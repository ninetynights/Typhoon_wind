#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大风玫瑰_按站点总体_色标统一版.py（统一色标：所有站点 + 超阈值与持续共用）
-----------------------------------------------------------------
功能概述（增强注释版）：
  - 按站点统计风速/风向观测，在“所有台风影响时段”内计算两类玫瑰：
      1) exceed_bins: 风速 > THRESHOLD 的小时数按风向扇区统计（总小时）
      2) sustain_bins: 连续 >= MIN_CONSEC_HOURS 小时均 > THRESHOLD 的小时数按扇区统计（持续小时）
  - 输出：
      * 每站点 CSV（每扇区小时数 + 百分比）
      * 每站点双子图（左：总小时 / 右：持续小时），颜色使用统一 colorbar（可选）
      * 地图视图：在地图上绘制若干站点的“迷你风玫瑰”以便空间比较
  - 设计要点：
      * 统一色标（norm_global）计算方式：以所有站点、两类指标的扇区最大值为 vmax，
        从而颜色可以直接用于不同站点/不同指标间的可视化对比。
      * 迷你玫瑰地图上的指定经纬度处，避免复杂投影转换。
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.projections import PolarAxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset

plt.rcParams['font.sans-serif'] = ['Heiti TC']

# ===================== 参数区（可按需修改） =====================
NC_PATH    = r"/Users/momo/Desktop/业务相关/2025 影响台风大风/All_Typhoons_ExMaxWind.nc"
OUTPUT_DIR = Path("/Users/momo/Desktop/业务相关/2025 影响台风大风") / "输出_风玫瑰_按站点"

THRESHOLD        = 17.3       # m/s：超阈值定义
MIN_CONSEC_HOURS = 2          # “持续”定义：连续小时数阈值（建议 2 或 3）

N_BINS           = 16         # 风向扇区数（16=22.5°；24=15°）
EDGES_DEG        = np.linspace(0, 360, N_BINS+1)

MAP_EXTENT = [118, 123, 27, 32]   # 地图范围（None 自适应）
TOPK_FOR_MAP = 103                 # 地图上显示总小时最多的前 K 个站点（避免拥挤）

CMAP_NAME  = "viridis"           # 颜色映射（可改为 "plasma"/"magma"/"cividis" 等）
UNIFY_RLIM = False               # 是否统一迷你玫瑰半径上限（True=所有小图同半径；False=各自最大值）
# =============================================================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def rose_bin_counts(deg_values: np.ndarray, edges_deg: np.ndarray) -> np.ndarray:
    """
    将角度数组（来向，单位度）按指定边界分箱计数。
    - deg_values: 1D 数组，角度可能存在 NaN（需在调用前滤除）
    - edges_deg: 扇区边界数组（例如 np.linspace(0,360,N+1)）
    返回值：每个扇区的计数（长度为 len(edges_deg)-1）
    说明：使用 np.mod 将角度折回 [0,360) 以避免负角度或超过 360 的值干扰统计。
    """
    if deg_values.size == 0:
        return np.zeros(len(edges_deg)-1, dtype=int)
    vals = np.mod(deg_values, 360.0)  # 折回 0-360
    hist, _ = np.histogram(vals, bins=edges_deg)
    return hist.astype(int)

def rle_segments(bool_arr: np.ndarray):
    """
    找到布尔序列中连续 True 的区间（左闭右开 [start, end)），返回列表 (start, end)。
    - 常用于识别“持续”时间段，例如连续多小时满足超阈值条件。
    - 若输入为空返回空列表。
    """
    if bool_arr.size == 0:
        return []
    b = np.array(bool_arr, dtype=bool)
    padded = np.concatenate([[False], b, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]
    return list(zip(starts, ends))

def compute_station_roses(wind_speeds, wind_dirs, ty_ids, threshold, min_consec, edges_deg):
    """
    对所有站点计算两种玫瑰（总小时与持续小时），按站点返回矩阵与总小时数。
    输入：
      - wind_speeds, wind_dirs: shape (time, n_stations)
      - ty_ids: 每个时刻每站点所属台风内部索引（>=0 表示某台风；<0 表示非台风影响）
    输出：
      - exceed_bins: (n_stations, n_bins) 总小时扇区计数（WS > threshold）
      - sustain_bins: (n_stations, n_bins) 持续小时扇区计数（连续 >= min_consec 小时）
      - totals_exceed, totals_sustain: 每站点对应的总小时数（用于排序/规模控制）
    重点说明：
      - exceed 统计跨所有台风时间的总小时（任何 ty_id >=0 的时刻均考虑，但须同时有风速与风向有效值）。
      - sustain 在每个台风的连续时间段内独立判定，避免跨台风拼接连续段。
    """
    n_time, n_sta = wind_speeds.shape
    n_bins = len(edges_deg) - 1
    exceed_bins  = np.zeros((n_sta, n_bins), dtype=int)
    sustain_bins = np.zeros((n_sta, n_bins), dtype=int)
    totals_exceed  = np.zeros(n_sta, dtype=int)
    totals_sustain = np.zeros(n_sta, dtype=int)

    for i in range(n_sta):
        # —— 汇总所有台风影响期（任何 ty_id >= 0 的小时）
        mask_any = ty_ids[:, i] >= 0
        if not np.any(mask_any):
            continue
        ws_any = wind_speeds[mask_any, i]
        wd_any = wind_dirs[mask_any, i]

        # 只统计风速/风向同时有效的时刻
        valid_any = np.isfinite(ws_any) & np.isfinite(wd_any)
        gt_any = (ws_any > threshold) & valid_any

        # exceed：把每个满足条件的小时按风向扇区计数
        exceed_bins[i, :] = rose_bin_counts(wd_any[gt_any], edges_deg)
        totals_exceed[i]  = int(np.sum(gt_any))

        # sustain：在每个台风段内独立判断连续 True 段（避免不同台风段拼接）
        uniq_ty = sorted({int(k) for k in np.unique(ty_ids[:, i]) if int(k) >= 0})
        for ty in uniq_ty:
            mask_ty = (ty_ids[:, i] == ty)
            ws = wind_speeds[mask_ty, i]
            wd = wind_dirs[mask_ty, i]

            valid = np.isfinite(ws) & np.isfinite(wd)
            gt = (ws > threshold) & valid
            if not np.any(gt):
                continue

            # 对 True 序列做 RLE：找到所有连续段
            for s, e in rle_segments(gt):
                if (e - s) >= min_consec:
                    # 把该段内的每个小时的风向都计入扇区
                    sustain_bins[i, :] += rose_bin_counts(wd[s:e], edges_deg)
                    totals_sustain[i]  += (e - s)

    return exceed_bins, sustain_bins, totals_exceed, totals_sustain

def plot_station_rose_pair_colored(exceed_counts, sustain_counts, edges_deg, title, out_png,
                                   cmap_name="viridis", norm=None):
    """
    绘制单站点左右并列的两个极坐标玫瑰：
      - 左图：总小时（exceed_counts）
      - 右图：持续小时（sustain_counts）
    参数说明：
      - norm: matplotlib.colors.Normalize 对象，若提供可保证不同站点间颜色一致（用于统一色标）
      - title: 图像总标题（包含站点编号与总小时信息）
      - out_png: 输出文件路径
    额外细节：
      - 为防止副标题遮挡北向刻度，将子图标题上移（y=1.08）。
      - 在条形顶部标注小时数（如需可关闭）。
    """
    angles_center_deg = 0.5 * (edges_deg[:-1] + edges_deg[1:])
    angles_rad = np.deg2rad(angles_center_deg)
    widths_rad = np.deg2rad(np.diff(edges_deg))

    if norm is None:
        # 未传统一色标时，使用该站点自身的扇区最大值
        vmax = max(int(np.max(exceed_counts)), int(np.max(sustain_counts)), 1)
        norm = mcolors.Normalize(vmin=0, vmax=vmax)

    cmap = plt.get_cmap(cmap_name)

    fig = plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(121, polar=True)
    ax2 = plt.subplot(122, polar=True)

    for ax, counts, subtitle in [(ax1, exceed_counts, "Exceedance Hours"),
                                 (ax2, sustain_counts, "Sustained Hours")]:
        ax.set_theta_zero_location('N')  # 0° 指向北（图上方）
        ax.set_theta_direction(-1)       # 顺时针增大（气象来向）
        colors = cmap(norm(counts))
        ax.bar(angles_rad, counts, width=widths_rad, align='center',
               color=colors, edgecolor='none')
        ax.set_title(subtitle, va='bottom', fontsize=11, y=1.08)  # ← 避免与 0° 字标重叠

        # 在条顶标注小时数（可按需注释关闭）
        for a, r in zip(angles_rad, counts):
            if r <= 0:
                continue
            ax.text(a, r*1.02, f"{int(r)}", ha='center', va='bottom', fontsize=8)

    # 统一色标的颜色条（两个子图共用一根）
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], shrink=0.9, pad=0.05)
    cbar.set_label("Hours per sector (Unified)")

    fig.suptitle(title, fontsize=13)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

def add_mini_rose_colored(ax_map, lon, lat, counts, edges_deg, size_inch, norm, cmap, rlim_max=None):
    """
    在地图上将一个迷你极坐标玫瑰以 inset 的方式叠加到指定经纬度位置。
    实现与注意事项：
      - 使用 inset_axes 创建一个极坐标小轴（PolarAxes），位置由 bbox_to_anchor 与 ax.transData 控制（经纬度空间）。
      - size_inch 表示插入图像的宽高（英寸），可按站点重要性放缩。
      - rlim_max: 若非 None 则所有小图使用相同 r 轴上限（便于大小比较），否则每图自适应。
      - 该方法在省级范围内可视化效果良好；若地图缩放很大或处于高纬度区域，极坐标到经纬度的近似偏差需注意。
    """
    angles_center_deg = 0.5 * (edges_deg[:-1] + edges_deg[1:])
    angles_rad = np.deg2rad(angles_center_deg)
    widths_rad = np.deg2rad(np.diff(edges_deg))

    # 创建极坐标 inset 轴
    iax = inset_axes(
        ax_map, width=size_inch, height=size_inch, loc='center',
        bbox_to_anchor=(lon, lat), bbox_transform=ax_map.transData,
        axes_class=PolarAxes, borderpad=0
    )
    iax.set_theta_zero_location('N')
    iax.set_theta_direction(-1)
    iax.set_xticks([]); iax.set_yticks([])  # 小图不显示刻度
    # 半径上限（统一 or 自适应）
    if rlim_max is None:
        iax.set_rlim(0, max(1, int(np.max(counts))))
    else:
        iax.set_rlim(0, max(1, int(rlim_max)))
    iax.grid(False)
    iax.spines['polar'].set_visible(False)

    colors = cmap(norm(counts))
    iax.bar(angles_rad, counts, width=widths_rad, align='center', color=colors, edgecolor='none')

def main():
    # —— 输出目录组织
    out_root = OUTPUT_DIR
    out_csv_dir  = out_root / "per_station" / "csv"
    out_fig_dir  = out_root / "per_station" / "figs"
    out_map_dir  = out_root / "map"
    ensure_dir(out_csv_dir); ensure_dir(out_fig_dir); ensure_dir(out_map_dir)

    # —— 读数据
    nc = Dataset(NC_PATH)
    ws = np.array(nc.variables['wind_velocity'][:, 0, :], copy=True)   # [time, station]
    wd = np.array(nc.variables['wind_direction'][:, 0, :], copy=True)  # [time, station]（度，来向）
    ty = np.array(nc.variables['typhoon_id_index'][:, 0, :], copy=True)
    stids = np.array(nc.variables['STID'][:]).astype(str)
    lats  = np.array(nc.variables['lat'][:], dtype=float)
    lons  = np.array(nc.variables['lon'][:], dtype=float)

    # ty 转 int（缺测置 -1）
    if np.issubdtype(ty.dtype, np.floating):
        ty_int = np.full_like(ty, fill_value=-1, dtype=int)
        valid = ~np.isnan(ty)
        ty_int[valid] = ty[valid].astype(int)
        ty = ty_int
    else:
        ty = ty.astype(int)

    # —— 计算每站点两类玫瑰（扇区小时数矩阵）
    ex_bins, su_bins, ex_totals, su_totals = compute_station_roses(ws, wd, ty, THRESHOLD, MIN_CONSEC_HOURS, EDGES_DEG)

    # —— 统一色标：所有站点 + 两类指标共用同一把尺（0 ~ GLOBAL_VMAX）
    GLOBAL_VMAX = int(max(ex_bins.max(), su_bins.max(), 1))
    norm_global = mcolors.Normalize(vmin=0, vmax=GLOBAL_VMAX)
    cmap = plt.get_cmap(CMAP_NAME)

    # —— 导出每站点 CSV 与双图（使用统一色标）
    for i in range(ex_bins.shape[0]):
        df = pd.DataFrame({
            "bin_from_deg": EDGES_DEG[:-1],
            "bin_to_deg":   EDGES_DEG[1:],
            "exceed_hours": ex_bins[i, :],
            "exceed_percent": (ex_bins[i, :] / max(1, ex_bins[i, :].sum())) * 100.0,
            "sustain_hours": su_bins[i, :],
            "sustain_percent": (su_bins[i, :] / max(1, su_bins[i, :].sum())) * 100.0
        })
        df.to_csv(out_csv_dir / f"{stids[i]}_rose_{THRESHOLD:.1f}_ge{MIN_CONSEC_HOURS}h.csv",
                  index=False, encoding="utf-8")

        title = f"Station {stids[i]}  (Exceed={ex_totals[i]} h, Sustain={su_totals[i]} h)"
        plot_station_rose_pair_colored(
            ex_bins[i, :], su_bins[i, :], EDGES_DEG, title,
            out_fig_dir / f"{stids[i]}_rose_{THRESHOLD:.1f}_ge{MIN_CONSEC_HOURS}h.png",
            cmap_name=CMAP_NAME, norm=norm_global
        )

    # —— 地图：选择 TOP K 站点（按总超阈值小时数）
    n_sta = ex_bins.shape[0]
    order = np.argsort(-ex_totals)            # 降序
    idxs = order[:min(TOPK_FOR_MAP, n_sta)]   # Top K

    # 统一色标（仅用于地图 TopK 范围，也可直接用 norm_global）
    GLOBAL_VMAX_TOPK = int(max(ex_bins[idxs, :].max(), su_bins[idxs, :].max(), 1))
    norm_map = mcolors.Normalize(vmin=0, vmax=GLOBAL_VMAX_TOPK)

    # 迷你玫瑰尺寸：按站点总超阈值小时的平方根缩放，避免极端差异
    if ex_totals[idxs].max() > 0:
        size_norm = np.sqrt(ex_totals[idxs] / ex_totals[idxs].max())
    else:
        size_norm = np.ones_like(idxs, dtype=float)

    # —— 地图 1：总小时迷你玫瑰（统一色标）
    fig = plt.figure(figsize=(10, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
    ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')
    if MAP_EXTENT and len(MAP_EXTENT) == 4:
        ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    else:
        pad_lon = max(0.5, (np.nanmax(lons) - np.nanmin(lons)) * 0.1)
        pad_lat = max(0.5, (np.nanmax(lats) - np.nanmin(lats)) * 0.1)
        ax.set_extent([np.nanmin(lons) - pad_lon, np.nanmax(lons) + pad_lon,
                       np.nanmin(lats) - pad_lat, np.nanmax(lats) + pad_lat],
                      crs=ccrs.PlateCarree())
    # 半径统一上限（可选）：若 UNIFY_RLIM=True，则把 rlim_max 设为 GLOBAL_VMAX_TOPK；否则为 None（各自最大值）
    rlim_max = GLOBAL_VMAX_TOPK if UNIFY_RLIM else None
    for j, i in enumerate(idxs):
        size_inch = 0.7 * (1.0 + 0.6 * size_norm[j])
        add_mini_rose_colored(ax, lons[i], lats[i], ex_bins[i, :], EDGES_DEG, size_inch,
                              norm_map, cmap, rlim_max=rlim_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_map)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Hours per sector (Unified scale)")
    ax.set_title(f"Top {len(idxs)} Stations: Exceedance Wind Roses (> {THRESHOLD} m/s)")
    fig.savefig(out_map_dir / f"Map_MiniRoses_Exceedance_{THRESHOLD:.1f}.png",
                dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    # —— 地图 2：持续小时迷你玫瑰（统一色标）
    fig = plt.figure(figsize=(10, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
    ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')
    if MAP_EXTENT and len(MAP_EXTENT) == 4:
        ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    else:
        pad_lon = max(0.5, (np.nanmax(lons) - np.nanmin(lons)) * 0.1)
        pad_lat = max(0.5, (np.nanmax(lats) - np.nanmin(lats)) * 0.1)
        ax.set_extent([np.nanmin(lons) - pad_lon, np.nanmax(lons) + pad_lon,
                       np.nanmin(lats) - pad_lat, np.nanmax(lats) + pad_lat],
                      crs=ccrs.PlateCarree())
    for j, i in enumerate(idxs):
        size_inch = 0.7 * (1.0 + 0.6 * size_norm[j])
        add_mini_rose_colored(ax, lons[i], lats[i], su_bins[i, :], EDGES_DEG, size_inch,
                              norm_map, cmap, rlim_max=rlim_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_map)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Hours per sector (Unified scale, ≥ {MIN_CONSEC_HOURS}h)")
    ax.set_title(f"Top {len(idxs)} Stations: Sustained Wind Roses (> {THRESHOLD} m/s, ≥ {MIN_CONSEC_HOURS}h)")
    fig.savefig(out_map_dir / f"Map_MiniRoses_Sustained_{THRESHOLD:.1f}_ge{MIN_CONSEC_HOURS}h.png",
                dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"[OK] 每站点玫瑰已输出，且已启用“统一色标”。目录：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()
