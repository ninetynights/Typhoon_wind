#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大风玫瑰_按站点总体.py（统一色标：所有站点 + 超阈值与持续共用）
-----------------------------------------------------------------
更新要点：
  1) 统一色标范围（colorbar）：先计算所有站点、两类指标（总小时/持续小时）的扇区最大值，得到 GLOBAL_VMAX，
     然后全程用同一个 Normalize(0, GLOBAL_VMAX) 着色 → 所有站点、两种指标可直接对比颜色深浅。
  2) 代码内增加了详细注释与“可调开关”，便于后续理解与调整。
  3) 地图上的两张迷你玫瑰图（总小时、持续小时）也使用同一把色标（norm_map = (0, GLOBAL_VMAX_TOPK)）。

你可以在“参数区”调整：
  - THRESHOLD、MIN_CONSEC_HOURS、N_BINS、TOPK_FOR_MAP 等。
  - UNIFY_RLIM：是否把小玫瑰的半径也统一（默认 False，保持每站点自适应半径，便于观察形状）。

依赖：numpy, pandas, matplotlib, cartopy, netCDF4
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
    """将角度（度，0-360）按扇区计数。"""
    if deg_values.size == 0:
        return np.zeros(len(edges_deg)-1, dtype=int)
    vals = np.mod(deg_values, 360.0)  # 折回 0-360
    hist, _ = np.histogram(vals, bins=edges_deg)
    return hist.astype(int)

def rle_segments(bool_arr: np.ndarray):
    """返回 True 连续段 (start, end)（end 为开区间）。"""
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
    逐站点计算两类玫瑰的扇区小时数：
      - exceed_bins[i, :]  : 该站点在“所有台风影响时段”内，WS>threshold 的小时分布（按风向扇区）
      - sustain_bins[i, :] : “连续 >= min_consec 小时均 > threshold”的小时分布（按风向扇区）
      - totals_exceed[i]   : 该站点总的超阈值小时数
      - totals_sustain[i]  : 该站点总的持续小时数
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

        valid_any = np.isfinite(ws_any) & np.isfinite(wd_any)
        gt_any = (ws_any > threshold) & valid_any

        # 总小时玫瑰（把每个满足 gt 的小时，按其风向分扇区计数）
        exceed_bins[i, :] = rose_bin_counts(wd_any[gt_any], edges_deg)
        totals_exceed[i]  = int(np.sum(gt_any))

        # 按“站点×台风”拆开做连续段（防跨台风拼接）
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
    单站点双图：左=总小时，右=持续小时。
    - 颜色由传入的 `norm` 控制（若为 None，则退化为“本站点自适应”）。
    - 为了避免副标题与 0°(N) 刻度重叠，标题位置向上（y=1.12）。
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
        ax.set_theta_zero_location('N')  # 北在上
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
    在地图上叠加“迷你玫瑰”插图：
      - 颜色：使用统一 `norm`，实现所有站点与两类指标共用同一色标。
      - 半径：默认自适应（该站点该图的扇区最大值），也可通过 rlim_max 统一到全局
              （当 UNIFY_RLIM=True 时，传入全局上限）。
    """
    angles_center_deg = 0.5 * (edges_deg[:-1] + edges_deg[1:])
    angles_rad = np.deg2rad(angles_center_deg)
    widths_rad = np.deg2rad(np.diff(edges_deg))

    iax = inset_axes(
        ax_map, width=size_inch, height=size_inch, loc='center',
        bbox_to_anchor=(lon, lat), bbox_transform=ax_map.transData,
        axes_class=PolarAxes, borderpad=0
    )
    iax.set_theta_zero_location('N')
    iax.set_theta_direction(-1)
    iax.set_xticks([]); iax.set_yticks([])
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
