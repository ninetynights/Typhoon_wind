#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大风玫瑰_按站点总体.py（带颜色映射与色标 + 地图迷你玫瑰全局色标）
-----------------------------------------------------------------
变化点：
  - 每个站点的双图（总小时/持续小时）使用相同的颜色映射与色标（Hours per sector）。
  - 地图上的迷你玫瑰按“绝对扇区小时数”着色，并在整张图右侧放一个全局 colorbar。
  - 去掉 tight_layout 警告，改用 bbox_inches='tight' 保存。
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

# ======= 路径与参数 =======
NC_PATH    = r"/Users/momo/Desktop/业务相关/2025 影响台风大风/All_Typhoons_ExMaxWind.nc"
OUTPUT_DIR = Path("/Users/momo/Desktop/业务相关/2025 影响台风大风") / "输出_风玫瑰_按站点"

THRESHOLD        = 17.3
MIN_CONSEC_HOURS = 2
N_BINS           = 16
EDGES_DEG        = np.linspace(0, 360, N_BINS+1)

# 地图与显示
MAP_EXTENT = [118, 123, 27, 32] # 设为 None 则自适应
TOPK_FOR_MAP = 103  # 地图上显示的站点数（按总超阈值小时排序）
ROSE_SIZE_INCH_BASE = 0.7   # 迷你玫瑰基础尺寸（英寸）
ROSE_SIZE_SCALE = 0.6   # 迷你玫瑰尺寸缩放（按总超阈值小时平方根缩放）
CMAP_NAME = "viridis" # 颜色映射

# ----------------------------- 工具函数 -----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def rose_bin_counts(deg_values: np.ndarray, edges_deg: np.ndarray) -> np.ndarray:
    if deg_values.size == 0:
        return np.zeros(len(edges_deg)-1, dtype=int)
    vals = np.mod(deg_values, 360.0)
    hist, _ = np.histogram(vals, bins=edges_deg)
    return hist.astype(int)

def rle_segments(bool_arr: np.ndarray):
    if bool_arr.size == 0:
        return []
    b = np.array(bool_arr, dtype=bool)
    padded = np.concatenate([[False], b, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]
    return list(zip(starts, ends))

def compute_station_roses(wind_speeds, wind_dirs, ty_ids, threshold, min_consec, edges_deg):
    n_time, n_sta = wind_speeds.shape
    n_bins = len(edges_deg) - 1
    exceed_bins  = np.zeros((n_sta, n_bins), dtype=int)
    sustain_bins = np.zeros((n_sta, n_bins), dtype=int)
    totals_exceed  = np.zeros(n_sta, dtype=int)
    totals_sustain = np.zeros(n_sta, dtype=int)

    for i in range(n_sta):
        mask_any = ty_ids[:, i] >= 0
        if not np.any(mask_any):
            continue
        ws_any = wind_speeds[mask_any, i]
        wd_any = wind_dirs[mask_any, i]
        valid_any = np.isfinite(ws_any) & np.isfinite(wd_any)
        gt_any = (ws_any > threshold) & valid_any
        exceed_bins[i, :] = rose_bin_counts(wd_any[gt_any], edges_deg)
        totals_exceed[i]  = int(np.sum(gt_any))

        uniq_ty = sorted({int(k) for k in np.unique(ty_ids[:, i]) if int(k) >= 0})
        for ty in uniq_ty:
            mask_ty = (ty_ids[:, i] == ty)
            ws = wind_speeds[mask_ty, i]
            wd = wind_dirs[mask_ty, i]
            valid = np.isfinite(ws) & np.isfinite(wd)
            gt = (ws > threshold) & valid
            if not np.any(gt):
                continue
            for s, e in rle_segments(gt):
                if (e - s) >= min_consec:
                    sustain_bins[i, :] += rose_bin_counts(wd[s:e], edges_deg)
                    totals_sustain[i]  += (e - s)

    return exceed_bins, sustain_bins, totals_exceed, totals_sustain

def plot_station_rose_pair_colored(exceed_counts, sustain_counts, edges_deg, title, out_png, cmap_name="viridis"):
    angles_center_deg = 0.5 * (edges_deg[:-1] + edges_deg[1:])
    angles_rad = np.deg2rad(angles_center_deg)
    widths_rad = np.deg2rad(np.diff(edges_deg))

    vmax = max(int(np.max(exceed_counts)), int(np.max(sustain_counts)), 1)
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    fig = plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(121, polar=True)
    ax2 = plt.subplot(122, polar=True)

    for ax, counts, subtitle in [(ax1, exceed_counts, "Exceedance Hours"),
                                 (ax2, sustain_counts, "Sustained Hours")]:
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        colors = cmap(norm(counts))
        ax.bar(angles_rad, counts, width=widths_rad, align='center', color=colors, edgecolor='none')
        ax.set_title(subtitle, va='bottom', fontsize=11, y=1.08)
        for a, r in zip(angles_rad, counts):
            if r <= 0:
                continue
            ax.text(a, r*1.02, f"{int(r)}", ha='center', va='bottom', fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], shrink=0.9, pad=0.05)
    cbar.set_label("Hours per sector")

    fig.suptitle(title, fontsize=13)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

def add_mini_rose_colored(ax_map, lon, lat, counts, edges_deg, size_inch, norm, cmap):
    angles_center_deg = 0.5 * (edges_deg[:-1] + edges_deg[1:])
    angles_rad = np.deg2rad(angles_center_deg)
    widths_rad = np.deg2rad(np.diff(edges_deg))

    # 极坐标插图
    iax = inset_axes(
        ax_map, width=size_inch, height=size_inch, loc='center',
        bbox_to_anchor=(lon, lat), bbox_transform=ax_map.transData,
        axes_class=PolarAxes, borderpad=0
    )
    iax.set_theta_zero_location('N')
    iax.set_theta_direction(-1)
    iax.set_xticks([]); iax.set_yticks([])
    iax.set_rlim(0, max(1, int(np.max(counts))))
    iax.grid(False)
    iax.spines['polar'].set_visible(False)

    colors = cmap(norm(counts))
    iax.bar(angles_rad, counts, width=widths_rad, align='center', color=colors, edgecolor='none')

# ----------------------------- 主逻辑 -----------------------------
def main():
    out_root = OUTPUT_DIR
    out_csv_dir  = out_root / "per_station" / "csv"
    out_fig_dir  = out_root / "per_station" / "figs"
    out_map_dir  = out_root / "map"
    ensure_dir(out_csv_dir); ensure_dir(out_fig_dir); ensure_dir(out_map_dir)

    nc = Dataset(NC_PATH)
    ws = np.array(nc.variables['wind_velocity'][:, 0, :], copy=True)
    wd = np.array(nc.variables['wind_direction'][:, 0, :], copy=True)
    ty = np.array(nc.variables['typhoon_id_index'][:, 0, :], copy=True)
    stids = np.array(nc.variables['STID'][:]).astype(str)
    lats  = np.array(nc.variables['lat'][:], dtype=float)
    lons  = np.array(nc.variables['lon'][:], dtype=float)

    if np.issubdtype(ty.dtype, np.floating):
        ty_int = np.full_like(ty, fill_value=-1, dtype=int)
        valid = ~np.isnan(ty)
        ty_int[valid] = ty[valid].astype(int)
        ty = ty_int
    else:
        ty = ty.astype(int)

    n_time, n_sta = ws.shape

    ex_bins, su_bins, ex_totals, su_totals = compute_station_roses(ws, wd, ty, THRESHOLD, MIN_CONSEC_HOURS, EDGES_DEG)

    # 单站点导出（彩色 + 色标）
    for i in range(n_sta):
        df = pd.DataFrame({
            "bin_from_deg": EDGES_DEG[:-1],
            "bin_to_deg":   EDGES_DEG[1:],
            "exceed_hours": ex_bins[i, :],
            "exceed_percent": (ex_bins[i, :]/ max(1, ex_bins[i, :].sum())) * 100.0,
            "sustain_hours": su_bins[i, :],
            "sustain_percent": (su_bins[i, :]/ max(1, su_bins[i, :].sum())) * 100.0
        })
        df.to_csv(out_csv_dir / f"{stids[i]}_rose_{THRESHOLD:.1f}_ge{MIN_CONSEC_HOURS}h.csv", index=False, encoding="utf-8")

        title = f"Station {stids[i]}  (Exceed={ex_totals[i]} h, Sustain={su_totals[i]} h)"
        plot_station_rose_pair_colored(ex_bins[i, :], su_bins[i, :], EDGES_DEG, title,
                                       out_fig_dir / f"{stids[i]}_rose_{THRESHOLD:.1f}_ge{MIN_CONSEC_HOURS}h.png",
                                       cmap_name=CMAP_NAME)

    # 地图：按总超阈值小时排前 TOPK
    order = np.argsort(-ex_totals)
    idxs = order[:min(TOPK_FOR_MAP, n_sta)]

    # 全局颜色映射（绝对扇区小时数）
    global_vmax_ex = max(int(np.max(ex_bins[idxs, :])), 1)
    global_vmax_su = max(int(np.max(su_bins[idxs, :])), 1)
    norm_ex = mcolors.Normalize(vmin=0, vmax=global_vmax_ex)
    norm_su = mcolors.Normalize(vmin=0, vmax=global_vmax_su)
    cmap = plt.get_cmap(CMAP_NAME)

    # 玫瑰尺寸：总超阈值小时平方根缩放
    if ex_totals[idxs].max() > 0:
        size_norm = np.sqrt(ex_totals[idxs]/ ex_totals[idxs].max())
    else:
        size_norm = np.ones_like(idxs, dtype=float)

    # 地图 1：总小时
    fig = plt.figure(figsize=(10, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
    ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')
    if MAP_EXTENT and len(MAP_EXTENT)==4:
        ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    else:
        pad_lon = max(0.5, (np.nanmax(lons) - np.nanmin(lons)) * 0.1)
        pad_lat = max(0.5, (np.nanmax(lats) - np.nanmin(lats)) * 0.1)
        ax.set_extent([np.nanmin(lons)-pad_lon, np.nanmax(lons)+pad_lon,
                       np.nanmin(lats)-pad_lat, np.nanmax(lats)+pad_lat],
                      crs=ccrs.PlateCarree())
    for j, i in enumerate(idxs):
        size_inch = ROSE_SIZE_INCH_BASE * (1.0 + ROSE_SIZE_SCALE*size_norm[j])
        add_mini_rose_colored(ax, lons[i], lats[i], ex_bins[i, :], EDGES_DEG, size_inch, norm_ex, cmap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_ex)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Exceedance hours per sector")
    ax.set_title(f"Top {len(idxs)} Stations: Exceedance Wind Roses (> {THRESHOLD} m/s)")
    fig.savefig(out_map_dir / f"Map_MiniRoses_Exceedance_{THRESHOLD:.1f}.png", dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    # 地图 2：持续小时
    fig = plt.figure(figsize=(10, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
    ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')
    if MAP_EXTENT and len(MAP_EXTENT)==4:
        ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    else:
        pad_lon = max(0.5, (np.nanmax(lons) - np.nanmin(lons)) * 0.1)
        pad_lat = max(0.5, (np.nanmax(lats) - np.nanmin(lats)) * 0.1)
        ax.set_extent([np.nanmin(lons)-pad_lon, np.nanmax(lons)+pad_lon,
                       np.nanmin(lats)-pad_lat, np.nanmax(lats)+pad_lat],
                      crs=ccrs.PlateCarree())
    for j, i in enumerate(idxs):
        size_inch = ROSE_SIZE_INCH_BASE * (1.0 + ROSE_SIZE_SCALE*size_norm[j])
        add_mini_rose_colored(ax, lons[i], lats[i], su_bins[i, :], EDGES_DEG, size_inch, norm_su, cmap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_su)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Sustained hours per sector (≥ {MIN_CONSEC_HOURS}h)")
    ax.set_title(f"Top {len(idxs)} Stations: Sustained Wind Roses (> {THRESHOLD} m/s, ≥ {MIN_CONSEC_HOURS}h)")
    fig.savefig(out_map_dir / f"Map_MiniRoses_Sustained_{THRESHOLD:.1f}_ge{MIN_CONSEC_HOURS}h.png", dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"[OK] 每站点玫瑰（含色标）与地图迷你玫瑰（含色标）已输出：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()
