#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大风玫瑰图.py（总体：统一色标，超阈值与持续共用）
-----------------------------------------------
更新要点：
  - 在计算出“超阈值扇区小时数”和“持续扇区小时数”后，取两者的全局最大值 GLOBAL_VMAX，
    用同一 Normalize(0, GLOBAL_VMAX) 着色，从而两张图的颜色具有可比性。

依赖：numpy, pandas, matplotlib, netCDF4
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from netCDF4 import Dataset

plt.rcParams['font.sans-serif'] = ['Heiti TC']

# ============= 参数 =============
NC_PATH    = r"/Users/momo/Desktop/业务相关/2025 影响台风大风/All_Typhoons_ExMaxWind.nc"
OUTPUT_DIR = Path("/Users/momo/Desktop/业务相关/2025 影响台风大风") / "输出_风玫瑰"

THRESHOLD        = 17.3
MIN_CONSEC_HOURS = 2
N_BINS           = 16
EDGES_DEG        = np.linspace(0, 360, N_BINS+1)
CMAP_NAME        = "viridis"
ANNOTATE_BARS    = True
# ===============================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def rle_segments(bool_arr: np.ndarray):
    if bool_arr.size == 0:
        return []
    b = np.array(bool_arr, dtype=bool)
    padded = np.concatenate([[False], b, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]
    return list(zip(starts, ends))

def rose_bin_counts(deg_values: np.ndarray, edges_deg: np.ndarray) -> np.ndarray:
    if deg_values.size == 0:
        return np.zeros(len(edges_deg)-1, dtype=int)
    vals = np.mod(deg_values, 360.0)
    hist, _ = np.histogram(vals, bins=edges_deg)
    return hist.astype(int)

def plot_wind_rose_colored(counts: np.ndarray, edges_deg: np.ndarray, title: str, out_png: Path,
                           cmap_name="viridis", annotate=True, norm=None):
    angles_center_deg = 0.5 * (edges_deg[:-1] + edges_deg[1:])
    angles_rad = np.deg2rad(angles_center_deg)
    widths_rad = np.deg2rad(np.diff(edges_deg))

    if norm is None:
        norm = mcolors.Normalize(vmin=0, vmax=max(int(np.max(counts)), 1))

    cmap = plt.get_cmap(cmap_name)
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    colors = cmap(norm(counts))
    ax.bar(angles_rad, counts, width=widths_rad, align='center', color=colors, edgecolor='none')

    if annotate:
        for a, r in zip(angles_rad, counts):
            if r <= 0:
                continue
            ax.text(a, r*1.02, f"{int(r)}", ha='center', va='bottom', fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.9)
    cbar.set_label("Hours per sector (Unified scale)")

    ax.set_title(title, va='bottom', fontsize=13, pad=20)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

def main():
    out_csv = OUTPUT_DIR / "csv"
    out_fig = OUTPUT_DIR / "figs"
    ensure_dir(out_csv); ensure_dir(out_fig)

    # —— 读数据
    nc = Dataset(NC_PATH)
    wind_speeds = np.array(nc.variables['wind_velocity'][:, 0, :], copy=True)
    wind_dirs   = np.array(nc.variables['wind_direction'][:, 0, :], copy=True)
    ty_ids      = np.array(nc.variables['typhoon_id_index'][:, 0, :], copy=True)

    # ty 转 int
    if np.issubdtype(ty_ids.dtype, np.floating):
        ty_ids_int = np.full_like(ty_ids, fill_value=-1, dtype=int)
        valid = ~np.isnan(ty_ids)
        ty_ids_int[valid] = ty_ids[valid].astype(int)
        ty_ids = ty_ids_int
    else:
        ty_ids = ty_ids.astype(int)

    # —— 总体：跨所有站点×台风累积两类扇区计数
    exceed_bins = np.zeros(len(EDGES_DEG)-1, dtype=int)
    sustain_bins = np.zeros(len(EDGES_DEG)-1, dtype=int)

    # 1) 超阈值总小时（总体）
    for i in range(wind_speeds.shape[1]):
        mask_any = ty_ids[:, i] >= 0
        if not np.any(mask_any):
            continue
        ws = wind_speeds[mask_any, i]
        wd = wind_dirs[mask_any, i]
        valid = np.isfinite(ws) & np.isfinite(wd)
        gt = (ws > THRESHOLD) & valid
        if not np.any(gt):
            continue
        exceed_bins += rose_bin_counts(wd[gt], EDGES_DEG)

    # 2) 持续小时（总体）：逐站点×台风做连续段，满足阈值的每小时计入
    for i in range(wind_speeds.shape[1]):
        uniq_ty = sorted({int(k) for k in np.unique(ty_ids[:, i]) if int(k) >= 0})
        if not uniq_ty:
            continue
        for ty in uniq_ty:
            mask_ty = (ty_ids[:, i] == ty)
            ws = wind_speeds[mask_ty, i]
            wd = wind_dirs[mask_ty, i]
            valid = np.isfinite(ws) & np.isfinite(wd)
            gt = (ws > THRESHOLD) & valid
            if not np.any(gt):
                continue
            for s, e in rle_segments(gt):
                if (e - s) >= MIN_CONSEC_HOURS:
                    sustain_bins += rose_bin_counts(wd[s:e], EDGES_DEG)

    # —— 统一色标
    GLOBAL_VMAX = int(max(exceed_bins.max(), sustain_bins.max(), 1))
    norm_global = mcolors.Normalize(vmin=0, vmax=GLOBAL_VMAX)

    # —— CSV
    df_ex = pd.DataFrame({
        "bin_from_deg": EDGES_DEG[:-1],
        "bin_to_deg":   EDGES_DEG[1:],
        "hours":        exceed_bins,
        "percent":      (exceed_bins / max(1, exceed_bins.sum())) * 100.0
    })
    df_ex.to_csv(out_csv / f"Overall_Exceedance_WindRose_{THRESHOLD:.1f}.csv", index=False, encoding="utf-8")

    df_su = pd.DataFrame({
        "bin_from_deg": EDGES_DEG[:-1],
        "bin_to_deg":   EDGES_DEG[1:],
        "hours":        sustain_bins,
        "percent":      (sustain_bins / max(1, sustain_bins.sum())) * 100.0
    })
    df_su.to_csv(out_csv / f"Overall_Sustained_WindRose_{THRESHOLD:.1f}_ge{MIN_CONSEC_HOURS}h.csv", index=False, encoding="utf-8")

    # —— 图（统一色标）
    plot_wind_rose_colored(exceed_bins, EDGES_DEG,
                           title=f"Overall Wind Rose (Hours > {THRESHOLD} m/s)",
                           out_png=out_fig / f"Overall_Exceedance_WindRose_{THRESHOLD:.1f}.png",
                           cmap_name=CMAP_NAME, annotate=ANNOTATE_BARS, norm=norm_global)

    plot_wind_rose_colored(sustain_bins, EDGES_DEG,
                           title=f"Overall Wind Rose (Sustained > {THRESHOLD} m/s, ≥ {MIN_CONSEC_HOURS}h)",
                           out_png=out_fig / f"Overall_Sustained_WindRose_{THRESHOLD:.1f}_ge{MIN_CONSEC_HOURS}h.png",
                           cmap_name=CMAP_NAME, annotate=ANNOTATE_BARS, norm=norm_global)

    print(f"[OK] 总体玫瑰已输出（统一色标）：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()
