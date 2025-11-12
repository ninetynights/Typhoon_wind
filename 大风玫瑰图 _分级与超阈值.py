#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
大风玫瑰图_分级.py — 基于风速分级的风玫瑰统计与可视化

总体目的：
- 按蒲氏风力等级（示例为 8–16 级）统计观测时序中不同风向扇区的小时数，
  分别计算两类指标：区间小时（Exactly N 级）与阈值小时（Over N 级）。
- 为每个风力等级生成统一色标的风玫瑰图与对应的 CSV 报表，便于等级间的横向比较与归档。

主要功能：
- 从指定 NetCDF（NC_PATH）读取风速 wind_velocity、风向 wind_direction、台风索引 typhoon_id_index 等变量；
- 仅统计“台风影响时段”（typhoon_id_index >= 0）内的有效观测（同时有风速与风向）；
- 对每个等级：
    1) 计算区间小时：风速位于该等级上下限之间的小时数，按风向扇区累积；
    2) 计算阈值小时：风速大于等于该等级下限的小时数，按风向扇区累积；
- 输出每等级的 CSV（扇区小时数与百分比）与 PNG 风玫瑰图（区间 / 阈值，使用该等级统一色标）。

输入（需在脚本顶部配置）：
- NC_PATH：包含变量 'wind_velocity','wind_direction','typhoon_id_index' 的 NetCDF 文件路径。
- WIND_BINS：风速等级定义列表（等级名称、最小风速、最大风速）。
- 输出目录 OUTPUT_DIR、绘图扇区数 N_BINS 和 colormap 等绘图参数。

输出（保存到 OUTPUT_DIR 下的子目录）：
- csv/Overall_WindRose_Level_{等级}_Exactly_*.csv
- csv/Overall_WindRose_Level_{等级}_Over_*.csv
- figs/Overall_WindRose_Level_{等级}_Exactly_*.png
- figs/Overall_WindRose_Level_{等级}_Over_*.png

"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from netCDF4 import Dataset

plt.rcParams['font.sans-serif'] = ['Heiti TC']

# ============= 参数 =============
NC_PATH    = r"/Users/momo/Desktop/业务相关/2025 影响台风大风/数据/Combine_Stations_ExMaxWind.nc"
OUTPUT_DIR = Path("/Users/momo/Desktop/业务相关/2025 影响台风大风") / "输出_风玫瑰_分级与超阈值"

# --- 风速分级参数 (蒲氏风力) ---
# (等级名称, 最小风速, 最大风速)
WIND_BINS = [
    ("8级", 17.2, 20.7),
    ("9级", 20.8, 24.4),
    ("10级", 24.5, 28.4),
    ("11级", 28.5, 32.6),
    ("12级", 32.7, 36.9),
    ("13级", 37.0, 41.4),
    ("14级", 41.5, 46.1),
    ("15级", 46.2, 50.9),
    ("16级", 51.0, 56.0),
]

# --- 绘图参数 ---
N_BINS           = 16
EDGES_DEG        = np.linspace(0, 360, N_BINS+1)
CMAP_NAME        = "viridis"
ANNOTATE_BARS    = True
# ===============================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def rose_bin_counts(deg_values: np.ndarray, edges_deg: np.ndarray) -> np.ndarray:
    if deg_values.size == 0:
        return np.zeros(len(edges_deg)-1, dtype=int)
    vals = np.mod(deg_values, 360.0)
    hist, _ = np.histogram(vals, bins=edges_deg)
    return hist.astype(int)

def plot_wind_rose_colored(counts: np.ndarray, edges_deg: np.ndarray, title: str, out_png: Path,
                           cmap_name="viridis", annotate=True, norm=None, cbar_label=None):
    """
    绘制风玫瑰图 (可指定色标和标签)
    """
    angles_center_deg = 0.5 * (edges_deg[:-1] + edges_deg[1:])
    angles_rad = np.deg2rad(angles_center_deg)
    widths_rad = np.deg2rad(np.diff(edges_deg))

    if norm is None:
        # 如果未指定norm，则自动根据当前数据创建
        vmax = max(int(np.max(counts)), 1)
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
        if cbar_label is None:
             cbar_label = f"Hours per sector (Scale 0-{vmax})"
    else:
        # 如果指定了norm (统一色标)
        if cbar_label is None:
            cbar_label = f"Hours per sector (Unified scale 0-{int(norm.vmax)})"

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
    cbar.set_label(cbar_label) # 使用指定的标签

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
    nc.close() # 及时关闭文件

    # ty 转 int
    if np.issubdtype(ty_ids.dtype, np.floating):
        ty_ids_int = np.full_like(ty_ids, fill_value=-1, dtype=int)
        valid_ty = ~np.isnan(ty_ids)
        ty_ids_int[valid_ty] = ty_ids[valid_ty].astype(int)
        ty_ids = ty_ids_int
    else:
        ty_ids = ty_ids.astype(int)

    # ======== 1. 初始化所有统计容器 ========
    
    # "区间" (Exactly N-level)
    binned_stats = {
        level: np.zeros(len(EDGES_DEG)-1, dtype=int) 
        for level, _, _ in WIND_BINS
    }
    # "阈值" (Over N-level)
    exceed_stats = {
        level: np.zeros(len(EDGES_DEG)-1, dtype=int) 
        for level, _, _ in WIND_BINS
    }

    # ======== 2. 计算：区间 (Exactly) 和 阈值 (Over) ========
    print("--- 正在计算 分级区间 (Exactly) 和 分级阈值 (Over) ---")
    
    for i in range(wind_speeds.shape[1]): # 遍历所有站点
        mask_any = ty_ids[:, i] >= 0
        if not np.any(mask_any):
            continue
        
        ws = wind_speeds[mask_any, i]
        wd = wind_dirs[mask_any, i]
        valid = np.isfinite(ws) & np.isfinite(wd)
        
        if not np.any(valid):
            continue

        # 对每个风速等级
        for level, min_ws, max_ws in WIND_BINS:
            
            # 逻辑 a: 区间 (Exactly N-level)
            in_bin_mask = (ws >= min_ws) & (ws <= max_ws) & valid
            if np.any(in_bin_mask):
                binned_stats[level] += rose_bin_counts(wd[in_bin_mask], EDGES_DEG)
                
            # 逻辑 b: 阈值 (Over N-level)
            exceed_mask = (ws >= min_ws) & valid
            if np.any(exceed_mask):
                exceed_stats[level] += rose_bin_counts(wd[exceed_mask], EDGES_DEG)

    # ======== 3. 输出：分级 区间(Exactly) vs 阈值(Over) (每级统一色标) ========
    print(f"[OK] 正在输出 分级统计 (区间 vs 阈值) ...")

    for level, min_ws, max_ws in WIND_BINS:
        
        binned_counts = binned_stats[level]
        exceed_counts = exceed_stats[level]
        
        # --- CSV (区间) ---
        df_binned = pd.DataFrame({
            "bin_from_deg": EDGES_DEG[:-1],
            "bin_to_deg":   EDGES_DEG[1:],
            "hours":        binned_counts,
            "percent":      (binned_counts / max(1, binned_counts.sum())) * 100.0
        })
        csv_fname = f"Overall_WindRose_Level_{level}_Exactly_{min_ws:.1f}-{max_ws:.1f}.csv"
        df_binned.to_csv(out_csv / csv_fname, index=False, encoding="utf-8")

        # --- CSV (阈值) ---
        df_exceed = pd.DataFrame({
            "bin_from_deg": EDGES_DEG[:-1],
            "bin_to_deg":   EDGES_DEG[1:],
            "hours":        exceed_counts,
            "percent":      (exceed_counts / max(1, exceed_counts.sum())) * 100.0
        })
        csv_fname = f"Overall_WindRose_Level_{level}_Over_{min_ws:.1f}.csv"
        df_exceed.to_csv(out_csv / csv_fname, index=False, encoding="utf-8")
        
        # --- 绘图 (统一色标) ---
        # 寻找这一级 (区间 vs 阈值) 的全局最大值
        GLOBAL_VMAX_LEVEL = int(max(binned_counts.max(), exceed_counts.max(), 1))
        norm_level = mcolors.Normalize(vmin=0, vmax=GLOBAL_VMAX_LEVEL)
        cbar_label_unified = f"Hours per sector (Unified scale 0-{GLOBAL_VMAX_LEVEL})"
        
        # --- 图 (区间) ---
        title = f"Overall Wind Rose (Exactly {level}: {min_ws:.1f} - {max_ws:.1f} m/s)"
        png_fname = f"Overall_WindRose_Level_{level}_Exactly_{min_ws:.1f}-{max_ws:.1f}.png"
        plot_wind_rose_colored(binned_counts, EDGES_DEG,
                               title=title,
                               out_png=out_fig / png_fname,
                               cmap_name=CMAP_NAME, 
                               annotate=ANNOTATE_BARS, 
                               norm=norm_level, # 使用本级统一色标
                               cbar_label=cbar_label_unified)

        # --- 图 (阈值) ---
        title = f"Overall Wind Rose (Over {level}: ≥ {min_ws:.1f} m/s)"
        png_fname = f"Overall_WindRose_Level_{level}_Over_{min_ws:.1f}.png"
        plot_wind_rose_colored(exceed_counts, EDGES_DEG,
                               title=title,
                               out_png=out_fig / png_fname,
                               cmap_name=CMAP_NAME, 
                               annotate=ANNOTATE_BARS, 
                               norm=norm_level, # 使用本级统一色标
                               cbar_label=cbar_label_unified)

    print(f"[All Done] 所有输出已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()