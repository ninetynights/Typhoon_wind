#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按台风路径类别分析8级以上大风风玫瑰并叠加地形.py
------------------------------------------------------------------------------------------------
主要功能：
- [优化] 采用非线性颜色映射(PowerNorm)和高对比度色图(YlOrRd)，增强颜色分辨度。
- [优化] 在每个风玫瑰后增加白色光环，解决重叠问题，提升视觉分离度。
- 全局统一色标，确保跨类别可比性。
- 精细化风玫瑰绘制，效果美观。
- 为每个路径类别（簇）生成CSV和叠加地形的地图。
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from netCDF4 import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import xarray as xr

# --- 配置区 ---
# 1. 输入文件路径
CLUSTER_MAP_CSV = r"/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_影响段聚类(修改轨迹点和弧长)/影响段_簇映射.csv"
NC_PATH         = r"/Users/momo/Desktop/业务相关/2025 影响台风大风/All_Typhoons_ExMaxWind.nc"
DEM_PATH        = r"/Users/momo/Desktop/业务相关/2025 影响台风大风/地形文件/DEM_0P05_CHINA.nc"

# 2. 输出目录
OUTPUT_DIR = Path("./输出_各路径类别大风玫瑰分析_15级及以上")

# 3. 参数设置
WIND_THRESHOLD = 46.2
N_BINS         = 16
EDGES_DEG      = np.linspace(0, 360, N_BINS + 1)
MAP_EXTENT     = [117.5, 123.5, 26.5, 31.5]
# --- 【改进3】调整基础缩放因子 ---
ROSE_SCALE     = 0.3  # 可以尝试 0.25, 0.3, 0.35 等值

# 4. 绘图样式
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False


# --- 工具函数 ---

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parse_nc_mappings(nc_file):
    id_to_index_str = nc_file.getncattr('id_to_index')
    id_to_index = {k.strip(): int(v.strip()) for k, v in (p.split(":") for p in id_to_index_str.split(";") if ":" in p)}
    index_to_id = {v: k for k, v in id_to_index.items()}
    return index_to_id

def add_topography_to_map(ax, dem_path, map_extent):
    try:
        ds_dem = xr.open_dataset(dem_path)
        dem_var_name = 'dhgt_gfs' if 'dhgt_gfs' in ds_dem else list(ds_dem.data_vars)[0]
        dem_data = ds_dem[dem_var_name]
        dem_cropped = dem_data.sel(
            Lon=slice(map_extent[0] - 0.5, map_extent[1] + 0.5),
            Lat=slice(map_extent[2] - 0.5, map_extent[3] + 0.5)
        )
        lons_dem, lats_dem = dem_cropped.Lon.values, dem_cropped.Lat.values
        levels = np.arange(0, 1501, 100)
        ax.contourf(lons_dem, lats_dem, dem_cropped.values, levels=levels, cmap='Greys', alpha=0.6, transform=ccrs.PlateCarree(), zorder=1)
        ds_dem.close()
    except Exception as e:
        print(f"[警告] 添加地形背景失败: {e}")

def plot_rose_on_map(ax_map, lon, lat, counts, edges_deg, norm, cmap, scale_factor=0.5):
    """在地图上绘制带光环的迷你风玫瑰图"""
    if np.sum(counts) == 0:
        return

    total_hours = np.sum(counts)
    # 动态调整大小
    dynamic_scale = scale_factor * (0.6 + 0.4 * np.sqrt(total_hours / 50.0))

    # --- 【改进4】增加白色背景光环 ---
    # 光环比玫瑰图本身稍大一点，形成一个边界
    halo_radius_deg = dynamic_scale * 1.15
    # 用一个大的散点来模拟圆
    ax_map.scatter(lon, lat, s=(halo_radius_deg * 72)**2, color='white', alpha=0.6, zorder=8, transform=ccrs.PlateCarree())


    edges_rad = np.deg2rad(edges_deg)
    angles_rad = edges_rad[:-1]
    widths_rad = np.diff(edges_rad)
    
    max_radius_in_rose = max(1, np.max(counts))
    radii = (counts / max_radius_in_rose) * dynamic_scale
    
    colors = cmap(norm(counts))

    for i in range(len(angles_rad)):
        if counts[i] > 0:
            start_angle = angles_rad[i]
            end_angle = angles_rad[i] + widths_rad[i]
            
            theta = np.linspace(start_angle, end_angle, 20)
            r = np.linspace(0, radii[i], 2)
            theta_grid, r_grid = np.meshgrid(theta, r)

            x_offset = r_grid * np.sin(theta_grid)
            y_offset = r_grid * np.cos(theta_grid)
            
            x_points = lon + x_offset
            y_points = lat + y_offset

            ax_map.fill(
                np.concatenate([x_points[0, :], x_points[1, ::-1]]),
                np.concatenate([y_points[0, :], y_points[1, ::-1]]),
                facecolor=colors[i],
                edgecolor='black',
                linewidth=0.2,
                alpha=0.95,
                transform=ccrs.PlateCarree(),
                zorder=10
            )


# --- 主逻辑 (与之前版本基本一致) ---

def analyze_and_plot_rose_by_cluster():
    ensure_dir(OUTPUT_DIR)
    csv_out_dir = OUTPUT_DIR / "csv"
    fig_out_dir = OUTPUT_DIR / "figs"
    ensure_dir(csv_out_dir)
    ensure_dir(fig_out_dir)

    # 1. 读取路径分类数据
    print(f"正在读取路径分类文件: {CLUSTER_MAP_CSV}")
    df_clusters = pd.read_csv(CLUSTER_MAP_CSV)
    tid_to_cluster = df_clusters.set_index('tid')['cluster_id'].to_dict()
    cluster_ids = sorted(df_clusters['cluster_id'].unique())
    print(f"识别到 {len(cluster_ids)} 个路径类别: {cluster_ids}")

    # 2. 读取NetCDF数据
    print(f"正在读取NetCDF数据: {NC_PATH}")
    nc = Dataset(NC_PATH)
    index_to_tid = parse_nc_mappings(nc)
    stids = np.array(nc.variables['STID'][:]).astype(str)
    lats = np.array(nc.variables['lat'][:])
    lons = np.array(nc.variables['lon'][:])
    wind_speeds = np.array(nc.variables['wind_velocity'][:, 0, :])
    wind_dirs = np.array(nc.variables['wind_direction'][:, 0, :])
    ty_indices = np.array(nc.variables['typhoon_id_index'][:, 0, :]).astype(int)
    nc.close()

    # 3. 筛选并统计8级以上大风的风向
    print(f"正在筛选风速 >= {WIND_THRESHOLD}m/s 的数据并统计风向...")
    cluster_wind_dirs = {cid: {s: [] for s in stids} for cid in cluster_ids}

    strong_wind_mask = wind_speeds >= WIND_THRESHOLD
    t_indices, s_indices = np.where(strong_wind_mask)

    for t_idx, s_idx in zip(t_indices, s_indices):
        ty_idx = ty_indices[t_idx, s_idx]
        if ty_idx < 0: continue
        
        tid = index_to_tid.get(ty_idx)
        if tid is None: continue

        cluster_id = tid_to_cluster.get(int(tid))
        if cluster_id is None: continue

        wd = wind_dirs[t_idx, s_idx]
        if not np.isnan(wd):
            cluster_wind_dirs[cluster_id][stids[s_idx]].append(wd)

    # 4. 计算所有类别的风玫瑰数据，并找到全局最大值以统一色标
    print("正在计算所有类别的风玫瑰数据...")
    cluster_rose_data = {}
    GLOBAL_VMAX = 1

    for cid in cluster_ids:
        rose_counts_list = []
        station_indices_with_wind = []
        for s_idx, stid in enumerate(stids):
            dirs = np.array(cluster_wind_dirs[cid][stid])
            if dirs.size > 0:
                hist, _ = np.histogram(np.mod(dirs, 360.0), bins=EDGES_DEG)
                rose_counts_list.append(hist)
                station_indices_with_wind.append(s_idx)
        
        if rose_counts_list:
            rose_matrix = np.array(rose_counts_list)
            cluster_rose_data[cid] = {
                "matrix": rose_matrix,
                "s_indices": station_indices_with_wind
            }
            current_max = rose_matrix.max()
            if current_max > GLOBAL_VMAX:
                GLOBAL_VMAX = current_max

    print(f"所有类别计算完成，用于统一色标的全局最大扇区小时数为: {GLOBAL_VMAX}")

    # 5. 使用统一色标生成CSV和地图
    # --- 【改进2】使用PowerNorm进行非线性颜色映射 ---
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=GLOBAL_VMAX)
    # --- 【改进1】换用高对比度色图 ---
    cmap = plt.get_cmap('YlOrRd')

    for cid in cluster_ids:
        print(f"\n--- 正在为路径类别 {cid} 生成结果 ---")
        if cid not in cluster_rose_data:
            print(f"类别 {cid} 没有站点出现8级以上大风，跳过。")
            continue

        data = cluster_rose_data[cid]
        rose_counts_matrix = data["matrix"]
        s_indices = data["s_indices"]

        # 保存CSV
        bin_headers = [f'deg_{int(d1)}-{int(d2)}' for d1, d2 in zip(EDGES_DEG[:-1], EDGES_DEG[1:])]
        df_csv = pd.DataFrame(rose_counts_matrix, columns=bin_headers)
        df_csv.insert(0, 'STID', stids[s_indices])
        df_csv.insert(1, 'lon', lons[s_indices])
        df_csv.insert(2, 'lat', lats[s_indices])
        df_csv['total_hours'] = df_csv[bin_headers].sum(axis=1)
        csv_path = csv_out_dir / f"Cluster_{cid}_StrongWind_Rose.csv"
        df_csv.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"已保存风玫瑰数据到: {csv_path}")

        # 绘图
        fig = plt.figure(figsize=(12, 12))
        proj = ccrs.PlateCarree()
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_extent(MAP_EXTENT, crs=proj)

        add_topography_to_map(ax, DEM_PATH, MAP_EXTENT)
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), lw=0.8, zorder=5)
        ax.add_feature(cfeature.OCEAN, zorder=0, facecolor='#cbe4f5')
        ax.add_feature(cfeature.BORDERS, linestyle='-', lw=0.5, zorder=5)
        provinces = cfeature.NaturalEarthFeature(
            category='cultural', name='admin_1_states_provinces_lines',
            scale='10m', facecolor='none')
        ax.add_feature(provinces, edgecolor='gray', linestyle='--', linewidth=0.6, zorder=5)

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False; gl.right_labels = False
        gl.xformatter = LongitudeFormatter(); gl.yformatter = LatitudeFormatter()
        
        for i, s_idx in enumerate(s_indices):
            plot_rose_on_map(
                ax, lons[s_idx], lats[s_idx], rose_counts_matrix[i],
                EDGES_DEG, norm, cmap, scale_factor=ROSE_SCALE
            )
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.05, shrink=0.6)
        cbar.set_label(f"单个扇区内的大风小时数 (全局统一色标, max={GLOBAL_VMAX})", fontsize=10)

        ax.set_title(f"路径类别 {cid} 的15级以上大风风玫瑰图", fontsize=16, pad=20)

        fig_path = fig_out_dir / f"Cluster_{cid}_StrongWind_Rose_Map.png"
        fig.savefig(fig_path, dpi=200, bbox_inches='tight')
        print(f"已保存风玫瑰地图到: {fig_path}")
        plt.close(fig)

if __name__ == "__main__":
    analyze_and_plot_rose_by_cluster()
    print("\n[--- 所有任务完成 ---]")