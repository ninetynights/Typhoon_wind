#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按台风路径类别分析多个大风等级的风玫瑰并叠加地形.py
------------------------------------------------------------------------------------------------
主要功能：
- [新] 批处理：可一次性循环分析多个大风等级 (8级-15级)
- [新] 动态输出：为每个大风等级自动生成带等级名称的文件夹和图表标题
- [优化] 采用非线性颜色映射(PowerNorm)和高对比度色图(YlOrRd)，增强颜色分辨度。
- [优化] 在每个风玫瑰后增加白色光环，解决重叠问题，提升视觉分离度。
- 全局统一色标 (在单个风力等级内)，确保跨类别可比性。
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
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource # 确保引入 LightSource
import scipy.ndimage as ndimage # 引入 scipy 进行高程数据平滑，效果更好

# --- 配置区 ---

# 1. 输入文件路径
# Kmeans 增加经纬度特征 去除圆规 选择K=3 后的路径分类映射文件
CLUSTER_MAP_CSV = r"/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_路径分类/输出_影响段聚类_Kmeans_增加经纬度特征_去除圆规_K=3/影响段_簇映射.csv"
# HDBSCAN + UMAP 去除圆规 min samples=2 后的路径分类映射文件
#CLUSTER_MAP_CSV = r"/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_路径分类/输出_影响段聚类_HDBSCAN_优化版_去除圆规_MINSAMPLE=2_计算DBCV/影响段_簇映射.csv"
NC_PATH         = r"/Users/momo/Desktop/业务相关/2025 影响台风大风/All_Typhoons_ExMaxWind.nc"
DEM_PATH        = r"/Users/momo/Desktop/业务相关/2025 影响台风大风/地形文件/DEM_0P05_CHINA.nc"

# 2. 基础输出目录 (不含风力等级)
BASE_OUTPUT_DIR = Path("/Users/momo/Desktop/业务相关/2025 影响台风大风")

# 3. 参数设置
N_BINS         = 16
EDGES_DEG      = np.linspace(0, 360, N_BINS + 1)
MAP_EXTENT     = [117.5, 123.5, 26.5, 31.5]
ROSE_SCALE     = 0.3  # 可以尝试 0.25, 0.3, 0.35 等值

# 4. 绘图样式
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False


# --- 工具函数 (保持不变) ---

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parse_nc_mappings(nc_file):
    id_to_index_str = nc_file.getncattr('id_to_index')
    id_to_index = {k.strip(): int(v.strip()) for k, v in (p.split(":") for p in id_to_index_str.split(";") if ":" in p)}
    index_to_id = {v: k for k, v in id_to_index.items()}
    return index_to_id

def add_topography_to_map(ax, dem_path, map_extent):
    """
    [改进版 v2] 
    1. 使用 xarray.interp() 进行三次样条插值，解决马赛克问题
    2. 使用晕渲（Hillshade）地形图，增强立体感
    """
    try:
        ds_dem = xr.open_dataset(dem_path)
        dem_var_name = 'dhgt_gfs' if 'dhgt_gfs' in ds_dem else list(ds_dem.data_vars)[0]
        dem_data = ds_dem[dem_var_name]
        
        # 1. 裁剪原始低分辨率数据 (比地图范围稍大一点)
        dem_cropped_lowres = dem_data.sel(
            Lon=slice(map_extent[0] - 0.5, map_extent[1] + 0.5),
            Lat=slice(map_extent[2] - 0.5, map_extent[3] + 0.5)
        )
        
        # --- 核心改进：插值升采样 ---
        
        # 2. 定义一个新的、更高分辨率的网格
        interp_factor = 10
        orig_lons = dem_cropped_lowres.Lon.values
        orig_lats = dem_cropped_lowres.Lat.values
        
        new_lons = np.linspace(orig_lons.min(), orig_lons.max(), num=len(orig_lons) * interp_factor)
        new_lats = np.linspace(orig_lats.min(), orig_lats.max(), num=len(orig_lats) * interp_factor)

        # 3. 执行插值
        print(f"[提示] 正在对 DEM 数据进行 {interp_factor}x 插值，使其更平滑...")
        dem_interp_highres = dem_cropped_lowres.interp(
            Lon=new_lons, 
            Lat=new_lats, 
            method='cubic'
        )
        
        # 4. 从插值后的高分辨率数据中获取网格和高程
        lons_dem, lats_dem = dem_interp_highres.Lon.values, dem_interp_highres.Lat.values
        elevation = dem_interp_highres.values
        elevation = np.nan_to_num(elevation)
        
        # --- 核心改进：创建晕渲图 (与之前相同) ---
        elevation_smoothed = ndimage.gaussian_filter(elevation, sigma=2.0)
        ls = LightSource(azdeg=315, altdeg=45)
        rgb = ls.shade(elevation_smoothed, 
                       cmap=plt.get_cmap('terrain'), 
                       vert_exag=1.5,
                       blend_mode='soft',
                       norm=mcolors.Normalize(vmin=0, vmax=1500)
                      )
        
        # 8. 在地图上绘制晕渲图
        ax.imshow(rgb, 
                  origin='upper', 
                  extent=[lons_dem.min(), lons_dem.max(), lats_dem.min(), lats_dem.max()],
                  transform=ccrs.PlateCarree(), 
                  zorder=1,
                  alpha=0.75
                 )
        ds_dem.close()

    except Exception as e:
        print(f"[警告] 添加(改进版V2 - 插值)地形背景失败: {e}")

def plot_rose_on_map(ax_map, lon, lat, counts, edges_deg, norm, cmap, scale_factor=0.5):
    """在地图上绘制带光环的迷你风玫瑰图"""
    if np.sum(counts) == 0:
        return

    total_hours = np.sum(counts)
    # 动态调整大小
    dynamic_scale = scale_factor * (0.6 + 0.4 * np.sqrt(total_hours / 50.0))

    # --- 【改进4】增加白色背景光环 ---
    halo_radius_deg = dynamic_scale * 1.15
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


# --- 主逻辑 (已参数化) ---

def analyze_and_plot_rose_by_cluster(wind_threshold, level_name):
    """
    参数化的主分析函数
    
    Args:
        wind_threshold (float): 用于筛选的风速阈值 (m/s)
        level_name (str): 风力等级的名称 (如 "8级及以上")
    """
    
    # 1. [新] 根据风力等级动态创建输出目录
    output_folder_name = f"输出_Kmeans_各路径类别大风玫瑰分析_{level_name}"
    current_output_dir = BASE_OUTPUT_DIR / output_folder_name
    ensure_dir(current_output_dir)
    csv_out_dir = current_output_dir / "csv"
    fig_out_dir = current_output_dir / "figs"
    ensure_dir(csv_out_dir)
    ensure_dir(fig_out_dir)
    print(f"输出目录已设置为: {current_output_dir}")

    # 2. 读取路径分类数据
    print(f"正在读取路径分类文件: {CLUSTER_MAP_CSV}")
    df_clusters = pd.read_csv(CLUSTER_MAP_CSV)
    tid_to_cluster = df_clusters.set_index('tid')['cluster_id'].to_dict()
    cluster_ids = sorted(df_clusters['cluster_id'].unique())
    print(f"识别到 {len(cluster_ids)} 个路径类别: {cluster_ids}")

    # 3. 读取NetCDF数据
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

    # 4. [新] 筛选并统计指定等级大风的风向
    print(f"正在筛选风速 >= {wind_threshold}m/s ({level_name}) 的数据并统计风向...")
    cluster_wind_dirs = {cid: {s: [] for s in stids} for cid in cluster_ids}

    strong_wind_mask = wind_speeds >= wind_threshold # 使用传入的参数
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

    # 5. 计算所有类别的风玫瑰数据，并找到全局最大值以统一色标
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

    print(f"[{level_name}] 级别计算完成，用于统一色标的全局最大扇区小时数为: {GLOBAL_VMAX}")

    # 6. 使用统一色标生成CSV和地图
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=GLOBAL_VMAX)
    cmap = plt.get_cmap('YlOrRd')

    for cid in cluster_ids:
        print(f"\n--- 正在为路径类别 {cid} (风力 {level_name}) 生成结果 ---")
        if cid not in cluster_rose_data:
            print(f"类别 {cid} 没有站点出现 {level_name} 大风，跳过。")
            continue

        data = cluster_rose_data[cid]
        rose_counts_matrix = data["matrix"]
        s_indices = data["s_indices"]

        # 保存CSV (路径已在步骤1中动态设置)
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

        # [新] 动态设置标题
        ax.set_title(f"路径类别 {cid} 的{level_name}大风风玫瑰图", fontsize=16, pad=20)

        fig_path = fig_out_dir / f"Cluster_{cid}_StrongWind_Rose_Map.png"
        fig.savefig(fig_path, dpi=200, bbox_inches='tight')
        print(f"已保存风玫瑰地图到: {fig_path}")
        plt.close(fig)


# --- [新] 主程序入口 (循环调用) ---

if __name__ == "__main__":
    
    # 1. 定义你需要的风力等级和对应的风速阈值 (m/s)
    WIND_LEVELS_TO_PROCESS = {
        "8级及以上": 17.2,
        "9级及以上": 20.8,
        "10级及以上": 24.5,
        "11级及以上": 28.5,
        "12级及以上": 32.7,
        "13级及以上": 37.0,
        "14级及以上": 41.5,
        "15级及以上": 46.2,
    }
    
    # 2. 循环处理每个等级
    print(f"--- 即将开始批处理 {len(WIND_LEVELS_TO_PROCESS)} 个风力等级 ---")
    
    for level_name, threshold in WIND_LEVELS_TO_PROCESS.items():
        print(f"\n=======================================================")
        print(f"      开始处理风力等级: {level_name} (>= {threshold} m/s)")
        print(f"=======================================================")
        
        try:
            analyze_and_plot_rose_by_cluster(
                wind_threshold=threshold,
                level_name=level_name
            )
        except Exception as e:
            print(f"[!!! 严重错误 !!!] 处理 {level_name} 时发生失败: {e}")
            # 你可以选择在这里 'continue' (继续下一个) 或 'break' (终止)
            continue
            
        print(f"--- 风力等级: {level_name} 处理完成 ---")

    print("\n[--- 所有任务完成 ---]")