#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
台风大风分布_空间聚类.py — 基于大风空间分布对台风过程进行 K-Means 聚类

目的与功能：
- 核心目标：不再是聚类站点，而是【聚类台风】。
- 识别出哪些台风在“8级及以上”大风的空间分布上具有相似性。
- 
- 功能：
- 1. 为每个台风生成一个“大风空间分布特征向量”，向量的维度等于总站点数，
     值=该台风在该站点的“8级及以上”大风小时数。
- 2. 构建一个 (N个台风, M个站点) 的特征矩阵。
- 3. 使用 K-Means 算法对【台风】（即矩阵的行）进行聚类。
- 4. 评估不同 K 值（聚类个数）的轮廓系数，以供参考。
- 5. 输出：
    * 聚类结果CSV：每个台风（TID, 中文名）被分到了哪个簇 (Cluster)。
    * K值评估CSV：不同 K 值对应的轮廓系数等指标。
    * 聚类地图 (PNG)：为【每个簇】生成一张“平均大风空间分布图”，
      展示“属于这一类的台风，其平均影响范围是什么样的”。

输入 (来自 统计_大风分级.py)：
- NC_PATH: 包含所有站点、所有台风原始风速数据的 NetCDF 文件。
- SHP_CITY_PATH: 市界 Shapefile 路径（用于绘图）。

输出 (保存至新的子目录)：
- {BASE_OUTPUT_DIR}/输出_台风聚类_8级及以上/
    * Typhoon_Cluster_Assignments_8级及以上.csv (台风归类表)
    * Typhoon_K_Metrics_8级及以上.csv (K值评估表)
    * (K=N, Cluster=M).png (K=N时，第M类的平均大风分布图)

"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from netCDF4 import Dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

plt.rcParams['font.sans-serif'] = ['Heiti TC']

# ======= 1. NetCDF 路径 (来自 统计_大风分级.py) =======
NC_PATH = r"/Users/momo/Desktop/业务相关/2025 影响台风大风/数据/Refined_Combine_Stations_ExMaxWind_Fixed.nc"

# ======= 2. Shapefile 路径 (来自 统计_大风分级.py) =======
SHP_CITY_PATH = r"/Users/momo/Desktop/业务相关/2025 影响台风大风/地形文件/shapefile/市界/浙江市界.shp"

# ======= 3. 基础输出目录 (来自 统计_大风分级.py) =======
# 我们将在此目录下创建一个新的子目录
BASE_OUTPUT_DIR = Path("/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_大风分级统计")

# ======= 4. 【新】聚类任务配置 =======

# --- 我们要分析哪个级别？---
# 先从 "8级及以上" 开始
LEVEL_CONFIG = {
    "thresh_min": 17.2,
    "thresh_max": np.inf,
    "name": "8级及以上",
    "output_subdir": f"输出_台风聚类_Kmeans_8级及以上" 
}

# --- K-Means 测试的 K 值范围 (来自 大风分布聚类_指定级别.py) ---
K_RANGE = range(2, 7) # 测试 k=2, 3, 4, 5, 6

# --- 绘图参数 (来自 统计_大风分级.py) ---
EXTENT = [118, 123, 27, 31.5] # 设为 None 则自适应
TEXT_SIZE = 8
CMAP_NAME = "viridis"
SHOW_ZERO = False 


# ----------------------------- 辅助函数 (来自 统计_大风分级.py) -----------------------------
def parse_mapping(attr_str: str):
    if not attr_str:
        return {}
    pairs = (p for p in attr_str.strip().split(";") if ":" in p)
    return {k.strip(): v.strip() for k, v in (q.split(":", 1) for q in pairs)}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sanitize_filename(name: str) -> str:
    return re.sub(r"[\\/:*?\"<>|\\s]+", "_", name).strip("_")

def draw_station_count_text_map(
    lons, lats, counts, stids, title, out_png,
    threshold_val, # 用于色标 (这里可以理解为均值的色标)
    extent=None, text_size=8, cmap_name="viridis", show_zero=True,
    vmin=None, vmax=None
):
    """
    用“彩色数字”标注每个站点的【平均】超阈值小时数。
    (从 统计_大风分级.py 复制而来)
    
    【重要修改】: 
    - counts 可能是浮点数 (平均值)，所以标注时改为 f"{val:.1f}"
    """
    fig = plt.figure(figsize=(10, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_title(title, fontsize=14)

    # 底图要素
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
    ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')

    # 加载并绘制市界
    try:
        city_shapes = list(shpreader.Reader(SHP_CITY_PATH).geometries())
        ax.add_geometries(
            city_shapes, ccrs.PlateCarree(),
            edgecolor='gray', facecolor='none',
            linewidth=0.5, linestyle='--'
        )
    except Exception as e:
        print(f"\n[WARN] 无法加载市界 Shapefile: {SHP_CITY_PATH}")
        print(f"[WARN] 错误: {e}")
        pass

    # 范围
    if extent and len(extent) == 4:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        pad_lon = max(0.5, (np.nanmax(lons) - np.nanmin(lons)) * 0.1)
        pad_lat = max(0.5, (np.nanmax(lats) - np.nanmin(lats)) * 0.1)
        ax.set_extent([np.nanmin(lons)-pad_lon, np.nanmax(lons)+pad_lon,
                       np.nanmin(lats)-pad_lat, np.nanmax(lats)+pad_lat],
                      crs=ccrs.PlateCarree())

    # 颜色映射
    if vmin is None:
        vmin = np.nanmin(counts)
    if vmax is None:
        vmax = np.nanmax(counts)
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax):
        vmax = 1.0
    if vmin == vmax:
        vmax = vmin + 1.0

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    # 逐站点标注彩色数字
    for x, y, val in zip(lons, lats, counts):
        if (not show_zero) and (val == 0):
            continue
        color = cmap(norm(val if np.isfinite(val) else 0.0))
        
        # ---【核心修改】---
        # 因为可能是平均值(float)，所以用 .1f 格式化，而不是 int(val)
        # 如果值太小，显示 0.0，否则显示具体数值
        display_val = f"{val:.1f}" if val > 0.05 else "0.0"
        if (not show_zero) and (display_val == "0.0"):
             continue
             
        txt = ax.text(
            x, y, display_val, fontsize=text_size,
            ha='center', va='center', color=color,
            transform=ccrs.PlateCarree()
        )
        # ---【结束修改】---
        
        txt.set_path_effects([
            pe.withStroke(linewidth=1.5, foreground="white")
        ])

    # Colorbar
    # --- 【修正 2】 ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label(f"平均统计小时数 (m/s)")

    # 网格
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, crs=ccrs.PlateCarree())
    gl.right_labels = False
    gl.top_labels = False

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# ----------------------------- 【新】主逻辑 -----------------------------
def main():
    
    # 1. 定义配置和输出目录
    config = LEVEL_CONFIG
    level_name = config['name']
    thresh_min = config['thresh_min']
    thresh_max = config['thresh_max']
    
    output_dir = BASE_OUTPUT_DIR / config['output_subdir']
    ensure_dir(output_dir)
    
    print(f"{'='*70}")
    print(f"--- 任务：基于“{level_name}”大风空间分布的【台风聚类】---")
    print(f"--- 输出目录: {output_dir.resolve()} ---")
    print(f"{'='*70}")

    # 2. 读数据 (来自 统计_大风分级.py)
    print(f"正在读取 NetCDF 文件: {NC_PATH}")
    nc = Dataset(NC_PATH)

    id_to_index = parse_mapping(getattr(nc, 'id_to_index', None))
    index_to_cn = parse_mapping(getattr(nc, 'index_to_cn', None))
    index_to_en = parse_mapping(getattr(nc, 'index_to_en', None))

    stids = np.array(nc.variables['STID'][:]).astype(str)
    lats = np.array(nc.variables['lat'][:], dtype=float)
    lons = np.array(nc.variables['lon'][:], dtype=float)
    
    wind_speeds = np.array(nc.variables['wind_velocity'][:, 0, :], copy=True)
    ty_ids = np.array(nc.variables['typhoon_id_index'][:, 0, :], copy=True)
    if np.issubdtype(ty_ids.dtype, np.floating):
        ty_ids_int = np.full_like(ty_ids, fill_value=-1, dtype=int)
        valid = ~np.isnan(ty_ids)
        ty_ids_int[valid] = ty_ids[valid].astype(int)
        ty_ids = ty_ids_int
    else:
        ty_ids = ty_ids.astype(int)

    n_time, n_sta = wind_speeds.shape

    if id_to_index:
        items = sorted([(tid, int(str(idx).strip())) for tid, idx in id_to_index.items() if str(idx).strip().isdigit()], key=lambda kv: kv[0])
    else:
        uniq = sorted({int(x) for x in np.unique(ty_ids) if int(x) >= 0})
        items = [(str(idx), idx) for idx in uniq]
    
    print(f"数据读取完毕。共 {n_sta} 个站点，{len(items)} 个台风过程。")

    # 3. 【核心】构建 (N台风, M站点) 特征矩阵
    print(f"\n正在为 {len(items)} 个台风构建“{level_name}”空间分布特征向量...")
    
    feature_vectors = []
    typhoon_metadata = [] # 存储台风信息
    
    for tid_str, ty_idx in items:
        
        # 为这个台风创建一个 M站 维度的向量
        typhoon_hours_vector = np.zeros(n_sta, dtype=float)
        
        for i in range(n_sta): # 遍历 M 个站点
            mask_ty = (ty_ids[:, i] == ty_idx)
            if not np.any(mask_ty):
                typhoon_hours_vector[i] = 0.0
                continue
            
            ws = wind_speeds[mask_ty, i]
            
            # 统计 "8级及以上" 小时数
            mask_wind = (ws >= thresh_min) & (ws <= thresh_max)
            hours = int(np.sum(mask_wind))
            typhoon_hours_vector[i] = hours
            
        # 仅当该台风在至少一个站点造成了 > 0 小时的大风时，才将其纳入聚类
        # (否则所有 0 向量会聚成无意义的一类)
        if np.sum(typhoon_hours_vector) > 0:
            feature_vectors.append(typhoon_hours_vector)
            typhoon_metadata.append({
                "TID": tid_str,
                "Index": ty_idx,
                "CN_Name": index_to_cn.get(str(ty_idx), ""),
                "EN_Name": index_to_en.get(str(ty_idx), "")
            })
        else:
            print(f"  [INFO] 台风 {tid_str} ({index_to_cn.get(str(ty_idx), '')}) 无 {level_name} 影响, 已跳过。")

    # 4. 准备聚类
    
    # 原始特征矩阵 (N_valid_typhoons, M_stations)
    X = np.array(feature_vectors)
    # 描述台风信息的 DataFrame
    df_typhoons_meta = pd.DataFrame(typhoon_metadata)

    if X.shape[0] < min(K_RANGE):
        print(f"[ERROR] 有效台风数 ({X.shape[0]}) 少于最小 K 值 ({min(K_RANGE)})，无法执行聚类。")
        return

    print(f"\n特征矩阵构建完毕: (N_台风 = {X.shape[0]}, M_站点 = {X.shape[1]})")

    # 标准化 (非常重要，K-Means 依赖距离)
    print("正在对特征矩阵进行标准化...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. K-Means 聚类 (循环测试 K)
    print(f"正在评估 K 值 {list(K_RANGE)} 并为每个K保存结果...")
    
    metrics_list = []
    
    for k in K_RANGE:
        print(f"\n--- 正在处理 K={k} ---")
        
        # A. 执行聚类
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertia = kmeans.inertia_
        
        try:
            score = silhouette_score(X_scaled, labels)
        except ValueError:
            score = float('nan')
        
        metrics_list.append({"k": k, "Inertia": inertia, "Silhouette_Score": score})
        
        # B. 将标签存入台风信息表
        df_typhoons_meta[f'Cluster_k{k}'] = labels
        
        print(f"  K={k} 聚类完毕. 轮廓系数: {score:.4f}")
        
        # C. 【可视化】为 K={k} 的每个簇绘制“平均空间分布图”
        print(f"  正在为 K={k} 的 {k} 个簇绘制平均分布图...")
        
        for cluster_id in range(k):
            # 找出这个簇包含的所有台风的【索引】
            indices_in_cluster = np.where(labels == cluster_id)[0]
            
            if len(indices_in_cluster) == 0:
                print(f"    [WARN] K={k}, 簇 {cluster_id} 为空，跳过绘图。")
                continue
                
            # 从【原始】特征矩阵 X 中提取这些台风的向量
            cluster_vectors = X[indices_in_cluster]
            
            # 计算“平均空间分布” (沿台风轴求均值)
            avg_footprint = np.mean(cluster_vectors, axis=0)
            
            # 绘图
            n_typhoons = len(indices_in_cluster)
            title = f"台风聚类: {level_name} (K={k}, 簇={cluster_id})\n包含 {n_typhoons} 个台风的【平均】空间分布"
            
            # 文件名
            safe_name = sanitize_filename(level_name)
            fname = f"Typhoon_Cluster_{safe_name}_K{k}_C{cluster_id}_AvgFootprint.png"
            png_path = output_dir / fname
            
            # --- 【修正 1】 ---
            draw_station_count_text_map(
                lons, lats, avg_footprint, stids, title, str(png_path),
                threshold_val=thresh_min, # 使用 17.2 作为色标基准
                extent=EXTENT, text_size=TEXT_SIZE, cmap_name=CMAP_NAME, show_zero=SHOW_ZERO
            )
            print(f"    [OK] 已保存地图: {fname}")

    # 6. 保存最终结果
    print(f"\n{'='*50}")
    print("--- 聚类完成，正在保存汇总文件 ---")
    
    # A. K值评估
    df_metrics = pd.DataFrame(metrics_list)
    metrics_csv_path = output_dir / f"Typhoon_K_Metrics_{sanitize_filename(level_name)}.csv"
    df_metrics.to_csv(metrics_csv_path, index=False, encoding='utf-8-sig')
    print(f"[OK] K值评估指标已保存: {metrics_csv_path.resolve()}")
    
    # B. 台风聚类结果
    # df_typhoons_meta 现在包含: TID, CN_Name, EN_Name, Cluster_k2, Cluster_k3, ...
    assignments_csv_path = output_dir / f"Typhoon_Cluster_Assignments_{sanitize_filename(level_name)}.csv"
    df_typhoons_meta.to_csv(assignments_csv_path, index=False, encoding='utf-8-sig')
    print(f"[OK] 台风聚类归属表已保存: {assignments_csv_path.resolve()}")
    
    print(f"{'='*50}")
    print("--- 所有台风聚类任务完成 ---")


if __name__ == "__main__":
    main()