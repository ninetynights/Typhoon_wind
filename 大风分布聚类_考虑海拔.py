"""
大风分布聚类_海拔优化版.py — 引入海拔高度特征的空间约束聚类

【核心升级】：
1. 特征维度扩展：聚类特征从 [经度, 纬度, 时长] 升级为 [经度, 纬度, 海拔, 时长]。
   - 效果：能有效区分"高山大风区"和"平原大风区"，使山区和平原的界限更清晰。
2. 数据清洗：自动检测并修复 CSV 中的无效高度值 (-999.9)。

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# 引入聚类相关的库
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 引入地图绘制库
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

# ==========================================
# 1. 全局配置 (Config)
# ==========================================

plt.rcParams['font.sans-serif'] = ['Heiti TC'] # Mac: Heiti TC, Win: SimHei
plt.rcParams['axes.unicode_minus'] = False

# --- 文件路径配置 ---
CSV_EXCEED_PATH = "/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_大风分级统计/AllTyphoons_Exceed.csv"
CSV_EXACT_PATH  = "/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_大风分级统计/AllTyphoons_Exact.csv"
SHP_CITY_PATH   = r"/Users/momo/Desktop/业务相关/2025 影响台风大风/地形文件/shapefile/市界/浙江市界.shp"

# 输出根目录
OUTPUT_DIR = Path("/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_大风分级统计/大风累积空间聚类结果(海拔优化版)") 

# --- 任务定义 ---
ANALYSIS_TASKS = [
    # --- 8级 (17.2 m/s) ---
    {
        "file": CSV_EXCEED_PATH,
        "column": "TotalHours_gt_17.2",
        "name": "8级及以上",
        "output_subdir": "超阈值 (Exceed)"
    },
    {
        "file": CSV_EXACT_PATH,
        "column": "TotalHours_eq_17.2",
        "name": "8级",
        "output_subdir": "指定级别 (Exact)"
    },
    
    # --- 9级 (20.8 m/s) ---
    {
        "file": CSV_EXCEED_PATH,
        "column": "TotalHours_gt_20.8",
        "name": "9级及以上",
        "output_subdir": "超阈值 (Exceed)"
    },
    {
        "file": CSV_EXACT_PATH,
        "column": "TotalHours_eq_20.8",
        "name": "9级",
        "output_subdir": "指定级别 (Exact)"
    },

    # --- 10级 (24.5 m/s) ---
    {
        "file": CSV_EXCEED_PATH,
        "column": "TotalHours_gt_24.5",
        "name": "10级及以上",
        "output_subdir": "超阈值 (Exceed)"
    },
    {
        "file": CSV_EXACT_PATH,
        "column": "TotalHours_eq_24.5",
        "name": "10级",
        "output_subdir": "指定级别 (Exact)"
    },
    
    # --- 11级 (28.5 m/s) ---
    {
        "file": CSV_EXCEED_PATH,
        "column": "TotalHours_gt_28.5",
        "name": "11级及以上",
        "output_subdir": "超阈值 (Exceed)"
    },
    {
        "file": CSV_EXACT_PATH,
        "column": "TotalHours_eq_28.5",
        "name": "11级",
        "output_subdir": "指定级别 (Exact)"
    },
    
    # --- 12级 (32.7 m/s) ---
    {
        "file": CSV_EXCEED_PATH,
        "column": "TotalHours_gt_32.7",
        "name": "12级及以上",
        "output_subdir": "超阈值 (Exceed)"
    },
    {
        "file": CSV_EXACT_PATH,
        "column": "TotalHours_eq_32.7",
        "name": "12级",
        "output_subdir": "指定级别 (Exact)"
    }
]

# K值范围
K_RANGE = range(2, 7) 
# 空间约束邻居数
N_NEIGHBORS = 15 

# ==========================================
# 2. 主逻辑
# ==========================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"🚀 任务开始，总输出目录: {OUTPUT_DIR.resolve()}")

for task in ANALYSIS_TASKS:
    file_path = Path(task['file'])
    column = task['column']
    name = task['name']
    
    task_output_dir = OUTPUT_DIR / task['output_subdir']
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"正在处理任务: [{name}] | 特征包含: [Lon, Lat, Height, Hours]")
    print(f"{'='*70}")
    
    try:
        # 1. 加载数据
        df = pd.read_csv(file_path)
        if df.empty: continue
        
        # --- 【新】数据清洗：处理 Height 列 ---
        # 检查是否有 -999.9 等无效值
        invalid_mask = df['Height'] < -500
        if invalid_mask.any():
            invalid_count = invalid_mask.sum()
            print(f"⚠️ 检测到 {invalid_count} 个站点高度无效 (<-500)，已修正为 0。")
            df.loc[invalid_mask, 'Height'] = 0
            
        # 2. 准备特征矩阵
        # 【关键修改】加入 'Height'
        features = df[['Lon', 'Lat', 'Height', column]]
        
        # 标准化 (极其重要！海拔0-1000m，经度118-123，必须统一量纲)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 3. 构建空间约束图
        # 注意：约束图依然只基于【经纬度】构建。
        # 原因：我们希望只有"地理上相邻"的点才能合并。
        # 如果把高度也放进约束图，山顶和山脚可能就不算邻居了，导致无法形成连片区域。
        # 高度的作用是在"features_scaled"里，作为"相似度"的判断依据。
        print(f"🔗 正在构建空间约束图 (Neighbor k={N_NEIGHBORS})...")
        connectivity = kneighbors_graph(
            df[['Lon', 'Lat']], 
            n_neighbors=N_NEIGHBORS, 
            include_self=False
        )
        
        silhouette_list = []
        k_range_list = list(K_RANGE)
        
        # 4. 循环测试 K 值
        for k in k_range_list:
            print(f"  👉 K={k} ...")
            
            # A. 聚类
            model = AgglomerativeClustering(
                n_clusters=k, 
                connectivity=connectivity, 
                linkage='ward'
            )
            labels = model.fit_predict(features_scaled)
            
            # B. 评分
            try:
                score = silhouette_score(features_scaled, labels)
            except ValueError:
                score = -1.0
            silhouette_list.append(score)
            
            # C. 保存数据
            df_k = df.copy()
            df_k['Cluster'] = labels
            score_str = f"{score:.3f}"
            
            data_csv_path = task_output_dir / f"Clustered_Data_{name}_k{k}_Score{score_str}.csv"
            df_k.to_csv(data_csv_path, index=False, encoding='utf-8-sig') 
            
            # D. 绘制地图
            fig, ax = plt.subplots(figsize=(10, 9), subplot_kw={'projection': ccrs.PlateCarree()})
            
            # 标题增加 "Height" 提示
            ax.set_title(f"空间聚类(含海拔) (K={k}): {name} | Score: {score_str}", fontsize=16)
            
            ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8)
            ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
            try:
                city_shapes = list(shpreader.Reader(SHP_CITY_PATH).geometries())
                ax.add_geometries(city_shapes, ccrs.PlateCarree(), 
                                  edgecolor='gray', facecolor='none', 
                                  linewidth=0.5, linestyle='--')
            except Exception: pass

            ax.set_extent([118, 123, 27, 31.5], crs=ccrs.PlateCarree())
            
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False

            colors = matplotlib.colormaps['tab10']
            unique_labels = sorted(df_k['Cluster'].unique())
            
            for i, label in enumerate(unique_labels):
                cluster_data = df_k[df_k['Cluster'] == label]
                
                # 计算统计值：平均时长 & 平均海拔
                avg_hours = cluster_data[column].mean()
                avg_height = cluster_data['Height'].mean()
                
                ax.scatter(cluster_data['Lon'], cluster_data['Lat'], 
                           color=colors(i), 
                           label=f'区域{label}: {avg_hours:.0f}h | {avg_height:.0f}m', # 图例显示海拔
                           s=20, 
                           transform=ccrs.PlateCarree(),
                           alpha=0.8, 
                           edgecolors='none')

            ax.legend(title="聚类特征(时长|海拔)", loc='upper right', fontsize=9)
            
            map_png_path = task_output_dir / f"Clustered_Map_{name}_k{k}_Score{score_str}.png"
            fig.savefig(map_png_path, dpi=180, bbox_inches='tight')
            plt.close(fig)

        # 保存指标
        df_metrics = pd.DataFrame({
            'k': k_range_list,
            'Silhouette_Score': silhouette_list
        })
        metrics_csv_path = task_output_dir / f"K_Metrics_{name}.csv"
        df_metrics.to_csv(metrics_csv_path, index=False, encoding='utf-8-sig')

    except Exception as e:
        print(f"❌ [ERROR] 任务 {name} 出错: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*70}")
print("🎉 海拔优化版聚类完成！")