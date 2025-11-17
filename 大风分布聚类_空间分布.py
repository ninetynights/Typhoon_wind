"""
大风分布聚类_指定级别.py — 按风速等级的大风站点空间聚类与可视化工具

总体目的：
- 针对不同风速等级（如 ≥17.2 m/s 的"超阈值"或恰好在某等级范围内的"精确等级"），
  对全部台风期间受影响的观测站进行 K-Means 聚类分析，识别大风易发区域。
- 为每个等级生成聚类结果（CSV）、聚类地图（PNG）与轮廓系数评估，便于理解大风的空间分布特征。

主要功能：
- 从输入 CSV（AllTyphoons_Exceed.csv / AllTyphoons_Exact.csv）读取各站点的大风小时数与经纬；
- 对每个等级采用 K-Means进行聚类，并计算轮廓系数以评估聚类质量；
- 将聚类结果（站点编号、经纬、大风小时数、聚类标签）保存为 CSV；
- 在地图上用颜色区分各聚类，并叠加浙江市界 shapefile 作为参考边界；
- 输出每等级的聚类地图（PNG）与统计摘要（聚类个数、轮廓系数、各类站点计数）。

输入（脚本顶部配置）：
- CSV_EXCEED_PATH：所有台风叠加的"超阈值"大风小时数 CSV（期望列：STID、lon、lat、hours）；
- CSV_EXACT_PATH：所有台风叠加的"精确等级"大风小时数 CSV（同列结构）；
- SHP_CITY_PATH：浙江市界 shapefile 路径（用于地图背景参考）；
- OUTPUT_DIR：聚类结果输出根目录（脚本会自动为每个等级创建子目录）。

输出（保存至 OUTPUT_DIR 下的子目录）：
- 聚类结果 CSV：{等级名}_聚类结果.csv（包含站点编号、经纬、大风小时数、聚类标签）
- 聚类地图 PNG：{等级名}_聚类地图.png（彩色地图显示各聚类，圆大小与大风小时数对应）
- 聚类摘要打印：控制台输出每等级的聚类个数、轮廓系数、各类站点统计

"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from pathlib import Path
import matplotlib
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

# ======= 1. 配置 =======

# 使用您在 测试.py 中的字体设置
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 输入文件路径 (请按需修改)
CSV_EXCEED_PATH = "/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_大风分级统计/AllTyphoons_Exceed.csv"
CSV_EXACT_PATH  = "/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_大风分级统计/AllTyphoons_Exact.csv"
# 使用您在 测试.py 中的 SHP 路径
SHP_CITY_PATH   = r"/Users/momo/Desktop/业务相关/2025 影响台风大风/地形文件/shapefile/市界/浙江市界.shp"

# 【修改】这里是 *总的* 输出根目录
OUTPUT_DIR = Path("/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_大风分级统计/大风累积空间聚类结果") 

# 定义我们要分析的任务 (已扩展)
# 【修改】为每个任务增加了 "output_subdir" 键
ANALYSIS_TASKS = [
    # --- 8级 (17.2 m/s) ---
    {
        "file": CSV_EXCEED_PATH,
        "column": "TotalHours_gt_17.2", # 8级及以上
        "name": "8级及以上",
        "output_subdir": "超阈值 (Exceed)" # 指定输出子目录
    },
    {
        "file": CSV_EXACT_PATH,
        "column": "TotalHours_eq_17.2", # 正好是8级
        "name": "8级",
        "output_subdir": "指定级别 (Exact)" # 指定输出子目录
    },
    
    # --- 9级 (20.8 m/s) ---
    {
        "file": CSV_EXCEED_PATH,
        "column": "TotalHours_gt_20.8", # 9级及以上
        "name": "9级及以上",
        "output_subdir": "超阈值 (Exceed)"
    },
    {
        "file": CSV_EXACT_PATH,
        "column": "TotalHours_eq_20.8", # 正好是9级
        "name": "9级",
        "output_subdir": "指定级别 (Exact)"
    },

    # --- 10级 (24.5 m/s) ---
    {
        "file": CSV_EXCEED_PATH,
        "column": "TotalHours_gt_24.5", # 10级及以上
        "name": "10级及以上",
        "output_subdir": "超阈值 (Exceed)"
    },
    {
        "file": CSV_EXACT_PATH,
        "column": "TotalHours_eq_24.5", # 正好是10级
        "name": "10级",
        "output_subdir": "指定级别 (Exact)"
    },
    
    # --- 11级 (28.5 m/s) ---
    {
        "file": CSV_EXCEED_PATH,
        "column": "TotalHours_gt_28.5", # 11级及以上
        "name": "11级及以上",
        "output_subdir": "超阈值 (Exceed)"
    },
    {
        "file": CSV_EXACT_PATH,
        "column": "TotalHours_eq_28.5", # 正好是11级
        "name": "11级",
        "output_subdir": "指定级别 (Exact)"
    },
    
    # --- 12级 (32.7 m/s) ---
    {
        "file": CSV_EXCEED_PATH,
        "column": "TotalHours_gt_32.7", # 12级及以上
        "name": "12级及以上",
        "output_subdir": "超阈值 (Exceed)"
    },
    {
        "file": CSV_EXACT_PATH,
        "column": "TotalHours_eq_32.7", # 正好是12级
        "name": "12级",
        "output_subdir": "指定级别 (Exact)"
    }
]

# K-Means 测试的 K 值范围
K_RANGE = range(2, 7) # 测试 k=2, 3, 4, 5

# -----------------------------
# 确保 *总的* 输出目录存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"总输出目录: {OUTPUT_DIR.resolve()}")

# 存储生成的图像路径
generated_files = []

for task in ANALYSIS_TASKS:
    file_path = Path(task['file'])
    column = task['column']
    name = task['name']
    
    # --- 【新增】---
    # 根据配置定义并创建特定任务的输出子目录
    task_output_dir = OUTPUT_DIR / task['output_subdir']
    task_output_dir.mkdir(parents=True, exist_ok=True)
    # --- 【结束新增】---
    
    print(f"\n{'='*70}")
    print(f"--- 正在处理任务: {name} ---")
    print(f"--- 输出至: {task_output_dir.resolve()} ---") # 打印新的输出路径
    print(f"{'='*70}")
    
    try:
        # 1. 加载数据
        df = pd.read_csv(file_path)
        if df.empty:
            print(f"[WARN] 文件 {file_path} 为空，跳过任务。")
            continue
        print(f"加载文件: {file_path}")
        
        # 2. 准备特征
        features = df[['Lon', 'Lat', column]]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        print("特征已提取并标准化 (Lon, Lat, Hours)")
        
        # 3. 评估K值 (并为每个K保存结果)
        inertia_list = []
        silhouette_list = []
        k_range_list = list(K_RANGE)
        
        print(f"正在评估 K 值 {k_range_list} 并为每个K保存结果...")
        
        for k in k_range_list:
            print(f"\n--- 正在处理 K={k} ---")
            
            # A. 执行聚类
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(features_scaled)
            labels = kmeans.labels_
            inertia = kmeans.inertia_
            try:
                score = silhouette_score(features_scaled, labels)
            except ValueError:
                score = float('nan') 

            # B. 收集指标
            inertia_list.append(inertia)
            silhouette_list.append(score)
            
            # C. 保存聚类结果数据CSV
            df_k = df.copy()
            df_k['Cluster'] = labels 
            
            # 【修改】使用 task_output_dir
            data_csv_path = task_output_dir / f"Clustered_Data_{name}_k{k}.csv"
            df_k.to_csv(data_csv_path, index=False, encoding='utf-8-sig') 
            generated_files.append(str(data_csv_path))
            print(f"[OK] K={k} 聚类数据已保存: {data_csv_path.resolve()}")

            # D. 保存聚类结果地图
            print(f"正在为 K={k} 绘制地图...")
            fig, ax = plt.subplots(figsize=(10, 9), subplot_kw={'projection': ccrs.PlateCarree()})
            ax.set_title(f"K-Means 聚类 (K={k}): {name}", fontsize=16) 

            # 加载底图
            ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
            ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
            
            try:
                city_shapes = list(shpreader.Reader(SHP_CITY_PATH).geometries())
                ax.add_geometries(city_shapes, ccrs.PlateCarree(), edgecolor='gray', facecolor='none', linewidth=0.5, linestyle='--')
            except Exception:
                print(f"[WARN] 无法加载市界 Shapefile (路径: {SHP_CITY_PATH})，跳过绘制。")

            # 设置地图范围
            ax.set_extent([118, 123, 27, 31.5], crs=ccrs.PlateCarree())
            
            # 绘制网格
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, crs=ccrs.PlateCarree())
            gl.right_labels = False
            gl.top_labels = False

            # 绘制聚类散点
            colors = plt.cm.get_cmap('tab10', k) 
            unique_labels = sorted(df_k['Cluster'].unique())
            
            for i, label in enumerate(unique_labels):
                cluster_data = df_k[df_k['Cluster'] == label] 
                ax.scatter(cluster_data['Lon'], cluster_data['Lat'], 
                           color=colors(i), 
                           label=f'簇 {label} (N={len(cluster_data)})', 
                           s=15, 
                           transform=ccrs.PlateCarree(),
                           alpha=0.8)

            ax.legend(title="聚类结果", loc='upper right')
            
            # 【修改】使用 task_output_dir
            map_png_path = task_output_dir / f"Clustered_Map_{name}_k{k}.png" 
            fig.savefig(map_png_path, dpi=180, bbox_inches='tight')
            plt.close(fig) 
            generated_files.append(str(map_png_path))
            print(f"[OK] K={k} 聚类地图已保存: {map_png_path.resolve()}")
            
        # --- K循环结束 ---
            
        # 4. 【保留】保存K值评估指标到CSV (在K循环之后，仍在task循环内)
        print(f"\n--- 任务 {name} 的指标汇总 ---")
        df_metrics = pd.DataFrame({
            'k': k_range_list,
            'Inertia': inertia_list,
            'Silhouette_Score': silhouette_list
        })
        
        # 【修改】使用 task_output_dir
        metrics_csv_path = task_output_dir / f"K_Metrics_{name}.csv"
        df_metrics.to_csv(metrics_csv_path, index=False, encoding='utf-8-sig')
        generated_files.append(str(metrics_csv_path))
        print(f"[OK] 整体K值评估指标已保存: {metrics_csv_path.resolve()}")


    except FileNotFoundError:
        print(f"[ERROR] 文件未找到: {file_path}")
    except KeyError:
        print(f"[ERROR] 列 {column} 在文件 {file_path} 中未找到")
    except Exception as e:
        print(f"[ERROR] 处理任务 {name} 时发生严重错误: {e}")

print(f"\n{'='*70}")
print("--- 全部聚类任务完成 ---")
print("生成的文件 (已自动归类到子目录):")
# 打印子目录
subdirs = sorted(list(set([task['output_subdir'] for task in ANALYSIS_TASKS])))
for subdir in subdirs:
    print(f"  - 子目录: {OUTPUT_DIR / subdir}")
print("="*70)