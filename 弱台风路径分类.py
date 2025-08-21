import os
import re
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from datetime import datetime, timedelta  # 添加timedelta用于时间计算
import matplotlib.cm as cm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 台风等级颜色映射
LEVEL_COLORS = {
    "TD": "#A0A0A0",
    "TS": "#00CED1",
    "STS": "#1E90FF",
    "TY": "#FF8C00",
    "STY": "#FF4500",
    "SuperTY": "#8B0000",
    "Unknown": "black",
    "ET": "green"  # Extra-tropical
}


# 强度编号与等级映射
LEVEL_MAP = {
    "0": "Unknown",
    "1": "TD",
    "2": "TS",
    "3": "STS",
    "4": "TY",
    "5": "STY",
    "6": "SuperTY",
    "9": "ET"
}


def read_weak_typhoon_list(excel_path):
    """读取浙江省弱台风列表"""
    df = pd.read_excel(excel_path)
    print("Excel文件列名:", df.columns.tolist())

    # 转换中央台编号为4位字符串
    df['中央台编号'] = df['中央台编号'].astype(str).str.zfill(4)
    print("处理后的编号样例:", df['中央台编号'].head().tolist())

    # 确保时间列是datetime类型
    df['开始时间'] = pd.to_datetime(df['开始时间'])
    df['结束时间'] = pd.to_datetime(df['结束时间'])

    return df


def parse_bst_time(time_str):
    """解析BST文件中的时间格式(YYYYMMDDHH)"""
    try:
        return datetime.strptime(time_str, "%Y%m%d%H")
    except ValueError:
        # 尝试处理可能的2位数年份
        if len(time_str) == 10:
            return datetime.strptime(time_str, "%y%m%d%H")
        raise


def interpolate_track_hourly(track):
    """将台风路径插值到逐小时"""
    if not track:
        return []

    # 转成 DataFrame
    df = pd.DataFrame(track)
    df = df.sort_values("time")  # 确保按时间顺序

    # 构造每小时时间序列
    time_index = pd.date_range(start=df['time'].iloc[0],
                               end=df['time'].iloc[-1],
                               freq='1H')

    # 设置 time 为索引，方便插值
    df = df.set_index('time')

    # 经纬度线性插值
    df_interp = df[['lat', 'lon']].reindex(time_index).interpolate()

    # 台风等级（level）可选：直接前向填充
    if 'level' in df:
        df_interp['level'] = df['level'].reindex(time_index).ffill()

    # 重置索引
    df_interp = df_interp.reset_index().rename(columns={'index': 'time'})

    # 转回 list[dict]
    return df_interp.to_dict(orient='records')

def read_typhoon_tracks_with_time_filter(folder_path, typhoon_list):
    track_data = {}
    typhoon_info = {}
    for _, row in typhoon_list.iterrows():
        tid = row['中央台编号']
        typhoon_info[tid] = {
            'start': row['开始时间'],
            'end': row['结束时间'],
            'name': row['中文名称']
        }

    for year in range(2000, 2025):
        filename = f"CH{year}BST.txt"
        filepath = os.path.join(folder_path, filename)
        if not os.path.exists(filepath):
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            current_id = None
            current_path = []
            current_info = None

            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("66666"):
                    # 保存上一个台风
                    if current_id and current_info and current_path:
                        # 插值成逐小时
                        hourly_path = interpolate_track_hourly(current_path)
                        # 再按 start ~ end 筛选
                        filtered_path = [p for p in hourly_path
                                         if current_info['start'] <= p['time'] <= current_info['end']]
                        if filtered_path:
                            track_data[current_id] = {
                                'path': filtered_path,
                                'name': current_info['name']
                            }

                    # 新台风头信息
                    parts = re.split(r'\s+', line)
                    if len(parts) >= 5:
                        current_id = parts[4]
                        current_info = typhoon_info.get(current_id)
                        current_path = []
                    else:
                        current_id = None
                        current_info = None
                        current_path = []

                elif current_id and current_info:
                    parts = re.split(r'\s+', line)
                    if len(parts) >= 6:
                        try:
                            dt = parse_bst_time(parts[0])
                            lat = float(parts[2]) / 10.0
                            lon = float(parts[3]) / 10.0
                            level_code = parts[1]
                            level = LEVEL_MAP.get(level_code, "Unknown")

                            # 不筛选，先全存
                            current_path.append({
                                'time': dt,
                                'lat': lat,
                                'lon': lon,
                                'level': level
                            })
                        except Exception as e:
                            print(f"解析错误: {line} | {str(e)}")
                            continue

            # 文件结束后处理最后一个台风
            if current_id and current_info and current_path:
                hourly_path = interpolate_track_hourly(current_path)
                filtered_path = [p for p in hourly_path
                                 if current_info['start'] <= p['time'] <= current_info['end']]
                if filtered_path:
                    track_data[current_id] = {
                        'path': filtered_path,
                        'name': current_info['name']
                    }

    return track_data


def cluster_tracks(track_data, n_clusters=3):
    """将轨迹转成向量并进行聚类"""
    X = []
    ids = []
    max_len = max(len(data['path']) for data in track_data.values())
    for tid, data in track_data.items():
        path = data['path']
        coords = []
        for p in path:
            coords.extend([p['lat'], p['lon']])
        # 补齐到统一长度
        while len(coords) < max_len * 2:
            coords.append(0)
        X.append(coords)
        ids.append(tid)
    X = np.array(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    return dict(zip(ids, labels))

def plot_filtered_tracks(track_data, n_clusters=4, output_path=None):
    # 先做聚类
    cluster_labels = cluster_tracks(track_data, n_clusters=n_clusters)
    # 准备颜色映射    
    color_map_manual = {
    0: 'red',
    1: 'orange',
    2: 'blue',
    3: 'purple'}
    cluster_colors = {label: color_map_manual[label] for label in set(cluster_labels.values())}


    """绘制过滤后的台风路径"""
    print("\n=== 绘图前的路径数据 ===")
    for tid, data in track_data.items():
        print(f"台风 {tid} ({data['name']}): {len(data['path'])}个点, 类别 {cluster_labels[tid]}")
        if len(data['path']) > 0:
            print(f"  首点: {data['path'][0]['lon']}, {data['path'][0]['lat']}")
            print(f"  末点: {data['path'][-1]['lon']}, {data['path'][-1]['lat']}")

    try:
        fig = plt.figure(figsize=(8, 7))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # 设置地图范围
        ax.set_extent([110, 130, 20, 40], crs=ccrs.PlateCarree())

        # 添加地图元素
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="lightgray")
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"))
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linestyle=':')
        ax.add_feature(cfeature.OCEAN.with_scale("50m"))

        # 尝试添加省界
        try:
            provinces = cfeature.NaturalEarthFeature(
                category='cultural',
                name='admin_1_states_provinces_lines',
                scale='50m',
                facecolor='none')
            ax.add_feature(provinces, edgecolor='gray', linestyle=':')
        except Exception as e:
            print(f"无法添加省界: {str(e)}")

        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False   # 隐藏顶部纬度
        gl.right_labels = False # 隐藏右侧经度

        # 绘制每个台风的路径
        for tid, data in track_data.items():
            path = data['path']
            if len(path) < 2:
                continue

            lons = [p['lon'] for p in path]
            lats = [p['lat'] for p in path]

            cluster_id = cluster_labels[tid]
            color = cluster_colors[cluster_id]

            ax.plot(lons, lats, color=color, linewidth=2, zorder=2)

            # 绘制强度点
            for point in path:
                ax.scatter(
                    point['lon'], point['lat'],
                    color='lightgray', s=5,
                    transform=ccrs.PlateCarree(),
                    zorder=1
                )

            # 添加台风名称标签
            if len(path) > 0:
                ax.text(
                    path[-1]['lon'], path[-1]['lat'],
                    f"{data['name']}({tid})",
                    fontsize=10,
                    ha='left',
                    va='top',
                    color=color,
                    transform=ccrs.PlateCarree()
                )

        ax.set_title("影响浙江省的弱台风个例（聚类结果）", fontsize=20)

        if output_path:
            plt.savefig(output_path, dpi=300)
            print(f"图表已保存到: {output_path}")
        else:
            plt.show()

    except Exception as e:
        print(f"绘图出错: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # 配置路径
        folder = "./热带气旋最佳路径数据集"
        excel_path = "./浙江省弱台风.xlsx"

        # 1. 读取弱台风列表
        print("正在读取Excel文件...")
        weak_typhoon_list = read_weak_typhoon_list(excel_path)
        print(f"找到{len(weak_typhoon_list)}个弱台风记录")

        # 2. 读取并过滤台风路径
        print("正在读取台风路径数据...")
        track_data = read_typhoon_tracks_with_time_filter(folder, weak_typhoon_list)

        # 3. 绘制图表
        print("正在绘制图表...")
        plot_filtered_tracks(track_data, output_path='./track.png')

    except Exception as e:
        print(f"程序运行出错: {str(e)}")