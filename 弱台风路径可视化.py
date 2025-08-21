import os
import re
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from datetime import datetime, timedelta  # 添加timedelta用于时间计算

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


def read_typhoon_tracks_with_time_filter(folder_path, typhoon_list):
    """读取并过滤台风路径数据"""
    track_data = {}

    # 准备台风信息（精确匹配4位编号）
    typhoon_info = {}
    for _, row in typhoon_list.iterrows():
        tid = row['中央台编号']  # 4位格式 (0407)
        typhoon_info[tid] = {
            'start': row['开始时间'],
            'end': row['结束时间'],
            'name': row['中文名称']
        }

    print("\n=== 查找以下台风编号 ===")
    print(typhoon_info.keys())

    # 正式读取路径数据
    for year in range(2000, 2025):
        filename = f"CH{year}BST.txt"
        filepath = os.path.join(folder_path, filename)
        if not os.path.exists(filepath):
            continue

        print(f"\n处理文件: {filename}")

        with open(filepath, 'r', encoding='utf-8') as f:
            current_id = None
            current_path = []
            current_info = None

            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("66666"):
                    # 保存上一个台风的数据
                    if current_id and current_info and current_path:
                        if current_id in track_data:
                            track_data[current_id]['path'].extend(current_path)
                        else:
                            track_data[current_id] = {
                                'path': current_path,
                                'name': current_info['name']
                            }

                    # 解析新台风头信息
                    parts = re.split(r'\s+', line)
                    if len(parts) >= 5:
                        current_id = parts[4]  # 获取台风编号
                        current_info = typhoon_info.get(current_id)
                        current_path = []

                        if current_info:
                            print(f"找到匹配台风: {current_id} ({current_info['name']})")
                    else:
                        current_id = None
                        current_info = None
                        current_path = []

                elif current_id and current_info:
                    parts = re.split(r'\s+', line)
                    if len(parts) >= 6:
                        try:
                            time_str = parts[0]
                            # 解析时间 (格式: YYYYMMDDHH 或 YYMMDDHH)
                            dt = parse_bst_time(time_str)

                            # 检查时间范围
                            if current_info['start'] <= dt <= current_info['end']:
                                lat = float(parts[2]) / 10.0
                                lon = float(parts[3]) / 10.0
                                level_code = parts[1]
                                level = LEVEL_MAP.get(level_code, "Unknown")

                                current_path.append({
                                    'time': dt,
                                    'lat': lat,
                                    'lon': lon,
                                    'level': level
                                })
                        except Exception as e:
                            print(f"解析错误: {line} | {str(e)}")
                            continue

            # 处理最后一个台风
            if current_id and current_info and current_path:
                if current_id in track_data:
                    track_data[current_id]['path'].extend(current_path)
                else:
                    track_data[current_id] = {
                        'path': current_path,
                        'name': current_info['name']
                    }

    print("\n=== 最终匹配结果 ===")
    for tid, data in track_data.items():
        print(f"{tid} ({data['name']}): {len(data['path'])}个点")
        if len(data['path']) > 0:
            print(f"  时间范围: {data['path'][0]['time']} 到 {data['path'][-1]['time']}")

    return track_data


def plot_filtered_tracks(track_data, output_path=None):
    """绘制过滤后的台风路径"""
    print("\n=== 绘图前的路径数据 ===")
    for tid, data in track_data.items():
        print(f"台风 {tid} ({data['name']}): {len(data['path'])}个点")
        if len(data['path']) > 0:
            print(f"  首点: {data['path'][0]['lon']}, {data['path'][0]['lat']}")
            print(f"  末点: {data['path'][-1]['lon']}, {data['path'][-1]['lat']}")

    try:
        fig = plt.figure(figsize=(16, 14))
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

        ax.gridlines(draw_labels=True)

        # 绘制每个台风的路径
        for tid, data in track_data.items():
            path = data['path']
            if len(path) < 2:
                continue

            lons = [p['lon'] for p in path]
            lats = [p['lat'] for p in path]

            # 绘制路径线
            ax.plot(lons, lats, color='gray', linewidth=1, alpha=0.6, zorder=1)

            # 绘制强度点
            for point in path:
                color = LEVEL_COLORS.get(point['level'], "black")
                ax.scatter(
                    point['lon'], point['lat'],
                    color=color, s=15,
                    transform=ccrs.PlateCarree(),
                    zorder=2
                )

            # 添加台风名称标签
            if len(path) > 0:
                ax.text(
                    path[0]['lon'], path[0]['lat'],
                    f"{data['name']}({tid})",
                    fontsize=11,
                    ha='left',
                    va='top',
                    transform=ccrs.PlateCarree()
                )

        # 添加图例
        for lvl, color in LEVEL_COLORS.items():
            ax.scatter([], [], color=color, label=lvl, s=30)

        ax.legend(title="Typhoon Level", loc="lower left")
        ax.set_title("影响浙江省的弱台风个例", fontsize=14)
        plt.tight_layout()

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
        folder = "/Users/momo/Desktop/业务相关/2025 影响台风大风/热带气旋最佳路径数据集"
        excel_path = "/Users/momo/Desktop/业务相关/2025 影响台风大风/浙江省弱台风.xlsx"

        # 1. 读取弱台风列表
        print("正在读取Excel文件...")
        weak_typhoon_list = read_weak_typhoon_list(excel_path)
        print(f"找到{len(weak_typhoon_list)}个弱台风记录")

        # 2. 读取并过滤台风路径
        print("正在读取台风路径数据...")
        track_data = read_typhoon_tracks_with_time_filter(folder, weak_typhoon_list)

        # 3. 绘制图表
        print("正在绘制图表...")
        plot_filtered_tracks(track_data)

    except Exception as e:
        print(f"程序运行出错: {str(e)}")