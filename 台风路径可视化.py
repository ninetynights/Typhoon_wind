"""
台风路径可视化.py — 浙江省受影响台风路径绘图工具

功能概述：
- 从最佳路径文本（CHYYYYBST.txt 系列）和 NetCDF 元数据中筛选出影响浙江的台风，
  并在地图上绘制这些台风的路径与每个时刻的位置点（按强度等级着色）。
- 支持 2010–2024 年的数据批量读取并一次性绘制或保存图片。
- 标题会显示台风总数;图例会显示以该等级为最强等级的台风个数。

主要输入：
- folder（脚本中变量）：包含 CHYYYYBST.txt 格式的最佳路径文件目录（按年分文件）。
- nc_file（脚本中变量）：包含属性 id_to_index 的 NetCDF 文件，用于获取“影响浙江”的台风编号集合。

主要输出：
- 屏幕显示或保存的地图图像（如果传入 output_path 则保存为 PNG；否则弹窗显示）。
- 控制台打印已加载的台风编号与处理进度信息。

"""

import os
import re
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from collections import defaultdict
from netCDF4 import Dataset

plt.rcParams['font.sans-serif'] = ['Heiti TC']

# 台风等级颜色映射
LEVEL_COLORS = {
    "TD": "#A0A0A0",
    "TS": "#00CED1",
    "STS": "#1E90FF",
    "TY": "#FF8C00",
    "STY": "#FF4500",
    "SuperTY": "#8B0000",
    "ET": "green",  # Extra-tropical
    "Unknown": "black",
}

# 强度编号与等级映射（来自国家标准）
LEVEL_MAP = {
    "0": "Unknown",     # 弱于热带低压或未知
    "1": "TD",
    "2": "TS",
    "3": "STS",
    "4": "TY",
    "5": "STY",
    "6": "SuperTY",
    "9": "ET"  # 变性
}

# [新增] 台风等级的强度排序（用于确定最大强度）
# 数字越大，强度越强
STRENGTH_RANKING = {
    "SuperTY": 6,
    "STY": 5,
    "TY": 4,
    "STS": 3,
    "TS": 2,
    "TD": 1,
    "ET": 0,
    "Unknown": -1
}


def get_typhoon_ids_from_nc(nc_path):
    """从NC文件中读取所有影响浙江的台风编号"""
    nc = Dataset(nc_path)
    id_attr = nc.getncattr('id_to_index')
    id_map = {k.strip(): int(v.strip()) for k, v in [p.split(":") for p in id_attr.split(";") if ":" in p]}
    nc.close()
    print(f"从 {nc_path} 加载了 {len(id_map)} 个台风ID。") # 优化打印信息
    # print(id_attr) # 原始打印信息太长，可以选择注释掉
    return set(id_map.keys())


def read_selected_typhoon_tracks(folder_path, valid_ids):
    """读取所有txt文件中属于valid_ids的路径数据"""
    track_data = {}  # key: 中国台风编号，value: list of (time, lat, lon, level)

    print(f"开始从 {folder_path} 目录读取数据，筛选 {len(valid_ids)} 个有效ID...")
    
    for year in range(2010, 2025):
        filename = f"CH{year}BST.txt"
        filepath = os.path.join(folder_path, filename)
        if not os.path.exists(filepath):
            # print(f"警告: 未找到文件 {filepath}")
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            current_id = None
            current_path = []

            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("66666"):
                    if current_id and current_id in valid_ids and current_path:
                        track_data[current_id] = current_path
                        # print(f"    -> 成功加载台风 {current_id} ({year})")
                    
                    parts = re.split(r'\s+', line)
                    if len(parts) >= 5:
                        current_id = parts[4]  # 中国台风编号 EEEE
                        current_path = []
                    else:
                        current_id = None
                        current_path = []
                elif current_id and current_id in valid_ids: # 优化：如果ID无效，则不处理后续行
                    parts = re.split(r'\s+', line)
                    if len(parts) >= 6:
                        try:
                            time_str = parts[0]
                            lat = float(parts[2]) / 10.0
                            lon = float(parts[3]) / 10.0
                            level_code = parts[1]  # I 位，强度标记
                            level = LEVEL_MAP.get(level_code, "Unknown")

                            current_path.append((time_str, lat, lon, level))
                        except:
                            continue

            if current_id and current_id in valid_ids and current_path:
                track_data[current_id] = current_path
                # print(f"    -> 成功加载台风 {current_id} ({year}) (文件尾)")

    print(f"数据读取完成。共加载 {len(track_data)} 个台风的路径。")
    return track_data


def plot_all_tracks(track_data, output_path=None, title="Typhoon Tracks"):
    """
    [修改] 
    1. 接受完整的标题 (已包含总数)。
    2. 计算每个等级的最大强度台风数并显示在图例中。
    """
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent([110, 135, 15, 40], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"))
    ax.add_feature(cfeature.BORDERS.with_scale("50m"))
    ax.gridlines(draw_labels=True, linestyle='--', alpha=0.7)

    # --- 添加省界 ---
    # 定义省界特征，'admin_1_states_provinces' 代表省/州级行政区划
    # scale='10m' 提供最高分辨率，'50m' 或 '110m' 是较低分辨率
    province_boundaries = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces',
        scale='10m', # 可以选择 '10m', '50m', '110m'
        facecolor='none', # 不填充省份内部
        edgecolor='grey', # 省界颜色
        linestyle='--', # 虚线
        linewidth=0.8 # 线宽
    )
    ax.add_feature(province_boundaries, zorder=3)

    # --- 计算最大强度统计 ---
    level_max_counts = defaultdict(int)
    for tid, path in track_data.items():
        if not path:
            continue
        
        max_strength_rank = -2 # 低于 "Unknown"
        max_level_name = "Unknown"
        
        for (time_str, lat, lon, level) in path:
            current_rank = STRENGTH_RANKING.get(level, -1)
            if current_rank > max_strength_rank:
                max_strength_rank = current_rank
                max_level_name = level
        
        level_max_counts[max_level_name] += 1

    # --- 绘制路径和散点 (不变) ---
    for tid, path in track_data.items():
        if len(path) < 2:
            continue

        lons = [p[2] for p in path]
        lats = [p[1] for p in path]
        levels = [p[3] for p in path]

        # 绘制灰色路径底线
        ax.plot(lons, lats, color='gray', linewidth=1, alpha=0.6, zorder=1)

        # 绘制彩色强度点
        for lon, lat, lvl in zip(lons, lats, levels):
            color = LEVEL_COLORS.get(lvl, "black")
            ax.scatter(lon, lat, color=color, s=10, transform=ccrs.PlateCarree(), zorder=2)

        '''
        # 在起点标注ID (优化：避免重叠)
        if lons:
             ax.text(lons[0], lats[0], tid, fontsize=6, ha='right', transform=ccrs.PlateCarree(),
                     bbox=dict(facecolor='white', alpha=0.3, pad=0.1, edgecolor='none'))
     '''
    # --- [修改] 创建图例 ---
    legend_handles = []
    # 按照 LEVEL_COLORS 中定义的顺序创建图例
    for lvl, color in LEVEL_COLORS.items():
        count = level_max_counts.get(lvl, 0)
        label_text = f"{lvl} ({count}个)"
        # 使用 ax.plot 创建一个不可见的点用于图例，s=10 匹配散点大小
        handle = ax.scatter([], [], color=color, s=30, label=label_text)
        legend_handles.append(handle)

    ax.legend(handles=legend_handles, title="强度", loc="lower left")
    # --- [修改] 结束 ---

    ax.set_title(title, fontsize=16) # [修改] 使用传入的完整标题
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"图像已保存至: {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    folder = "/Users/momo/Desktop/业务相关/2025 影响台风大风/热带气旋最佳路径数据集"
    
    # [修改点 1] 更改NC文件路径
    nc_file = "/Users/momo/Desktop/业务相关/2025 影响台风大风/数据/Refined_Combine_Stations_ExMaxWind_Fixed.nc"
    
    # 加载数据
    valid_typhoon_ids = get_typhoon_ids_from_nc(nc_file)
    track_data = read_selected_typhoon_tracks(folder, valid_typhoon_ids)
    
    # [修改点 2] 准备包含总数的标题
    total_count = len(track_data)
    plot_title = f"2010–2024年浙江省大风影响台风 (总数: {total_count}个)"

    # [修改点 3] (在 plot_all_tracks 函数内部实现)
    plot_all_tracks(track_data, title=plot_title)