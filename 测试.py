import os
import re
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
import matplotlib.font_manager as fm

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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

# 强度编号与等级映射（来自国家标准）
LEVEL_MAP = {
    "0": "Unknown",  # 弱于热带低压或未知
    "1": "TD",
    "2": "TS",
    "3": "STS",
    "4": "TY",
    "5": "STY",
    "6": "SuperTY",
    "9": "ET"  # 变性
}


def get_typhoon_ids_from_nc(nc_path):
    """从NC文件中读取所有影响浙江的台风编号"""
    nc = Dataset(nc_path)
    id_attr = nc.getncattr('id_to_index')
    id_map = {k.strip(): int(v.strip()) for k, v in [p.split(":") for p in id_attr.split(";") if ":" in p]}
    nc.close()
    print(id_attr)  # 打印ID映射属性
    return set(id_map.keys())


def read_selected_typhoon_tracks(folder_path, valid_ids):
    """读取所有txt文件中属于valid_ids的路径数据"""
    track_data = {}  # key: 中国台风编号，value: list of (time, lat, lon, level)

    for year in range(2010, 2025):
        filename = f"CH{year}BST.txt"
        filepath = os.path.join(folder_path, filename)
        if not os.path.exists(filepath):
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
                    parts = re.split(r'\s+', line)
                    if len(parts) >= 5:
                        current_id = parts[4]  # 中国台风编号 EEEE
                        current_path = []
                    else:
                        current_id = None
                        current_path = []
                elif current_id:
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

    return track_data


def plot_typhoon_tracks(track_data, specific_ids=None, output_path=None,
                        title="Typhoon Tracks Affecting Zhejiang (2010–2024)"):
    """
    绘制台风路径
    :param track_data: 所有台风路径数据
    :param specific_ids: 需要绘制的特定台风ID列表（例如['2421', '2425']），若为None则绘制所有
    :param output_path: 图片输出路径
    :param title: 图片标题
    """
    fig = plt.figure(figsize=(24, 20))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # 设置地图范围
    ax.set_extent([110, 135, 15, 40], crs=ccrs.PlateCarree())

    # 添加地理特征
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"))
    ax.add_feature(cfeature.BORDERS.with_scale("50m"))

    # 添加网格线
    ax.gridlines(draw_labels=True)

    # 确定要绘制的台风ID
    if specific_ids is None:
        ids_to_plot = list(track_data.keys())
    else:
        # 处理全角和半角逗号
        processed_ids = []
        for tid in specific_ids:
            # 分割可能包含全角逗号的字符串
            if '，' in tid:
                processed_ids.extend(tid.split('，'))
            elif ',' in tid:
                processed_ids.extend(tid.split(','))
            else:
                processed_ids.append(tid)

        # 去除空白字符
        ids_to_plot = [tid.strip() for tid in processed_ids if tid.strip()]

        # 检查有效性
        valid_ids = [tid for tid in ids_to_plot if tid in track_data]
        invalid_ids = set(ids_to_plot) - set(valid_ids)

        if invalid_ids:
            print(f"警告: 以下台风ID不在数据集中: {', '.join(invalid_ids)}")

        if valid_ids:
            ids_to_plot = valid_ids
            title = f"Typhoon Tracks ({', '.join(valid_ids)})"
        else:
            print("错误: 没有找到有效的台风ID!")
            plt.close(fig)
            return

    # 绘制路径
    for tid in ids_to_plot:
        path = track_data[tid]
        if len(path) < 2:
            continue

        lons = [p[2] for p in path]
        lats = [p[1] for p in path]
        levels = [p[3] for p in path]

        # 绘制路径线 - 使用原始风格的灰色细线
        ax.plot(lons, lats, color='gray', linewidth=1, alpha=0.6, zorder=1)

        # 绘制强度点 - 保持原始大小和风格
        for lon, lat, lvl in zip(lons, lats, levels):
            color = LEVEL_COLORS.get(lvl, "black")
            ax.scatter(lon, lat, color=color, s=10, transform=ccrs.PlateCarree(), zorder=2)

        # 标注台风编号（起点） - 保持原始字体大小
        ax.text(lons[0], lats[0], tid, fontsize=6, ha='right', va='bottom',
                transform=ccrs.PlateCarree(), zorder=3)

        # 如果是特定台风，用红色圆圈标记路径点
        if specific_ids:
            for lon, lat in zip(lons, lats):
                ax.scatter(lon, lat, color='red', s=30, edgecolor='black',
                           linewidth=0.5, alpha=0.7, transform=ccrs.PlateCarree(), zorder=4)

    # 创建图例 - 保持原始风格
    handles = []
    for lvl, color in LEVEL_COLORS.items():
        handles.append(plt.scatter([], [], color=color, s=30, label=lvl))

    # 图例位置在左下角
    ax.legend(handles=handles, title="Typhoon Level", loc="lower left", fontsize=10)

    # 添加标题
    plt.title(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"图片已保存至: {output_path}")
    else:
        plt.show()


def list_available_typhoons(track_data):
    """列出所有可用的台风ID"""
    print("\n可用的台风编号列表:")
    print("-" * 40)
    for i, tid in enumerate(sorted(track_data.keys())):
        print(f"{tid}", end="\t")
        if (i + 1) % 8 == 0:
            print()  # 每8个换行
    print("\n" + "-" * 40)


if __name__ == "__main__":
    folder = "/Users/momo/Desktop/业务相关/2025 影响台风大风/热带气旋最佳路径数据集"
    nc_file = "/Users/momo/Desktop/业务相关/2025 影响台风大风/All_Typhoons_ExMaxWind.nc"

    # 获取所有影响浙江的台风编号
    print("正在从NC文件中读取台风编号...")
    valid_typhoon_ids = get_typhoon_ids_from_nc(nc_file)

    # 读取所有台风路径数据
    print("正在读取台风路径数据...")
    track_data = read_selected_typhoon_tracks(folder, valid_typhoon_ids)
    print(f"成功加载 {len(track_data)} 个台风的路径数据")

    # 列出可用台风
    list_available_typhoons(track_data)

    # 用户交互
    while True:
        user_input = input(
            "\n请输入要绘制的台风编号（多个编号用逗号分隔，输入 'all' 绘制所有，输入 'exit' 退出）: ").strip()

        if user_input.lower() == 'exit':
            print("程序已退出")
            break

        if user_input.lower() == 'all':
            specific_ids = None
            title = "Typhoon Tracks Affecting Zhejiang (2010–2024)"
        elif user_input:
            specific_ids = [user_input]
            title = ""  # 标题将在绘图函数中生成
        else:
            print("输入无效，请重新输入")
            continue

        # 绘制选定台风
        plot_typhoon_tracks(
            track_data,
            specific_ids=specific_ids,
            title=title
        )

        continue_input = input("是否继续绘制其他台风？(y/n): ").strip().lower()
        if continue_input != 'y':
            print("程序已退出")
            break