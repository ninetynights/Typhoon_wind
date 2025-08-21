import os
import re
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from collections import defaultdict
from netCDF4 import Dataset

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
    "0": "Unknown",     # 弱于热带低压或未知
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
    print(id_attr)
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


def plot_all_tracks(track_data, output_path=None, title="Typhoon Tracks Affecting Zhejiang (2010–2024)"):
    fig = plt.figure(figsize=(24, 20))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent([110, 135, 15, 40], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"))
    ax.add_feature(cfeature.BORDERS.with_scale("50m"))
    ax.gridlines(draw_labels=True)

    for tid, path in track_data.items():
        if len(path) < 2:
            continue

        lons = [p[2] for p in path]
        lats = [p[1] for p in path]
        levels = [p[3] for p in path]

        ax.plot(lons, lats, color='gray', linewidth=1, alpha=0.6, zorder=1)

        for lon, lat, lvl in zip(lons, lats, levels):
            color = LEVEL_COLORS.get(lvl, "black")
            ax.scatter(lon, lat, color=color, s=10, transform=ccrs.PlateCarree(), zorder=2)

        ax.text(lons[0], lats[0], tid, fontsize=6, ha='right', transform=ccrs.PlateCarree())

    for lvl, color in LEVEL_COLORS.items():
        ax.scatter([], [], color=color, label=lvl, s=30)
    ax.legend(title="Typhoon Level", loc="lower left")
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    folder = "/Users/momo/Desktop/业务相关/2025 影响台风大风/热带气旋最佳路径数据集"
    nc_file = "/Users/momo/Desktop/业务相关/2025 影响台风大风/All_Typhoons_ExMaxWind.nc"
    valid_typhoon_ids = get_typhoon_ids_from_nc(nc_file)
    track_data = read_selected_typhoon_tracks(folder, valid_typhoon_ids)
    plot_all_tracks(track_data, title="Typhoon Tracks Affecting Zhejiang (2010–2024)")
