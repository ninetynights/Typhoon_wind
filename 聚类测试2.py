import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d

# ======================== 配置路径 ========================
path_dir = "/Users/momo/Desktop/业务相关/2025 影响台风大风/热带气旋最佳路径数据集"
nc_typhoon_ids = [
    '2008', '1006', '1521', '2421', '2112', '1010', '1209', '1211', '1315', '1323', '1416', '1509',
    '1513', '1614', '1622', '1710', '1810', '1812', '1814', '1818', '1909', '1918', '2004', '2106',
    '2212', '2311', '2314', '2413', '2414'
]

# ======================== 定义函数 ========================
def parse_typhoon_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    paths = []
    i = 0
    while i < len(lines):
        if len(lines[i]) < 20:
            i += 1
            continue

        header = lines[i].strip()
        if len(header) < 20 or not header[:5].isdigit():
            i += 1
            continue

        year = int(header[0:4])
        cma_id = header[12:16].strip()
        n_points = int(header[16:20])

        i += 1
        points = []
        actual_points = min(n_points, len(lines) - i)
        if actual_points < n_points:
            print(f"⚠️ 文件 {file_path} 中台风 {cma_id} 的路径点仅有 {actual_points} 条，少于预期 {n_points}，部分读取。")

        for _ in range(actual_points):
            info = lines[i].strip()
            try:
                lat = int(info[14:18]) / 10.0
                lon = int(info[18:23]) / 10.0
                strength = int(info[23:25])
                points.append((lat, lon, strength))
            except:
                print(f"⚠️ 跳过格式异常的路径点：{info}")
            i += 1

        if cma_id in nc_typhoon_ids and len(points) >= 5:
            paths.append({
                "台风编号": cma_id,
                "年份": year,
                "路径点": points
            })

    return paths


def classify_path(row):
    start_lat, start_lon = row["起点"]
    end_lat, end_lon = row["终点"]
    delta_lat = end_lat - start_lat
    delta_lon = end_lon - start_lon

    angle = np.arctan2(delta_lat, delta_lon) * 180 / np.pi

    if angle > 30 and angle < 75:
        return "东北偏北型"
    elif angle >= 75:
        return "北上型"
    else:
        return "其他"


def interpolate_path(path, n=100):
    lats = [pt[0] for pt in path]
    lons = [pt[1] for pt in path]
    t = np.linspace(0, 1, len(path))
    ti = np.linspace(0, 1, n)
    lat_i = interp1d(t, lats, kind='linear')(ti)
    lon_i = interp1d(t, lons, kind='linear')(ti)
    return np.vstack([lat_i, lon_i]).T

# ======================== 主流程 ========================
all_paths = []
for year in range(2010, 2025):
    txt_file = os.path.join(path_dir, f"CH{year}BST.txt")
    ty_paths = parse_typhoon_file(txt_file)
    all_paths.extend(ty_paths)

# 构建 DataFrame
records = []
interpolated = []
for p in all_paths:
    latlon = [(lat, lon) for lat, lon, _ in p["路径点"]]
    records.append({
        "台风编号": p["台风编号"],
        "年份": p["年份"],
        "路径点数": len(latlon),
        "起点": latlon[0],
        "终点": latlon[-1]
    })
    interpolated.append(interpolate_path(latlon, n=50).flatten())

path_df = pd.DataFrame(records)
X = np.array(interpolated)

# 路径规则分型
path_df["路径分类"] = path_df.apply(classify_path, axis=1)

# 聚类
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
path_df["聚类类型"] = kmeans.labels_

# 保存结果
path_df.to_excel("浙江台风路径规则与聚类分型结果.xlsx", index=False)

# 可视化
plt.figure(figsize=(10, 8))
colors = ['red', 'green', 'blue', 'orange', 'purple']
for idx, row in path_df.iterrows():
    pts = all_paths[idx]["路径点"]
    lats = [pt[0] for pt in pts]
    lons = [pt[1] for pt in pts]
    cluster = row["聚类类型"]
    plt.plot(lons, lats, color=colors[cluster % len(colors)], alpha=0.6)
    plt.text(lons[0], lats[0], row["台风编号"], fontsize=8)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("浙江影响台风路径聚类分型图")
plt.grid(True)
plt.savefig("浙江路径聚类分型图.png", dpi=300)
plt.show()