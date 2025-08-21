import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d

# 设置路径
base_dir = "/Users/momo/Desktop/业务相关/2025 影响台风大风/热带气旋最佳路径数据集"
years = list(range(2010, 2025))

# 读取浙江省有大风影响的台风编号（来自NC文件提取的）
zj_typhoon_ids = [
    '2008', '1006', '1521', '2421', '2112', '1010', '1209', '1211', '1315', '1323', '1416', '1509',
    '1513', '1614', '1622', '1710', '1810', '1812', '1814', '1818', '1909', '1918', '2004', '2106',
    '2212', '2311', '2314', '2413', '2414'
]

# 存储路径信息
all_paths = []
path_vectors = []
vec_typhoon_ids = []

for year in years:
    file_path = os.path.join(base_dir, f"CH{year}BST.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if len(line) >= 14 and line[4:8].isdigit():
            ty_id = line[4:8]  # 中国编号（4位）
            n_points = int(line[12:16].strip())
            i += 1
            path_points = []
            for _ in range(n_points):
                info = lines[i].strip()
                lat = int(info[14:18]) / 10.0
                lon = int(info[18:22]) / 10.0
                path_points.append((lat, lon))
                i += 1

            if ty_id in zj_typhoon_ids:
                lat1, lon1 = path_points[0]
                lat2, lon2 = path_points[-1]
                all_paths.append({
                    "台风编号": ty_id,
                    "起点纬度": lat1,
                    "起点经度": lon1,
                    "终点纬度": lat2,
                    "终点经度": lon2,
                    "路径点数": len(path_points)
                })

                # 插值成固定长度路径向量
                lats, lons = zip(*path_points)
                f_lat = interp1d(np.linspace(0, 1, len(lats)), lats, kind='linear')
                f_lon = interp1d(np.linspace(0, 1, len(lons)), lons, kind='linear')
                interp_lats = f_lat(np.linspace(0, 1, 20))
                interp_lons = f_lon(np.linspace(0, 1, 20))
                feature = np.concatenate([interp_lats, interp_lons])
                path_vectors.append(feature)
                vec_typhoon_ids.append(ty_id)
        else:
            i += 1

# 转换为 DataFrame
path_df = pd.DataFrame(all_paths)


# 添加规则路径分类函数
def classify_path(row):
    lat1, lon1 = row['起点纬度'], row['起点经度']
    lat2, lon2 = row['终点纬度'], row['终点经度']
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    if delta_lat > 5 and abs(delta_lon) < 3:
        return "北上型"
    elif delta_lat > 5 and delta_lon > 3:
        return "东北偏北型"
    elif delta_lon < -5:
        return "偏西型"
    elif np.hypot(delta_lat, delta_lon) < 3:
        return "徘徊型"
    else:
        return "其他"


path_df["路径分类"] = path_df.apply(classify_path, axis=1)

# 几何聚类分类
feature_array = np.array(path_vectors)
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(feature_array)
path_df['聚类分型'] = ""
for tid, label in zip(vec_typhoon_ids, labels):
    path_df.loc[path_df['台风编号'] == tid, '聚类分型'] = f"类型{label + 1}"

# 保存结果
output_file = "浙江台风路径规则与聚类分型结果.xlsx"
path_df.to_excel(output_file, index=False)
print(f"规则+聚类路径分型完成，已保存：{output_file}")
