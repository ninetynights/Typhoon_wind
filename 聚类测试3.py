import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# ===== 用户参数设置 =====
txt_folder = "/Users/momo/Desktop/业务相关/2025 影响台风大风/热带气旋最佳路径数据集"
nc_typhoon_ids = [
    '1006', '1007', '1010', '1105', '1109', '1111', '1209', '1211', '1215',
    '1307', '1312', '1315', '1323', '1410', '1416', '1509', '1513', '1521',
    '1601', '1614', '1616', '1617', '1618', '1622', '1709', '1710', '1718',
    '1808', '1810', '1812', '1814', '1818', '1825', '1905', '1909', '1911',
    '1913', '1917', '1918', '2004', '2008', '2009', '2106', '2109', '2112',
    '2114', '2204', '2211', '2212', '2305', '2306', '2311', '2314', '2403',
    '2413', '2414', '2418', '2421', '2425']


# ===== 辅助函数：解析路径数据 =====
def parse_typhoon_paths(txt_folder, nc_typhoon_ids):
    all_paths = []
    used_ids = set()

    for year in range(2010, 2024 + 1):
        txt_file = os.path.join(txt_folder, f"CH{year}BST.txt")
        if not os.path.exists(txt_file):
            print(f"⚠️ 文件不存在：{txt_file}")
            continue

        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if len(line) >= 30 and line[:5].isdigit():
                ch_id = line[21:24]  # 四位中国编号
                path_count = int(line[11:15])

                if ch_id in nc_typhoon_ids:
                    used_ids.add(ch_id)
                    lats, lons = [], []
                    for j in range(path_count):
                        try:
                            info = lines[i + 1 + j].strip()
                            lat = int(info[14:18]) / 10.0
                            lon = int(info[18:23]) / 10.0
                            lats.append(lat)
                            lons.append(lon)
                        except:
                            print(f"⚠️ 台风 {ch_id} 第 {j} 个路径点读取失败，跳过")
                            continue

                    if len(lats) >= 2:
                        all_paths.append({
                            "台风编号": ch_id,
                            "年份": year,
                            "起点纬度": lats[0],
                            "起点经度": lons[0],
                            "终点纬度": lats[-1],
                            "终点经度": lons[-1],
                            "路径点数": len(lats),
                            "路径": list(zip(lons, lats))  # for clustering
                        })
                    else:
                        print(f"⚠️ 台风 {ch_id} 路径点数过少，跳过")
                i += path_count + 1
            else:
                i += 1

    print(f"✅ 共有 {len(all_paths)} 个路径成功读取，原始编号应为 {len(nc_typhoon_ids)} 个")
    missed_ids = set(nc_typhoon_ids) - used_ids
    if missed_ids:
        print(f"❌ 以下编号未匹配成功：{sorted(missed_ids)}")

    return pd.DataFrame(all_paths)


# ===== 聚类分析 =====
def do_kmeans_clustering(path_df, n_clusters=3):
    # 仅取起点和终点经纬度作为特征
    X = path_df[["起点经度", "起点纬度", "终点经度", "终点纬度"]].values
    model = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    path_df['聚类类别'] = model.fit_predict(X)
    return path_df, model


# ===== 主程序入口 =====
if __name__ == "__main__":
    df = parse_typhoon_paths(txt_folder, nc_typhoon_ids)
    if df.empty:
        print("❗未成功读取任何台风路径")
    else:
        df, model = do_kmeans_clustering(df, n_clusters=3)
        df.to_excel("路径分类_聚类结果.xlsx", index=False)
        print("✅ 聚类结果已保存为 '路径分类_聚类结果.xlsx'")