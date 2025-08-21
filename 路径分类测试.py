import numpy as np
import pandas as pd
import re
from collections import defaultdict
from netCDF4 import Dataset
import os

# 读取 NetCDF 中的浙江影响台风编号
def get_impact_typhoons(nc_path):
    nc = Dataset(nc_path)
    attr = nc.getncattr('id_to_index')
    id_to_index = {k.strip(): int(v) for k, v in (pair.split(':') for pair in attr.split(';') if ':' in pair)}
    nc.close()
    return set(id_to_index.keys())

# 读取 CMA 路径数据
def read_typhoon_track_file(filepath):
    from collections import defaultdict
    import re

    data = {}  # 使用中国编号（EEEE）作为键
    current_id = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("66666"):
                parts = re.split(r'\s+', line)
                if len(parts) >= 5:
                    # ✅ EEEE 是中国编号（NetCDF 中使用）
                    current_id = parts[4].strip()
                    data[current_id] = []  # 新建记录
                else:
                    current_id = None
            elif current_id:
                parts = re.split(r'\s+', line)
                if len(parts) >= 5:
                    try:
                        time_str = parts[0]
                        lat = float(parts[2]) / 10.0
                        lon = float(parts[3]) / 10.0
                        data[current_id].append((time_str, lat, lon))
                    except:
                        continue
    return data

# 分类路径方向
def classify_typhoon_path(track):
    if len(track) < 2:
        return "轨迹不足"
    _, slat, slon = track[0]
    _, elat, elon = track[-1]
    dlat, dlon = elat - slat, elon - slon
    angle = np.rad2deg(np.arctan2(dlat, dlon)) % 360
    if 45 <= angle <= 135:
        return '北上型'
    elif 135 < angle <= 180 or 0 <= angle < 45:
        return '东北偏北型'
    elif 225 <= angle <= 315:
        return '西行型'
    elif 180 < angle < 225:
        return '登陆型'
    else:
        return '其他型'

# 判断路径是否进入浙江范围（118–123E，27–32N）
def path_affect_zhejiang(track):
    for _, lat, lon in track:
        if 118 <= lon <= 123 and 27 <= lat <= 32:
            return True
    return False

# 主函数：处理所有年份文件
def analyze_paths(txt_dir, nc_path, out_csv='浙江影响台风路径分类.csv'):
    impact_typhoons = get_impact_typhoons(nc_path)
    all_records = []

    for year in range(2010, 2025):
        txt_file = os.path.join(txt_dir, f'CH{year}BST.txt')
        if not os.path.exists(txt_file):
            print(f"未找到文件：{txt_file}")
            continue

        track_data = read_typhoon_track_file(txt_file)
        for tid, track in track_data.items():
            if tid not in impact_typhoons:
                continue  # 跳过非浙江影响台风
            category = classify_typhoon_path(track)
            in_zhejiang = path_affect_zhejiang(track)
            all_records.append({
                "年份": year,
                "台风编号": tid,
                "路径点数": len(track),
                "路径分类": category,
                "路径是否进入浙江": in_zhejiang
            })

    df = pd.DataFrame(all_records)
    df.to_csv(out_csv, index=False)

    return df

# 示例调用
if __name__ == '__main__':
    txt_dir = '/Users/momo/Desktop/业务相关/2025 影响台风大风/热带气旋最佳路径数据集'
    nc_path = '/Users/momo/Desktop/业务相关/2025 影响台风大风/All_Typhoons_ExMaxWind.nc'
    df = analyze_paths(txt_dir, nc_path)
    print(df)
