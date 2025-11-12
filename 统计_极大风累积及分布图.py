"""
极大风累积分布.py — 台风期间站点极大风速统计与可视化

总体目的：
- 对多次台风过程中每个观测站的极大风速进行统计与可视化，帮助识别在台风影响下风速较大的站点和空间分布特征。
- 提供单台风级别的风场图（平均风速/方向 与 极大风速/对应风向）以及全部台风的出现次数汇总与地图展示。

主要功能：
- 逐台风读取 NetCDF 中风速、风向与台风内部索引（id_to_index / index_to_cn / index_to_en 等），
  计算并绘制：
    * 每站点在该台风影响期的平均风速与平均风向（并保存地图）；
    * 每站点在该台风影响期的最大风速与对应风向（并保存地图）。
- 统计所有台风中每个站点出现“极大风速最大站”的次数，保存为 CSV 并绘制计数分布地图。
- 对缺测值做基本容错，输出过程信息并记录保存路径。

输入（在脚本中配置）：
- nc_path: 含有变量 id_to_index、index_to_cn、index_to_en、STID、lat、lon、height、wind_velocity、wind_direction、typhoon_id_index 的 NetCDF 文件路径。
- typhoon_ids: 要处理的台风外部编号列表（脚本底部配置，可修改）。

输出（保存到脚本中指定的 output_dir）：
- 每台风的平均风场图：Typhoon_{id}_Average_Wind.png
- 每台风的极大风速图：Typhoon_{id}_Max_Wind.png
- 全体台风汇总站点计数 CSV：max_wind_count.csv
- 全体台风计数地图：Max_Wind_Count_Map.png

"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
import os
import warnings
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Heiti TC']


def parse_mapping(attr_str):
    """解析映射字符串为字典"""
    pairs = attr_str.strip().split(";")
    return {k.strip(): v.strip() for k, v in (p.split(":", 1) for p in pairs if ":" in p)}


def draw_station_wind_map(lons, lats, U, V, ws_values, stids, heights, title, save_path=None):
    """绘制站点风场图并保存"""
    fig = plt.figure(figsize=(10, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_title(title, fontsize=14)

    ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
    ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')
    ax.set_extent([np.min(lons) - 1, np.max(lons) + 1, np.min(lats) - 1, np.max(lats) + 1])

    # 绘制风羽
    ax.barbs(lons, lats, U, V, ws_values,
             length=6, pivot='middle', barb_increments=dict(half=2, full=4, flag=20),
             linewidth=0.6, zorder=3)

    # 添加站点信息
    for i in range(len(stids)):
        if np.isnan(ws_values[i]):
            continue
        ax.text(lons[i], lats[i] - 0.1, f"{stids[i]}\n{ws_values[i]:.1f} m/s\n{int(heights[i])} m",
                fontsize=7, ha='center', va='top', zorder=4)

    plt.tight_layout()

    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
        plt.close(fig)  # 关闭图形以释放内存
    else:
        plt.show()


def draw_count_map(lons, lats, counts, stids, heights, title, save_path=None):
    """绘制站点出现最大风速次数的地图"""
    fig = plt.figure(figsize=(10, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_title(title, fontsize=14)

    ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
    ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')
    ax.set_extent([np.min(lons) - 1, np.max(lons) + 1, np.min(lats) - 1, np.max(lats) + 1])

    # 绘制散点图，点的大小表示出现次数
    scatter = ax.scatter(lons, lats, s=counts * 20 + 10, c=counts, cmap='viridis',
                         alpha=0.7, transform=ccrs.PlateCarree())

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('出现次数', fontsize=10)

    # 添加站点信息
    for i in range(len(stids)):
        if counts[i] == 0:
            continue
        ax.text(lons[i], lats[i] + 0.1, f"{stids[i]}\n{int(heights[i])}m\n{counts[i]}次",
                fontsize=7, ha='center', va='bottom', zorder=4,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

    plt.tight_layout()

    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存计数图: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def process_typhoon(nc, typhoon_id, output_dir, station_count):
    """处理单个台风并生成图表，同时更新站点计数"""
    id_to_index = parse_mapping(nc.getncattr('id_to_index'))
    index_to_cn = parse_mapping(nc.getncattr('index_to_cn'))
    index_to_en = parse_mapping(nc.getncattr('index_to_en'))

    if typhoon_id not in id_to_index:
        print(f"警告: 台风编号 {typhoon_id} 不在数据中，跳过处理")
        return station_count

    ty_idx = int(id_to_index[typhoon_id])
    cn_name = index_to_cn[str(ty_idx)]
    en_name = index_to_en[str(ty_idx)]

    print(f"处理台风: {typhoon_id} ({cn_name}/{en_name})")

    stids = nc.variables['STID'][:].astype(str)
    lats = nc.variables['lat'][:]
    lons = nc.variables['lon'][:]
    heights = nc.variables['height'][:]

    # 确保所有数组都是可写的
    wind_speeds = np.array(nc.variables['wind_velocity'][:, 0, :], copy=True)
    wind_dirs = np.array(nc.variables['wind_direction'][:, 0, :], copy=True)
    typhoon_ids = np.array(nc.variables['typhoon_id_index'][:, 0, :], copy=True)

    n_stations = len(stids)

    # ----------- 极大风速和对应风向 -----------
    max_ws_list = []

    for i in range(n_stations):
        mask = typhoon_ids[:, i] == ty_idx
        if not np.any(mask):
            max_ws_list.append(np.nan)
            continue

        ws_values = wind_speeds[mask, i]

        # 检查有效数据
        valid_ws = ws_values[~np.isnan(ws_values)]
        if len(valid_ws) == 0:
            max_ws = np.nan
        else:
            max_ws = np.max(valid_ws)

        max_ws_list.append(max_ws)

    max_ws_arr = np.array(max_ws_list)

    # 找出该台风过程中风速最大的站点
    if not np.all(np.isnan(max_ws_arr)):
        max_value = np.nanmax(max_ws_arr)
        max_indices = np.where(max_ws_arr == max_value)[0]

        # 更新站点计数
        for idx in max_indices:
            station_count[idx] += 1
        print(f"  风速最大值: {max_value:.1f} m/s, 出现在 {len(max_indices)} 个站点")
    else:
        print("  无有效风速数据")

    # ----------- 平均风速和平均风向 -----------
    avg_ws_list, avg_wd_list = [], []

    for i in range(n_stations):
        mask = typhoon_ids[:, i] == ty_idx
        if not np.any(mask):
            avg_ws_list.append(np.nan)
            avg_wd_list.append(np.nan)
            continue

        # 提取数据
        ws_values = wind_speeds[mask, i]
        wd_values = wind_dirs[mask, i]

        # 检查有效数据
        valid_ws = ws_values[~np.isnan(ws_values)]
        if len(valid_ws) == 0:
            avg_ws = np.nan
            avg_wd = np.nan
        else:
            # 计算平均风速
            avg_ws = np.mean(valid_ws)

            # 计算平均风向
            valid_wd = wd_values[~np.isnan(wd_values)]
            if len(valid_wd) == 0:
                avg_wd = np.nan
            else:
                theta = np.deg2rad(valid_wd)
                u = -np.sin(theta)
                v = -np.cos(theta)
                u_mean = np.mean(u)
                v_mean = np.mean(v)
                avg_theta = np.arctan2(-u_mean, -v_mean)
                avg_wd = (np.rad2deg(avg_theta) + 360) % 360

        avg_ws_list.append(avg_ws)
        avg_wd_list.append(avg_wd)

    avg_ws_arr = np.array(avg_ws_list)
    avg_wd_arr = np.array(avg_wd_list)
    U_avg = -avg_ws_arr * np.sin(np.deg2rad(avg_wd_arr))
    V_avg = -avg_ws_arr * np.cos(np.deg2rad(avg_wd_arr))

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存平均风速图
    avg_save_path = os.path.join(output_dir, f"Typhoon_{typhoon_id}_Average_Wind.png")
    draw_station_wind_map(
        lons, lats, U_avg, V_avg, avg_ws_arr, stids, heights,
        f"Typhoon {typhoon_id} - {cn_name} ({en_name})\nAverage Wind Speed and Direction",
        save_path=avg_save_path
    )

    # ----------- 极大风速和对应风向 -----------
    max_ws_list, max_wd_list = [], []

    for i in range(n_stations):
        mask = typhoon_ids[:, i] == ty_idx
        if not np.any(mask):
            max_ws_list.append(np.nan)
            max_wd_list.append(np.nan)
            continue

        ws_values = wind_speeds[mask, i]
        wd_values = wind_dirs[mask, i]

        # 检查有效数据
        valid_ws = ws_values[~np.isnan(ws_values)]
        if len(valid_ws) == 0:
            max_ws = np.nan
            max_wd = np.nan
        else:
            max_idx = np.argmax(valid_ws)
            max_ws = valid_ws[max_idx]

            # 获取对应的风向
            valid_wd = wd_values[~np.isnan(wd_values)]
            if len(valid_wd) > 0:
                max_wd = valid_wd[max_idx % len(valid_wd)]  # 确保索引在有效范围内
            else:
                max_wd = np.nan

        max_ws_list.append(max_ws)
        max_wd_list.append(max_wd)

    max_ws_arr = np.array(max_ws_list)
    max_wd_arr = np.array(max_wd_list)
    U_max = -max_ws_arr * np.sin(np.deg2rad(max_wd_arr))
    V_max = -max_ws_arr * np.cos(np.deg2rad(max_wd_arr))

    # 保存最大风速图
    max_save_path = os.path.join(output_dir, f"Typhoon_{typhoon_id}_Max_Wind.png")
    draw_station_wind_map(
        lons, lats, U_max, V_max, max_ws_arr, stids, heights,
        f"Typhoon {typhoon_id} - {cn_name} ({en_name})\nMax Wind Speed and Corresponding Direction",
        save_path=max_save_path
    )

    return station_count


def save_count_data(stids, lats, lons, heights, counts, output_dir):
    """保存站点计数数据到CSV文件"""
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "max_wind_count.csv")

    # 创建DataFrame
    data = {
        'StationID': stids,
        'Latitude': lats,
        'Longitude': lons,
        'Height': heights,
        'Count': counts
    }
    df = pd.DataFrame(data)

    # 按计数降序排序
    df = df.sort_values(by='Count', ascending=False)

    # 保存到CSV
    df.to_csv(save_path, index=False)
    print(f"已保存计数数据: {save_path}")

    return df


def process_all_typhoons(nc_path, typhoon_ids, output_dir="台风风场图"):
    """处理所有台风"""
    print(f"开始处理 {len(typhoon_ids)} 个台风...")

    # 初始化站点计数数组
    station_count = None

    # 打开NetCDF文件
    with Dataset(nc_path) as nc:
        # 获取站点信息
        stids = nc.variables['STID'][:].astype(str)
        lats = nc.variables['lat'][:]
        lons = nc.variables['lon'][:]
        heights = nc.variables['height'][:]

        # 初始化站点计数数组
        n_stations = len(stids)
        station_count = np.zeros(n_stations, dtype=int)

        for typhoon_id in typhoon_ids:
            try:
                station_count = process_typhoon(nc, typhoon_id, output_dir, station_count)
            except Exception as e:
                print(f"处理台风 {typhoon_id} 时出错: {e}")

    # 保存计数数据
    df = save_count_data(stids, lats, lons, heights, station_count, output_dir)

    # 打印计数最多的前10个站点
    print("\n站点出现最大风速次数排名:")
    print(df.head(10))

    # 绘制计数图
    count_save_path = os.path.join(output_dir, "Max_Wind_Count_Map.png")
    draw_count_map(
        lons, lats, station_count, stids, heights,
        "台风过程中极大风速最大站点出现次数",
        save_path=count_save_path
    )

    print("所有台风处理完成!")


# 台风ID列表
typhoon_ids = ['1006', '1007', '1010', '1105', '1109', '1111', '1209', '1211', '1215',
               '1307', '1312', '1315', '1323', '1410', '1416', '1509', '1513', '1521',
               '1601', '1614', '1616', '1617', '1618', '1622', '1709', '1710', '1718',
               '1808', '1810', '1812', '1814', '1818', '1825', '1905', '1909', '1911',
               '1913', '1917', '1918', '2004', '2008', '2009', '2106', '2109', '2112',
               '2114', '2204', '2211', '2212', '2305', '2306', '2311', '2314', '2403',
               '2413', '2414', '2418', '2421', '2425']

# 忽略空切片的警告
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

# 处理所有台风
nc_path = '/Users/momo/Desktop/业务相关/2025 影响台风大风/All_Typhoons_MaxWind.nc'
process_all_typhoons(nc_path, typhoon_ids)