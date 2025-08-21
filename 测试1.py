import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
import warnings

plt.rcParams['font.sans-serif'] = ['Heiti TC']


def parse_mapping(attr_str):
    pairs = attr_str.strip().split(";")
    return {k.strip(): v.strip() for k, v in (p.split(":", 1) for p in pairs if ":" in p)}


def draw_station_wind_map(lons, lats, U, V, ws_values, stids, heights, title):
    fig = plt.figure(figsize=(10, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_title(title, fontsize=14)

    ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
    ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')
    ax.set_extent([np.min(lons) - 1, np.max(lons) + 1, np.min(lats) - 1, np.max(lats) + 1])

    ax.barbs(lons, lats, U, V, ws_values,
             length=6, pivot='middle', barb_increments=dict(half=2, full=4, flag=20),
             linewidth=0.6, zorder=3)

    for i in range(len(stids)):
        if np.isnan(ws_values[i]):
            continue
        ax.text(lons[i], lats[i] - 0.1, f"{stids[i]}\n{ws_values[i]:.1f} m/s\n{int(heights[i])} m",
                fontsize=7, ha='center', va='top', zorder=4)

    plt.tight_layout()
    plt.show()


def plot_typhoon_avg_and_max_wind_map(typhoon_id, nc_path):
    nc = Dataset(nc_path)

    id_to_index = parse_mapping(nc.getncattr('id_to_index'))
    index_to_cn = parse_mapping(nc.getncattr('index_to_cn'))
    index_to_en = parse_mapping(nc.getncattr('index_to_en'))

    if typhoon_id not in id_to_index:
        raise ValueError(f"台风编号 {typhoon_id} 不在数据中")

    ty_idx = int(id_to_index[typhoon_id])
    cn_name = index_to_cn[str(ty_idx)]
    en_name = index_to_en[str(ty_idx)]

    stids = nc.variables['STID'][:].astype(str)
    lats = nc.variables['lat'][:]
    lons = nc.variables['lon'][:]
    heights = nc.variables['height'][:]

    # 确保所有数组都是可写的
    wind_speeds = np.array(nc.variables['wind_velocity'][:, 0, :], copy=True)
    wind_dirs = np.array(nc.variables['wind_direction'][:, 0, :], copy=True)
    typhoon_ids = np.array(nc.variables['typhoon_id_index'][:, 0, :], copy=True)

    n_stations = len(stids)

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

    draw_station_wind_map(
        lons, lats, U_avg, V_avg, avg_ws_arr, stids, heights,
        f"Typhoon {typhoon_id} - {cn_name} ({en_name})\nAverage Wind Speed and Direction"
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

    draw_station_wind_map(
        lons, lats, U_max, V_max, max_ws_arr, stids, heights,
        f"Typhoon {typhoon_id} - {cn_name} ({en_name})\nMax Wind Speed and Corresponding Direction"
    )


# 忽略空切片的警告
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

plot_typhoon_avg_and_max_wind_map('1315', '/Users/momo/Desktop/业务相关/2025 影响台风大风/All_Typhoons_ExMaxWind.nc')