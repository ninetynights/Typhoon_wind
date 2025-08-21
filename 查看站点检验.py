from netCDF4 import Dataset, num2date
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import re

plt.rcParams['font.sans-serif'] = ['Heiti TC']

# 用户输入设置
target_typhoon_id = '1307'
target_station_id = '58663'
nc_path = '/Users/momo/Desktop/业务相关/2025 影响台风大风/All_Typhoons_ExMaxWind.nc'

# 打开 NetCDF 文件
nc = Dataset(nc_path)

# 解析映射属性
def parse_mapping(attr_str):
    pairs = attr_str.strip().split(";")
    return {k.strip(): v.strip() for k, v in (p.split(":", 1) for p in pairs if ":" in p)}

id_to_index = parse_mapping(nc.getncattr('id_to_index'))
index_to_cn = parse_mapping(nc.getncattr('index_to_cn'))
index_to_en = parse_mapping(nc.getncattr('index_to_en'))

# 获取台风索引及名称
if target_typhoon_id not in id_to_index:
    raise ValueError(f"台风编号 {target_typhoon_id} 不在数据中")
typhoon_index = int(id_to_index[target_typhoon_id])

# 获取台风中英文名
cn_name = index_to_cn[str(typhoon_index)]
en_name = index_to_en[str(typhoon_index)]

# 查找站点索引
stid_list = nc.variables['STID'][:].astype(str)
if target_station_id not in stid_list:
    raise ValueError(f"站点编号 {target_station_id} 不在数据中")
station_index = np.where(stid_list == target_station_id)[0][0]

# 提取数据
typhoon_ids = nc.variables['typhoon_id_index'][:, 0, station_index]
wind_speeds = nc.variables['wind_velocity'][:, 0, station_index]
wind_dirs = nc.variables['wind_direction'][:, 0, station_index]
init_times = nc.variables['INITTIME'][:]
init_units = nc.variables['INITTIME'].units
time_dt = num2date(init_times, init_units)

# 找出台风影响时间段索引
valid_idx = np.where(typhoon_ids == typhoon_index)[0]
if len(valid_idx) == 0:
    raise ValueError("该站点没有此台风影响数据")

# 数据处理
times = [datetime.datetime(t.year, t.month, t.day, t.hour, t.minute) for t in time_dt[valid_idx]]
ws = wind_speeds[valid_idx]
wd = wind_dirs[valid_idx]

# 计算风速方向分量（单位向量 * 风速）
theta_rad = np.deg2rad(wd)
U = -ws * np.sin(theta_rad)
V = -ws * np.cos(theta_rad)

# 计算平均风速与总时长
avg_ws = np.nanmean(ws)
max_ws = np.nanmax(ws)
duration_hr = len(times)

# 可视化
fig, ax = plt.subplots(figsize=(14, 6))

# 折线图：风速
ax.plot(times, ws, color='darkgrey', label='Max Wind Speed (m/s)', linewidth=0.8, zorder=3)

# 标注每个点的风速数值（位于点下方）
for x, y in zip(times, ws):
    if not np.isnan(y):
        ax.text(x, y - 0.1, f"{y:.1f}", ha='center', va='top', fontsize=8, color='grey', zorder=5)

# 绘制风向杆（barbs）
ax.barbs(times, ws, U, V, length=6, color='black', linewidth=1,
         barb_increments=dict(half=2, full=4, flag=20), zorder=4)

# 格式化时间轴
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
fig.autofmt_xdate(rotation=45)

# 标签和标题
ax.set_xlabel('Time (Hourly)')
ax.set_ylabel('Wind Speed (m/s)')
ax.set_title(f"Typhoon {target_typhoon_id} - {cn_name} ({en_name}) - Station {target_station_id}\n"
             f"Hourly Max Wind Speed and Direction", fontsize=14)

# 在图下方显示统计信息
text_stats = f"Duration: {duration_hr} hours  |  Avg Wind Speed: {avg_ws:.2f} m/s  |  Max Wind Speed: {max_ws:.2f} m/s"
plt.figtext(0.5, 0.01, text_stats, ha='center', fontsize=11)

# 其他设置
ax.grid(True, linestyle='--', color='gray', alpha=0.3, zorder=0)
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()