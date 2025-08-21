import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# ------------------------
# 1. 基本设置
# ------------------------
plt.rcParams['font.sans-serif'] = ['Heiti TC']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据路径
obs_file = "/Users/momo/Desktop/业务相关/2025 影响台风大风/All_Typhoons_MaxWind.nc"

# 读取观测数据
ds_obs = xr.open_dataset(obs_file)

# ------------------------
# 2. 提取指定时刻的观测数据
# ------------------------
target_time = np.datetime64("2024-11-01T04:00:00")
time_idx = np.where(ds_obs['INITTIME'].values == target_time)[0][0]

# 提取站点风速和风向
wind_speed_obs = ds_obs['wind_velocity'][time_idx, 0, :].values
wind_dir_obs = ds_obs['wind_direction'][time_idx, 0, :].values
lat_obs = ds_obs['lat'].values
lon_obs = ds_obs['lon'].values

# 转换为 U/V 分量（气象风向：风从该方向吹来）
u_obs = -wind_speed_obs * np.sin(np.radians(wind_dir_obs))
v_obs = -wind_speed_obs * np.cos(np.radians(wind_dir_obs))

# ------------------------
# 3. 绘图
# ------------------------
fig = plt.figure(figsize=(10, 8))
proj = ccrs.PlateCarree()

# 设置绘图区域（可调整到更精确范围）
map_extent = [118, 123, 27, 32]  # [lon_min, lon_max, lat_min, lat_max]


ax = fig.add_subplot(1, 1, 1, projection=proj)
ax.set_extent(map_extent, crs=proj)

# 添加地理要素
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle='-', linewidth=1, edgecolor='black')  # 国界
ax.add_feature(cfeature.STATES.with_scale('50m'), linestyle=':', linewidth=0.8, edgecolor='gray')  # 省界

# 添加经纬度网格（只在下方和左侧显示标签）
gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}

# 绘制风场箭头
q = ax.quiver(lon_obs, lat_obs, u_obs, v_obs,
              wind_speed_obs,  # 颜色按风速
              cmap='jet', scale=200, width=0.003, transform=proj)

# 颜色条放在图右侧
cb = plt.colorbar(q, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
cb.set_label('风速 (m/s)', fontsize=10)

# 标题
ax.set_title(f"站点实况风场 ({str(target_time)})", fontsize=14)

plt.tight_layout()
plt.show()