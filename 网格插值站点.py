import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# ------------------------
# 1. 数据路径
# ------------------------
u_file = "/Users/momo/Desktop/业务相关/2025 影响台风大风/HRCLDAS/UWIN/2024/20241101/Z_NAFP_C_BABJ_20241101041731_P_HRCLDAS_RT_BEHZ_0P01_HOR-UWIN-2024110104.GRB2"
v_file = "/Users/momo/Desktop/业务相关/2025 影响台风大风/HRCLDAS/VWIN/2024/20241101/Z_NAFP_C_BABJ_20241101041721_P_HRCLDAS_RT_BEHZ_0P01_HOR-VWIN-2024110104.GRB2"
obs_file = "/Users/momo/Desktop/业务相关/2025 影响台风大风/All_Typhoons_ExMaxWind.nc"

# ------------------------
# 2. 读取数据
# ------------------------
ds_u = xr.open_dataset(u_file, engine="cfgrib", decode_timedelta=True)
ds_v = xr.open_dataset(v_file, engine="cfgrib", decode_timedelta=True)
ds_obs = xr.open_dataset(obs_file)

# ------------------------
# 3. 获取观测站点位置
# ------------------------
target_time = np.datetime64("2024-11-01T04:00:00")
time_idx = np.where(ds_obs['INITTIME'].values == target_time)[0][0]

lat_obs = ds_obs['lat'].values
lon_obs = ds_obs['lon'].values

# ------------------------
# 4. 将网格实况插值到站点位置
# ------------------------
lon_grid, lat_grid = ds_u['longitude'].values, ds_u['latitude'].values
grid_lon, grid_lat = np.meshgrid(lon_grid, lat_grid)
points = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])

u_interp = griddata(points, ds_u['u10'].values.ravel(), (lon_obs, lat_obs), method='linear')
v_interp = griddata(points, ds_v['v10'].values.ravel(), (lon_obs, lat_obs), method='linear')

# 计算插值风速
wind_speed_interp = np.sqrt(u_interp**2 + v_interp**2)

# ------------------------
# 5. 绘制站点化的网格实况风场
# ------------------------
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(10, 8))
proj = ccrs.PlateCarree()

# 地图范围自动根据站点数据设定
extent = [lon_obs.min()-0.5, lon_obs.max()+0.5,
          lat_obs.min()-0.5, lat_obs.max()+0.5]

ax = fig.add_subplot(1, 1, 1, projection=proj)
ax.set_extent(extent, crs=proj)

# 地理要素
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle='-', linewidth=1)
ax.add_feature(cfeature.STATES.with_scale('50m'), linestyle=':', linewidth=0.8)

# 经纬度网格（左+下）
gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.top_labels = False
gl.right_labels = False

# 绘制箭头
q = ax.quiver(lon_obs, lat_obs, u_interp, v_interp,
              wind_speed_interp, cmap='jet',
              scale=200, width=0.003, transform=proj)

# 颜色条
cb = plt.colorbar(q, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
cb.set_label('风速 (m/s)', fontsize=10)

ax.set_title(f"网格实况插值到站点 ({str(target_time)})", fontsize=14)

plt.tight_layout()
plt.show()