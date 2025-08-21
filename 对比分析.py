import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.metrics import mean_squared_error, r2_score
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 读取数据
u_file = "/Users/momo/Desktop/业务相关/2025 影响台风大风/HRCLDAS/UWIN/2024/20241101/Z_NAFP_C_BABJ_20241101041731_P_HRCLDAS_RT_BEHZ_0P01_HOR-UWIN-2024110104.GRB2"
v_file = "/Users/momo/Desktop/业务相关/2025 影响台风大风/HRCLDAS/VWIN/2024/20241101/Z_NAFP_C_BABJ_20241101041721_P_HRCLDAS_RT_BEHZ_0P01_HOR-VWIN-2024110104.GRB2"
obs_file = "/Users/momo/Desktop/业务相关/2025 影响台风大风/All_Typhoons_MaxWind.nc"

ds_u = xr.open_dataset(u_file, engine="cfgrib", decode_timedelta=True)
ds_v = xr.open_dataset(v_file, engine="cfgrib", decode_timedelta=True)
ds_obs = xr.open_dataset(obs_file)

# 2. 数据处理
target_time = np.datetime64("2024-11-01T04:00:00")
time_idx = np.where(ds_obs['INITTIME'].values == target_time)[0][0]

# 提取观测数据
wind_speed_obs = ds_obs['wind_velocity'][time_idx, 0, :].values
wind_dir_obs = ds_obs['wind_direction'][time_idx, 0, :].values
lat_obs = ds_obs['lat'].values
lon_obs = ds_obs['lon'].values
u_obs = -wind_speed_obs * np.sin(np.radians(wind_dir_obs))
v_obs = -wind_speed_obs * np.cos(np.radians(wind_dir_obs))

# 插值处理
lon_grid, lat_grid = ds_u['longitude'].values, ds_u['latitude'].values
points = np.column_stack([np.meshgrid(lon_grid, lat_grid)[0].ravel(), 
                         np.meshgrid(lon_grid, lat_grid)[1].ravel()])
u_interp = griddata(points, ds_u['u10'].values.ravel(), (lon_obs, lat_obs), method='linear')
v_interp = griddata(points, ds_v['v10'].values.ravel(), (lon_obs, lat_obs), method='linear')

# 过滤NaN值
valid_mask = (~np.isnan(u_obs)) & (~np.isnan(v_obs)) & (~np.isnan(u_interp)) & (~np.isnan(v_interp))
u_obs_clean, v_obs_clean = u_obs[valid_mask], v_obs[valid_mask]
u_interp_clean, v_interp_clean = u_interp[valid_mask], v_interp[valid_mask]

# 3. 计算评估指标
def calculate_metrics(obs, pred):
    cor = np.corrcoef(obs, pred)[0, 1]
    mre = np.mean(np.abs(obs - pred) / (np.abs(obs) + 1e-6))
    rmse = np.sqrt(mean_squared_error(obs, pred))
    return cor, mre, rmse

cor_u, mre_u, rmse_u = calculate_metrics(u_obs_clean, u_interp_clean)
cor_v, mre_v, rmse_v = calculate_metrics(v_obs_clean, v_interp_clean)

# 4. 创建大图
fig = plt.figure(figsize=(15, 15))

# 定义地图投影和范围
proj = ccrs.PlateCarree()
china_bbox = [115, 125, 27, 35]  # 经度最小/最大，纬度最小/最大

# 第一行：观测数据
ax1 = fig.add_subplot(3, 2, 1, projection=proj)
ax1.set_extent(china_bbox, crs=proj)
ax1.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax1.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':')
ax1.add_feature(cfeature.STATES.with_scale('50m'), linestyle=':')
sc1 = ax1.scatter(lon_obs, lat_obs, c=u_obs, cmap='coolwarm', 
                 s=30, transform=proj, vmin=-15, vmax=15)
plt.colorbar(sc1, ax=ax1, label='U分量 (m/s)')
ax1.set_title('观测U分量分布')

ax2 = fig.add_subplot(3, 2, 2, projection=proj)
ax2.set_extent(china_bbox, crs=proj)
ax2.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax2.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':')
ax2.add_feature(cfeature.STATES.with_scale('50m'), linestyle=':')
sc2 = ax2.scatter(lon_obs, lat_obs, c=v_obs, cmap='coolwarm', 
                 s=30, transform=proj, vmin=-15, vmax=15)
plt.colorbar(sc2, ax=ax2, label='V分量 (m/s)')
ax2.set_title('观测V分量分布')

# 第二行：插值数据
ax3 = fig.add_subplot(3, 2, 3, projection=proj)
ax3.set_extent(china_bbox, crs=proj)
ax3.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax3.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':')
ax3.add_feature(cfeature.STATES.with_scale('50m'), linestyle=':')
sc3 = ax3.scatter(lon_obs[valid_mask], lat_obs[valid_mask], c=u_interp_clean, 
                 cmap='coolwarm', s=30, transform=proj, vmin=-15, vmax=15)
plt.colorbar(sc3, ax=ax3, label='U分量 (m/s)')
ax3.set_title('HRCLDAS插值U分量')

ax4 = fig.add_subplot(3, 2, 4, projection=proj)
ax4.set_extent(china_bbox, crs=proj)
ax4.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax4.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':')
ax4.add_feature(cfeature.STATES.with_scale('50m'), linestyle=':')
sc4 = ax4.scatter(lon_obs[valid_mask], lat_obs[valid_mask], c=v_interp_clean, 
                 cmap='coolwarm', s=30, transform=proj, vmin=-15, vmax=15)
plt.colorbar(sc4, ax=ax4, label='V分量 (m/s)')
ax4.set_title('HRCLDAS插值V分量')

# 第三行：评估结果
ax5 = fig.add_subplot(3, 2, 5)
ax5.scatter(u_obs_clean, u_interp_clean, alpha=0.6, color='blue')
ax5.plot([-20, 20], [-20, 20], 'r--')
ax5.set_xlabel('观测U分量 (m/s)')
ax5.set_ylabel('插值U分量 (m/s)')
ax5.set_title(f'U分量评估: COR={cor_u:.2f}, RMSE={rmse_u:.2f} m/s')
ax5.set_xlim(-20, 20)
ax5.set_ylim(-20, 20)

ax6 = fig.add_subplot(3, 2, 6)
ax6.scatter(v_obs_clean, v_interp_clean, alpha=0.6, color='green')
ax6.plot([-20, 20], [-20, 20], 'r--')
ax6.set_xlabel('观测V分量 (m/s)')
ax6.set_ylabel('插值V分量 (m/s)')
ax6.set_title(f'V分量评估: COR={cor_v:.2f}, RMSE={rmse_v:.2f} m/s')
ax6.set_xlim(-20, 20)
ax6.set_ylim(-20, 20)

# 调整布局
plt.tight_layout()
# plt.savefig('/Users/momo/Desktop/wind_analysis.png', dpi=300, bbox_inches='tight')
plt.show()