import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 读取HRCLDAS网格数据
u_file = "/Users/momo/Desktop/业务相关/2025 影响台风大风/HRCLDAS/UWIN/2024/20241101/Z_NAFP_C_BABJ_20241101041731_P_HRCLDAS_RT_BEHZ_0P01_HOR-UWIN-2024110104.GRB2"
v_file = "/Users/momo/Desktop/业务相关/2025 影响台风大风/HRCLDAS/VWIN/2024/20241101/Z_NAFP_C_BABJ_20241101041721_P_HRCLDAS_RT_BEHZ_0P01_HOR-VWIN-2024110104.GRB2"

ds_u = xr.open_dataset(u_file, engine="cfgrib", decode_timedelta=True)
ds_v = xr.open_dataset(v_file, engine="cfgrib", decode_timedelta=True)

# 提取U/V风场和经纬度网格
u_grid = ds_u['u10'].values  # 纬向风 (东向为正)
v_grid = ds_v['v10'].values  # 经向风 (北向为正)
lon_grid = ds_u['longitude'].values
lat_grid = ds_u['latitude'].values

# 生成网格点坐标矩阵
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

# 2. 读取站点观测数据
obs_file = "/Users/momo/Desktop/业务相关/2025 影响台风大风/All_Typhoons_ExMaxWind.nc"
ds_obs = xr.open_dataset(obs_file)

# 提取指定时间（2024-11-01 04:00）的数据
target_time = np.datetime64("2024-11-01T04:00:00")
time_idx = np.where(ds_obs['INITTIME'].values == target_time)[0][0]

# 获取站点风速和风向
wind_speed_obs = ds_obs['wind_velocity'][time_idx, 0, :].values  # 风速 (m/s)
wind_dir_obs = ds_obs['wind_direction'][time_idx, 0, :].values   # 风向 (度)
stid = ds_obs['STID'].values                                     # 站点ID
lat_obs = ds_obs['lat'].values                                   # 纬度
lon_obs = ds_obs['lon'].values                                   # 经度

# 将站点风速转换为U/V分量 (气象学定义: 0°为正北，顺时针增加)
u_obs = -wind_speed_obs * np.sin(np.radians(wind_dir_obs))
v_obs = -wind_speed_obs * np.cos(np.radians(wind_dir_obs))

# 3. 将HRCLDAS网格数据插值到站点位置 (双线性插值)
points = np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()])
u_interp = griddata(points, u_grid.ravel(), (lon_obs, lat_obs), method='linear')
v_interp = griddata(points, v_grid.ravel(), (lon_obs, lat_obs), method='linear')

# 4. 处理NaN值：移除观测或插值中存在NaN的站点
valid_mask = (~np.isnan(u_obs)) & (~np.isnan(v_obs)) & (~np.isnan(u_interp)) & (~np.isnan(v_interp))
u_obs_clean = u_obs[valid_mask]
v_obs_clean = v_obs[valid_mask]
u_interp_clean = u_interp[valid_mask]
v_interp_clean = v_interp[valid_mask]

# 5. 计算评估指标
def calculate_metrics(obs, pred):
    cor = np.corrcoef(obs, pred)[0, 1]
    mre = np.mean(np.abs(obs - pred) / (np.abs(obs) + 1e-6))  # 避免除以0
    rmse = np.sqrt(mean_squared_error(obs, pred))
    return cor, mre, rmse

# U分量评估
cor_u, mre_u, rmse_u = calculate_metrics(u_obs_clean, u_interp_clean)
# V分量评估
cor_v, mre_v, rmse_v = calculate_metrics(v_obs_clean, v_interp_clean)

# 6. 结果输出
print(f"有效站点数量: {len(u_obs_clean)}/{len(u_obs)}")
print(f"U分量评估结果: COR={cor_u:.3f}, MRE={mre_u:.3f}, RMSE={rmse_u:.3f} m/s")
print(f"V分量评估结果: COR={cor_v:.3f}, MRE={mre_v:.3f}, RMSE={rmse_v:.3f} m/s")

# 7. 绘制散点对比图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# U分量散点图
ax1.scatter(u_obs_clean, u_interp_clean, alpha=0.6, color='blue')
ax1.plot([-20, 20], [-20, 20], 'r--')
ax1.set_xlabel('观测U分量 (m/s)')
ax1.set_ylabel('HRCLDAS插值U分量 (m/s)')
ax1.set_title(f'U分量对比 (COR={cor_u:.2f})')

# V分量散点图
ax2.scatter(v_obs_clean, v_interp_clean, alpha=0.6, color='green')
ax2.plot([-20, 20], [-20, 20], 'r--')
ax2.set_xlabel('观测V分量 (m/s)')
ax2.set_ylabel('HRCLDAS插值V分量 (m/s)')
ax2.set_title(f'V分量对比 (COR={cor_v:.2f})')

plt.tight_layout()
# plt.savefig('/Users/momo/Desktop/wind_validation.png', dpi=300)
plt.show()