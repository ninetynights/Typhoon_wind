import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------------------------
# 1. 数据路径
# ------------------------
u_file = "/Users/momo/Desktop/业务相关/2025 影响台风大风/HRCLDAS/UWIN/2024/20241101/Z_NAFP_C_BABJ_20241101041731_P_HRCLDAS_RT_BEHZ_0P01_HOR-UWIN-2024110104.GRB2"
v_file = "/Users/momo/Desktop/业务相关/2025 影响台风大风/HRCLDAS/VWIN/2024/20241101/Z_NAFP_C_BABJ_20241101041721_P_HRCLDAS_RT_BEHZ_0P01_HOR-VWIN-2024110104.GRB2"
obs_file = "/Users/momo/Desktop/业务相关/2025 影响台风大风/All_Typhoons_MaxWind.nc"

# ------------------------
# 2. 读取数据
# ------------------------
ds_u = xr.open_dataset(u_file, engine="cfgrib", decode_timedelta=True)
ds_v = xr.open_dataset(v_file, engine="cfgrib", decode_timedelta=True)
ds_obs = xr.open_dataset(obs_file)

# ------------------------
# 3. 获取观测站点位置与实况
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
# 4. 将网格实况插值到站点位置
# ------------------------
lon_grid, lat_grid = ds_u['longitude'].values, ds_u['latitude'].values
grid_lon, grid_lat = np.meshgrid(lon_grid, lat_grid)
points = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])

u_interp = griddata(points, ds_u['u10'].values.ravel(), (lon_obs, lat_obs), method='linear')
v_interp = griddata(points, ds_v['v10'].values.ravel(), (lon_obs, lat_obs), method='linear')
wind_speed_interp = np.sqrt(u_interp**2 + v_interp**2)

# ------------------------
# 5. 计算误差分量和指标，内部过滤NaN
# ------------------------
u_diff = u_interp - u_obs
v_diff = v_interp - v_obs
wind_speed_diff = np.sqrt(u_diff**2 + v_diff**2)

def calc_metrics(obs, pred):
    # 过滤NaN，确保输入有效
    mask = (~np.isnan(obs)) & (~np.isnan(pred))
    obs_valid = obs[mask]
    pred_valid = pred[mask]
    mae = mean_absolute_error(obs_valid, pred_valid)
    rmse = np.sqrt(mean_squared_error(obs_valid, pred_valid))
    cor = np.corrcoef(obs_valid, pred_valid)[0, 1]
    return mae, rmse, cor

mae_u, rmse_u, cor_u = calc_metrics(u_obs, u_interp)
mae_v, rmse_v, cor_v = calc_metrics(v_obs, v_interp)

# ------------------------
# 6. 绘图
# ------------------------
import matplotlib.gridspec as gridspec

plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

proj = ccrs.PlateCarree()
extent = [lon_obs.min()-0.5, lon_obs.max()+0.5,
          lat_obs.min()-0.5, lat_obs.max()+0.5]


fig = plt.figure(figsize=(14, 16), constrained_layout=True)
gs = gridspec.GridSpec(3, 2, width_ratios=[3, 2])



titles = [
    f"站点实况风场 ({str(target_time)})",
    f"网格实况插值到站点风场 ({str(target_time)})",
    f"误差风场 (插值-实况)\nU MAE={mae_u:.2f}, RMSE={rmse_u:.2f}, COR={cor_u:.2f}\nV MAE={mae_v:.2f}, RMSE={rmse_v:.2f}, COR={cor_v:.2f}"
]
u_data = [u_obs, u_interp, u_diff]
v_data = [v_obs, v_interp, v_diff]
spd_data = [wind_speed_obs, wind_speed_interp, wind_speed_diff]

for i in range(3):
    ax = fig.add_subplot(gs[i, 0], projection=proj)
    ax.set_extent(extent, crs=proj)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle='-', linewidth=1)
    ax.add_feature(cfeature.STATES.with_scale('50m'), linestyle=':', linewidth=0.8)
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.top_labels = False
    gl.right_labels = False
    
    q = ax.quiver(lon_obs, lat_obs, u_data[i], v_data[i], spd_data[i],
                  cmap='jet', scale=200, width=0.003, transform=proj)
    ax.set_title(titles[i], fontsize=13)

cb_ax = fig.add_axes([0.12, 0.07, 0.38, 0.02])
cb = fig.colorbar(q, cax=cb_ax, orientation='horizontal')
cb.set_label('风速 (m/s)', fontsize=12)

ax_u = fig.add_subplot(gs[0, 1])
ax_u.scatter(u_obs, u_interp, alpha=0.7, edgecolors='k', color='blue')
ax_u.plot([-20, 20], [-20, 20], 'r--')
ax_u.set_xlabel('观测U分量 (m/s)')
ax_u.set_ylabel('插值U分量 (m/s)')
ax_u.set_title('U分量散点图')
ax_u.set_xlim(-20, 20)
ax_u.set_ylim(-20, 20)
ax_u.grid(True)

ax_v = fig.add_subplot(gs[1, 1])
ax_v.scatter(v_obs, v_interp, alpha=0.7, edgecolors='k', color='green')
ax_v.plot([-20, 20], [-20, 20], 'r--')
ax_v.set_xlabel('观测V分量 (m/s)')
ax_v.set_ylabel('插值V分量 (m/s)')
ax_v.set_title('V分量散点图')
ax_v.set_xlim(-20, 20)
ax_v.set_ylim(-20, 20)
ax_v.grid(True)

ax_spd = fig.add_subplot(gs[2, 1])
ax_spd.scatter(wind_speed_obs, wind_speed_interp, alpha=0.7, edgecolors='k', color='purple')
ax_spd.plot([0, 25], [0, 25], 'r--')
ax_spd.set_xlabel('观测风速 (m/s)')
ax_spd.set_ylabel('插值风速 (m/s)')
ax_spd.set_title('风速散点图')
ax_spd.set_xlim(0, 25)
ax_spd.set_ylim(0, 25)
ax_spd.grid(True)


plt.show()