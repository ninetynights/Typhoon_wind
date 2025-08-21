import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# ===============================
# 1. 参数设置
# ===============================

# 数据文件路径（U 分量和 V 分量的 GRIB2 文件）
u_file = "/Users/momo/Desktop/业务相关/2025 影响台风大风/HRCLDAS/UWIN/2024/20241101/Z_NAFP_C_BABJ_20241101041731_P_HRCLDAS_RT_BEHZ_0P01_HOR-UWIN-2024110104.GRB2"
v_file = "/Users/momo/Desktop/业务相关/2025 影响台风大风/HRCLDAS/VWIN/2024/20241101/Z_NAFP_C_BABJ_20241101041721_P_HRCLDAS_RT_BEHZ_0P01_HOR-VWIN-2024110104.GRB2"

# 绘图范围（这里用浙江省）
region_bbox = {
    'lon_min': 118.0,
    'lon_max': 123.0,
    'lat_min': 27.0,
    'lat_max': 32.0
}

# 箭头稀疏间隔（避免箭头过密）
arrow_skip = 10

# ===============================
# 2. 数据读取
# ===============================

# 读取 GRIB 文件
ds_u = xr.open_dataset(u_file, engine="cfgrib", decode_timedelta=True)
ds_v = xr.open_dataset(v_file, engine="cfgrib", decode_timedelta=True)

# 提取 U/V 分量（10m 高度）
u10 = ds_u['u10']  # U 分量（东向为正）
v10 = ds_v['v10']  # V 分量（北向为正）

# ===============================
# 3. 确定纬度顺序并裁剪区域
# ===============================
# HRCLDAS 数据可能纬度是从北到南递减，所以需要判断 slice 顺序
if u10.latitude.values[0] > u10.latitude.values[-1]:  # 纬度递减
    lat_slice = slice(region_bbox['lat_max'], region_bbox['lat_min'])
else:  # 纬度递增
    lat_slice = slice(region_bbox['lat_min'], region_bbox['lat_max'])

lon_slice = slice(region_bbox['lon_min'], region_bbox['lon_max'])

# 裁剪区域数据
u_crop = u10.sel(longitude=lon_slice, latitude=lat_slice)
v_crop = v10.sel(longitude=lon_slice, latitude=lat_slice)

# ===============================
# 4. 计算风速（m/s）
# ===============================
wind_speed = np.sqrt(u_crop**2 + v_crop**2)

# ===============================
# 5. 绘制风场图
# ===============================
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

# 添加地理要素
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle='-', linewidth=1, edgecolor='black')  # 国界
ax.add_feature(cfeature.STATES.with_scale('50m'), linestyle=':', linewidth=0.8, edgecolor='gray')  # 省界
ax.add_feature(cfeature.RIVERS.with_scale('50m'))
ax.add_feature(cfeature.LAKES.with_scale('50m'))

# 绘制风速填色图
cf = ax.contourf(
    u_crop.longitude,
    u_crop.latitude,
    wind_speed,
    levels=np.arange(0, 20.5, 1),  # 风速分级（可根据数据调整）
    cmap='YlOrRd',
    extend='max'
)
plt.colorbar(cf, label='Wind Speed (m/s)', orientation='horizontal', pad=0.05)

# 绘制风向箭头（用 U/V 分量直接控制箭头方向）
qv = ax.quiver(
    u_crop.longitude.values[::arrow_skip],
    u_crop.latitude.values[::arrow_skip],
    u_crop.values[::arrow_skip, ::arrow_skip],
    v_crop.values[::arrow_skip, ::arrow_skip],
    wind_speed.values[::arrow_skip, ::arrow_skip],  # 用风速给箭头着色
    cmap='coolwarm',
    scale=100,    # 控制箭头长度
    width=0.002
)

# 给箭头加颜色条
plt.colorbar(qv, label='Wind Speed (m/s) [Quiver]', orientation='vertical', pad=0.02)

# 设置标题
ax.set_title(
    f"10m Wind Speed and Direction over Zhejiang\n"
    f"Valid Time: {ds_u['valid_time'].values}",
    fontsize=14
)

# 添加经纬度网格
gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5)
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
# 只保留下方和左侧的经纬度标签
gl.top_labels = False
gl.right_labels = False

gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}

# 显示图像
plt.show()