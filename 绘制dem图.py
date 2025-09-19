import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# 文件路径
DEM_PATH = r"/Users/momo/Desktop/业务相关/2025 影响台风大风/地形文件/DEM_0P05_CHINA.nc"

plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

def get_province_boundaries(extent):
    """获取省界数据（这里使用内置的省界，如果需要更精确的可以下载shapefile）"""
    try:
        # 使用Cartopy内置的省界数据
        return cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none'
        )
    except:
        print("警告: 无法加载省界数据，将使用默认边界")
        return None

def plot_dem():
    try:
        # 打开DEM文件
        ds = xr.open_dataset(DEM_PATH)
        
        # 选择合适的地形变量 - 根据描述，HGT_M 或 dhgt_gfs 都可用
        # HGT_M 单位是Degree，但描述为height，可能需要进行单位转换
        # dhgt_gfs 单位是meters MSL，更直接
        dem_data = ds['dhgt_gfs']  # 或者使用 ds['HGT_M']
        
        # 获取经纬度坐标
        lons = ds['Lon'].values
        lats = ds['Lat'].values
        
        # 创建地图
        fig = plt.figure(figsize=(12, 10))
        
        # 使用合适的投影（这里使用PlateCarree，适合中国区域）
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # 设置地图范围（根据DEM文件的经纬度范围）
        ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], 
                      crs=ccrs.PlateCarree())
        
        # 添加地图特征
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.RIVERS, linewidth=0.5, alpha=0.5)
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.set_extent([118, 123, 27, 31.5], crs=ccrs.PlateCarree())

        # 添加省界
        province_boundary = get_province_boundaries([118, 123, 27, 31.5])
        if province_boundary:
            ax.add_feature(province_boundary, linewidth=1.0, edgecolor='grey', linestyle='-')

        
        # 绘制地形等高线或填色图
        # 方法1：填色图
        contourf = ax.contourf(lons, lats, dem_data.values, 
                              levels=50, 
                              cmap='terrain', 
                              alpha=0.7, 
                              transform=ccrs.PlateCarree())
        
        # 方法2：等高线（可选）
        contour = ax.contour(lons, lats, dem_data.values, 
                           levels=20, 
                           colors='black', 
                           linewidths=0.5, 
                           alpha=0.5, 
                           transform=ccrs.PlateCarree())
        
        # 添加颜色条
        cbar = plt.colorbar(contourf, ax=ax, shrink=0.8)
        cbar.set_label('Elevation (meters MSL)', fontsize=12)
        
        # 添加网格和坐标标签
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                         linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}
        
        # 设置标题
        ax.set_title('地形图', fontsize=16, pad=20)
        
        # 这里可以添加你的风玫瑰图数据
        # 示例：在特定位置添加风玫瑰（你需要替换为实际的风数据）
        # add_wind_roses(ax, wind_data)
        
        # 关闭文件
        ds.close()
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")

def plot_dem_3d_surface():
    """3D地形表面图"""
    try:
        ds = xr.open_dataset(DEM_PATH)
        dem_data = ds['dhgt_gfs']
        lons = ds['Lon'].values
        lats = ds['Lat'].values
        
        # 创建3D图
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 创建网格（为了性能，可以降采样）
        X, Y = np.meshgrid(lons[::10], lats[::10])  # 每10个点取一个
        Z = dem_data.values[::10, ::10]
        
        # 绘制3D表面
        surf = ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.8, 
                              rstride=1, cstride=1, linewidth=0)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Elevation (m)')
        ax.set_title('3D 地形图')
        
        # 添加颜色条
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
        
        ds.close()
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in 3D plot: {e}")

def get_dem_at_location(lon, lat):
    """获取特定位置的地形高度"""
    try:
        ds = xr.open_dataset(DEM_PATH)
        # 使用最近邻插值获取指定位置的地形高度
        height = ds['dhgt_gfs'].sel(Lon=lon, Lat=lat, method='nearest').values
        ds.close()
        return height
    except Exception as e:
        print(f"Error getting DEM at location: {e}")
        return None

# 如果你想在现有风玫瑰图代码中添加地形，可以使用这个函数
def add_topography_to_existing_map(ax):
    """在现有的地图轴对象上添加地形"""
    try:
        ds = xr.open_dataset(DEM_PATH)
        dem_data = ds['dhgt_gfs']
        lons = ds['Lon'].values
        lats = ds['Lat'].values
        
        # 添加地形填色
        contourf = ax.contourf(lons, lats, dem_data.values, 
                              levels=30, 
                              cmap='terrain', 
                              alpha=0.6, 
                              transform=ccrs.PlateCarree())
        
        # 添加等高线
        ax.contour(lons, lats, dem_data.values, 
                  levels=15, 
                  colors='black', 
                  linewidths=0.3, 
                  alpha=0.4, 
                  transform=ccrs.PlateCarree())
        
        ds.close()
        return contourf
        
    except Exception as e:
        print(f"Error adding topography: {e}")
        return None

if __name__ == "__main__":
    # 绘制带地形的2D地图
    plot_dem()
    
    # 如果需要3D视图，取消注释下面这行
    plot_dem_3d_surface()
    
    # 示例：获取上海位置的地形高度
    # shanghai_height = get_dem_at_location(121.47, 31.23)
    # print(f"Shanghai elevation: {shanghai_height} meters")