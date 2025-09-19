import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import warnings
from cartopy.io import DownloadWarning


plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题


# 忽略下载警告
warnings.filterwarnings("ignore", category=DownloadWarning)

# 文件路径
DEM_PATH = r"/Users/momo/Desktop/业务相关/2025 影响台风大风/地形文件/DEM_0P05_CHINA.nc"

# 浙江省的大致经纬度范围
ZHEJIANG_EXTENT = [118.0, 123.0, 27.0, 31.5]  # [min_lon, max_lon, min_lat, max_lat]


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


def plot_dem_2d(extent, title="指定区域地形图"):
    """绘制2D地形图"""
    try:
        # 打开DEM文件
        ds = xr.open_dataset(DEM_PATH)
        dem_data = ds['dhgt_gfs']
        
        # 提取指定范围内的数据
        mask = (ds.Lon >= extent[0]) & (ds.Lon <= extent[1]) & \
               (ds.Lat >= extent[2]) & (ds.Lat <= extent[3])
        
        region_data = dem_data.where(mask, drop=True)
        sub_lons = region_data.Lon.values
        sub_lats = region_data.Lat.values
        sub_dem = region_data.values
        
        # 创建地图
        fig = plt.figure(figsize=(12, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # 设置地图范围
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        # 添加地图特征
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5)
        ax.add_feature(cfeature.RIVERS.with_scale('50m'), linewidth=0.5, alpha=0.5)
        
        # 添加省界
        province_boundary = get_province_boundaries(extent)
        if province_boundary:
            ax.add_feature(province_boundary, linewidth=1.5, edgecolor='red', linestyle='-')

        # 绘制地形填色图
        contourf = ax.contourf(sub_lons, sub_lats, sub_dem, 
                              levels=50, 
                              cmap='terrain', 
                              alpha=0.7)
        
        # 添加等高线
        contour = ax.contour(sub_lons, sub_lats, sub_dem, 
                           levels=15, 
                           colors='black', 
                           linewidths=0.5, 
                           alpha=0.5)
        
        # 添加颜色条
        cbar = plt.colorbar(contourf, ax=ax, shrink=0.8)
        cbar.set_label('Elevation (meters MSL)', fontsize=12)
        
        # 添加网格
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        
        ax.set_title(title, fontsize=16, pad=20)
        
        ds.close()
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in 2D plot: {e}")

def plot_dem_3d(extent, title="3D地形图", elevation_angle=30, azimuth_angle=45, scale_factor=0.001):
    """绘制3D地形图"""
    try:
        # 打开DEM文件
        ds = xr.open_dataset(DEM_PATH)
        dem_data = ds['dhgt_gfs']
        
        # 提取指定范围内的数据
        mask = (ds.Lon >= extent[0]) & (ds.Lon <= extent[1]) & \
               (ds.Lat >= extent[2]) & (ds.Lat <= extent[3])
        
        region_data = dem_data.where(mask, drop=True)
        sub_lons = region_data.Lon.values
        sub_lats = region_data.Lat.values
        sub_dem = region_data.values
        
        # 创建网格
        X, Y = np.meshgrid(sub_lons, sub_lats)
        Z = sub_dem
        
        # 创建3D图
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 设置视角
        ax.view_init(elev=elevation_angle, azim=azimuth_angle)
        
        # 绘制3D表面（为了性能可以降采样）
        stride = max(1, len(sub_lons) // 100)  # 自动计算步长
        surf = ax.plot_surface(X[::stride, ::stride], Y[::stride, ::stride], 
                              Z[::stride, ::stride] * scale_factor,  # 缩放高度以便更好显示
                              cmap='terrain', 
                              alpha=0.9,
                              linewidth=0, 
                              antialiased=True)
        
        # 设置标签
        ax.set_xlabel('Longitude (°E)', labelpad=15)
        ax.set_ylabel('Latitude (°N)', labelpad=15)
        ax.set_zlabel(f'Elevation (m × {scale_factor})', labelpad=15)
        
        # 设置标题
        ax.set_title(title, fontsize=16, pad=20)
        
        # 添加颜色条
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, pad=0.1)
        cbar.set_label('Elevation (meters MSL)', fontsize=12)
        
        # 设置坐标轴格式
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}°'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}°'))
        
        ds.close()
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in 3D plot: {e}")

def plot_dem_3d_advanced(extent, title="高级3D地形图"):
    """高级3D地形图，包含更多自定义选项"""
    try:
        ds = xr.open_dataset(DEM_PATH)
        dem_data = ds['dhgt_gfs']
        
        # 提取数据
        mask = (ds.Lon >= extent[0]) & (ds.Lon <= extent[1]) & \
               (ds.Lat >= extent[2]) & (ds.Lat <= extent[3])
        
        region_data = dem_data.where(mask, drop=True)
        sub_lons = region_data.Lon.values
        sub_lats = region_data.Lat.values
        sub_dem = region_data.values
        
        X, Y = np.meshgrid(sub_lons, sub_lats)
        Z = sub_dem
        
        # 创建多个视角的3D图
        fig = plt.figure(figsize=(20, 15))
        
        # 视角1: 俯视
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.view_init(elev=60, azim=0)
        surf1 = ax1.plot_surface(X[::2, ::2], Y[::2, ::2], Z[::2, ::2] * 0.002, 
                                cmap='terrain', alpha=0.8)
        ax1.set_title('俯视视角', fontsize=12)
        
        # 视角2: 侧面
        ax2 = fig.add_subplot(222, projection='3d')
        ax2.view_init(elev=20, azim=45)
        surf2 = ax2.plot_surface(X[::2, ::2], Y[::2, ::2], Z[::2, ::2] * 0.002, 
                                cmap='terrain', alpha=0.8)
        ax2.set_title('侧面视角', fontsize=12)
        
        # 视角3: 正面
        ax3 = fig.add_subplot(223, projection='3d')
        ax3.view_init(elev=10, azim=90)
        surf3 = ax3.plot_surface(X[::2, ::2], Y[::2, ::2], Z[::2, ::2] * 0.002, 
                                cmap='terrain', alpha=0.8)
        ax3.set_title('正面视角', fontsize=12)
        
        # 视角4: 等距
        ax4 = fig.add_subplot(224, projection='3d')
        ax4.view_init(elev=30, azim=135)
        surf4 = ax4.plot_surface(X[::2, ::2], Y[::2, ::2], Z[::2, ::2] * 0.002, 
                                cmap='terrain', alpha=0.8)
        ax4.set_title('等距视角', fontsize=12)
        
        fig.suptitle(title, fontsize=16)
        
        ds.close()
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in advanced 3D plot: {e}")

def plot_dem_comparison(extent, title="地形图对比"):
    """同时显示2D和3D地形图"""
    try:
        ds = xr.open_dataset(DEM_PATH)
        dem_data = ds['dhgt_gfs']
        
        # 提取数据
        mask = (ds.Lon >= extent[0]) & (ds.Lon <= extent[1]) & \
               (ds.Lat >= extent[2]) & (ds.Lat <= extent[3])
        
        region_data = dem_data.where(mask, drop=True)
        sub_lons = region_data.Lon.values
        sub_lats = region_data.Lat.values
        sub_dem = region_data.values
        
        X, Y = np.meshgrid(sub_lons, sub_lats)
        Z = sub_dem
        
        # 创建对比图
        fig = plt.figure(figsize=(20, 10))
        
        # 2D图
        ax1 = fig.add_subplot(121, projection=ccrs.PlateCarree())
        ax1.set_extent(extent, crs=ccrs.PlateCarree())
        contourf = ax1.contourf(sub_lons, sub_lats, sub_dem, levels=50, cmap='terrain', alpha=0.8)
        ax1.set_title('2D地形图', fontsize=14)
        plt.colorbar(contourf, ax=ax1, shrink=0.8)
        
        # 3D图
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.view_init(elev=35, azim=45)
        surf = ax2.plot_surface(X[::3, ::3], Y[::3, ::3], Z[::3, ::3] * 0.002, 
                               cmap='terrain', alpha=0.9)
        ax2.set_title('3D地形图', fontsize=14)
        plt.colorbar(surf, ax=ax2, shrink=0.8)
        
        fig.suptitle(title, fontsize=16)
        
        ds.close()
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in comparison plot: {e}")

def get_dem_statistics(extent):
    """获取指定范围内的地形统计信息"""
    try:
        ds = xr.open_dataset(DEM_PATH)
        dem_data = ds['dhgt_gfs']
        
        mask = (ds.Lon >= extent[0]) & (ds.Lon <= extent[1]) & \
               (ds.Lat >= extent[2]) & (ds.Lat <= extent[3])
        
        region_data = dem_data.where(mask, drop=True)
        
        print(f"区域地形统计 ({extent[0]}~{extent[1]}E, {extent[2]}~{extent[3]}N):")
        print(f"  最小值: {float(region_data.min().values):.2f} m")
        print(f"  最大值: {float(region_data.max().values):.2f} m")
        print(f"  平均值: {float(region_data.mean().values):.2f} m")
        print(f"  标准差: {float(region_data.std().values):.2f} m")
        print(f"  数据点数: {np.prod(region_data.shape)}")
        
        ds.close()
        
    except Exception as e:
        print(f"Error getting statistics: {e}")

if __name__ == "__main__":
    # 获取全局经纬度范围
    ds = xr.open_dataset(DEM_PATH)
    lons = ds['Lon'].values
    lats = ds['Lat'].values
    ds.close()
    
    print(f"DEM数据覆盖范围: 经度 {lons.min():.2f}~{lons.max():.2f}E, 纬度 {lats.min():.2f}~{lats.max():.2f}N")
    
    # 显示浙江省地形统计
    get_dem_statistics(ZHEJIANG_EXTENT)
    
    # 选择要绘制的图形类型
    print("\n选择绘图类型:")
    print("1. 2D地形图")
    print("2. 3D地形图")
    print("3. 高级3D地形图（多视角）")
    print("4. 2D+3D对比图")
    
    choice = input("请输入选择 (1-4, 默认1): ") or "1"
    
    if choice == "1":
        plot_dem_2d(ZHEJIANG_EXTENT, "浙江省2D地形图")
    elif choice == "2":
        # 可以调整视角参数
        plot_dem_3d(ZHEJIANG_EXTENT, "浙江省3D地形图", elevation_angle=35, azimuth_angle=45)
    elif choice == "3":
        plot_dem_3d_advanced(ZHEJIANG_EXTENT, "浙江省多视角3D地形图")
    elif choice == "4":
        plot_dem_comparison(ZHEJIANG_EXTENT, "浙江省地形图对比")
    else:
        plot_dem_2d(ZHEJIANG_EXTENT, "浙江省2D地形图")