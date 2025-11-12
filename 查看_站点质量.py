import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import os
import sys

def plot_station_quality_map_categorized(nc_file, csv_quality_report, output_png, shp_paths):
    """
    绘制站点数据质量分布图。
    站点将按4个缺测等级（0%, 0-20%, 20-50%, >50%）分色显示。
    """
    print("开始绘制分档的站点数据质量分布图...")
    
    # --- 1. 加载数据 ---
    print(f"  正在读取 NC 文件: {nc_file}")
    try:
        with xr.open_dataset(nc_file) as ds:
            # 提取所有站点的经纬度信息
            df_meta = ds[['lat', 'lon']].to_dataframe()
            # 确保索引是字符串
            df_meta.index = df_meta.index.astype(str)
    except FileNotFoundError:
        print(f"  [!!] 错误: 找不到 NetCDF 文件: {nc_file}")
        return
    
    print(f"  正在读取 CSV 质量报告: {csv_quality_report}")
    try:
        # 强制将 STID 读作字符串，以匹配 NC 文件的索引
        df_quality = pd.read_csv(csv_quality_report, dtype={'STID': str})
    except FileNotFoundError:
        print(f"  [!!] 错误: 找不到 CSV 文件: {csv_quality_report}")
        return
        
    # --- 2. 合并元数据和质量数据 ---
    try:
        # 将经纬度和缺测百分比合并到一个 DataFrame 中
        df_all = pd.merge(df_meta, df_quality, left_index=True, right_on='STID', how='inner')
    except Exception as e:
        print(f"  [!!] 错误: 合并数据时出错: {e}")
        return

    # --- 3. 按你的4档规则分离站点 ---
    df_green = df_all[df_all['missing_percent'] == 0.0]
    df_yellow = df_all[(df_all['missing_percent'] > 0.0) & 
                       (df_all['missing_percent'] <= 20.0)]
    df_orange = df_all[(df_all['missing_percent'] > 20.0) & 
                       (df_all['missing_percent'] <= 50.0)]
    df_red = df_all[df_all['missing_percent'] > 50.0]

    print(f"  站点质量分类:")
    print(f"    - [绿色] 无缺测 (0%): {len(df_green)} 个")
    print(f"    - [黄色] 缺测 0%-20%: {len(df_yellow)} 个")
    print(f"    - [橙色] 缺测 20%-50%: {len(df_orange)} 个")
    print(f"    - [红色] 缺测 > 50%: {len(df_red)} 个")
    print(f"  总计: {len(df_green) + len(df_yellow) + len(df_orange) + len(df_red)} / {len(df_all)} 个站点")


    # --- 4. 开始绘图 (参考你的样式) ---
    plt.rcParams['font.sans-serif'] = ['Heiti TC']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    fig = plt.figure(figsize=(10, 10))
    proj = ccrs.PlateCarree()  # 投影
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    # 设置地图范围 (来自你的参考代码)
    lon_min, lon_max = 118, 123
    lat_min, lat_max = 27, 31.5
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)

    # --- 5. 添加地理底图 ---
    print("  正在加载地理底图...")
    try:
        # 优先尝试加载你指定的 SHP 文件
        zj_shp = shp_paths['zhejiang_province']
        city_shp = shp_paths['zhejiang_city']
        
        if not os.path.exists(zj_shp) or not os.path.exists(city_shp):
            raise FileNotFoundError("SHP 文件路径无效")

        zj_reader = shpreader.Reader(zj_shp)
        ax.add_geometries(zj_reader.geometries(), crs=proj, edgecolor='black', facecolor='None', lw=1.5)
        
        city_reader = shpreader.Reader(city_shp)
        ax.add_geometries(city_reader.geometries(), crs=proj, edgecolor='gray', facecolor='None', lw=0.5)
        
        print(f"  [✓] 成功加载本地 SHP 文件。")

    except Exception as e:
        # 备用方案
        print(f"  [!] 警告: 加载本地 SHP 文件失败 ({e})。")
        print("         将使用 Cartopy 默认的省界和海岸线作为备用底图。")
        
        provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces',
            scale='10m',
            facecolor='none',
            edgecolor='black'
        )
        ax.add_feature(provinces, lw=0.8)
        ax.add_feature(cfeature.COASTLINE, lw=1.0)
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS)

    # 绘制网格线 (来自你的参考代码)
    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}

    # --- 6. 绘制四类站点 ---
    # 为了让高亮站点显示在最上层，我们按 Green -> Yellow -> Orange -> Red 的顺序绘制
    print("  正在绘制站点...")
    
    # [蓝色] 无缺测 (0%)
    ax.scatter(df_green['lon'], df_green['lat'],
               s=12, 
               color='blue',
               label=f'无缺测 (0%) ({len(df_green)})',
               alpha=0.6, 
               transform=proj)

    # [绿色] 缺测 0-20%
    ax.scatter(df_yellow['lon'], df_yellow['lat'],
               s=12, 
               color='green',
               linewidth=0.5,
               label=f'缺测 0-20% ({len(df_yellow)})',
               alpha=0.7, 
               transform=proj)

    # [橙色] 缺测 20-50%
    ax.scatter(df_orange['lon'], df_orange['lat'],
               s=12, 
               color='orange',
               linewidth=0.5,
               label=f'缺测 20-50% ({len(df_orange)})',
               alpha=0.9, 
               transform=proj)

    # [红色] 缺测 > 50%
    ax.scatter(df_red['lon'], df_red['lat'],
               s=15, # 让红点最大，最显眼
               color='red',
               linewidth=0.5,
               label=f'缺测 > 50% ({len(df_red)})',
               alpha=1.0,
               transform=proj)

    # --- 7. 添加图例和标题并保存 ---
    ax.legend(loc='upper right', markerscale=2, fontsize=10) # 缩小一点字号以放下4个条目
    ax.set_title(f'站点数据质量分布图 (共 {len(df_all)} 站)', fontsize=16)

    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"\n✅ 成功! 站点质量分档图已保存到:\n   {output_png}")


# --- 脚本主程序入口 ---
if __name__ == "__main__":
    
    # --- 1. 文件路径设置 ---
    base_dir = "/Users/momo/Desktop/业务相关/2025 影响台风大风/数据/"
    
    # 输入文件
    nc_file = os.path.join(base_dir, "Combine_Stations_ExMaxWind.nc")
    csv_quality_report = os.path.join(base_dir, "4_合并后缺测报告_按站点.csv")
    
    # 输出文件
    output_png = os.path.join(base_dir, "station_quality_map.png")
    
    # SHP 路径 (来自你的参考文件)
    # **注意**: 'N:' 路径在 macOS 上很可能无效。
    # 请确保这些路径对你当前的系统是正确的，否则脚本将自动使用备用底图。
    shp_paths = {
        'zhejiang_province': 'N:/00_GIS/shp/zhejiang/zhejiang_province/Zhejiang_province.shp',
        'zhejiang_city': 'N:/00_GIS/shp/zhejiang/zhejiang_city/Zhejiang_city.shp'
    }

    # --- 2. 运行绘图函数 ---
    plot_station_quality_map_categorized(nc_file, csv_quality_report, output_png, shp_paths)