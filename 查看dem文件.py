import xarray as xr
import numpy as np

# 文件路径
DEM_PATH = r"/Users/momo/Desktop/业务相关/2025 影响台风大风/地形文件/DEM_0P05_CHINA.nc"

try:
    # 打开NetCDF文件
    ds = xr.open_dataset(DEM_PATH)
    
    print("=" * 50)
    print("DEM文件基本信息:")
    print("=" * 50)
    
    # 1. 显示文件整体信息
    print("\n1. 数据集信息:")
    print(ds)
    
    # 2. 显示所有变量
    print("\n2. 数据集中的变量:")
    for var_name in ds.data_vars:
        print(f"  - {var_name}: {ds[var_name].dtype} {ds[var_name].shape}")
    
    # 3. 显示坐标变量
    print("\n3. 坐标变量:")
    for coord_name in ds.coords:
        coord = ds.coords[coord_name]
        print(f"  - {coord_name}: {coord.dtype} {coord.shape}")
        if len(coord) > 0:
            print(f"    范围: {float(coord.min().values):.2f} 到 {float(coord.max().values):.2f}")
    
    # 4. 显示全局属性
    print("\n4. 全局属性:")
    for attr_name in ds.attrs:
        print(f"  - {attr_name}: {ds.attrs[attr_name]}")
    
    # 5. 显示主要DEM数据变量的详细信息
    dem_vars = [var for var in ds.data_vars if var.lower() in ['dem', 'elevation', 'height', 'altitude']]
    if not dem_vars:
        dem_vars = list(ds.data_vars)  # 如果没有明确命名的DEM变量，显示所有数据变量
    
    for dem_var in dem_vars:
        print(f"\n5. DEM变量 '{dem_var}' 的详细信息:")
        var_data = ds[dem_var]
        print(f"   - 数据类型: {var_data.dtype}")
        print(f"   - 维度: {var_data.dims}")
        print(f"   - 形状: {var_data.shape}")
        print(f"   - 单位: {var_data.attrs.get('units', '未知')}")
        print(f"   - 描述: {var_data.attrs.get('long_name', '未知')}")
        
        # 显示数据统计信息（如果数据不是太大）
        if var_data.size < 1000000:  # 避免处理过大的数据
            print(f"   - 最小值: {float(var_data.min().values):.2f}")
            print(f"   - 最大值: {float(var_data.max().values):.2f}")
            print(f"   - 平均值: {float(var_data.mean().values):.2f}")
        
        # 显示变量属性
        if var_data.attrs:
            print(f"   - 变量属性:")
            for attr_name, attr_value in var_data.attrs.items():
                print(f"     * {attr_name}: {attr_value}")
    
    # 6. 显示文件大小信息
    print(f"\n6. 文件大小: {ds.nbytes / (1024 * 1024):.2f} MB")
    
    # 关闭文件
    ds.close()
    
except FileNotFoundError:
    print(f"错误: 找不到文件 {DEM_PATH}")
except Exception as e:
    print(f"读取文件时出错: {e}")