import xarray as xr
import pandas as pd

# 指定NetCDF文件路径
file_path = "/Users/momo/Desktop/业务相关/2025 影响台风大风/All_Typhoons_ExMaxWind.nc"

# 打开NetCDF文件
ds = xr.open_dataset(file_path)

# 打印文件基本信息
print("=" * 80)
print(f"NetCDF文件基本信息: {file_path}")
print("=" * 80)
print(f"文件维度: {list(ds.dims)}")
print(f"文件变量: {list(ds.variables)}")
print(f"全局属性: {ds.attrs}")

# 打印时间维度信息
print("\n" + "=" * 80)
print("时间维度信息 (INITTIME):")
print("=" * 80)
if 'INITTIME' in ds:
    times = ds['INITTIME'].values
    print(f"时间点数量: {len(times)}")
    print(f"起始时间: {pd.to_datetime(times[0])}")
    print(f"结束时间: {pd.to_datetime(times[-1])}")
    print(f"时间间隔: {(times[1] - times[0]).astype('timedelta64[h]')}小时")

# 打印站点信息
print("\n" + "=" * 80)
print("站点信息 (STID):")
print("=" * 80)
if 'STID' in ds:
    stations = ds['STID'].values
    print(f"站点数量: {len(stations)}")
    print(f"前10个站点ID: {stations[:10]}")

    # 打印站点坐标信息
    if 'lat' in ds:
        lats = ds['lat'].values
        lons = ds['lon'].values
        print(f"\n前5个站点的经纬度:")
        for i in range(min(5, len(stations))):
            print(f"站点 {stations[i]}: 纬度={lats[i]:.4f}, 经度={lons[i]:.4f}")

# 打印变量信息
print("\n" + "=" * 80)
print("变量详细信息:")
print("=" * 80)
for var_name in ds.variables:
    var = ds[var_name]
    print(f"\n变量名: {var_name}")
    print(f"  维度: {var.dims}")
    print(f"  形状: {var.shape}")
    print(f"  数据类型: {var.dtype}")
    print(f"  属性: {var.attrs}")

    # 打印部分数据样本
    if var.ndim > 0 and var.size > 0:
        sample_size = min(3, len(var))
        if var.ndim == 1:  # 一维变量
            print(f"  数据样本: {var.values[:sample_size]}")
        elif var.ndim == 3:  # 三维变量
            print("  数据样本 (时间, TL, 站点):")
            # 打印第一个时间点、第一个TL、前几个站点的数据
            print(f"    - 时间点0, TL0: {var[0, 0, :sample_size].values}")

# 打印台风信息映射
print("\n" + "=" * 80)
print("台风信息映射关系:")
print("=" * 80)
for key in ['id_to_index', 'index_to_id', 'cn_to_index', 'index_to_cn', 'en_to_index', 'index_to_en']:
    if key in ds.attrs:
        mapping_str = ds.attrs[key]
        # 解析映射关系
        mapping = {}
        for pair in mapping_str.split(';'):
            if ':' in pair:
                k, v = pair.split(':', 1)
                mapping[k] = v
        print(f"\n{key}映射关系:")
        for k, v in list(mapping.items())[:5]:  # 只打印前5个
            print(f"  {k} -> {v}")
        print(f"  总共 {len(mapping)} 个映射")

# 关闭数据集
ds.close()