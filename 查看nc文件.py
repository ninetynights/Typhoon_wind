import xarray as xr
import pandas as pd
import os  # <--- 新增: 用于处理文件路径
import sys  # <--- 新增: 用于重定向标准输出

# --- 1. 定义文件路径 ---

# 指定NetCDF文件路径
# 乡镇站点
file_path = "/Users/momo/Desktop/业务相关/2025 影响台风大风/combine_stations_ExMaxWind.nc"
# 国家站站点
# file_path = "/Users/momo/Desktop/业务相关/2025 影响台风大风/All_Typhoons_ExMaxWind.nc"

# --- 2. 自动生成输出的 .txt 文件路径 ---
# os.path.splitext(file_path) 会将路径分割成 ('文件名', '.nc')
output_txt_path = os.path.splitext(file_path)[0] + ".txt"  # <--- 新增

# --- 3. 重定向标准输出 (stdout) ---
print(f"正在分析文件: {file_path}")  # <--- 新增: 这句会打印在控制台
print(f"分析结果将保存到: {output_txt_path}")  # <--- 新增: 这句也会打印在控制台

original_stdout = sys.stdout  # <--- 新增: 保存原始的标准输出 (控制台)

try:
    # <--- 新增: 使用 'with' 语句安全地打开文件 ---
    # 之后所有的 'print()' 都会被重定向到这个 'f' 文件中
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        sys.stdout = f  # <--- 新增: 将标准输出重定向到文件

        # === 以下是你的原始代码，无需任何修改 ===

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
            # (这里我还是帮你加了个小检查，防止只有一个时间点时脚本报错)
            if len(times) > 0:
                print(f"起始时间: {pd.to_datetime(times[0])}")
                print(f"结束时间: {pd.to_datetime(times[-1])}")
            if len(times) > 1:
                print(f"时间间隔: {(times[1] - times[0]).astype('timedelta64[h]')}小时")
            else:
                print("时间间隔: 只有一个时间点，无法计算间隔")


        # 打印站点信息
        print("\n" + "=" * 80)
        print("站点信息 (STID):")
        print("=" * 80)
        if 'STID' in ds:
            stations = ds['STID'].values
            print(f"站点数量: {len(stations)}")
            print(f"前10个站点ID: {stations[:10]}")

            # 打印站点坐标信息
            # (也加了个检查，防止 'lon' 或 'lat' 不存在时报错)
            if 'lat' in ds and 'lon' in ds:
                lats = ds['lat'].values
                lons = ds['lon'].values
                print(f"\n前5个站点的经纬度:")
                for i in range(min(5, len(stations))):
                    # 确保经纬度数组和站点数组一样长
                    if i < len(lats) and i < len(lons):
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
                sample_size = min(3, var.shape[-1] if var.ndim > 0 else 3) # 确保样本大小不超过最后一维
                if var.ndim == 1:  # 一维变量
                    print(f"  数据样本: {var.values[:sample_size]}")
                elif var.ndim == 3:  # 三维变量
                    print("  数据样本 (时间, TL, 站点):")
                    # 确保索引不越界
                    if var.shape[0] > 0 and var.shape[1] > 0:
                         print(f"    - 时间点0, TL0: {var[0, 0, :sample_size].values}")
                    else:
                        print("    - (维度为空，无法采样)")

        # 打印台风信息映射
        print("\n" + "=" * 80)
        print("台风信息映射关系:")
        print("=" * 80)
        for key in ['id_to_index', 'index_to_id', 'cn_to_index', 'index_to_cn', 'en_to_index', 'index_to_en']:
            if key in ds.attrs:
                mapping_str = ds.attrs[key]
                # 解析映射关系
                mapping = {}
                # (这里也加了个小检查，过滤掉空字符串)
                for pair in filter(None, mapping_str.split(';')):
                    if ':' in pair:
                        try:
                            k, v = pair.split(':', 1)
                            mapping[k.strip()] = v.strip()
                        except ValueError:
                            pass # 跳过格式错误的
                print(f"\n{key}映射关系:")
                for k, v in list(mapping.items())[:5]:  # 只打印前5个
                    print(f"  {k} -> {v}")
                print(f"  总共 {len(mapping)} 个映射")

        # 关闭数据集
        ds.close()

        # === 你的原始代码结束 ===

finally:
    # --- 4. 恢复标准输出 ---
    sys.stdout = original_stdout  # <--- 新增: 无论如何，最后都要恢复标准输出到控制台

print("--- 分析完成! ---")  # <--- 新增: 这句会打印在控制台