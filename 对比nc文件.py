import xarray as xr
import numpy as np
import pandas as pd
import sys

def compare_typhoon_nc_files(file_path_national, file_path_township):
    """
    对比分析两个台风 NetCDF 文件 (国家站 vs. 乡镇站)。
    """
    print("正在加载 NetCDF 文件...")
    print(f"  国家站文件: {file_path_national}")
    print(f"  乡镇站文件: {file_path_township}")

    # 使用 'with' 语句同时打开两个文件
    try:
        with xr.open_dataset(file_path_national) as ds_nat, \
             xr.open_dataset(file_path_township) as ds_town:
            
            print("文件加载完毕。\n")

            # --- 1. 站点 (STID) 集合分析 ---
            print("=" * 80)
            print("1. 站点 (STID) 集合分析")
            print("=" * 80)
            
            # 提取站点ID并转换为 set 集合，便于比较
            try:
                stids_nat = set(np.char.strip(ds_nat['STID'].values.astype(str)))
                stids_town = set(np.char.strip(ds_town['STID'].values.astype(str)))
            except KeyError:
                print("[错误] 'STID' 变量在文件中未找到。")
                return

            print(f"国家站 (All_Typhoons) 文件中站点数量: {len(stids_nat)}")
            print(f"乡镇站 (New_Stations) 文件中站点数量: {len(stids_town)}")

            # 计算交集和差集
            common_stations = stids_nat.intersection(stids_town)
            national_only = stids_nat.difference(stids_town)
            township_only = stids_town.difference(stids_nat)

            print(f"\n共同(重叠)的站点数量: {len(common_stations)}")
            print(f"仅在 '国家站' 文件中存在的站点数量: {len(national_only)}")
            print(f"仅在 '乡镇站' 文件中存在的站点数量: {len(township_only)}")

            if not common_stations:
                print("\n两个文件没有共同的站点ID。对比结束。")
                return
            
            # 将 set 转换回 list 并排序，便于后续索引
            common_stations_list = sorted(list(common_stations))

            # --- 2. 重叠站点元数据 (Lat, Lon, Height) 对比 ---
            print("\n" + "=" * 80)
            print("2. 重叠站点元数据 (Lat, Lon, Height) 对比")
            print("=" * 80)
            
            mismatch_count = 0
            # 遍历所有重叠站点
            for stid in common_stations_list:
                # .sel() 用于按维度标签（这里是站点ID）选择数据
                meta_nat = ds_nat.sel(STID=stid)
                meta_town = ds_town.sel(STID=stid)
                
                # .item() 用于从0维数组中提取单个数值
                lat_nat, lon_nat, hgt_nat = meta_nat['lat'].item(), meta_nat['lon'].item(), meta_nat['height'].item()
                lat_town, lon_town, hgt_town = meta_town['lat'].item(), meta_town['lon'].item(), meta_town['height'].item()

                # 使用 np.isclose() 来安全地比较浮点数
                if not np.isclose(lat_nat, lat_town) or \
                   not np.isclose(lon_nat, lon_town) or \
                   not np.isclose(hgt_nat, hgt_town):
                    
                    mismatch_count += 1
                    if mismatch_count <= 100:  # 只打印前10个不匹配的
                        print(f"  [!!] 站点 {stid} 元数据不匹配:")
                        if not np.isclose(lat_nat, lat_town):
                            print(f"    - 纬度 (Lat): {lat_nat} (国家站) vs {lat_town} (乡镇站)")
                        if not np.isclose(lon_nat, lon_town):
                            print(f"    - 经度 (Lon): {lon_nat} (国家站) vs {lon_town} (乡镇站)")
                        if not np.isclose(hgt_nat, hgt_town):
                            print(f"    - 高度 (Height): {hgt_nat} (国家站) vs {hgt_town} (乡镇站)")
            
            if mismatch_count == 0:
                print("  [✓] 所有重叠站点的元数据 (Lat, Lon, Height) 完全一致。")
            else:
                if mismatch_count > 10:
                    print(f"  ... (已省略其余 {mismatch_count - 10} 个不匹配的站点)")
                print(f"\n  [!] 总结: 总共有 {mismatch_count} 个重叠站点的元数据不匹配。")

            # --- 3. 重叠站点数据值 (wind_velocity) 抽样对比 ---
            print("\n" + "=" * 80)
            print("3. 重叠站点数据值 (wind_velocity) 抽样对比")
            print("=" * 80)
            
            # 随机挑选一个重叠站点进行时间序列对比
            sample_stid = common_stations_list[0]
            print(f"抽样站点: {sample_stid}\n")
            
            # 提取该站点的时间序列数据
            # .squeeze() 用于移除大小为 1 的维度 (如 'TL')
            ts_nat = ds_nat['wind_velocity'].sel(STID=sample_stid).squeeze()
            ts_town = ds_town['wind_velocity'].sel(STID=sample_stid).squeeze()
            
            # 检查时间轴是否一致 (根据 .txt 文件，它们应该是一致的)
            if not np.array_equal(ts_nat['INITTIME'].values, ts_town['INITTIME'].values):
                print("[警告] 两个文件的时间轴 (INITTIME) 不一致，无法直接对比。")
            else:
                # 计算两个时间序列的绝对差值
                diff = np.abs(ts_nat - ts_town)
                
                # 仅在两个数据都有效(非NaN)的地方计算
                valid_diff = diff.where(np.isfinite(ts_nat) & np.isfinite(ts_town))
                
                max_diff = valid_diff.max().item()
                mean_diff = valid_diff.mean().item()
                
                print(f"对比变量: wind_velocity (风速)")
                print(f"  时间点数量: {len(ts_nat['INITTIME'])}")
                
                if np.isnan(max_diff):
                    print("  [!] 无法计算差异 (可能所有数据均为NaN或一侧为NaN)。")
                elif max_diff == 0:
                    print("  [✓] 数据完全一致: 两个文件在该站点的风速数据相同。")
                else:
                    print(f"  [!!] 数据不一致:")
                    print(f"    - 最大绝对差异: {max_diff:.4f} m/s")
                    print(f"    - 平均绝对差异: {mean_diff:.4f} m/s")
                    # 找出最大差异发生的时间
                    max_diff_time = valid_diff.idxmax().values
                    print(f"    - 最大差异发生在时间点: {pd.to_datetime(str(max_diff_time))}")

            # --- 4. 站点名称 (station_name) 编码问题检查 ---
            print("\n" + "=" * 80)
            print("4. 站点名称 (station_name) 编码问题检查")
            print("=" * 80)

            # 检查 '58443' 站 (根据 .txt 文件的线索)
            check_stid = '58443'
            if check_stid in common_stations:
                name_nat = ds_nat['station_name'].sel(STID=check_stid).item()
                name_town = ds_town['station_name'].sel(STID=check_stid).item()
                
                print(f"抽样对比 (站点 {check_stid}):")
                print(f"  '国家站' 文件 (All_Typhoons) 名称: '{name_nat}'")
                print(f"  '乡镇站' 文件 (New_Stations) 名称: '{name_town}'")
                
                # '长兴' vs '³¤ÐË'
                if name_nat == '长兴' and '³' in name_town:
                    print("\n  [!] 确认: '乡镇站' 文件的站点名称存在编码问题 (乱码)。")
                elif name_nat != name_town:
                    print("\n  [!] 发现: 两个文件的站点名称不一致。")
                else:
                    print("\n  [✓] 该站点的名称一致。")
            else:
                print(f"  [信息] 站点 '58443' 不是重叠站点，跳过特定编码对比。")


    except FileNotFoundError as e:
        print(f"[错误] 文件未找到: {e.filename}")
    except Exception as e:
        print(f"分析过程中发生意外错误: {e}")

# --- 脚本主程序入口 ---
if __name__ == "__main__":
    # *** 请根据你的实际路径修改这里 ***
    # 国家站站点
    path_national = "/Users/momo/Desktop/业务相关/2025 影响台风大风/All_Typhoons_ExMaxWind.nc"
    # 乡镇站点
    path_township = "/Users/momo/Desktop/业务相关/2025 影响台风大风/New_Stations_Typhoons_ExMaxWind.nc"
    
    compare_typhoon_nc_files(path_national, path_township)