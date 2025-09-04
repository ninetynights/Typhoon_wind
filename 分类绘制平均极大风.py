#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按簇计算“过程极大风”的簇均值（风速算术均值 + 风向矢量均值），并绘图。
- 输入：
    1) All_Typhoons_ExMaxWind.nc  （含 wind_velocity, wind_direction, typhoon_id_index, STID/lat/lon/height 及属性映射）
    2) 影响段_簇映射.csv         （含 tid 与 cluster_id）
- 输出：
    输出目录下为每个簇生成：
      a) Cluster_<cid>_MeanMaxWind.png   风羽图
      b) Cluster_<cid>_MeanMaxWind.csv   明细（每站均值与有效样本数）

运行指令示例（请根据实际路径调整）：
(wind_zj) momo@Momo-2 代码 % python 分类绘制平均极大风.py \  
  --nc "/Users/momo/Desktop/业务相关/2025 影响台风大风/All_Typhoons_ExMaxWind.nc" \
  --cluster_csv "/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_影响段聚类(修改轨迹点和弧长)/影响段_簇映射.csv" \
  --out "/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_簇平均风场"
"""

import os
import argparse
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import chartostring

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ------------------ 工具函数 ------------------

def parse_mapping(attr_str):
    """解析全局属性中的 'k1:v1; k2:v2; ...' 映射为 dict（值若可转 int 就转 int）"""
    m = {}
    for seg in str(attr_str).strip().split(";"):
        if ":" in seg:
            k, v = seg.split(":", 1)
            k = k.strip()
            v = v.strip()
            try:
                v = int(v)
            except Exception:
                pass
            m[k] = v
    return m



def to_str_array(STID_raw):
    a = np.array(STID_raw)

    # 情况 A：本来就是一维字符串/字节串
    if a.ndim == 1 and a.dtype.kind in {'S', 'U', 'O'}:
        return a.astype(str)

    # 情况 B：二维字符数组（常见为 S1，形如 (n, m)）
    if a.ndim == 2 and a.dtype.kind == 'S':
        # 例如 b'5', b'8', b'4', b'5', b'7' → '58457'
        return chartostring(a).astype(str)

    # 情况 C：二维数值矩阵（每格是 ASCII 码）
    if a.ndim == 2 and a.dtype.kind in {'i', 'u'}:
        return np.array([''.join(map(chr, row)).strip() for row in a])

    # 兜底：都转成 str
    return a.astype(str)


def squeeze_time_station(arr):
    """
    将变量 squeeze 成 (time, station) 形状。
    允许原始为 (time, 1, station) 或 (time, station)；若维度不匹配则抛错。
    """
    a = np.array(arr)
    a = np.squeeze(a)
    if a.ndim != 2:
        raise ValueError(f"变量维度期望为2，但得到 {a.ndim} 维，shape={a.shape}")
    # 约定第0维为time，第1维为station
    return a

def barb_uv_from_wswd(ws, wd_deg):
    """
    气象风向（来自方向，度）+ 风速 -> U/V（指向流向），用于 barbs。
    U = -ws * sin(theta), V = -ws * cos(theta)
    """
    theta = np.deg2rad(wd_deg)
    U = -ws * np.sin(theta)
    V = -ws * np.cos(theta)
    return U, V

def circular_mean_dir(deg_array):
    """
    对风向做矢量（圆周）平均，输入为度的一维数组（已过滤 NaN）。
    返回：平均风向（度，0-360）。
    """
    th = np.deg2rad(deg_array)
    # 气象风向的“来自方向”，做平均时仍按方向角做矢量平均（取来向向量）
    u = -np.sin(th)
    v = -np.cos(th)
    u_m = u.mean()
    v_m = v.mean()
    # 还原为气象风向（来自方向）
    avg_theta = np.arctan2(-u_m, -v_m)
    avg_deg = (np.rad2deg(avg_theta) + 360) % 360
    return avg_deg

def draw_station_wind_map(lons, lats, U, V, ws_values, stids, heights,
                          title, save_path, extent=None, annotate=True):
    """
    复用“多台风多站点”的风羽绘图风格：风羽 + 站点三行标注（站号/风速/海拔）
    """
    fig = plt.figure(figsize=(10, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_title(title, fontsize=14)

    ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
    ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')

    if extent is None:
        # 根据站点范围自动设置，并四周留白 1°
        xmin, xmax = np.nanmin(lons), np.nanmax(lons)
        ymin, ymax = np.nanmin(lats), np.nanmax(lats)
        extent = [xmin - 1, xmax + 1, ymin - 1, ymax + 1]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # 仅绘制有风速的站点
    valid = ~np.isnan(ws_values) & ~np.isnan(U) & ~np.isnan(V)
    if valid.any():
        ax.barbs(lons[valid], lats[valid], U[valid], V[valid],
                 ws_values[valid],
                 length=6, linewidth=0.8, transform=ccrs.PlateCarree(),
                 barb_increments=dict(half=2, full=4, flag=20))  # 参考你的设置

    if annotate:
        for i in range(len(lons)):
            if np.isnan(ws_values[i]):
                continue
            txt = f"{stids[i]}\n{ws_values[i]:.1f} m/s\n{int(heights[i])} m"
            ax.text(lons[i], lats[i], txt, fontsize=7, ha='left', va='bottom',
                    transform=ccrs.PlateCarree())

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

# ------------------ 核心计算 ------------------

def compute_cluster_mean_max(nc_path, cluster_csv, output_dir,
                             extent=None, annotate=True,
                             require_dir_pair=True):
    """
    按簇计算“过程极大风”的簇平均（风速/风向），并绘图+输出CSV。

    参数:
    - nc_path: NetCDF 路径
    - cluster_csv: 影响段_簇映射.csv（含 tid, cluster_id）
    - output_dir: 输出目录
    - extent: [minlon, maxlon, minlat, maxlat]，不传则按站点外扩1°
    - annotate: 是否标注站号/值/海拔
    - require_dir_pair: True 则风速与风向配对统计（若最大风对应风向为 NaN，则该台风对该站不计入均值）
                       False 则风速与风向各自用各自有效样本做平均（可能出现分母不同）
    """
    # 读聚类映射
    cmap = pd.read_csv(cluster_csv)
    # 聚类簇列表
    clusters = sorted(cmap['cluster_id'].unique())

    # 打开NC
    nc = Dataset(nc_path)

    # 映射：中央台编号 -> 内部索引
    id_to_index = parse_mapping(nc.getncattr('id_to_index'))

    # 站点信息
    # STID 有可能是字符串数组或字符矩阵，统一成一维字符串数组
    STID_raw = nc.variables['STID'][:]
    stids = to_str_array(STID_raw)
    # 去掉可能的空白
    stids = np.char.strip(stids)

    lats = np.array(nc.variables['lat'][:], dtype=float)
    lons = np.array(nc.variables['lon'][:], dtype=float)
    heights = np.array(nc.variables['height'][:], dtype=float)

    # 主变量
    ws_all = squeeze_time_station(nc.variables['wind_velocity'][:])
    wd_all = squeeze_time_station(nc.variables['wind_direction'][:])
    tf_idx = squeeze_time_station(nc.variables['typhoon_id_index'][:])

    n_time, n_sta = ws_all.shape

    # 输出目录
    fig_dir = os.path.join(output_dir, "figs")
    csv_dir = os.path.join(output_dir, "csv")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # 逐簇处理
    for cid in clusters:
        sub = cmap[cmap['cluster_id'] == cid]
        # 该簇的台风编号列表（去重）
        tids = list(pd.unique(sub['tid'].astype(str)))

        # 容器：每站收集“台风→最大风”的样本
        # 为了做配对统计，收集成列表
        per_sta_speeds = [[] for _ in range(n_sta)]
        per_sta_dirs   = [[] for _ in range(n_sta)]

        # 遍历该簇台风
        for tid in tids:
            if tid not in id_to_index:
                # 数据对不上的情况直接跳过
                continue
            ty_idx = int(id_to_index[tid])

            # 找出该台风影响掩膜：shape = (time, station)
            mask = (tf_idx == ty_idx)

            # 逐站提取“过程极大风速及对应风向”
            for j in range(n_sta):
                m = mask[:, j]
                if not np.any(m):
                    continue
                ws_j = ws_all[m, j]
                if ws_j.size == 0 or np.all(np.isnan(ws_j)):
                    continue
                # 最大风速
                try:
                    k = int(np.nanargmax(ws_j))
                except ValueError:
                    # 全 NaN
                    continue
                max_ws = ws_j[k]
                if np.isnan(max_ws):
                    continue
                # 对应时刻的风向（从同一“掩膜后的子序列”取第 k 个）
                wd_j = wd_all[m, j]
                max_wd = wd_j[k] if k < wd_j.size else np.nan

                if require_dir_pair:
                    # 要求成对（风向缺失则都不计）
                    if np.isnan(max_wd):
                        continue
                    per_sta_speeds[j].append(float(max_ws))
                    per_sta_dirs[j].append(float(max_wd))
                else:
                    # 分别计数：速度先收
                    per_sta_speeds[j].append(float(max_ws))
                    # 风向若有才收
                    if not np.isnan(max_wd):
                        per_sta_dirs[j].append(float(max_wd))

        # 计算“簇→站点”的均值
        mean_ws = np.full(n_sta, np.nan, dtype=float)
        mean_wd = np.full(n_sta, np.nan, dtype=float)
        cnt_pair = np.zeros(n_sta, dtype=int)
        cnt_ws   = np.zeros(n_sta, dtype=int)
        cnt_wd   = np.zeros(n_sta, dtype=int)

        for j in range(n_sta):
            spd_list = np.array(per_sta_speeds[j], dtype=float)
            dir_list = np.array(per_sta_dirs[j], dtype=float)
            if require_dir_pair:
                # 成对统计：两者长度必然相等
                cnt_pair[j] = len(spd_list)
                if cnt_pair[j] > 0:
                    mean_ws[j] = np.nanmean(spd_list)
                    mean_wd[j] = circular_mean_dir(dir_list)
            else:
                # 分开统计
                if len(spd_list) > 0:
                    cnt_ws[j] = len(spd_list)
                    mean_ws[j] = np.nanmean(spd_list)
                if len(dir_list) > 0:
                    cnt_wd[j] = len(dir_list)
                    mean_wd[j] = circular_mean_dir(dir_list)

        # 生成 U/V（仅对有风速且有风向的站点）
        U = np.full(n_sta, np.nan, dtype=float)
        V = np.full(n_sta, np.nan, dtype=float)
        valid_for_vec = ~np.isnan(mean_ws) & ~np.isnan(mean_wd)
        if valid_for_vec.any():
            U[valid_for_vec], V[valid_for_vec] = barb_uv_from_wswd(
                mean_ws[valid_for_vec],
                mean_wd[valid_for_vec]
            )

        # 保存 CSV
        out_csv = os.path.join(csv_dir, f"Cluster_{cid}_MeanMaxWind.csv")
        if require_dir_pair:
            df = pd.DataFrame({
                "station": stids,
                "lon": lons,
                "lat": lats,
                "height": heights,
                "mean_max_ws": mean_ws,
                "mean_max_wd": mean_wd,
                "count_pairs": cnt_pair,   # 参与均值的“台风-站”样本数
            })
        else:
            df = pd.DataFrame({
                "station": stids,
                "lon": lons,
                "lat": lats,
                "height": heights,
                "mean_max_ws": mean_ws,
                "mean_max_wd": mean_wd,
                "count_ws": cnt_ws,
                "count_wd": cnt_wd,
            })
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")

        # 绘图
        title = f"Cluster {cid} | 过程极大风 簇均值（风速+风向）\n样本台风数={len(tids)} | 站点有效数={int(np.sum(valid_for_vec))}"
        out_fig = os.path.join(fig_dir, f"Cluster_{cid}_MeanMaxWind.png")
        draw_station_wind_map(
            lons=lons, lats=lats, U=U, V=V, ws_values=mean_ws,
            stids=stids, heights=heights,
            title=title, save_path=out_fig,
            extent=extent, annotate=annotate
        )

    nc.close()


# ------------------ CLI ------------------

def main():
    parser = argparse.ArgumentParser(description="按簇绘制过程极大风簇均值风场图")
    parser.add_argument("--nc", type=str, default="All_Typhoons_ExMaxWind.nc",
                        help="NetCDF 文件路径（含 wind_velocity / wind_direction / typhoon_id_index 等）")
    parser.add_argument("--cluster_csv", type=str, default="影响段_簇映射.csv",
                        help="聚类结果 CSV（含 tid, cluster_id）")
    parser.add_argument("--out", type=str, default="输出_簇平均风场",
                        help="输出目录")
    parser.add_argument("--extent", type=float, nargs=4, default=None,
                        help="绘图范围 [minlon maxlon minlat maxlat]，不传则按站点范围自动+1°留白")
    parser.add_argument("--no-annotate", action="store_true",
                        help="不在图上标注站号/风速/海拔")
    parser.add_argument("--loose-pair", action="store_true",
                        help="风速与风向分别按各自有效样本做平均（默认成对统计）")

    args = parser.parse_args()

    compute_cluster_mean_max(
        nc_path=args.nc,
        cluster_csv=args.cluster_csv,
        output_dir=args.out,
        extent=args.extent,
        annotate=not args.no_annotate,
        require_dir_pair=(not args.loose_pair)
    )

if __name__ == "__main__":
    main()