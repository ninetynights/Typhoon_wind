#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按簇绘制两类统计风场：
  1) meanmax   : "过程极大风" 的簇均值（你之前已经实现的逻辑：每个台风-站点取过程内最大风，然后对台风取平均；风向用矢量平均）
  2) meantime  : "过程平均风" 的簇均值（每个台风-站点在影响期内对时间做平均；风向用时间矢量平均；再在台风层面做均值/或按样本小时加权）

输入：
  - NetCDF: All_Typhoons_ExMaxWind.nc （含 wind_velocity, wind_direction, typhoon_id_index, STID/lat/lon/height, 以及全局属性 id_to_index 等）
  - CSV    : 影响段_簇映射.csv （含 tid, cluster_id）

输出：
  - 每个簇一张风羽图：输出目录/figs/Cluster_<cid>_<stat>.png
  - 每个簇一个 CSV：输出目录/csv/Cluster_<cid>_<stat>.csv

用法示例：
  # 过程极大风（和你之前一致）
  python 分类绘制风场特征.py \
    --stat meanmax \
    --nc "/Users/momo/Desktop/业务相关/2025 影响台风大风/All_Typhoons_ExMaxWind.nc" \
    --cluster_csv "/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_影响段聚类(修改轨迹点和弧长)/影响段_簇映射.csv" \
    --out "/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_簇平均风场"

  # 过程平均风（新需求），并按小时数加权（可选）
  python 分类绘制风场特征.py \
    --stat meantime --weight-by-hours \
    --nc "/Users/momo/Desktop/业务相关/2025 影响台风大风/All_Typhoons_ExMaxWind.nc" \
    --cluster_csv "/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_影响段聚类(修改轨迹点和弧长)/影响段_簇映射.csv" \
    --out "/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_簇平均风场"
"""

import os
import argparse
import numpy as np
import pandas as pd
from netCDF4 import Dataset
try:
    from netCDF4 import chartostring
except Exception:  # chartostring 可能不存在（旧版 netCDF4）
    chartostring = None

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ------------------ 工具函数 ------------------

def parse_mapping(attr_str):
    """解析 'k1:v1; k2:v2; ...' 为 dict（值若可转 int 则转 int）"""
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
    # 一维字符串/字节串
    if a.ndim == 1 and a.dtype.kind in {"S", "U", "O"}:
        return a.astype(str)
    # 二维字符数组（S1），如 (n, m)
    if a.ndim == 2 and a.dtype.kind == "S":
        if chartostring is not None:
            return chartostring(a).astype(str)
        # 退化处理：逐行拼接
        return np.array([b"".join(row).decode("utf-8", errors="ignore").strip() for row in a])
    # 二维数值矩阵：ASCII 码
    if a.ndim == 2 and a.dtype.kind in {"i", "u"}:
        return np.array(["".join(map(chr, row)).strip() for row in a])
    return a.astype(str)


def squeeze_time_station(arr):
    a = np.array(arr)
    a = np.squeeze(a)
    if a.ndim != 2:
        raise ValueError(f"变量维度期望为2，但得到 {a.ndim} 维，shape={a.shape}")
    return a  # (time, station)


def barb_uv_from_wswd(ws, wd_deg):
    theta = np.deg2rad(wd_deg)
    U = -ws * np.sin(theta)
    V = -ws * np.cos(theta)
    return U, V


def circular_mean_dir(deg_array):
    """无权重的角度平均（气象风向：来自方向）"""
    th = np.deg2rad(deg_array)
    u = -np.sin(th)
    v = -np.cos(th)
    u_m = u.mean()
    v_m = v.mean()
    ang = np.arctan2(-u_m, -v_m)
    return (np.rad2deg(ang) + 360) % 360


def weighted_circular_mean_dir(deg_array, weights):
    w = np.asarray(weights, dtype=float)
    d = np.asarray(deg_array, dtype=float)
    mask = (~np.isnan(d)) & (w > 0)
    if not np.any(mask):
        return np.nan
    th = np.deg2rad(d[mask])
    wv = w[mask]
    u = -np.sin(th)
    v = -np.cos(th)
    u_w = np.sum(wv * u) / np.sum(wv)
    v_w = np.sum(wv * v) / np.sum(wv)
    ang = np.arctan2(-u_w, -v_w)
    return (np.rad2deg(ang) + 360) % 360


def draw_station_wind_map(lons, lats, U, V, ws_values, stids, heights,
                          title, save_path, extent=None, annotate=True):
    fig = plt.figure(figsize=(10, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_title(title, fontsize=14)

    ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
    ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')

    if extent is None:
        xmin, xmax = np.nanmin(lons), np.nanmax(lons)
        ymin, ymax = np.nanmin(lats), np.nanmax(lats)
        extent = [xmin - 1, xmax + 1, ymin - 1, ymax + 1]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    valid = ~np.isnan(ws_values) & ~np.isnan(U) & ~np.isnan(V)
    if valid.any():
        ax.barbs(lons[valid], lats[valid], U[valid], V[valid], ws_values[valid],
                 length=6, linewidth=0.8, transform=ccrs.PlateCarree(),
                 barb_increments=dict(half=2, full=4, flag=20))

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

def compute_cluster_mean(nc_path, cluster_csv, output_dir,
                         stat="meanmax", extent=None, annotate=True,
                         require_dir_pair=True, weight_by_hours=False):
    """
    stat: 'meanmax' 或 'meantime'
      - meanmax : 每台风-站点在影响掩膜内取最大风速及其对应风向，然后对台风做平均
      - meantime: 每台风-站点在影响掩膜内对时间平均（风速算术均值；风向时间矢量均值），然后对台风做平均
                  若 weight_by_hours=True，则按“有效小时数”加权（相当于直接合并所有小时做总体平均）
    require_dir_pair:
      - True  : 速度与方向按配对时刻统计（时刻上任一缺测则跳过该时刻；meanmax 时要求最大风对应风向非缺测）
      - False : 速度与方向各自按各自有效样本统计（分母可能不同）
    """
    print(f"读取聚类映射: {cluster_csv}")
    cmap = pd.read_csv(cluster_csv)
    clusters = sorted(cmap['cluster_id'].unique())

    print(f"打开 NC: {nc_path}")
    nc = Dataset(nc_path)

    id_to_index = parse_mapping(nc.getncattr('id_to_index'))

    STID_raw = nc.variables['STID'][:]
    stids = to_str_array(STID_raw)
    stids = np.char.strip(stids)

    lats = np.array(nc.variables['lat'][:], dtype=float)
    lons = np.array(nc.variables['lon'][:], dtype=float)
    heights = np.array(nc.variables['height'][:], dtype=float)

    ws_all = squeeze_time_station(nc.variables['wind_velocity'][:])
    wd_all = squeeze_time_station(nc.variables['wind_direction'][:])
    tf_idx = squeeze_time_station(nc.variables['typhoon_id_index'][:])

    n_time, n_sta = ws_all.shape

    fig_dir = os.path.join(output_dir, "figs")
    csv_dir = os.path.join(output_dir, "csv")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    for cid in clusters:
        sub = cmap[cmap['cluster_id'] == cid]
        tids = list(pd.unique(sub['tid'].astype(str)))
        print(f"处理簇 {cid}，台风数={len(tids)} …")

        # 收集器：每站点存放 台风级统计（用于再平均）
        per_sta_speed_vals = [[] for _ in range(n_sta)]      # 台风级“值”（meanmax: max_ws；meantime: mean_ws）
        per_sta_speed_wts  = [[] for _ in range(n_sta)]      # 台风级“权重”（meantime 才用到：有效小时数）
        per_sta_dir_vals   = [[] for _ in range(n_sta)]      # 台风级“方向值”（meanmax: 对应风向；meantime: 时间平均风向）
        per_sta_dir_wts    = [[] for _ in range(n_sta)]      # 台风级“方向权重”（meantime 才用：有效小时数）

        for tid in tids:
            if tid not in id_to_index:
                continue
            ty_idx = int(id_to_index[tid])
            mask_ts = (tf_idx == ty_idx)  # (time, station)

            # 逐站处理
            for j in range(n_sta):
                m = mask_ts[:, j]
                if not np.any(m):
                    continue

                ws_seq = ws_all[m, j]
                wd_seq = wd_all[m, j]

                if stat == "meanmax":
                    # 过程极大风：找最大风速及其对应风向
                    if ws_seq.size == 0 or np.all(np.isnan(ws_seq)):
                        continue
                    try:
                        k = int(np.nanargmax(ws_seq))
                    except ValueError:
                        continue
                    max_ws = ws_seq[k]
                    if np.isnan(max_ws):
                        continue

                    if require_dir_pair:
                        # 最大风对应风向必须有效
                        if k >= wd_seq.size or np.isnan(wd_seq[k]):
                            continue
                        per_sta_speed_vals[j].append(float(max_ws))
                        per_sta_dir_vals[j].append(float(wd_seq[k]))
                    else:
                        per_sta_speed_vals[j].append(float(max_ws))
                        if k < wd_seq.size and not np.isnan(wd_seq[k]):
                            per_sta_dir_vals[j].append(float(wd_seq[k]))

                elif stat == "meantime":
                    # 过程平均风：对时间序列做平均
                    if require_dir_pair:
                        valid = (~np.isnan(ws_seq)) & (~np.isnan(wd_seq))
                        ws_valid = ws_seq[valid]
                        wd_valid = wd_seq[valid]
                    else:
                        ws_valid = ws_seq[~np.isnan(ws_seq)]
                        wd_valid = wd_seq[~np.isnan(wd_seq)]

                    # 风速时间均值
                    if ws_valid.size > 0:
                        per_sta_speed_vals[j].append(float(np.nanmean(ws_valid)))
                        per_sta_speed_wts[j].append(int(ws_valid.size) if weight_by_hours else 1)

                    # 风向时间矢量均值（来自方向）
                    if wd_valid.size > 0:
                        # 无权时间平均（台风级）
                        dir_mean_deg = circular_mean_dir(wd_valid)
                        per_sta_dir_vals[j].append(float(dir_mean_deg))
                        per_sta_dir_wts[j].append(int(wd_valid.size) if weight_by_hours else 1)
                else:
                    raise ValueError("stat must be 'meanmax' or 'meantime'")

        # 台风级 → 簇级（对台风做再平均）
        mean_ws = np.full(n_sta, np.nan, dtype=float)
        mean_wd = np.full(n_sta, np.nan, dtype=float)
        cnt_pairs = np.zeros(n_sta, dtype=int)  # meanmax/配对时可当作参与样本数
        cnt_ws_ty = np.zeros(n_sta, dtype=int)  # 参与风速均值的台风个数
        cnt_wd_ty = np.zeros(n_sta, dtype=int)  # 参与风向均值的台风个数

        for j in range(n_sta):
            spd_vals = np.array(per_sta_speed_vals[j], dtype=float)
            dir_vals = np.array(per_sta_dir_vals[j], dtype=float)

            # meanmax: 简单平均（或配对样本数量记录）
            if stat == "meanmax":
                if spd_vals.size > 0:
                    mean_ws[j] = np.nanmean(spd_vals)
                    cnt_ws_ty[j] = spd_vals.size
                if dir_vals.size > 0:
                    mean_wd[j] = circular_mean_dir(dir_vals)
                    cnt_wd_ty[j] = dir_vals.size
                cnt_pairs[j] = min(cnt_ws_ty[j], cnt_wd_ty[j])  # 近似记录

            # meantime: 台风级时间均值 → 簇级
            elif stat == "meantime":
                # 风速：按台风平均，或按有效小时加权
                if spd_vals.size > 0:
                    cnt_ws_ty[j] = spd_vals.size
                    if weight_by_hours:
                        w = np.array(per_sta_speed_wts[j], dtype=float)
                        w = w[~np.isnan(spd_vals)]
                        v = spd_vals[~np.isnan(spd_vals)]
                        if w.size > 0 and np.sum(w) > 0:
                            mean_ws[j] = np.sum(w * v) / np.sum(w)
                    else:
                        mean_ws[j] = np.nanmean(spd_vals)

                # 风向：对台风级“时间平均风向”做矢量平均；若加权则做加权矢量平均
                if dir_vals.size > 0:
                    cnt_wd_ty[j] = dir_vals.size
                    if weight_by_hours:
                        w = np.array(per_sta_dir_wts[j], dtype=float)
                        w = w[~np.isnan(dir_vals)]
                        v = dir_vals[~np.isnan(dir_vals)]
                        if w.size > 0 and np.sum(w) > 0:
                            mean_wd[j] = weighted_circular_mean_dir(v, w)
                    else:
                        mean_wd[j] = circular_mean_dir(dir_vals)

        # 生成 U/V（仅对风速与风向同时有效的站点）
        U = np.full(n_sta, np.nan, dtype=float)
        V = np.full(n_sta, np.nan, dtype=float)
        valid_vec = ~np.isnan(mean_ws) & ~np.isnan(mean_wd)
        if valid_vec.any():
            U[valid_vec], V[valid_vec] = barb_uv_from_wswd(mean_ws[valid_vec], mean_wd[valid_vec])

        # 保存 CSV
        suffix = stat
        out_csv = os.path.join(csv_dir, f"Cluster_{cid}_{suffix}.csv")
        df = pd.DataFrame({
            "station": stids,
            "lon": lons,
            "lat": lats,
            "height": heights,
            f"{suffix}_ws": mean_ws,
            f"{suffix}_wd": mean_wd,
            "count_ws_ty": cnt_ws_ty,
            "count_wd_ty": cnt_wd_ty,
        })
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"  保存: {out_csv}")

        # 绘图
        title = (
            f"Cluster {cid} | {'过程极大风' if stat=='meanmax' else '过程平均风'} 簇均值\n"
            f"样本台风数={len(tids)} | 站点有效数={int(np.sum(valid_vec))}"
            + (" | 按小时加权" if (stat=='meantime' and weight_by_hours) else "")
        )
        out_fig = os.path.join(fig_dir, f"Cluster_{cid}_{suffix}.png")
        draw_station_wind_map(
            lons=lons, lats=lats, U=U, V=V, ws_values=mean_ws,
            stids=stids, heights=heights,
            title=title, save_path=out_fig,
            extent=extent, annotate=annotate
        )
        print(f"  保存: {out_fig}")

    nc.close()
    print("全部完成！")


# ------------------ CLI ------------------

def main():
    parser = argparse.ArgumentParser(description="按簇绘制 过程极大风/过程平均风 的簇均值风场图")
    parser.add_argument("--nc", type=str, default="All_Typhoons_ExMaxWind.nc", help="NetCDF 文件路径")
    parser.add_argument("--cluster_csv", type=str, default="影响段_簇映射.csv", help="聚类结果 CSV（含 tid, cluster_id）")
    parser.add_argument("--out", type=str, default="输出_簇平均风场", help="输出目录")
    parser.add_argument("--extent", type=float, nargs=4, default=None, help="绘图范围 [minlon maxlon minlat maxlat]")
    parser.add_argument("--no-annotate", action="store_true", help="不在图上标注站号/风速/海拔")
    parser.add_argument("--loose-pair", action="store_true", help="风速与风向分别按各自有效样本做平均（默认要求配对）")
    parser.add_argument("--stat", type=str, choices=["meanmax", "meantime"], default="meanmax", help="统计口径：meanmax=过程极大风；meantime=过程平均风")
    parser.add_argument("--weight-by-hours", action="store_true", help="仅在 --stat meantime 时生效：按有效小时数加权（等价于把所有小时样本放在一起整体平均）")

    args = parser.parse_args()

    compute_cluster_mean(
        nc_path=args.nc,
        cluster_csv=args.cluster_csv,
        output_dir=args.out,
        stat=args.stat,
        extent=args.extent,
        annotate=not args.no_annotate,
        require_dir_pair=(not args.loose_pair),
        weight_by_hours=args.weight_by_hours,
    )


if __name__ == "__main__":
    main()
