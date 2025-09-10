#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
台风路径聚类（HDBSCAN 版）

目标：复用 KMeans 版本的输入/输出格式与流程，但用 HDBSCAN 做密度聚类；
- 支持体检报告（found_in_lib / win_points / win_arclength_km / participate_cluster）
- 支持簇映射（idx / tid / start / end / cluster_id，噪声= -1）
- 支持成图（所有影响段路径按簇着色；噪声灰色）

依赖：
  pip install hdbscan umap-learn pandas numpy matplotlib cartopy

用法示例：
python 台风路径聚类_切换版.py \
    --algo kmeans \
    --excel "/Users/momo/Desktop/业务相关/2025 影响台风大风/2010_2024_影响台风_大风.xlsx" \
    --besttrack-dir "/Users/momo/Desktop/业务相关/2025 影响台风大风/热带气旋最佳路径数据集" \
    --out-dir "/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_影响段聚类_KMeans" \
    --resample-N 30 --min-arclen 100 --min-points 3 \
    --k-candidates 3 4 5 6 7 8 9 10 11 12 13 14 15

python 台风路径聚类_切换版.py \
    --algo hdbscan \
    --excel "/Users/momo/Desktop/业务相关/2025 影响台风大风/2010_2024_影响台风_大风.xlsx" \
    --besttrack-dir "/Users/momo/Desktop/业务相关/2025 影响台风大风/热带气旋最佳路径数据集" \
    --out-dir "/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_影响段聚类_HDBSCAN" \
    --resample-N 30 --min-arclen 100 --min-points 3 \
    --use-umap --umap-neigh 20 --umap-min-dist 0.1 \
    --hdb-min-cluster-size 10 --hdb-min-samples 7

    python 台风路径聚类_HDBSCAN.py \
    --excel "/Users/momo/Desktop/业务相关/2025 影响台风大风/2010_2024_影响台风_大风.xlsx" \
    --besttrack-dir "/Users/momo/Desktop/业务相关/2025 影响台风大风/热带气旋最佳路径数据集" \
    --out-dir "/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_影响段聚类_HDBSCAN" \
    --resample-N 30 --min-arclen 100 --min-points 3 \
    --use-umap --umap-neigh 20 --umap-min-dist 0.1 \
    --hdb-min-cluster-size 10 --hdb-min-samples 7
说明：
- 若你已有从“台风路径可视化脚本”里复用的读取函数，可在本脚本替换 `read_besttrack_for_ids()` 部分；
- 当前内置了一个 66666-格式的兜底解析（常见 CMA-Best Track 文本），文件夹下批量搜索匹配台风编号；
- 输出与 KMeans 脚本一致：
  1) 影响段_体检报告.csv
  2) 影响段_簇映射.csv（cluster_id=-1 表示噪声/未入簇）
  3) 影响段_路径簇.png（按簇着色的路径分布）
"""

import os
import re
import glob
import math
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 可选：先降维到 2D/3D 再聚类
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    import hdbscan
except Exception as e:
    raise SystemExit("请先安装 hdbscan ： pip install hdbscan\n" + str(e))

# ---------------------- 地理/轨迹工具 ----------------------

R_EARTH = 6371.0  # km


def haversine(lon1, lat1, lon2, lat2):
    """两点大圆距离（km）"""
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R_EARTH * c


def path_length_km(lons, lats):
    if len(lons) < 2:
        return 0.0
    d = 0.0
    for i in range(1, len(lons)):
        d += haversine(lons[i-1], lats[i-1], lons[i], lats[i])
    return d


def interp_hourly(times, lons, lats):
    """将离散路径插值到逐小时时间轴（线性插值，假设区域尺度较小）"""
    if len(times) == 0:
        return [], [], []
    tmin, tmax = min(times), max(times)
    if tmin == tmax:
        return [tmin], [lons[0]], [lats[0]]
    # 构建整点小时轴
    tgrid = []
    t = datetime(tmin.year, tmin.month, tmin.day, tmin.hour)
    while t <= tmax:
        tgrid.append(t)
        t += timedelta(hours=1)

    # 转换为数值时间
    t0 = times[0]
    x = np.array([(tt - t0).total_seconds()/3600.0 for tt in times], dtype=float)
    xi = np.array([(tt - t0).total_seconds()/3600.0 for tt in tgrid], dtype=float)

    lons = np.array(lons, dtype=float)
    lats = np.array(lats, dtype=float)

    # 简单线性插值
    lon_i = np.interp(xi, x, lons)
    lat_i = np.interp(xi, x, lats)

    return tgrid, lon_i.tolist(), lat_i.tolist()


def arclen_resample(lons, lats, N=30):
    """按弧长均匀重采样到 N 个点"""
    if len(lons) < 2:
        lons = lons * N
        lats = lats * N
        return np.array(lons[:N]), np.array(lats[:N])
    seg = [0.0]
    for i in range(1, len(lons)):
        seg.append(seg[-1] + haversine(lons[i-1], lats[i-1], lons[i], lats[i]))
    total = seg[-1]
    if total == 0:
        return np.array([lons[0]]*N), np.array([lats[0]]*N)
    seg = np.array(seg)
    s_target = np.linspace(0, total, N)
    lon_i = np.interp(s_target, seg, lons)
    lat_i = np.interp(s_target, seg, lats)
    return lon_i, lat_i

# ---------------------- 数据读取 ----------------------

DT_RE = re.compile(r"\\d{10}")  # yyyymmddHH


def parse_bt_file(filepath):
    """兜底解析一个最佳路径文本文件（含 66666 头），返回 {id: [(time, lon, lat), ...]}"""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    data = defaultdict(list)
    cur_id = None
    for ln in lines:
        if ln.strip().startswith('66666'):
            # 约定：中国编号位于 EEEE（22-25）位置或用正则捕获 4 位数字
            try:
                tid = ln[21:25].strip()
                if not tid.isdigit():
                    # 回退：抓取首个 4 位数字段
                    m = re.search(r"(\d{4})", ln)
                    tid = m.group(1) if m else None
            except Exception:
                tid = None
            cur_id = tid
            continue
        # 记录行：包含时间/经纬度
        if cur_id is None:
            continue
        m = re.search(DT_RE, ln)
        if not m:
            continue
        ts = m.group(0)
        try:
            t = datetime.strptime(ts, "%Y%m%d%H")
        except Exception:
            continue
        # 尝试抓经纬度（常见格式带 E/W/N/S 标记或十进制度）
        # 这里给出一个简化例子：找两段类似 '120.3' '25.1'
        nums = re.findall(r"[-+]?\d+\.\d+", ln)
        if len(nums) >= 2:
            lon = float(nums[-2]); lat = float(nums[-1])
            data[cur_id].append((t, lon, lat))
    # 按时间排序
    for k in data:
        data[k].sort(key=lambda x: x[0])
    return dict(data)


def read_besttrack_for_ids(besttrack_dir, id_list):
    """在文件夹内搜索包含这些 id 的文本，解析并合并，返回 {id: (times[], lons[], lats[])}"""
    id_list = list(pd.unique([str(x) for x in id_list]))
    agg = {tid: ([], [], []) for tid in id_list}

    files = glob.glob(os.path.join(besttrack_dir, "**", "*.txt"), recursive=True)
    files += glob.glob(os.path.join(besttrack_dir, "**", "*.dat"), recursive=True)

    for fp in files:
        parsed = parse_bt_file(fp)
        for tid, rows in parsed.items():
            if tid in agg and rows:
                times, lons, lats = zip(*rows)
                agg[tid][0].extend(times)
                agg[tid][1].extend(lons)
                agg[tid][2].extend(lats)

    # 排序去重
    out = {}
    for tid in id_list:
        t, x, y = agg[tid]
        if len(t) == 0:
            continue
        # zip 排序
        arr = sorted(zip(t, x, y), key=lambda z: z[0])
        t2, x2, y2 = zip(*arr)
        out[tid] = (list(t2), list(x2), list(y2))
    return out

# ---------------------- 聚类准备：样本构建与向量化 ----------------------

def build_window_segments(excel_path, bt_dict, resample_N=30,
                          min_points_for_cluster=3, min_arclen_km=100.0):
    """
    从 Excel（tid/start/end）与最佳路径字典，构建“影响段样本”。
    返回：
      samples: 列表，每个元素 dict：{
        'idx', 'tid', 'start', 'end',
        'win_times', 'win_lons', 'win_lats',
        'arclen_km', 'ok_for_cluster',
        'feat'  # 向量化特征（若 ok）
      }
    """
    df = pd.read_excel(excel_path)
    # 兼容常见列名
    col_id = '中央台编号'
    col_s = '大风开始时间'
    col_e = '大风结束时间'
    if col_id not in df.columns:
        raise ValueError("Excel 缺少列：中央台编号")
    if col_s not in df.columns or col_e not in df.columns:
        raise ValueError("Excel 缺少列：大风开始时间 / 大风结束时间")

    # 规范类型
    df[col_id] = df[col_id].astype(str).str.zfill(4)
    df[col_s] = pd.to_datetime(df[col_s])
    df[col_e] = pd.to_datetime(df[col_e])

    samples = []
    for i, row in df.iterrows():
        tid = row[col_id]
        t0 = row[col_s].to_pydatetime()
        t1 = row[col_e].to_pydatetime()
        rec = {
            'idx': int(i), 'tid': tid, 'start': t0, 'end': t1,
            'win_times': [], 'win_lons': [], 'win_lats': [],
            'arclen_km': 0.0, 'ok_for_cluster': False, 'feat': None
        }
        if tid not in bt_dict:
            samples.append(rec)
            continue
        times, lons, lats = bt_dict[tid]
        # 插值到逐小时
        tgrid, xl, yl = interp_hourly(times, lons, lats)
        # 取窗口
        mask = [(t>=t0 and t<=t1) for t in tgrid]
        if not any(mask):
            samples.append(rec)
            continue
        win_t = [t for t, m in zip(tgrid, mask) if m]
        win_x = [x for x, m in zip(xl, mask) if m]
        win_y = [y for y, m in zip(yl, mask) if m]
        rec['win_times'] = win_t
        rec['win_lons']  = win_x
        rec['win_lats']  = win_y
        rec['arclen_km'] = path_length_km(win_x, win_y)
        # 质控
        if (len(win_t) >= min_points_for_cluster) and (rec['arclen_km'] >= min_arclen_km):
            # 弧长均匀重采样 + 差分向量化（平移不敏感）
            rx, ry = arclen_resample(win_x, win_y, N=resample_N)
            dx = np.diff(rx); dy = np.diff(ry)
            feat = np.concatenate([dx, dy])  # 长度 2*(N-1)
            rec['feat'] = feat
            rec['ok_for_cluster'] = True
        samples.append(rec)
    return samples

# ---------------------- HDBSCAN 聚类 ----------------------

def cluster_hdbscan(samples, use_umap=True, umap_neigh=20, umap_min_dist=0.1,
                    hdb_min_cluster_size=10, hdb_min_samples=7, random_state=42):
    # 取有效样本
    feats = [s['feat'] for s in samples if s['ok_for_cluster'] and s['feat'] is not None]
    idxs  = [i for i, s in enumerate(samples) if s['ok_for_cluster'] and s['feat'] is not None]
    if len(feats) == 0:
        return np.array([-1]*len(samples)), None, None

    X = np.vstack(feats)
    X = StandardScaler().fit_transform(X)

    Z = X
    reducer = None
    if use_umap:
        if not HAS_UMAP:
            raise SystemExit("未安装 umap-learn，请先 pip install umap-learn 或关闭 --use-umap")
        reducer = umap.UMAP(n_neighbors=umap_neigh, min_dist=umap_min_dist,
                            n_components=2, metric='euclidean', random_state=random_state)
        Z = reducer.fit_transform(X)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=hdb_min_cluster_size,
                                min_samples=hdb_min_samples,
                                metric='euclidean',
                                prediction_data=True)
    labels = clusterer.fit_predict(Z)

    # 回填到所有样本（未参与聚类/失败者= -1）
    cluster_id = np.array([-1]*len(samples))
    for lab, pos in zip(labels, idxs):
        cluster_id[pos] = int(lab) if lab != -1 else -1

    return cluster_id, Z, reducer

# ---------------------- 输出与成图 ----------------------

def save_reports_and_plot(samples, cluster_id, out_dir, title_tag="HDBSCAN"):
    os.makedirs(out_dir, exist_ok=True)

    # 体检报告
    report_rows = []
    for s in samples:
        report_rows.append({
            'idx': s['idx'], 'tid': s['tid'], 'start': s['start'], 'end': s['end'],
            'found_in_lib': len(s['win_times'])>0,
            'win_points': len(s['win_times']),
            'win_arclength_km': round(float(s['arclen_km']), 1),
            'participate_cluster': bool(s['ok_for_cluster'])
        })
    df_rep = pd.DataFrame(report_rows)
    rep_path = os.path.join(out_dir, '影响段_体检报告.csv')
    df_rep.to_csv(rep_path, index=False, encoding='utf-8-sig')

    # 簇映射
    map_rows = []
    for s, cid in zip(samples, cluster_id):
        map_rows.append({
            'idx': s['idx'], 'tid': s['tid'], 'start': s['start'], 'end': s['end'],
            'cluster_id': int(cid)
        })
    df_map = pd.DataFrame(map_rows)
    map_path = os.path.join(out_dir, '影响段_簇映射.csv')
    df_map.to_csv(map_path, index=False, encoding='utf-8-sig')

    # 成图（所有段放一张图）
    fig = plt.figure(figsize=(9, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_title(f"影响段路径聚类（{title_tag}）", fontsize=14)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
    ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')

    # 颜色表（预设 10 种，不够则循环）；噪声用灰色
    colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#a65628','#f781bf','#999999','#66c2a5','#fc8d62']

    # 自动范围
    all_lons = []; all_lats = []
    for s in samples:
        all_lons += s['win_lons']
        all_lats += s['win_lats']
    if all_lons and all_lats:
        xmin, xmax = np.nanmin(all_lons), np.nanmax(all_lons)
        ymin, ymax = np.nanmin(all_lats), np.nanmax(all_lats)
        ax.set_extent([xmin-2, xmax+2, ymin-2, ymax+2], crs=ccrs.PlateCarree())

    # 逐段画线
    for s, cid in zip(samples, cluster_id):
        if len(s['win_lons']) < 2:
            continue
        if cid == -1:
            ax.plot(s['win_lons'], s['win_lats'], color='0.7', linewidth=1.2, alpha=0.8, transform=ccrs.PlateCarree())
        else:
            c = colors[int(cid) % len(colors)]
            ax.plot(s['win_lons'], s['win_lats'], color=c, linewidth=1.6, alpha=0.9, transform=ccrs.PlateCarree())

    out_png = os.path.join(out_dir, '影响段_路径簇.png')
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches='tight')
    plt.close(fig)

    print(f"→ 体检报告： {rep_path}")
    print(f"→ 簇映射： {map_path}")
    print(f"→ 聚类成图： {out_png}")

# ---------------------- 主流程 ----------------------

def main():
    ap = argparse.ArgumentParser(description='台风影响段路径聚类（HDBSCAN 版）')
    ap.add_argument('--excel', type=str, required=True, help='影响台风 Excel（含 中央台编号/大风开始时间/大风结束时间）')
    ap.add_argument('--besttrack-dir', type=str, required=True, help='最佳路径库根目录（含 66666 格式 txt/dat 文件）')
    ap.add_argument('--out-dir', type=str, required=True, help='输出目录')

    ap.add_argument('--resample-N', type=int, default=30, help='弧长均匀重采样点数')
    ap.add_argument('--min-arclen', type=float, default=100.0, help='参与聚类的最小弧长（km）')
    ap.add_argument('--min-points', type=int, default=3, help='参与聚类的最少小时点数')

    ap.add_argument('--use-umap', action='store_true', help='是否先用 UMAP 降维到2D后再聚类')
    ap.add_argument('--umap-neigh', type=int, default=20, help='UMAP n_neighbors')
    ap.add_argument('--umap-min-dist', type=float, default=0.1, help='UMAP min_dist')

    ap.add_argument('--hdb-min-cluster-size', type=int, default=10, help='HDBSCAN min_cluster_size')
    ap.add_argument('--hdb-min-samples', type=int, default=7, help='HDBSCAN min_samples')

    args = ap.parse_args()

    print('读取 Excel…')
    df = pd.read_excel(args.excel)
    id_list = df['中央台编号'].astype(str).str.zfill(4).tolist()

    print('从最佳路径库中仅提取这些编号的轨迹…')
    bt_dict = read_besttrack_for_ids(args.besttrack_dir, id_list)

    print('逐小时插值 & 构建影响段样本…')
    samples = build_window_segments(
        excel_path=args.excel,
        bt_dict=bt_dict,
        resample_N=args.resample_N,
        min_points_for_cluster=args.min_points,
        min_arclen_km=args.min_arclen
    )

    print('向量化 & HDBSCAN 聚类…')
    cluster_id, Z, reducer = cluster_hdbscan(
        samples,
        use_umap=args.use_umap,
        umap_neigh=args.umap_neigh,
        umap_min_dist=args.umap_min_dist,
        hdb_min_cluster_size=args.hdb_min_cluster_size,
        hdb_min_samples=args.hdb_min_samples
    )

    print('输出报告与成图…')
    save_reports_and_plot(samples, cluster_id, args.out_dir, title_tag='HDBSCAN')


if __name__ == '__main__':
    main()
