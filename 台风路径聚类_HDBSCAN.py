#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
台风路径聚类（KMeans / HDBSCAN 一键切换，**复用可视化脚本的读取与插值**）

要点：
- 与你【已跑通】的 KMeans 脚本保持同一条读数链：
  read_excel_windows → iter_needed_segments → hourly_interp
  （三者优先从 --vis-script 加载；若缺失则使用本文件的 Fallback）
- 向量化/输出格式完全一致：
  1) 影响段_体检报告.csv  2) 影响段_簇映射.csv  3) 影响段_路径簇.pn
- 仅替换聚类器：可用 --algo 在 KMeans 与 HDBSCAN 之间一键切换。

依赖：
  pip install scikit-learn hdbscan umap-learn pandas numpy matplotlib cartopy

示例：
  HDBSCAN：
  python 台风路径聚类_切换版.py \
    --algo hdbscan \
    --excel "/Users/momo/Desktop/业务相关/2025 影响台风大风/2010_2024_影响台风_大风.xlsx" \
    --besttrack-dir "/Users/momo/Desktop/业务相关/2025 影响台风大风/热带气旋最佳路径数据集" \
    --vis-script "/Users/momo/Desktop/业务相关/2025 影响台风大风/代码/台风路径可视化_插值小时.py" \
    --out-dir "/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_影响段聚类_HDBSCAN" \
    --resample-N 30 --min-arclen 100 --min-points 3 \
    --use-umap --umap-neigh 15 --umap-min-dist 0.2 \
    --hdb-min-cluster-size 6 --hdb-min-samples 4

  KMeans：
  python 台风路径聚类_切换版.py \
    --algo kmeans \
    --excel "/Users/momo/Desktop/业务相关/2025 影响台风大风/2010_2024_影响台风_大风.xlsx" \
    --besttrack-dir "/Users/momo/Desktop/业务相关/2025 影响台风大风/热带气旋最佳路径数据集" \
    --vis-script "/Users/momo/Desktop/业务相关/2025 影响台风大风/代码/台风路径可视化_插值小时.py" \
    --out-dir "/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_影响段聚类_KMeans" \
    --resample-N 30 --min-arclen 100 --min-points 3 \
    --k-candidates 3 4 5 6 7 8 9 10 11 12 13 14 15
"""

import os, re, math, glob, types, importlib.util, hashlib, argparse
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Set

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 可选降维
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    import hdbscan
    HAS_HDB = True
except Exception:
    HAS_HDB = False

# ------------------ 地理工具 ------------------
EARTH_R = 6371.0

def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return float(EARTH_R*c)

def slerp_lonlat(lon1, lat1, lon2, lat2, f):
    # 球面线性插值
    a = np.deg2rad([lon1, lat1]); b = np.deg2rad([lon2, lat2])
    a = np.array([np.cos(a[1])*np.cos(a[0]), np.cos(a[1])*np.sin(a[0]), np.sin(a[1])])
    b = np.array([np.cos(b[1])*np.cos(b[0]), np.cos(b[1])*np.sin(b[0]), np.sin(b[1])])
    dot = float(np.clip(np.dot(a,b), -1.0, 1.0))
    omega = math.acos(dot); so = math.sin(omega) if omega!=0 else 1.0
    v = (math.sin((1-f)*omega)/so)*a + (math.sin(f*omega)/so)*b
    lat = math.degrees(math.asin(np.clip(v[2], -1.0, 1.0)))
    lon = math.degrees(math.atan2(v[1], v[0]))
    return lon, lat

# ------------------ Fallback：读取/插值/Excel ------------------
TIME_RE = re.compile(r"(\d{10})")
NUM_RE  = re.compile(r"[-+]?\d+\.?\d*")

@dataclass
class BTPoint:
    t: pd.Timestamp
    lon: float
    lat: float

def _fallback_parse_header_tid(line: str, file_year: Optional[int]) -> Optional[str]:
    s = line.lstrip()
    if not s.startswith("66666"):
        return None
    toks = re.findall(r"\b(\d{4})\b", s)
    if not toks:
        return None
    if file_year is not None:
        yy = f"{file_year%100:02d}"
        for t in toks:
            if t.startswith(yy):
                return t
    for t in reversed(toks):
        if t != "0000":
            return t
    return toks[-1]

def _fallback_parse_record_line(line: str) -> Optional[BTPoint]:
    s = line.strip()
    if not s or s.lstrip().startswith("66666"):
        return None
    m = TIME_RE.search(s)
    if not m:
        return None
    ts = pd.to_datetime(m.group(1), format="%Y%m%d%H", errors="coerce")
    if pd.isna(ts):
        return None
    after = s[m.end():].strip()
    nums = [float(x) for x in NUM_RE.findall(after)]
    if len(nums) < 3:
        return None
    lat_raw = nums[1]; lon_raw = nums[2]
    if 0 <= abs(lat_raw) <= 900 and 0 <= abs(lon_raw) <= 1800 and (abs(lon_raw)>180 or abs(lat_raw)>90):
        lat, lon = lat_raw/10.0, lon_raw/10.0
    else:
        lat, lon = lat_raw, lon_raw
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return BTPoint(ts, float(lon), float(lat))

def _fallback_iter_needed_segments(folder: str, target_tids: Set[str]) -> Dict[str, List[BTPoint]]:
    out: Dict[str, List[BTPoint]] = {tid: [] for tid in target_tids}
    for root, _, files in os.walk(folder):
        for fn in sorted(files):
            if not fn.lower().endswith((".txt",".dat")):
                continue
            path = os.path.join(root, fn)
            m = re.search(r"(19|20)\d{2}", fn); file_year = int(m.group(0)) if m else None
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
            except Exception:
                continue
            cur_tid: Optional[str] = None
            cur_pts: List[BTPoint] = []
            def flush():
                nonlocal cur_tid, cur_pts
                if cur_tid and cur_pts and cur_tid in out:
                    out[cur_tid].extend(cur_pts)
                cur_tid = None; cur_pts = []
            for ln in lines:
                if ln.lstrip().startswith("66666"):
                    flush(); cur_tid = _fallback_parse_header_tid(ln, file_year); continue
                if cur_tid is None: continue
                p = _fallback_parse_record_line(ln)
                if p: cur_pts.append(p)
            flush()
    # 排序并按时间去重
    for tid, arr in out.items():
        arr.sort(key=lambda x: x.t)
        uniq = {p.t: p for p in arr}
        out[tid] = [uniq[t] for t in sorted(uniq.keys())]
    return out

def _fallback_hourly_interp(points: List[BTPoint]) -> List[BTPoint]:
    if not points: return []
    points = sorted(points, key=lambda x: x.t)
    times = [p.t for p in points]
    out: List[BTPoint] = []
    cur = times[0].floor('h'); end = times[-1].ceil('h')
    idx = 0
    while cur <= end:
        while idx+1 < len(times) and times[idx+1] < cur: idx += 1
        if cur <= times[0]: p = points[0]; out.append(BTPoint(cur, p.lon, p.lat))
        elif cur >= times[-1]: p = points[-1]; out.append(BTPoint(cur, p.lon, p.lat))
        else:
            i = max(0, np.searchsorted(times, cur) - 1); j = i + 1
            t0, t1 = times[i], times[j]; p0, p1 = points[i], points[j]
            if t0 == t1: out.append(BTPoint(cur, p0.lon, p0.lat))
            else:
                f = float((cur - t0) / (t1 - t0)); f = float(np.clip(f, 0.0, 1.0))
                lon, lat = slerp_lonlat(p0.lon, p0.lat, p1.lon, p1.lat, f)
                out.append(BTPoint(cur, lon, lat))
        cur += pd.Timedelta(hours=1)
    return out

def _fallback_read_excel_windows(path: str, id_col: str, st_col: str, en_col: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df[[id_col, st_col, en_col]].copy()
    df[id_col] = df[id_col].astype(str).str.strip().str.zfill(4)
    df[st_col] = pd.to_datetime(df[st_col]); df[en_col] = pd.to_datetime(df[en_col])
    df = df.dropna().sort_values([id_col, st_col]).reset_index(drop=True)
    return df

# ------------------ 绑定函数（优先使用可视化脚本） ------------------

def load_vis_module(path: str) -> types.ModuleType:
    mname = "ty_vis_hourly_" + hashlib.md5(path.encode("utf-8")).hexdigest()[:10]
    spec = importlib.util.spec_from_file_location(mname, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载可视化脚本：{path}")
    mod = importlib.util.module_from_spec(spec)
    sys_modules = __import__('sys').modules
    sys_modules[mname] = mod
    spec.loader.exec_module(mod)
    return mod

# ------------------ 质控 / 重采样 / 向量化 ------------------
@dataclass
class ImpactSeg:
    idx: int
    tid: str
    start: pd.Timestamp
    end: pd.Timestamp
    pts: List[BTPoint]  # 逐小时

def extract_impact_segment(hourly_pts: List[BTPoint], st: pd.Timestamp, en: pd.Timestamp) -> List[BTPoint]:
    return [p for p in hourly_pts if st <= p.t <= en]

def seg_arclength(coords: List[Tuple[float,float]]) -> float:
    s = 0.0
    for i in range(1, len(coords)):
        s += haversine_km(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1])
    return s

def resample_by_arclength(coords: List[Tuple[float,float]], N: int) -> np.ndarray:
    if len(coords) < 2:
        raise ValueError("Not enough points to resample")
    d = [0.0]
    for i in range(1, len(coords)):
        d.append(d[-1] + haversine_km(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1]))
    total = d[-1]
    if total == 0: return np.repeat(np.asarray(coords[:1]), N, axis=0)
    targets = np.linspace(0.0, total, N)
    res = []
    j = 0
    for s in targets:
        while j+1 < len(d) and d[j+1] < s: j += 1
        if j+1 >= len(d): res.append(coords[-1])
        else:
            if d[j+1] == d[j]: res.append(coords[j])
            else:
                f = (s - d[j])/(d[j+1]-d[j])
                lon = coords[j][0] + f*(coords[j+1][0]-coords[j][0])
                lat = coords[j][1] + f*(coords[j+1][1]-coords[j][1])
                res.append((lon,lat))
    return np.asarray(res)

def vectorize_segments(segments: List[ImpactSeg], resample_N: int, min_points: int, min_arclen_km: float):
    X, keep_idx, curves = [], [], []
    for idx, seg in enumerate(segments):
        coords = [(p.lon, p.lat) for p in seg.pts]
        npts = len(coords)
        if npts < min_points:
            continue
        arc = seg_arclength(coords)
        if arc < min_arclen_km:
            continue
        curve = resample_by_arclength(coords, N=resample_N)
        dxy = np.diff(curve, axis=0).reshape(-1)  # 平移不敏感
        X.append(dxy); keep_idx.append(idx); curves.append(curve)
    if not X:
        return np.empty((0, resample_N*2)), [], []
    return np.vstack(X), keep_idx, curves

# ------------------ 聚类器 ------------------

def run_kmeans(X: np.ndarray, k_candidates: List[int], random_state=42):
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    best = (None, -np.inf, None, None)  # (k, score, labels, model)
    for k in k_candidates:
        km = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
        labs = km.fit_predict(Xs)
        try:
            sc = float(silhouette_score(Xs, labs)) if k >= 2 else -np.inf
        except Exception:
            sc = -np.inf
        if sc > best[1]:
            best = (k, sc, labs, km)
    return best  # (k, score, labels, model)

def run_hdbscan(X: np.ndarray, use_umap=True, umap_neigh=20, umap_min_dist=0.1,
                min_cluster_size=10, min_samples=7, random_state=42):
    if not HAS_HDB:
        raise SystemExit("未安装 hdbscan，请先 pip install hdbscan")
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    Z = Xs
    reducer = None
    if use_umap:
        if not HAS_UMAP:
            raise SystemExit("未安装 umap-learn，请先 pip install umap-learn 或移除 --use-umap")
        reducer = umap.UMAP(n_neighbors=umap_neigh, min_dist=umap_min_dist,
                            n_components=2, metric='euclidean', random_state=random_state)
        Z = reducer.fit_transform(Xs)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                metric='euclidean', prediction_data=True)
    labels = clusterer.fit_predict(Z)
    # 去噪轮廓
    mask = labels >= 0
    if mask.any() and len(np.unique(labels[mask])) >= 2:
        try:
            s = float(silhouette_score(Z[mask], labels[mask]))
        except Exception:
            s = float('nan')
    else:
        s = float('nan')
    return labels, clusterer, reducer, s

# ------------------ 成图与输出 ------------------
CLUSTER_COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f"]

def plot_clusters(segments: List[ImpactSeg], labels_map: Dict[int,int], out_png: str, extent=None, title='大风影响段路径聚类'):
    if HAS_CARTOPY:
        proj = ccrs.PlateCarree(); fig = plt.figure(figsize=(11,9)); ax = plt.axes(projection=proj)
        if extent: ax.set_extent(extent, crs=proj)
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
        ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
        ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    else:
        fig = plt.figure(figsize=(10,8)); ax = plt.gca()
        if extent: ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.grid(True, ls='--', alpha=0.4); ax.set_xlabel('Lon'); ax.set_ylabel('Lat')
    counts = {}
    for i, seg in enumerate(segments):
        cid = labels_map.get(i, -1)
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)] if cid >= 0 else "#999999"
        xs = [p.lon for p in seg.pts]; ys = [p.lat for p in seg.pts]
        if HAS_CARTOPY:
            plt.plot(xs, ys, '-', lw=2.2 if cid>=0 else 1.2, color=color, alpha=0.95 if cid>=0 else 0.6, transform=ccrs.PlateCarree())
        else:
            plt.plot(xs, ys, '-', lw=2.2 if cid>=0 else 1.2, color=color, alpha=0.95 if cid>=0 else 0.6)
        counts[cid] = counts.get(cid, 0) + 1
    plt.title(title)
    plt.tight_layout(); os.makedirs(os.path.dirname(out_png), exist_ok=True); plt.savefig(out_png, dpi=200); plt.close()

# ------------------ 主流程 ------------------

def main():
    ap = argparse.ArgumentParser(description='台风影响段路径聚类（KMeans/HDBSCAN，一键切换，复用可视化读取）')
    ap.add_argument('--algo', choices=['kmeans','hdbscan'], default='hdbscan')
    ap.add_argument('--excel', required=True)
    ap.add_argument('--besttrack-dir', required=True)
    ap.add_argument('--vis-script', required=True, help='你的台风路径可视化_插值小时.py 的绝对路径')
    ap.add_argument('--out-dir', required=True)
    # Excel 列名（保持与你工程一致）
    ap.add_argument('--id-col', default='中央台编号')
    ap.add_argument('--st-col', default='大风开始时间')
    ap.add_argument('--en-col', default='大风结束时间')
    # 可选地图范围（不设则自动）
    ap.add_argument('--extent', type=float, nargs=4, default=None, help='lon_min lon_max lat_min lat_max')
    # 质控
    ap.add_argument('--resample-N', type=int, default=30)
    ap.add_argument('--min-arclen', type=float, default=100.0)
    ap.add_argument('--min-points', type=int, default=3)
    # HDBSCAN / UMAP
    ap.add_argument('--use-umap', action='store_true')
    ap.add_argument('--umap-neigh', type=int, default=20)
    ap.add_argument('--umap-min-dist', type=float, default=0.1)
    ap.add_argument('--hdb-min-cluster-size', type=int, default=10)
    ap.add_argument('--hdb-min-samples', type=int, default=7)
    # KMeans
    ap.add_argument('--k-candidates', type=int, nargs='*', default=[3,4,5,6,7,8,9,10,11,12,13,14,15])

    args = ap.parse_args()

    # 绑定可视化脚本函数
    print('加载可视化脚本函数…')
    VIS = load_vis_module(args.vis_script)
    iter_needed_segments = getattr(VIS, 'iter_needed_segments', _fallback_iter_needed_segments)
    hourly_interp        = getattr(VIS, 'hourly_interp',        _fallback_hourly_interp)
    read_excel_windows   = getattr(VIS, 'read_excel_windows',   lambda p: _fallback_read_excel_windows(p, args.id_col, args.st_col, args.en_col))

    print('读取 Excel…')
    dfx = read_excel_windows(args.excel)
    dfx = dfx[[args.id_col, args.st_col, args.en_col]].copy()
    dfx[args.id_col] = dfx[args.id_col].astype(str).str.zfill(4)
    target_tids: Set[str] = set(dfx[args.id_col])

    print('从最佳路径库中仅提取这些编号的轨迹（复用可视化脚本）…')
    raw_tracks: Dict[str, List[BTPoint]] = iter_needed_segments(args.besttrack_dir, target_tids)
    print(f"已识别到路径的台风个数：{sum(1 for k,v in raw_tracks.items() if v)} / {len(target_tids)}")

    print('逐小时插值（全生命史，复用可视化脚本）…')
    hourly: Dict[str, List[BTPoint]] = {tid: hourly_interp(pts) if pts else [] for tid, pts in raw_tracks.items()}

    # 构建影响段样本
    print('构建影响段样本…')
    segments: List[ImpactSeg] = []
    for i, row in dfx.iterrows():
        tid = str(row[args.id_col]).zfill(4)
        st  = pd.to_datetime(row[args.st_col])
        en  = pd.to_datetime(row[args.en_col])
        pts_all = hourly.get(tid, [])
        pts_win = extract_impact_segment(pts_all, st, en) if pts_all else []
        segments.append(ImpactSeg(idx=int(i), tid=tid, start=st, end=en, pts=pts_win))

    # 体检报告
    rows = []
    for seg in segments:
        coords = [(p.lon, p.lat) for p in seg.pts]
        n = len(coords); arc = seg_arclength(coords) if n>=2 else 0.0
        participate = (n >= args.min_points and arc >= args.min_arclen)
        rows.append({
            'idx': seg.idx, 'tid': seg.tid, 'start': seg.start, 'end': seg.end,
            'found_in_lib': bool(hourly.get(seg.tid)),
            'win_points': n,
            'win_arclength_km': round(arc, 1),
            'participate_cluster': participate,
        })
    rep_df = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    rep_path = os.path.join(args.out_dir, '影响段_体检报告.csv')
    rep_df.to_csv(rep_path, index=False, encoding='utf-8-sig')
    print('→ 体检报告：', rep_path)

    # 向量化
    print('向量化 & 聚类…')
    X, keep_idx, _ = vectorize_segments(segments, args.resample_N, args.min_points, args.min_arclen)
    if X.shape[0] < 2:
        print('样本不足，跳过聚类。可放宽门槛或检查 Excel 时段。')
        map_path = os.path.join(args.out_dir, '影响段_簇映射.csv')
        pd.DataFrame(columns=['idx','tid','start','end','cluster_id']).to_csv(map_path, index=False, encoding='utf-8-sig')
        print('→ 簇映射：', map_path)
        return

    if args.algo == 'kmeans':
        k_best, score_best, labels_best, _ = run_kmeans(X, args.k_candidates)
        print(f"KMeans：选定 K={k_best}，轮廓系数={score_best:.3f}（候选 {args.k_candidates}）")
        labels = labels_best
        title = 'KMeans'
    else:
        labels, model, reducer, s = run_hdbscan(X,
            use_umap=args.use_umap, umap_neigh=args.umap_neigh, umap_min_dist=args.umap_min_dist,
            min_cluster_size=args.hdb_min_cluster_size, min_samples=args.hdb_min_samples)
        k_eff = int(len(np.unique(labels[labels>=0]))) if (labels.size>0) else 0
        noise_ratio = float((labels<0).sum()/labels.size)
        print(f"HDBSCAN：簇数(不含噪声)={k_eff}，轮廓(去噪)={s if not np.isnan(s) else float('nan'):.3f}，噪声占比={noise_ratio:.2%}")
        title = 'HDBSCAN'

    # 回填映射
    idx2cid = {keep_idx[i]: int(labels[i]) for i in range(len(labels))}
    rows2 = [{
        'idx': seg.idx, 'tid': seg.tid, 'start': seg.start, 'end': seg.end,
        'cluster_id': idx2cid.get(i, -1)
    } for i, seg in enumerate(segments)]
    map_path = os.path.join(args.out_dir, '影响段_簇映射.csv')
    pd.DataFrame(rows2).to_csv(map_path, index=False, encoding='utf-8-sig')
    print('→ 簇映射：', map_path)

    # 成图
    png_path = os.path.join(args.out_dir, '影响段_路径簇.png')
    plot_clusters(segments, idx2cid, png_path, extent=args.extent, title=f'大风影响段路径聚类（{title}）')
    print('→ 聚类成图：', png_path)

if __name__ == '__main__':
    main()
