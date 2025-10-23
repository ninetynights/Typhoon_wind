#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cluster_impact_segments.py (已修改：同时考虑坐标和形状)

目的
- 复用你现有的《台风路径可视化_插值小时.py》里的解析/插值/抽取函数，只在此脚本里做“影响时段内的路径聚类”。
- 仅处理 Excel 中出现的台风编号；聚类时只用每条记录的“影响窗口内”那一段轨迹。
- [新] 聚类特征向量同时包含绝对坐标和形状（通过重采样的坐标序列实现）。

特点
- 用 importlib **直接加载你的可视化脚本**作为模块，优先复用其中函数；
- 如果该脚本里函数名不完全一致，本脚本含有**安全兜底实现**，会自动切换；
- 聚类：弧长重采样→拉平坐标序列→KMeans，K 由轮廓系数在候选里自动选择；
- 输出：
  1) 影响段_体检报告.csv（每条 Excel 记录是否命中、有效点数、弧长、是否参与聚类）；
  2) 影响段_簇映射.csv（idx, tid, start, end, cluster_id）；
  3) 影响段_路径簇.png（浙江近海范围按簇上色，未参与样本灰色）。

使用
1) 修改下方路径常量为你的本机路径；
2) 运行本文件即可。
"""

from __future__ import annotations
import os, re, math, json, importlib.util, types
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import sys, hashlib


# 可选底图
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Heiti TC', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# ------------------ 路径配置 ------------------
BESTTRACK_DIR = "/Users/momo/Desktop/业务相关/2025 影响台风大风/热带气旋最佳路径数据集"
EXCEL_PATH    = "/Users/momo/Desktop/业务相关/2025 影响台风大风/2010_2024_影响台风_大风_去除圆规.xlsx"
VIS_SCRIPT    = "/Users/momo/Desktop/业务相关/2025 影响台风大风/代码/台风路径可视化_插值小时.py"  # 你的可视化脚本完整路径
OUT_DIR       = os.path.join(os.path.dirname(EXCEL_PATH), "输出_影响段聚类_Kmeans_增加经纬度特征_去除圆规") # 改个新目录名

# Excel 列名
ID_COL = "中央台编号"
ST_COL = "大风开始时间"
EN_COL = "大风结束时间"

# 浙江及周边的地图范围
EXTENT = [105, 140, 15, 45]  # [lon_min, lon_max, lat_min, lat_max]

# 可调参数
PARAMS = dict(
    # 质控门槛（进入聚类前）
    min_points_for_cluster=2, # 轨迹点数量
    min_arclength_km=30.0, # 轨迹弧长（公里）
    # 重采样与聚类
    resample_N=30,
    k_candidates=[3,4,5,6,7,8,9,10,11,12,13,14,15]  # KMeans 聚类候选 K 值,
)

CLUSTER_COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f"]

# ------------------ 从可视化脚本加载函数 ------------------

def load_vis_module(path: str) -> types.ModuleType:
    """按文件路径加载脚本并注册到 sys.modules，避免 dataclass 导入期出错。"""
    mname = "ty_vis_hourly_" + hashlib.md5(path.encode("utf-8")).hexdigest()[:10]
    spec = importlib.util.spec_from_file_location(mname, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载可视化脚本：{path}")
    mod = importlib.util.module_from_spec(spec)
    # 关键：先注册，再执行
    sys.modules[mname] = mod
    spec.loader.exec_module(mod)
    return mod

@dataclass
class BTPoint:
    t: pd.Timestamp
    lon: float
    lat: float

# 兜底实现（当外部模块没有对应函数时启用）
EARTH_R = 6371.0
TIME_RE = re.compile(r"(\d{10})")
NUM_RE  = re.compile(r"[-+]?\d+\.?\d*")

def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return float(EARTH_R * c)

def slerp_lonlat(lon1, lat1, lon2, lat2, f: float):
    a = np.array([
        math.cos(math.radians(lat1))*math.cos(math.radians(lon1)),
        math.cos(math.radians(lat1))*math.sin(math.radians(lon1)),
        math.sin(math.radians(lat1))
    ])
    b = np.array([
        math.cos(math.radians(lat2))*math.cos(math.radians(lon2)),
        math.cos(math.radians(lat2))*math.sin(math.radians(lon2)),
        math.sin(math.radians(lat2))
    ])
    dot = float(np.clip(np.dot(a,b), -1.0, 1.0))
    omega = math.acos(dot)
    if omega < 1e-12:
        return float(lon1), float(lat1)
    so = math.sin(omega)
    v = (math.sin((1-f)*omega)/so)*a + (math.sin(f*omega)/so)*b
    lat = math.degrees(math.asin(np.clip(v[2], -1.0, 1.0)))
    lon = math.degrees(math.atan2(v[1], v[0]))
    return lon, lat

# Fallback：iter_needed_segments / hourly_interp / read_excel_windows

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
                cur_tid, cur_pts = None, []
            for ln in lines:
                if ln.lstrip().startswith("66666"):
                    flush(); cur_tid = _fallback_parse_header_tid(ln, file_year); continue
                if cur_tid and cur_tid in out:
                    p = _fallback_parse_record_line(ln)
                    if p: cur_pts.append(p)
            flush()
    for tid in list(out.keys()):
        arr = out[tid]
        if not arr: continue
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

def _fallback_read_excel_windows(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df[[ID_COL, ST_COL, EN_COL]].copy()
    df[ID_COL] = df[ID_COL].astype(str).str.strip().str.zfill(4)
    df[ST_COL] = pd.to_datetime(df[ST_COL]); df[EN_COL] = pd.to_datetime(df[EN_COL])
    df = df.dropna().sort_values([ID_COL, ST_COL]).reset_index(drop=True)
    return df

# 绑定函数句柄（优先外部模块）
try:
    VIS = load_vis_module(VIS_SCRIPT)
    iter_needed_segments = getattr(VIS, 'iter_needed_segments', _fallback_iter_needed_segments)
    hourly_interp        = getattr(VIS, 'hourly_interp',        _fallback_hourly_interp)
    read_excel_windows   = getattr(VIS, 'read_excel_windows',   _fallback_read_excel_windows)
    print(f"成功从 {VIS_SCRIPT} 加载函数。")
except Exception as e:
    print(f"警告：无法从 {VIS_SCRIPT} 加载函数 ({e})，将使用内置的兜底实现。")
    iter_needed_segments = _fallback_iter_needed_segments
    hourly_interp        = _fallback_hourly_interp
    read_excel_windows   = _fallback_read_excel_windows


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

def resample_by_arclength(coords: List[Tuple[float,float]], N:int) -> np.ndarray:
    if len(coords) < 2:
        # 点不够，用第一个点（或一个nan点）重复N次
        if coords:
            return np.repeat(np.asarray(coords[:1]), N, axis=0)
        else:
             # 理论上不应发生，因为 vectorize_segments 有 npts 检查
             return np.full((N, 2), np.nan) 
             
    d = [0.0]
    for i in range(1, len(coords)):
        d.append(d[-1] + haversine_km(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1]))
    total = d[-1]
    
    # 如果路径没有移动（总长为0）
    if total < 1e-6: 
        return np.repeat(np.asarray(coords[:1]), N, axis=0)
        
    targets = np.linspace(0.0, total, N)
    res = []
    j = 0
    for s in targets:
        # 找到 s 所在的原始线段 [j, j+1]
        while j+1 < len(d) and d[j+1] < s: 
            j += 1
            
        if j+1 >= len(d):
            # 如果 s 超过了总长（浮点数误差），就取最后一个点
            res.append(coords[-1])
        else:
            if (d[j+1] - d[j]) < 1e-9:
                # 两个原始点重合
                res.append(coords[j])
            else:
                # 线性插值（用于经纬度，更准确应用球面插值，但线性通常也够用）
                f = (s - d[j]) / (d[j+1] - d[j])
                lon = coords[j][0] + f * (coords[j+1][0] - coords[j][0])
                lat = coords[j][1] + f * (coords[j+1][1] - coords[j][1])
                # (lon, lat) = slerp_lonlat(coords[j][0], coords[j][1], coords[j+1][0], coords[j+1][1], f) # 用球面插值更准，但线性插值更快
                res.append((lon,lat))
                
    return np.asarray(res)

def vectorize_segments(segments: List[ImpactSeg], P=PARAMS):
    X, keep_idx, curves = [], [], []
    for idx, seg in enumerate(segments):
        coords = [(p.lon, p.lat) for p in seg.pts]
        npts = len(coords)
        if npts < P["min_points_for_cluster"]:
            continue
        arc = seg_arclength(coords)
        if arc < P["min_arclength_km"]:
            continue
            
        # 核心步骤：重采样
        curve = resample_by_arclength(coords, P["resample_N"])  # (N, 2)
        
        # ******** 这是关键修改 *********
        # 原方法 (只看形状):
        # dxy = np.diff(curve, axis=0).reshape(-1)  # (N-1)*2
        # X.append(dxy)
        
        # 新方法 (同时看坐标和形状):
        # 直接拉平重采样后的坐标序列 [lon1, lat1, lon2, lat2, ...]
        feature_vector = curve.reshape(-1) # 长度 N*2
        X.append(feature_vector)
        # ******************************
        
        keep_idx.append(idx)
        curves.append(curve)
        
    if not X:
        # 注意：这里的维度也要跟着修改
        return np.zeros((0, P["resample_N"] * 2)), [], []
        
    return np.vstack(X), keep_idx, curves

# ------------------ 聚类与绘图 ------------------

def choose_k_and_cluster(X: np.ndarray, k_candidates: List[int]):
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    best = (-1.0, None, None, None) # (score, k, model, labels)
    print(f"开始自动选择K值 (候选: {k_candidates})，样本数: {X.shape[0]}")
    
    valid_candidates = [k for k in k_candidates if 1 < k < X.shape[0]]
    if not valid_candidates:
         print("警告: 样本数太少，无法在候选K值中进行有效聚类。")
         if X.shape[0] > 1:
             k = X.shape[0] - 1
         else:
             k = 1
         km = KMeans(n_clusters=k, n_init=10, random_state=42)
         labels = km.fit_predict(Xs)
         return labels, k, np.nan, km

    for k in valid_candidates:
        km = KMeans(n_clusters=k, n_init=50, random_state=42)
        labels = km.fit_predict(Xs)
        try: 
            s = silhouette_score(Xs, labels)
            print(f"  K = {k}, 轮廓系数 = {s:.4f}")
        except Exception: 
            s = -1.0
            print(f"  K = {k}, 轮廓系数计算失败")
            
        if s > best[0]: 
            best = (s, k, km, labels)
            
    if best[1] is None:
        print("警告：所有K值均未成功计算轮廓系数，将默认使用 k=4")
        k = 4
        if k >= X.shape[0]: k = max(1, X.shape[0] - 1)
        km = KMeans(n_clusters=k, n_init=50, random_state=42)
        labels = km.fit_predict(Xs)
        return labels, k, float('nan'), km
        
    return best[3], best[1], best[0], best[2]

def plot_clusters(segments: List[ImpactSeg], labels_map: Dict[int,int], k: int, score: float, out_png: str):
    if HAS_CARTOPY:
        proj = ccrs.PlateCarree(); fig = plt.figure(figsize=(11,9)); ax = plt.axes(projection=proj)
        ax.set_extent(EXTENT, crs=proj)
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
        ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
        ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    else:
        fig = plt.figure(figsize=(10,8)); ax = plt.gca()
        ax.set_xlim(EXTENT[0], EXTENT[1]); ax.set_ylim(EXTENT[2], EXTENT[3])
        ax.grid(True, ls='--', alpha=0.4); ax.set_xlabel('Lon'); ax.set_ylabel('Lat')

    counts = {}
    for i, seg in enumerate(segments):
        cid = labels_map.get(i, -1) # -1 表示未参与聚类
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)] if cid >= 0 else "#cccccc"
        lw = 2.2 if cid >= 0 else 1.0
        alpha = 0.95 if cid >= 0 else 0.5
        
        xs = [p.lon for p in seg.pts]; ys = [p.lat for p in seg.pts]
        if not xs: continue
        
        if HAS_CARTOPY:
            ax.plot(xs, ys, '-', lw=lw, color=color, alpha=alpha, transform=ccrs.PlateCarree())
        else:
            ax.plot(xs, ys, '-', lw=lw, color=color, alpha=alpha)
        counts[cid] = counts.get(cid, 0) + 1
    
    # 创建图例
    handles = []
    for i in range(k):
         handles.append(plt.Line2D([0],[0], color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)], lw=2, label=f'簇 {i} (N={counts.get(i, 0)})'))
    if -1 in counts:
         handles.append(plt.Line2D([0],[0], color="#cccccc", lw=1, label=f'未参与 (N={counts[-1]})'))
    ax.legend(handles=handles, title="聚类结果", loc="lower left", fontsize=9)
    
    plt.title(f'大风影响段路径聚类（K={k}, 轮廓系数={score:.3f}）\n(特征: 重采样坐标序列)')
    plt.tight_layout(); os.makedirs(OUT_DIR, exist_ok=True); plt.savefig(out_png, dpi=180); plt.close()

# ------------------ 主流程 ------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("读取 Excel…")
    dfx = read_excel_windows(EXCEL_PATH)
    target_tids: Set[str] = set(dfx[ID_COL].astype(str).str.zfill(4))

    print("从最佳路径库中仅提取这些编号的轨迹（复用可视化脚本）…")
    raw_tracks = iter_needed_segments(BESTTRACK_DIR, target_tids)

    print("逐小时插值（全生命史，复用可视化脚本）…")
    hourly = {tid: hourly_interp(pts) if pts else [] for tid, pts in raw_tracks.items()}

    print("构建影响段样本…")
    segments: List[ImpactSeg] = []
    P = PARAMS
    rows = []
    for idx, row in dfx.iterrows():
        tid = str(row[ID_COL]).zfill(4); st = pd.to_datetime(row[ST_COL]); en = pd.to_datetime(row[EN_COL])
        pts_all = hourly.get(tid, [])
        pts_win = [p for p in pts_all if st <= p.t <= en]
        coords = [(p.lon, p.lat) for p in pts_win]
        n = len(coords)
        arc = 0.0
        if n > 1:
            arc = seg_arclength(coords)
            
        participate = (n >= P["min_points_for_cluster"]) and (arc >= P["min_arclength_km"])
        segments.append(ImpactSeg(idx, tid, st, en, pts_win))
        rows.append({
            "idx": idx,
            "tid": tid,
            "start": st,
            "end": en,
            "found_in_lib": bool(pts_all),
            "win_points": n,
            "win_arclength_km": round(arc,1),
            "participate_cluster": participate,
        })

    rep_df = pd.DataFrame(rows)
    rep_path = os.path.join(OUT_DIR, "影响段_体检报告.csv")
    rep_df.to_csv(rep_path, index=False, encoding='utf-8-sig')
    print("→ 体检报告：", rep_path)

    print("向量化 & 聚类…")
    X, keep_idx, _ = vectorize_segments(segments, PARAMS)
    if X.shape[0] < min(PARAMS["k_candidates"]):
        print(f"样本不足 ({X.shape[0]} 个)，无法进行聚类。请检查 Excel 时段或放宽质控门槛。")
        map_path = os.path.join(OUT_DIR, "影响段_簇映射.csv")
        pd.DataFrame(columns=["idx","tid","cluster_id"]).to_csv(map_path, index=False, encoding='utf-8-sig')
        
        # 即使不聚类，也画一张图
        png_path = os.path.join(OUT_DIR, "影响段_路径簇_(样本不足).png")
        plot_clusters(segments, {}, 0, np.nan, png_path)
        print("→ 聚类成图（仅显示样本）：", png_path)
        return
        
    labels, k, score, model = choose_k_and_cluster(X, PARAMS["k_candidates"])
    print(f"聚类完成：选定 K={k}，轮廓系数={score:.3f}")

    idx2cid = {keep_idx[i]: int(labels[i]) for i in range(len(labels))}

    rows2 = [{"idx": seg.idx, "tid": seg.tid, "start": seg.start, "end": seg.end, "cluster_id": idx2cid.get(i, -1)} for i, seg in enumerate(segments)]
    map_path = os.path.join(OUT_DIR, "影响段_簇映射.csv")
    pd.DataFrame(rows2).to_csv(map_path, index=False, encoding='utf-8-sig')
    print("→ 簇映射：", map_path)

    png_path = os.path.join(OUT_DIR, "影响段_路径簇.png")
    plot_clusters(segments, idx2cid, k, score, png_path)
    print("→ 聚类成图：", png_path)

if __name__ == "__main__":
    main()