#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
台风路径聚类 (HDBSCAN 专用版，使用绝对坐标特征 + 强度特征)

目的:
- 专门使用 HDBSCAN 算法对台风“影响时段”路径进行聚类。
- [新功能] 聚类特征向量 = “重采样坐标序列” + “时段内的平均/峰值强度”，
  使得聚类同时考虑【地理位置】、【形状】和【强度】。
- [新功能] 强制使用本脚本内置的 fallback 数据处理函数，以确保强度信息被正确读取和插值，
  忽略 'VIS_SCRIPT_PATH' 中可能存在的同名"精简版"函数。
- 打印出未被聚类（噪声/未通过质控）的台风名称。
- 计算并打印 DBCV (密度聚类有效性) 指标。

依赖:
  pip install scikit-learn hdbscan umap-learn pandas numpy matplotlib cartopy

使用:
  直接修改下方 CONFIG 字典中的路径和参数，然后运行本文件。
"""

import os, re, math, glob, types, importlib.util, hashlib, sys
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
from sklearn.metrics import silhouette_score

# 确保中文字体正确加载
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
    print("错误：未找到 hdbscan 库。请先运行: pip install hdbscan", file=sys.stderr)
    sys.exit(1)

# ------------------ CONFIG: 在此修改所有配置 ------------------
CONFIG = {
    # --- 1. 路径配置 ---
    # [MODIFIED] 修改输出目录名以反映新特性
    "EXCEL_PATH": "/Users/momo/Desktop/业务相关/2025 影响台风大风/2010_2024_影响台风_大风_去除圆规.xlsx",
    "BESTTRACK_DIR": "/Users/momo/Desktop/业务相关/2025 影响台风大风/热带气旋最佳路径数据集",
    "VIS_SCRIPT_PATH": "/Users/momo/Desktop/业务相关/2025 影响台风大风/代码/台风路径可视化_插值小时.py",
    "OUT_DIR": "/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_影响段聚类_HDBSCAN_v4_Coords_Intensity",

    # --- 2. Excel 列名 ---
    "ID_COL": "中央台编号",
    "ST_COL": "大风开始时间",
    "EN_COL": "大风结束时间",
    "CN_NAME_COL": "中文名称",
    "EN_NAME_COL": "国外名称",

    # --- 3. 地图范围 [lon_min, lon_max, lat_min, lat_max] ---
    "MAP_EXTENT": [105, 140, 15, 45],

    # --- 4. 质控与特征工程 ---
    "RESAMPLE_N": 30,      # 将每条路径重采样为 N 个点
    "MIN_ARCLEN_KM": 30.0, # 路径总长小于此值(公里)的，视为噪声
    "MIN_POINTS": 3,       # 原始路径点数小于此值的，视为噪声

    # --- 5. HDBSCAN 与 UMAP 参数 (调优重点) ---
    "USE_UMAP": True,
    "UMAP_NEIGH": 15,
    "UMAP_MIN_DIST": 0.1,
    "HDB_MIN_CLUSTER_SIZE": 5,
    "HDB_MIN_SAMPLES": 2,         
    "RANDOM_STATE": 42
}
# ------------------ CONFIG END ------------------


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
    omega = math.acos(dot); so = math.sin(omega) if omega>1e-9 else 1.0
    v = (math.sin((1-f)*omega)/so)*a + (math.sin(f*omega)/so)*b
    lat = math.degrees(math.asin(np.clip(v[2], -1.0, 1.0)))
    lon = math.degrees(math.atan2(v[1], v[0]))
    return lon, lat

# ------------------ Fallback：读取/插值/Excel ------------------
TIME_RE = re.compile(r"(\d{10})")
NUM_RE  = re.compile(r"[-+]?\d+\.?\d*")

# [MODIFIED] 数据结构增加强度
@dataclass
class BTPoint:
    t: pd.Timestamp
    lon: float
    lat: float
    intensity: float # 强度编码 (0-9)

def _fallback_parse_header_tid(line: str, file_year: Optional[int]) -> Optional[str]:
    s = line.lstrip()
    if not s.startswith("66666"): return None
    toks = re.findall(r"\b(\d{4})\b", s)
    if not toks: return None
    if file_year is not None:
        yy = f"{file_year%100:02d}"
        for t in toks:
            if t.startswith(yy): return t
    for t in reversed(toks):
        if t != "0000": return t
    return toks[-1] if toks else None

# [MODIFIED] 解析时增加强度
def _fallback_parse_record_line(line: str) -> Optional[BTPoint]:
    s = line.strip()
    if not s or s.lstrip().startswith("66666"): return None
    m = TIME_RE.search(s)
    if not m: return None
    ts = pd.to_datetime(m.group(1), format="%Y%m%d%H", errors="coerce")
    if pd.isna(ts): return None
    
    after = s[m.end():].strip()
    nums = [float(x) for x in NUM_RE.findall(after)]
    
    # 路径文件格式通常是:
    # YYYYMMDDHH (I) (LAT*10) (LON*10) (PRES) (WND)
    # nums[0] = I (强度编码)
    # nums[1] = LAT*10
    # nums[2] = LON*10
    if len(nums) < 3: return None 
    
    intensity_code = nums[0]
    lat_raw = nums[1]
    lon_raw = nums[2]
    
    if 0 <= abs(lat_raw) <= 900 and 0 <= abs(lon_raw) <= 1800 and (abs(lon_raw)>180 or abs(lat_raw)>90):
        lat, lon = lat_raw/10.0, lon_raw/10.0
    else:
        lat, lon = lat_raw, lon_raw
        
    if not (-90 <= lat <= 90 and -180 <= lon <= 180): return None
    
    return BTPoint(ts, float(lon), float(lat), float(intensity_code)) # 返回包含强度的点

# [MODIFIED] iter_needed_segments 现在会返回 List[BTPoint]，其中 BTPoint 包含强度
def _fallback_iter_needed_segments(folder: str, target_tids: Set[str]) -> Dict[str, List[BTPoint]]:
    out: Dict[str, List[BTPoint]] = {tid: [] for tid in target_tids}
    for root, _, files in os.walk(folder):
        for fn in sorted(files):
            if not fn.lower().endswith((".txt",".dat")): continue
            path = os.path.join(root, fn)
            m = re.search(r"(19|20)\d{2}", fn); file_year = int(m.group(0)) if m else None
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f: lines = f.readlines()
            except Exception: continue
            cur_tid: Optional[str] = None
            cur_pts: List[BTPoint] = []
            def flush():
                nonlocal cur_tid, cur_pts
                if cur_tid and cur_pts and cur_tid in out: out[cur_tid].extend(cur_pts)
                cur_tid = None; cur_pts = []
            for ln in lines:
                if ln.lstrip().startswith("66666"):
                    flush(); cur_tid = _fallback_parse_header_tid(ln, file_year); continue
                if cur_tid is None or cur_tid not in target_tids: continue
                
                # [MODIFIED] parse_record_line 现在会返回带强度的点
                p = _fallback_parse_record_line(ln) 
                if p: cur_pts.append(p)
            flush()
    for tid, arr in out.items():
        if not arr: continue
        uniq = {p.t: p for p in arr}
        out[tid] = [uniq[t] for t in sorted(uniq.keys())]
    return out

# [MODIFIED] 插值时增加强度（使用"时间最近点"逻辑）
def _fallback_hourly_interp(points: List[BTPoint]) -> List[BTPoint]:
    if not points: return []
    points = sorted(points, key=lambda x: x.t)
    times = [p.t for p in points]
    out: List[BTPoint] = []
    cur = times[0].floor('h'); end = times[-1].ceil('h')
    idx = 0
    while cur <= end:
        while idx+1 < len(times) and times[idx+1] < cur: idx += 1
        
        if cur <= times[0]:
            p = points[0]
            out.append(BTPoint(cur, p.lon, p.lat, p.intensity))
            
        elif cur >= times[-1]:
            p = points[-1]
            out.append(BTPoint(cur, p.lon, p.lat, p.intensity))
            
        else:
            i = max(0, np.searchsorted(times, cur) - 1); j = i + 1
            t0, t1 = times[i], times[j]; p0, p1 = points[i], points[j]
            
            if t0 == t1:
                out.append(BTPoint(cur, p0.lon, p0.lat, p0.intensity))
            else:
                # 经纬度使用球面线性插值
                f = float((cur - t0) / (t1 - t0)); f = float(np.clip(f, 0.0, 1.0))
                lon, lat = slerp_lonlat(p0.lon, p0.lat, p1.lon, p1.lat, f)
                
                # [NEW] 强度使用"时间最近点"逻辑 (f < 0.5 说明离 p0 更近)
                intensity = p0.intensity if f < 0.5 else p1.intensity
                
                out.append(BTPoint(cur, lon, lat, intensity))
                
        cur += pd.Timedelta(hours=1)
    return out

def _fallback_read_excel_windows(path: str, id_col: str, st_col: str, en_col: str, cn_name_col: str, en_name_col: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(path)
    except FileNotFoundError:
        print(f"错误: Excel 文件未找到: {path}", file=sys.stderr)
        sys.exit(1)
        
    required_cols = [id_col, st_col, en_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"错误: Excel 文件缺少必要列: {missing_cols}", file=sys.stderr)
        sys.exit(1)
        
    cols_to_read = required_cols
    
    if cn_name_col in df.columns:
        cols_to_read.append(cn_name_col)
    else:
        print(f"警告: Excel 中未找到中文名列 '{cn_name_col}'，将使用 'N/A' 填充。")
        df[cn_name_col] = "N/A"
        
    if en_name_col in df.columns:
        cols_to_read.append(en_name_col)
    else:
        print(f"警告: Excel 中未找到英文名列 '{en_name_col}'，将使用 'N/A' 填充。")
        df[en_name_col] = "N/A"
        
    df = df[cols_to_read].copy()
    
    df[id_col] = df[id_col].astype(str).str.strip().str.zfill(4)
    df[st_col] = pd.to_datetime(df[st_col])
    df[en_col] = pd.to_datetime(df[en_col])
    
    df[cn_name_col] = df[cn_name_col].fillna("N/A")
    df[en_name_col] = df[en_name_col].fillna("N/A")
    
    df = df.dropna(subset=[id_col, st_col, en_col]).sort_values([id_col, st_col]).reset_index(drop=True)
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
    pts: List[BTPoint]  # [MODIFIED] pts 列表中现在包含了强度
    cn_name: str
    en_name: str

def extract_impact_segment(hourly_pts: List[BTPoint], st: pd.Timestamp, en: pd.Timestamp) -> List[BTPoint]:
    # hourly_pts 是 List[BTPoint]，BTPoint 包含强度
    return [p for p in hourly_pts if st <= p.t <= en]

def seg_arclength(coords: List[Tuple[float,float]]) -> float:
    s = 0.0
    for i in range(1, len(coords)):
        s += haversine_km(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1])
    return s

def resample_by_arclength(coords: List[Tuple[float,float]], N: int) -> np.ndarray:
    if len(coords) < 2:
        if coords:
            return np.repeat(np.asarray(coords[:1]), N, axis=0)
        else:
             return np.full((N, 2), np.nan) 
             
    d = [0.0]
    for i in range(1, len(coords)):
        d.append(d[-1] + haversine_km(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1]))
    total = d[-1]
    
    if total < 1e-6: 
        return np.repeat(np.asarray(coords[:1]), N, axis=0)
        
    targets = np.linspace(0.0, total, N)
    res = []
    j = 0
    for s in targets:
        while j+1 < len(d) and d[j+1] < s: 
            j += 1
        if j+1 >= len(d):
            res.append(coords[-1])
        else:
            if (d[j+1] - d[j]) < 1e-9:
                res.append(coords[j])
            else:
                f = (s - d[j]) / (d[j+1] - d[j])
                lon = coords[j][0] + f * (coords[j+1][0] - coords[j][0])
                lat = coords[j][1] + f * (coords[j+1][1] - coords[j][1])
                res.append((lon,lat))
                
    return np.asarray(res)

# [MODIFIED] 向量化函数 (核心修改)
def vectorize_segments(segments: List[ImpactSeg], resample_N: int, min_points: int, min_arclen_km: float):
    """
    特征向量 = [坐标序列] + [强度统计特征]
    """
    X, keep_idx, curves_for_plot = [], [], []
    for idx, seg in enumerate(segments):
        
        # 1. 质控
        npts = len(seg.pts)
        if npts < min_points:
            continue
            
        coords = [(p.lon, p.lat) for p in seg.pts]
        arc = seg_arclength(coords)
        if arc < min_arclen_km:
            continue
            
        # 2. 坐标特征 (例如 60维)
        curve = resample_by_arclength(coords, N=resample_N)  # (N, 2)
        path_feature_vector = curve.reshape(-1) # 长度 N*2
        
        if np.isnan(path_feature_vector).any():
            print(f"警告: 路径 idx={idx}, tid={seg.tid} 坐标向量化失败，已跳过。")
            continue
            
        # 3. [NEW] 强度特征 (2维)
        #    提取该影响时段内的强度列表
        segment_intensities = [p.intensity for p in seg.pts]
        
        if not segment_intensities:
            avg_intensity = 0.0
            max_intensity = 0.0
        else:
            avg_intensity = np.mean(segment_intensities)
            max_intensity = np.max(segment_intensities)
            
        intensity_features = np.array([avg_intensity, max_intensity])

        # 4. [NEW] 组合特征 (例如 60 + 2 = 62维)
        full_feature_vector = np.concatenate([path_feature_vector, intensity_features])
        
        X.append(full_feature_vector)
        keep_idx.append(idx)
        # 绘图时我们仍然只需要坐标
        curves_for_plot.append(curve) 
        
    if not X:
        # [MODIFIED] 维度变为 N*2 + 2
        return np.empty((0, resample_N * 2 + 2)), [], []
        
    return np.vstack(X), keep_idx, curves_for_plot

# ------------------ 聚类器 ------------------

def run_hdbscan(X: np.ndarray, use_umap=True, umap_neigh=20, umap_min_dist=0.1,
                min_cluster_size=10, min_samples=7, random_state=42):
    if not HAS_HDB:
        raise SystemExit("未安装 hdbscan，请先 pip install hdbscan")
        
    # [MODIFIED] StandardScaler 现在会处理 N*2 + 2 维向量
    scaler = StandardScaler(); 
    Xs = scaler.fit_transform(X)
    
    Z = Xs
    reducer = None
    
    if use_umap:
        if not HAS_UMAP:
            print("警告: 未安装 umap-learn (pip install umap-learn)，已跳过 UMAP 降维。")
        else:
            print(f"正在使用 UMAP 降维: n_neighbors={umap_neigh}, min_dist={umap_min_dist}")
            # [MODIFIED] UMAP 将在 N*2 + 2 维空间上工作
            reducer = umap.UMAP(n_neighbors=umap_neigh, min_dist=umap_min_dist,
                                n_components=2, metric='euclidean', random_state=random_state)
            Z = reducer.fit_transform(Xs)
            
    print(f"正在运行 HDBSCAN: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                metric='euclidean', 
                                prediction_data=True,
                                gen_min_span_tree=True)
    
    labels = clusterer.fit_predict(Z)
    
    mask = labels >= 0
    silhouette_score_val = np.nan
    if mask.sum() > 0 and len(np.unique(labels[mask])) >= 2:
        try:
            silhouette_score_val = float(silhouette_score(Z[mask], labels[mask]))
        except Exception as e:
            print(f"轮廓系数计算失败: {e}")
            
    dbcv_score_val = np.nan
    try:
        dbcv_score_val = clusterer.relative_validity_
    except Exception as e:
        print(f"DBCV (relative_validity_) 获取失败: {e}")
            
    return labels, clusterer, reducer, silhouette_score_val, dbcv_score_val

# ------------------ 成图与输出 ------------------
CLUSTER_COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f", "#bcbd22", "#17becf"]

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
        cid = labels_map.get(i, -1) # -1 = 噪声或未参与
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)] if cid >= 0 else "#cccccc"
        lw = 2.2 if cid >= 0 else 1.0
        alpha = 0.95 if cid >= 0 else 0.5
        
        # 绘图仍然只使用坐标
        xs = [p.lon for p in seg.pts]; ys = [p.lat for p in seg.pts]
        if not xs: continue
        
        if HAS_CARTOPY:
            ax.plot(xs, ys, '-', lw=lw, color=color, alpha=alpha, transform=ccrs.PlateCarree())
        else:
            ax.plot(xs, ys, '-', lw=lw, color=color, alpha=alpha)
        counts[cid] = counts.get(cid, 0) + 1
    
    handles = []
    unique_clusters = sorted([c for c in counts if c >= 0])
    cluster_map = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}

    for old_id, new_id in cluster_map.items():
         handles.append(plt.Line2D([0],[0], color=CLUSTER_COLORS[new_id % len(CLUSTER_COLORS)], lw=2, label=f'簇 {new_id} (N={counts.get(old_id, 0)})'))
    if -1 in counts:
         handles.append(plt.Line2D([0],[0], color="#cccccc", lw=1, label=f'噪声/未参与 (N={counts[-1]})'))
    ax.legend(handles=handles, title="聚类结果", loc="lower left", fontsize=9)
    
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f'→ 聚类成图：{out_png}')

# ------------------ 主流程 ------------------

def main():
    # 从 CONFIG 加载配置
    cfg = CONFIG
    os.makedirs(cfg["OUT_DIR"], exist_ok=True)
    
    # --- [MODIFIED] 绑定函数 (强制使用本脚本的 fallback) ---
    fallback_loader = lambda p: _fallback_read_excel_windows(p, 
                                                             cfg["ID_COL"], 
                                                             cfg["ST_COL"], 
                                                             cfg["EN_COL"], 
                                                             cfg["CN_NAME_COL"], 
                                                             cfg["EN_NAME_COL"])
    
    print(f'加载可视化脚本函数从: {cfg["VIS_SCRIPT_PATH"]} ...')
    try:
        # 仍然加载 VIS 模块，以防万一它提供了其他辅助函数
        VIS = load_vis_module(cfg["VIS_SCRIPT_PATH"])
        print("...模块加载成功。")
    except Exception as e:
        print(f"警告：无法从 {cfg['VIS_SCRIPT_PATH']} 加载函数 ({e})。")

    # [NEW] 强制使用本脚本内置的、支持强度的函数
    print("...[强制] 使用本脚本内置的 fallback 函数 (iter_needed_segments, hourly_interp) 以确保强度被处理。")
    iter_needed_segments = _fallback_iter_needed_segments
    hourly_interp        = _fallback_hourly_interp
    read_excel_windows   = fallback_loader # read_excel 总是用 fallback
    
    print(f'读取 Excel: {cfg["EXCEL_PATH"]} ...')
    dfx = read_excel_windows(cfg["EXCEL_PATH"])
    target_tids: Set[str] = set(dfx[cfg["ID_COL"]])

    print(f'从 {cfg["BESTTRACK_DIR"]} 提取轨迹...')
    raw_tracks: Dict[str, List[BTPoint]] = iter_needed_segments(cfg["BESTTRACK_DIR"], target_tids)
    print(f"已识别到路径的台风个数：{sum(1 for k,v in raw_tracks.items() if v)} / {len(target_tids)}")

    print('逐小时插值 (包含强度)...')
    hourly: Dict[str, List[BTPoint]] = {tid: hourly_interp(pts) if pts else [] for tid, pts in raw_tracks.items()}

    print('构建影响段样本...')
    segments: List[ImpactSeg] = []
    for i, row in dfx.iterrows():
        tid = str(row[cfg["ID_COL"]])
        st  = pd.to_datetime(row[cfg["ST_COL"]])
        en  = pd.to_datetime(row[cfg["EN_COL"]])
        
        cn_name = str(row.get(cfg["CN_NAME_COL"], "N/A"))
        en_name = str(row.get(cfg["EN_NAME_COL"], "N/A"))
        
        pts_all = hourly.get(tid, [])
        # [MODIFIED] pts_win 现在是 List[BTPoint]，且 BTPoint 包含强度
        pts_win = extract_impact_segment(pts_all, st, en) if pts_all else [] 
        segments.append(ImpactSeg(
            idx=int(i), 
            tid=tid, 
            start=st, 
            end=en, 
            pts=pts_win,
            cn_name=cn_name,
            en_name=en_name
        ))

    # 体检报告
    rows = []
    for seg in segments:
        coords = [(p.lon, p.lat) for p in seg.pts]
        n = len(coords); arc = seg_arclength(coords) if n>=2 else 0.0
        participate = (n >= cfg["MIN_POINTS"] and arc >= cfg["MIN_ARCLEN_KM"])
        rows.append({
            'idx': seg.idx, 
            'tid': seg.tid,
            'cn_name': seg.cn_name,
            'en_name': seg.en_name,
            'start': seg.start, 
            'end': seg.end,
            'found_in_lib': bool(hourly.get(seg.tid)),
            'win_points': n,
            'win_arclength_km': round(arc, 1),
            'participate_cluster': participate,
        })
    rep_df = pd.DataFrame(rows)
    rep_path = os.path.join(cfg["OUT_DIR"], '影响段_体检报告.csv')
    rep_df.to_csv(rep_path, index=False, encoding='utf-8-sig')
    print('→ 体检报告：', rep_path)

    # [MODIFIED] 向量化 (包含强度)
    print('向量化 (包含坐标与强度)...')
    X, keep_idx, _ = vectorize_segments(segments, 
        cfg["RESAMPLE_N"], cfg["MIN_POINTS"], cfg["MIN_ARCLEN_KM"])
    
    print(f'共 {X.shape[0]} 条路径进入聚类 (特征维度: {X.shape[1]})。') # [MODIFIED] 打印维度

    if X.shape[0] < cfg["HDB_MIN_CLUSTER_SIZE"]:
        print(f'样本不足 (少于 min_cluster_size={cfg["HDB_MIN_CLUSTER_SIZE"]})，跳过聚类。')
        map_path = os.path.join(cfg["OUT_DIR"], '影响段_簇映射.csv')
        pd.DataFrame(columns=['idx','tid','cn_name','en_name','start','end','cluster_id']).to_csv(map_path, index=False, encoding='utf-8-sig')
        print('→ 簇映射：', map_path)
        png_path = os.path.join(cfg["OUT_DIR"], '影响段_路径簇_(样本不足).png')
        plot_clusters(segments, {}, png_path, extent=cfg["MAP_EXTENT"], title=f'大风影响段路径 (样本不足)')
        return

    # 运行 HDBSCAN (接收 dbcv_score)
    labels, model, reducer, s_score, dbcv_score = run_hdbscan(X,
        use_umap=cfg["USE_UMAP"], 
        umap_neigh=cfg["UMAP_NEIGH"], 
        umap_min_dist=cfg["UMAP_MIN_DIST"],
        min_cluster_size=cfg["HDB_MIN_CLUSTER_SIZE"], 
        min_samples=cfg["HDB_MIN_SAMPLES"],
        random_state=cfg["RANDOM_STATE"])
    
    k_eff = int(len(np.unique(labels[labels>=0]))) if (labels.size>0) else 0
    noise_ratio = float((labels<0).sum()/labels.size) if labels.size > 0 else 0
    
    print(f"HDBSCAN：簇数(不含噪声)={k_eff}，轮廓(去噪)={s_score:.3f}，DBCV(稳定性)={dbcv_score:.3f}，噪声占比={noise_ratio:.2%}")

    # 回填映射 & 打印噪声台风
    idx2cid = {keep_idx[i]: int(labels[i]) for i in range(len(labels))}

    print("\n--- 未进入有效聚类的台风 (唯一ID) ---")
    noise_typhoons = {} 
    not_in_cluster_count = 0
    
    for i, seg in enumerate(segments):
        cluster_id = idx2cid.get(i, -1)
        
        if cluster_id == -1:
            not_in_cluster_count += 1
            if seg.tid not in noise_typhoons:
                noise_typhoons[seg.tid] = (seg.cn_name, seg.en_name)
    
    if not noise_typhoons:
        print("  所有台风均已成功聚类。")
    else:
        for tid, (cn, en) in noise_typhoons.items():
            print(f"  - 编号: {tid}, 中文名: {cn}, 英文名: {en}")
        print(f"\n  (总计 {not_in_cluster_count} 个路径段被标记为噪声或未通过质控，")
        print(f"   涉及 {len(noise_typhoons)} 个唯一台风编号。)")
    print("------------------------------------\n")

    # rows2 包含名称，用于保存CSV
    rows2 = [{
        'idx': seg.idx, 
        'tid': seg.tid, 
        'cn_name': seg.cn_name,
        'en_name': seg.en_name,
        'start': seg.start, 
        'end': seg.end,
        'cluster_id': idx2cid.get(i, -1)
    } for i, seg in enumerate(segments)]
    
    map_path = os.path.join(cfg["OUT_DIR"], '影响段_簇映射.csv')
    pd.DataFrame(rows2).to_csv(map_path, index=False, encoding='utf-8-sig')
    print('→ 簇映射：', map_path)

    # [MODIFIED] 成图 (标题包含分数和新特征)
    title = (
        f'HDBSCAN (k={k_eff}, 噪声={noise_ratio:.1%}, 轮廓={s_score:.3f}, DBCV={dbcv_score:.3f})\n'
        f'[特征: 路径坐标(N={cfg["RESAMPLE_N"]}) + 强度(均值/峰值)]'
    )
    png_path = os.path.join(cfg["OUT_DIR"], '影响段_路径簇_HDBSCAN.png')
    plot_clusters(segments, idx2cid, png_path, extent=cfg["MAP_EXTENT"], title=title)

    
    

if __name__ == '__main__':
    main()