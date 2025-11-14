#!/usr/bin-env python3
# -*- coding: utf-8 -*-
"""
====================================================================
台风路径聚类 (HDBSCAN 专用版，使用绝对坐标特征)
====================================================================

[版本：新 - 独立版 (v3 - 已修复所有已知Bug)]

--- 核心功能 ---
本脚本用于对台风的“大风影响时段”路径进行自动聚类（分类）。
它使用 HDBSCAN 算法，该算法无需预先指定簇数，并能自动识别噪声点。

--- 关键技术点 ---
1.  **独立运行**: 本脚本已移除对所有外部 .py 脚本的依赖。所有必需的数据读取、
    解析和插值功能均由内置的 `_fallback_` 函数提供。
2.  **特征工程**: 聚类的核心特征是“重采样后的绝对坐标序列”。
    -   脚本首先提取逐小时的“影响时段”路径。
    -   然后使用 `resample_by_arclength` 函数将每条路径（无论长短）
        重采样为 N 个 (例如30个) 等距的点。
    -   最后，将这 N 个点的 `(lon, lat)` 坐标展平，形成一个长向量：
        `[lon1, lat1, lon2, lat2, ..., lonN, latN]`
    -   这个向量同时编码了路径的**绝对地理位置**和**形状**。
3.  **降维与聚类**:
    -   (可选) 使用 UMAP 对高维特征向量进行降维，有助于 HDBSCAN 发现结构。
    -   使用 HDBSCAN 进行密度聚类。
4.  **质量评估**:
    -   计算**轮廓系数 (Silhouette Score)** (基于UMAP降维后的空间) 来评估簇的“分离度”。
    -   计算**DBCV (Density-Based Cluster Validation)** (基于HDBSCAN的`relative_validity_`)
        来评估聚类结果的“稳定性”和“密度合理性”。

--- 输入 (Inputs) ---
1.  `CONFIG["EXCEL_PATH"]`:
    一个 Excel (.xlsx) 文件，必须包含台风编号、中文名、英文名以及
    大风影响的“开始时间”和“结束时间”。
2.  `CONFIG["BESTTRACK_DIR"]`:
    一个目录，存放着 CMA 格式的最佳路径数据（例如 `CH1990BST.txt`）。

--- 输出 (Outputs) ---
脚本将在 `CONFIG["OUT_DIR"]` 目录中生成三个文件：
1.  `影响段_体检报告.csv`:
    一个详细的质控(QC)报告，列出 Excel 中的每一条记录，
    并报告其路径点数、长度以及是否通过质控进入聚类。
2.  `影响段_簇映射.csv`:
    **核心结果**。将 Excel 中的每一行映射到一个 `cluster_id`。
    `cluster_id = -1` 表示该路径被识别为“噪声”。
3.  `影响段_路径簇_HDBSCAN.png`:
    一个可视化的聚类结果地图，不同颜色的路径代表不同的簇。

--- 依赖 (Dependencies) ---
  pip install scikit-learn hdbscan umap-learn pandas numpy matplotlib cartopy

--- 使用 (Usage) ---
  直接修改下方的 `CONFIG` 字典中的路径和参数，然后运行本文件。
"""

import os, re, math, glob, sys
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Set

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
try:
    # 尝试导入 Cartopy 用于专业地图绘制
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False # 如果导入失败，则回退到 Matplotlib 基础绘图

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 确保中文字体正确加载 (覆盖常见的中文字体名称)
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False # 正确显示负号


# 可选降维库
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

# 核心聚类库
try:
    import hdbscan
    HAS_HDB = True
except Exception:
    print("错误：未找到 hdbscan 库。请先运行: pip install hdbscan", file=sys.stderr)
    sys.exit(1)

# ------------------ CONFIG: 在此修改所有配置 ------------------
CONFIG = {
    # --- 1. 路径配置 ---
    "EXCEL_PATH": "/Users/momo/Desktop/业务相关/2025 影响台风大风/数据/2010_2024_影响台风_大风.xlsx",
    "BESTTRACK_DIR": "/Users/momo/Desktop/业务相关/2025 影响台风大风/热带气旋最佳路径数据集",
    # "VIS_SCRIPT_PATH": "...", # <-- [移除] 不再需要此配置
    "OUT_DIR": "/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_影响段聚类_HDBSCAN_优化版",

    # --- 2. Excel 列名 ---
    # 脚本将从 EXCEL_PATH 中读取这些列
    "ID_COL": "中央台编号",
    "ST_COL": "大风开始时间",
    "EN_COL": "大风结束时间",
    "CN_NAME_COL": "中文名称",   # (可选, 但推荐) 用于在终端打印噪声台风
    "EN_NAME_COL": "国外名称",   # (可选, 但推荐) 用于在终端打印噪声台风

    # --- 3. 地图范围 [lon_min, lon_max, lat_min, lat_max] ---
    "MAP_EXTENT": [105, 140, 15, 45], # 绘图时使用的地图范围

    # --- 4. 质控与特征工程 ---
    "RESAMPLE_N": 30,      # (特征工程) 将每条路径重采样为 N 个点
    "MIN_ARCLEN_KM": 30.0, # (质控) 路径总长小于此值(公里)的，视为噪声
    "MIN_POINTS": 3,       # (质控) 原始路径点数小于此值的，视为噪声

    # --- 5. HDBSCAN 与 UMAP 参数 (调优重点) ---
    "USE_UMAP": True,             # 是否在聚类前使用UMAP降维 (推荐)
    "UMAP_NEIGH": 15,             # UMAP: 邻域点数 (值越小越关注局部结构)
    "UMAP_MIN_DIST": 0.1,         # UMAP: 点之间的最小距离 (值越小簇越紧凑)
    "HDB_MIN_CLUSTER_SIZE": 5,    # HDBSCAN: 形成一个簇所需的最小路径数
    "HDB_MIN_SAMPLES": 2,         # HDBSCAN: 成为核心点的所需邻居数 (值越小越能容忍噪声)
    "RANDOM_STATE": 42            # 随机种子，确保 UMAP 和聚类结果可复现
}
# ------------------ CONFIG END ------------------


# ------------------ 地理工具 ------------------
EARTH_R = 6371.0 # 地球平均半径 (公里)

def haversine_km(lon1, lat1, lon2, lat2):
    """使用 Haversine (半正矢) 公式计算两个经纬度点之间的地球表面距离 (公里)"""
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return float(EARTH_R*c)


def slerp_lonlat(lon1, lat1, lon2, lat2, f):
    """
    球面线性插值 (Slerp)。
    在两个经纬度点之间按比例 f (0.0到1.0) 进行平滑插值。
    [注]: 这是来自 '台风路径可视化_插值小时.py' 的健壮版本，
           修复了原始 fallback 版本中的浮点数处理问题。
    """
    # 将经纬度转换为笛卡尔坐标 (x, y, z)
    a = np.deg2rad([lon1, lat1]); b = np.deg2rad([lon2, lat2])
    a = np.array([np.cos(a[1])*np.cos(a[0]), np.cos(a[1])*np.sin(a[0]), np.sin(a[1])])
    b = np.array([np.cos(b[1])*np.cos(b[0]), np.cos(b[1])*np.sin(b[0]), np.sin(b[1])])
    
    # 计算两个向量的点积
    dot = float(np.clip(np.dot(a,b), -1.0, 1.0))
    omega = math.acos(dot) # 两个向量之间的夹角
    
    # --- [关键修复] ---
    # 如果两个点几乎重合 (omega 极小)，直接返回起点，避免除零
    if omega < 1e-12:
        return float(lon1), float(lat1)
    
    so = math.sin(omega)
    # ----------------
    
    # 标准 Slerp 插值公式
    v = (math.sin((1-f)*omega)/so)*a + (math.sin(f*omega)/so)*b
    
    # 将插值后的笛卡尔坐标转换回经纬度
    lat = math.degrees(math.asin(np.clip(v[2], -1.0, 1.0)))
    lon = math.degrees(math.atan2(v[1], v[0]))
    return lon, lat

# ------------------ Fallback：读取/插值/Excel ------------------
#
# 本节包含了所有的数据读取、解析和插值功能。
# 这些函数是本脚本独立运行的保证。
#
# ---------------------------------------------------------------

# 正则表达式，用于解析 CMA 最佳路径文件
TIME_RE = re.compile(r"(\d{10})") # 匹配 YYYYMMDDHH (10位时间)
NUM_RE  = re.compile(r"[-+]?\d+\.?\d*") # 匹配所有数字 (包括小数和负数)

@dataclass
class BTPoint:
    """一个简单的数据结构，用于存储最佳路径上的一个点 (不含强度)"""
    t: pd.Timestamp
    lon: float
    lat: float

def _fallback_parse_header_tid(line: str, file_year: Optional[int]) -> Optional[str]:
    """解析 CMA 文件的 '66666' 头部行，提取台风编号 (TID)"""
    s = line.lstrip()
    if not s.startswith("66666"): return None
    # 查找所有4位数字
    toks = re.findall(r"\b(\d{4})\b", s)
    if not toks: return None
    # 优先：与文件年份的后两位匹配 (例如 2013 -> '13xx')
    if file_year is not None:
        yy = f"{file_year%100:02d}"
        for t in toks:
            if t.startswith(yy): return t
    # 兜底：返回从右到左第一个非 '0000' 的编号
    for t in reversed(toks):
        if t != "0000": return t
    return toks[-1] if toks else None

def _fallback_parse_record_line(line: str) -> Optional[BTPoint]:
    """解析 CMA 文件的路径数据行，提取 (时间, 经度, 纬度)"""
    s = line.strip()
    if not s or s.lstrip().startswith("66666"): return None
    
    # 1. 查找时间
    m = TIME_RE.search(s)
    if not m: return None
    ts = pd.to_datetime(m.group(1), format="%Y%m%d%H", errors="coerce")
    if pd.isna(ts): return None
    
    # 2. 查找时间戳之后的所有数字
    after = s[m.end():].strip()
    nums = [float(x) for x in NUM_RE.findall(after)]
    if len(nums) < 3: return None
    
    # 3. 提取经纬度 (通常在第2、3个数字，即索引1和2)
    lat_raw = nums[1]; lon_raw = nums[2]
    
    # 4. 处理“十分度” (例如 305 -> 30.5)
    if 0 <= abs(lat_raw) <= 900 and 0 <= abs(lon_raw) <= 1800 and (abs(lon_raw)>180 or abs(lat_raw)>90):
        lat, lon = lat_raw/10.0, lon_raw/10.0
    else:
        lat, lon = lat_raw, lon_raw
        
    # 5. 最终校验
    if not (-90 <= lat <= 90 and -180 <= lon <= 180): return None
    
    return BTPoint(ts, float(lon), float(lat))

def _fallback_iter_needed_segments(folder: str, target_tids: Set[str]) -> Dict[str, List[BTPoint]]:
    """
    遍历最佳路径目录，读取所有 .txt/.dat 文件，
    并只解析 `target_tids` 集合中指定的台风的路径点。
    返回一个字典：{tid -> [BTPoint, BTPoint, ...]}
    """
    out: Dict[str, List[BTPoint]] = {tid: [] for tid in target_tids}
    
    # 递归遍历目录
    for root, _, files in os.walk(folder):
        for fn in sorted(files):
            if not fn.lower().endswith((".txt",".dat")): continue
            
            path = os.path.join(root, fn)
            # 尝试从文件名获取年份提示，用于辅助解析TID
            m = re.search(r"(19|20)\d{2}", fn); file_year = int(m.group(0)) if m else None
            
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f: lines = f.readlines()
            except Exception: continue
            
            cur_tid: Optional[str] = None
            cur_pts: List[BTPoint] = []
            
            def flush():
                """辅助函数：将当前收集的点存入 out 字典"""
                nonlocal cur_tid, cur_pts
                if cur_tid and cur_pts and cur_tid in out: 
                    out[cur_tid].extend(cur_pts)
                cur_tid = None; cur_pts = []
            
            # 逐行解析文件
            for ln in lines:
                if ln.lstrip().startswith("66666"):
                    flush() # 遇到新头部，先“刷出”上一段
                    cur_tid = _fallback_parse_header_tid(ln, file_year)
                    continue
                
                # 如果当前TID不是我们想要的，跳过该行
                if cur_tid is None or cur_tid not in target_tids: continue
                
                p = _fallback_parse_record_line(ln)
                if p: cur_pts.append(p)
            
            flush() # 刷出文件末尾的最后一段
            
    # 合并与去重：处理同一台风分布在多个文件中的情况
    for tid, arr in out.items():
        if not arr: continue
        uniq = {p.t: p for p in arr} # 使用字典去重
        out[tid] = [uniq[t] for t in sorted(uniq.keys())] # 按时间排序
        
    return out

def _fallback_hourly_interp(points: List[BTPoint]) -> List[BTPoint]:
    """
    将稀疏的路径点 (例如6小时一次) 插值为逐小时路径。
    使用 `slerp_lonlat` 进行球面插值。
    """
    if not points: return []
    points = sorted(points, key=lambda x: x.t)
    times = [p.t for p in points]
    out: List[BTPoint] = []
    
    # 定义插值的时间范围 (从第一个点的整点到最后一个点的整点)
    cur = times[0].floor('h'); end = times[-1].ceil('h')
    
    idx = 0
    while cur <= end:
        # 找到当前 'cur' 时间点所在的原始路径段 (p0 -> p1)
        while idx+1 < len(times) and times[idx+1] < cur: idx += 1
        
        if cur <= times[0]:
            # 在第一个点之前：外推 (使用第一个点)
            p = points[0]; out.append(BTPoint(cur, p.lon, p.lat))
        elif cur >= times[-1]:
            # 在最后一个点之后：外推 (使用最后一个点)
            p = points[-1]; out.append(BTPoint(cur, p.lon, p.lat))
        else:
            # 在两个原始点之间：内插
            i = max(0, np.searchsorted(times, cur) - 1); j = i + 1
            t0, t1 = times[i], times[j]; p0, p1 = points[i], points[j]
            
            if t0 == t1:
                # 两个点时间相同 (不应发生, 但做保护)
                out.append(BTPoint(cur, p0.lon, p0.lat))
            else:
                # 计算插值因子 f
                f = float((cur - t0) / (t1 - t0)); f = float(np.clip(f, 0.0, 1.0))
                # 执行插值
                lon, lat = slerp_lonlat(p0.lon, p0.lat, p1.lon, p1.lat, f)
                out.append(BTPoint(cur, lon, lat))
                
        cur += pd.Timedelta(hours=1)
    return out

def _fallback_read_excel_windows(path: str, id_col: str, st_col: str, en_col: str, cn_name_col: str, en_name_col: str) -> pd.DataFrame:
    """
    [健壮版] 读取 Excel，包含可选的名称列。
    - 检查所有必需列 (ID, ST, EN) 是否存在。
    - 检查可选列 (CN_NAME, EN_NAME) 是否存在，如果不存在，则用 'N/A' 填充
      以确保后续代码能安全访问。
    """
    try:
        df = pd.read_excel(path)
    except FileNotFoundError:
        print(f"错误: Excel 文件未找到: {path}", file=sys.stderr)
        sys.exit(1)
        
    # 检查基本列
    required_cols = [id_col, st_col, en_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"错误: Excel 文件缺少必要列: {missing_cols}", file=sys.stderr)
        sys.exit(1)
        
    cols_to_read = required_cols
    
    # 检查并添加可选的名称列
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
    
    # 标准化数据
    df[id_col] = df[id_col].astype(str).str.strip().str.zfill(4) # 编号补全为4位
    df[st_col] = pd.to_datetime(df[st_col])
    df[en_col] = pd.to_datetime(df[en_col])
    
    # 填充可能存在的空名称
    df[cn_name_col] = df[cn_name_col].fillna("N/A")
    df[en_name_col] = df[en_name_col].fillna("N/A")
    
    # 丢弃无效数据并排序
    df = df.dropna(subset=[id_col, st_col, en_col]).sort_values([id_col, st_col]).reset_index(drop=True)
    return df

# ------------------ [移除] 绑定函数（不再需要） ------------------
# [移除] load_vis_module 函数被删除


# ------------------ 质控 / 重采样 / 向量化 ------------------
#
# 本节是特征工程的核心。
#
# -----------------------------------------------------------

@dataclass
class ImpactSeg:
    """数据结构：存储一个“影响段”的完整信息"""
    idx: int            # 在 Excel 中的原始索引
    tid: str            # 台风编号
    start: pd.Timestamp # 影响开始时间
    end: pd.Timestamp   # 影响结束时间
    pts: List[BTPoint]  # 该时段内的逐小时路径点
    cn_name: str        # 中文名
    en_name: str        # 英文名

def extract_impact_segment(hourly_pts: List[BTPoint], st: pd.Timestamp, en: pd.Timestamp) -> List[BTPoint]:
    """从逐小时路径(hourly_pts)中，根据开始(st)和结束(en)时间，切片出影响段"""
    return [p for p in hourly_pts if st <= p.t <= en]

def seg_arclength(coords: List[Tuple[float,float]]) -> float:
    """计算一条路径的总弧长 (公里)，用于质控"""
    s = 0.0
    for i in range(1, len(coords)):
        s += haversine_km(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1])
    return s

def resample_by_arclength(coords: List[Tuple[float,float]], N: int) -> np.ndarray:
    """
    【特征工程核心】按弧长重采样。
    将一条路径（无论长短）转换为 N 个等距的点。
    [注]: 这是修复了线性插值错误的健壮版本。
    """
    if len(coords) < 2:
        # 路径点太少，无法插值
        if coords:
            return np.repeat(np.asarray(coords[:1]), N, axis=0) # 返回 N 个重复的第1个点
        else:
             return np.full((N, 2), np.nan) # 返回 N 个 nan
             
    # 1. 计算路径上每个点到起点的累计距离
    d = [0.0]
    for i in range(1, len(coords)):
        d.append(d[-1] + haversine_km(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1]))
    total = d[-1] # 路径总长
    
    if total < 1e-6: 
        # 路径总长几乎为0 (例如，一个静止的点)
        return np.repeat(np.asarray(coords[:1]), N, axis=0)
        
    # 2. 定义 N 个目标采样点在总长度上的位置
    targets = np.linspace(0.0, total, N)
    
    # 3. 线性插值
    res = []
    j = 0 # 当前所在原始路径段的索引
    for s in targets: # s 是目标距离
        # 找到 s 所在的原始路径段 (j -> j+1)
        while j+1 < len(d) and d[j+1] < s: 
            j += 1
            
        if j+1 >= len(d):
            # 目标距离超出了路径总长 (浮点数误差)，返回最后一个点
            res.append(coords[-1])
        else:
            if (d[j+1] - d[j]) < 1e-9:
                # 原始路径段长度为0，返回前一个点
                res.append(coords[j])
            else:
                # 计算在当前段 (j -> j+1) 上的插值因子 f
                f = (s - d[j]) / (d[j+1] - d[j])
                
                # --- [关键修复] ---
                # 在 (lon[j], lat[j]) 和 (lon[j+1], lat[j+1]) 之间进行线性插值
                lon = coords[j][0] + f * (coords[j+1][0] - coords[j][0])
                lat = coords[j][1] + f * (coords[j+1][1] - coords[j][1])
                # ----------------
                
                res.append((lon,lat))
                
    return np.asarray(res) # 返回 (N, 2) 的数组

def vectorize_segments(segments: List[ImpactSeg], resample_N: int, min_points: int, min_arclen_km: float):
    """
    遍历所有“影响段”，执行质控(QC)和向量化。
    """
    X, keep_idx, curves = [], [], []
    
    for idx, seg in enumerate(segments):
        coords = [(p.lon, p.lat) for p in seg.pts]
        npts = len(coords)
        
        # --- 质控(QC) 1: 点数太少 ---
        if npts < min_points:
            continue
            
        # --- 质控(QC) 2: 弧长太短 ---
        arc = seg_arclength(coords)
        if arc < min_arclen_km:
            continue
            
        # --- 特征工程 ---
        # 1. 重采样为 N 个点
        curve = resample_by_arclength(coords, N=resample_N)  # (N, 2)
        
        # 2. 拉平 (Flatten)
        # 将 (N, 2) 数组 [ [lon1, lat1], [lon2, lat2], ... ]
        # 展平为 (N*2,) 向量 [lon1, lat1, lon2, lat2, ...]
        feature_vector = curve.reshape(-1)
        
        # 3. 质控(QC) 3: 检查重采样是否失败 (例如返回nan)
        if np.isnan(feature_vector).any():
            print(f"警告: 路径 idx={idx}, tid={seg.tid} 向量化失败，已跳过。")
            continue
            
        # 通过所有质控，加入数据集
        X.append(feature_vector)
        keep_idx.append(idx) # 记录这个向量对应 `segments` 列表中的原始索引
        curves.append(curve) # 存储 (N, 2) 路径，用于后续绘图（虽然本脚本没用上）
        
    if not X:
        return np.empty((0, resample_N * 2)), [], []
        
    # 将列表堆叠为 numpy 数组 (M, N*2)，其中 M 是通过质控的路径数
    return np.vstack(X), keep_idx, curves

# ------------------ 聚类器 ------------------

def run_hdbscan(X: np.ndarray, use_umap=True, umap_neigh=20, umap_min_dist=0.1,
                min_cluster_size=10, min_samples=7, random_state=42):
    """
    运行 HDBSCAN 聚类流程。
    """
    if not HAS_HDB:
        raise SystemExit("未安装 hdbscan，请先 pip install hdbscan")
        
    # 1. 标准化 (Standard Scaling)
    # 使每个特征 (即 lon1, lat1, lon2...) 的均值为0，方差为1
    # 这对于 UMAP 和 HDBSCAN (基于距离) 都非常重要
    scaler = StandardScaler(); 
    Xs = scaler.fit_transform(X)
    
    Z = Xs # Z 是用于聚类的最终数据
    reducer = None
    
    # 2. (可选) UMAP 降维
    if use_umap:
        if not HAS_UMAP:
            print("警告: 未安装 umap-learn (pip install umap-learn)，已跳过 UMAP 降维。")
        else:
            print(f"正在使用 UMAP 降维: n_neighbors={umap_neigh}, min_dist={umap_min_dist}")
            reducer = umap.UMAP(n_neighbors=umap_neigh, min_dist=umap_min_dist,
                                n_components=2, # 降到2维 (主要用于可视化和辅助聚类)
                                metric='euclidean', 
                                random_state=random_state)
            Z = reducer.fit_transform(Xs) # Z 变为 (M, 2) 数组
            
    # 3. 运行 HDBSCAN
    print(f"正在运行 HDBSCAN: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                metric='euclidean', 
                                prediction_data=True,
                                gen_min_span_tree=True) # 必须为 True 才能计算 DBCV
    
    labels = clusterer.fit_predict(Z)
    
    # --- 4. 计算评估指标 ---
    
    # 4a. 轮廓系数 (Silhouette Score)
    # 衡量簇的“分离度”。越接近1越好。
    # 注意：我们在 UMAP 降维后的空间 (Z) 中计算它。
    mask = labels >= 0 # 只选非噪声点
    silhouette_score_val = np.nan
    if mask.sum() > 0 and len(np.unique(labels[mask])) >= 2: # 必须至少有2个簇
        try:
            silhouette_score_val = float(silhouette_score(Z[mask], labels[mask]))
        except Exception as e:
            print(f"轮廓系数计算失败: {e}")
            
    # 4b. DBCV (Density-Based Cluster Validation)
    # 衡量密度聚类的“稳定性”。越接近1越好。
    # HDBSCAN 库内置了 DBCV，称为 "relative_validity_"
    dbcv_score_val = np.nan
    try:
        dbcv_score_val = clusterer.relative_validity_
    except Exception as e:
        print(f"DBCV (relative_validity_) 获取失败: {e}")
            
    return labels, clusterer, reducer, silhouette_score_val, dbcv_score_val

# ------------------ 成图与输出 ------------------

# 定义一组循环使用的颜色
CLUSTER_COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f", "#bcbd22", "#17becf"]

def plot_clusters(segments: List[ImpactSeg], labels_map: Dict[int,int], out_png: str, extent=None, title='大风影响段路径聚类'):
    """
    绘制聚类结果地图。
    - segments: 包含 *所有* 路径段 (包括未通过QC的)
    - labels_map: 一个字典 {seg_idx: cluster_id}，只包含 *通过QC* 的路径
    """
    if HAS_CARTOPY:
        # 使用 Cartopy 绘制带地图背景的图
        proj = ccrs.PlateCarree(); fig = plt.figure(figsize=(11,9)); ax = plt.axes(projection=proj)
        if extent: ax.set_extent(extent, crs=proj)
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
        ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
        ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    else:
        # 备用方案：使用 Matplotlib 基础绘图
        fig = plt.figure(figsize=(10,8)); ax = plt.gca()
        if extent: ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.grid(True, ls='--', alpha=0.4); ax.set_xlabel('Lon'); ax.set_ylabel('Lat')
        
    counts = {} # 统计每个簇的路径数
    
    # 遍历 *所有* 路径段
    for i, seg in enumerate(segments):
        # 从 labels_map 中获取该路径的 cluster_id
        # 如果 i 不在 map 中 (说明未通过QC)，或值为 -1 (HDBSCAN噪声)，则默认为 -1
        cid = labels_map.get(i, -1)
        
        if cid >= 0:
            # 属于某个簇
            color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
            lw = 2.2
            alpha = 0.95
        else:
            # 噪声或未通过QC
            color = "#cccccc"
            lw = 1.0
            alpha = 0.5
        
        xs = [p.lon for p in seg.pts]; ys = [p.lat for p in seg.pts]
        if not xs: continue # 跳过空路径
        
        if HAS_CARTOPY:
            ax.plot(xs, ys, '-', lw=lw, color=color, alpha=alpha, transform=ccrs.PlateCarree())
        else:
            ax.plot(xs, ys, '-', lw=lw, color=color, alpha=alpha)
            
        counts[cid] = counts.get(cid, 0) + 1
    
    # --- 创建图例 ---
    handles = []
    
    # 确保簇ID从0开始连续显示 (例如 0, 1, 2...)
    # 即使 HDBSCAN 返回的簇ID可能是 0, 2, 5
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
    plt.savefig(out_png, dpi=200) # 保存图像
    plt.close()
    print(f'→ 聚类成图：{out_png}')

# ------------------ 主流程 ------------------

def main():
    # 从 CONFIG 加载配置
    cfg = CONFIG
    os.makedirs(cfg["OUT_DIR"], exist_ok=True)
    
    # --- [核心修改] 绑定函数 (方案一：完全独立) ---
    print("...[方案一] 本脚本已独立，将使用内置的 'fallback' 函数。")
    
    # 定义一个 lambda 函数，用于调用 _fallback_read_excel_windows
    # 并传入所有在 CONFIG 中定义的列名参数
    fallback_loader = lambda p: _fallback_read_excel_windows(p, 
                                                             cfg["ID_COL"], 
                                                             cfg["ST_COL"], 
                                                             cfg["EN_COL"], 
                                                             cfg["CN_NAME_COL"], 
                                                             cfg["EN_NAME_COL"])

    # 永久绑定 Fallback 函数
    # 后续代码将直接使用这些变量，无需关心它们是来自外部还是内部
    iter_needed_segments = _fallback_iter_needed_segments
    hourly_interp        = _fallback_hourly_interp
    read_excel_windows   = fallback_loader
    
    print("...函数绑定完成。")


    # --- 步骤 1: 读取 Excel ---
    print(f'读取 Excel: {cfg["EXCEL_PATH"]} ...')
    dfx = read_excel_windows(cfg["EXCEL_PATH"])
    target_tids: Set[str] = set(dfx[cfg["ID_COL"]]) # 获取所有需要处理的台风ID

    # --- 步骤 2: 读取最佳路径 ---
    print(f'从 {cfg["BESTTRACK_DIR"]} 提取轨迹...')
    raw_tracks: Dict[str, List[BTPoint]] = iter_needed_segments(cfg["BESTTRACK_DIR"], target_tids)
    print(f"已识别到路径的台风个数：{sum(1 for k,v in raw_tracks.items() if v)} / {len(target_tids)}")

    # --- 步骤 3: 逐小时插值 ---
    print('逐小时插值...')
    hourly: Dict[str, List[BTPoint]] = {tid: hourly_interp(pts) if pts else [] for tid, pts in raw_tracks.items()}

    # --- 步骤 4: 构建影响段样本 ---
    print('构建影响段样本...')
    segments: List[ImpactSeg] = []
    # 遍历 Excel 中的 *每一行*
    for i, row in dfx.iterrows():
        tid = str(row[cfg["ID_COL"]])
        st  = pd.to_datetime(row[cfg["ST_COL"]])
        en  = pd.to_datetime(row[cfg["EN_COL"]])
        
        # 获取名称 (fallback 已处理空值和缺列)
        cn_name = str(row.get(cfg["CN_NAME_COL"], "N/A"))
        en_name = str(row.get(cfg["EN_NAME_COL"], "N/A"))
        
        # 获取该台风的完整逐小时路径
        pts_all = hourly.get(tid, [])
        # 从中切片出影响时段
        pts_win = extract_impact_segment(pts_all, st, en) if pts_all else []
        
        segments.append(ImpactSeg(
            idx=int(i), # 原始索引
            tid=tid, 
            start=st, 
            end=en, 
            pts=pts_win, # 切片后的路径
            cn_name=cn_name,
            en_name=en_name
        ))

    # --- 步骤 5: 生成体检报告 (QC Report) ---
    rows = []
    for seg in segments:
        coords = [(p.lon, p.lat) for p in seg.pts]
        n = len(coords); arc = seg_arclength(coords) if n>=2 else 0.0
        # 检查是否满足质控条件
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
            'participate_cluster': participate, # 标记是否会进入聚类
        })
    rep_df = pd.DataFrame(rows)
    rep_path = os.path.join(cfg["OUT_DIR"], '影响段_体检报告.csv')
    rep_df.to_csv(rep_path, index=False, encoding='utf-8-sig')
    print('→ 体检报告：', rep_path)

    # --- 步骤 6: 向量化 (QC + 特征工程) ---
    print('向量化...')
    # X: 特征矩阵 (M, N*2)
    # keep_idx: 通过QC的路径，它们在 `segments` 列表中的索引
    X, keep_idx, _ = vectorize_segments(segments, 
        cfg["RESAMPLE_N"], cfg["MIN_POINTS"], cfg["MIN_ARCLEN_KM"])
    
    print(f'共 {X.shape[0]} 条路径进入聚类。')

    # 样本太少，无法聚类
    if X.shape[0] < cfg["HDB_MIN_CLUSTER_SIZE"]:
        print(f'样本不足 (少于 min_cluster_size={cfg["HDB_MIN_CLUSTER_SIZE"]})，跳过聚类。')
        map_path = os.path.join(cfg["OUT_DIR"], '影响段_簇映射.csv')
        pd.DataFrame(columns=['idx','tid','cn_name','en_name','start','end','cluster_id']).to_csv(map_path, index=False, encoding='utf-8-sig')
        print('→ 簇映射：', map_path)
        png_path = os.path.join(cfg["OUT_DIR"], '影响段_路径簇_(样本不足).png')
        plot_clusters(segments, {}, png_path, extent=cfg["MAP_EXTENT"], title=f'大风影响段路径 (样本不足)')
        return

    # --- 步骤 7: 运行聚类 ---
    labels, model, reducer, s_score, dbcv_score = run_hdbscan(X,
        use_umap=cfg["USE_UMAP"], 
        umap_neigh=cfg["UMAP_NEIGH"], 
        umap_min_dist=cfg["UMAP_MIN_DIST"],
        min_cluster_size=cfg["HDB_MIN_CLUSTER_SIZE"], 
        min_samples=cfg["HDB_MIN_SAMPLES"],
        random_state=cfg["RANDOM_STATE"])
    
    k_eff = int(len(np.unique(labels[labels>=0]))) # 有效簇数 (不含噪声)
    noise_ratio = float((labels<0).sum()/labels.size) if labels.size > 0 else 0.0
    
    # 打印评估分数
    print(f"HDBSCAN：簇数(不含噪声)={k_eff}，轮廓(去噪)={s_score:.3f}，DBCV(稳定性)={dbcv_score:.3f}，噪声占比={noise_ratio:.2%}")

    # --- 步骤 8: 结果回填 & 打印噪声 ---
    # `labels` 是对 `X` (M条路径) 的标签
    # `idx2cid` 将 `segments` 列表的索引 (0到58) 映射到 cluster_id
    idx2cid = {keep_idx[i]: int(labels[i]) for i in range(len(labels))}

    print("\n--- 未进入有效聚类的台风 (唯一ID) ---")
    noise_typhoons = {} # 存储 (tid -> (cn_name, en_name))
    not_in_cluster_count = 0
    
    for i, seg in enumerate(segments):
        cluster_id = idx2cid.get(i, -1) # 'i' 是 segments 列表的索引
        
        if cluster_id == -1: # -1 = 噪声 或 未通过QC
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

    # --- 步骤 9: 保存最终输出 ---
    
    # 9a. 保存簇映射 CSV
    rows2 = [{
        'idx': seg.idx, 
        'tid': seg.tid, 
        'cn_name': seg.cn_name,
        'en_name': seg.en_name,
        'start': seg.start, 
        'end': seg.end,
        'cluster_id': idx2cid.get(i, -1) # 将 cluster_id 赋给Excel中的每一行
    } for i, seg in enumerate(segments)]
    
    map_path = os.path.join(cfg["OUT_DIR"], '影响段_簇映射.csv')
    pd.DataFrame(rows2).to_csv(map_path, index=False, encoding='utf-8-sig')
    print('→ 簇映射：', map_path)

    # 9b. 保存聚类地图
    title = (
        f'HDBSCAN (k={k_eff}, 噪声={noise_ratio:.1%}, 轮廓={s_score:.3f}, DBCV={dbcv_score:.3f})\n'
        f'(min_size={cfg["HDB_MIN_CLUSTER_SIZE"]}, min_samples={cfg["HDB_MIN_SAMPLES"]})'
    )
    png_path = os.path.join(cfg["OUT_DIR"], '影响段_路径簇_HDBSCAN.png')
    plot_clusters(segments, idx2cid, png_path, extent=cfg["MAP_EXTENT"], title=title)

if __name__ == '__main__':
    main()