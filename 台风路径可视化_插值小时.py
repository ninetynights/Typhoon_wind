"""
台风路径可视化（逐小时插值，仅处理 Excel 中列出的台风 + 影响段加粗）

本机路径：
  BESTTRACK_DIR = "/Users/momo/Desktop/业务相关/2025 影响台风大风/热带气旋最佳路径数据集"
  EXCEL_PATH    = "/Users/momo/Desktop/业务相关/2025 影响台风大风/2010_2024_影响台风_大风.xlsx"
输出：Excel 同级目录下 “输出_逐小时仅Excel/”。
"""
from __future__ import annotations
import os, re, math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Iterable, Set

import numpy as np
import pandas as pd

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

# ------------------ 路径 ------------------
BESTTRACK_DIR = "/Users/momo/Desktop/业务相关/2025 影响台风大风/热带气旋最佳路径数据集"
EXCEL_PATH    = "/Users/momo/Desktop/业务相关/2025 影响台风大风/2010_2024_影响台风_大风.xlsx"
OUTPUT_DIR    = os.path.join(os.path.dirname(EXCEL_PATH), "输出_逐小时仅Excel")

# Excel 列名
ID_COL = "中央台编号"
ST_COL = "大风开始时间"
EN_COL = "大风结束时间"

# 地图范围（浙江近海）
EXTENT = [110.0, 135.0, 15.0, 40.0]

# 颜色与线宽
COLOR_ALL = "#888888"     # 全轨迹底色（仅对被引用到的编号绘制）
COLOR_HL  = "#d62728"     # 影响段加粗色
LW_ALL    = 0.9
LW_HL     = 2.8

# 不确定性圈设置（用于“停止编号/消散后仍有影响”的个例）
VMAX_KMH       = 40.0
UNCERT_RMAX_KM = 300.0

# ------------------ 数据结构 ------------------
@dataclass
class BTPoint:
    t: pd.Timestamp
    lon: float
    lat: float

# ------------------ 基础工具 ------------------
EARTH_R = 6371.0
TIME_RE = re.compile(r"(\d{10})")
NUM_RE  = re.compile(r"[-+]?\d+\.?\d*")

def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
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


def hourly_range(st: pd.Timestamp, en: pd.Timestamp) -> List[pd.Timestamp]:
    return list(pd.date_range(st, en, freq="1h"))

# ------------------ 只取需要的编号：扫描命中即收 ------------------

def parse_header_tid(line: str, file_year: Optional[int]) -> Optional[str]:
    s = line.lstrip()
    if not s.startswith("66666"):
        return None
    toks = re.findall(r"\b(\d{4})\b", s)  # 抓出所有4位数字
    if not toks:
        return None
    # 优先：与文件年份后两位匹配（2014 -> '14**'）
    if file_year is not None:
        yy = f"{file_year%100:02d}"
        for t in toks:
            if t.startswith(yy):
                return t
    # 兜底：自右向左第一个非'0000'
    for t in reversed(toks):
        if t != "0000":
            return t
    return toks[-1]


def parse_record_line(line: str) -> Optional[BTPoint]:
    s = line.strip()
    if not s or s.lstrip().startswith("66666"):
        return None
    m = TIME_RE.search(s)
    if not m:
        return None
    ts = pd.to_datetime(m.group(1), format="%Y%m%d%H", errors="coerce")
    if pd.isna(ts):
        return None

    # 取出时间戳后的数字序列
    # 形如：2014011706 1  98 1279 1006  13
    after = s[m.end():].strip()
    nums = [float(x) for x in NUM_RE.findall(after)]
    if len(nums) < 3:
        return None
    # 一般顺序：状态码、纬(十分度)、经(十分度)...
    status = nums[0]
    lat_raw = nums[1]
    lon_raw = nums[2]

    # 判别“十分度”并转成“度”
    def to_deg(latv, lonv):
        # 典型：纬 0-900，经 0-1800（十分度）
        if 0 <= abs(latv) <= 900 and 0 <= abs(lonv) <= 1800 and (abs(lonv) > 180 or abs(latv) > 90):
            return latv/10.0, lonv/10.0
        # 已经是度
        return latv, lonv

    lat, lon = to_deg(lat_raw, lon_raw)

    # 最后再做一次合理性检查
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return BTPoint(ts, float(lon), float(lat))


def iter_needed_segments(folder: str, target_tids: Set[str]) -> Dict[str, List[BTPoint]]:
    """只返回 Excel 指定的 tid → 点序列。扫描到匹配段即收集，尽量少读。"""
    out: Dict[str, List[BTPoint]] = {tid: [] for tid in target_tids}
    left = set(target_tids)
    for root, _, files in os.walk(folder):
        if not left:
            break
        for fn in sorted(files):
            if not left:
                break
            if not fn.lower().endswith((".txt",".dat")):
                continue
            path = os.path.join(root, fn)
            # 年份提示
            m = re.search(r"(19|20)\d{2}", fn)
            file_year = int(m.group(0)) if m else None
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
            except Exception:
                continue
            cur_tid: Optional[str] = None
            cur_pts: List[BTPoint] = []
            def flush():
                nonlocal cur_tid, cur_pts, out, left
                if cur_tid and cur_pts and cur_tid in out:
                    out[cur_tid].extend(cur_pts)
                cur_tid, cur_pts = None, []
            for ln in lines:
                if ln.lstrip().startswith("66666"):
                    # 进入新段前，把上一段落盘
                    flush()
                    cur_tid = parse_header_tid(ln, file_year)
                    # 只保留目标编号，其它编号直接跳过但仍需推进到下一段
                    continue
                if cur_tid and cur_tid in left:
                    p = parse_record_line(ln)
                    if p:
                        cur_pts.append(p)
            flush()
            # 如果有编号已收集到点，检查是否还能继续匹配其它文件（不同文件可能有同编号补点）
            for tid in list(left):
                if out.get(tid):
                    # 只要出现过点，就先保留；不立即从 left 移除，允许跨文件追加
                    pass
    # 合并去重排序
    for tid in list(out.keys()):
        arr = out[tid]
        if not arr:
            continue
        uniq = {p.t: p for p in arr}
        out[tid] = [uniq[t] for t in sorted(uniq.keys())]
    return out

# ------------------ 逐小时插值（仅对命中的编号） ------------------

def hourly_interp(points: List[BTPoint]) -> List[BTPoint]:
    if not points:
        return []
    points = sorted(points, key=lambda x: x.t)
    times = [p.t for p in points]
    out: List[BTPoint] = []
    cur = times[0].floor('h')
    end = times[-1].ceil('h')
    idx = 0
    while cur <= end:
        while idx+1 < len(times) and times[idx+1] < cur:
            idx += 1
        if cur <= times[0]:
            p = points[0]; out.append(BTPoint(cur, p.lon, p.lat))
        elif cur >= times[-1]:
            p = points[-1]; out.append(BTPoint(cur, p.lon, p.lat))
        else:
            i = max(0, np.searchsorted(times, cur) - 1)
            j = i + 1
            t0, t1 = times[i], times[j]
            p0, p1 = points[i], points[j]
            if t0 == t1:
                out.append(BTPoint(cur, p0.lon, p0.lat))
            else:
                f = (cur - t0) / (t1 - t0)
                f = float(np.clip(f, 0.0, 1.0))
                lon, lat = slerp_lonlat(p0.lon, p0.lat, p1.lon, p1.lat, f)
                out.append(BTPoint(cur, lon, lat))
        cur += pd.Timedelta(hours=1)
    return out

# ------------------ Excel 读取 ------------------

def read_excel_windows(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df[[ID_COL, ST_COL, EN_COL]].copy()
    df[ID_COL] = df[ID_COL].astype(str).str.strip().str.zfill(4)
    df[ST_COL] = pd.to_datetime(df[ST_COL])
    df[EN_COL] = pd.to_datetime(df[EN_COL])
    df = df.dropna().sort_values([ID_COL, ST_COL])
    return df

# ------------------ 绘图 ------------------

def plot_selected(tracks_hourly: Dict[str, List[BTPoint]], excel_df: pd.DataFrame, out_png: str, report_csv: str):
    if HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(11, 9))
        ax = plt.axes(projection=proj)
        ax.set_extent(EXTENT, crs=proj)
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
        ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
        ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    else:
        fig = plt.figure(figsize=(10, 8)); ax = plt.gca()
        ax.set_xlim(EXTENT[0], EXTENT[1]); ax.set_ylim(EXTENT[2], EXTENT[3])
        ax.grid(True, ls='--', alpha=0.4); ax.set_xlabel('Lon'); ax.set_ylabel('Lat')

    # 先画所有被命中的编号的全生命史（细线）
    for tid, pts in tracks_hourly.items():
        if not pts:
            continue
        xs = [p.lon for p in pts]
        ys = [p.lat for p in pts]
        if HAS_CARTOPY:
            ax.plot(xs, ys, '-', lw=LW_ALL, color=COLOR_ALL, alpha=0.75, transform=ccrs.PlateCarree())
        else:
            ax.plot(xs, ys, '-', lw=LW_ALL, color=COLOR_ALL, alpha=0.75)

    # 再对 Excel 逐条加粗影响段
    rows = []
    tids_in_lib = set([k for k,v in tracks_hourly.items() if v])
    for _, r in excel_df.iterrows():
        tid = str(r[ID_COL]).zfill(4)
        st  = pd.to_datetime(r[ST_COL])
        en  = pd.to_datetime(r[EN_COL])
        status = ""
        has_highlight = False
        if tid in tids_in_lib:
            pts = tracks_hourly[tid]
            xs_h, ys_h = [], []
            for p in pts:
                if st <= p.t <= en:
                    xs_h.append(p.lon); ys_h.append(p.lat)
            if len(xs_h) >= 2:
                has_highlight = True
                if HAS_CARTOPY:
                    ax.plot(xs_h, ys_h, '-', lw=LW_HL, color=COLOR_HL, alpha=0.98, transform=ccrs.PlateCarree())
                else:
                    ax.plot(xs_h, ys_h, '-', lw=LW_HL, color=COLOR_HL, alpha=0.98)
            else:
                status = "窗口与生命史无交集（停止编号/消散后仍影响）"
                last = pts[-1]
                if HAS_CARTOPY:
                    ax.scatter([last.lon],[last.lat], marker='x', s=40, color='#444', transform=ccrs.PlateCarree(), zorder=6)
                else:
                    ax.scatter([last.lon],[last.lat], marker='x', s=40, color='#444', zorder=6)
                rkm = min(VMAX_KMH*6.0, UNCERT_RMAX_KM)
                ddeg = rkm/111.0
                tt = np.linspace(0, 2*np.pi, 121)
                cx = last.lon + ddeg*np.cos(tt)/max(math.cos(math.radians(last.lat)), 0.3)
                cy = last.lat + ddeg*np.sin(tt)
                if HAS_CARTOPY:
                    ax.plot(cx, cy, '--', lw=0.8, color='#666', transform=ccrs.PlateCarree())
                else:
                    ax.plot(cx, cy, '--', lw=0.8, color='#666')
        else:
            status = "最佳路径库未发现该编号"
        rows.append({
            "tid": tid,
            "start": st,
            "end": en,
            "found_in_lib": (tid in tids_in_lib),
            "has_highlight_segment": has_highlight,
            "note": status
        })

    # 图例与标题
    h_all = plt.Line2D([0],[0], color=COLOR_ALL, lw=LW_ALL, label='全轨迹（逐小时）')
    h_hl  = plt.Line2D([0],[0], color=COLOR_HL, lw=LW_HL, label='大风影响时段（加粗）')
    h_x   = plt.Line2D([0],[0], color='#666', lw=1.0, ls='--', label='停止编号后影响：最后点× + 不确定性圈')
    plt.legend(handles=[h_all, h_hl, h_x], loc='lower left', fontsize=9, frameon=True)
    plt.title('2010–2024 台风路径（逐小时，仅Excel编号）与影响段加粗')
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()

    pd.DataFrame(rows).to_csv(report_csv, index=False, encoding='utf-8-sig')

# ------------------ 主流程 ------------------

def main():
    print("读取 Excel 影响时段…")
    dfx = read_excel_windows(EXCEL_PATH)
    need_ids: Set[str] = set(dfx[ID_COL].astype(str).str.zfill(4))
    print(f"Excel 里一共 {len(need_ids)} 个唯一编号")

    print("从最佳路径库中只提取这些编号的轨迹…")
    segs = iter_needed_segments(BESTTRACK_DIR, need_ids)
    # 逐小时插值（仅对命中的）
    hourly_tracks: Dict[str, List[BTPoint]] = {}
    for tid in need_ids:
        pts = segs.get(tid, [])
        hourly_tracks[tid] = hourly_interp(pts) if pts else []

    print("绘图并输出报告…")
    out_png = os.path.join(OUTPUT_DIR, "台风路径_逐小时_仅Excel_影响段加粗.png")
    report_csv = os.path.join(OUTPUT_DIR, "仅Excel编号_影响段加粗_报告.csv")
    plot_selected(hourly_tracks, dfx, out_png, report_csv)
    print("→ 图：", out_png)
    print("→ 报告：", report_csv)

if __name__ == "__main__":
    main()
