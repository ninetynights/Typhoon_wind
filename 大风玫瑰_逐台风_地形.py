#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大风玫瑰_逐台风_地形.py 
------------------------------------------------------------------------------------------------
主要功能：
- 针对每个台风，统计所有站点的风玫瑰图（超阈值/持续性），并输出统一色标的图和CSV。
- 输出每个台风的总结性地图（所有站点风玫瑰叠加）；增加DEM地形背景。
- 输出每个站点的独立风玫瑰图和CSV。
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from netCDF4 import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import xarray as xr
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# ===================== 参数区（可按需修改） =====================
NC_PATH    = r"/Users/momo/Desktop/业务相关/2025 影响台风大风/All_Typhoons_ExMaxWind.nc"
OUTPUT_DIR = Path("/Users/momo/Desktop/业务相关/2025 影响台风大风") / "输出_风玫瑰_逐台风_地形"
# 新增全局变量
DEM_PATH = r"/Users/momo/Desktop/业务相关/2025 影响台风大风/地形文件/DEM_0P05_CHINA.nc"

# ★ 台风信息 Excel（用于映射真实编号与名称）
TYPHOON_META_XLSX = r"/Users/momo/Desktop/业务相关/2025 影响台风大风/2010_2024_影响台风_大风.xlsx"

THRESHOLD        = 17.3       # m/s：超阈值定义
MIN_CONSEC_HOURS = 2          # “持续段”定义：连续小时数阈值（建议 2 或 3）

N_BINS           = 16         # 风向扇区数（16=22.5°；24=15°）
EDGES_DEG        = np.linspace(0, 360, N_BINS+1)

CMAP_NAME  = "viridis"        # 颜色映射（可改为 "plasma"/"magma"/"cividis" 等）
UNIFY_RLIM = False            # 是否统一每张玫瑰图的半径上限（True=统一；False=自适应本站点）

# ----- 新增地图参数 -----
MAP_EXTENT = [117, 124, 27, 32]  # 地图显示范围 [lon_min, lon_max, lat_min, lat_max]
ROSE_SCALE = 0.5                 # 地图上风玫瑰图的直径(英寸)，可根据出图效果调整
# =============================================================


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def rle_segments(bool_arr: np.ndarray):
    if bool_arr.size == 0: return []
    b = np.array(bool_arr, dtype=bool)
    padded = np.concatenate([[False], b, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]
    return list(zip(starts, ends))


def _norm(s: str) -> str:
    if s is None: return ""
    s = str(s).strip()
    s = re.sub(r"[ _\-　]+", "", s, flags=re.IGNORECASE)
    return s.lower()

REALNO_KEYS = {"中央台编号"}
TY_EN_KEYS = {"国外名称"}
TY_CN_KEYS = {"中文名称"}

def _pick_col(df: pd.DataFrame, candidates: set):
    cols = {_norm(c): c for c in df.columns}
    cand_norm = {_norm(x) for x in candidates}
    for nk, c in cols.items():
        if nk in cand_norm: return c
    return None


def load_typhoon_meta_from_excel(path: str):
    if not path:
        print("[WARN] 未提供台风信息 Excel 路径，无法映射真实编号。")
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[WARN] 台风信息 Excel 不存在：{p}")
        return {}
    df = pd.read_excel(p, sheet_name=0)
    col_no  = _pick_col(df, REALNO_KEYS)
    col_en  = _pick_col(df, TY_EN_KEYS)
    col_cn  = _pick_col(df, TY_CN_KEYS)
    if col_no is None:
        print("[WARN] Excel 未识别到“中央台编号”列，将不使用真实编号映射。")
        return {}
    mapping = {}
    bad_rows = 0
    for idx, row in df.iterrows():
        real_no = str(row[col_no]).strip() if pd.notna(row[col_no]) else ""
        if not real_no:
            bad_rows += 1
            continue
        en = str(row[col_en]).strip() if (col_en and pd.notna(row[col_en])) else ""
        cn = str(row[col_cn]).strip() if (col_cn and pd.notna(row[col_cn])) else ""
        mapping[idx] = {"real_no": real_no, "en": en, "cn": cn}
    print(f"[INFO] 载入台风映射 {len(mapping)} 条。{('（有 %d 条记录跳过）' % bad_rows) if bad_rows else ''}")
    return mapping


def compute_roses_for_one_typhoon(ws_ty: np.ndarray, wd_ty: np.ndarray, threshold: float,
                                  min_consec: int, edges_deg: np.ndarray):
    n_time, n_sta = ws_ty.shape
    n_bins = len(edges_deg) - 1
    ex_bins = np.zeros((n_sta, n_bins), dtype=int)
    su_bins = np.zeros((n_sta, n_bins), dtype=int)
    ex_totals = np.zeros(n_sta, dtype=int)
    su_totals = np.zeros(n_sta, dtype=int)
    for i in range(n_sta):
        ws, wd = ws_ty[:, i], wd_ty[:, i]
        valid = np.isfinite(ws) & np.isfinite(wd)
        gt = (ws > threshold) & valid
        if np.any(gt):
            ex_bins[i, :] += np.histogram(np.mod(wd[gt], 360.0), bins=edges_deg)[0].astype(int)
            ex_totals[i] = int(np.sum(gt))
            for s, e in rle_segments(gt):
                if (e - s) >= min_consec:
                    su_bins[i, :] += np.histogram(np.mod(wd[s:e], 360.0), bins=edges_deg)[0].astype(int)
                    su_totals[i]  += (e - s)
    return ex_bins, su_bins, ex_totals, su_totals


def format_pair(a: str, b: str, sep=" / "):
    a, b = (a or "").strip(), (b or "").strip()
    return f"{a}{sep}{b}" if a and b else a or b or ""


def plot_station_rose_pair_colored(exceed_counts, sustain_counts, edges_deg,
                                   ty_code, ty_en, ty_cn, stid,
                                   out_png, cmap_name="viridis", norm=None, unify_rlim=False):
    angles_center_deg = 0.5 * (edges_deg[:-1] + edges_deg[1:])
    angles_rad = np.deg2rad(angles_center_deg)
    widths_rad = np.deg2rad(np.diff(edges_deg))
    if norm is None:
        vmax = max(int(np.max(exceed_counts)), int(np.max(sustain_counts)), 1)
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
    rlim = max(int(np.max(exceed_counts)), int(np.max(sustain_counts)), 1) if unify_rlim else None
    cmap = plt.get_cmap(cmap_name)
    fig = plt.figure(figsize=(12, 9))
    ax1, ax2 = plt.subplot(121, polar=True), plt.subplot(122, polar=True)
    for ax, counts, subtitle in [(ax1, exceed_counts, "Exceedance Hours (>17.3 m/s)"),
                                 (ax2, sustain_counts, f"Sustained Hours (≥{MIN_CONSEC_HOURS}h)")]:
        ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
        ax.set_rlim(0, max(1, int(np.max(counts))) if rlim is None else max(1, int(rlim)))
        ax.set_xticks(np.deg2rad(np.arange(0, 360, 45))); ax.set_yticks([])
        ax.grid(True, alpha=0.3, linestyle=':'); [s.set_visible(False) for s in ax.spines.values()]
        colors = plt.get_cmap(cmap_name)(norm(counts))
        ax.bar(angles_rad, counts, width=widths_rad, align='center', color=colors, edgecolor='none')
        ax.set_title(subtitle, y=1.10, fontsize=12)
    name_pair = format_pair(ty_cn, ty_en)
    l1 = f"{ty_code}  {name_pair}".strip()
    l2 = f"Station {stid}  (Exceed={int(exceed_counts.sum())} h, Sustain={int(sustain_counts.sum())} h)"
    fig.suptitle(l1 + "\n" + l2, y=0.98, fontsize=14)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], fraction=0.03, pad=0.08)
    cbar.set_label("Hours per sector (Unified within typhoon)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

# ----- [修正] 函数：在地图上绘制风玫瑰 -----
def plot_rose_on_map(ax_map, lon, lat, counts, edges_deg, norm, cmap, scale_factor=0.5):
    """在地图投影的指定经纬度点上直接绘制风玫瑰图"""
    if np.sum(counts) == 0: 
        return
    
    # 将角度转换为弧度
    edges_rad = np.deg2rad(edges_deg)
    angles_rad = edges_rad[:-1]
    widths_rad = np.diff(edges_rad)
    
    # 计算每个扇区的半径（按比例缩放）
    max_count = max(1, np.max(counts))
    radii = counts / max_count * scale_factor
    
    # 获取颜色
    colors = cmap(norm(counts))
    
    # 在地图坐标系中绘制每个扇区
    for i in range(len(angles_rad)):
        if counts[i] > 0:
            # 计算扇区的起点和终点角度
            start_angle = angles_rad[i]
            end_angle = angles_rad[i] + widths_rad[i]
            
            # 计算扇区的轮廓点
            theta = np.linspace(start_angle, end_angle, 20)
            r = np.linspace(0, radii[i], 2)
            theta_grid, r_grid = np.meshgrid(theta, r)
            
            # 转换为笛卡尔坐标（注意：极坐标的0度是正北，90度是正东）
            x_offset = r_grid * np.sin(theta_grid)  # 东方向
            y_offset = r_grid * np.cos(theta_grid)  # 北方向
            
            # 将偏移量加到站点坐标上
            x_points = lon + x_offset
            y_points = lat + y_offset
            
            # 绘制填充的多边形
            ax_map.fill(
                np.concatenate([x_points[0, :], x_points[1, ::-1]]),
                np.concatenate([y_points[0, :], y_points[1, ::-1]]),
                facecolor=colors[i],
                edgecolor='none',
                alpha=0.8,
                transform=ccrs.PlateCarree(),  # 使用地理坐标系
                zorder=10
            )

# 新增函数：在现有地图上叠加地形
def add_topography_to_existing_map(ax):
    """在现有的地图轴对象上添加地形（灰度填色+等高线）"""
    try:
        ds = xr.open_dataset(DEM_PATH)
        dem_data = ds['dhgt_gfs']
        lons = ds['Lon'].values
        lats = ds['Lat'].values

        levels = np.arange(-500, 2001, 100)
        norm = mcolors.Normalize(vmin=-500, vmax=2000)

        # 灰度填色
        contourf = ax.contourf(
            lons, lats, dem_data.values,
            levels=levels,
            cmap='Greys_r',      # 反转灰度，高山为深色
            norm=norm,
            alpha=0.7,
            transform=ccrs.PlateCarree(),
            zorder=6
        )
        '''
        # 等高线
        ax.contour(
            lons, lats, dem_data.values,
            levels=levels,
            colors='black',
            linewidths=0.3,
            alpha=0.5,
            transform=ccrs.PlateCarree(),
            zorder=7           # 等高线在填色之上
        )
        '''
        ds.close()
        return contourf

    except Exception as e:
        print(f"Error adding topography: {e}")
        return None
    
# ----- [修正] 函数：创建并保存总结性地图 -----
def create_summary_map(stids, lons, lats, rose_data, edges_deg, norm, cmap,
                       title, out_png, scale_factor, map_extent):
    """创建一个包含所有站点风玫瑰的总结性地图"""
    fig = plt.figure(figsize=(12, 12))
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    ax.set_extent(map_extent, crs=proj)
    add_topography_to_existing_map(ax)# 先添加地形
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.8, zorder=5)
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='whitesmoke', zorder=1)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='aliceblue', zorder=1)
    ax.add_feature(cfeature.RIVERS.with_scale('50m'), lw=0.5, zorder=5)
    ax.add_feature(cfeature.BORDERS, linestyle='-', lw=0.8, zorder=5)

    # 绘制省界线
    provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none')
    ax.add_feature(provinces, edgecolor='gray', linestyle='--', linewidth=0.6, zorder=5)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False; gl.right_labels = False
    gl.xformatter = LongitudeFormatter(zero_direction_label=True)
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {'size': 10}; gl.ylabel_style = {'size': 10}

    # 在地图上绘制每个站点的风玫瑰
    for i, stid in enumerate(stids):
        if np.sum(rose_data[i,:]) > 0:
            plot_rose_on_map(ax, lons[i], lats[i], rose_data[i, :], edges_deg, norm, cmap, scale_factor)

    # 绘制站点位置标记
    ax.scatter(lons, lats, c='black', s=5, marker='o', 
               edgecolors='black', linewidth=0.1, 
               transform=ccrs.PlateCarree(), zorder=15)

    ax.set_title(title, fontsize=16, pad=20)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.05, shrink=0.6)
    cbar.set_label("Hours per sector", fontsize=10)

    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)

def main():
    # —— 读 NetCDF (新增读取经纬度)
    nc = Dataset(NC_PATH)
    ws_all = np.array(nc.variables['wind_velocity'][:, 0, :], copy=True)
    wd_all = np.array(nc.variables['wind_direction'][:, 0, :], copy=True)
    ty_all = np.array(nc.variables['typhoon_id_index'][:, 0, :], copy=True)
    stids  = np.array(nc.variables['STID'][:]).astype(str)
    lats   = np.array(nc.variables['lat'][:])   # 读取纬度
    lons   = np.array(nc.variables['lon'][:])  # 读取经度
    nc.close()

    # —— 载入台风映射
    ty_map = load_typhoon_meta_from_excel(TYPHOON_META_XLSX)

    # —— 收集所有“出现过的台风内部索引”
    uniq_idx = sorted({int(k) for k in np.unique(ty_all) if int(k) >= 0})
    if not uniq_idx:
        print("[WARN] 没有发现任何 typhoon_id_index >= 0 的记录。程序结束。")
        return

    # —— 输出根目录
    root = OUTPUT_DIR; ensure_dir(root)

    # —— 按“内部索引”循环
    for idx in uniq_idx:
        meta = ty_map.get(int(idx), None)
        if meta:
            ty_code, ty_en, ty_cn = meta.get("real_no", f"TY{idx}"), meta.get("en", ""), meta.get("cn", "")
        else:
            ty_code, ty_en, ty_cn = f"TY{idx}", "", ""
        
        ty_name_pair = format_pair(ty_cn, ty_en)
        print(f"\n--- 开始处理台风: {ty_code} ({ty_name_pair}) ---")

        mask_ty_any_station = np.any(ty_all == idx, axis=1)
        if not np.any(mask_ty_any_station):
            print(f"[INFO] 台风 {ty_code} 在所有站点均无记录，跳过。")
            continue

        ws_ty = np.where(ty_all == idx, ws_all, np.nan)[mask_ty_any_station, :]
        wd_ty = np.where(ty_all == idx, wd_all, np.nan)[mask_ty_any_station, :]

        # 统计所有站点的玫瑰（该台风内）
        ex_bins, su_bins, ex_totals, su_totals = compute_roses_for_one_typhoon(
            ws_ty, wd_ty, THRESHOLD, MIN_CONSEC_HOURS, EDGES_DEG
        )

        GLOBAL_VMAX_TY = int(max(ex_bins.max(), su_bins.max(), 1))
        norm_ty = mcolors.Normalize(vmin=0, vmax=GLOBAL_VMAX_TY)
        cmap_ty = plt.get_cmap(CMAP_NAME)

        # —— 输出目录
        ty_dir_name = f"{ty_code} {format_pair(ty_cn, ty_en, sep=' ')}".strip()
        ty_dir = root / ty_dir_name
        ensure_dir(ty_dir)

        # 【新增】===== 1. 输出总结性 CSV 文件 =====
        bin_headers = [f'deg_{int(d)}' for d in EDGES_DEG[:-1]]
        df_ex = pd.DataFrame(ex_bins, columns=bin_headers)
        df_su = pd.DataFrame(su_bins, columns=bin_headers)
        
        # 插入站点信息
        for df in [df_ex, df_su]:
            df.insert(0, 'STID', stids)
            df.insert(1, 'latitude', lats)
            df.insert(2, 'longitude', lons)
        
        csv_ex_path = ty_dir / f"{ty_code}_summary_exceed.csv"
        csv_su_path = ty_dir / f"{ty_code}_summary_sustain.csv"
        df_ex.to_csv(csv_ex_path, index=False, encoding='utf-8-sig')
        df_su.to_csv(csv_su_path, index=False, encoding='utf-8-sig')
        print(f"[OK] 已输出总结性CSV: {csv_ex_path.name}, {csv_su_path.name}")

        # 【新增】===== 2. 输出总结性地图 =====
        title_ex = f"台风: {ty_code} {ty_name_pair}\n超阈值大风时数 (> {THRESHOLD} m/s) 风玫瑰图"
        map_ex_path = ty_dir / f"{ty_code}_map_exceed.png"
        create_summary_map(stids, lons, lats, ex_bins, EDGES_DEG, norm_ty, cmap_ty,
                           title_ex, map_ex_path, ROSE_SCALE, MAP_EXTENT)

        title_su = f"台风: {ty_code} {ty_name_pair}\n持续性大风时数 (≥ {MIN_CONSEC_HOURS}h) 风玫瑰图"
        map_su_path = ty_dir / f"{ty_code}_map_sustain.png"
        create_summary_map(stids, lons, lats, su_bins, EDGES_DEG, norm_ty, cmap_ty,
                           title_su, map_su_path, ROSE_SCALE, MAP_EXTENT)
        print(f"[OK] 已输出总结性地图: {map_ex_path.name}, {map_su_path.name}")

        # ===== 3. (原功能) 每站点输出独立的 CSV + 双图 =====
        out_csv_dir = ty_dir / "per_station" / "csv"
        out_fig_dir = ty_dir / "per_station" / "figs"
        ensure_dir(out_csv_dir); ensure_dir(out_fig_dir)

        stations_with_data = 0
        for i, stid in enumerate(stids):
            ex_counts, su_counts = ex_bins[i, :], su_bins[i, :]
            if ex_counts.sum() == 0 and su_counts.sum() == 0:
                continue
            
            stations_with_data += 1
            df = pd.DataFrame({
                "sector_start_deg": EDGES_DEG[:-1], "sector_end_deg": EDGES_DEG[1:],
                "exceed_hours": ex_counts, "exceed_percent": (ex_counts / max(1, ex_counts.sum())) * 100.0,
                "sustain_hours": su_counts, "sustain_percent": (su_counts / max(1, su_counts.sum())) * 100.0,
            })
            df.to_csv(out_csv_dir / f"{stid}_rose_{THRESHOLD:.1f}_ge{MIN_CONSEC_HOURS}h.csv", index=False)
            out_png = out_fig_dir / f"{stid}_rose_{THRESHOLD:.1f}_ge{MIN_CONSEC_HOURS}h.png"
            plot_station_rose_pair_colored(
                ex_counts, su_counts, EDGES_DEG, ty_code, ty_en, ty_cn, str(stid),
                out_png, CMAP_NAME, norm_ty, UNIFY_RLIM
            )
        print(f"[OK] 已输出 {stations_with_data} 个站点的独立图表和CSV。 （统一色标 vmax={GLOBAL_VMAX_TY}）")

    print(f"\n[ALL DONE] 所有台风的“每站点玫瑰”和“总结地图”已完成。根目录：{root}")

if __name__ == "__main__":
    main()