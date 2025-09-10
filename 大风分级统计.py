#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大风分级统计.py（简洁版：固定 NC 路径 + 文字着色标注）
-------------------------------------------------
改动：
  - 输出目录：固定为 /Users/momo/Desktop/业务相关/2025 影响台风大风/输出_大风分级统计
  - 绘图：使用“不同颜色的数字”标注站点的超阈值小时数（不再使用散点）

依赖：numpy, pandas, matplotlib, cartopy, netCDF4
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset

plt.rcParams['font.sans-serif'] = ['Heiti TC']

# ======= 这里按你的实际情况改一行就够了（NC 路径） =======
NC_PATH    = r"/Users/momo/Desktop/业务相关/2025 影响台风大风/All_Typhoons_ExMaxWind.nc"
# ======================================================

# 其余参数在这里：
OUTPUT_DIR = Path("/Users/momo/Desktop/业务相关/2025 影响台风大风") / "输出_大风分级统计"
THRESHOLD  = 17.3
EXTENT     = [118, 123, 27, 32]   # 设为 None 则自适应
TEXT_SIZE  = 8                    # 数字字号
CMAP_NAME  = "viridis"            # 颜色映射
DRAW_GRID  = True                 # 是否画经纬网
SHOW_ZERO  = True                 # 小时数为 0 时是否也标注
# 注：如果标注太密，可把 SHOW_ZERO=False 或调小 TEXT_SIZE

# ----------------------------- 小工具 -----------------------------
def parse_mapping(attr_str: str):
    """把全局属性里的 'a:1; b:2' 解析为 dict[str, str]"""
    if not attr_str:
        return {}
    pairs = (p for p in attr_str.strip().split(";") if ":" in p)
    return {k.strip(): v.strip() for k, v in (q.split(":", 1) for q in pairs)}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sanitize_filename(name: str) -> str:
    return re.sub(r"[\\/:*?\"<>|\\s]+", "_", name).strip("_")

def draw_station_count_text_map(
    lons, lats, counts, stids, title, out_png,
    extent=None, text_size=8, cmap_name="viridis", show_zero=True,
    vmin=None, vmax=None
):
    """
    用“彩色数字”标注每个站点的超阈值小时数。
    - 颜色根据 counts 映射到 colormap
    - 使用描边增强可读性
    """
    fig = plt.figure(figsize=(10, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_title(title, fontsize=14)

    # 底图要素
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
    ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')

    # 范围
    if extent and len(extent) == 4:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        pad_lon = max(0.5, (np.nanmax(lons) - np.nanmin(lons)) * 0.1)
        pad_lat = max(0.5, (np.nanmax(lats) - np.nanmin(lats)) * 0.1)
        ax.set_extent([np.nanmin(lons)-pad_lon, np.nanmax(lons)+pad_lon,
                       np.nanmin(lats)-pad_lat, np.nanmax(lats)+pad_lat],
                      crs=ccrs.PlateCarree())

    # 颜色映射
    if vmin is None:
        vmin = np.nanmin(counts)
    if vmax is None:
        vmax = np.nanmax(counts)
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax):
        vmax = 1.0
    if vmin == vmax:
        vmax = vmin + 1.0

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    # 逐站点标注彩色数字
    for x, y, val in zip(lons, lats, counts):
        if (not show_zero) and (val == 0):
            continue
        color = cmap(norm(val if np.isfinite(val) else 0.0))
        txt = ax.text(
            x, y, f"{int(val)}", fontsize=text_size,
            ha='center', va='center', color=color,
            transform=ccrs.PlateCarree()
        )
        # 白色描边增强可读性
        txt.set_path_effects([
            pe.withStroke(linewidth=1.5, foreground="white")
        ])

    # 加一个虚拟的 ScalarMappable 用于 colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label(f"Hours with wind speed > {THRESHOLD} m/s")

    # 网格
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, crs=ccrs.PlateCarree())
    try:
        gl.right_labels = False
        gl.top_labels = False
    except Exception:
        pass

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# ----------------------------- 主逻辑 -----------------------------
def main():
    # 1) 读数据
    nc = Dataset(NC_PATH)

    # 全局属性映射（若不存在则回退）
    id_to_index = parse_mapping(getattr(nc, 'id_to_index', None))
    index_to_cn = parse_mapping(getattr(nc, 'index_to_cn', None))
    index_to_en = parse_mapping(getattr(nc, 'index_to_en', None))

    # 站点信息
    stids   = np.array(nc.variables['STID'][:]).astype(str)
    lats    = np.array(nc.variables['lat'][:], dtype=float)
    lons    = np.array(nc.variables['lon'][:], dtype=float)
    heights = np.array(nc.variables['height'][:], dtype=float) if 'height' in nc.variables else np.full_like(lats, np.nan)

    # 风速/台风索引（简化为 [time, station]）
    wind_speeds = np.array(nc.variables['wind_velocity'][:, 0, :], copy=True)
    ty_ids      = np.array(nc.variables['typhoon_id_index'][:, 0, :], copy=True)

    # 将 ty_ids 统一为整数（若是浮点/缺测）
    if np.issubdtype(ty_ids.dtype, np.floating):
        ty_ids_int = np.full_like(ty_ids, fill_value=-1, dtype=int)
        valid = ~np.isnan(ty_ids)
        ty_ids_int[valid] = ty_ids[valid].astype(int)
        ty_ids = ty_ids_int
    else:
        ty_ids = ty_ids.astype(int)

    n_time, n_sta = wind_speeds.shape

    # 2) 输出目录
    out_csv = OUTPUT_DIR / "csv"
    out_fig = OUTPUT_DIR / "figs"
    ensure_dir(out_csv); ensure_dir(out_fig)

    # 3) 确定要遍历的台风索引与 TID 名称
    if id_to_index:
        items = []
        for tid_str, idx_str in id_to_index.items():
            try:
                idx = int(str(idx_str).strip())
            except Exception:
                continue
            items.append((tid_str, idx))
        items.sort(key=lambda kv: kv[0])  # 以 tid 字典序
    else:
        uniq = sorted({int(x) for x in np.unique(ty_ids) if int(x) >= 0})
        items = [(str(idx), idx) for idx in uniq]

    # 4) 统计 + 输出
    total_counts = np.zeros(n_sta, dtype=int)

    for tid_str, ty_idx in items:
        cn_name = index_to_cn.get(str(ty_idx), "")
        en_name = index_to_en.get(str(ty_idx), "")

        counts = np.zeros(n_sta, dtype=int)
        for i in range(n_sta):
            mask = (ty_ids[:, i] == ty_idx)
            if not np.any(mask):
                counts[i] = 0
                continue
            ws = wind_speeds[mask, i]
            # 计数大于阈值的小时数
            counts[i] = int(np.sum(ws > THRESHOLD))

        # CSV（单台风）
        df = pd.DataFrame({
            "STID": stids,
            "Lon": lons,
            "Lat": lats,
            "Height": heights,
            f"Hours_WS_gt_{THRESHOLD}": counts
        })
        fname = f"Exceed_{THRESHOLD:.1f}_{sanitize_filename(tid_str)}_{sanitize_filename(cn_name)}_{sanitize_filename(en_name)}.csv"
        df.to_csv(out_csv / fname, index=False, encoding="utf-8")

        # 图（单台风，彩色数字）
        title = f"TID {tid_str} - {cn_name} ({en_name})\nHours with WS > {THRESHOLD} m/s"
        png = out_fig / f"Exceed_{THRESHOLD:.1f}_{sanitize_filename(tid_str)}.png"
        draw_station_count_text_map(
            lons, lats, counts, stids, title, str(png),
            extent=EXTENT, text_size=TEXT_SIZE, cmap_name=CMAP_NAME, show_zero=SHOW_ZERO
        )

        total_counts += counts

    # 总体 CSV + 图
    df_total = pd.DataFrame({
        "STID": stids,
        "Lon": lons,
        "Lat": lats,
        "Height": heights,
        f"TotalHours_WS_gt_{THRESHOLD}": total_counts
    })
    df_total.to_csv(out_csv / f"AllTyphoons_Exceed_{THRESHOLD:.1f}.csv", index=False, encoding="utf-8")

    title_total = f"All Typhoons\nTotal Hours with WS > {THRESHOLD} m/s"
    png_total = out_fig / f"AllTyphoons_Exceed_{THRESHOLD:.1f}.png"
    draw_station_count_text_map(
        lons, lats, total_counts, stids, title_total, str(png_total),
        extent=EXTENT, text_size=TEXT_SIZE, cmap_name=CMAP_NAME, show_zero=SHOW_ZERO
    )

    print(f"[OK] 单台风 CSV/图 已输出至: {OUTPUT_DIR}")
    print(f"[OK] 总体 CSV/图 已输出至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
