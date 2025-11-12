
"""
持续大风统计.py（持续超过阈值的小时数；与“总超阈值小时数”不同）
-------------------------------------------------
定义（可调）
  - 持续事件：连续 >= MIN_CONSEC_HOURS 个小时风速 > THRESHOLD 的时间段
  - 统计值：把所有满足条件的“连续段”的小时数累加（若把 MIN_CONSEC_HOURS 设为 1，就等价于总超阈值小时数）

输出（与上一脚本一致的结构与风格）：
  - 每个台风：CSV + 地图（彩色数字）
  - 总体：CSV + 地图（彩色数字）
  - 目录：/Users/momo/Desktop/业务相关/2025 影响台风大风/输出_持续大风统计/{csv,figs}
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

# ======= 路径与参数（与上一脚本保持一致的写法） =======
NC_PATH    = r"/Users/momo/Desktop/业务相关/2025 影响台风大风/All_Typhoons_ExMaxWind.nc"
OUTPUT_DIR = Path("/Users/momo/Desktop/业务相关/2025 影响台风大风") / "输出_持续大风统计"

THRESHOLD          = 17.3     # 风速阈值 (m/s)
MIN_CONSEC_HOURS   = 2        # “持续”定义：连续小时数阈值（建议 2 或 3；设为 1 则退化为总小时数）

EXTENT     = [118, 123, 27, 32]  # 设为 None 则自适应
TEXT_SIZE  = 8                   # 数字字号
CMAP_NAME  = "plasma"            # 颜色映射（与上一脚本不同以便识别，可自由修改）
DRAW_GRID  = True
SHOW_ZERO  = True                # 是否显示 0

# ----------------------------- 工具函数 -----------------------------
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
    """用“彩色数字”标注每个站点的统计值。"""
    fig = plt.figure(figsize=(10, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_title(title, fontsize=14)

    ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
    ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')

    if extent and len(extent) == 4:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        pad_lon = max(0.5, (np.nanmax(lons) - np.nanmin(lons)) * 0.1)
        pad_lat = max(0.5, (np.nanmax(lats) - np.nanmin(lats)) * 0.1)
        ax.set_extent([np.nanmin(lons)-pad_lon, np.nanmax(lons)+pad_lon,
                       np.nanmin(lats)-pad_lat, np.nanmax(lats)+pad_lat],
                      crs=ccrs.PlateCarree())

    if vmin is None:
        vmin = np.nanmin(counts) if np.isfinite(np.nanmin(counts)) else 0.0
    if vmax is None:
        vmax = np.nanmax(counts) if np.isfinite(np.nanmax(counts)) else 1.0
    if vmin == vmax:
        vmax = vmin + 1.0

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    for x, y, val in zip(lons, lats, counts):
        if (not show_zero) and (val == 0):
            continue
        color = cmap(norm(val if np.isfinite(val) else 0.0))
        txt = ax.text(
            x, y, f"{int(val)}", fontsize=text_size,
            ha='center', va='center', color=color,
            transform=ccrs.PlateCarree()
        )
        txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground="white")])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label(f"Sustained hours (> {THRESHOLD} m/s, ≥ {MIN_CONSEC_HOURS}h)")

    if DRAW_GRID:
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, crs=ccrs.PlateCarree())
        try:
            gl.right_labels = False
            gl.top_labels = False
        except Exception:
            pass

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def rle_sustained_count(bool_arr: np.ndarray, min_len: int) -> int:
    """
    计算布尔序列中，值为 True 的“连续段”长度 >= min_len 的这些段的总长度之和。
    例如：T T F T T T F  且 min_len=2 -> 段1长度2计入，段2长度3计入，总和=5。
    """
    if bool_arr.size == 0:
        return 0

    # 将 NaN/缺测视为 False
    b = np.array(bool_arr, dtype=bool)

    # 在首尾添加 False 以便捕捉边界变化
    padded = np.concatenate([[False], b, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]       # 段起始索引（相对于原始数组）
    ends   = np.where(diff == -1)[0]      # 段结束位置（结束后一位）

    lengths = ends - starts
    good = lengths >= min_len
    return int(np.sum(lengths[good]))

# ----------------------------- 主逻辑 -----------------------------
def main():
    # 1) 读数据
    nc = Dataset(NC_PATH)

    id_to_index = parse_mapping(getattr(nc, 'id_to_index', None))
    index_to_cn = parse_mapping(getattr(nc, 'index_to_cn', None))
    index_to_en = parse_mapping(getattr(nc, 'index_to_en', None))

    stids   = np.array(nc.variables['STID'][:]).astype(str)
    lats    = np.array(nc.variables['lat'][:], dtype=float)
    lons    = np.array(nc.variables['lon'][:], dtype=float)
    heights = np.array(nc.variables['height'][:], dtype=float) if 'height' in nc.variables else np.full_like(lats, np.nan)

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

    # 3) 要遍历的台风集合
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
            # 取该站点属于该台风影响期的所有小时
            mask_ty = (ty_ids[:, i] == ty_idx)
            if not np.any(mask_ty):
                counts[i] = 0
                continue

            ws = wind_speeds[mask_ty, i]
            # 超阈值布尔序列（NaN 作为 False）
            gt = np.zeros_like(ws, dtype=bool)
            valid = np.isfinite(ws)
            gt[valid] = ws[valid] > THRESHOLD

            # 计算“连续 >= MIN_CONSEC_HOURS 的段”的总长度（小时）
            counts[i] = rle_sustained_count(gt, MIN_CONSEC_HOURS)

        # CSV（单台风）
        df = pd.DataFrame({
            "STID": stids,
            "Lon": lons,
            "Lat": lats,
            "Height": heights,
            f"SustainedHours_WS_gt_{THRESHOLD}_ge{MIN_CONSEC_HOURS}h": counts
        })
        fname = f"Sustained_{THRESHOLD:.1f}mps_ge{MIN_CONSEC_HOURS}h_{sanitize_filename(tid_str)}_{sanitize_filename(cn_name)}_{sanitize_filename(en_name)}.csv"
        df.to_csv(out_csv / fname, index=False, encoding="utf-8")

        # 图（单台风）
        title = (f"TID {tid_str} - {cn_name} ({en_name})\n"
                 f"Sustained hours > {THRESHOLD} m/s (≥ {MIN_CONSEC_HOURS}h)")
        png = out_fig / f"Sustained_{THRESHOLD:.1f}_ge{MIN_CONSEC_HOURS}h_{sanitize_filename(tid_str)}.png"
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
        f"TotalSustainedHours_WS_gt_{THRESHOLD}_ge{MIN_CONSEC_HOURS}h": total_counts
    })
    df_total.to_csv(out_csv / f"AllTyphoons_Sustained_{THRESHOLD:.1f}_ge{MIN_CONSEC_HOURS}h.csv", index=False, encoding="utf-8")

    title_total = f"All Typhoons\nSustained hours > {THRESHOLD} m/s (≥ {MIN_CONSEC_HOURS}h)"
    png_total = out_fig / f"AllTyphoons_Sustained_{THRESHOLD:.1f}_ge{MIN_CONSEC_HOURS}h.png"
    draw_station_count_text_map(
        lons, lats, total_counts, stids, title_total, str(png_total),
        extent=EXTENT, text_size=TEXT_SIZE, cmap_name=CMAP_NAME, show_zero=SHOW_ZERO
    )

    print(f"[OK] 单台风 CSV/图 已输出至: {OUTPUT_DIR}")
    print(f"[OK] 总体 CSV/图 已输出至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
