import xarray as xr

file_path = "/Users/momo/Desktop/业务相关/2025 影响台风大风/HRCLDAS/VWIN/2024/20241101/Z_NAFP_C_BABJ_20241101041721_P_HRCLDAS_RT_BEHZ_0P01_HOR-VWIN-2024110104.GRB2"

# 使用 cfgrib 引擎读取
ds = xr.open_dataset(file_path, engine="cfgrib")

print(ds)       # 查看文件内容
print(ds.variables)  # 列出变量