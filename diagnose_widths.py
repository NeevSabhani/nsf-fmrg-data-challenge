"""
Standalone diagnostic — figure out why so many width extractions land at 0.0
Run: python diagnose_widths.py
"""
import numpy as np
from build_pairs_local import load_wyko_asc  # reuses the loader you already have

def diagnose_width(height_data, x_query, window_mm=0.3, height_thresh_um=5.0):
    x = height_data['x_actual_mm']; y = height_data['y_mm']; Z = height_data['Z_mm']
    col_mask = np.abs(x - x_query) <= window_mm
    Z_win = Z[:, col_mask] * 1000.0
    profile = np.nanmedian(Z_win, axis=1)
    n_valid_rows = np.sum(~np.isnan(profile))
    col_median = np.nanmedian(profile)
    print(f"x={x_query:.1f} | valid rows: {n_valid_rows}/{len(profile)} | "
          f"baseline: {col_median:.2f}um | profile range: [{np.nanmin(profile):.2f}, {np.nanmax(profile):.2f}]um | "
          f"max above baseline: {np.nanmax(profile)-col_median:.2f}um")

if __name__ == "__main__":
    print("Loading Track 21 height data...")
    height_data_21 = load_wyko_asc(21, crop_to_common=True)
    for xq in [30, 40, 50, 60, 70, 80, 90]:
        diagnose_width(height_data_21, xq)