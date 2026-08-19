import numpy as np
from build_pairs_local import load_wyko_asc, robust_line_detrend_profile

hd = load_wyko_asc(10, crop_to_common=True)
x, y = hd['x_actual_mm'], hd['y_mm']
for xq in [30, 40, 50, 60, 70, 80, 90]:
    m = np.abs(x - xq) <= 0.3
    profile = np.nanmedian(hd['Z_mm'][:, m] * 1000.0, axis=1)
    detrended, sigma = robust_line_detrend_profile(profile, y)
    thresh = max(2.0, 3.0*sigma) if sigma else None
    peak = np.nanmax(detrended) if detrended is not None else None
    print(f"x={xq} | sigma={sigma} | thresh={thresh} | peak-above-baseline={peak}")