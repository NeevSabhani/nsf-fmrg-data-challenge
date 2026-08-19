"""
Why are the Wyko-derived width labels ~3x smaller than the SEM-measured
width, with ~25% exact zeros?

extract_local_width_v3 detrends the height profile across the track, then
takes the largest contiguous run of rows ABOVE max(2um, 3*sigma_noise).
That threshold is a NOISE floor, not a track EDGE. A melt track has
sloped shoulders, so a 3-sigma cut slices off everything except the very
crown -- underestimating width, and returning 0 whenever the crown itself
fails to clear the cut.

This plots the actual detrended profile at several x positions with the
current threshold drawn on it, next to a half-maximum edge criterion (the
standard way to measure a feature's width), so the mechanism is visible.

Run: python diagnose_wyko_profile.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_pairs_local import load_wyko_asc, robust_line_detrend_profile

OUT_DIR = "./processed_data/qa_sem_band"
import os
os.makedirs(OUT_DIR, exist_ok=True)

TRACK = 10
X_QUERIES = [35.0, 45.0, 55.0, 65.0, 75.0, 85.0]


def half_max_width(detrended, y):
    """Width at half the crown height -- measures the track to its
    shoulders instead of to the noise floor."""
    d = np.where(np.isnan(detrended), -np.inf, detrended)
    peak = np.nanmax(detrended)
    if not np.isfinite(peak) or peak <= 0:
        return np.nan, np.nan
    half = peak * 0.5
    above = d >= half
    best_len, best_start, cur_start, cur_len = 0, None, None, 0
    for i, b in enumerate(above):
        if b:
            if cur_start is None:
                cur_start = i
            cur_len += 1
        else:
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
            cur_start, cur_len = None, 0
    if cur_len > best_len:
        best_len, best_start = cur_len, cur_start
    if best_start is None:
        return np.nan, half
    dy = abs(float(np.median(np.diff(y))))
    return best_len * dy, half


hd = load_wyko_asc(TRACK, crop_to_common=True)
x, y = hd["x_actual_mm"], hd["y_mm"]

fig, axes = plt.subplots(2, 3, figsize=(15, 7))
print(f"Track {TRACK}: comparing 3-sigma-threshold width vs half-max width\n")
for ax, xq in zip(axes.ravel(), X_QUERIES):
    m = np.abs(x - xq) <= 0.3
    profile = np.nanmedian(hd["Z_mm"][:, m] * 1000.0, axis=1)  # um
    detrended, sigma = robust_line_detrend_profile(profile, y)
    if detrended is None:
        continue
    thresh = max(2.0, 3.0 * sigma)
    hm_w, half = half_max_width(detrended, y)

    # current criterion's width
    above = np.where(np.isnan(detrended), False, detrended > thresh)
    best_len, cur_len = 0, 0
    for b in above:
        cur_len = cur_len + 1 if b else 0
        best_len = max(best_len, cur_len)
    dy = abs(float(np.median(np.diff(y))))
    cur_w = best_len * dy

    print(f"  x={xq:5.1f} | peak {np.nanmax(detrended):6.2f}um | 3-sigma thresh {thresh:5.2f}um "
          f"-> width {cur_w:.3f}mm | half-max thresh {half:5.2f}um -> width {hm_w:.3f}mm")

    ax.plot(y, detrended, "k-", lw=1.0, label="detrended height (um)")
    ax.axhline(thresh, color="tab:red", ls="--", lw=1.2, label=f"3-sigma cut ({thresh:.1f}um)")
    ax.axhline(half, color="tab:green", ls="--", lw=1.2, label=f"half-max ({half:.1f}um)")
    ax.set_title(f"x = {xq} mm   |   3-sigma w={cur_w:.3f}mm   half-max w={hm_w:.3f}mm", fontsize=9)
    ax.set_xlabel("y (mm)"); ax.set_ylabel("height (um)")
    ax.legend(fontsize=7)

fig.suptitle(f"Track {TRACK}: the 3-sigma NOISE cut slices the track's shoulders off; "
             f"half-max measures to the shoulders", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/wyko_profile_threshold_track_{TRACK}.png", dpi=140)
print(f"\nSaved {OUT_DIR}/wyko_profile_threshold_track_{TRACK}.png")
