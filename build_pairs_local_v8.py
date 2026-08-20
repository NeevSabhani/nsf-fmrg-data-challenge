"""
Builds the corrected (thermal, SEM, metadata) -> width dataset.

Two fixes over the original builder:

1. Labels. The old extractor thresholded the detrended height profile at
   a noise floor, max(2um, 3*sigma), rather than at a track edge — at
   track 10 x=85mm that cut (7.5um) sat above the track's own crown
   (7.0um) and reported 0.000mm for a plainly-present track. Width is
   now measured at HALF MAXIMUM of the smoothed crown, and returns NaN
   rather than 0 when no track is detectable.

2. SEM masking. The old code took one row range per tile and painted it
   across the full image width, but the band is absent from part of the
   first tile and drifts vertically within a tile. Bands are now
   detected per column, with each row crop centred on the local band.

Run: python build_pairs_local_v8.py
"""
import gc
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import uniform_filter, median_filter

from build_pairs_local import (
    TRACK_IDS, TRACK_LASER_POWER_W, COMMON_X_START_MM, COMMON_X_END_MM,
    EDGE_MARGIN_MM, K_THERMAL, SEM_PATCH_SIZE, SEM_TILE_WIDTH_MM,
    extract_final_thermal_frames, resize_thermal_stack,
    load_wyko_asc, get_sem_tile_paths, load_sem_tile, sem_tile_index_for_x,
    robust_line_detrend_profile,
)

CACHE_DIR = Path("./processed_data/cache_v8")
QA_DIR = Path("./processed_data/qa_v8")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
QA_DIR.mkdir(parents=True, exist_ok=True)

CROP_WIDTH_MM = 1.0
PROFILE_SMOOTH_ROWS = 7    # median-smooth the height profile before peak/half-max
MIN_PEAK_UM = 3.0          # below this crown height there is no usable track
MIN_RUN_PX = 20
SMOOTH_COLS = 15
ROW_HALF_SPAN_FRAC = 1.25  # row crop half-height, as a multiple of local band thickness


# ------------------------------------------------------------------
# FIX 1: half-maximum width extraction
# ------------------------------------------------------------------
def extract_local_width_v8(height_data, x_query, half_win_mm=0.3):
    x, y = height_data["x_actual_mm"], height_data["y_mm"]
    m = np.abs(x - x_query) <= half_win_mm
    if not np.any(m):
        return np.nan
    with np.errstate(all="ignore"):
        profile = np.nanmedian(height_data["Z_mm"][:, m] * 1000.0, axis=1)  # um
    if np.all(np.isnan(profile)):
        return np.nan

    detrended, sigma = robust_line_detrend_profile(profile, y)
    if detrended is None:
        return np.nan

    # Smooth before measuring so a single noise spike cannot set the crown.
    filled = np.where(np.isnan(detrended), np.nanmin(detrended), detrended)
    smooth = median_filter(filled, size=PROFILE_SMOOTH_ROWS)

    peak = float(np.nanmax(smooth))
    if not np.isfinite(peak) or peak < MIN_PEAK_UM:
        return np.nan  # no detectable track -- NaN, never a silent 0.0

    half = peak * 0.5
    above = smooth >= half
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
        return np.nan
    dy = abs(float(np.median(np.diff(y))))
    return float(best_len * dy)


# ------------------------------------------------------------------
# FIX 2: per-column SEM band detection
# ------------------------------------------------------------------
def per_column_band(img, win=15, pct=30):
    img = img.astype(np.float32)
    mean = uniform_filter(img, size=win)
    var = np.clip(uniform_filter(img ** 2, size=win) - mean ** 2, 0, None)
    low = var < np.percentile(var, pct)

    H, W = img.shape
    starts = np.full(W, np.nan)
    ends = np.full(W, np.nan)
    for c in range(W):
        col = low[:, c]
        best_len, best_start, cur_start, cur_len = 0, None, None, 0
        for i, b in enumerate(col):
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
        if best_start is not None and best_len >= MIN_RUN_PX:
            starts[c], ends[c] = best_start, best_start + best_len

    # Smooth along x; the track edge is physically continuous.
    def smooth_nan(a):
        out = a.copy()
        good = ~np.isnan(a)
        if good.sum() > SMOOTH_COLS:
            out[good] = median_filter(a[good], size=SMOOTH_COLS)
        return out

    return smooth_nan(starts), smooth_nan(ends)


def get_sem_patch_v8(x_query, tile_paths, tile_cache):
    tile_idx = sem_tile_index_for_x(x_query, len(tile_paths))
    if tile_idx not in tile_cache:
        img = load_sem_tile(tile_paths[tile_idx]).astype(np.float32)
        starts, ends = per_column_band(img)
        tile_left = 100.0 - (tile_idx + 1) * SEM_TILE_WIDTH_MM
        tile_cache[tile_idx] = (img, starts, ends, tile_left)
    img, starts, ends, tile_left = tile_cache[tile_idx]

    H, W = img.shape
    px_per_mm = W / SEM_TILE_WIDTH_MM
    frac = min(max((x_query - tile_left) / SEM_TILE_WIDTH_MM, 0.0), 1.0)
    col_center = int(min(frac * W, W - 1))
    half_px = int(round(CROP_WIDTH_MM * 0.5 * px_per_mm))
    col_lo, col_hi = col_center - half_px, col_center + half_px
    pad_lo, pad_hi = max(0, -col_lo), max(0, col_hi - W)
    col_lo_c, col_hi_c = max(0, col_lo), min(W, col_hi)

    # Row crop follows the LOCAL band, not a tile-wide average.
    win = slice(max(0, col_center - half_px), min(W, col_center + half_px))
    s_loc, e_loc = np.nanmedian(starts[win]), np.nanmedian(ends[win])
    band_present = float(np.mean(~np.isnan(starts[win]))) if win.stop > win.start else 0.0
    if np.isnan(s_loc) or np.isnan(e_loc):
        # No track at this x: fall back to a centered crop, flag it as absent.
        center, thickness = H // 2, H // 4
    else:
        center, thickness = 0.5 * (s_loc + e_loc), (e_loc - s_loc)
    span = max(int(round(thickness * ROW_HALF_SPAN_FRAC)), 24)
    row_lo, row_hi = int(center - span), int(center + span)
    rpad_lo, rpad_hi = max(0, -row_lo), max(0, row_hi - H)
    row_lo_c, row_hi_c = max(0, row_lo), min(H, row_hi)

    img_crop = img[row_lo_c:row_hi_c, col_lo_c:col_hi_c]
    if rpad_lo or rpad_hi or pad_lo or pad_hi:
        img_crop = np.pad(img_crop, ((rpad_lo, rpad_hi), (pad_lo, pad_hi)), mode="edge")

    # Mask channel: exactly the detected band within this crop, per column.
    mask_crop = np.zeros_like(img_crop, dtype=np.float32)
    cols = np.arange(col_lo, col_hi)
    for k, c in enumerate(cols):
        if 0 <= c < W and not np.isnan(starts[c]):
            a = int(np.clip(starts[c] - row_lo, 0, mask_crop.shape[0]))
            b = int(np.clip(ends[c] - row_lo, 0, mask_crop.shape[0]))
            if b > a and k < mask_crop.shape[1]:
                mask_crop[a:b, k] = 1.0

    img_r = np.array(Image.fromarray(img_crop).resize((SEM_PATCH_SIZE, SEM_PATCH_SIZE)))
    mask_r = np.array(Image.fromarray(mask_crop).resize((SEM_PATCH_SIZE, SEM_PATCH_SIZE), Image.NEAREST))
    sem_width_mm = (e_loc - s_loc) / px_per_mm if not np.isnan(s_loc) else np.nan
    return np.stack([img_r, mask_r], axis=0).astype(np.float32), sem_width_mm, band_present


def build_pairs_for_track_v8(track_id, k=K_THERMAL, edge_margin=EDGE_MARGIN_MM):
    thermal = extract_final_thermal_frames(track_id)
    frames, x_mm = thermal["frames"], thermal["x_mm_center"]
    n = len(x_mm)

    height_data = load_wyko_asc(track_id, crop_to_common=True)
    sem_tiles = get_sem_tile_paths(track_id)
    power = TRACK_LASER_POWER_W[track_id]
    tile_cache = {}

    T_l, S_l, M_l, G_l, X_l, W_l = [], [], [], [], [], []
    for j in range(n):
        xj = x_mm[j]
        if xj < COMMON_X_START_MM + edge_margin or xj > COMMON_X_END_MM - edge_margin:
            continue
        lo, hi = max(0, j - k), min(n, j + k + 1)
        T_ij = resize_thermal_stack(frames[lo:hi])
        want = 2 * k + 1
        if T_ij.shape[0] < want:
            pad = want - T_ij.shape[0]
            T_ij = (np.concatenate([T_ij[:1]] * pad + [T_ij], axis=0) if lo == 0
                    else np.concatenate([T_ij] + [T_ij[-1:]] * pad, axis=0))

        S_ij, sem_w, band_present = get_sem_patch_v8(xj, sem_tiles, tile_cache)
        G_j = extract_local_width_v8(height_data, xj)

        T_l.append(T_ij); S_l.append(S_ij)
        M_l.append([xj / 100.0, power / 400.0])
        G_l.append(G_j); X_l.append(xj); W_l.append(sem_w)

    return {
        "T": np.stack(T_l).astype(np.float32),
        "S": np.stack(S_l).astype(np.float32),
        "M": np.array(M_l, dtype=np.float32),
        "G": np.array(G_l, dtype=np.float32),
        "X": np.array(X_l, dtype=np.float32),
        "W_sem": np.array(W_l, dtype=np.float32),
    }


if __name__ == "__main__":
    summary = {}
    for track_id in TRACK_IDS:
        print(f"Building track {track_id} (v8)...")
        d = build_pairs_for_track_v8(track_id)
        G, W = d["G"], d["W_sem"]
        good = ~np.isnan(G)
        both = good & ~np.isnan(W)
        corr = np.corrcoef(G[both], W[both])[0, 1] if both.sum() > 10 else np.nan
        print(f"  widths: median {np.nanmedian(G):.3f}mm | std {np.nanstd(G):.3f} | "
              f"NaN {np.sum(~good)}/{len(G)}")
        print(f"  corr(Wyko half-max width, SEM band width) = {corr:+.3f}  (n={both.sum()})")
        summary[track_id] = float(corr)

        out = CACHE_DIR / f"track_{track_id}_v8.npz"
        np.savez_compressed(out, T=d["T"], S=d["S"], M=d["M"], G=d["G"], X=d["X"], W_sem=d["W_sem"])
        print(f"  saved {out} ({out.stat().st_size / 1e6:.1f} MB)")

        fig, ax = plt.subplots(figsize=(11, 3.2))
        ax.plot(d["X"], G, ".", markersize=4, label="Wyko half-max width (v8 label)")
        ax.plot(d["X"], W, "-", lw=0.8, alpha=0.7, label="SEM band width (independent)")
        ax.set_xlabel("x (mm)"); ax.set_ylabel("width (mm)")
        ax.set_title(f"Track {track_id}: v8 labels vs independent SEM measurement — corr {corr:+.3f}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(QA_DIR / f"width_qa_track_{track_id}.png", dpi=140)
        plt.close(fig)
        del d
        gc.collect()

    print("\nAgreement between the two INDEPENDENT width measurements, per track:")
    for t, c in summary.items():
        print(f"  track {t}: {c:+.3f}")
    print("\n(v3 labels scored +0.045 on track 10. Meaningfully positive values here mean the "
          "labels now track real physical width.)")
