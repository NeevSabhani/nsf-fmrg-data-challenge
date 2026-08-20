"""
Testing:

  A. Whether the track spans the full width of every tile (it does not —
     tile 01 has no track across 17.4% of its columns).
  B. Whether the track drifts vertically within a tile (it does, so one
     flat row range per tile cannot follow it).
  C. Whether SEM band thickness — an independent measurement of local
     width — agrees with the Wyko labels.
Run: python diagnose_sem_band.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, median_filter

from build_pairs_local import (
    SEM_TILE_WIDTH_MM, get_sem_tile_paths, load_sem_tile,
    load_wyko_asc, extract_local_width_v3,
)

OUT_DIR = "./processed_data/qa_sem_band"
import os
os.makedirs(OUT_DIR, exist_ok=True)

TRACK = 10
MIN_RUN_PX = 20          # shorter low-texture runs are noise, not the track
SMOOTH_COLS = 15         # median-smooth the per-column result along x


def per_column_band(img, win=15, pct=30):
    """For each column, find the largest contiguous run of LOW-texture rows
    (the melted track is smooth; the substrate is speckled). Returns
    (row_start, row_end, thickness_px) arrays, NaN where no run qualifies."""
    img = img.astype(np.float32)
    mean = uniform_filter(img, size=win)
    var = np.clip(uniform_filter(img ** 2, size=win) - mean ** 2, 0, None)
    thresh = np.percentile(var, pct)
    low = var < thresh  # (H, W) boolean

    H, W = img.shape
    starts = np.full(W, np.nan)
    ends = np.full(W, np.nan)
    for c in range(W):
        col = low[:, c]
        best_len, best_start = 0, None
        cur_start, cur_len = None, 0
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
            starts[c] = best_start
            ends[c] = best_start + best_len
    thickness = ends - starts
    return starts, ends, thickness


def analyze_track(track_id):
    tiles = get_sem_tile_paths(track_id)
    print(f"Track {track_id}: {len(tiles)} SEM tiles")

    all_x, all_thick_mm, all_start, all_end = [], [], [], []
    per_tile = []
    for ti, path in enumerate(tiles):
        img = load_sem_tile(path).astype(np.float32)
        H, W = img.shape
        mm_per_px = SEM_TILE_WIDTH_MM / W
        s, e, th = per_column_band(img)
        th_s = median_filter(np.nan_to_num(th, nan=0.0), size=SMOOTH_COLS)
        th_s[np.isnan(th)] = np.nan

        tile_left = 100.0 - (ti + 1) * SEM_TILE_WIDTH_MM
        x_cols = tile_left + (np.arange(W) + 0.5) * mm_per_px

        frac_missing = float(np.mean(np.isnan(th)))
        drift = float(np.nanmax(s) - np.nanmin(s)) if np.any(~np.isnan(s)) else np.nan
        per_tile.append((path.stem, frac_missing, drift, float(np.nanmedian(th))))
        print(f"  {path.stem}: no-band columns {100*frac_missing:5.1f}% | "
              f"band-top drift {drift:6.1f}px | median thickness {np.nanmedian(th):6.1f}px "
              f"({np.nanmedian(th)*mm_per_px:.3f}mm)")

        all_x.append(x_cols)
        all_thick_mm.append(th_s * mm_per_px)
        all_start.append(s); all_end.append(e)

    x = np.concatenate(all_x)
    thick = np.concatenate(all_thick_mm)
    order = np.argsort(x)
    return x[order], thick[order], per_tile, tiles


def compare_to_wyko(track_id, x_sem, w_sem):
    """Correlate SEM-derived width against the Wyko ground-truth labels at
    the same x, for BOTH assumed x-orientations of the SEM tiles."""
    hd = load_wyko_asc(track_id, crop_to_common=True)
    xs = np.arange(28.0, 92.0, 0.2)
    w_wyko = np.array([extract_local_width_v3(hd, xq) for xq in xs], dtype=float)

    def interp_sem(x_axis):
        good = ~np.isnan(w_sem)
        return np.interp(xs, x_axis[good], w_sem[good], left=np.nan, right=np.nan)

    results = {}
    for label, x_axis in [("assumed (col0 = low x)", x_sem),
                          ("flipped (col0 = high x)", x_sem.max() + x_sem.min() - x_sem)]:
        if label.startswith("flipped"):
            o = np.argsort(x_axis)
            w_i = np.interp(xs, x_axis[o], w_sem[o], left=np.nan, right=np.nan)
        else:
            w_i = interp_sem(x_axis)
        m = (~np.isnan(w_i)) & (~np.isnan(w_wyko))
        c = np.corrcoef(w_i[m], w_wyko[m])[0, 1] if m.sum() > 10 else np.nan
        results[label] = (c, int(m.sum()))
        print(f"  corr(SEM width, Wyko width) [{label}]: {c:+.3f}  (n={m.sum()})")
    return xs, w_wyko, results


if __name__ == "__main__":
    x_sem, w_sem, per_tile, tiles = analyze_track(TRACK)

    print("\n--- Comparing SEM-measured width to Wyko ground-truth labels ---")
    xs, w_wyko, corr = compare_to_wyko(TRACK, x_sem, w_sem)

    # ---- Figure 1: per-column band on a few tiles (does it slant / stop?) ----
    picks = [0, len(tiles) // 2, len(tiles) - 1]
    fig, axes = plt.subplots(1, len(picks), figsize=(5.0 * len(picks), 3.6))
    for ax, ti in zip(np.atleast_1d(axes), picks):
        img = load_sem_tile(tiles[ti]).astype(np.float32)
        s, e, th = per_column_band(img)
        ax.imshow(img, cmap="gray")
        ax.plot(np.arange(len(s)), s, "r-", lw=1.0, label="band top (per column)")
        ax.plot(np.arange(len(e)), e, "c-", lw=1.0, label="band bottom (per column)")
        ax.set_title(f"{tiles[ti].stem}\n{100*np.mean(np.isnan(th)):.0f}% columns with NO band", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Track {TRACK}: per-COLUMN band detection — red/cyan should hug the real track, "
                 f"gaps = no track present")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/per_column_band_track_{TRACK}.png", dpi=140)
    print(f"\nSaved {OUT_DIR}/per_column_band_track_{TRACK}.png")

    # ---- Figure 2: SEM-measured width vs Wyko labels along the track ----
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x_sem, w_sem, "-", color="tab:blue", lw=0.9, label="SEM-measured width (per column)")
    ax.plot(xs, w_wyko, "k.", markersize=4, label="Wyko ground-truth label (used for training)")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("local width (mm)")
    c_assumed = corr["assumed (col0 = low x)"][0]
    c_flipped = corr["flipped (col0 = high x)"][0]
    ax.set_title(f"Track {TRACK}: SEM-measured vs Wyko-labelled width — "
                 f"corr {c_assumed:+.3f} (assumed orientation), {c_flipped:+.3f} (flipped)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/sem_vs_wyko_width_track_{TRACK}.png", dpi=140)
    print(f"Saved {OUT_DIR}/sem_vs_wyko_width_track_{TRACK}.png")
