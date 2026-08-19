"""
v5: fixes the two structural problems in the SEM branch found via
permutation importance in interpret_v3.py (near-zero importance for the
SEM branch on the trained v3/v4 models):

  1. get_sem_patch_masked_v3 blanked out the track band's own pixels
     (overwritten with the substrate's median value) before the model
     ever saw them -- deleting the one signal (track surface texture)
     that should carry substrate-driven width variation.
  2. Each 6.41mm-wide SEM tile was reused UNCHANGED for every sample
     that fell inside it (samples are 0.2mm apart -> ~32 samples share
     one cached image), so the SEM branch had almost no resolution at
     the scale needed to explain point-to-point width variation.

v5 crops a real ~1mm local window around each sample's x-position from
the full-resolution tile (1024px wide / 6.41mm =~ 0.006mm/px, so a 1mm
crop is ~160px -- plenty of headroom) and keeps the true pixel content,
no blanking. Thermal extraction, width extraction, and the train/val/test
split are all unchanged from v3; this only replaces the SEM patch step.

Run: python build_pairs_local_v5.py
Requires: numpy, pillow, matplotlib (for QA)
"""
import gc
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from build_pairs_local import (
    TRACK_IDS, TRACK_LASER_POWER_W, COMMON_X_START_MM, COMMON_X_END_MM,
    EDGE_MARGIN_MM, K_THERMAL, SEM_PATCH_SIZE, SEM_TILE_WIDTH_MM,
    extract_final_thermal_frames, resize_thermal_stack,
    load_wyko_asc, extract_local_width_v3,
    get_sem_tile_paths, load_sem_tile, detect_track_band, sem_tile_index_for_x,
)

CACHE_DIR = Path("./processed_data/cache_v5")
QA_DIR = Path("./processed_data/qa_v5")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
QA_DIR.mkdir(parents=True, exist_ok=True)

CROP_WIDTH_MM = 1.0  # total physical width of the local SEM crop per sample


def get_sem_patch_local_v5(track_id, x_query, tile_paths, tile_cache):
    tile_idx = sem_tile_index_for_x(x_query, len(tile_paths))
    if tile_idx not in tile_cache:
        img = load_sem_tile(tile_paths[tile_idx]).astype(np.float32)
        row_start, row_end = detect_track_band(img)
        mask = np.zeros(img.shape, dtype=np.float32)
        mask[row_start:row_end, :] = 1.0
        tile_left = 100.0 - (tile_idx + 1) * SEM_TILE_WIDTH_MM
        tile_cache[tile_idx] = (img, mask, tile_left)
    img, mask, tile_left = tile_cache[tile_idx]

    H, W = img.shape
    px_per_mm = W / SEM_TILE_WIDTH_MM
    frac = min(max((x_query - tile_left) / SEM_TILE_WIDTH_MM, 0.0), 1.0)
    col_center = int(frac * W)
    half_px = int(round(CROP_WIDTH_MM * 0.5 * px_per_mm))
    col_lo, col_hi = col_center - half_px, col_center + half_px
    pad_lo, pad_hi = max(0, -col_lo), max(0, col_hi - W)
    col_lo_c, col_hi_c = max(0, col_lo), min(W, col_hi)

    img_crop = img[:, col_lo_c:col_hi_c]
    mask_crop = mask[:, col_lo_c:col_hi_c]
    if pad_lo or pad_hi:
        img_crop = np.pad(img_crop, ((0, 0), (pad_lo, pad_hi)), mode="edge")
        mask_crop = np.pad(mask_crop, ((0, 0), (pad_lo, pad_hi)), mode="edge")

    img_r = np.array(Image.fromarray(img_crop).resize((SEM_PATCH_SIZE, SEM_PATCH_SIZE)))
    mask_r = np.array(Image.fromarray(mask_crop).resize((SEM_PATCH_SIZE, SEM_PATCH_SIZE), Image.NEAREST))
    return np.stack([img_r, mask_r], axis=0).astype(np.float32)


def build_pairs_for_track_v5(track_id, k=K_THERMAL, edge_margin=EDGE_MARGIN_MM):
    thermal = extract_final_thermal_frames(track_id)
    frames = thermal["frames"]
    x_mm = thermal["x_mm_center"]
    n = len(x_mm)

    height_data = load_wyko_asc(track_id, crop_to_common=True)
    sem_tiles = get_sem_tile_paths(track_id)
    if not sem_tiles:
        raise FileNotFoundError(f"No SEM tiles for track {track_id}")
    power = TRACK_LASER_POWER_W[track_id]
    tile_cache = {}

    T_list, S_list, M_list, G_list, X_list = [], [], [], [], []
    for j in range(n):
        xj = x_mm[j]
        if xj < COMMON_X_START_MM + edge_margin or xj > COMMON_X_END_MM - edge_margin:
            continue
        lo, hi = max(0, j - k), min(n, j + k + 1)
        T_ij = resize_thermal_stack(frames[lo:hi])
        want = 2 * k + 1
        if T_ij.shape[0] < want:
            pad = want - T_ij.shape[0]
            if lo == 0:
                T_ij = np.concatenate([T_ij[:1]] * pad + [T_ij], axis=0)
            else:
                T_ij = np.concatenate([T_ij] + [T_ij[-1:]] * pad, axis=0)

        S_ij = get_sem_patch_local_v5(track_id, xj, sem_tiles, tile_cache)
        G_j = extract_local_width_v3(height_data, xj)

        T_list.append(T_ij)
        S_list.append(S_ij)
        M_list.append([xj / 100.0, power / 400.0])
        G_list.append(G_j)
        X_list.append(xj)

    return {
        "T": np.stack(T_list).astype(np.float32),
        "S": np.stack(S_list).astype(np.float32),
        "M": np.array(M_list, dtype=np.float32),
        "G": np.array(G_list, dtype=np.float32),
        "X": np.array(X_list, dtype=np.float32),
    }


def save_sem_crop_qa(track_id, d, n_show=6):
    """Show a handful of consecutive local crops so we can visually verify
    they actually differ from each other (unlike the v3 per-tile-constant
    patches) and contain real (unblanked) track texture."""
    idx = np.linspace(0, len(d["X"]) - 1, n_show).astype(int)
    fig, axes = plt.subplots(1, n_show, figsize=(2.6 * n_show, 3.2))
    for ax, i in zip(axes, idx):
        ax.imshow(d["S"][i, 0], cmap="gray")
        ax.set_title(f"x={d['X'][i]:.2f}mm\nwidth={d['G'][i]:.3f}mm", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Track {track_id}: local SEM crops (v5) — should differ sample-to-sample, track band visible (not blanked)")
    fig.tight_layout()
    out = QA_DIR / f"sem_crop_qa_track_{track_id}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for track_id in TRACK_IDS:
        print(f"Building track {track_id} (v5)...")
        d = build_pairs_for_track_v5(track_id)
        out_path = CACHE_DIR / f"track_{track_id}_v5.npz"
        np.savez_compressed(out_path, T=d["T"], S=d["S"], M=d["M"], G=d["G"], X=d["X"])
        print(f"  T{d['T'].shape} S{d['S'].shape} saved to {out_path} "
              f"({out_path.stat().st_size / 1e6:.1f} MB)")
        print("  QA:", save_sem_crop_qa(track_id, d))
        del d
        gc.collect()
    print("\nDone. Inspect processed_data/qa_v5/sem_crop_qa_track_*.png before training.")
