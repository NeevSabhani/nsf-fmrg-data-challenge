"""
Shared data loaders and the original (v3) pair builder.

Kept mainly for its loaders — load_wyko_asc, robust_line_detrend_profile
and the SEM tile helpers are imported by the v8 builder, the diagnostics
and the figure script. Its own width extractor is superseded: it
thresholds at a noise floor rather than a track edge. See
build_pairs_local_v8.py.

Run: python build_pairs_local.py
"""
import re
import gc
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import uniform_filter
from PIL import Image, ImageOps

# ------------------------------------------------------------------
PROJECT_DIR = Path("./")
RAW_DIR = PROJECT_DIR / "data" / "raw"
THERMAL_DIR = RAW_DIR / "thermal"
SEM_DIR = RAW_DIR / "sem"
HEIGHT_DIR = RAW_DIR / "height_maps"

CACHE_DIR = PROJECT_DIR / "processed_data" / "cache_v3"
QA_DIR = PROJECT_DIR / "processed_data" / "qa_v3"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
QA_DIR.mkdir(parents=True, exist_ok=True)

TRACK_IDS = [8, 10, 14, 21]
TRACK_LASER_POWER_W = {8: 200, 10: 300, 14: 350, 21: 400}

COMMON_X_START_MM = 20.0
COMMON_X_END_MM = 100.0
THERMAL_FPS = 50.0
SCAN_SPEED_MM_PER_S = 10.0
THERMAL_MM_PER_FRAME = SCAN_SPEED_MM_PER_S / THERMAL_FPS
EXTRACTED_THERMAL_FRAMES = int(round((COMMON_X_END_MM - COMMON_X_START_MM) / THERMAL_MM_PER_FRAME))
SEM_TILE_WIDTH_MM = 6.41

EDGE_MARGIN_MM = 8.0
K_THERMAL = 2
THERMAL_RESIZE = 64
SEM_PATCH_SIZE = 96

MIN_VALID_PROFILE_ROWS = 40   # below this, the y-profile is too gappy to trust
MIN_RUN_ROWS = 3              # elevated runs shorter than this are treated as noise


# ------------------------------------------------------------------
# Core loaders (1:1 with the repo's notebook 02 logic)
# ------------------------------------------------------------------
def natural_key(s):
    s = str(s)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def find_track_file(root, track_id, suffixes):
    root = Path(root)
    suffixes = {s.lower() for s in suffixes}
    matches = []
    if not root.exists():
        return None
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in suffixes:
            name = p.name.lower()
            if re.search(rf"(^|[_\-\s]){track_id}($|[_\-\s\.])", name):
                matches.append(p)
    matches = sorted(matches, key=natural_key)
    return matches[0] if matches else None


def _loadmat_any(path):
    path = Path(path)
    try:
        data = loadmat(path)
        return {k: v for k, v in data.items() if not k.startswith("__")}
    except NotImplementedError:
        import h5py
        out = {}
        with h5py.File(path, "r") as f:
            def visit(name, obj):
                if hasattr(obj, "shape"):
                    try:
                        out[name] = np.array(obj)
                    except Exception:
                        pass
            f.visititems(visit)
        return out


def find_thermal_array(mat_dict):
    candidates = []
    for k, v in mat_dict.items():
        arr = np.asarray(v)
        arr = np.squeeze(arr)
        if arr.ndim not in (3, 4):
            continue
        if not np.issubdtype(arr.dtype, np.number):
            continue
        if arr.ndim == 4:
            small_dims = [i for i, d in enumerate(arr.shape) if d in (1, 3, 4)]
            if small_dims:
                arr = np.take(arr, indices=0, axis=small_dims[-1])
                arr = np.squeeze(arr)
        if arr.ndim != 3:
            continue
        score = arr.size * (10 if 400 in arr.shape else 1)
        candidates.append((score, k, arr))
    if not candidates:
        raise ValueError("No thermal-like array found in MAT file.")
    candidates.sort(key=lambda x: x[0], reverse=True)
    key, arr = candidates[0][1], candidates[0][2]
    shape = arr.shape
    if shape[0] == shape[1] and shape[2] != shape[0]:
        arr_t = np.moveaxis(arr, 2, 0)
    elif shape[1] == shape[2]:
        arr_t = arr
    else:
        arr_t = np.moveaxis(arr, int(np.argmax(shape)), 0)
    return np.asarray(arr_t, dtype=np.float32), key


def thermal_frame_score(frames, top_percentile=99.5):
    return np.array([np.nanpercentile(fr, top_percentile) for fr in frames], dtype=np.float64)


def largest_true_run(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None, None
    idx = np.flatnonzero(mask)
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    stops = np.r_[idx[breaks] + 1, idx[-1] + 1]
    j = int(np.argmax(stops - starts))
    return int(starts[j]), int(stops[j])


def detect_laser_on_interval(frames):
    score = thermal_frame_score(frames)
    n = len(score)
    pre = score[: max(5, n // 10)]
    med = np.nanmedian(pre)
    mad = 1.4826 * np.nanmedian(np.abs(pre - med))
    range_thr = np.nanmin(score) + 0.20 * (np.nanmax(score) - np.nanmin(score))
    mad_thr = med + 8.0 * max(mad, 1e-12)
    threshold = max(range_thr, mad_thr)
    mask = score > threshold
    on_start, on_stop = largest_true_run(mask)
    if on_start is None:
        raise ValueError("Could not detect laser-on interval.")
    return on_start, on_stop, score, threshold


def extract_final_thermal_frames(track_id):
    path = find_track_file(THERMAL_DIR, track_id, [".mat"])
    if path is None:
        raise FileNotFoundError(f"No MAT file found for track {track_id} in {THERMAL_DIR}")
    mat = _loadmat_any(path)
    frames, key = find_thermal_array(mat)
    on_start, on_stop, score, threshold = detect_laser_on_interval(frames)
    stop_idx = int(on_stop)
    start_idx = max(0, stop_idx - EXTRACTED_THERMAL_FRAMES)
    segment = frames[start_idx:stop_idx]
    indices = np.arange(start_idx, stop_idx)
    x_mm_center = COMMON_X_END_MM - ((stop_idx - indices) - 0.5) * THERMAL_MM_PER_FRAME
    return {"frames": segment, "x_mm_center": x_mm_center}


def get_sem_tile_paths(track_id):
    root = SEM_DIR / f"SEM_{track_id}" / "PlainImages"
    if not root.exists():
        return []
    suffixes = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in suffixes]
    return sorted(files, key=natural_key)


def load_sem_tile(path):
    return np.asarray(ImageOps.grayscale(Image.open(path)))


def parse_wyko_header(path):
    header = {}
    with open(path, "r", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0].lower() == "x" and parts[1].lower() == "size":
                header["x_size"] = int(float(parts[2]))
            elif len(parts) >= 3 and parts[0].lower() == "y" and parts[1].lower() == "size":
                header["y_size"] = int(float(parts[2]))
            elif parts and parts[0].lower() == "pixel_size":
                header["pixel_size_mm"] = float(parts[-1])
            if parts and parts[0].upper() == "RAW_DATA":
                break
    return header


def load_wyko_asc(track_id, crop_to_common=True):
    path = find_track_file(HEIGHT_DIR, track_id, [".asc", ".txt"])
    if path is None:
        raise FileNotFoundError(f"No ASC/TXT file found for track {track_id} in {HEIGHT_DIR}")
    header = parse_wyko_header(path)
    x_size = int(header["x_size"])
    y_size = int(header["y_size"])
    pixel = float(header.get("pixel_size_mm", 0.003982))
    n_expected = x_size * y_size

    z_mm_flat = np.empty(n_expected, dtype=np.float32)
    z_mm_flat.fill(np.nan)

    count = 0
    in_raw = False
    with open(path, "r", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if not in_raw:
                if parts and parts[0].upper() == "RAW_DATA":
                    in_raw = True
                continue
            if len(parts) < 3:
                continue
            z_tok = parts[2]
            z_mm_flat[count] = np.nan if z_tok.lower() == "bad" else float(z_tok) * 1e-6
            count += 1
            if count >= n_expected:
                break

    if count < n_expected:
        warnings.warn(f"Read {count} rows but expected {n_expected}. Results may be truncated.")

    Z_x_y = z_mm_flat.reshape((x_size, y_size))
    Z_yx = Z_x_y.T
    x_local = np.arange(x_size, dtype=np.float64) * pixel
    y_mm = np.arange(y_size, dtype=np.float64) * pixel
    x_actual_raw = 100.0 - x_local
    sort_idx = np.argsort(x_actual_raw)
    x_actual = x_actual_raw[sort_idx]
    Z_yx = Z_yx[:, sort_idx]

    if crop_to_common:
        mask = (x_actual >= COMMON_X_START_MM) & (x_actual <= COMMON_X_END_MM)
        x_actual = x_actual[mask]
        Z_yx = Z_yx[:, mask]

    return {"Z_mm": Z_yx, "x_actual_mm": x_actual, "y_mm": y_mm}


# ------------------------------------------------------------------
# SEM masking (texture-variance band detection) — QA figures verify it
# ------------------------------------------------------------------
def detect_track_band(img, win=15, min_frac=0.08, max_frac=0.45):
    img = img.astype(np.float32)
    mean = uniform_filter(img, size=win)
    mean_sq = uniform_filter(img ** 2, size=win)
    local_var = np.clip(mean_sq - mean ** 2, 0, None)
    row_texture = np.mean(local_var, axis=1)

    h = img.shape[0]
    thresh = np.percentile(row_texture, 30)
    below = row_texture < thresh

    best_start, best_len, cur_start, cur_len = None, 0, None, 0
    for i, b in enumerate(below):
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
        return int(h * 0.4), int(h * 0.6)

    row_start, row_end = best_start, best_start + best_len
    min_px, max_px = int(h * min_frac), int(h * max_frac)
    band_len = row_end - row_start
    if band_len < min_px:
        pad = (min_px - band_len) // 2
        row_start, row_end = max(0, row_start - pad), min(h, row_end + pad)
    elif band_len > max_px:
        c = (row_start + row_end) // 2
        row_start, row_end = c - max_px // 2, c + max_px // 2
    return row_start, row_end


def sem_tile_index_for_x(x_query, n_tiles):
    """Tile 0 (file 01) spans [100 - 6.41, 100]; tile i spans
    [100 - (i+1)*6.41, 100 - i*6.41]. floor() is the correct mapping."""
    idx = int(np.floor((100.0 - x_query) / SEM_TILE_WIDTH_MM))
    return max(0, min(n_tiles - 1, idx))


def get_sem_patch_masked_v3(track_id, x_query, tile_paths, tile_cache):
    tile_idx = sem_tile_index_for_x(x_query, len(tile_paths))
    if tile_idx not in tile_cache:
        img = load_sem_tile(tile_paths[tile_idx]).astype(np.float32)
        row_start, row_end = detect_track_band(img)
        mask = np.zeros(img.shape, dtype=np.float32)
        mask[row_start:row_end, :] = 1.0
        fill_val = np.median(img[mask == 0]) if np.any(mask == 0) else np.median(img)
        img_filled = img.copy()
        img_filled[mask == 1] = fill_val
        img_r = np.array(Image.fromarray(img_filled).resize((SEM_PATCH_SIZE, SEM_PATCH_SIZE)))
        mask_r = np.array(Image.fromarray(mask).resize((SEM_PATCH_SIZE, SEM_PATCH_SIZE), Image.NEAREST))
        tile_cache[tile_idx] = np.stack([img_r, mask_r], axis=0).astype(np.float32)
    return tile_cache[tile_idx]


# ------------------------------------------------------------------
# Width extraction v3: robust per-profile detrend + adaptive threshold
# + largest-contiguous-run width
# ------------------------------------------------------------------
def robust_line_detrend_profile(profile, y):
    """Iteratively fit a line to the y-profile, excluding elevated rows,
    and subtract it. Returns (detrended, sigma_noise_um) or (None, None)."""
    valid = ~np.isnan(profile)
    if valid.sum() < MIN_VALID_PROFILE_ROWS:
        return None, None
    yy = y[valid]
    pp = profile[valid]
    keep = np.ones(pp.size, dtype=bool)
    coef = None
    for _ in range(3):
        if keep.sum() < MIN_VALID_PROFILE_ROWS // 2:
            break
        coef = np.polyfit(yy[keep], pp[keep], 1)
        resid = pp - np.polyval(coef, yy)
        cutoff = np.percentile(resid, 70)  # keep the lower 70% (substrate rows)
        keep = resid <= cutoff
    if coef is None:
        return None, None
    detrended = profile - np.polyval(coef, y)
    substrate_resid = (pp - np.polyval(coef, yy))[keep]
    sigma = 1.4826 * np.median(np.abs(substrate_resid - np.median(substrate_resid)))
    return detrended, float(max(sigma, 1e-6))


def extract_local_width_v3(height_data, x_query, window_mm=0.3,
                           floor_thresh_um=2.0, sigma_mult=3.0):
    x = height_data["x_actual_mm"]
    y = height_data["y_mm"]
    Z = height_data["Z_mm"]
    col_mask = np.abs(x - x_query) <= window_mm
    if not col_mask.any():
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        profile = np.nanmedian(Z[:, col_mask] * 1000.0, axis=1)  # -> um
    detrended, sigma = robust_line_detrend_profile(profile, y)
    if detrended is None:
        return np.nan
    thresh = max(floor_thresh_um, sigma_mult * sigma)
    elevated = np.zeros_like(detrended, dtype=bool)
    ok = ~np.isnan(detrended)
    elevated[ok] = detrended[ok] > thresh
    start, stop = largest_true_run(elevated)
    if start is None or (stop - start) < MIN_RUN_ROWS:
        return 0.0
    return float(y[stop - 1] - y[start])


def resize_thermal_stack(frames):
    out = np.zeros((frames.shape[0], THERMAL_RESIZE, THERMAL_RESIZE), dtype=np.float32)
    for i, f in enumerate(frames):
        out[i] = np.array(Image.fromarray(f.astype(np.float32)).resize((THERMAL_RESIZE, THERMAL_RESIZE)))
    return out


def build_pairs_for_track(track_id, k=K_THERMAL, edge_margin=EDGE_MARGIN_MM):
    thermal = extract_final_thermal_frames(track_id)
    frames = thermal["frames"]
    x_mm = thermal["x_mm_center"]
    n = len(x_mm)

    height_data = load_wyko_asc(track_id, crop_to_common=True)
    sem_tiles = get_sem_tile_paths(track_id)
    if not sem_tiles:
        raise FileNotFoundError(f"No SEM tiles found for track {track_id} under {SEM_DIR}")
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

        S_ij = get_sem_patch_masked_v3(track_id, xj, sem_tiles, tile_cache)
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
    }, sem_tiles


# ------------------------------------------------------------------
# QA outputs
# ------------------------------------------------------------------
def save_mask_qa_figure(track_id, sem_tiles):
    picks = [0, len(sem_tiles) // 2, len(sem_tiles) - 1]
    fig, axes = plt.subplots(1, len(picks), figsize=(4.5 * len(picks), 3.2))
    for ax, ti in zip(np.atleast_1d(axes), picks):
        img = load_sem_tile(sem_tiles[ti]).astype(np.float32)
        rs, re_ = detect_track_band(img)
        ax.imshow(img, cmap="gray")
        ax.axhline(rs, color="red", lw=1.5)
        ax.axhline(re_, color="red", lw=1.5)
        ax.set_title(f"{sem_tiles[ti].stem}\nmask rows [{rs}, {re_}]", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Track {track_id}: detected mask band (red lines) — VERIFY these cover the track")
    fig.tight_layout()
    out = QA_DIR / f"mask_qa_track_{track_id}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def save_width_qa_figure(track_id, X, G):
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(X, G, ".", markersize=3)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("local width (mm)")
    ax.set_title(f"Track {track_id}: extracted local width (v3)")
    fig.tight_layout()
    out = QA_DIR / f"width_qa_track_{track_id}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("THERMAL_DIR:", THERMAL_DIR, THERMAL_DIR.exists())
    print("SEM_DIR:", SEM_DIR, SEM_DIR.exists())
    print("HEIGHT_DIR:", HEIGHT_DIR, HEIGHT_DIR.exists())
    print()

    for track_id in TRACK_IDS:
        print(f"Building track {track_id}...")
        d, sem_tiles = build_pairs_for_track(track_id)
        G = d["G"]
        n = len(G)
        n_nan = int(np.sum(np.isnan(G)))
        n_zero = int(np.sum(G == 0.0))
        n_big = int(np.sum(G > 1.5))
        med = float(np.nanmedian(G))
        p95 = float(np.nanpercentile(G, 95))
        print(f"  T{d['T'].shape} S{d['S'].shape} M{d['M'].shape} G{d['G'].shape}")
        print(f"  widths: median {med:.3f} mm | p95 {p95:.3f} mm | "
              f"NaN {n_nan}/{n} | exactly-zero {n_zero}/{n} | >1.5mm {n_big}/{n}")
        if n_zero / n > 0.15:
            print(f"  *** WARNING: {100*n_zero/n:.0f}% zero widths — extraction still suspect for this track")
        if n_big / n > 0.05:
            print(f"  *** WARNING: {100*n_big/n:.0f}% widths >1.5mm — outlier contamination suspected")

        out_path = CACHE_DIR / f"track_{track_id}_v3.npz"
        np.savez_compressed(out_path, T=d["T"], S=d["S"], M=d["M"], G=d["G"], X=d["X"])
        print(f"  Saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

        print("  QA:", save_mask_qa_figure(track_id, sem_tiles))
        print("  QA:", save_width_qa_figure(track_id, d["X"], d["G"]))

        del d
        gc.collect()

    print("\nAll done. INSPECT the figures in", QA_DIR,
          "before training — especially the mask overlays.")