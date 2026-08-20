"""
v9 — reframed target, built on the fixed v8 pipeline.

WHY REFRAME. Per-0.2mm pointwise width sits below the noise floor of the
available metrology: two INDEPENDENT good measurements of it (Wyko
half-max, SEM band thickness) correlate only +0.03..+0.20 with each
other, and every model plateaued at corr ~0.00 on held-out data even
after the label bug was fixed (see PROGRESS.md). Averaging over a
segment suppresses independent measurement noise as ~1/sqrt(n) while
preserving real process-driven structure, so this predicts SEGMENT-level
quantities instead:

    target 1: segment MEAN width          (the width itself)
    target 2: segment WIDTH VARIABILITY   (std within the segment --
                                           this is "local width variation"
                                           as a descriptor)

MODEL. ~16 segments/track x 4 tracks = 64 samples, so a CNN is
inappropriate; this uses ridge regression on physically-interpretable
scalar features. Feature groups are kept separate (thermal = process,
SEM = substrate) so their contributions can be compared directly, which
is what the "process- vs substrate-driven" judging criterion asks for.

VALIDATION. Leave-one-TRACK-out across all 4 tracks. Each fold trains on
3 laser powers and tests on the 4th, so the CV is simultaneously the
generalization test and the "robustness across laser powers" evidence.
Every fold is reported, including the hard extrapolation folds (200W and
400W are the range ends).

Run: python v9_segment_model.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE_DIR = "./processed_data/cache_v8"
OUT_DIR = "./processed_data/model_outputs_v9"
os.makedirs(OUT_DIR, exist_ok=True)

TRACKS = [8, 10, 14, 21]
TRACK_POWER = {8: 200, 10: 300, 14: 350, 21: 400}
SEGMENT_MM = 4.0
MIN_PTS_PER_SEG = 12
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


# ------------------------------------------------------------------
# Per-sample scalar features (thermal = process, SEM = substrate)
# ------------------------------------------------------------------
def thermal_features(T):
    """T: (N, 5, 64, 64), already in physical-fraction units."""
    peak = T.max(axis=(1, 2, 3))
    mean = T.mean(axis=(1, 2, 3))
    # melt-pool size proxies: fraction of pixels above fixed fractions of scale
    hot_hi = (T > 0.6).mean(axis=(1, 2, 3))
    hot_lo = (T > 0.3).mean(axis=(1, 2, 3))
    # cooling proxy: how much the frame peak decays across the 5-frame window
    fpeak = T.max(axis=(2, 3))               # (N, 5)
    cool = fpeak[:, 0] - fpeak[:, -1]
    slope = np.polyfit(np.arange(fpeak.shape[1]), fpeak.T, 1)[0]
    # spatial sharpness of the hot zone
    gy = np.abs(np.diff(T, axis=2)).mean(axis=(1, 2, 3))
    gx = np.abs(np.diff(T, axis=3)).mean(axis=(1, 2, 3))
    return np.column_stack([peak, mean, hot_hi, hot_lo, cool, slope, gy, gx]), \
        ["th_peak", "th_mean", "th_areahot", "th_areawarm", "th_cool", "th_slope", "th_grady", "th_gradx"]


def sem_features(S):
    """S: (N, 2, 96, 96) -- channel 0 image (0-255), channel 1 band mask."""
    img, mask = S[:, 0], S[:, 1]
    col_thick = mask.sum(axis=1)                      # (N, 96) band thickness per column
    band_mean = col_thick.mean(axis=1)
    band_std = col_thick.std(axis=1)                  # edge roughness within the patch
    band_rng = col_thick.max(axis=1) - col_thick.min(axis=1)
    present = (mask.sum(axis=(1, 2)) > 0).astype(np.float32)

    m = mask > 0.5
    inv = ~m
    def masked_stat(a, sel, fn):
        out = np.zeros(len(a), dtype=np.float32)
        for i in range(len(a)):
            v = a[i][sel[i]]
            out[i] = fn(v) if v.size > 5 else 0.0
        return out
    track_tex = masked_stat(img, m, np.std)           # melted surface texture
    sub_tex = masked_stat(img, inv, np.std)           # substrate texture
    track_lvl = masked_stat(img, m, np.mean)
    sub_lvl = masked_stat(img, inv, np.mean)
    contrast = track_lvl - sub_lvl
    return np.column_stack([band_mean, band_std, band_rng, present,
                            track_tex, sub_tex, contrast]), \
        ["sem_bandmean", "sem_bandstd", "sem_bandrng", "sem_present",
         "sem_tracktex", "sem_subtex", "sem_contrast"]


def build_segments():
    rows, meta = [], []
    feat_names = None
    for tid in TRACKS:
        d = np.load(f"{CACHE_DIR}/track_{tid}_v8.npz")
        T, S, G, X = d["T"], d["S"], d["G"], d["X"]
        Ft, tn = thermal_features(T)
        Fs, sn = sem_features(S)
        if feat_names is None:
            feat_names = tn + sn
        F = np.column_stack([Ft, Fs])

        edges = np.arange(X.min(), X.max() + SEGMENT_MM, SEGMENT_MM)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (X >= lo) & (X < hi)
            g = G[m]
            valid = ~np.isnan(g)
            if valid.sum() < MIN_PTS_PER_SEG:
                continue
            gv = g[valid]
            fseg = np.concatenate([np.nanmean(F[m], axis=0), np.nanstd(F[m], axis=0)])
            rows.append(fseg)
            meta.append({
                "track": tid, "power": TRACK_POWER[tid], "x_lo": float(lo), "x_hi": float(hi),
                "width_mean": float(gv.mean()), "width_std": float(gv.std()),
                "n": int(valid.sum()),
            })
    names = [f"{n}_mean" for n in feat_names] + [f"{n}_std" for n in feat_names]
    return np.array(rows, dtype=np.float64), meta, names


# ------------------------------------------------------------------
# Ridge regression (numpy; sklearn not installed in this env)
# ------------------------------------------------------------------
def ridge_fit(X, y, alpha):
    n, p = X.shape
    Xb = np.column_stack([np.ones(n), X])
    A = Xb.T @ Xb + alpha * np.eye(p + 1)
    A[0, 0] -= alpha  # do not penalize the intercept
    return np.linalg.solve(A, Xb.T @ y)


def ridge_predict(w, X):
    return np.column_stack([np.ones(len(X)), X]) @ w


def standardize(train, *others):
    mu, sd = train.mean(0), train.std(0)
    sd[sd < 1e-9] = 1.0
    return ((train - mu) / sd,) + tuple((o - mu) / sd for o in others)


def loo_track_cv(Xall, yall, tracks, feature_mask=None, label=""):
    """Leave-one-track-out. Inner LOO over the 3 training tracks picks alpha."""
    X = Xall[:, feature_mask] if feature_mask is not None else Xall
    preds = np.full(len(yall), np.nan)
    per_fold = {}
    for held in TRACKS:
        te = tracks == held
        tr = ~te
        inner_tracks = [t for t in TRACKS if t != held]
        best_alpha, best_err = None, np.inf
        for a in RIDGE_ALPHAS:
            errs = []
            for ho in inner_tracks:
                itr = tr & (tracks != ho)
                ite = tracks == ho
                Xi, Xj = standardize(X[itr], X[ite])
                w = ridge_fit(Xi, yall[itr], a)
                errs.append(np.mean(np.abs(ridge_predict(w, Xj) - yall[ite])))
            if np.mean(errs) < best_err:
                best_err, best_alpha = np.mean(errs), a
        Xtr, Xte = standardize(X[tr], X[te])
        w = ridge_fit(Xtr, yall[tr], best_alpha)
        preds[te] = ridge_predict(w, Xte)
        # Predictive sigma from OUT-OF-FOLD residuals on the 3 training
        # tracks, not in-fold residuals: in-fold residuals are optimistic
        # (the model saw those points) and badly underestimate the error
        # incurred when extrapolating to a held-out laser power.
        oof = []
        for ho in inner_tracks:
            itr = tr & (tracks != ho)
            ite = tracks == ho
            Xi, Xj = standardize(X[itr], X[ite])
            wi = ridge_fit(Xi, yall[itr], best_alpha)
            oof.append(yall[ite] - ridge_predict(wi, Xj))
        per_fold[held] = {
            "power": TRACK_POWER[held], "alpha": best_alpha,
            "mae": float(np.mean(np.abs(preds[te] - yall[te]))),
            "baseline_mae": float(np.mean(np.abs(yall[tr].mean() - yall[te]))),
            "resid_std_train": float(np.concatenate(oof).std()),
            "n_test": int(te.sum()),
        }
    mae = float(np.mean(np.abs(preds - yall)))
    base = float(np.mean([per_fold[t]["baseline_mae"] * per_fold[t]["n_test"] for t in TRACKS])
                 / np.mean([per_fold[t]["n_test"] for t in TRACKS]))
    corr = float(np.corrcoef(preds, yall)[0, 1])
    ss_res = np.sum((yall - preds) ** 2)
    ss_tot = np.sum((yall - yall.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)
    print(f"  {label:38s} MAE {mae:.4f} (baseline {base:.4f}) | corr {corr:+.3f} | R2 {r2:+.3f}")
    return preds, {"mae": mae, "baseline_mae": base, "corr": corr, "r2": r2, "folds": per_fold}


if __name__ == "__main__":
    Xall, meta, names = build_segments()
    tracks = np.array([m["track"] for m in meta])
    powers = np.array([m["power"] for m in meta], dtype=float)
    y_mean = np.array([m["width_mean"] for m in meta])
    y_std = np.array([m["width_std"] for m in meta])
    print(f"Built {len(meta)} segments ({SEGMENT_MM}mm each) across {len(TRACKS)} tracks, "
          f"{Xall.shape[1]} features")
    for t in TRACKS:
        m = tracks == t
        print(f"  track {t:2d} ({TRACK_POWER[t]}W): {m.sum():2d} segments | "
              f"width_mean {y_mean[m].mean():.3f} +/- {y_mean[m].std():.3f} | "
              f"width_std {y_std[m].mean():.3f}")

    n_th = 8  # thermal features come first in each half
    half = Xall.shape[1] // 2
    th_mask = np.zeros(Xall.shape[1], bool); th_mask[:n_th] = True; th_mask[half:half + n_th] = True
    sem_mask = ~th_mask

    results = {}
    print("\n=== Target 1: segment MEAN width (leave-one-track-out) ===")
    p_all, r_all = loo_track_cv(Xall, y_mean, tracks, None, "all features")
    p_th, r_th = loo_track_cv(Xall, y_mean, tracks, th_mask, "thermal only (process)")
    p_sem, r_sem = loo_track_cv(Xall, y_mean, tracks, sem_mask, "SEM only (substrate)")
    results["width_mean"] = {"all": r_all, "thermal": r_th, "sem": r_sem}

    print("\n=== Target 2: segment WIDTH VARIABILITY / std (leave-one-track-out) ===")
    v_all, rv_all = loo_track_cv(Xall, y_std, tracks, None, "all features")
    v_th, rv_th = loo_track_cv(Xall, y_std, tracks, th_mask, "thermal only (process)")
    v_sem, rv_sem = loo_track_cv(Xall, y_std, tracks, sem_mask, "SEM only (substrate)")
    results["width_std"] = {"all": rv_all, "thermal": rv_th, "sem": rv_sem}

    print("\n=== Per-power robustness (mean-width model, all features) ===")
    for t in TRACKS:
        f = r_all["folds"][t]
        flag = "OK " if f["mae"] < f["baseline_mae"] else "!! "
        print(f"  {flag}{f['power']}W (track {t:2d}): MAE {f['mae']:.4f} vs baseline "
              f"{f['baseline_mae']:.4f} | alpha {f['alpha']}")

    # Uncertainty: per-fold OUT-OF-FOLD residual std as the predictive sigma.
    # Computed for BOTH feature sets: the headline model is thermal-only, so
    # its coverage must be quoted rather than the all-feature model's.
    def coverage(res, preds):
        sig = np.array([res["folds"][m["track"]]["resid_std_train"] for m in meta])
        z = (y_mean - preds) / sig
        return sig, float(np.mean(np.abs(z) <= 1.0)), float(np.mean(np.abs(z) <= 2.0))

    sigma, cov1, cov2 = coverage(r_all, p_all)
    sigma_th, cov1_th, cov2_th = coverage(r_th, p_th)
    print(f"\nUncertainty (mean-width, all features): 68% {cov1:.3f} | 95% {cov2:.3f}")
    print(f"Uncertainty (mean-width, thermal only):  68% {cov1_th:.3f} | 95% {cov2_th:.3f}")
    results["coverage_1sigma"] = cov1
    results["coverage_2sigma"] = cov2
    results["coverage_1sigma_thermal"] = cov1_th
    results["coverage_2sigma_thermal"] = cov2_th

    # Within-track correlation: strip each track's mean from prediction and
    # truth. This separates the between-power effect (which works) from
    # within-track variation (which does not).
    def within_corr(preds):
        pc, yc = preds.copy(), y_mean.copy()
        for t in TRACKS:
            m = tracks == t
            pc[m] -= pc[m].mean(); yc[m] -= yc[m].mean()
        per = {int(t): float(np.corrcoef(pc[tracks == t], yc[tracks == t])[0, 1])
               for t in TRACKS}
        return float(np.corrcoef(pc, yc)[0, 1]), per

    wc_all, wc_all_per = within_corr(p_all)
    wc_th, wc_th_per = within_corr(p_th)
    print(f"Within-track corr (all features): {wc_all:+.3f}  per-track {wc_all_per}")
    print(f"Within-track corr (thermal only): {wc_th:+.3f}  per-track {wc_th_per}")
    results["within_track_corr"] = wc_all
    results["within_track_corr_thermal"] = wc_th
    results["within_track_corr_per_track_thermal"] = wc_th_per

    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    np.savez(f"{OUT_DIR}/segment_predictions.npz",
             pred_mean=p_all, true_mean=y_mean, pred_std=v_all, true_std=y_std,
             pred_mean_thermal=p_th, sigma_thermal=sigma_th,
             tracks=tracks, powers=powers, sigma=sigma,
             x_lo=np.array([m["x_lo"] for m in meta]))

    # ---------------- figures ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (pred, true, name) in zip(axes, [(p_all, y_mean, "segment mean width"),
                                             (v_all, y_std, "segment width variability")]):
        for t in TRACKS:
            m = tracks == t
            ax.errorbar(true[m], pred[m], fmt="o", ms=6, alpha=0.8, label=f"{TRACK_POWER[t]}W")
        lim = [min(true.min(), pred.min()), max(true.max(), pred.max())]
        ax.plot(lim, lim, "k--", lw=1, label="perfect")
        ax.set_xlabel(f"measured {name} (mm)"); ax.set_ylabel(f"predicted {name} (mm)")
        c = np.corrcoef(pred, true)[0, 1]
        ax.set_title(f"{name}\nleave-one-track-out, corr {c:+.3f}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/parity_plots.png", dpi=150)
    print("Saved", f"{OUT_DIR}/parity_plots.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for t in TRACKS:
        m = tracks == t
        ax.scatter(np.full(m.sum(), TRACK_POWER[t]), y_mean[m], alpha=0.6, s=28,
                   label=f"track {t}")
    pw = np.array([TRACK_POWER[t] for t in TRACKS], float)
    mw = np.array([y_mean[tracks == t].mean() for t in TRACKS])
    ax.plot(pw, mw, "k-o", lw=2, ms=8, label="track mean")
    ax.set_xlabel("laser power (W)"); ax.set_ylabel("segment mean width (mm)")
    ax.set_title("Track width vs laser power (v8 half-max labels)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/width_vs_power.png", dpi=150)
    print("Saved", f"{OUT_DIR}/width_vs_power.png")
