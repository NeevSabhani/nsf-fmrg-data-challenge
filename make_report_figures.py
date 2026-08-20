"""
Publication figures for the report.

Three figures, both sized to the 6.5in text column of a 1"-margin US Letter
page, with Arial at >=8pt so they stay legible when placed inline:

  fig1_labels_and_bug.png  — why every model through v7 failed: the
      3-sigma noise cut sits above the track's own crown, and the label
      distribution it produced (25% exact zeros) vs the half-max fix.
  fig2_results.png         — what the fixed pipeline supports: width vs
      power, leave-one-power-out parity for the thermal-only model with
      out-of-fold uncertainty, and the within-track null result.

Run: python make_report_figures.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_pairs_local import load_wyko_asc, robust_line_detrend_profile

OUT_DIR = "./report/figures"
os.makedirs(OUT_DIR, exist_ok=True)

TRACK_POWER = {8: 200, 10: 300, 14: 350, 21: 400}
POWER_COLOR = {200: "tab:blue", 300: "tab:orange", 350: "tab:green", 400: "tab:red"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})


def half_max_width(detrended, y):
    d = np.where(np.isnan(detrended), -np.inf, detrended)
    peak = np.nanmax(detrended)
    if not np.isfinite(peak) or peak <= 0:
        return np.nan, np.nan
    half = peak * 0.5
    above = d >= half
    best, cur = 0, 0
    for b in above:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    dy = abs(float(np.median(np.diff(y))))
    return best * dy, half


# ---------------------------------------------------------------- figure 1
fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.0))

# (a) the smoking gun: 3-sigma cut above the crown at track 10, x=85mm
hd = load_wyko_asc(10, crop_to_common=True)
x, y = hd["x_actual_mm"], hd["y_mm"]
m = np.abs(x - 85.0) <= 0.3
profile = np.nanmedian(hd["Z_mm"][:, m] * 1000.0, axis=1)
detrended, sigma = robust_line_detrend_profile(profile, y)
thresh = max(2.0, 3.0 * sigma)
hm_w, half = half_max_width(detrended, y)
peak = float(np.nanmax(detrended))

ax = axes[0]
# v3 thresholded the raw detrended profile; v8 median-filters first, so show
# both — otherwise the "crown" is just whichever noise spike ran highest.
from scipy.ndimage import median_filter
smooth = median_filter(np.nan_to_num(detrended, nan=0.0), size=7)
ax.plot(y, detrended, "-", color="0.75", lw=0.6, label="raw profile")
ax.plot(y, smooth, "k-", lw=1.1, label="smoothed (v8)")
ax.axhline(thresh, color="tab:red", ls="--", lw=1.1,
           label=f"3$\\sigma$ cut, {thresh:.1f} $\\mu$m")
ax.axhline(half, color="tab:green", ls="--", lw=1.1,
           label=f"half-max, {half:.1f} $\\mu$m")
ax.set_xlabel("y (mm)")
ax.set_ylabel("detrended height ($\\mu$m)")
ax.set_title("(a) Height profile, track 10, x = 85 mm")
ax.legend(loc="lower left", framealpha=0.9, fontsize=5.5)
ax.set_ylim(-11, 11)

# (b) what that did to the labels, per power
ax = axes[1]
zero_frac, v3_med, v8_med, powers = [], [], [], []
for t in sorted(TRACK_POWER, key=lambda k: TRACK_POWER[k]):
    g3 = np.load(f"./processed_data/cache_v3/track_{t}_v3.npz")["G"]
    g8 = np.load(f"./processed_data/cache_v8/track_{t}_v8.npz")["G"]
    powers.append(TRACK_POWER[t])
    zero_frac.append(float(np.mean(g3 <= 1e-9)) * 100)
    v3_med.append(float(np.median(g3)))
    v8_med.append(float(np.nanmedian(g8)))
w = 12
ax.bar(np.array(powers) - w / 2, v3_med, width=w, color="tab:red", alpha=0.75,
       label="v3 (3$\\sigma$ noise cut)")
ax.bar(np.array(powers) + w / 2, v8_med, width=w, color="tab:green", alpha=0.85,
       label="v8 (half-max)")
for p, z, v in zip(powers, zero_frac, v3_med):
    ax.text(p - w / 2, v + 0.01, f"{z:.0f}%\nzeros", ha="center", va="bottom",
            fontsize=5.5, color="tab:red")
ax.set_xlabel("laser power (W)")
ax.set_ylabel("median width (mm)")
ax.set_title("(b) Labels before vs after the fix")
ax.set_xticks(powers)
ax.set_ylim(0, 0.72)
ax.legend(loc="upper right", framealpha=0.9, fontsize=5.5)

# (c) two independent metrologies disagree pointwise
ax = axes[2]
d8 = np.load("./processed_data/cache_v8/track_10_v8.npz")
gw, sw = d8["G"], d8["W_sem"]
ok = np.isfinite(gw) & np.isfinite(sw)
r = float(np.corrcoef(gw[ok], sw[ok])[0, 1])
ax.scatter(gw[ok], sw[ok], s=5, alpha=0.45, color="tab:purple", edgecolors="none")
ax.set_xlabel("Wyko half-max width (mm)")
ax.set_ylabel("SEM band width (mm)")
ax.set_title("(c) Wyko vs SEM width, track 10")
ax.text(0.04, 0.95, f"pointwise corr {r:+.3f}", transform=ax.transAxes,
        fontsize=6.5, va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.5))

fig.tight_layout(pad=0.4)
fig.savefig(f"{OUT_DIR}/fig1_labels_and_bug.png", dpi=300)
print(f"Saved {OUT_DIR}/fig1_labels_and_bug.png  "
      f"(track10 x=85: crown {peak:.2f}um, 3sigma cut {thresh:.2f}um, half-max w {hm_w:.3f}mm)")
print(f"  pointwise Wyko-vs-SEM corr on track 10: {r:+.3f}")

# ---------------------------------------------------------------- figure 2
d = np.load("./processed_data/model_outputs_v9/segment_predictions.npz")
pm, tm = d["pred_mean_thermal"], d["true_mean"]
sig, tracks, powers_seg = d["sigma_thermal"], d["tracks"], d["powers"]

fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.0))

# (a) the physical trend the model has to reproduce
ax = axes[0]
for t in sorted(TRACK_POWER, key=lambda k: TRACK_POWER[k]):
    mk = tracks == t
    p = TRACK_POWER[t]
    ax.scatter(np.full(mk.sum(), p), tm[mk], s=9, alpha=0.5,
               color=POWER_COLOR[p], edgecolors="none")
means = [tm[tracks == t].mean() for t in sorted(TRACK_POWER, key=lambda k: TRACK_POWER[k])]
ps = sorted(TRACK_POWER.values())
ax.plot(ps, means, "k.-", lw=1.2, markersize=6, label="track mean")
ax.set_xlabel("laser power (W)")
ax.set_ylabel("segment width (mm)")
ax.set_title("(a) Measured width vs power\n(4 mm segments, v8 labels)")
ax.legend(loc="upper right", framealpha=0.9)

# (b) the headline: leave-one-power-out parity, thermal features only
ax = axes[1]
lo, hi = min(tm.min(), pm.min()) - 0.02, max(tm.max(), pm.max()) + 0.02
ax.plot([lo, hi], [lo, hi], "k--", lw=0.9, label="perfect")
for t in sorted(TRACK_POWER, key=lambda k: TRACK_POWER[k]):
    mk = tracks == t
    p = TRACK_POWER[t]
    ax.errorbar(tm[mk], pm[mk], yerr=sig[mk], fmt="o", markersize=3.2,
                elinewidth=0.6, capsize=0, alpha=0.75,
                color=POWER_COLOR[p], label=f"{p} W")
corr = float(np.corrcoef(pm, tm)[0, 1])
r2 = float(1 - np.sum((tm - pm) ** 2) / np.sum((tm - tm.mean()) ** 2))
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("measured width (mm)")
ax.set_ylabel("predicted width (mm)")
ax.set_title(f"(b) Leave-one-power-out, thermal only\n$R^2$ {r2:+.3f}, corr {corr:+.3f}, $\\pm1\\sigma$ shown")
ax.legend(loc="lower right", framealpha=0.9, ncol=2, fontsize=5.5,
          handletextpad=0.3, columnspacing=0.8)

# (c) the honest limit: nothing survives once the power level is removed
ax = axes[2]
pc, yc = pm.copy(), tm.copy()
for t in TRACK_POWER:
    mk = tracks == t
    pc[mk] -= pc[mk].mean(); yc[mk] -= yc[mk].mean()
for t in sorted(TRACK_POWER, key=lambda k: TRACK_POWER[k]):
    mk = tracks == t
    ax.scatter(yc[mk], pc[mk], s=9, alpha=0.6,
               color=POWER_COLOR[TRACK_POWER[t]], edgecolors="none")
wc = float(np.corrcoef(pc, yc)[0, 1])
lim = max(np.abs(np.r_[pc, yc])) * 1.1
ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.9)
ax.axhline(0, color="0.7", lw=0.5); ax.axvline(0, color="0.7", lw=0.5)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("measured deviation (mm)")
ax.set_ylabel("predicted deviation (mm)")
ax.set_title(f"(c) Within-track variation\ncorr {wc:+.3f} — not predictable")

fig.tight_layout(pad=0.4)
fig.savefig(f"{OUT_DIR}/fig2_results.png", dpi=300)
print(f"Saved {OUT_DIR}/fig2_results.png  (thermal-only R2 {r2:+.3f}, corr {corr:+.3f}, "
      f"within-track {wc:+.3f})")

# ---------------------------------------------------------------- figure 3
# Uncertainty calibration. Out-of-fold sigma vs the in-fold sigma the same
# procedure would have produced, to show why the distinction matters.
from scipy.stats import norm

z_oof = (tm - pm) / sig
levels = np.linspace(0.05, 0.99, 60)
emp_oof = [np.mean(np.abs(z_oof) <= norm.ppf(0.5 + L / 2)) for L in levels]

fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.1))

ax = axes[0]
ax.plot([0, 1], [0, 1], "k--", lw=0.9, label="perfect calibration")
ax.plot(levels, emp_oof, "-", color="tab:green", lw=1.6, label="out-of-fold $\\sigma$")
for L, c in [(0.68, "tab:blue"), (0.95, "tab:orange")]:
    e = np.mean(np.abs(z_oof) <= norm.ppf(0.5 + L / 2))
    ax.plot([L], [e], "o", color=c, markersize=4.5, zorder=5)
    ax.annotate(f"{L:.2f}$\\rightarrow${e:.3f}", (L, e), textcoords="offset points",
                xytext=(-6, -12), fontsize=6, color=c, ha="right")
ax.set_xlabel("nominal coverage")
ax.set_ylabel("empirical coverage")
ax.set_title("(a) Reliability curve")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(loc="upper left", fontsize=6, framealpha=0.9)

ax = axes[1]
ax.hist(z_oof, bins=14, density=True, color="tab:green", alpha=0.55,
        edgecolor="white", linewidth=0.4, label="out-of-fold $z$")
xs = np.linspace(-3.5, 3.5, 200)
ax.plot(xs, norm.pdf(xs), "k-", lw=1.2, label="$\\mathcal{N}(0,1)$")
ax.set_xlabel("standardised residual $z=(y-\\hat{y})/\\sigma$")
ax.set_ylabel("density")
ax.set_title("(b) Residual distribution")
ax.legend(loc="upper right", fontsize=6, framealpha=0.9)

fig.tight_layout(pad=0.4)
fig.savefig(f"{OUT_DIR}/fig3_calibration.png", dpi=300)
print(f"Saved {OUT_DIR}/fig3_calibration.png  (z mean {z_oof.mean():+.3f}, "
      f"z std {z_oof.std():.3f})")
