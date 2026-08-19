"""
v7 — best-effort model, combining everything that measurably worked plus
targeted fixes for the failure modes v6 exposed.

Data: processed_data/cache_v5/ (the local ~1mm SEM crop fix from
build_pairs_local_v5.py — the one data change with a proven benefit:
test MAE 0.084 -> 0.063). v6's tighter ROW crop is deliberately NOT used;
it regressed to 0.118 (see PROGRESS.md).

Changes vs v5:
  1. DEEP ENSEMBLE (N_MEMBERS seeds). v6 showed single-run results swing
     wildly on a 2-track training set — "best val NLL" can lock onto a
     member that fits track 14 but not the unseen 400W track. Averaging
     independent members is the standard fix for that variance, and it
     removes the seed lottery instead of hoping to win it.
  2. ENSEMBLE UNCERTAINTY, properly decomposed (Lakshminarayanan et al.):
     total_var = mean(var_m)          <- aleatoric (per-member noise)
               + var(mu_m)            <- epistemic (member disagreement)
     The epistemic term is what a single model structurally cannot report,
     and it is exactly the term that should grow when extrapolating to an
     unseen laser power.
  3. POST-HOC VARIANCE CALIBRATION fit on validation (the one part of v6
     that worked: 68% coverage 0.994 -> 0.769).
  4. BEST-VAL-MAE CHECKPOINTING IN PHASE A. v5/v6 ran all 40 MSE epochs
     and kept the final weights, but val MAE bottoms out around epoch
     ~10 and then degrades (0.134 -> 0.180) as it overfits. Phase B now
     starts from the best Phase-A weights, not the overfit ones.
  5. FLIP AUGMENTATION across the track centerline. Local width is
     symmetric under that flip, so it is a physically valid way to
     double an otherwise tiny (640-sample) training set. Applied to the
     thermal and SEM tensors together (same flip for both — they are two
     views of the same physical location).

Run: python train_v7_ensemble.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

from model_common_v3 import (
    FusionModel, PairDataset, concat_tracks, load_track, gaussian_nll, evaluate,
)

CACHE_DIR = "./processed_data/cache_v5"
SUFFIX = "v5"
OUT_DIR = "./processed_data/model_outputs_v7"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_TRACKS = [8, 10]
VAL_TRACKS = [14]
TEST_TRACK = 21

N_MEMBERS = 5
MSE_EPOCHS = 40
NLL_EPOCHS = 40
BATCH_SIZE = 32
LR = 5e-4
AUGMENT = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


class FlipAugDataset(Dataset):
    """Random flip across the track centerline (the row axis). Width is
    invariant under it, and the same flip is applied to thermal and SEM
    so the two views stay consistent."""

    def __init__(self, T, S, M, G, rng_seed=0):
        self.T, self.S, self.M, self.G = T, S, M, G
        self.rng = np.random.default_rng(rng_seed)

    def __len__(self):
        return len(self.G)

    def __getitem__(self, i):
        T, S = self.T[i], self.S[i]
        if self.rng.random() < 0.5:
            T = T[:, ::-1, :].copy()
            S = S[:, ::-1, :].copy()
        return (torch.from_numpy(T), torch.from_numpy(S),
                torch.from_numpy(self.M[i]), torch.tensor(self.G[i], dtype=torch.float32))


T_tr, S_tr, M_tr, G_tr, X_tr = concat_tracks(TRAIN_TRACKS, cache_dir=CACHE_DIR, suffix=SUFFIX)
T_va, S_va, M_va, G_va, X_va = concat_tracks(VAL_TRACKS, cache_dir=CACHE_DIR, suffix=SUFFIX)
T_te, S_te, M_te, G_te, X_te = load_track(TEST_TRACK, cache_dir=CACHE_DIR, suffix=SUFFIX)

print(f"Train {len(G_tr)} | Val {len(G_va)} | Test {len(G_te)}")

val_loader = DataLoader(PairDataset(T_va, S_va, M_va, G_va), batch_size=64)
test_loader = DataLoader(PairDataset(T_te, S_te, M_te, G_te), batch_size=64)


def train_member(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if AUGMENT:
        train_ds = FlipAugDataset(T_tr, S_tr, M_tr, G_tr, rng_seed=seed)
    else:
        train_ds = PairDataset(T_tr, S_tr, M_tr, G_tr)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = FusionModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # ---- Phase A: MSE only, keep the BEST-val-MAE weights ----
    best_mae, best_A_state, best_A_epoch = float("inf"), None, -1
    for epoch in range(1, MSE_EPOCHS + 1):
        model.train()
        for T, S, M, G in train_loader:
            T, S, M, G = T.to(device), S.to(device), M.to(device), G.to(device)
            opt.zero_grad()
            mu, _ = model(T, S, M)
            loss = torch.mean((mu - G) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        _, _, _, _, val_mae = evaluate(val_loader, model, device)
        if val_mae < best_mae:
            best_mae, best_A_epoch = val_mae, epoch
            best_A_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_A_state)
    print(f"  [seed {seed}] Phase A best val MAE {best_mae:.4f} @ epoch {best_A_epoch}")

    # ---- Phase B: NLL fine-tune, keep the BEST-val-NLL weights ----
    opt = torch.optim.Adam(model.parameters(), lr=LR * 0.5)
    best_nll, best_B_state, best_B_epoch = float("inf"), None, -1
    for epoch in range(1, NLL_EPOCHS + 1):
        model.train()
        for T, S, M, G in train_loader:
            T, S, M, G = T.to(device), S.to(device), M.to(device), G.to(device)
            opt.zero_grad()
            mu, lv = model(T, S, M)
            loss = gaussian_nll(mu, lv, G)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        _, _, _, val_nll, _ = evaluate(val_loader, model, device)
        if val_nll < best_nll:
            best_nll, best_B_epoch = val_nll, epoch
            best_B_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_B_state)
    print(f"  [seed {seed}] Phase B best val NLL {best_nll:.4f} @ epoch {best_B_epoch}")
    return model


def member_predictions(model, loader):
    mu, lv, gt, _, _ = evaluate(loader, model, device)
    return mu, np.exp(lv), gt


print(f"\n=== Training {N_MEMBERS}-member ensemble (augment={AUGMENT}) ===")
members = []
for seed in range(N_MEMBERS):
    print(f"Member {seed + 1}/{N_MEMBERS} (seed {seed})...")
    m = train_member(seed)
    torch.save(m.state_dict(), f"{OUT_DIR}/model_seed{seed}.pt")
    members.append(m)


def ensemble_predict(loader):
    """Deep-ensemble mean and variance decomposition."""
    mus, varis = [], []
    gt = None
    for m in members:
        mu_m, var_m, gt = member_predictions(m, loader)
        mus.append(mu_m); varis.append(var_m)
    mus = np.stack(mus); varis = np.stack(varis)
    mu = mus.mean(axis=0)
    aleatoric = varis.mean(axis=0)
    epistemic = mus.var(axis=0)
    return mu, aleatoric, epistemic, gt


mu_va, alea_va, epi_va, gt_va = ensemble_predict(val_loader)
std_va = np.sqrt(alea_va + epi_va)
calib_scale = float(np.sqrt(np.mean(((gt_va - mu_va) / std_va) ** 2)))
print(f"\nPost-hoc calibration scale (fit on val track {VAL_TRACKS}): {calib_scale:.3f}")

mu, alea, epi, gt = ensemble_predict(test_loader)
std_raw = np.sqrt(alea + epi)
std_cal = std_raw * calib_scale

mae = float(np.mean(np.abs(mu - gt)))
rmse = float(np.sqrt(np.mean((mu - gt) ** 2)))
nll_cal = float(np.mean(0.5 * np.log(std_cal ** 2) + 0.5 * (gt - mu) ** 2 / std_cal ** 2))
const_pred = float(np.mean(G_tr))
baseline_mae = float(np.mean(np.abs(const_pred - gt)))
oracle_mae = float(np.mean(np.abs(gt.mean() - gt)))

z_raw = (gt - mu) / std_raw
z_cal = (gt - mu) / std_cal
cov1_raw = float(np.mean(np.abs(z_raw) <= 1.0))
cov1_cal = float(np.mean(np.abs(z_cal) <= 1.0))
cov2_cal = float(np.mean(np.abs(z_cal) <= 2.0))

# Per-member MAE, to show what the ensemble bought us over one model.
member_maes = []
for m in members:
    mu_m, _, gt_m = member_predictions(m, test_loader)
    member_maes.append(float(np.mean(np.abs(mu_m - gt_m))))

print(f"\n=== v7 ensemble — Held-out Track 21 (400W, unseen power) ===")
print(f"Individual member MAEs: {['%.4f' % v for v in member_maes]}")
print(f"  mean {np.mean(member_maes):.4f} | best {np.min(member_maes):.4f} | worst {np.max(member_maes):.4f}")
print(f"ENSEMBLE MAE:  {mae:.4f} mm")
print(f"  constant-baseline MAE:     {baseline_mae:.4f} mm")
print(f"  oracle per-track-mean MAE: {oracle_mae:.4f} mm")
print(f"RMSE: {rmse:.4f} mm")
print(f"68% coverage: raw {cov1_raw:.3f} -> calibrated {cov1_cal:.3f} (target ~0.68)")
print(f"95% coverage: calibrated {cov2_cal:.3f} (target ~0.95)")
print(f"Predicted mu std: {mu.std():.4f} (ground-truth std: {gt.std():.4f})")
print(f"Mean aleatoric std {np.sqrt(alea).mean():.4f} | mean epistemic std {np.sqrt(epi).mean():.4f}")
if mae < oracle_mae:
    print(">>> BEATS the oracle per-track-mean — genuine local signal.")
else:
    print("*** Still does not beat the oracle per-track-mean.")

metrics = {
    "n_members": N_MEMBERS, "augment": AUGMENT,
    "member_test_maes": member_maes,
    "ensemble_test_mae": mae, "ensemble_test_rmse": rmse, "test_nll_calibrated": nll_cal,
    "constant_baseline_mae": baseline_mae, "oracle_track_mean_mae": oracle_mae,
    "calib_scale": calib_scale,
    "coverage_1sigma_raw": cov1_raw, "coverage_1sigma_calibrated": cov1_cal,
    "coverage_2sigma_calibrated": cov2_cal,
    "pred_mu_std": float(mu.std()), "gt_std": float(gt.std()),
    "mean_aleatoric_std": float(np.sqrt(alea).mean()),
    "mean_epistemic_std": float(np.sqrt(epi).mean()),
    "train_tracks": TRAIN_TRACKS, "val_tracks": VAL_TRACKS, "test_track": TEST_TRACK,
}
with open(f"{OUT_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved metrics to", f"{OUT_DIR}/metrics.json")

# ------------------------------------------------------------------
order = np.argsort(X_te)
xs, mus_s, stds_s, gts_s = X_te[order], mu[order], std_cal[order], gt[order]
epi_s, alea_s = np.sqrt(epi)[order], np.sqrt(alea)[order]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                gridspec_kw={"height_ratios": [3, 1]})
ax1.plot(xs, gts_s, "k.", markersize=4, label="Actual width")
ax1.plot(xs, mus_s, "-", color="tab:orange", lw=1.6, label="Ensemble mean (v7)")
ax1.fill_between(xs, mus_s - stds_s, mus_s + stds_s, color="tab:orange", alpha=0.25,
                 label="±1 std (calibrated)")
ax1.axhline(gt.mean(), color="gray", ls=":", lw=1.2, label="oracle track mean")
ax1.set_ylabel("local width (mm)")
ax1.set_title(f"v7 ensemble — Track 21 (held-out, 400W) — MAE {mae:.3f} mm "
              f"(oracle {oracle_mae:.3f}), 68% cov {cov1_cal:.2f}")
ax1.legend(fontsize=8, ncol=2)

ax2.plot(xs, alea_s, color="tab:blue", lw=1.2, label="aleatoric std (data noise)")
ax2.plot(xs, epi_s, color="tab:red", lw=1.2, label="epistemic std (model disagreement)")
ax2.set_xlabel("x (mm)"); ax2.set_ylabel("std (mm)")
ax2.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/track21_prediction.png", dpi=150)
print("Saved plot to", f"{OUT_DIR}/track21_prediction.png")
