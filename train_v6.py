"""
v6 training: same two-phase recipe as v4/v5, on processed_data/cache_v6/
(tight row+column SEM crops -- see build_pairs_local_v6.py), plus
post-hoc variance calibration: fit a single scale factor on the
validation set so the predicted uncertainty band actually achieves its
target coverage, instead of relying on the network to self-calibrate.

Run: python train_v6.py
Requires: processed_data/cache_v6/*.npz (build_pairs_local_v6.py)
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from model_common_v3 import (
    FusionModel, PairDataset, concat_tracks, load_track, gaussian_nll, evaluate,
)

CACHE_DIR = "./processed_data/cache_v6"
SUFFIX = "v6"
OUT_DIR = "./processed_data/model_outputs_v6"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_TRACKS = [8, 10]
VAL_TRACKS = [14]
TEST_TRACK = 21

MSE_EPOCHS = 40
NLL_EPOCHS = 40
BATCH_SIZE = 32
LR = 5e-4
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

T_tr, S_tr, M_tr, G_tr, X_tr = concat_tracks(TRAIN_TRACKS, cache_dir=CACHE_DIR, suffix=SUFFIX)
T_va, S_va, M_va, G_va, X_va = concat_tracks(VAL_TRACKS, cache_dir=CACHE_DIR, suffix=SUFFIX)
T_te, S_te, M_te, G_te, X_te = load_track(TEST_TRACK, cache_dir=CACHE_DIR, suffix=SUFFIX)

train_loader = DataLoader(PairDataset(T_tr, S_tr, M_tr, G_tr), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(PairDataset(T_va, S_va, M_va, G_va), batch_size=BATCH_SIZE)
test_loader = DataLoader(PairDataset(T_te, S_te, M_te, G_te), batch_size=BATCH_SIZE)

model = FusionModel().to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)

print(f"\n=== Phase A: MSE-only, {MSE_EPOCHS} epochs ===")
for epoch in range(1, MSE_EPOCHS + 1):
    model.train()
    train_loss, n_seen = 0.0, 0
    for T, S, M, G in train_loader:
        T, S, M, G = T.to(device), S.to(device), M.to(device), G.to(device)
        opt.zero_grad()
        mu, _ = model(T, S, M)
        loss = torch.mean((mu - G) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        train_loss += loss.item() * len(G)
        n_seen += len(G)
    train_loss /= n_seen
    _, _, _, val_nll, val_mae = evaluate(val_loader, model, device)
    print(f"[A] Epoch {epoch:3d}/{MSE_EPOCHS} | train MSE {train_loss:.4f} | val MAE {val_mae:.4f}")

print(f"\n=== Phase B: Gaussian NLL fine-tune, {NLL_EPOCHS} epochs ===")
opt = torch.optim.Adam(model.parameters(), lr=LR * 0.5)
best_val_nll = float("inf")
best_state = None
best_epoch = -1

for epoch in range(1, NLL_EPOCHS + 1):
    model.train()
    train_loss, n_seen = 0.0, 0
    for T, S, M, G in train_loader:
        T, S, M, G = T.to(device), S.to(device), M.to(device), G.to(device)
        opt.zero_grad()
        mu, lv = model(T, S, M)
        loss = gaussian_nll(mu, lv, G)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        train_loss += loss.item() * len(G)
        n_seen += len(G)
    train_loss /= n_seen
    _, _, _, val_nll, val_mae = evaluate(val_loader, model, device)
    marker = ""
    if val_nll < best_val_nll:
        best_val_nll, best_epoch = val_nll, epoch
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        marker = "  <- best"
    print(f"[B] Epoch {epoch:3d}/{NLL_EPOCHS} | train NLL {train_loss:7.3f} | "
          f"val NLL {val_nll:7.3f} | val MAE {val_mae:.3f}{marker}")

print(f"\nRestoring best Phase-B model from epoch {best_epoch} (val NLL {best_val_nll:.3f})")
model.load_state_dict(best_state)
torch.save(best_state, f"{OUT_DIR}/model_best.pt")

# ------------------------------------------------------------------
# Post-hoc variance calibration: fit one scalar on VAL so that the
# predicted std actually matches the empirical residual spread, then
# apply that same scalar to the test predictions.
# ------------------------------------------------------------------
mu_va, lv_va, gt_va, _, _ = evaluate(val_loader, model, device)
std_va = np.exp(0.5 * lv_va)
z_va = (gt_va - mu_va) / std_va
calib_scale = float(np.sqrt(np.mean(z_va ** 2)))
print(f"\nPost-hoc calibration scale (fit on val): {calib_scale:.3f} "
      f"({'inflating' if calib_scale > 1 else 'shrinking'} predicted std)")

mu, lv, gt, nll, mae = evaluate(test_loader, model, device)
std_raw = np.exp(0.5 * lv)
std_cal = std_raw * calib_scale

for tag, std in [("raw", std_raw), ("calibrated", std_cal)]:
    z = (gt - mu) / std
    cov1 = float(np.mean(np.abs(z) <= 1.0))
    cov2 = float(np.mean(np.abs(z) <= 2.0))
    print(f"[{tag}] 68% coverage (target ~0.68): {cov1:.3f} | 95% coverage (target ~0.95): {cov2:.3f}")

rmse = float(np.sqrt(np.mean((mu - gt) ** 2)))
const_pred = float(np.mean(G_tr))
baseline_mae = float(np.mean(np.abs(const_pred - gt)))
oracle_track_mean_mae = float(np.mean(np.abs(gt.mean() - gt)))

z_cal = (gt - mu) / std_cal
cov1_cal = float(np.mean(np.abs(z_cal) <= 1.0))
cov2_cal = float(np.mean(np.abs(z_cal) <= 2.0))

print("\n=== v6 (tight SEM crop + calibrated uncertainty) Held-out Track 21 ===")
print(f"MAE:   {mae:.4f} mm    (constant-baseline MAE: {baseline_mae:.4f} mm, "
      f"oracle per-track-mean MAE: {oracle_track_mean_mae:.4f} mm)")
print(f"RMSE:  {rmse:.4f} mm")
print(f"Predicted mu std: {mu.std():.4f}  (ground-truth std: {gt.std():.4f})")
if mae >= oracle_track_mean_mae:
    print("*** STILL does not beat the oracle per-track-mean.")
else:
    print(">>> Beats the oracle per-track-mean.")

metrics = {
    "best_epoch_phaseB": best_epoch, "val_nll_best": best_val_nll,
    "test_mae": mae, "test_rmse": rmse, "test_nll": nll,
    "calib_scale": calib_scale,
    "coverage_1sigma_raw": cov1, "coverage_1sigma_calibrated": cov1_cal,
    "coverage_2sigma_raw": cov2, "coverage_2sigma_calibrated": cov2_cal,
    "constant_baseline_mae": baseline_mae,
    "oracle_track_mean_mae": oracle_track_mean_mae,
    "pred_mu_std": float(mu.std()), "gt_std": float(gt.std()),
    "train_tracks": TRAIN_TRACKS, "val_tracks": VAL_TRACKS, "test_track": TEST_TRACK,
}
with open(f"{OUT_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved metrics to", f"{OUT_DIR}/metrics.json")

order = np.argsort(X_te)
xs, mus_s, stds_s, gts_s = X_te[order], mu[order], std_cal[order], gt[order]
plt.figure(figsize=(11, 4))
plt.plot(xs, gts_s, "k.", markersize=4, label="Actual width")
plt.plot(xs, mus_s, "-", color="tab:purple", label="Predicted mean (v6)")
plt.fill_between(xs, mus_s - stds_s, mus_s + stds_s, color="tab:purple", alpha=0.2,
                  label="±1 std (calibrated)")
plt.xlabel("x (mm)"); plt.ylabel("local width (mm)")
plt.title(f"v6 tight SEM crop + calibrated uncertainty — Track 21 — MAE {mae:.3f} mm "
          f"(oracle {oracle_track_mean_mae:.3f}), 68% cov {cov1_cal:.2f}")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/track21_prediction.png", dpi=150)
print("Saved plot to", f"{OUT_DIR}/track21_prediction.png")
