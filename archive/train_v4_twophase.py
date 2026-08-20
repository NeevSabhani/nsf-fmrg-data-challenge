"""
v4 fix attempt: two-phase training to address the v3 finding that the
Gaussian-NLL model collapses toward a near-constant mu (predicted std
0.017-0.027mm vs. ground-truth std 0.077-0.155mm) instead of using local
thermal/SEM content — likely because NLL lets the model buy down loss by
inflating uncertainty instead of sharpening the mean.

Phase A (mse_epochs): pure MSE on mu only. No variance escape hatch —
forces the shared trunk + mu head to fit as tightly as the data allows.
Phase B (nll_epochs): switch to full Gaussian NLL (mu path continues
fine-tuning, log_var head trained from scratch) to calibrate uncertainty
around the now-tightly-fit mean.

Everything else (architecture, split, bounded log-var, grad clipping) is
identical to train_local.py v3, so results are directly comparable.

Run: python train_v4_twophase.py
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from model_common_v3 import (
    FusionModel, PairDataset, concat_tracks, load_track, gaussian_nll, evaluate,
)

OUT_DIR = "./processed_data/model_outputs_v4"
import os
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

T_tr, S_tr, M_tr, G_tr, X_tr = concat_tracks(TRAIN_TRACKS)
T_va, S_va, M_va, G_va, X_va = concat_tracks(VAL_TRACKS)
T_te, S_te, M_te, G_te, X_te = load_track(TEST_TRACK)

train_loader = DataLoader(PairDataset(T_tr, S_tr, M_tr, G_tr), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(PairDataset(T_va, S_va, M_va, G_va), batch_size=BATCH_SIZE)
test_loader = DataLoader(PairDataset(T_te, S_te, M_te, G_te), batch_size=BATCH_SIZE)

model = FusionModel().to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)

history = []

# ------------------------------------------------------------------
# Phase A: pure MSE on mu, no uncertainty escape hatch
# ------------------------------------------------------------------
print(f"\n=== Phase A: MSE-only, {MSE_EPOCHS} epochs ===")
for epoch in range(1, MSE_EPOCHS + 1):
    model.train()
    train_loss = 0.0
    n_seen = 0
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
    history.append(("A", epoch, train_loss, val_nll, val_mae))
    print(f"[A] Epoch {epoch:3d}/{MSE_EPOCHS} | train MSE {train_loss:.4f} | val MAE {val_mae:.4f}")

mu_mid, _, gt_mid, _, _ = evaluate(val_loader, model, device)
print(f"\nAfter Phase A: val mu std={mu_mid.std():.4f} (target std={gt_mid.std():.4f})")

# ------------------------------------------------------------------
# Phase B: full Gaussian NLL fine-tune to calibrate uncertainty
# ------------------------------------------------------------------
print(f"\n=== Phase B: Gaussian NLL fine-tune, {NLL_EPOCHS} epochs ===")
opt = torch.optim.Adam(model.parameters(), lr=LR * 0.5)  # gentler LR for fine-tune
best_val_nll = float("inf")
best_state = None
best_epoch = -1

for epoch in range(1, NLL_EPOCHS + 1):
    model.train()
    train_loss = 0.0
    n_seen = 0
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
    history.append(("B", epoch, train_loss, val_nll, val_mae))
    marker = ""
    if val_nll < best_val_nll:
        best_val_nll = val_nll
        best_epoch = epoch
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        marker = "  <- best"
    print(f"[B] Epoch {epoch:3d}/{NLL_EPOCHS} | train NLL {train_loss:7.3f} | "
          f"val NLL {val_nll:7.3f} | val MAE {val_mae:.3f}{marker}")

print(f"\nRestoring best Phase-B model from epoch {best_epoch} (val NLL {best_val_nll:.3f})")
model.load_state_dict(best_state)
torch.save(best_state, f"{OUT_DIR}/model_best.pt")

# ------------------------------------------------------------------
# Final evaluation on held-out Track 21
# ------------------------------------------------------------------
mu, lv, gt, nll, mae = evaluate(test_loader, model, device)
std = np.exp(0.5 * lv)
z = (gt - mu) / std
cov1 = float(np.mean(np.abs(z) <= 1.0))
cov2 = float(np.mean(np.abs(z) <= 2.0))
rmse = float(np.sqrt(np.mean((mu - gt) ** 2)))

const_pred = float(np.mean(G_tr))
baseline_mae = float(np.mean(np.abs(const_pred - gt)))
oracle_track_mean_mae = float(np.mean(np.abs(gt.mean() - gt)))

print("\n=== v4 (two-phase) Held-out Track 21 evaluation ===")
print(f"MAE:   {mae:.4f} mm    (constant-baseline MAE: {baseline_mae:.4f} mm, "
      f"oracle per-track-mean MAE: {oracle_track_mean_mae:.4f} mm)")
print(f"RMSE:  {rmse:.4f} mm")
print(f"NLL:   {nll:.4f}")
print(f"68% coverage (target ~0.68): {cov1:.3f}")
print(f"95% coverage (target ~0.95): {cov2:.3f}")
print(f"Predicted mu std: {mu.std():.4f}  (ground-truth std: {gt.std():.4f})")
if mae >= baseline_mae:
    print("*** WARNING: model does not beat the constant-mean baseline.")
if mae >= oracle_track_mean_mae:
    print("*** NOTE: model still does not beat the oracle per-track-mean — "
          "still not extracting usable local signal from images.")
else:
    print(">>> Model NOW beats the oracle per-track-mean baseline — "
          "genuine local-signal improvement over v3.")

metrics = {
    "best_epoch_phaseB": best_epoch, "val_nll_best": best_val_nll,
    "test_mae": mae, "test_rmse": rmse, "test_nll": nll,
    "coverage_1sigma": cov1, "coverage_2sigma": cov2,
    "constant_baseline_mae": baseline_mae,
    "oracle_track_mean_mae": oracle_track_mean_mae,
    "pred_mu_std": float(mu.std()), "gt_std": float(gt.std()),
    "train_tracks": TRAIN_TRACKS, "val_tracks": VAL_TRACKS, "test_track": TEST_TRACK,
}
with open(f"{OUT_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved metrics to", f"{OUT_DIR}/metrics.json")

order = np.argsort(X_te)
xs, mus_s, stds_s, gts_s = X_te[order], mu[order], std[order], gt[order]
plt.figure(figsize=(11, 4))
plt.plot(xs, gts_s, "k.", markersize=4, label="Actual width")
plt.plot(xs, mus_s, "-", color="tab:red", label="Predicted mean (v4)")
plt.fill_between(xs, mus_s - stds_s, mus_s + stds_s, color="tab:red", alpha=0.2, label="±1 std")
plt.xlabel("x (mm)"); plt.ylabel("local width (mm)")
plt.title(f"v4 two-phase — Track 21 (held-out) — MAE {mae:.3f} mm "
          f"(oracle {oracle_track_mean_mae:.3f}), NLL {nll:.3f}")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/track21_prediction.png", dpi=150)
print("Saved plot to", f"{OUT_DIR}/track21_prediction.png")

phases, eps, tr, vl, vmae = zip(*history)
fig, ax1 = plt.subplots(figsize=(9, 4))
b_start = phases.index("B")
ax1.plot(range(len(vl)), vl, label="val NLL (Phase A NLL is uncalibrated, ignore scale)")
ax1.axvline(b_start, color="gray", ls="--", lw=1, label="Phase B starts")
ax1.axvline(b_start + best_epoch - 1, color="red", ls="--", lw=1, label=f"best Phase-B epoch ({best_epoch})")
ax1.set_xlabel("step (A then B)"); ax1.set_ylabel("val NLL"); ax1.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/training_curve.png", dpi=150)
print("Saved training curve to", f"{OUT_DIR}/training_curve.png")
