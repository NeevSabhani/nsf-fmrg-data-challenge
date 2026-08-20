"""
v8 training — same ensemble recipe as train_v7_ensemble.py, on the fixed
cache_v8 labels (half-maximum width extraction + per-column SEM masking,
see build_pairs_local_v8.py).

The decisive metric here is NOT MAE (which the old broken labels could
already "win" by predicting a constant), but corr(prediction, truth) on
the held-out track. Every previous version scored ~0.00 there. Two extra
reference points are printed so the number can be judged honestly:
  - corr of the ensemble mean
  - the per-track constant and oracle-mean MAEs, as before

Run: python train_v8.py
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

CACHE_DIR = "./processed_data/cache_v8"
SUFFIX = "v8"
OUT_DIR = "./processed_data/model_outputs_v8"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_TRACKS = [8, 10]
VAL_TRACKS = [14]
TEST_TRACK = 21

N_MEMBERS = 5
MSE_EPOCHS = 40
NLL_EPOCHS = 40
BATCH_SIZE = 32
LR = 5e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


class FlipAugDataset(Dataset):
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
print(f"Label stats — train median {np.median(G_tr):.3f} std {G_tr.std():.3f} | "
      f"test median {np.median(G_te):.3f} std {G_te.std():.3f}")

val_loader = DataLoader(PairDataset(T_va, S_va, M_va, G_va), batch_size=64)
test_loader = DataLoader(PairDataset(T_te, S_te, M_te, G_te), batch_size=64)


def train_member(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader = DataLoader(FlipAugDataset(T_tr, S_tr, M_tr, G_tr, rng_seed=seed),
                              batch_size=BATCH_SIZE, shuffle=True)
    model = FusionModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_mae, best_state = float("inf"), None
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
            best_mae = val_mae
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    opt = torch.optim.Adam(model.parameters(), lr=LR * 0.5)
    best_nll, best_state = float("inf"), None
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
            best_nll = val_nll
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    print(f"  [seed {seed}] Phase A best val MAE {best_mae:.4f} | Phase B best val NLL {best_nll:.4f}")
    return model


print(f"\n=== Training {N_MEMBERS}-member ensemble on FIXED v8 labels ===")
members = []
for seed in range(N_MEMBERS):
    print(f"Member {seed + 1}/{N_MEMBERS}...")
    m = train_member(seed)
    torch.save(m.state_dict(), f"{OUT_DIR}/model_seed{seed}.pt")
    members.append(m)


def ensemble_predict(loader):
    mus, varis, gt = [], [], None
    for m in members:
        mu_m, lv_m, gt, _, _ = evaluate(loader, m, device)
        mus.append(mu_m); varis.append(np.exp(lv_m))
    mus = np.stack(mus); varis = np.stack(varis)
    return mus.mean(0), varis.mean(0), mus.var(0), gt, mus


mu_va, alea_va, epi_va, gt_va, _ = ensemble_predict(val_loader)
std_va = np.sqrt(alea_va + epi_va)
calib_scale = float(np.sqrt(np.mean(((gt_va - mu_va) / std_va) ** 2)))
corr_va = float(np.corrcoef(mu_va, gt_va)[0, 1])
print(f"\nVal  corr(pred, truth) = {corr_va:+.3f} | calibration scale {calib_scale:.3f}")

mu, alea, epi, gt, mus_all = ensemble_predict(test_loader)
std_cal = np.sqrt(alea + epi) * calib_scale

mae = float(np.mean(np.abs(mu - gt)))
rmse = float(np.sqrt(np.mean((mu - gt) ** 2)))
corr = float(np.corrcoef(mu, gt)[0, 1])
member_corrs = [float(np.corrcoef(m_, gt)[0, 1]) for m_ in mus_all]
member_maes = [float(np.mean(np.abs(m_ - gt))) for m_ in mus_all]
baseline_mae = float(np.mean(np.abs(np.mean(G_tr) - gt)))
oracle_mae = float(np.mean(np.abs(gt.mean() - gt)))
z = (gt - mu) / std_cal
cov1 = float(np.mean(np.abs(z) <= 1.0))
cov2 = float(np.mean(np.abs(z) <= 2.0))

print(f"\n=== v8 — Held-out Track 21 (400W, unseen power) ===")
print(f"Per-member corr(pred, truth): {['%+.3f' % c for c in member_corrs]}")
print(f"Per-member MAE:               {['%.4f' % v for v in member_maes]}")
print(f"ENSEMBLE corr(pred, truth):   {corr:+.3f}   <-- the metric that matters")
print(f"ENSEMBLE MAE:  {mae:.4f} mm   (constant baseline {baseline_mae:.4f}, oracle mean {oracle_mae:.4f})")
print(f"RMSE: {rmse:.4f} mm | pred std {mu.std():.4f} (truth std {gt.std():.4f})")
print(f"68% coverage {cov1:.3f} | 95% coverage {cov2:.3f}")
if mae < oracle_mae:
    print(">>> Beats the oracle per-track-mean.")
else:
    print("*** Does not beat the oracle per-track-mean.")

metrics = {
    "ensemble_corr": corr, "val_corr": corr_va,
    "member_corrs": member_corrs, "member_maes": member_maes,
    "ensemble_test_mae": mae, "ensemble_test_rmse": rmse,
    "constant_baseline_mae": baseline_mae, "oracle_track_mean_mae": oracle_mae,
    "coverage_1sigma": cov1, "coverage_2sigma": cov2,
    "calib_scale": calib_scale,
    "pred_mu_std": float(mu.std()), "gt_std": float(gt.std()),
}
with open(f"{OUT_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

order = np.argsort(X_te)
xs, mus_s, stds_s, gts_s = X_te[order], mu[order], std_cal[order], gt[order]
plt.figure(figsize=(11, 4))
plt.plot(xs, gts_s, "k.-", markersize=4, lw=0.5, alpha=0.7, label="Actual width (v8 labels)")
plt.plot(xs, mus_s, "-", color="tab:green", lw=1.6, label="Ensemble mean (v8)")
plt.fill_between(xs, mus_s - stds_s, mus_s + stds_s, color="tab:green", alpha=0.2, label="±1 std")
plt.xlabel("x (mm)"); plt.ylabel("local width (mm)")
plt.title(f"v8 (fixed labels) — Track 21 held-out — corr {corr:+.3f}, MAE {mae:.3f} mm "
          f"(oracle {oracle_mae:.3f})")
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/track21_prediction.png", dpi=150)
print("Saved", f"{OUT_DIR}/track21_prediction.png")
