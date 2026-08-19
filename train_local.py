"""
Standalone LOCAL training + evaluation script (v3).
Reads processed_data/cache_v3/track_*_v3.npz (built by build_pairs_local.py v3).

Fixes vs v2:
  - Input normalization (thermal / 2500 physical scale, SEM image / 255)
  - log_var bounded ARCHITECTURALLY via 6*tanh(raw/6) in forward() — the
    loss-only clamp allowed a degenerate huge-variance collapse
  - Gradient clipping (max_norm=1.0)
  - Best-validation-NLL checkpointing (not last epoch)
  - Track-based validation: train [8, 10], validate [14], test 21 — random
    coordinate splits leak spatially-correlated neighbors (paper's warning)
  - Constant-baseline comparison so MAE numbers have a reference point

Run: python train_local.py
Requires: numpy, torch, matplotlib
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------
CACHE_DIR = Path("./processed_data/cache_v3")
OUT_DIR = Path("./processed_data/model_outputs_v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_TRACKS = [8, 10]
VAL_TRACKS = [14]
TEST_TRACK = 21

EPOCHS = 60
BATCH_SIZE = 32
LR = 5e-4
SEED = 0

THERMAL_SCALE = 2500.0  # physical intensity scale (THERMAL_VMAX in starter code)
SEM_SCALE = 255.0

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def load_track(track_id):
    d = np.load(CACHE_DIR / f"track_{track_id}_v3.npz")
    T, S, M, G, X = d["T"], d["S"], d["M"], d["G"], d["X"]
    valid = ~np.isnan(G)
    T, S, M, G, X = T[valid], S[valid], M[valid], G[valid], X[valid]
    # Normalize inputs (in-place on copies)
    T = np.clip(T / THERMAL_SCALE, 0.0, 2.0).astype(np.float32)
    S = S.copy()
    S[:, 0] = S[:, 0] / SEM_SCALE  # channel 0 = image; channel 1 = mask (already 0/1)
    return T, S.astype(np.float32), M.astype(np.float32), G.astype(np.float32), X.astype(np.float32)


def concat_tracks(track_ids):
    Ts, Ss, Ms, Gs, Xs = [], [], [], [], []
    for tid in track_ids:
        T, S, M, G, X = load_track(tid)
        Ts.append(T); Ss.append(S); Ms.append(M); Gs.append(G); Xs.append(X)
    return (np.concatenate(Ts), np.concatenate(Ss), np.concatenate(Ms),
            np.concatenate(Gs), np.concatenate(Xs))


T_tr, S_tr, M_tr, G_tr, X_tr = concat_tracks(TRAIN_TRACKS)
T_va, S_va, M_va, G_va, X_va = concat_tracks(VAL_TRACKS)
T_te, S_te, M_te, G_te, X_te = load_track(TEST_TRACK)

print(f"Train (tracks {TRAIN_TRACKS}): {len(G_tr)} samples")
print(f"Val   (tracks {VAL_TRACKS}):   {len(G_va)} samples")
print(f"Test  (track {TEST_TRACK}):     {len(G_te)} samples")
print(f"Train width stats: median {np.median(G_tr):.3f}, p95 {np.percentile(G_tr,95):.3f}, "
      f"zero-frac {np.mean(G_tr==0):.2f}")


class PairDataset(Dataset):
    def __init__(self, T, S, M, G):
        self.T, self.S, self.M, self.G = T, S, M, G

    def __len__(self):
        return len(self.G)

    def __getitem__(self, i):
        return (torch.from_numpy(self.T[i]), torch.from_numpy(self.S[i]),
                torch.from_numpy(self.M[i]), torch.tensor(self.G[i], dtype=torch.float32))


train_loader = DataLoader(PairDataset(T_tr, S_tr, M_tr, G_tr), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(PairDataset(T_va, S_va, M_va, G_va), batch_size=BATCH_SIZE)
test_loader = DataLoader(PairDataset(T_te, S_te, M_te, G_te), batch_size=BATCH_SIZE)


class ConvEncoder(nn.Module):
    def __init__(self, in_ch, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


class FusionModel(nn.Module):
    LOG_VAR_BOUND = 6.0

    def __init__(self):
        super().__init__()
        self.thermal_enc = ConvEncoder(in_ch=5)
        self.sem_enc = ConvEncoder(in_ch=2)
        self.meta_enc = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 16))
        self.head = nn.Sequential(
            nn.Linear(32 + 32 + 16, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, T, S, M):
        z = torch.cat([self.thermal_enc(T), self.sem_enc(S), self.meta_enc(M)], dim=1)
        out = self.head(z)
        mu = out[:, 0]
        # Bound log-variance in the forward pass itself (smooth, gradient everywhere).
        log_var = self.LOG_VAR_BOUND * torch.tanh(out[:, 1] / self.LOG_VAR_BOUND)
        return mu, log_var


def gaussian_nll(mu, log_var, target):
    return (0.5 * log_var + 0.5 * (target - mu) ** 2 / torch.exp(log_var)).mean()


def evaluate(loader, model):
    model.eval()
    mus, lvs, gts = [], [], []
    with torch.no_grad():
        for T, S, M, G in loader:
            mu, lv = model(T.to(device), S.to(device), M.to(device))
            mus.append(mu.cpu().numpy()); lvs.append(lv.cpu().numpy()); gts.append(G.numpy())
    mu = np.concatenate(mus); lv = np.concatenate(lvs); gt = np.concatenate(gts)
    nll = float(np.mean(0.5 * lv + 0.5 * (gt - mu) ** 2 / np.exp(lv)))
    mae = float(np.mean(np.abs(mu - gt)))
    return mu, lv, gt, nll, mae


model = FusionModel().to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)

best_val_nll = float("inf")
best_state = None
best_epoch = -1
history = []

for epoch in range(1, EPOCHS + 1):
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

    _, _, _, val_nll, val_mae = evaluate(val_loader, model)
    history.append((epoch, train_loss, val_nll, val_mae))
    marker = ""
    if val_nll < best_val_nll:
        best_val_nll = val_nll
        best_epoch = epoch
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        marker = "  <- best"
    print(f"Epoch {epoch:3d}/{EPOCHS} | train NLL {train_loss:7.3f} | "
          f"val NLL {val_nll:7.3f} | val MAE {val_mae:.3f}{marker}")

print(f"\nRestoring best model from epoch {best_epoch} (val NLL {best_val_nll:.3f})")
model.load_state_dict(best_state)
torch.save(best_state, OUT_DIR / "model_best.pt")

# ------------------------------------------------------------------
# Final evaluation on held-out Track 21
# ------------------------------------------------------------------
mu, lv, gt, nll, mae = evaluate(test_loader, model)
std = np.exp(0.5 * lv)
z = (gt - mu) / std
cov1 = float(np.mean(np.abs(z) <= 1.0))
cov2 = float(np.mean(np.abs(z) <= 2.0))
rmse = float(np.sqrt(np.mean((mu - gt) ** 2)))

# Trivial baseline: predict the training-set mean everywhere.
const_pred = float(np.mean(G_tr))
baseline_mae = float(np.mean(np.abs(const_pred - gt)))

print("\n=== Held-out Track 21 evaluation (best-epoch model) ===")
print(f"MAE:   {mae:.4f} mm    (constant-baseline MAE: {baseline_mae:.4f} mm)")
print(f"RMSE:  {rmse:.4f} mm")
print(f"NLL:   {nll:.4f}")
print(f"68% coverage (target ~0.68): {cov1:.3f}")
print(f"95% coverage (target ~0.95): {cov2:.3f}")
if mae >= baseline_mae:
    print("*** WARNING: model does not beat the constant-mean baseline — "
          "it has not learned a usable signal on the held-out track.")

metrics = {
    "best_epoch": best_epoch, "val_nll_best": best_val_nll,
    "test_mae": mae, "test_rmse": rmse, "test_nll": nll,
    "coverage_1sigma": cov1, "coverage_2sigma": cov2,
    "constant_baseline_mae": baseline_mae,
    "train_tracks": TRAIN_TRACKS, "val_tracks": VAL_TRACKS, "test_track": TEST_TRACK,
}
with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved metrics to", OUT_DIR / "metrics.json")

# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
order = np.argsort(X_te)
xs, mus_s, stds_s, gts_s = X_te[order], mu[order], std[order], gt[order]

plt.figure(figsize=(11, 4))
plt.plot(xs, gts_s, "k.", markersize=4, label="Actual width")
plt.plot(xs, mus_s, "-", color="tab:blue", label="Predicted mean")
plt.fill_between(xs, mus_s - stds_s, mus_s + stds_s, color="tab:blue", alpha=0.25, label="±1 std")
plt.xlabel("x (mm)"); plt.ylabel("local width (mm)")
plt.title(f"Track 21 (held-out) — MAE {mae:.3f} mm (baseline {baseline_mae:.3f}), NLL {nll:.3f}")
plt.legend(); plt.tight_layout()
plt.savefig(OUT_DIR / "track21_prediction.png", dpi=150)
print("Saved plot to", OUT_DIR / "track21_prediction.png")

ep, tr, vl, vmae = zip(*history)
fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(ep, tr, label="train NLL")
ax1.plot(ep, vl, label="val NLL")
ax1.axvline(best_epoch, color="gray", ls="--", lw=1, label=f"best epoch ({best_epoch})")
ax1.set_xlabel("epoch"); ax1.set_ylabel("NLL"); ax1.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "training_curve.png", dpi=150)
print("Saved training curve to", OUT_DIR / "training_curve.png")