"""
Shared model/data code for v3, factored out of train_local.py so
interpret_v3.py and robustness_v3.py don't duplicate it. train_local.py
itself is left untouched since its output is already verified.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

CACHE_DIR = Path("./processed_data/cache_v3")
THERMAL_SCALE = 2500.0
SEM_SCALE = 255.0
TRACK_LASER_POWER_W = {8: 200, 10: 300, 14: 350, 21: 400}


def load_track(track_id, cache_dir=None, suffix="v3"):
    cache_dir = cache_dir or CACHE_DIR
    d = np.load(Path(cache_dir) / f"track_{track_id}_{suffix}.npz")
    T, S, M, G, X = d["T"], d["S"], d["M"], d["G"], d["X"]
    valid = ~np.isnan(G)
    T, S, M, G, X = T[valid], S[valid], M[valid], G[valid], X[valid]
    T = np.clip(T / THERMAL_SCALE, 0.0, 2.0).astype(np.float32)
    S = S.copy()
    S[:, 0] = S[:, 0] / SEM_SCALE
    return T, S.astype(np.float32), M.astype(np.float32), G.astype(np.float32), X.astype(np.float32)


def concat_tracks(track_ids, cache_dir=None, suffix="v3"):
    Ts, Ss, Ms, Gs, Xs = [], [], [], [], []
    for tid in track_ids:
        T, S, M, G, X = load_track(tid, cache_dir=cache_dir, suffix=suffix)
        Ts.append(T); Ss.append(S); Ms.append(M); Gs.append(G); Xs.append(X)
    return (np.concatenate(Ts), np.concatenate(Ss), np.concatenate(Ms),
            np.concatenate(Gs), np.concatenate(Xs))


class PairDataset(Dataset):
    def __init__(self, T, S, M, G):
        self.T, self.S, self.M, self.G = T, S, M, G

    def __len__(self):
        return len(self.G)

    def __getitem__(self, i):
        return (torch.from_numpy(self.T[i]), torch.from_numpy(self.S[i]),
                torch.from_numpy(self.M[i]), torch.tensor(self.G[i], dtype=torch.float32))


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
        log_var = self.LOG_VAR_BOUND * torch.tanh(out[:, 1] / self.LOG_VAR_BOUND)
        return mu, log_var


def gaussian_nll(mu, log_var, target):
    return (0.5 * log_var + 0.5 * (target - mu) ** 2 / torch.exp(log_var)).mean()


def evaluate(loader, model, device):
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


def train_model(train_tracks, val_tracks, epochs=60, batch_size=32, lr=5e-4, seed=0, device=None,
                 cache_dir=None, suffix="v3"):
    """Same training loop as train_local.py, factored out for reuse across
    multiple leave-one-out splits (robustness_v3.py)."""
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T_tr, S_tr, M_tr, G_tr, X_tr = concat_tracks(train_tracks, cache_dir=cache_dir, suffix=suffix)
    T_va, S_va, M_va, G_va, X_va = concat_tracks(val_tracks, cache_dir=cache_dir, suffix=suffix)

    train_loader = DataLoader(PairDataset(T_tr, S_tr, M_tr, G_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(PairDataset(T_va, S_va, M_va, G_va), batch_size=batch_size)

    model = FusionModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_nll = float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(1, epochs + 1):
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
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_epoch, best_val_nll, float(np.mean(G_tr))
