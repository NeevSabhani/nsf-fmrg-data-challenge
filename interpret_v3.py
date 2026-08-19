"""
Interpretability for the v3 fusion model: permutation importance per input
branch (thermal, SEM, metadata), evaluated on the held-out Track 21 test
set and the Track 14 validation set.

For each branch, that branch's inputs are shuffled across the batch
(breaking its association with the target while leaving the other two
branches intact) and the resulting increase in MAE/NLL is measured,
repeated over multiple random permutations for stability. A branch whose
shuffle causes a large error increase is one the model actually relies on;
this is what separates process-driven (thermal) from substrate-driven
(SEM) contribution to the width prediction, per the judging criteria.

Run: python interpret_v3.py
Requires: processed_data/model_outputs_v3/model_best.pt (train_local.py)
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from model_common_v3 import FusionModel, PairDataset, load_track, evaluate

OUT_DIR_MODEL = "./processed_data/model_outputs_v3"
OUT_DIR = "./processed_data/interpret_v3"
N_REPEATS = 20
SEED = 0

import os
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(SEED)

model = FusionModel().to(device)
model.load_state_dict(torch.load(f"{OUT_DIR_MODEL}/model_best.pt", map_location=device))
model.eval()


def permutation_importance(T, S, M, G, branch, n_repeats=N_REPEATS):
    """branch in {'T', 'S', 'M', None}. None = baseline (no shuffle)."""
    maes, nlls = [], []
    n = len(G)
    for r in range(n_repeats if branch is not None else 1):
        Tp, Sp, Mp = T, S, M
        if branch == "T":
            Tp = T[rng.permutation(n)]
        elif branch == "S":
            Sp = S[rng.permutation(n)]
        elif branch == "M":
            Mp = M[rng.permutation(n)]
        loader = DataLoader(PairDataset(Tp, Sp, Mp, G), batch_size=64)
        _, _, _, nll, mae = evaluate(loader, model, device)
        maes.append(mae); nlls.append(nll)
    return float(np.mean(maes)), float(np.std(maes)), float(np.mean(nlls)), float(np.std(nlls))


results = {}
for split_name, track_id in [("val_track14", 14), ("test_track21", 21)]:
    T, S, M, G, X = load_track(track_id)
    base_mae, _, base_nll, _ = permutation_importance(T, S, M, G, None)

    split_results = {"baseline_mae": base_mae, "baseline_nll": base_nll, "branches": {}}
    for branch, label in [("T", "thermal (process)"), ("S", "sem (substrate)"), ("M", "metadata (x-pos, power)")]:
        mae_mean, mae_std, nll_mean, nll_std = permutation_importance(T, S, M, G, branch)
        split_results["branches"][branch] = {
            "label": label,
            "shuffled_mae_mean": mae_mean, "shuffled_mae_std": mae_std,
            "delta_mae": mae_mean - base_mae,
            "shuffled_nll_mean": nll_mean, "shuffled_nll_std": nll_std,
            "delta_nll": nll_mean - base_nll,
        }
        print(f"[{split_name}] shuffle {label:28s} | "
              f"delta MAE {mae_mean - base_mae:+.4f} (+/-{mae_std:.4f}) | "
              f"delta NLL {nll_mean - base_nll:+.4f} (+/-{nll_std:.4f})")
    results[split_name] = split_results
    print(f"[{split_name}] baseline MAE {base_mae:.4f}, baseline NLL {base_nll:.4f}\n")

with open(f"{OUT_DIR}/permutation_importance.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved", f"{OUT_DIR}/permutation_importance.json")

# ------------------------------------------------------------------
# Plot: delta MAE per branch, per split
# ------------------------------------------------------------------
branches = ["T", "S", "M"]
labels = [results["test_track21"]["branches"][b]["label"] for b in branches]
fig, ax = plt.subplots(figsize=(7, 4))
width = 0.35
x = np.arange(len(branches))
for i, split_name in enumerate(["val_track14", "test_track21"]):
    deltas = [results[split_name]["branches"][b]["delta_mae"] for b in branches]
    errs = [results[split_name]["branches"][b]["shuffled_mae_std"] for b in branches]
    ax.bar(x + (i - 0.5) * width, deltas, width, yerr=errs, capsize=3, label=split_name)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("increase in MAE when branch is shuffled (mm)")
ax.set_title("Permutation importance by input branch")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/permutation_importance.png", dpi=150)
print("Saved", f"{OUT_DIR}/permutation_importance.png")
