# Archive — superseded versions

These are earlier versions, kept because they are the evidence behind the
"ruled out" claims in the report rather than because they are useful models.
None of them should be used for new work.

| File | What it was | Why it is superseded |
|---|---|---|
| `train_local.py` | v3 baseline: fusion CNN, Gaussian NLL | Trained on the broken labels. Beat a constant baseline only by predicting a near-constant per track. |
| `train_v4_twophase.py` | Pure MSE phase, then NLL fine-tune | **Ruled out the loss function** as the cause of the zero correlation — MSE reproduced the same plateau. |
| `build_pairs_local_v5.py` | SEM patch fix: real ~1mm column crop, no blanking | Superseded by v8's per-column band detection. |
| `build_pairs_local_v6.py` | v5 plus a tighter row crop | **Regressed** (MAE 0.118 vs v5's 0.063): cropping around a *tile-average* band position truncates the track where it drifts. |
| `train_v7_ensemble.py` | 5-seed deep ensemble, flip augmentation, post-hoc calibration | **Ruled out added capacity** — scored worse than a single lucky seed, members correlating −0.62 to +0.96 with each other. |
| `train_v8.py` | The v7 recipe rerun on the corrected v8 labels | Held-out pointwise correlation was still +0.002. This is the result that motivated reframing to segments. |
| `model_common_v3.py` | Shared model/dataset code for the `train_*` scripts | Only used by the above. |

Every one of these was trained on **pointwise** width, which the report shows
lies below the noise floor of the available metrology. The working model is
`v9_segment_model.py` in the repository root.

They import from the repository root, so run them from there:

```bash
PYTHONPATH=. python archive/train_v7_ensemble.py
```

Full evidence trail for each: [`../PROGRESS.md`](../PROGRESS.md).
