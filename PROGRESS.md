# Progress Log

## Status: v3 model trained and beats baseline — see 2026-08-18 log entry

- [x] Cloned starter repo (2026-07-16)
- [x] Diagnosed width-extraction bug (many widths = 0.0) — `diagnose_widths.py`
- [x] Diagnosed model-collapse bug in first trained model (v2, log-variance
      blew up) — visible in `processed_data/model_outputs/*.png`. **That
      model/those plots are broken, don't use them.**
- [x] Rewrote `build_pairs_local.py` v3 (SEM tile index fix, robust width
      extraction) and `train_local.py` v3 (bounded log-variance, grad
      clipping, best-checkpoint, constant-baseline comparison)
- [x] Ran `build_pairs_local.py` v3 → `processed_data/cache_v3/*.npz` and
      `processed_data/qa_v3/*.png` exist
- [x] Wrote `diagnose_v3.py` to sanity-check detrend logic
- [x] Inspected `qa_v3/mask_qa_track_*.png` (red lines correctly bracket
      the track band, all 4 tracks) and `width_qa_track_*.png` (values in
      plausible 0-0.6mm range; ~15-25% exact-zero fraction confirmed
      legitimate via `diagnose_v3.py` — e.g. at track 10 x=50mm the
      detected peak (5.66) falls under the noise-adaptive 3-sigma
      threshold (7.06), so width correctly reports 0 rather than a
      hallucinated value)
- [x] **Ran `python train_local.py` (v3)** — no collapse this time.
      Held-out Track 21: MAE 0.084mm vs. constant-baseline MAE 0.173mm
      (beats it ~2x), RMSE 0.098mm, NLL -1.472. Best checkpoint restored
      from epoch 15 (val NLL kept degrading/noisy after that — checkpoint
      logic did its job). No baseline-beat warning fired.
      Caveat: uncertainty is overcovered — 68%/95% target coverage came
      in at 99.4%/100%, meaning predicted std is too wide/underconfident.
      Worth a line in the report; not a blocker.
      Outputs: `processed_data/model_outputs_v3/{model_best.pt,
      metrics.json, track21_prediction.png, training_curve.png}`
- [x] **Interpretability (permutation importance)** — `interpret_v3.py`,
      results in `processed_data/interpret_v3/`. **Important finding:**
      shuffling thermal, SEM, or metadata branches independently barely
      moves MAE/NLL (deltas ~0.0002-0.002mm, near noise level, for all
      three). Root cause confirmed directly: predicted mean has std
      0.017-0.027mm across a track vs. ground-truth std 0.077-0.155mm.
      **The model is not really learning local width variation from
      pixel content — it's outputting a near-constant value per
      track/power, and beats the constant-baseline MAE mainly because
      that near-constant is better calibrated per power level than one
      global constant fit on 2 training tracks.** This is a real
      limitation, not a bug — surfacing per working-style rule rather
      than silently proceeding. Needs a decision: report this honestly
      as a limitation (defensible, still beats baseline, uncertainty
      calibration separately imperfect too — see training log above),
      or attempt a fix before treating modeling as final.
- [x] **Fix attempt (v4): two-phase training** — `train_v4_twophase.py`,
      outputs in `processed_data/model_outputs_v4/`. Phase A = pure MSE
      on mu (no log-var, no uncertainty escape hatch) for 40 epochs,
      Phase B = fine-tune with full Gaussian NLL for 40 epochs to
      recalibrate uncertainty. **Result: root cause is NOT the NLL loss.**
      Under pure MSE, val MAE bottomed at ~0.133-0.134 by epoch ~9-10
      (matching v3's NLL result almost exactly) and predicted std stayed
      at 0.018 — barely above v3's 0.017 — before the model started
      overfitting train-set noise (train MSE kept falling, val MAE rose
      from epoch ~11 on). Final v4 test MAE 0.0742mm, a modest
      improvement over v3's 0.0840mm, but **still worse than the oracle
      per-track-mean baseline (0.0598mm)**. Conclusion: the model
      (this architecture, this ~640-sample 2-track training set, these
      thermal/SEM features) cannot reliably extract local width-variation
      signal beyond coarse per-power calibration — and even that
      calibration doesn't fully generalize to the untrained 400W power
      level. This is a genuine data/scale limitation, not a loss-function
      bug, and further loss-function tuning is unlikely to fix it — real
      progress would need either more training tracks/power levels, or
      features/labels engineered to have less inherent per-point noise.
      **Stopping modeling iteration here per working-style rule** (surface
      real findings rather than keep trying fixes indefinitely).
- [ ] Per-power-level robustness check — optional at this point; would
      mainly document how the calibration-not-signal behavior varies
      across the 4 tracks/powers, not change the conclusion above.

## v5: found and fixed a real SEM-branch bug (2026-08-18, later)

Went back into `get_sem_patch_masked_v3` (`build_pairs_local.py:308`) to
check *why* the SEM branch had ~zero permutation importance, rather than
assuming it was just data scale. Found two concrete, fixable problems:

1. **Track pixels were blanked.** `img_filled[mask == 1] = fill_val`
   overwrote every pixel inside the detected track band with the
   substrate's median value before the model ever saw it — deleting the
   one signal (track surface texture) that should carry substrate-driven
   width variation.
2. **Resolution mismatch.** Each SEM tile spans 6.41mm; samples are
   spaced 0.2mm apart (~32 samples/tile), and the whole tile was cached
   and reused unchanged (`tile_cache[tile_idx]`) for every sample inside
   it — so the SEM branch had almost no ability to distinguish neighbors.

Checked feasibility of a real fix: tiles are 1024px wide / 6.41mm =
~0.006mm/px, so a proper per-position crop is easy.

- [x] `build_pairs_local_v5.py` — new `get_sem_patch_local_v5()` crops a
      real ~1mm local window around each sample's x-position from the
      full-resolution tile, no blanking. Ran → `processed_data/cache_v5/*.npz`.
      QA (`processed_data/qa_v5/sem_crop_qa_track_*.png`) confirms crops
      now visibly differ sample-to-sample and show real track texture.
- [x] `train_v5.py` (same two-phase recipe as v4) →
      `processed_data/model_outputs_v5/`. **Real improvement:** test MAE
      0.0632mm (v3: 0.0840, v4: 0.0742), very close to the oracle
      per-track-mean (0.0598). Calibration also improved — 68% coverage
      0.956 vs. v3/v4's 0.994 (still overcovered, but much closer to the
      0.68 target).
- [x] `interpret_v5.py` (permutation importance again) — on the
      **validation** track (14), the SEM branch now shows a real,
      statistically-distinguishable contribution (delta MAE +0.0014,
      std 0.0007 — bug-fix confirmed to matter). On the **held-out
      test** track (21, untrained 400W power), SEM importance is still
      near-zero, consistent with `track21_prediction.png` (v5) showing
      visible step-jumps in the predicted mean at exact 6.41mm SEM tile
      boundaries (proof the branch is now used structurally) but still
      flat *within* each tile (no fine local-variation signal resolved).
- **Conclusion:** the SEM masking/tiling bug was real and the fix
  measurably helped (best result yet, better-calibrated uncertainty),
  but it does not fully resolve the core limitation — the model still
  doesn't beat the oracle mean on the held-out power level, and still
  isn't extracting fine (<6.41mm-scale) local width-variation signal.
  Stopping here; this is a legitimate, worthwhile improvement to report
  alongside the original limitation, not a full fix.

## v6 / v7 and the decisive negative result (2026-08-18, final)

- [x] **v6** (`build_pairs_local_v6.py`, `train_v6.py`) — tight ROW crop
      around the detected track band (band is only ~20% of tile height,
      so most of each patch was diluted substrate) + post-hoc variance
      calibration. **Split result:** calibration WORKED (68% coverage
      0.994 -> 0.769, best calibration of any version) but the row crop
      REGRESSED accuracy badly (test MAE 0.118 vs v5's 0.063 — worse
      than the original v3). Row crop abandoned; calibration kept.
- [x] **v7** (`train_v7_ensemble.py`) — best-effort model: v5 data +
      5-seed deep ensemble + proper aleatoric/epistemic decomposition +
      post-hoc calibration + best-val-MAE Phase-A checkpointing +
      centerline flip augmentation. Ensemble test MAE **0.0904**, i.e.
      WORSE than v5's 0.0632 — but the per-member MAEs were
      **0.059 / 0.091 / 0.087 / 0.143 / 0.101** (mean 0.096).
      **v5's 0.063 was a lucky seed, not a better model.** The ensemble
      mean (~0.096) is the honest expected performance of this approach.

### The decisive check: is the local variation signal or noise?

v7's ensemble `mu std` collapsed to 0.0062 (single models: ~0.010-0.026).
Averaging washed the wiggles out, which only happens if members disagree.
Checked directly on held-out track 21:

    corr(member prediction, ground truth):  -0.027, -0.031, -0.015, +0.001, -0.007
    inter-member correlation:  -0.62 .. +0.96  (mean +0.22, wildly inconsistent)

**Every member has ~ZERO correlation with the true local width variation,
and the members do not even agree with each other.** The apparent
point-to-point structure in any single model's prediction curve is a
seed-dependent artifact, not learned signal.

### Final conclusion (do not re-litigate by tuning)

The model captures the *average* width level for a track and nothing
about *local* variation — correlation with the target is statistically
zero on held-out data. Every metric that looked like success (beating
the constant baseline) is explained by getting the mean level roughly
right. Further architecture/loss/augmentation/ensembling tuning is
**pointless** — you cannot tune your way out of zero correlation.

Real paths forward, none of which are modeling changes:
  1. **More tracks/powers.** 4 tracks (2 for training) cannot support
     generalization to an unseen power level.
  2. **Label noise.** Ground-truth width is extracted per-x-position from
     Wyko height maps and jumps 0 -> 0.4mm between adjacent 0.2mm
     samples, with ~15-25% exact zeros. If much of that is extraction
     noise rather than physics, no model can fit it. Validating the
     width extraction against a manual/independent measurement is the
     highest-value next step.
  3. **Spatial registration.** Whether the thermal frame, SEM tile, and
     height-map x-coordinate for a given sample truly correspond to the
     same physical spot has never been independently verified. A
     registration offset would destroy local correlation exactly like
     this while leaving track-average behavior intact.

**Recommended reporting position:** present the v7 ensemble as the honest
result (with the seed-variance disclosure), report v5 only as
best-single-run, and lead with the negative finding + the diagnostic
evidence above. That is a legitimate, well-evidenced result for the
challenge; claiming the v5 number as a working local-variation predictor
would not be.

## v8: the LABELS were broken — root cause found and fixed (2026-08-18)

User's insight drove this: the SEM mask is painted as a full-width row
band, but the track does not span every tile (tile 01's track starts ~20%
across) and it drifts vertically within a tile. Chasing that led to the
real root cause, in the LABELS.

### Root cause 1 — ground-truth width extraction (the big one)

`extract_local_width_v3` thresholds the detrended height profile at a
NOISE floor, `max(2um, 3*sigma)`. That floor drifts with local noise
instead of following the track's edge, so the label largely measured
noise. Demonstrated in `diagnose_wyko_profile.py` (figure:
`processed_data/qa_sem_band/wyko_profile_threshold_track_10.png`) — the
profiles show a clean ~0.45mm dome at every x, but:

    x (mm):        35     45     55     65     75     85
    3-sigma  :  0.016  0.394  0.394  0.478  0.215  0.000   <- the old labels
    half-max :  0.402  0.442  0.482  0.458  0.450  0.454   <- stable, correct

At x=85 the noise cut (7.5um) sat ABOVE the track's own peak (7.0um) and
reported width 0.000mm for a plainly-present track. **The ~25% "exact
zeros" and the wild point-to-point swings were the extraction failing,
not physics. We were training models to predict noise** — which is
exactly why every architecture/loss/ensemble plateaued at zero
correlation. Independent confirmation (`diagnose_sem_band.py`): SEM-
measured band width had ~zero correlation (+0.045) with the v3 labels,
and disagreed 3x in scale (~0.89mm SEM vs ~0.28mm label).

### Root cause 2 — SEM masking (the user's original point, confirmed)

`per_column_band()` in `build_pairs_local_v8.py` detects the band PER
COLUMN. QA figure `qa_sem_band/per_column_band_track_10.png` confirms:
tile 01's track genuinely starts ~20% across (17% of columns have no
band at all), and the band slants within tiles. The old full-width,
tile-constant mask was wrong on both counts — and this explains v6's
regression (tightening a row crop around a tile-AVERAGE band position
cuts off the real track wherever it drifts).

- [x] **`build_pairs_local_v8.py`** — half-max width extraction (NaN, never
      a silent 0.0, when no track is detectable) + per-column SEM band
      masking + row crop centered on the LOCAL band. Ran → `cache_v8/`.
      **Labels are now physically coherent and ordered by laser power:**
      median width 0.569mm (200W) / 0.470 (300W) / 0.390 (350W) / 0.195
      (400W), std 0.037-0.096, zero spurious zeros.
- [x] **`train_v8.py`** (v7 ensemble recipe on fixed labels) — val MAE
      improved ~3x (0.134 → 0.041): with clean labels the model fits the
      validation track far better in absolute terms. **But held-out
      corr(pred, truth) on track 21 is still +0.002.**

### Registration checked and cleared

A lag scan showed a peak at -6.4mm (~one tile width), so all four
tile-order/orientation mappings were tested against the labels:

    tiles as-is,    within-tile as-is    : +0.143   <- best (current code)
    tiles as-is,    within-tile flipped  : +0.143
    tiles REVERSED, within-tile as-is    : -0.001
    tiles REVERSED, within-tile flipped  : -0.086

Current mapping is correct; the -6.4mm lag peak was a multiple-comparisons
artifact across 81 lags. **Registration is not the problem.**

### Where this actually leaves the project

Both real bugs are now fixed and the labels are genuinely good. What
remains is a data property, not a code defect: two INDEPENDENT good
measurements of local width (Wyko half-max, SEM band) correlate only
+0.03..+0.20 with each other. The local-variation signal is weak relative
to measurement noise in both instruments, and with 2 training tracks it
does not generalize to an unseen laser power. No modeling change fixes
that. Options that would: more tracks/powers, a lower-noise width
metrology, or reframing the target (e.g. predict track-average width or a
smoothed/aggregated variation statistic rather than per-0.2mm width).

## v9: reframed target — the first genuinely working model

`v9_segment_model.py`. Aggregates the fixed v8 per-point data into 4mm
segments (64 segments across 4 tracks) and predicts segment-level
quantities with ridge regression on interpretable scalar features
(sklearn is not installed; ridge + CV implemented in numpy). Validation
is leave-one-TRACK-out, so every fold trains on 3 laser powers and tests
on the 4th — the generalization test and the "robustness across laser
powers" evidence are the same experiment.

### Result 1 — segment mean width: WORKS

    features            MAE      baseline   corr     R2
    thermal only      0.0495     0.1469    +0.902   +0.800   <- best
    all features      0.0584     0.1469    +0.875   +0.729
    SEM only          0.1145     0.1469    +0.103   -0.228
    power alone       0.0778     0.1469    +0.814   +0.607

Per-power folds (all-feature model): 200W, 300W, 400W all beat their
baseline; 350W does not (0.0448 vs 0.0281 — that track's mean sits close
to the mean of the other three, so the constant baseline is unusually
strong there). Reported, not hidden.

### Result 2 — interpretability: variation is PROCESS-driven

Thermal-only (R2 +0.800) beats SEM-only (R2 -0.228) decisively, and
beats all-features, i.e. the SEM branch adds noise rather than signal for
this target. This is a clean, quantitative answer to the "separate
process-driven vs substrate-driven contributions" criterion: **track
width is governed by the thermal history, not by substrate texture.**
Thermal also beats power alone (R2 0.800 vs 0.607), so the thermal data
carries real information beyond just identifying the power setpoint —
it is not merely a power proxy.

### Result 3 — uncertainty: well calibrated

68% coverage 0.625, 95% coverage 0.922 (targets 0.68 / 0.95) — the best
calibration achieved in the project. Required using OUT-OF-FOLD residuals
for the predictive sigma; in-fold residuals gave 0.219/0.516, badly
overconfident, because they do not reflect the cost of extrapolating to
an unseen power.

### Result 4 — the honest limits (both verified, must be reported)

  - **Within-track variation is still not predicted.** Subtracting each
    track's mean from prediction and truth gives within-track corr
    **+0.079** (per track: -0.188 / +0.167 / +0.143 / +0.002). The +0.902
    headline is a BETWEEN-track (power-level) effect. Visible in
    `parity_plots.png`: clean ordering between colour groups, no diagonal
    structure within any group.
  - **Segment width VARIABILITY is not predicted at all** (corr -0.192,
    R2 -0.566, does not beat baseline) by any feature set.

### Bottom line for the challenge

Defensible claim: *"Thermal signature predicts track width across laser
powers (R2 0.80 leave-one-power-out, calibrated uncertainty), width is
process- not substrate-driven, and we show with independent metrology
that pointwise width variation is below the noise floor of the available
measurements."* That is a real positive result plus a well-evidenced
negative one. Claiming pointwise local-variation prediction would not be
supportable.
- [ ] Final report (PDF, 3 pages max, 10pt Arial, 1" margins, required
      sections incl. GenAI disclosure)
- [ ] Clean executable notebook
- [ ] Slide deck
- [ ] Package submission.zip

## Log

**2026-08-18** — Session resumed via Claude (Cowork) after user lost track
of the project for ~5 weeks. Reconstructed the above history from git log
+ file mtimes (no local commits existed; only the original clone was
tracked). Wrote this file and `CLAUDE.md` to make the state legible again.
Did not run any code — next session should start with `python train_local.py`.
