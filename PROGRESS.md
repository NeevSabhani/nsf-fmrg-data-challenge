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
