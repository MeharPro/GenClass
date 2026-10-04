# Z-68m mid-run dev check (step 1950 of ~2100, 2026-10-03 20:40 EDT)
Checkpoint z-mid1950, uncalibrated, clean dev set v1.1, main variant only.
Shared dev sets, accuracy: v2-68m -> z-mid1950
 20NG .302->.241 | TweetEval-sent .675->.658 | SNIPS .746->.715 | MRPC .718->.647 | SciTail .593->.670 | CB(n=56) .393->.500 | TweetEval-off .799->.784
Mean of the 7 shared sets: v2 0.604, z-mid 0.602 (parity). tasksource_heldout .497->.473 (set changed: strategy-qa removed).
z-mid1950 own indices (not comparable to v2's 36.8: different dev composition): skill 40.2, decision-score 5.6.
Reading: clean Z (no benchmark train data, ~27% fewer rows, ~1.0B vs 1.34B tokens) ~ matches v2 on average; training-loss drops at pass boundaries (steps ~860, ~1680) did not show up as dev gains.

## Final Z-68m (step 2010/2010, 2026-10-03) and calibration check
- Final checkpoint scores identically to step 1950 on the clean dev set (skill 40.2, decision-score 5.6): the last ~60 steps at decayed LR changed nothing; the run had converged.
- Calibration (engine/encoder/calibrate.py fit on the 10 clean dev sets) does NOT transfer. Split-half check (fit on half the datasets, score the other half):
  mean decision-score on held-out datasets 0.023 raw -> 0.029 calibrated (noise); per-dataset swings large (ATIS 22-way -0.32, CB +0.16, SciTail +0.12, offensive +0.08);
  accuracy unchanged. In-sample fit looked great (choice ECE 0.099 -> 0.022) = overfitting; noul Platt hit the clamp (a=2.0, b=-5.0) = unstable.
  Cause (likely): tau_k is fit per option-count bucket from few datasets and extrapolates badly (the B-half had no K>10, and ATIS has K=22).
  Decision: ship identity calibration for Z-68m; do not claim a calibration gain. Better route: fit on the large diverse dev_mix slice
  (data/bm/z/shards/dev_mix, ~1% of the training mix) and validate on the clean dev sets as out-of-sample.
- Model copied to Mac: models/meharsjev-68m-z (287 MB).
