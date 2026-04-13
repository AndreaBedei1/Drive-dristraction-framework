"""
tcn_dann_ext4.py

Domain-Adversarial Dual-Branch TCN — ext4 additions:
  1. Auxiliary physiological branch loss (AUX_PHYS_WEIGHT):
     Forces the physiology TCN to maintain discriminative representations
     independently of kinematics. Without this, strong kinematic gradients
     cause the gate to collapse toward kinematics and the phys branch
     converges to a near-trivial representation.
  2. Per-session HR normalization:
     HR is z-scored per (driver, route) session — the same treatment as
     vehicle signals — making it relative to each driver's intra-session
     baseline rather than absolute. Arousal is already baseline-relative;
     this brings HR into the same regime.

Publication fixes (applied to ext4):
  3. Consistent train/test normalisation:
     Training windows are now re-normalised using the same route-level
     training-fold statistics as test windows (instead of per-driver
     self-normalisation), eliminating the train/test distribution mismatch.
  4. Correct pos_weight for XGB:
     pw_uniform is now computed from the actual fine-stride training labels
     (y_tr_orig) rather than the uniform-stride test labels, matching the
     training data distribution that XGB actually sees.
  5. Spectral feature standardisation:
     A per-feature StandardScaler (fit on real training windows) is applied
     to the 18-d spectral vector so it enters the head on the same scale as
     the 96-d scaled fusion vector, avoiding implicit down-weighting.
  6. Per-fold PyTorch RNG seeding:
     torch.manual_seed is reset at the start of each LOPO fold so that
     DataLoader sampling and augmentation are independently reproducible
     per fold regardless of execution order.
  7. torch.set_num_threads(1):
     Ensures bit-exact CPU reproducibility across machines with different
     thread counts.
  8. Val windows built from per-fold df slice (not X_raw_te):
     Val windows for early stopping are now built by running build_windows on
     the val-driver rows of df, keeping val and test window pools completely
     disjoint.  The previous approach took val from X_raw_te, exposing each
     driver's test windows as val data in ~20% of other folds.
  9. CosineAnnealingLR T_max = EPOCHS:
     All three training functions now set T_max=EPOCHS so the cosine LR
     schedule spans the full training budget.  The previous T_max of
     PATIENCE*4 / DANN_PATIENCE*4 caused the scheduler to reach eta_min
     after 40-60 epochs, stalling training for up to 60 epochs at minimum LR.
 10. SMOTE spectral features from actual synthetic windows:
     Spectral features for SMOTE-synthesised windows are now computed from
     the interpolated window itself rather than copied from the anchor.  The
     previous approach introduced a temporal/spectral mismatch for ~50% of
     synthetic samples (those with interpolation weight λ ≠ 0).
 11. Per-model torch RNG reseeding within each fold:
     torch.manual_seed is reset independently before DB-TCN, DANN-DB-TCN,
     and each ablation model so weight initialisation and DataLoader sampling
     are independent between models trained in the same fold.
 12. LOO route re-normalisation for training drivers:
     When re-normalising training driver p_tr's windows with route-level
     statistics, driver p_tr's own session stats are now excluded from the
     route mean (leave-one-out).  Previously _route_tr_stats_d included
     p_tr's contribution, creating a subtle circular dependency where each
     driver was partially normalised using its own baseline.
 13. LOO route re-normalisation for validation drivers:
     Same fix for validation drivers used in early stopping.  Val drivers'
     own session stats are now excluded from the route mean, ensuring they
     are treated identically to the test driver.
 14. Platt calibrator sign guard:
     platt_adapt() now checks that the fitted LogisticRegression coefficient
     w > 0 before applying calibration.  With K=15 support samples and heavy
     class imbalance the calibrator can fit w ≤ 0 (inverted scores), which
     would break the monotonicity assumption required for mixing raw and
     Platt-scaled scores in a single AUC evaluation.  Falls back to
     population scores when the coefficient is non-positive.
 15. Sampling-rate validation:
     main() now checks that all sessions have median Timestamp Δt ≈ 1 s
     before proceeding.  A deviation would shift all three spectral band
     boundaries away from their documented Hz values.
 16. Platt scaling applied uniformly to all windows:
     platt_adapt() now applies the fitted calibrator to ALL population scores
     (support + eval) rather than returning raw scores for the support set and
     Platt-scaled scores for the eval set.  Mixing two different score scales
     invalidates cross-window rankings for AUC: monotonicity holds within each
     scale separately but not across them (a raw score of 0.7 and a Platt score
     of 0.65 have no well-defined relative ordering).
 17. Pooled standard deviation in LOO route re-normalisation:
     _loo_renorm() and _route_tr_stats_d now use pooled σ = sqrt(mean(σ²))
     instead of the arithmetic mean of σ when aggregating per-session standard
     deviations across drivers on the same route.  The arithmetic mean of σ
     underestimates the true population spread when session variances are
     heterogeneous (e.g. one calm and one aggressive driver on the same route).
 18. Uniform-stride training windows for TCN:
     The TCN training pool now uses fine_stride=False (uniform WINDOW_STEP),
     matching the test-time window distribution and eliminating up to 11
     consecutive windows with 89/90-step temporal overlap that arose when
     fine_stride=True oversampled the EVENT_VICINITY region at stride=1.
     SMOTE and class-weighted focal loss handle the class imbalance that was
     previously addressed by the fine-stride vicinity oversampling.
 19. Validation driver fraction increased to 25 %:
     _val_sample fraction raised from 0.20 to 0.25 (≈4 → ≈5 drivers per fold)
     to reduce early-stopping variance caused by overly small validation sets.
 20. Personalisation AUC evaluated on eval-only windows:
     gate_adapt / head_adapt / platt_adapt AUC is now computed on y_te[K:] /
     scores[K:] only, excluding the K support windows that were used to fit the
     calibrator or fine-tune the model.  Including them inflated AUC because the
     adapted model had already been trained on their labels.  The returned
     score arrays still cover all windows (for a single consistent scale) but
     only the eval portion is used for all metric computation.
 21. Spectral features regularised by GRL via spec_adapter:
     A learned linear projection (spec_adapter) is inserted between the raw
     spectral input and the head.  During DANN training the discriminator
     receives torch.cat([fusion, spec_adapted]) so its GRL-reversed gradients
     flow back through spec_adapter, making its weights learn a subject-invariant
     spectral representation.  Without this, subject-specific frequency signatures
     in the precomputed spectral features bypassed the adversarial objective
     entirely, since backprop cannot reach numpy-precomputed constants.
 22. PCA-reduced nearest-neighbour search in SMOTE:
     The k-NN distance matrix in smote_raw is now computed in a PCA-compressed
     space (min(n_pos-1, 50) components) rather than in the full 540-d raw-signal
     space (T=90, C=6).  In 540 dims Euclidean distances concentrate, making the
     5th nearest neighbour barely closer than the 50th, so SMOTE interpolations
     were effectively random.  PCA retains the principal axes of variance and
     makes distance-based neighbourhood meaningful.
 23. LOO renorm floor raised from 1e-12 to 1e-6:
     _loo_renorm() divided by max(rte_sig, 1e-12).  For near-constant routes
     (rte_sig → 0) this could silently scale features by up to 1e12×, producing
     numerically corrupted TCN inputs for specific drivers.  The floor is now
     1e-6, matching the additive epsilon used in normalize_signals().
 24. Inductive domain adaptation — BinaryDomainDiscriminator removed:
     A previous version used the held-out test subject's unlabeled windows as the
     target domain during DANN training (transductive DANN).  This was removed
     because it allows the feature extractor to see the test subject's feature
     distribution before evaluation, violating the inductive LOPO assumption.
     The multi-class SubjectDiscriminator across training subjects is retained;
     it provides cross-subject invariance without any access to the test subject.
 25. StandardScaler for LR baseline features:
     window_baseline_feats() returns raw mean/std/max statistics whose scales
     differ by orders of magnitude across signals (HR ~50-100 bpm, steering in
     degrees, speed in km/h).  Without scaling, L2 regularisation in
     LogisticRegression penalises large-scale features less, biasing LR toward
     lower performance.  A StandardScaler (fit on training windows only) is now
     applied before LR.fit(); XGB is scale-invariant and unchanged.
 26. Driver-level F1/Precision/Recall in threshold metrics:
     Consecutive windows overlap 94% of timesteps (WINDOW_STEP=5, LOOKBACK_S=90),
     making pooled threshold metrics over all windows appear more reliable than
     they are.  Per-driver F1/Prec/Rec (mean ± std across LOPO folds) is now
     reported alongside the pooled value, correctly treating each driver as one
     independent observation.
 27. Window demeaning before FFT in spectral features:
     compute_spectral_features_batch() now subtracts the per-column window mean
     before np.fft.rfft().  Previously, DC energy from the LOO renormalisation
     residual (µ_sess − µ_route_loo) leaked into Band 0 [0.0, 0.1) Hz, partly
     encoding driver identity rather than slow-drift frequency content.  Demeaning
     makes Band 0 represent genuine low-frequency oscillations and makes all band
     powers invariant to the absolute level shift introduced by renormalisation.
 28. Val-excluded normalization reference for test driver:
     The route-level statistics used to renormalise the test driver's windows
     (_route_mus_d / _route_sigs_d) now exclude validation drivers, making the
     reference pool identical to the one used for training-driver LOO renorm.
     Previously, val drivers contributed to the test reference but not to the
     train reference, creating a subtle distribution mismatch: test windows were
     normalised against a slightly different route mean than the training windows
     they were evaluated against.  The symmetric exclusion (non-test, non-val
     only) eliminates this asymmetry at the cost of a smaller reference pool.
 30. SMOTE synthetic windows included in adversarial training (anchor subject ID):
     Previously synthetic windows received sentinel subject ID -1 and were excluded
     from the adversarial discriminator loss, so ~50 % of augmented training data
     never received a GRL gradient.  Each synthetic window now inherits the anchor's
     subject ID, allowing the full training set to contribute to subject-invariant
     representation learning.
 31. ADV_WEIGHT raised from 0.1 to 0.3 (matched to AUX_PHYS_WEIGHT):
     With ADV_WEIGHT=0.1 the auxiliary physiology loss (weight=0.3) dominated the
     non-classification terms 3:1, suppressing the adversarial objective and reducing
     DANN's effective contribution to cross-subject invariance.  Setting both weights
     equal ensures neither term structurally dominates the other.
 32. Platt scaler lr.fit() wrapped in try/except:
     If the K=15 support samples produce a degenerate input (e.g. constant scores)
     sklearn's LogisticRegression.fit() can raise rather than returning a valid model.
     The exception is caught and platt_adapt falls back to population scores instead
     of propagating a runtime error.
 33. Pooled window-level CI removed from _print_pooled output:
     bootstrap_auc_ci_windows produces anticonservative CIs due to ~94 % window
     overlap and was already labelled "do not report" in the previous output.
     The window CI is no longer computed or displayed; only the point estimate is
     shown alongside the driver-level CI (the primary reported metric).
 29. WeightedRandomSampler bypassed when data is already balanced:
     get_dann_loader() now uses plain shuffle=True when the majority/minority
     class ratio is ≤ 1.5× (e.g. after SMOTE).  Previously it always used
     WeightedRandomSampler, which samples with replacement — on balanced data
     this means each sample appears on average once per epoch with high variance,
     whereas shuffle=True gives exactly one occurrence per epoch (lower variance,
     no wasted gradient steps).  The weighted path is retained for the imbalanced
     case (USE_SMOTE=False) where it is strictly necessary.
 34. train_single_tcn WeightedRandomSampler bypassed when already balanced:
     The same fix #29 ratio guard (ratio > 1.5×) is now applied inside
     train_single_tcn().  Previously it always used WeightedRandomSampler even
     after SMOTE had balanced the data, making the ablation training loop
     inconsistent with get_dann_loader() and introducing sampling-with-replacement
     bias on balanced data.  Now shuffle=True is used when the training set is
     balanced, making the ablation comparison fair vs DB-TCN / DANN-DB-TCN.
 36. train_single_tcn pos_weight parameter added:
     train_single_tcn() previously had no pos_weight parameter and always called
     focal_loss without class weighting, while train_db_tcn() and train_dann_tcn()
     both accept and use pos_weight.  When USE_SMOTE=False the training set is
     class-imbalanced; DB-TCN and DANN-DB-TCN received pw_bl_smote=pw_uniform
     (≈5-10×) via focal loss, but the ablation SingleBranchTCN models received
     no weighting — a systematic disadvantage.  pos_weight is now threaded through
     train_single_tcn() and the ablation call site passes pw_bl_smote, making the
     focal-loss treatment identical across all trained models.
 35. PYTHONHASHSEED guard no longer self-defeating:
     The os.environ["PYTHONHASHSEED"] = str(SEED) assignment has been removed.
     Setting the env var inside the running interpreter only propagates to child
     processes, not to the current interpreter's hash seed (which is fixed at
     startup).  The assignment therefore always made the guard's condition false,
     silently bypassing the exit() check.  Removing the assignment restores the
     guard's intended behaviour: the script exits with an actionable error if it
     was not launched with PYTHONHASHSEED=42.
 37. safe_auc / safe_auprc: ``or float("nan")`` replaced with _nan_or():
     All call sites previously used the pattern ``safe_auc(...) or float("nan")``.
     The ``or`` operator treats 0.0 as falsy, so a valid AUC / AUPRC of exactly
     0.0 (perfectly wrong or degenerate predictions) would be silently replaced by
     NaN, corrupting per-driver and pooled metric tables.  A dedicated helper
     _nan_or(x) performs an explicit None check and is used throughout.
 38. Dead constants MIN_POSITIVES and N_PERM_DRIVERS removed:
     MIN_POSITIVES = 1 was never referenced anywhere (MIN_EVAL_POSITIVES = 5 is
     used instead).  N_PERM_DRIVERS = 10 was also unused — the permutation loop
     uses all available folds (len(pool_models_dann)).  Removing them prevents
     future confusion about which constant is actually in effect.
 39. drv_aucs_* list comprehensions compute safe_auc only once per pair:
     All driver-AUC list comprehensions previously called safe_auc(y, s) twice —
     once in the filter condition and once in the value expression — doubling the
     computation and making the code harder to reason about.  Generator expressions
     now compute the value once and filter on it.
 40. SMOTE subject-ID fix: always use anchor ID (fix #30 implementation corrected):
     Fix #30's docstring stated synthetic windows inherit the anchor's subject ID,
     but the implementation used a λ-weighted rule (neighbour's ID when λ ≥ 0.5).
     This created inconsistent subject attribution: two synthetics from the same
     anchor-neighbour pair could carry different IDs based purely on λ, artificially
     inflating cross-subject diversity in the adversarial discriminator's training
     set and undermining the intended DANN invariance signal.  The implementation
     now always uses the anchor's subject ID, matching the documented intent.
 41. Platt calibrator |w| instability guard added:
     platt_adapt() previously only checked w > 0 (fix #14) but not for numerical
     blow-up.  With K=15 support windows and class imbalance, LogisticRegression
     can fit a very large |w| (>>1) when the population scores are well-separated
     within the support set.  A large |w| makes the calibrated probabilities nearly
     binary {0,1}, erasing score gradations and causing abrupt ranking changes
     for drivers with few positives in the adaptation window.  platt_adapt() now
     falls back to population scores when |w| > MAX_PLATT_W = 20.0.
 42. Platt calibrator exception now logged via warnings.warn:
     The silent except-pass on lr.fit() (fix #32) masked degenerate support sets
     without any diagnostic signal.  A warnings.warn is now emitted so callers
     can identify drivers whose support sets consistently fail calibration.

Domain-Adversarial Dual-Branch TCN (DANN-DB-TCN) for driving impairment prediction.

Extension of tcn_dual_branch.py with two complementary improvements:

1.  Domain-Adversarial Training (DANN)
    ───────────────────────────────────
    In LOPO evaluation the test subject is completely unseen during training.
    Standard ERM implicitly encodes subject-specific patterns in the feature
    extractor, degrading cross-subject generalisation.

    DANN (Ganin et al., 2016) attaches a subject discriminator to the shared
    fusion vector via a Gradient Reversal Layer (GRL):

        cat([fused_96d, spec_adapted_18d]) ──GRL(λ)──► SubjectDiscriminator ──► P(subject | x)

    Training objective (jointly):
        L_total = L_focal(y_pred, y)  +  α_adv · L_CE(subj_pred, subj_id)

    The GRL reverses gradients through the feature extractor, so it is trained
    to simultaneously predict risk well AND confuse the discriminator.
    Net effect: representations invariant to subject identity.

    Lambda annealing (modified DANN schedule, Ganin et al.):
        λ(p) = λ_max · (2 / (1 + exp(−10·p)) − 1),  p = epoch / T_max
    p is floored at LAMBDA_WARMUP_P=0.05 so λ ≈ 0.24 at epoch 0, avoiding a
    cold-start instability seen when the discriminator dominates at λ=0.

2.  Kinematic Spectral Features
    ─────────────────────────────
    Rolling statistics (mean, std) capture amplitude but miss frequency
    structure. Steering micro-corrections in the 0.1–0.5 Hz band are a
    well-documented distraction indicator (Verwey & Zaidel, 1999;
    Rommerskirchen et al., 2007; Patten et al., 2004).

    Per-window FFT band powers are computed over three bands:
        Band 1  [0.00, 0.10) Hz — slow drift, long-horizon manoeuvres
        Band 2  [0.10, 0.30) Hz — medium-frequency steering corrections
        Band 3  [0.30, 0.50] Hz — high-frequency micro-corrections (distraction)

    6 signals (4 kinematic + 2 physiological) × 3 bands = 18 spectral features per window.
    Band powers are normalised by total window power → amplitude-invariant,
    cross-subject relative spectral energy.

    Spectral features are injected at the head level (post-fusion), keeping the
    TCN branch architecture unchanged and maintaining interpretability of α.

Architecture changes relative to tcn_dual_branch.py:
    • Head input: 96 (fusion) + 18 (spectral) = 114-d
    • GRL + SubjectDiscriminator attached to cat([96-d fusion, 18-d spec_adapted]) during training
    • Separate StandardScaler for spectral features (fit on training fold only)

Outputs (mirrors tcn_dual_branch.py):
     1.  Label validity report
     2.  Pipeline configuration
     3.  Per-driver table: LR | XGB | DB-TCN | DANN-Pop | GateAdapt | Gain | gate(α)
     4.  Pooled AUC 95% CI, AUPRC, Brier, ECE
     5.  Threshold metrics (F1 / Precision / Recall @ Youden's J)
     6.  Gate distribution analysis + risk correlation
     7.  Permutation importance (by signal and by branch)
     8.  Per-band spectral feature importance
     9.  Modality ablation: Phys-only | Kin-only | DB-TCN | DANN-DB-TCN
    10.  DANN adversarial diagnostic (discriminator accuracy vs chance)
    11.  Wilcoxon signed-rank test: DANN-DB-TCN vs DB-TCN
    12.  Stratified evaluation (CLC vs non-CLC positive windows)
    13.  Per-driver summary statistics
    14.  JSON artefacts in impairment_results/dann_db_tcn_results.json
"""

import copy
import hashlib
import json
import math
import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                              average_precision_score, f1_score,
                              precision_score, recall_score)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu, wilcoxon, pearsonr
from scipy.spatial.distance import cdist
import xgboost as xgb
import random

# ── CONFIG ──────────────────────────────────────────────────────────────────────
SEED   = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SIGNAL_COLS  = ["arousal", "hr", "steeringWheelAngle", "steeringTorq", "acceleration.y", "speed.x"]
VEHICLE_COLS = ["steeringWheelAngle", "steeringTorq", "acceleration.y", "speed.x"]
PHYS_COLS    = ["arousal", "hr"]
KIN_COLS     = ["steeringWheelAngle", "steeringTorq", "acceleration.y", "speed.x"]
PHYS_IDX     = [SIGNAL_COLS.index(c) for c in PHYS_COLS]   # [0, 1]
KIN_IDX      = [SIGNAL_COLS.index(c) for c in KIN_COLS]    # [2, 3, 4, 5]

SEVERITY = {
    "Collision":               5,
    "Red_light_violation":     3,
    "panic_braking_with_stop": 2,
    "panic_braking":           1,
    "sharp_turn":              1,
}
# center_line_crossing excluded: dominated by kinematic precursors,
# suppresses physiological signal in population-level LOPO.

# Windowing
LOOKBACK_S     = 90
WINDOW_STEP    = 5
GAP, HORIZON   = 5, 10
EVENT_VICINITY = 10
ROLL_K         = 10
BATCH_SIZE     = 64
WEIGHT_DECAY   = 1e-4
JITTER_STD     = 0.01
CUTOUT_LEN     = 5
CUTOUT_PROB    = 0.2

# Training
EPOCHS  = 100
LR      = 1e-3
PATIENCE      = 10  # early-stopping patience for DB-TCN (epochs without val-AUC improvement)
DANN_PATIENCE = 15  # longer patience for DANN: adversarial + GRL annealing makes val-AUC noisier

# ── DANN hyperparameters ─────────────────────────────────────────────────────────
LAMBDA_MAX      = 1.0   # maximum GRL reversal strength
ADV_WEIGHT      = 0.3   # weight of adversarial loss in total loss; matched to AUX_PHYS_WEIGHT so adversarial regularisation is not dominated by the auxiliary physiology term
LAMBDA_WARMUP_P   = 0.05  # floor on p → λ ≈ 0.24 at epoch 0 (avoids cold-start instability)

# ── Auxiliary physiological branch loss ──────────────────────────────────────────
# Adds focal_loss(phys_branch_pred, y) * AUX_PHYS_WEIGHT to the training objective,
# forcing the physiology TCN to learn discriminative representations independently
# of the kinematics branch, preventing gradient collapse toward kinematics.
AUX_PHYS_WEIGHT = 0.3

# ── Spectral features ─────────────────────────────────────────────────────────────
# At fs = 1 Hz (arousal/HR/vehicle logged at ~1 Hz), Nyquist = 0.5 Hz.
# Three bands chosen to isolate known driving-distraction frequency signatures.
SPECTRAL_BANDS = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5)]   # inclusive upper bound on final band
SPECTRAL_DIM   = (len(KIN_COLS) + len(PHYS_COLS)) * len(SPECTRAL_BANDS)   # 6 × 3 = 18

# Gate personalisation
GATE_ADAPT_K     = 15
GATE_ADAPT_STEPS = 20
GATE_ADAPT_LR    = 5e-4
MIN_SUPPORT_POSITIVES = 3   # minimum positives in K support windows for personalisation
MAX_PLATT_W = 20.0          # |w| ceiling for Platt calibrator; above this the sigmoid is nearly binary (fix #41)

# Evaluation
N_BOOTSTRAP        = 2000
MIN_EVAL_POSITIVES = 5   # exclude drivers with < 5 real positives from evaluation
N_PERM_REPEATS     = 10  # permutation repeats per signal per fold (reduces variance of importance)

# SMOTE oversampling
USE_SMOTE          = True   # set False to disable SMOTE (uses class-weighted focal loss only)
SMOTE_K_NEIGHBORS  = 5      # k for SMOTE nearest-neighbour search
# Salt added to per-fold seed when constructing the SMOTE RNG so that it is
# independent of fold_rng (which uses the same SEED ^ seed_d base).
# Prevents SMOTE interpolation choices from being correlated with the val split.
SMOTE_SEED_SALT    = 0xABCD

# PYTHONHASHSEED must be set before interpreter startup (CPython limitation).
# If not already set correctly, re-exec this script with the env var in place.
if os.environ.get("PYTHONHASHSEED") != str(SEED):
    import subprocess
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(SEED)
    sys.exit(subprocess.call([sys.executable] + sys.argv, env=env))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True   # ensures bit-exact reproducibility
torch.backends.cudnn.benchmark     = False  # benchmark=True disables determinism
torch.set_num_threads(1)                    # strict cross-machine CPU reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # add this
torch.use_deterministic_algorithms(True)             # unconditional

OUT_DIR = Path(__file__).parent / "impairment_results"
OUT_DIR.mkdir(exist_ok=True)

# ── PREPROCESSING ────────────────────────────────────────────────────────────────

def mark_event_onsets(df):
    df = df.copy()
    for _, grp in df.groupby(["id", "route"]):
        idx = grp.index
        for col in SEVERITY:
            if col not in df.columns:
                continue
            vals  = grp[col].fillna(0).values
            onset = np.zeros(len(vals), dtype=float)
            for i in range(len(vals)):
                if vals[i] > 0 and (i == 0 or vals[i - 1] == 0):
                    onset[i] = 1.0
            df.loc[idx, col] = onset
    return df


def normalize_signals(df):
    df = df.copy()
    prefix = LOOKBACK_S + GAP   # rows used for computing statistics (avoids temporal leakage)
    for _, grp in df.groupby(["id", "route"]):
        idx = grp.index
        for col in VEHICLE_COLS:
            mu  = grp.iloc[:prefix][col].mean()
            sig = grp.iloc[:prefix][col].std() + 1e-6
            df.loc[idx, col] = (grp[col] - mu) / sig
        # Per-session z-score for physiological signals: same treatment as vehicle
        # signals, making both relative to each driver's intra-session baseline.
        for phys_col in ("hr", "arousal"):
            if phys_col in df.columns:
                mu  = grp.iloc[:prefix][phys_col].mean()
                sig = grp.iloc[:prefix][phys_col].std() + 1e-6
                df.loc[idx, phys_col] = (grp[phys_col] - mu) / sig
    return df

# ── FEATURE ENGINEERING ──────────────────────────────────────────────────────────

def engineer_features(window):
    """(T, C) → (T, 5C): raw | diff1 | roll_mean | roll_std | z-score."""
    T, C    = window.shape
    diff1   = np.diff(window, axis=0, prepend=window[:1])
    cs      = np.cumsum(window,      axis=0)
    cs_sq   = np.cumsum(window ** 2, axis=0)
    cs_lag    = np.zeros_like(cs)
    cs_sq_lag = np.zeros_like(cs_sq)
    if ROLL_K < T:
        cs_lag[ROLL_K:]    = cs[:T - ROLL_K]
        cs_sq_lag[ROLL_K:] = cs_sq[:T - ROLL_K]
    win_len   = np.minimum(np.arange(1, T + 1)[:, None], ROLL_K)
    roll_mean = (cs - cs_lag) / win_len
    roll_var  = (cs_sq - cs_sq_lag) / win_len - roll_mean ** 2
    roll_std  = np.sqrt(np.maximum(roll_var, 0.0)) + 1e-6
    w_mean    = window.mean(axis=0, keepdims=True)
    w_std     = window.std(axis=0,  keepdims=True) + 1e-6
    z_score   = (window - w_mean) / w_std
    return np.concatenate([window, diff1, roll_mean, roll_std, z_score],
                          axis=1).astype(np.float32)


def apply_features_branch(X_raw, col_idx):
    """Apply engineer_features to a subset of channels. → (N, T, len(col_idx)*5)."""
    return np.stack([engineer_features(w[:, col_idx]) for w in X_raw])


def apply_features_all(X_raw):
    return np.stack([engineer_features(w) for w in X_raw])


def window_baseline_feats(X_raw, col_idx=None):
    X = X_raw if col_idx is None else X_raw[:, :, col_idx]
    return np.concatenate([X.mean(1), X.std(1), X.max(1)], axis=1).astype(np.float32)

# ── SPECTRAL FEATURES ────────────────────────────────────────────────────────────

def compute_spectral_features_batch(X_kin_raw, sample_dt=1.0):
    """
    Compute normalised spectral band power features for a batch of windows.

    Parameters
    ----------
    X_kin_raw : ndarray, shape (N, T, C)
        Raw (pre-feature-engineering) signals for one branch (kinematic or physiological).
    sample_dt : float, optional
        Median sampling interval in seconds (default 1.0).  Used as ``d`` in
        np.fft.rfftfreq so that frequency bins are labelled correctly when the
        actual fs deviates from 1 Hz.  Obtain from ``_check_sampling_rate()``.

    Returns
    -------
    spectral : ndarray, shape (N, C * len(SPECTRAL_BANDS))
        Relative band powers (signal_power_in_band / total_signal_power).
        Layout: [band0_sig0, ..., band0_sigC-1,
                 band1_sig0, ..., band2_sigC-1]
        After interleaving kin+phys the final tensor is (N, SPECTRAL_DIM=18).

    Physiological motivation
    ─────────────────────────
    At 1 Hz sampling (T=90), freq resolution = 1/90 ≈ 0.011 Hz.
    The 0.1–0.5 Hz steering band contains micro-correction oscillations
    whose frequency increases under secondary-task cognitive load
    (Verwey & Zaidel, 1999; Patten et al., 2004).
    Normalised band power is amplitude-invariant → cross-subject robust.
    """
    N, T, C = X_kin_raw.shape
    if T < 2:
        # rfft on a length-1 window yields only DC — band powers are meaningless.
        return np.zeros((N, C * len(SPECTRAL_BANDS)), dtype=np.float32)
    freqs   = np.fft.rfftfreq(T, d=sample_dt)   # (T//2 + 1,) with correct Hz labels
    # All bands use half-open [lo, hi); the final band uses closed [lo, hi] so
    # the Nyquist bin (0.5 Hz) is always included regardless of FFT resolution.
    masks   = [(freqs >= lo) & (freqs < hi) for lo, hi in SPECTRAL_BANDS[:-1]]
    masks  += [(freqs >= SPECTRAL_BANDS[-1][0]) & (freqs <= SPECTRAL_BANDS[-1][1])]

    out = np.zeros((N, C * len(SPECTRAL_BANDS)), dtype=np.float32)
    for i, win in enumerate(X_kin_raw):           # win: (T, C)
        # Demean each column before FFT (fix #27): removes the DC component so
        # that Band 0 [0.0, 0.1) Hz captures genuine slow oscillations rather than
        # the level shift introduced by LOO route renormalisation (µ_sess − µ_route_loo).
        win_dm  = win - win.mean(axis=0, keepdims=True)    # (T, C), DC = 0
        fft_pow = np.abs(np.fft.rfft(win_dm, axis=0)) ** 2  # (F, C)
        total   = fft_pow.sum(axis=0) + 1e-10              # (C,)
        for b, mask in enumerate(masks):
            out[i, b * C : (b + 1) * C] = fft_pow[mask].sum(axis=0) / total
    return out   # (N, C * len(SPECTRAL_BANDS))

# ── SMOTE ────────────────────────────────────────────────────────────────────────

def smote_raw(X_raw, y, k=SMOTE_K_NEIGHBORS, rng=None,
              pids=None, routes=None, t_starts=None, channel_weights=None):
    """
    SMOTE oversampling in raw signal space (T×C dims).

    Operates on raw signal windows (N, T, C) — 540 dims at T=90, C=6 —
    rather than flattened engineered features (~2718 dims). Euclidean
    distances in raw signal space are more meaningful than in the
    high-dimensional engineered feature space, avoiding the curse of
    dimensionality that makes neighbour search unreliable in the old approach.

    Applied to training data only — never to validation or test sets.

    Parameters
    ----------
    X_raw : ndarray — (N, T, C) raw signal windows (pre-feature-engineering)
    y     : ndarray — (N,) binary labels
    k     : int     — number of nearest neighbours for SMOTE
    rng   : numpy Generator or None

    Returns
    -------
    X_raw_aug, y_aug, syn_anchor_idx, syn_neighbor_idx, syn_lambdas
        syn_anchor_idx   : ndarray (n_synthetic,) — anchor index into original X_raw
        syn_neighbor_idx : ndarray (n_synthetic,) — neighbour index into original X_raw
        syn_lambdas      : ndarray (n_synthetic,) float32 — interpolation weight λ ∈ [0,1]
            The synthetic sample is (1-λ)·anchor + λ·neighbour in raw signal space.
            Callers can use these to re-interpolate in any feature space.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    if n_pos == 0 or n_pos >= n_neg:
        return (X_raw, y,
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32))  # already balanced or no positives

    n_synthetic = n_neg - n_pos

    # Flatten raw windows for nearest-neighbour computation (T*C dims)
    X_flat = X_raw[pos_idx].reshape(n_pos, -1)   # (n_pos, T*C)

    # Apply optional channel weights before PCA/NN search only (not to interpolation).
    # This equalises the contribution of different branches (e.g. phys vs kin) to the
    # Euclidean distance used for nearest-neighbour selection.
    if channel_weights is not None:
        cw = np.asarray(channel_weights, dtype=np.float32)
        X_flat_nn_input = X_flat * cw[np.newaxis, :]  # (n_pos, T*C), weighted
    else:
        X_flat_nn_input = X_flat

    k_eff = min(k, n_pos - 1)
    if k_eff < 1:
        # Too few positives for SMOTE — fall back to random duplication
        print(f"  [WARN smote_raw] only {n_pos} positive window(s) — "
              f"SMOTE impossible (need ≥2); falling back to random duplication.")
        dup_idx = rng.choice(pos_idx, n_synthetic, replace=True)
        return (np.concatenate([X_raw, X_raw[dup_idx]], axis=0),
                np.concatenate([y,     np.ones(n_synthetic, dtype=y.dtype)]),
                dup_idx,
                dup_idx,  # neighbour == anchor for pure duplicates
                np.zeros(n_synthetic, dtype=np.float32))  # lambda=0: fully anchor
    if k_eff < k:
        print(f"  [WARN smote_raw] k reduced from {k} to {k_eff} "
              f"(only {n_pos} positive windows available); "
              f"after overlap exclusion neighbour diversity may be further limited.")

    # Reduce dimensionality before NN search to avoid distance concentration in
    # the full 540-d space (T=90, C=6): in high dimensions Euclidean distances
    # concentrate so the k-th neighbour is barely closer than a random point.
    # PCA retains the principal axes of variance; distances in this compressed
    # space are more meaningful proxies for signal similarity.
    # sklearn PCA uses full SVD by default (deterministic for fixed data).
    # Cap components at n_pos//5 so PCA is not under-determined when positives
    # are few.  Rule of thumb: ≥5 samples per component prevents the compressed
    # space from being nearly full-rank (where NN distances lose meaning).
    _pca_k = min(n_pos - 1, X_flat_nn_input.shape[1], 50, max(2, n_pos // 5))
    if _pca_k >= 2:
        _pca       = PCA(n_components=_pca_k)
        X_flat_nn  = _pca.fit_transform(X_flat_nn_input)
    else:
        X_flat_nn  = X_flat_nn_input

    # For each positive, find k nearest neighbours (among positives only)
    dists = cdist(X_flat_nn, X_flat_nn, metric="euclidean")   # (n_pos, n_pos)
    np.fill_diagonal(dists, np.inf)

    # Exclude overlapping pairs: two windows from the same (pid, route) session
    # that start within LOOKBACK_S timesteps of each other share the majority of
    # their signal data.  Interpolating between them produces a synthetic sample
    # nearly identical to existing real windows, adding no real diversity.
    # Assumes t_starts are in the same unit as LOOKBACK_S (seconds at ~1 Hz).
    if pids is not None and routes is not None and t_starts is not None:
        pos_pids   = pids[pos_idx]
        pos_routes = routes[pos_idx]
        pos_ts     = t_starts[pos_idx].astype(float)
        same_sess  = (pos_pids[:, None] == pos_pids[None, :]) & \
                     (pos_routes[:, None] == pos_routes[None, :])
        tdiff      = np.abs(pos_ts[:, None] - pos_ts[None, :])
        dists[same_sess & (tdiff < LOOKBACK_S)] = np.inf

    nn_idx = np.argsort(dists, axis=1)[:, :k_eff]      # (n_pos, k_eff)

    # Generate synthetic samples by interpolating between anchor and neighbour.
    # Return neighbour indices and lambdas so callers can re-interpolate in
    # scaled feature space (avoids nonlinear artifacts from roll_std / z-score).
    T, C             = X_raw.shape[1], X_raw.shape[2]
    syn              = np.zeros((n_synthetic, T, C), dtype=X_raw.dtype)
    syn_anchor_idx   = np.zeros(n_synthetic, dtype=np.int64)
    syn_neighbor_idx = np.zeros(n_synthetic, dtype=np.int64)
    syn_lambdas      = np.zeros(n_synthetic, dtype=np.float32)

    for i in range(n_synthetic):
        anchor = rng.integers(0, n_pos)
        nn     = nn_idx[anchor, rng.integers(0, k_eff)]
        lam    = rng.uniform(0.0, 1.0)
        syn[i] = (X_raw[pos_idx[anchor]]
                  + lam * (X_raw[pos_idx[nn]] - X_raw[pos_idx[anchor]]))
        syn_anchor_idx[i]   = pos_idx[anchor]  # index into original X_raw
        syn_neighbor_idx[i] = pos_idx[nn]       # index into original X_raw
        syn_lambdas[i]      = lam

    X_raw_aug = np.concatenate([X_raw, syn], axis=0)
    y_aug     = np.concatenate([y,     np.ones(n_synthetic, dtype=y.dtype)])

    return X_raw_aug, y_aug, syn_anchor_idx, syn_neighbor_idx, syn_lambdas

# ── WINDOWING ────────────────────────────────────────────────────────────────────

def composite_risk_score(future_df):
    return sum(
        SEVERITY[col] * int((future_df[col] > 0).any())
        for col in SEVERITY
    )


def future_error_types(future_df):
    return frozenset(
        col for col in SEVERITY
        if col in future_df.columns and (future_df[col] > 0).any()
    )


def build_windows(df):
    """Build uniform-stride windows (WINDOW_STEP) for all sessions in df."""
    windows, labels, scores, pids, etypes, routes, t_starts = [], [], [], [], [], [], []
    min_session_len = LOOKBACK_S + GAP + HORIZON
    nan_skip_counts: dict = {}   # {pid: (skipped, total)} — for bias audit
    for (pid, route), grp in df.groupby(["id", "route"]):
        grp = grp.sort_values("Timestamp").reset_index(drop=True)
        n   = len(grp)
        ts  = grp["Timestamp"].values

        if n < min_session_len:
            print(f"  [WARN] build_windows: session (pid={pid}, route={route}) has {n} rows "
                  f"< minimum {min_session_len} — skipping.")
            continue

        idx = 0
        while idx + LOOKBACK_S + GAP + HORIZON <= n:
            sig = grp.iloc[idx: idx + LOOKBACK_S][SIGNAL_COLS].values.astype(np.float32)
            prev_skipped, prev_total = nan_skip_counts.get(pid, (0, 0))
            if not np.isnan(sig).any():
                future = grp.iloc[idx + LOOKBACK_S + GAP: idx + LOOKBACK_S + GAP + HORIZON]
                score  = composite_risk_score(future)
                windows.append(sig);  labels.append(int(score > 0))
                scores.append(score); pids.append(pid)
                etypes.append(future_error_types(future))
                routes.append(route); t_starts.append(ts[idx])
                nan_skip_counts[pid] = (prev_skipped, prev_total + 1)
            else:
                nan_skip_counts[pid] = (prev_skipped + 1, prev_total + 1)
            idx += WINDOW_STEP

    # Report NaN-skipped windows so callers can audit selection bias.
    total_skip = sum(s for s, _ in nan_skip_counts.values())
    total_all  = sum(t for _, t in nan_skip_counts.values())
    if total_skip > 0:
        print(f"  [NaN-skip audit] {total_skip}/{total_all} windows dropped due to NaN "
              f"({100*total_skip/max(total_all,1):.1f}%) — per-participant breakdown:")
        for pid_k, (ns, nt) in sorted(nan_skip_counts.items()):
            if ns > 0:
                print(f"    pid={pid_k}: {ns}/{nt} skipped ({100*ns/max(nt,1):.1f}%)")

    return (np.array(windows,  dtype=np.float32),
            np.array(labels,   dtype=np.float32),
            np.array(scores,   dtype=np.float32),
            np.array(pids),
            np.array(etypes, dtype=object),
            np.array(routes),
            np.array(t_starts))

# ── DATASET ──────────────────────────────────────────────────────────────────────

class DANNDataset(Dataset):
    """
    Dataset for DANN training.

    Yields (x_phys, x_kin, x_spec, y, subj_id) tuples.
    subj_id is used only during training for the adversarial loss;
    at test time it is set to -1 and ignored.
    """
    def __init__(self, X_phys, X_kin, X_spec, y, subj_ids=None, augment=False):
        self.Xp      = torch.as_tensor(X_phys).float()
        self.Xk      = torch.as_tensor(X_kin).float()
        self.Xs      = torch.as_tensor(X_spec).float()
        self.y       = torch.as_tensor(y).float()
        self.sids    = (torch.as_tensor(subj_ids).long()
                        if subj_ids is not None
                        else torch.full((len(y),), -1, dtype=torch.long))
        self.augment = augment

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        xp = self.Xp[idx].clone()
        xk = self.Xk[idx].clone()
        xs = self.Xs[idx].clone()
        if self.augment:
            xp += torch.randn_like(xp) * JITTER_STD
            xk += torch.randn_like(xk) * JITTER_STD
            # Use torch random ops (not global Python random) so augmentation
            # respects torch.manual_seed and is deterministic across runs.
            if torch.rand(1).item() < CUTOUT_PROB and xp.shape[0] > CUTOUT_LEN:
                t0 = torch.randint(0, xp.shape[0] - CUTOUT_LEN, (1,)).item()
                xp[t0: t0 + CUTOUT_LEN] = 0.0
                xk[t0: t0 + CUTOUT_LEN] = 0.0
        return xp, xk, xs, self.y[idx], self.sids[idx]


def get_dann_loader(Xp, Xk, Xs, y, subj_ids=None, batch_size=BATCH_SIZE, augment=False):
    y_int  = y.astype(int)
    counts = np.bincount(y_int, minlength=2)
    ds     = DANNDataset(Xp, Xk, Xs, y, subj_ids, augment=augment)
    ratio  = counts.max() / (counts.min() + 1e-6)
    if ratio > 1.5:
        # Imbalanced data: WeightedRandomSampler oversamples the minority class.
        weights = 1.0 / (counts + 1e-6)
        sw      = torch.from_numpy(weights[y_int])
        return DataLoader(ds, batch_size=batch_size, sampler=WeightedRandomSampler(sw, len(sw)))
    # Balanced data (e.g. after SMOTE, fix #29): shuffle without replacement so
    # each sample appears exactly once per epoch instead of being drawn with
    # replacement from an approximately uniform distribution.
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

# ── GRADIENT REVERSAL LAYER ──────────────────────────────────────────────────────

class _GRL(torch.autograd.Function):
    """
    Gradient Reversal Layer (Ganin & Lempitsky, 2015).

    Forward  : identity  (x → x).
    Backward : gradient scaling by −λ.

    This makes the feature extractor maximise the domain classifier loss
    while the domain classifier itself minimises it, driving representations
    toward subject invariance without requiring alternating optimisation.
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(torch.as_tensor(float(lambda_)))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (lambda_,) = ctx.saved_tensors
        return -lambda_.item() * grad_output, None


def grad_reverse(x, lambda_):
    return _GRL.apply(x, lambda_)

# ── MODEL COMPONENTS ─────────────────────────────────────────────────────────────

def _gn_groups(out_c: int) -> int:
    """
    Number of groups for GroupNorm: largest power-of-2 divisor of out_c, capped at 8.

    GroupNorm replaces BatchNorm in domain-adversarial TCNs because BatchNorm
    computes per-batch statistics that can re-encode subject identity even after
    the GRL has driven the activations toward invariance.  GroupNorm normalises
    within each sample independently of the batch composition, eliminating this
    leakage channel.

    The cap of 8 ensures at least 4 channels per group on the smallest branch
    (PHYS_D=32), keeping normalisation stable.
    """
    g = min(out_c, 8)
    while g > 1 and out_c % g != 0:
        g //= 2
    return max(g, 1)


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, d):
        super().__init__()
        # Causal convolution: pad (kernel_size-1)*dilation = 2*d timesteps on the
        # left only, so position t can only attend to positions [t-2d, t-d, t].
        causal_pad = 2 * d
        self.conv = nn.Sequential(
            nn.ConstantPad1d((causal_pad, 0), 0.0),
            nn.Conv1d(in_c, out_c, 3, padding=0, dilation=d),
            # GroupNorm instead of BatchNorm: per-sample normalisation avoids
            # encoding subject identity via batch statistics (DANN requirement).
            nn.GroupNorm(_gn_groups(out_c), out_c), nn.ReLU(), nn.Dropout1d(0.1),
        )
        self.res = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x): return self.conv(x) + self.res(x)


class TemporalAttention(nn.Module):
    """Additive attention pooling: (B, C, T) → (B, C)."""
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Linear(channels, channels // 2)
        self.score = nn.Linear(channels // 2, 1)

    def forward(self, x):
        xt      = x.permute(0, 2, 1)                       # (B, T, C)
        h       = torch.tanh(self.query(xt))               # (B, T, C//2)
        weights = torch.softmax(self.score(h), dim=1)      # (B, T, 1)
        return (xt * weights).sum(dim=1)                   # (B, C)


class CrossModalAttention(nn.Module):
    """
    Query-context cross-attention between modalities.

    A pooled summary (query, B×d_q) attends over the temporal sequence of
    the other branch (key/value, B×d_s×T), returning a context vector that
    captures which temporal patterns in one modality are most informative
    given the current state of the other.
    """
    def __init__(self, d_query, d_seq):
        super().__init__()
        d_attn      = max(d_seq // 2, 8)
        self.proj_q = nn.Linear(d_query, d_attn, bias=False)
        self.proj_s = nn.Linear(d_seq,   d_attn, bias=False)
        self.score  = nn.Linear(d_attn,  1,      bias=False)

    def forward(self, query, seq):
        seq_t  = seq.permute(0, 2, 1)                         # (B, T, d_s)
        q_proj = self.proj_q(query).unsqueeze(1)              # (B, 1, d_attn)
        s_proj = self.proj_s(seq_t)                           # (B, T, d_attn)
        attn   = torch.softmax(
            self.score(torch.tanh(q_proj + s_proj)), dim=1)  # (B, T, 1)
        return (seq_t * attn).sum(dim=1)                      # (B, d_s)


class ModalityGate(nn.Module):
    """
    Input-dependent modality gate  α ∈ (0, 1).

    α → 1 : physiology-dominant driver/window.
    α → 0 : kinematics-dominant driver/window.

    Computed dynamically from joint branch summaries, so the model can
    modulate its modality reliance within a session based on signal content.
    Exposed after every forward pass for interpretability.
    """
    def __init__(self, d_phys, d_kin):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(d_phys + d_kin, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, phys_pool, kin_pool):
        return self.gate_net(torch.cat([phys_pool, kin_pool], dim=-1))   # (B, 1)


class SubjectDiscriminator(nn.Module):
    """
    Subject identity classifier applied to the fusion vector via GRL.

    Trained to identify which participant a window comes from.
    Via GRL, the shared feature extractor is adversarially trained to
    produce representations the discriminator cannot distinguish.

    At test time the discriminator output is never used; only the feature
    extractor benefits from the adversarial training signal.
    """
    def __init__(self, d_in: int, n_subjects: int):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.net = nn.Sequential(
            sn(nn.Linear(d_in, 128)),
            nn.ReLU(),
            nn.Dropout(0.3),
            sn(nn.Linear(128, 64)),
            nn.ReLU(),
            nn.Dropout(0.3),
            sn(nn.Linear(64, n_subjects)),
        )

    def forward(self, fusion: torch.Tensor, lambda_: float) -> torch.Tensor:
        """Apply GRL then classify subject identity."""
        return self.net(grad_reverse(fusion, lambda_))



# ── MAIN MODEL ───────────────────────────────────────────────────────────────────

class DANNDualBranchTCN(nn.Module):
    """
    Domain-Adversarial Dual-Branch TCN.

    Identical dual-branch TCN to tcn_dual_branch.py (same receptive fields,
    cross-modal attention, modality gate), with two modifications:

    (a) Spectral injection — 18-d normalised FFT band powers concatenated
        to the 96-d fusion vector before the head (→ 114-d).
        18 = 6 signals (4 kin + 2 phys) × 3 frequency bands.

    (b) DANN branch — cat([96-d fusion, 18-d spec_adapted]) passed through
        SubjectDiscriminator via GRL during training, making both the fusion
        vector and spec_adapter weights subject-invariant (fix #21).
        Not used at inference.

    PHYS_D = 32  (physiology branch output channels)
    KIN_D  = 64  (kinematics branch output channels)
    Fusion = 96-d; Head input = 114-d (96 + 18 spectral).
    """
    PHYS_D = 32
    KIN_D  = 64

    def __init__(self, n_phys_feats: int, n_kin_feats: int):
        super().__init__()

        # Physiology branch — RF ≈ 107 ts (d=1,4,16,32), captures tonic arousal/HR lag.
        # d=32 is the maximum useful dilation for LOOKBACK_S=90: its causal_pad=64 fits
        # within the window (all 3 kernel positions land on real data at any timestep).
        # The previous d=64 had causal_pad=128 > 90, so most output timesteps were
        # computed from zero-padded context, wasting the final TCN block.
        self.phys_branch = nn.Sequential(
            ResBlock(n_phys_feats, self.PHYS_D,  1),
            ResBlock(self.PHYS_D,  self.PHYS_D,  4),
            ResBlock(self.PHYS_D,  self.PHYS_D, 16),
            ResBlock(self.PHYS_D,  self.PHYS_D, 32),
        )
        self.phys_attn = TemporalAttention(self.PHYS_D)

        # Kinematics branch — RF ≈ 95 ts (d=1,2,4,8,32), captures reactive deviations
        self.kin_branch = nn.Sequential(
            ResBlock(n_kin_feats,      self.KIN_D // 2,  1),
            ResBlock(self.KIN_D // 2,  self.KIN_D,       2),
            ResBlock(self.KIN_D,       self.KIN_D,        4),
            ResBlock(self.KIN_D,       self.KIN_D,        8),
            ResBlock(self.KIN_D,       self.KIN_D,       32),
        )
        self.kin_attn = TemporalAttention(self.KIN_D)

        # Cross-modal attention (bidirectional)
        self.phys_attends_kin = CrossModalAttention(self.PHYS_D, self.KIN_D)
        self.kin_attends_phys = CrossModalAttention(self.KIN_D,  self.PHYS_D)

        # Modality gate
        self.gate = ModalityGate(self.PHYS_D, self.KIN_D)

        # Spectral adapter: learned linear projection applied to x_spec before the
        # head.  Its parameters are trained adversarially via GRL (the discriminator
        # receives torch.cat([fusion, spec_adapted])), so they learn a
        # subject-invariant representation of the spectral features.  Without this,
        # raw precomputed spectral features bypass the GRL entirely — backprop cannot
        # reach numpy constants, so subject-specific frequency signatures would leak
        # into predictions untouched by the adversarial objective.
        #
        # Identity init: adapter starts as a pass-through so no spectral information
        # is discarded at epoch 0.  GRL gradients progressively push it away from
        # identity only where needed for subject invariance.  Random init risks
        # destroying discriminative spectral content in early epochs before the
        # adversarial objective stabilises, particularly on folds with noisy val AUC.
        self.spec_adapter = nn.Linear(SPECTRAL_DIM, SPECTRAL_DIM)
        nn.init.eye_(self.spec_adapter.weight)
        nn.init.zeros_(self.spec_adapter.bias)

        # Head: fusion(96) + spec_adapted(18) → 114 → 48 → 1
        fusion_dim = self.PHYS_D + self.KIN_D   # 96
        self.head = nn.Sequential(
            nn.Linear(fusion_dim + SPECTRAL_DIM, 48),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(48, 1),
        )

        # Auxiliary physiology head: phys_pool(32) → 16 → 1
        # Supervised directly on y — forces phys branch to stay discriminative.
        self.phys_head = nn.Sequential(
            nn.Linear(self.PHYS_D, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

        self._last_gate: torch.Tensor | None = None

    def _encode(self, x_phys, x_kin):
        """Shared encoder returning fusion vector, gate value, and phys_pool."""
        phys_seq  = self.phys_branch(x_phys.permute(0, 2, 1))   # (B, PHYS_D, T)
        kin_seq   = self.kin_branch(x_kin.permute(0, 2, 1))     # (B, KIN_D,  T)
        phys_pool = self.phys_attn(phys_seq)                     # (B, PHYS_D)
        kin_pool  = self.kin_attn(kin_seq)                       # (B, KIN_D)

        cross_phys_kin = self.phys_attends_kin(phys_pool, kin_seq)  # (B, KIN_D)
        cross_kin_phys = self.kin_attends_phys(kin_pool,  phys_seq) # (B, PHYS_D)

        phys_enh  = phys_pool + cross_kin_phys   # residual enhancement
        kin_enh   = kin_pool  + cross_phys_kin

        alpha = self.gate(phys_pool, kin_pool)   # (B, 1)
        self._last_gate = alpha.detach()

        # α weights the *direct* contribution of each branch to the fused vector.
        # Physiology also indirectly influences kin_enh via cross_phys_kin, and
        # kinematics influences phys_enh via cross_kin_phys, regardless of α.
        # Gate values should therefore be interpreted as "direct modality weight",
        # not as total modality influence on the prediction.
        fused = torch.cat([alpha * phys_enh,
                           (1.0 - alpha) * kin_enh], dim=-1)    # (B, 96)
        return fused, alpha, phys_pool

    def spec_encode(self, x_spec: torch.Tensor) -> torch.Tensor:
        """
        Apply the spectral adapter to x_spec.

        Exposed so train_dann_tcn can pass spec_adapted to the SubjectDiscriminator
        alongside the fusion vector, making the adapter's parameters subject to the
        GRL-reversed adversarial gradient.
        """
        return self.spec_adapter(x_spec)

    def forward(self, x_phys, x_kin, x_spec=None):
        """
        Parameters
        ----------
        x_phys : (B, T, n_phys_feats)
        x_kin  : (B, T, n_kin_feats)
        x_spec : (B, SPECTRAL_DIM) or None → zeros

        Returns
        -------
        logits      : (B,)
        fusion      : (B, 96)   — exposed for SubjectDiscriminator
        phys_logits : (B,)      — auxiliary physiology head output
        """
        fused, _, phys_pool = self._encode(x_phys, x_kin)
        if x_spec is None:
            x_spec = torch.zeros(fused.shape[0], SPECTRAL_DIM,
                                 dtype=fused.dtype, device=fused.device)
        # spec_adapter applies a learnable linear projection; its weights are
        # adversarially regularised via GRL in train_dann_tcn (fix #21).
        spec_adapted = self.spec_encode(x_spec)                    # (B, SPECTRAL_DIM)
        head_in = torch.cat([fused, spec_adapted], dim=-1)         # (B, 114)
        phys_logits = self.phys_head(phys_pool).squeeze(-1)        # (B,)
        return self.head(head_in).squeeze(-1), fused, phys_logits  # (B,), (B, 96), (B,)

    def gate_values(self, x_phys, x_kin, device=DEVICE):
        """Per-sample gate α without storing gradients."""
        self.eval()
        with torch.no_grad():
            self.forward(x_phys.to(device), x_kin.to(device))
        if self._last_gate is None:
            raise RuntimeError(
                "gate_values(): _last_gate is None — forward() did not execute. "
                "Ensure the input tensors are non-empty."
            )
        return self._last_gate.cpu().squeeze(-1).numpy()


class SingleBranchTCN(nn.Module):
    """
    Single-branch TCN for modality ablation.

    spectral_dim : number of per-branch spectral features to inject at the head
        (signals_in_branch × len(SPECTRAL_BANDS)).  Matches the spectral injection
        used by DANNDualBranchTCN so the ablation comparison is architecturally fair.
        Defaults to 0 (no spectral) for backwards compatibility.
    """
    def __init__(self, in_channels, spectral_dim: int = 0, branch: str = "kin"):
        super().__init__()
        self.spectral_dim = spectral_dim
        # Match the architecture of the corresponding branch in DANNDualBranchTCN
        # so the ablation comparison does not confound modality with capacity.
        PHYS_D = DANNDualBranchTCN.PHYS_D
        KIN_D  = DANNDualBranchTCN.KIN_D
        if branch == "phys":
            # Mirrors DANNDualBranchTCN.phys_branch: 4 ResBlocks, d=1,4,16,32
            self.network = nn.Sequential(
                ResBlock(in_channels, PHYS_D,  1),
                ResBlock(PHYS_D,      PHYS_D,  4),
                ResBlock(PHYS_D,      PHYS_D, 16),
                ResBlock(PHYS_D,      PHYS_D, 32),
            )
            out_d = PHYS_D
        else:
            # Mirrors DANNDualBranchTCN.kin_branch: 5 ResBlocks, d=1,2,4,8,32
            self.network = nn.Sequential(
                ResBlock(in_channels,  KIN_D // 2,  1),
                ResBlock(KIN_D // 2,   KIN_D,       2),
                ResBlock(KIN_D,        KIN_D,        4),
                ResBlock(KIN_D,        KIN_D,        8),
                ResBlock(KIN_D,        KIN_D,       32),
            )
            out_d = KIN_D
        self._out_d = out_d
        self.attention = TemporalAttention(out_d)
        head_in = out_d + spectral_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1),
        )

    def forward(self, x, x_spec=None):
        feats = self.attention(self.network(x.permute(0, 2, 1)))   # (B, _out_d)
        if x_spec is not None and self.spectral_dim > 0:
            feats = torch.cat([feats, x_spec], dim=-1)             # (B, _out_d + spectral_dim)
        return self.head(feats).squeeze(-1)

# ── UTILITIES ────────────────────────────────────────────────────────────────────

def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2: return None
    return roc_auc_score(y_true, y_score)


def safe_auprc(y_true, y_score):
    if len(np.unique(y_true)) < 2: return None
    return average_precision_score(y_true, y_score)


def _nan_or(x: float | None) -> float:
    """
    Return float('nan') if x is None, otherwise return x unchanged.

    Fix #37: replaces the pattern ``safe_auc(...) or float("nan")`` throughout
    the codebase.  The ``or`` operator treats 0.0 as falsy, so a valid AUC of
    exactly 0.0 (perfectly wrong predictions) or AUPRC of 0.0 (no true positives
    ranked above any negative) would be silently replaced by NaN.  Using an
    explicit None check avoids this latent bug without changing behaviour for
    any other return value.
    """
    return float("nan") if x is None else x


def focal_loss(logits, targets, gamma=2.0, pos_weight=None):
    pw  = (torch.tensor(pos_weight, dtype=logits.dtype, device=logits.device)
           if pos_weight is not None else None)
    bce = F.binary_cross_entropy_with_logits(logits, targets,
                                             pos_weight=pw, reduction="none")
    # pt = p for positives, 1-p for negatives (standard focal loss, Lin et al. 2017).
    # Using exp(-bce) is incorrect because it conflates the pos_weight-scaled BCE
    # with the true probability; the sigmoid-based formula is exact.
    with torch.no_grad():
        p   = torch.sigmoid(logits)
        pt  = p * targets + (1.0 - p) * (1.0 - targets)
    return ((1.0 - pt) ** gamma * bce).mean()


def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0; n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask   = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo)
        if not mask.any(): continue
        ece += mask.sum() / n * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def threshold_metrics(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_idx  = int(np.argmax(tpr - fpr))
    thresh = float(thresholds[j_idx])
    preds  = (y_prob >= thresh).astype(int)
    return (float(f1_score(y_true, preds, zero_division=0)),
            float(precision_score(y_true, preds, zero_division=0)),
            float(recall_score(y_true, preds, zero_division=0)),
            thresh)


def _val_threshold(y_val, val_scores):
    """
    Select Youden's J threshold on validation data for application to the test fold.

    Using this threshold for test-set evaluation avoids selecting the threshold
    on the test set itself (which inflates F1/Precision/Recall).  Called once per
    LOPO fold using the fold's validation scores; the returned threshold is then
    applied to that fold's held-out test predictions.

    Falls back to 0.5 when the validation fold is single-class (AUC undefined).
    """
    if len(np.unique(y_val)) < 2:
        return 0.5
    fpr, tpr, thrs = roc_curve(y_val, val_scores)
    j_idx = int(np.argmax(tpr - fpr))
    return float(thrs[j_idx])


def bootstrap_auc_ci_windows(y_true, y_score, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed); n = len(y_true); aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2: continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    if len(aucs) < 10:
        print(f"  [WARN] bootstrap_auc_ci_windows: only {len(aucs)} valid bootstrap samples "
              f"(need ≥10); returning NaN CI.")
        return float("nan"), float("nan")
    return tuple(np.percentile(aucs, [2.5, 97.5]))


def bootstrap_auc_ci_drivers(driver_aucs, n_boot=N_BOOTSTRAP, seed=SEED):
    aucs = np.array(driver_aucs); n = len(aucs)
    if n < 2:
        print(f"  [WARN] bootstrap_auc_ci_drivers: need ≥2 drivers (got {n}); returning NaN CI.")
        return float("nan"), float("nan")
    rng  = np.random.default_rng(seed)
    boot = [rng.choice(aucs, n, replace=True).mean() for _ in range(n_boot)]
    return tuple(np.percentile(boot, [2.5, 97.5]))


def _val_sample(arr, frac, rng):
    n = min(max(1, int(frac * len(arr))), len(arr)) if len(arr) > 0 else 0
    return rng.choice(arr, n, replace=False) if n > 0 else np.array([], dtype=arr.dtype)


def _signal_feat_cols(local_signal_idx, n_branch_signals, n_feat_types=5):
    """
    Feature column indices for a single raw signal in an engineered branch tensor.

    engineer_features concatenates [raw, diff1, roll_mean, roll_std, z_score]
    each of shape (T, C_branch).  In the resulting (T, 5·C) tensor:
        signal i → columns i, C+i, 2C+i, 3C+i, 4C+i

    n_feat_types must match the number of feature blocks in engineer_features().
    If engineer_features() is ever changed, update this constant accordingly.
    """
    if n_feat_types != 5:
        raise ValueError(
            "_signal_feat_cols: n_feat_types must equal the number of concatenated "
            "blocks in engineer_features() [raw, diff1, roll_mean, roll_std, z_score]. "
            f"Got {n_feat_types}."
        )
    if not (0 <= local_signal_idx < n_branch_signals):
        raise ValueError(
            f"_signal_feat_cols: local_signal_idx {local_signal_idx} out of range "
            f"[0, {n_branch_signals})."
        )
    C = n_branch_signals
    return [local_signal_idx + k * C for k in range(n_feat_types)]

# ── TRAINING ─────────────────────────────────────────────────────────────────────

def _cycle(loader):
    """Infinite iterator over a DataLoader; restarts when exhausted."""
    while True:
        yield from loader



def train_dann_tcn(model, discriminator,
                   Xtr_p, Xtr_k, Xtr_s, y_tr, subj_ids_tr,
                   Xval_p, Xval_k, Xval_s, y_val,
                   pos_weight=None):
    """
    DANN training for DANNDualBranchTCN.

    Jointly optimises:
        L = L_focal(y_pred, y)
          + ADV_WEIGHT · L_CE(subj_pred, subj_id)   # cross-training-subject

    One adversarial objective operates through the GRL:
    1. SubjectDiscriminator (multi-class): aligns features across training subjects.

    The test subject's windows are NOT used during training (inductive LOPO).
    Domain adaptation is purely across training subjects via the multi-class GRL.

    Lambda is annealed from 0 → LAMBDA_MAX using the standard DANN schedule.
    Validation AUC (classification only) is used for early stopping.

    Parameters
    ----------
    pos_weight : float or None
        Positive-class weight for focal_loss (neg_count / pos_count).
    """
    # Separate optimisers: discriminators train 3× faster so they present a
    # challenging adversary before the GRL starts reversing gradients.
    opt_model = torch.optim.Adam(model.parameters(),         lr=LR,       weight_decay=WEIGHT_DECAY)
    opt_disc  = torch.optim.Adam(discriminator.parameters(), lr=LR * 3.0, weight_decay=WEIGHT_DECAY)
    sched_model = torch.optim.lr_scheduler.CosineAnnealingLR(opt_model, T_max=EPOCHS, eta_min=LR * 0.01)
    sched_disc  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_disc,  T_max=EPOCHS, eta_min=LR * 0.03)
    loader  = get_dann_loader(Xtr_p, Xtr_k, Xtr_s, y_tr, subj_ids_tr, augment=True)
    best_auc, best_w = float("-inf"), None
    best_train_loss  = float("inf")   # fallback criterion when val is single-class
    no_improve = 0
    diagnostics = {"class_loss": [], "adv_loss": [], "lambda": []}

    for epoch in range(EPOCHS):
        p       = max(epoch / max(EPOCHS - 1, 1), LAMBDA_WARMUP_P)  # floor → λ non-zero from epoch 0
        lambda_ = LAMBDA_MAX * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)

        model.train(); discriminator.train()
        ep_cls, ep_adv, n_b = 0.0, 0.0, 0

        for xp, xk, xs, yb, sid in loader:
            xp, xk, xs = xp.to(DEVICE), xk.to(DEVICE), xs.to(DEVICE)
            yb, sid    = yb.to(DEVICE),  sid.to(DEVICE)

            logits, fusion, phys_logits = model(xp, xk, xs)
            cls_loss      = focal_loss(logits,      yb, pos_weight=pos_weight)
            aux_phys_loss = focal_loss(phys_logits, yb, pos_weight=pos_weight)

            # spec_adapted is always computed: needed for both discriminators
            # (fix #21: GRL flows back through spec_adapter via both paths).
            spec_adapted = model.spec_encode(xs)
            disc_in      = torch.cat([fusion, spec_adapted], dim=-1)   # (B, 114)

            # Multi-class subject adversarial loss (all training windows including
            # SMOTE synthetic ones).  Synthetic windows carry the anchor's subject
            # ID (assigned in main()), so every sample — real or synthetic —
            # contributes an adversarial gradient through the GRL.
            real_mask = sid >= 0  # always True after fix #30; guard kept for safety
            if real_mask.any():
                subj_logits = discriminator(disc_in[real_mask], lambda_)
                adv_loss    = F.cross_entropy(subj_logits, sid[real_mask])
            else:
                adv_loss = torch.zeros(1, device=logits.device, dtype=logits.dtype).squeeze()

            loss = (cls_loss
                    + AUX_PHYS_WEIGHT * aux_phys_loss
                    + ADV_WEIGHT      * adv_loss)
            opt_model.zero_grad()
            opt_disc.zero_grad()
            loss.backward()
            # Clip model and discriminator separately: the GRL already scales
            # gradients into the feature extractor by −λ.
            torch.nn.utils.clip_grad_norm_(model.parameters(),         1.0)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
            opt_model.step()
            opt_disc.step()

            ep_cls += cls_loss.item(); ep_adv += adv_loss.item(); n_b += 1

        sched_model.step()
        sched_disc.step()
        diagnostics["class_loss"].append(ep_cls / max(n_b, 1))
        diagnostics["adv_loss"].append(ep_adv   / max(n_b, 1))
        diagnostics["lambda"].append(lambda_)

        model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(
                model(torch.as_tensor(Xval_p).to(DEVICE),
                      torch.as_tensor(Xval_k).to(DEVICE),
                      torch.as_tensor(Xval_s).to(DEVICE))[0]
            ).cpu().numpy()
        auc = safe_auc(y_val, preds)
        if auc is not None:
            if auc > best_auc:
                best_auc = auc
                best_w   = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= DANN_PATIENCE:
                    break
        else:
            # Single-class val fold: AUC is undefined.
            # Fall back to training-loss patience so all folds receive consistent
            # regularisation via early stopping rather than always running to EPOCHS.
            train_loss = ep_cls / max(n_b, 1)
            if train_loss < best_train_loss - 1e-4:
                best_train_loss = train_loss
                best_w   = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= DANN_PATIENCE:
                    break

    if best_w is not None:
        model.load_state_dict(best_w)
    return model, diagnostics


def train_db_tcn(model, Xtr_p, Xtr_k, Xtr_s, y_tr, Xval_p, Xval_k, Xval_s, y_val,
                 pos_weight=None):
    """
    Population training for DANNDualBranchTCN *without* domain adversarial loss.
    Used as the within-script fair-comparison baseline (DB-TCN w/ spectral).

    Parameters
    ----------
    pos_weight : float or None
        Positive-class weight for focal_loss (neg_count / pos_count).
        Pass the same value used for LR/XGB baselines for a fair comparison.
    """
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS, eta_min=LR * 0.01)
    dummy_ids = np.zeros(len(y_tr), dtype=np.int64)
    loader    = get_dann_loader(Xtr_p, Xtr_k, Xtr_s, y_tr, dummy_ids, augment=True)
    best_auc, best_w = float("-inf"), None
    best_train_loss  = float("inf")   # fallback criterion when val is single-class
    no_improve = 0

    for _ in range(EPOCHS):
        model.train()
        ep_loss, n_b = 0.0, 0
        for xp, xk, xs, yb, _ in loader:
            yb_d = yb.to(DEVICE)
            logits, _, phys_logits = model(xp.to(DEVICE), xk.to(DEVICE), xs.to(DEVICE))
            loss = (focal_loss(logits,      yb_d, pos_weight=pos_weight)
                    + AUX_PHYS_WEIGHT * focal_loss(phys_logits, yb_d,
                                                   pos_weight=pos_weight))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item(); n_b += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(
                model(torch.as_tensor(Xval_p).to(DEVICE),
                      torch.as_tensor(Xval_k).to(DEVICE),
                      torch.as_tensor(Xval_s).to(DEVICE))[0]
            ).cpu().numpy()
        auc = safe_auc(y_val, preds)
        if auc is not None:
            if auc > best_auc:
                best_auc = auc; best_w = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    break
        else:
            # Single-class val fold: fall back to training-loss patience so all
            # folds receive consistent regularisation via early stopping.
            train_loss = ep_loss / max(n_b, 1)
            if train_loss < best_train_loss - 1e-4:
                best_train_loss = train_loss
                best_w   = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    break

    if best_w is not None:
        model.load_state_dict(best_w)
    return model


def train_single_tcn(model, Xtr_sc, y_tr, Xval_sc, y_val,
                     Xtr_spec=None, Xval_spec=None, pos_weight=None):
    """
    Train a SingleBranchTCN.

    Xtr_spec / Xval_spec : optional (N, spectral_dim) arrays of per-branch
        spectral features.  When provided they are forwarded to model(x, x_spec)
        so the single-branch ablation receives the same spectral injection as
        DANNDualBranchTCN, making the modality ablation architecturally fair.
    pos_weight : float or None
        Positive-class weight for focal_loss (neg_count / pos_count).
        Pass the same value used for DB-TCN / DANN-DB-TCN so the ablation
        comparison is fair when USE_SMOTE=False (class imbalance unhandled by
        SMOTE requires consistent focal-loss weighting across all models).
    """
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS, eta_min=LR * 0.01)
    y_int   = y_tr.astype(int)
    counts  = np.bincount(y_int, minlength=2)
    # Fix #34: mirror the get_dann_loader fix #29 — use WeightedRandomSampler only
    # when the data is imbalanced (ratio > 1.5×).  After SMOTE the training set is
    # roughly balanced; applying WeightedRandomSampler on balanced data draws with
    # replacement, giving each sample on average one occurrence per epoch with high
    # variance.  shuffle=True gives exactly one occurrence per epoch (lower variance,
    # no wasted gradient steps) and makes the ablation comparison fair vs DB-TCN /
    # DANN-DB-TCN which use get_dann_loader with the same ratio guard.
    ratio   = counts.max() / (counts.min() + 1e-6)

    use_spec = (Xtr_spec is not None) and (model.spectral_dim > 0)
    if use_spec:
        ds = TensorDataset(torch.as_tensor(Xtr_sc).float(),
                           torch.as_tensor(y_tr).float(),
                           torch.as_tensor(Xtr_spec).float())
    else:
        ds = TensorDataset(torch.as_tensor(Xtr_sc).float(),
                           torch.as_tensor(y_tr).float())
    if ratio > 1.5:
        weights = 1.0 / (counts + 1e-6)
        sw      = torch.from_numpy(weights[y_int])
        loader  = DataLoader(ds, batch_size=BATCH_SIZE, sampler=WeightedRandomSampler(sw, len(sw)))
    else:
        loader  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    Xval_spec_t = (torch.as_tensor(Xval_spec).float().to(DEVICE)
                   if (use_spec and Xval_spec is not None) else None)

    best_auc, best_w = float("-inf"), None
    best_train_loss  = float("inf")   # fallback criterion when val is single-class
    no_improve = 0

    for _ in range(EPOCHS):
        model.train()
        ep_loss, n_b = 0.0, 0
        if use_spec:
            for xb, yb, xsb in loader:
                xb  = xb.to(DEVICE)
                xsb = xsb.to(DEVICE)
                # Apply per-sample augmentation matching DANNDataset.__getitem__:
                # jitter is already per-element (randn_like); cutout must also be
                # applied independently per sample so each gets its own random
                # position rather than all 64 samples sharing one batch-level position.
                xb = xb + torch.randn_like(xb) * JITTER_STD
                if xb.shape[1] > CUTOUT_LEN:
                    for bi in range(xb.shape[0]):
                        if torch.rand(1).item() < CUTOUT_PROB:
                            t0 = torch.randint(0, xb.shape[1] - CUTOUT_LEN, (1,)).item()
                            xb[bi, t0: t0 + CUTOUT_LEN, :] = 0.0
                loss = focal_loss(model(xb, xsb), yb.to(DEVICE), pos_weight=pos_weight)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item(); n_b += 1
        else:
            for xb, yb in loader:
                xb = xb.to(DEVICE)
                xb = xb + torch.randn_like(xb) * JITTER_STD
                if xb.shape[1] > CUTOUT_LEN:
                    for bi in range(xb.shape[0]):
                        if torch.rand(1).item() < CUTOUT_PROB:
                            t0 = torch.randint(0, xb.shape[1] - CUTOUT_LEN, (1,)).item()
                            xb[bi, t0: t0 + CUTOUT_LEN, :] = 0.0
                loss = focal_loss(model(xb), yb.to(DEVICE), pos_weight=pos_weight)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item(); n_b += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            Xval_t = torch.as_tensor(Xval_sc).to(DEVICE)
            preds  = torch.sigmoid(model(Xval_t, Xval_spec_t)).cpu().numpy()
        auc = safe_auc(y_val, preds)
        if auc is not None:
            if auc > best_auc:
                best_auc = auc; best_w = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    break
        else:
            # Single-class val fold: fall back to training-loss patience so all
            # folds receive consistent regularisation via early stopping.
            train_loss = ep_loss / max(n_b, 1)
            if train_loss < best_train_loss - 1e-4:
                best_train_loss = train_loss
                best_w   = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    break

    if best_w is not None:
        model.load_state_dict(best_w)
    return model


def platt_adapt(pop_scores: np.ndarray, y_te: np.ndarray) -> np.ndarray:
    """
    Online Platt-scaling personalisation on the first GATE_ADAPT_K test windows.

    Fits a 2-parameter sigmoid  p = σ(w·s + b)  on support scores/labels,
    then applies it to ALL windows (support + eval).

    - Support set  : pop_scores[:GATE_ADAPT_K] / y_te[:GATE_ADAPT_K]
    - Calibrator   : fitted on support set only (no test-set leakage into fitting)
    - Returns full-length Platt-scaled score array (all windows on a single scale)

    The calibrator is applied to all population scores so that the full returned
    array uses a single consistent score scale.  The previous approach returned
    raw population scores for the support set and Platt-scaled scores for the
    eval set; mixing two different scales invalidates cross-window rankings for
    AUC even when w > 0 — monotonicity holds within each scale separately but
    not across them (fix #16).
    """
    K = GATE_ADAPT_K
    if len(y_te) <= K:
        return pop_scores

    s_sup, y_sup = pop_scores[:K], y_te[:K]

    if len(np.unique(y_sup)) < 2:
        # Not enough class diversity — fall back to population scores
        return pop_scores

    if int(y_sup.sum()) < MIN_SUPPORT_POSITIVES:
        return pop_scores   # too few positives — fall back to population scores

    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    try:
        lr.fit(s_sup.reshape(-1, 1), y_sup.astype(int))
    except Exception as exc:
        warnings.warn(  # fix #42: log degenerate support sets instead of silently swallowing
            f"platt_adapt: lr.fit() failed ({exc}); falling back to population scores",
            RuntimeWarning, stacklevel=2,
        )
        return pop_scores   # degenerate support set — fall back to population scores
    w = lr.coef_[0][0]
    # Guard against non-positive coefficient (fix #14).
    # With K=15 support samples and heavy class imbalance the calibrator can
    # fit w ≤ 0 (inverted scores).  A non-positive w inverts all scores, making
    # higher population scores map to lower Platt probabilities.
    if w <= 0:
        return pop_scores   # fall back to un-calibrated population scores
    # Guard against numerically blown-up coefficient (fix #41).
    # A very large |w| (>> 1) collapses calibrated probabilities to near-binary
    # {0, 1}, erasing score gradations and destabilising cross-window rankings.
    if w > MAX_PLATT_W:
        return pop_scores   # coefficient blow-up — fall back to population scores

    # Apply Platt scaling to ALL population scores (support + eval) so the
    # full-length array uses a single consistent scale (fix #16).
    return lr.predict_proba(pop_scores.reshape(-1, 1))[:, 1]


def head_adapt(model_pop, Xte_p, Xte_k, Xte_s, y_te,
               n_steps: int = GATE_ADAPT_STEPS, lr: float = 1e-3):
    """
    Online head fine-tuning personalisation on the first GATE_ADAPT_K test windows.

    Freezes everything except self.head; overfits the classification head to
    this driver's support set so it learns their specific decision boundary.

    - Support set  : y_te[:GATE_ADAPT_K]  — used for head fine-tuning
    - Evaluation   : y_te[GATE_ADAPT_K:]  — fine-tuned model scores only
    - Support scores come from the frozen population model (no leakage)
    - Returns full-length score array: [pop_support | head_adapted_eval]

    NOTE: the returned array concatenates scores from two distinct models
    (population for [:K], adapted for [K:]).  AUC must be computed only on
    the eval slice [K:] where both labels and scores come from the same model.
    The full array must NOT be used for pooled ranking across windows — the
    two model score scales may differ, invalidating cross-window ordering.
    """
    K = GATE_ADAPT_K
    if len(y_te) <= K:
        return None

    if int(y_te[:K].sum()) < MIN_SUPPORT_POSITIVES:
        return None

    model = copy.deepcopy(model_pop)
    model.to(DEVICE)

    # Freeze everything except the head
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("head.")

    Xp = torch.as_tensor(Xte_p[:K]).to(DEVICE)
    Xk = torch.as_tensor(Xte_k[:K]).to(DEVICE)
    Xs = torch.as_tensor(Xte_s[:K]).to(DEVICE)
    ys = torch.as_tensor(y_te[:K], dtype=torch.float32).to(DEVICE)

    pos_w = float(np.clip(
        (y_te[:K] == 0).sum() / max((y_te[:K] == 1).sum(), 1), 0.2, 5.0))

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-3)

    model.train()
    for _ in range(n_steps):
        opt.zero_grad()
        logits, _, _ = model(Xp, Xk, Xs)
        loss = focal_loss(logits, ys, pos_weight=pos_w)
        loss.backward()
        opt.step()

    # Support windows [:K]: use population model scores (model_pop was not adapted
    # so its scores are uncontaminated by the support labels).
    # Eval windows [K:]: use adapted model scores.
    # This avoids the scale-mixing issue flagged in fix #16 while also preventing
    # the adapted model (which was explicitly trained on y_te[:K]) from producing
    # tainted scores for those same windows.
    model_pop.eval()
    model.eval()
    with torch.no_grad():
        pop_support = torch.sigmoid(
            model_pop(torch.as_tensor(Xte_p[:K]).to(DEVICE),
                      torch.as_tensor(Xte_k[:K]).to(DEVICE),
                      torch.as_tensor(Xte_s[:K]).to(DEVICE))[0]
        ).cpu().numpy()
        adapted_eval = torch.sigmoid(
            model(torch.as_tensor(Xte_p[K:]).to(DEVICE),
                  torch.as_tensor(Xte_k[K:]).to(DEVICE),
                  torch.as_tensor(Xte_s[K:]).to(DEVICE))[0]
        ).cpu().numpy()

    return np.concatenate([pop_support, adapted_eval])


def gate_adapt(model_pop, Xte_p, Xte_k, Xte_s, y_te):
    """
    Online gate-only personalisation on the first GATE_ADAPT_K test windows.

    Only gate_net parameters are updated; all other weights are frozen.

    This is *online personalisation*, not a held-out generalisation estimate:
    - Support set  : y_te[:GATE_ADAPT_K]  — used for gate adaptation (labels seen)
    - Evaluation   : y_te[GATE_ADAPT_K:]  — gate-adapted model scores only
    - Support scores come from the frozen population model (no adaptation leakage
      into the evaluation split), but the support labels are used during training.

    Results should be labelled "GateAdapt (online)" in comparisons to population
    models which never see test labels.
    """
    if len(y_te) < GATE_ADAPT_K + MIN_EVAL_POSITIVES:
        return None, 0

    if len(np.unique(y_te[:GATE_ADAPT_K])) < 2:
        return None, 0

    if int(y_te[:GATE_ADAPT_K].sum()) < MIN_SUPPORT_POSITIVES:
        return None, 0

    model_adapt = copy.deepcopy(model_pop)
    for p in model_adapt.parameters():
        p.requires_grad_(False)
    for p in model_adapt.gate.gate_net.parameters():
        p.requires_grad_(True)

    X_sup_p = torch.as_tensor(Xte_p[:GATE_ADAPT_K]).to(DEVICE)
    X_sup_k = torch.as_tensor(Xte_k[:GATE_ADAPT_K]).to(DEVICE)
    X_sup_s = torch.as_tensor(Xte_s[:GATE_ADAPT_K]).to(DEVICE)
    y_sup   = torch.as_tensor(y_te[:GATE_ADAPT_K]).to(DEVICE)

    pos_w   = float(np.clip(
        (y_te[:GATE_ADAPT_K] == 0).sum() / max((y_te[:GATE_ADAPT_K] == 1).sum(), 1),
        0.2, 5.0))
    gate_params = [p for p in model_adapt.parameters() if p.requires_grad]
    opt_g = torch.optim.Adam(gate_params, lr=GATE_ADAPT_LR, weight_decay=1e-2)

    # Freeze BN running stats while adapting the gate.
    # model_adapt.train() re-enables gradient tracking for gate_net.
    # GroupNorm has no running statistics (per-sample), so no freeze needed.
    model_adapt.train()

    for _ in range(GATE_ADAPT_STEPS):
        logits, _, _ = model_adapt(X_sup_p, X_sup_k, X_sup_s)
        loss = focal_loss(logits, y_sup, pos_weight=pos_w)
        opt_g.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(gate_params, 1.0)
        opt_g.step()

    # Support windows [:K]: use population model scores (model_pop, unmodified).
    # Eval windows [K:]: use gate-adapted model scores.
    # Avoids the adapted model (trained on y_te[:K]) producing tainted scores
    # for those same windows while keeping a single score scale per region.
    model_pop.eval()
    model_adapt.eval()
    with torch.no_grad():
        pop_support = torch.sigmoid(
            model_pop(
                torch.as_tensor(Xte_p[:GATE_ADAPT_K]).to(DEVICE),
                torch.as_tensor(Xte_k[:GATE_ADAPT_K]).to(DEVICE),
                torch.as_tensor(Xte_s[:GATE_ADAPT_K]).to(DEVICE),
            )[0]
        ).cpu().numpy()
        adapted_eval = torch.sigmoid(
            model_adapt(
                torch.as_tensor(Xte_p[GATE_ADAPT_K:]).to(DEVICE),
                torch.as_tensor(Xte_k[GATE_ADAPT_K:]).to(DEVICE),
                torch.as_tensor(Xte_s[GATE_ADAPT_K:]).to(DEVICE),
            )[0]
        ).cpu().numpy()

    return np.concatenate([pop_support, adapted_eval]), GATE_ADAPT_K


def _adv_accuracy(model, discriminator, Xval_p, Xval_k, Xval_s, subj_ids_val):
    """
    Subject discriminator accuracy on held-in training windows after training.

    Measures convergence: a near-chance score indicates the GRL successfully
    confused the discriminator ON THE TRAINING DISTRIBUTION.  This is a
    convergence diagnostic only — see _adv_entropy_val for the held-out
    invariance measure on unseen (validation) subjects.

    Note: lambda_=0 so GRL is transparent (pure forward inference).
    """
    with torch.no_grad():
        Xval_s_t = torch.as_tensor(Xval_s).to(DEVICE)
        _, fusion, _ = model(
            torch.as_tensor(Xval_p).to(DEVICE),
            torch.as_tensor(Xval_k).to(DEVICE),
            Xval_s_t,
        )
        spec_adapted = model.spec_encode(Xval_s_t)
        disc_in      = torch.cat([fusion, spec_adapted], dim=-1)
        subj_logits  = discriminator.net(disc_in)   # bypass GRL — pure forward inference
        preds        = subj_logits.argmax(dim=-1).cpu().numpy()
    return float((preds == subj_ids_val).mean())


def _adv_entropy_val(model, discriminator, Xval_p, Xval_k, Xval_s):
    """
    Discriminator entropy on validation windows from subjects UNSEEN during training.

    Val subjects are not in the discriminator's training set.  If the GRL drives
    genuine cross-subject invariance, features from unseen subjects should be
    indistinguishable — the discriminator should output a near-uniform distribution
    (high entropy → max_confidence ≈ 1/N_subjects).

    Returns
    -------
    entropy      : mean per-sample entropy H(p) = -Σ p·log(p)  (nats)
    max_conf     : mean per-sample max-class probability
    max_entropy  : log(N_subjects) — the theoretical maximum (uniform distribution)
    """
    model.eval(); discriminator.eval()
    with torch.no_grad():
        Xval_s_t = torch.as_tensor(Xval_s).to(DEVICE)
        _, fusion, _ = model(
            torch.as_tensor(Xval_p).to(DEVICE),
            torch.as_tensor(Xval_k).to(DEVICE),
            Xval_s_t,
        )
        spec_adapted = model.spec_encode(Xval_s_t)
        disc_in      = torch.cat([fusion, spec_adapted], dim=-1)
        subj_logits  = discriminator.net(disc_in)       # bypass GRL
        probs        = F.softmax(subj_logits, dim=-1)   # (N, n_subj)
        entropy      = -(probs * (probs + 1e-10).log()).sum(-1).mean()
        max_conf     = probs.max(-1).values.mean()
    max_entropy = math.log(subj_logits.shape[-1])
    return float(entropy.cpu()), float(max_conf.cpu()), max_entropy

# ── VALIDITY REPORT ──────────────────────────────────────────────────────────────

def print_validity_report(X_raw, y, scores, pids):
    pos   = y == 1; neg = y == 0
    unique, cnts = np.unique(scores[pos].astype(int), return_counts=True)
    total_pos = pos.sum(); total = len(y)

    print(f"\n{'='*72}")
    print("LABEL VALIDITY REPORT — COMPOSITE RISK TARGET")
    print(f"{'='*72}")
    print(f"Total windows    : {total}")
    print(f"Positive (risk>0): {total_pos} ({100*total_pos/total:.1f}%)")
    print(f"Negative         : {neg.sum()} ({100*neg.sum()/total:.1f}%)")
    print(f"\nRisk score distribution (positives only):")
    for sc, cnt in zip(unique, cnts):
        print(f"  score={sc:2d}  n={cnt:5d}  ({100*cnt/total_pos:.1f}%)")
    print(f"\nPredictive validity — Mann-Whitney U (pos vs neg, raw signals):")
    print(f"  {'Signal':<24}  {'Mean(neg)':>10}  {'Mean(pos)':>10}  {'p-value':>12}  Sig")
    mw_pvals = {}
    for ci, col in enumerate(SIGNAL_COLS):
        neg_v = X_raw[neg][:, :, ci].mean(axis=1)
        pos_v = X_raw[pos][:, :, ci].mean(axis=1)
        _, pv = mannwhitneyu(neg_v, pos_v, alternative="two-sided")
        sig = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
        print(f"  {col:<24}  {neg_v.mean():>10.4f}  {pos_v.mean():>10.4f}  {pv:>12.2e}  {sig}")
        mw_pvals[col] = pv

    print(f"\nPer-driver positive rate:")
    print(f"  {'Driver':<12}  {'N':>6}  {'Pos':>5}  {'Rate%':>6}  Tier")
    for d in np.unique(pids):
        mask_d = pids == d; nd = mask_d.sum(); n_pos = y[mask_d].sum()
        rate   = 100.0 * n_pos / nd if nd > 0 else 0.0
        tier   = "HIGH" if rate >= 10 else ("MED" if rate >= 5 else "LOW")
        print(f"  {d:<12}  {nd:>6}  {n_pos:>5}  {rate:>6.1f}%  {tier}")
    print(f"{'='*72}")
    return mw_pvals

# ── MAIN ─────────────────────────────────────────────────────────────────────────

def _check_sampling_rate(df):
    """
    Validate that all sessions are sampled at approximately 1 Hz (fix #15).

    The spectral band boundaries (0.1, 0.3, 0.5 Hz) and FFT frequency
    resolution are computed assuming d=median_dt s in np.fft.rfftfreq.  If the
    actual median Δt deviates from 1 s the band edges shift in proportion,
    silently changing what physiological/kinematic phenomena each band captures.

    Raises RuntimeError if any session's median Δt is outside [0.8, 1.2] s.

    Returns
    -------
    float — global median Δt across all sessions (seconds), to be passed as
    ``fs=1/median_dt`` to compute_spectral_features_batch so FFT frequency bins
    are labelled correctly even when the actual sampling interval is not exactly 1 s.
    """
    if "Timestamp" not in df.columns:
        print("[WARN] _check_sampling_rate: no Timestamp column — skipping fs check.")
        return 1.0
    bad, all_dts = [], []
    for (pid, route), grp in df.groupby(["id", "route"]):
        ts = grp["Timestamp"].sort_values().values
        if len(ts) < 2:
            continue
        median_dt = float(np.median(np.diff(ts)))
        all_dts.append(median_dt)
        if not (0.8 <= median_dt <= 1.2):
            bad.append((pid, route, median_dt))
    if bad:
        lines = "\n".join(f"  pid={p}  route={r}  median_Δt={dt:.3f}s" for p, r, dt in bad)
        raise RuntimeError(
            f"Sampling rate check failed for {len(bad)} session(s) — "
            f"spectral band edges assume fs≈1 Hz (Δt≈1 s):\n{lines}"
        )
    global_median_dt = float(np.median(all_dts)) if all_dts else 1.0
    print(f"  Sampling rate check: {len(all_dts)} sessions, "
          f"median Δt = {global_median_dt:.3f} s  (all within [0.8, 1.2] s — OK)")
    return global_median_dt


def _loo_renorm(X_arr, pid_arr, routes_arr,
                all_session_stats, route_sum_mus, route_sum_sigs2, route_counts,
                ci_norm_map, label="windows"):
    """
    Apply leave-one-out route re-normalization to a window array.

    For each (pid, route) group: undoes the per-session z-score applied by
    normalize_signals (raw prefix stats stored in all_session_stats), then
    applies the LOO route z-score — the pooled σ of every other non-test
    training driver's session stats for the same route.

    route_sum_sigs2 must contain the sum of σ² (not σ) across drivers so that
    the LOO pooled σ = sqrt(mean(σ²_loo)) is computed correctly.  Using the
    arithmetic mean of σ underestimates the true population spread when session
    variances are heterogeneous (fix #17).

    Singleton routes (only one non-test driver, i.e. this driver itself) are
    skipped because LOO is undefined there; a per-fold warning count is printed
    so the user can detect if this case occurs frequently.

    Returns a re-normalized copy of X_arr.
    """
    X_out = X_arr.copy()
    n_singleton = 0
    for pid in np.unique(pid_arr):
        for route in np.unique(routes_arr[pid_arr == pid]):
            mask = (pid_arr == pid) & (routes_arr == route)
            for col, ci in ci_norm_map.items():
                key_own = (pid, route, col)
                key_rte = (route, col)
                if key_own not in all_session_stats or key_rte not in route_sum_mus:
                    continue
                own_mu, own_sig_raw = all_session_stats[key_own]   # raw σ, no epsilon
                n_rte = route_counts[key_rte]
                if n_rte < 2:
                    # Singleton route: LOO undefined, per-session z-score retained.
                    n_singleton += 1
                    continue
                rte_mu  = (route_sum_mus[key_rte]  - own_mu)       / (n_rte - 1)
                # Pooled LOO σ: sqrt(mean(σ²)) avoids underestimating spread when
                # session variances are heterogeneous (fix #17).
                # route_sum_sigs2 accumulates raw σ² (without epsilon) so the
                # pooled estimate is not inflated by the per-sample epsilon (fix #4).
                rte_sig = math.sqrt(
                    max((route_sum_sigs2[key_rte] - own_sig_raw ** 2) / (n_rte - 1), 0.0)
                )
                # Undo per-session z-score: multiply by (σ+ε) because normalize_signals
                # divided by (σ+ε).  Apply LOO route z-score with floor (fix #23).
                X_out[mask, :, ci] = (
                    X_out[mask, :, ci] * (own_sig_raw + 1e-6) + own_mu - rte_mu
                ) / max(rte_sig, 1e-6)
    if n_singleton:
        print(f"  [WARN] _loo_renorm ({label}): {n_singleton} singleton-route "
              f"(driver, route, col) triplets skipped — LOO undefined, "
              f"per-session z-score retained.")
    return X_out


def main():
    df = pd.read_csv(Path(__file__).parent / "relab+unibo_dataset.csv")
    global_sample_dt = _check_sampling_rate(df)   # median Δt in seconds; used for FFT bin labels
    df = mark_event_onsets(df)

    # Pre-compute per-session normalization statistics from the RAW dataframe.
    # These are used inside each LOPO fold to re-normalize the test-driver's
    # windows using training-fold route statistics instead of the test driver's
    # own session statistics, eliminating transductive normalization leakage.
    COLS_TO_NORM = VEHICLE_COLS + [c for c in ["arousal", "hr"] if c in df.columns]
    NORM_PREFIX  = LOOKBACK_S + GAP   # rows used by normalize_signals
    all_session_stats: dict = {}       # {(pid, route, col): (mu, sig)}
    for (pid, route), grp in df.groupby(["id", "route"]):
        for col in COLS_TO_NORM:
            if col in grp.columns:
                mu      = float(grp.iloc[:NORM_PREFIX][col].mean())
                sig_raw = float(grp.iloc[:NORM_PREFIX][col].std())  # raw σ, no epsilon
                all_session_stats[(pid, route, col)] = (mu, sig_raw)

    df = normalize_signals(df)

    # Uniform-stride windows (WINDOW_STEP); LOPO splits by driver.
    # SMOTE + class-weighted focal loss handle class imbalance (fix #18).
    (X_raw_te, y_te_all, scores_te,
     pid_te_all, etypes_te, routes_te_all, ts_te_all) = build_windows(df)
    X_raw_tr, y_tr_all, pid_tr_all, routes_tr_all = (
        X_raw_te, y_te_all, pid_te_all, routes_te_all)

    n_phys_feat  = len(PHYS_COLS) * 5    # 10
    n_kin_feat   = len(KIN_COLS)  * 5    # 20
    n_all_feat   = len(SIGNAL_COLS) * 5  # 30
    fusion_dim   = DANNDualBranchTCN.PHYS_D + DANNDualBranchTCN.KIN_D   # 96
    head_dim     = fusion_dim + SPECTRAL_DIM                             # 114

    mw_pvals = print_validity_report(X_raw_te, y_te_all, scores_te, pid_te_all)

    print(f"\n{'='*72}")
    print("DANN-DB-TCN — PIPELINE CONFIGURATION")
    print(f"{'='*72}")
    print(f"Signals           : {SIGNAL_COLS}")
    print(f"Physiology branch : {PHYS_COLS}  ({n_phys_feat} engineered features)")
    print(f"  TCN blocks      : d=1,4,16,32  →  RF ≈ 107 timesteps")
    print(f"  Output channels : {DANNDualBranchTCN.PHYS_D}")
    print(f"Kinematics branch : {KIN_COLS}  ({n_kin_feat} engineered features)")
    print(f"  TCN blocks      : d=1,2,4,8,32  →  RF ≈ 95 timesteps")
    print(f"  Output channels : {DANNDualBranchTCN.KIN_D}")
    print(f"CLC excluded      : center_line_crossing removed from SEVERITY")
    print(f"SMOTE             : {'minority oversampling on training fold (k=' + str(SMOTE_K_NEIGHBORS) + ')' if USE_SMOTE else 'DISABLED — class-weighted focal loss only'}")
    print(f"Min eval positives: {MIN_EVAL_POSITIVES} (drivers below threshold excluded from eval)")
    print(f"Cross-modal attn  : bidirectional (phys↔kin) with residual enhancement")
    print(f"Modality gate     : α = σ(MLP([phys_pool, kin_pool]))  ∈ (0,1)")
    print(f"Fusion            : concat(α·phys_enh, (1−α)·kin_enh) → {fusion_dim}-d")
    print(f"Spectral features : {len(SIGNAL_COLS)} signals × {len(SPECTRAL_BANDS)} bands = {SPECTRAL_DIM}-d")
    print(f"  Bands (Hz)      : [0.00,0.10)  [0.10,0.30)  [0.30,0.50]")
    print(f"  Normalisation   : relative band power (÷ total window power)")
    print(f"Head input        : {fusion_dim} (fusion) + {SPECTRAL_DIM} (spectral) = {head_dim}-d")
    print(f"DANN              : GRL + SubjectDiscriminator ({fusion_dim + SPECTRAL_DIM}-d → 128 → 64 → N_subj)  (fusion+spec_adapted)")
    print(f"  λ schedule      : 0 → {LAMBDA_MAX}  (annealed, Ganin et al. schedule)")
    print(f"  Adversarial wt  : {ADV_WEIGHT}  (total loss = focal + {ADV_WEIGHT}·CE_subj)")
    print(f"Normalisation     : vehicle + HR + arousal z-scored per session (all relative to intra-session baseline)")
    print(f"  Spectral scaler : StandardScaler per feature (fit on real training windows only; "
          f"transforms real + SMOTE synthetic)")
    print(f"Aux phys loss     : {AUX_PHYS_WEIGHT}×focal(phys_head, y)  — forces phys branch to stay discriminative")
    print(f"Loss              : focal (γ=2) + {AUX_PHYS_WEIGHT}×focal(phys_head) + {ADV_WEIGHT}×CE(subject)")
    print(f"Epochs            : {EPOCHS}  |  LR : {LR}  |  Scheduler : CosineAnnealingLR")
    print(f"GAP / HORIZON     : {GAP}s / {HORIZON}s  →  predicts errors in [{GAP}, {GAP+HORIZON}]s")
    print(f"Gate adapt (pers) : {GATE_ADAPT_K} support windows  |  {GATE_ADAPT_STEPS} steps  |  gate_net only")
    print(f"Bootstrap CI      : {N_BOOTSTRAP} resamples (window-level pooled; driver-level summary)")
    print(f"Device            : {DEVICE}")
    print(f"{'='*72}\n")

    drivers  = [d for d in np.unique(pid_te_all)
                if y_te_all[pid_te_all == d].sum() >= MIN_EVAL_POSITIVES]

    hdr = (f"{'Driver':<10} | {'N_win':>5} {'PosR%':>6} | "
           f"{'LR':>7} {'XGB':>7} {'DB-TCN':>7} | "
           f"{'DANN-Pop':>8} {'GateAdpt†':>9} {'Platt†':>8} {'HeadFT†':>8} {'Gain':>6} | "
           f"{'EqBlend':>7} | {'gate(α)':>7}")
    print(hdr)
    print("-" * len(hdr))
    print(f"  † online personalisation: first {GATE_ADAPT_K} test-participant labels observed "
          f"during adaptation.  NOT comparable to population models (no †).")

    per_driver_results = []
    pool_y, pool_db, pool_dann, pool_ga, pool_blend = [], [], [], [], []
    pool_lr, pool_xgb = [], []
    pool_Xte_p, pool_Xte_k, pool_Xte_s = [], [], []
    pool_models_dann  = []
    pool_etypes, pool_gates, pool_gate_arrays = [], [], []
    dann_diagnostics  = []
    # Per-fold Youden's J thresholds: each fold's val predictions are used to
    # select the threshold applied exclusively to that fold's test predictions.
    # Pooling scores from different fold-specific model instances and computing
    # one global Youden threshold is invalid because scores from different models
    # live on potentially different marginal distributions — the resulting
    # threshold may not correspond to a meaningful operating point for any
    # individual test fold.
    pool_thresh_db    = []
    pool_thresh_dann  = []
    pool_thresh_blend = []

    ABLATION      = {"Physiology": PHYS_IDX, "Kinematics": KIN_IDX}
    pool_ablation = {k: [] for k in ABLATION}
    singleclass_val_folds: list = []   # folds where val set was single-class

    for d in drivers:
        mask_tr  = pid_tr_all != d
        X_tr     = X_raw_tr[mask_tr];     y_tr      = y_tr_all[mask_tr]
        pid_tr   = pid_tr_all[mask_tr];   routes_tr = routes_tr_all[mask_tr]
        ts_tr    = ts_te_all[mask_tr]

        mask_te  = pid_te_all == d
        X_te     = X_raw_te[mask_te]; y_te = y_te_all[mask_te]
        ts_te    = ts_te_all[mask_te]; routes_te = routes_te_all[mask_te]
        etypes_d = etypes_te[mask_te]

        # Sort by timestamp only — not by route then timestamp — so that the
        # first GATE_ADAPT_K windows used by online personalisation reflect
        # genuine chronological order across all routes rather than being
        # artificially concentrated in the lexicographically first route.
        order    = np.argsort(ts_te)
        X_te     = X_te[order]; y_te = y_te[order]; etypes_d = etypes_d[order]
        routes_te = routes_te[order]   # keep in sync with X_te after sorting

        # Compute fold seed, rng, and val_ids BEFORE building the LOO stats so
        # that val drivers can be excluded from the normalization reference used
        # for training/val windows.  Val drivers are excluded from model training
        # via vmask, so their session stats must not influence the route mean that
        # training-fold windows are normalized against.
        seed_d   = int(hashlib.md5(str(d).encode()).hexdigest(), 16) & 0xFFFFFFFF
        fold_rng = np.random.default_rng(SEED ^ seed_d)
        # Reset PyTorch RNG per fold so each fold is independently reproducible
        # regardless of which folds were executed before it.
        torch.manual_seed(SEED ^ seed_d)
        torch.cuda.manual_seed_all(SEED ^ seed_d)
        train_drivers = np.unique(pid_tr)
        has_pos  = np.array([y_tr[pid_tr == p].sum() > 0 for p in train_drivers])
        # 0.25 instead of 0.20: raises val fold from ~4 to ~5 drivers, reducing
        # early-stopping variance from overly small validation sets (fix #19).
        val_ids  = np.concatenate([
            _val_sample(train_drivers[has_pos],  0.25, fold_rng),
            _val_sample(train_drivers[~has_pos], 0.25, fold_rng),
        ])
        val_ids_set = set(val_ids.tolist())
        vmask = np.isin(pid_tr, val_ids)

        # ── Re-normalize test windows with training-fold route statistics ─────────
        # normalize_signals() used each driver's own session stats (transductive).
        # Undo the test driver's per-session z-score and apply the route-level mean
        # statistics from training drivers so the test driver is treated as truly
        # unseen at inference time.
        _route_mus_d:  dict = {}
        _route_sigs_d: dict = {}
        # _route_sum_mus/_sigs2/_counts: LOO reference for training/val window
        # renormalization.  Val drivers are excluded so their session characteristics
        # do not shift the normalization of training windows (val windows are excluded
        # from model training via vmask, but including their stats would still
        # introduce an indirect path from val data into training features).
        # Two separate reference pools with different exclusion rules:
        #
        # 1. Test-driver reference (_route_mus_d/_route_sigs_d):
        #    Includes ALL non-test drivers (val drivers included).
        #    Val drivers see the test driver's route more reliably with a larger
        #    pool (~29 drivers); the additional variance from the smaller pool
        #    outweighs the minor distribution asymmetry that fix #28 addressed.
        #
        # 2. Training LOO reference (_route_sum_mus/_sigs2/_counts):
        #    Excludes val drivers so their session characteristics cannot shift
        #    the normalization of training windows (val data must not influence
        #    the model's training distribution in any form).
        _route_sum_mus:   dict = {}
        _route_sum_sigs2: dict = {}   # sum of σ² per (route, col) for training LOO — val-excluded
        _route_counts:    dict = {}
        for (pid_s, route_s, col_s), (mu_s, sig_s) in all_session_stats.items():
            if pid_s == d:
                continue  # exclude test driver from both references
            key = (route_s, col_s)
            # Test reference: all non-test drivers (val included) for a larger,
            # lower-variance normalization pool.
            _route_mus_d.setdefault(key, []).append(mu_s)
            _route_sigs_d.setdefault(key, []).append(sig_s)
            # Training LOO reference: exclude val drivers to prevent any indirect
            # path from val data into training window normalization.
            if pid_s not in val_ids_set:
                _route_sum_mus[key]   = _route_sum_mus.get(key,   0.0) + mu_s
                _route_sum_sigs2[key] = _route_sum_sigs2.get(key, 0.0) + sig_s ** 2
                _route_counts[key]    = _route_counts.get(key,    0)   + 1
        # Pure-training reference (non-test, non-val): used for val window renorm
        # so val drivers are not in their own normalization reference.
        _route_pure_train_stats_d: dict = {
            k: (
                _route_sum_mus[k] / _route_counts[k],
                math.sqrt(max(_route_sum_sigs2[k] / _route_counts[k], 0.0))
            )
            for k in _route_sum_mus if _route_counts.get(k, 0) >= 1
        }
        # Route means for test-driver renormalisation — all non-test drivers
        # (including val), giving the largest possible reference pool.
        # Uses pooled σ = sqrt(mean(σ²)) consistent with _loo_renorm (fix #17).
        _route_tr_stats_d: dict = {
            k: (float(np.mean(_route_mus_d[k])),
                float(np.sqrt(np.mean(np.square(_route_sigs_d[k])))))
            for k in _route_mus_d
        }
        ci_norm_map = {col: SIGNAL_COLS.index(col)
                       for col in COLS_TO_NORM if col in SIGNAL_COLS}
        X_te = X_te.copy()
        for route_v in np.unique(routes_te):
            r_mask = routes_te == route_v
            for col_v, ci_v in ci_norm_map.items():
                key_test  = (d, route_v, col_v)
                key_route = (route_v, col_v)
                if key_test not in all_session_stats or key_route not in _route_tr_stats_d:
                    continue
                test_mu_v, test_sig_raw_v = all_session_stats[key_test]   # raw σ, no ε
                tr_mu_v,   tr_sig_v       = _route_tr_stats_d[key_route]
                # Undo test-driver z-score: multiply by (σ+ε) matching normalize_signals.
                # Apply LOO route z-score with floor.
                X_te[r_mask, :, ci_v] = (
                    X_te[r_mask, :, ci_v] * (test_sig_raw_v + 1e-6) + test_mu_v - tr_mu_v
                ) / max(tr_sig_v, 1e-6)

        # ── Re-normalise training windows with LOO route statistics ───────────────
        # normalize_signals() z-scored each driver by their own session stats.
        # To avoid circular dependency (fix #12), each training driver p_tr is
        # re-normalised using the route mean computed from all OTHER training
        # drivers (leave-one-out), so no driver is normalised by a mean that
        # includes its own contribution.  Singleton routes emit a warning.
        X_tr = _loo_renorm(X_tr, pid_tr, routes_tr,
                           all_session_stats, _route_sum_mus, _route_sum_sigs2,
                           _route_counts, ci_norm_map, f"train (test={d})")

        # Build val windows directly from the globally normalized df for val-driver
        # sessions only, using uniform stride (fine_stride=False).
        # Previous approach took val windows from X_raw_te, the official test pool:
        # each driver's test windows were used as val data in ~20% of other folds'
        # early stopping, creating cross-fold contamination.  Building from df
        # completely separates val and test window pools.
        df_val_fold = df[df["id"].isin(val_ids_set)].copy()
        X_val_raw, y_val_d, _, _pids_val_arr, _, _routes_val_arr, _ = \
            build_windows(df_val_fold)
        # Re-normalize val windows with training-driver route statistics (fix #13).
        # Val drivers are excluded from _route_sum_mus/_route_sum_sigs2/_route_counts
        # (fix #28), so they were never added to those sums.  Calling _loo_renorm on
        # them would compute (sum_train - own_mu_val) / (n_train - 1), incorrectly
        # subtracting a value that was never added and under-counting the pool by one.
        # The correct reference for val windows is _route_pure_train_stats_d —
        # the mean and pooled-σ of the pure-training drivers (non-test, non-val),
        # so val drivers are not in their own normalization reference pool.
        X_val_raw = X_val_raw.copy()
        for _vpid in np.unique(_pids_val_arr):
            for _vrt in np.unique(_routes_val_arr[_pids_val_arr == _vpid]):
                _vmask = (_pids_val_arr == _vpid) & (_routes_val_arr == _vrt)
                for _vcol, _vci in ci_norm_map.items():
                    _key_own = (_vpid, _vrt, _vcol)
                    _key_rte = (_vrt, _vcol)
                    if _key_own not in all_session_stats or _key_rte not in _route_pure_train_stats_d:
                        continue
                    _vmu, _vsig_raw = all_session_stats[_key_own]
                    _tr_mu, _tr_sig = _route_pure_train_stats_d[_key_rte]
                    X_val_raw[_vmask, :, _vci] = (
                        X_val_raw[_vmask, :, _vci] * (_vsig_raw + 1e-6) + _vmu - _tr_mu
                    ) / max(_tr_sig, 1e-6)

        if len(np.unique(y_val_d)) < 2:
            print(f"{d:<10} | [WARN] val fold single-class — "
                  f"AUC-based early stopping unavailable; falling back to "
                  f"training-loss patience (patience={DANN_PATIENCE})")
            singleclass_val_folds.append(d)

        # ── Uniform-stride training windows for LR/XGB baselines ─────────────────
        # Use the same val-excluded split as DANN so all models train on identical
        # data — necessary for a fair comparison.  X_tr is already LOO-renormalized.
        X_bl_raw = X_tr[~vmask]
        y_bl     = y_tr[~vmask]

        X_tr_train_raw  = X_tr[~vmask]   # raw training windows (pre-SMOTE)
        y_tr_train      = y_tr[~vmask]
        y_tr_orig       = y_tr_train.copy()
        pid_tr_train    = pid_tr[~vmask]
        routes_tr_train = routes_tr[~vmask]
        ts_tr_train     = ts_tr[~vmask]

        # === SMOTE: k-NN search in raw signal space (better distance geometry),
        # synthetic features generated by λ-weighted interpolation in *scaled
        # feature space* (avoids nonlinear artifacts when roll_std / z-score are
        # applied to the interpolated raw signal). ===
        if USE_SMOTE:
            smote_rng  = np.random.default_rng(SEED ^ seed_d ^ SMOTE_SEED_SALT)
            # Equal phys/kin branch weight: scale phys channels by sqrt(n_kin/n_phys)
            # so both branches contribute equally to the PCA-based NN distance.
            _smote_cw = np.ones(len(SIGNAL_COLS), dtype=np.float32)
            _smote_cw[list(PHYS_IDX)] = math.sqrt(len(KIN_IDX) / len(PHYS_IDX))
            # Tile over timesteps: flattened (T,C) layout is row-major, channel at t*C+c
            _smote_chan_weights = np.tile(_smote_cw, LOOKBACK_S)   # shape (T*C,) = (540,)
            _, y_tr_train, _smote_anchors, _smote_neighbors, _smote_lambdas = smote_raw(
                X_tr_train_raw, y_tr_orig, rng=smote_rng,
                pids=pid_tr_train, routes=routes_tr_train, t_starts=ts_tr_train,
                channel_weights=_smote_chan_weights)
        else:
            _smote_anchors   = np.empty(0, dtype=np.int64)
            _smote_neighbors = np.empty(0, dtype=np.int64)
            _smote_lambdas   = np.empty(0, dtype=np.float32)
            y_tr_train       = y_tr_orig
        n_real      = len(y_tr_orig)
        n_synthetic = len(_smote_anchors)

        # ── Feature engineering on real windows only ─────────────────────────────
        Xtr_p_feat_real = apply_features_branch(X_tr_train_raw, PHYS_IDX)
        Xval_p_feat     = apply_features_branch(X_val_raw,      PHYS_IDX)
        Xte_p_feat      = apply_features_branch(X_te,           PHYS_IDX)

        Xtr_k_feat_real = apply_features_branch(X_tr_train_raw, KIN_IDX)
        Xval_k_feat     = apply_features_branch(X_val_raw,      KIN_IDX)
        Xte_k_feat      = apply_features_branch(X_te,           KIN_IDX)

        # Fit scalers on real training windows; val/test use the same scaler.
        scaler_p = StandardScaler()
        scaler_p.fit(Xtr_p_feat_real.reshape(-1, n_phys_feat))
        Xtr_p_real_sc = scaler_p.transform(
            Xtr_p_feat_real.reshape(-1, n_phys_feat)).reshape(-1, LOOKBACK_S, n_phys_feat)
        Xval_p_sc = scaler_p.transform(
            Xval_p_feat.reshape(-1, n_phys_feat)).reshape(-1, LOOKBACK_S, n_phys_feat)
        Xte_p_sc  = scaler_p.transform(
            Xte_p_feat.reshape(-1, n_phys_feat)).reshape(-1, LOOKBACK_S, n_phys_feat)

        scaler_k = StandardScaler()
        scaler_k.fit(Xtr_k_feat_real.reshape(-1, n_kin_feat))
        Xtr_k_real_sc = scaler_k.transform(
            Xtr_k_feat_real.reshape(-1, n_kin_feat)).reshape(-1, LOOKBACK_S, n_kin_feat)
        Xval_k_sc = scaler_k.transform(
            Xval_k_feat.reshape(-1, n_kin_feat)).reshape(-1, LOOKBACK_S, n_kin_feat)
        Xte_k_sc  = scaler_k.transform(
            Xte_k_feat.reshape(-1, n_kin_feat)).reshape(-1, LOOKBACK_S, n_kin_feat)

        # Synthetic features: λ-weighted interpolation in scaled feature space.
        # (1-λ)·feat(anchor) + λ·feat(neighbour) is a convex combination of two
        # real, scaled feature vectors — it stays on the feature manifold without
        # the nonlinear distortion that occurs when roll_std / z-score are computed
        # from a raw-signal interpolation.
        if n_synthetic > 0:
            lams_bc = _smote_lambdas[:, None, None]   # (n_syn, 1, 1) broadcasts over T, feat
            Xtr_p_syn_sc = ((1.0 - lams_bc) * Xtr_p_real_sc[_smote_anchors]
                            + lams_bc        * Xtr_p_real_sc[_smote_neighbors])
            Xtr_k_syn_sc = ((1.0 - lams_bc) * Xtr_k_real_sc[_smote_anchors]
                            + lams_bc        * Xtr_k_real_sc[_smote_neighbors])
            Xtr_p_sc = np.concatenate([Xtr_p_real_sc, Xtr_p_syn_sc.astype(np.float32)])
            Xtr_k_sc = np.concatenate([Xtr_k_real_sc, Xtr_k_syn_sc.astype(np.float32)])
        else:
            Xtr_p_sc = Xtr_p_real_sc
            Xtr_k_sc = Xtr_k_real_sc

        # ── Spectral features ─────────────────────────────────────────────────────
        # Real windows: computed from pre-SMOTE raw signals.
        # Synthetic spectral features: λ-weighted interpolation in scaled spectral
        # space, consistent with the temporal branch treatment above.  This avoids
        # the nonlinear band-power distortion that occurs when computing spectral
        # features from a raw-signal interpolation.
        Xtr_sk_raw_real = compute_spectral_features_batch(X_tr_train_raw[:, :, KIN_IDX],  sample_dt=global_sample_dt)
        Xtr_sp_raw_real = compute_spectral_features_batch(X_tr_train_raw[:, :, PHYS_IDX], sample_dt=global_sample_dt)
        Xval_sk_raw = compute_spectral_features_batch(X_val_raw[:, :, KIN_IDX],  sample_dt=global_sample_dt)
        Xte_sk_raw  = compute_spectral_features_batch(X_te[:, :, KIN_IDX],       sample_dt=global_sample_dt)
        Xval_sp_raw = compute_spectral_features_batch(X_val_raw[:, :, PHYS_IDX], sample_dt=global_sample_dt)
        Xte_sp_raw  = compute_spectral_features_batch(X_te[:, :, PHYS_IDX],      sample_dt=global_sample_dt)

        # Interleave kin+phys per band into a single (N, SPECTRAL_DIM=18) vector.
        # Final layout per row (6 signals × 3 bands = 18 dims):
        #   [ steer|torq|acc|spd|aro|hr ]_band0
        #   [ steer|torq|acc|spd|aro|hr ]_band1
        #   [ steer|torq|acc|spd|aro|hr ]_band2
        # where each 6-element block is [kin_b(i) (4 cols) | phys_b(i) (2 cols)].
        # The per-band StandardScaler (below) is applied to each block independently.
        _n_kin  = len(KIN_COLS)
        _n_phys = len(PHYS_COLS)
        _n_spec = _n_kin + _n_phys

        def _concat_spec(sk, sp):
            # sk: (N, n_kin * n_bands)  — from compute_spectral_features_batch on KIN_IDX
            # sp: (N, n_phys * n_bands) — from compute_spectral_features_batch on PHYS_IDX
            # Returns (N, n_spec * n_bands) with kin and phys interleaved within each band.
            return np.hstack([
                np.hstack([sk[:, b*_n_kin:(b+1)*_n_kin],
                           sp[:, b*_n_phys:(b+1)*_n_phys]])
                for b in range(len(SPECTRAL_BANDS))
            ])

        Xtr_s_raw_real = _concat_spec(Xtr_sk_raw_real, Xtr_sp_raw_real)
        Xval_s_raw     = _concat_spec(Xval_sk_raw,     Xval_sp_raw)
        Xte_s_raw      = _concat_spec(Xte_sk_raw,      Xte_sp_raw)

        # Fit spectral scaler on real training windows only.
        scaler_s = StandardScaler()
        scaler_s.fit(Xtr_s_raw_real)
        Xtr_s_real_sc = scaler_s.transform(Xtr_s_raw_real).astype(np.float32)
        Xval_s_sc     = scaler_s.transform(Xval_s_raw).astype(np.float32)
        Xte_s_sc      = scaler_s.transform(Xte_s_raw).astype(np.float32)

        # Synthetic spectral features via λ-weighted interpolation in scaled space.
        if n_synthetic > 0:
            lams_s = _smote_lambdas[:, None]   # (n_syn, 1) broadcasts over SPECTRAL_DIM
            Xtr_s_syn_sc = ((1.0 - lams_s) * Xtr_s_real_sc[_smote_anchors]
                            + lams_s        * Xtr_s_real_sc[_smote_neighbors])
            Xtr_s_sc = np.concatenate([Xtr_s_real_sc, Xtr_s_syn_sc.astype(np.float32)])
        else:
            Xtr_s_sc = Xtr_s_real_sc

        # pos_weight for TCN: when SMOTE is active the training set is roughly balanced
        # (n_pos ≈ n_neg).  Applying pw_uniform (the true imbalance ratio, typically
        # 5–10×) on top of SMOTE-balanced data double-weights positives and degrades
        # calibration.  Use pw=1.0 when SMOTE handles balance; fall back to pw_uniform
        # when class-weighted focal loss is the sole rebalancing mechanism.
        # Computed from fine-stride training labels — the distribution the TCN trains on.
        # LR/XGB use pw_bl (derived from uniform-stride y_bl) computed below.
        pw_uniform  = float((y_tr_orig == 0).sum()) / max(float((y_tr_orig == 1).sum()), 1.0)
        pw_bl_smote = 1.0 if USE_SMOTE else pw_uniform

        # ── LR & XGB baselines ──────────────────────────────────────────────────
        bl_feats_tr = window_baseline_feats(X_bl_raw)   # uniform-stride, LOO-renormed
        bl_feats_te = window_baseline_feats(X_te)
        # LR/XGB train on uniform-stride windows (X_bl_raw) that match the test
        # distribution, avoiding the near-event density bias of fine_stride=True.
        # Imbalance ratio is derived from the same uniform-stride training labels
        # so scale_pos_weight reflects the actual test-time class balance.
        y_tr_bl = y_bl
        pw_bl   = float((y_bl == 0).sum()) / max(float((y_bl == 1).sum()), 1.0)

        # StandardScaler for LR (fix #25): window_baseline_feats returns raw
        # mean/std/max statistics whose scales differ by orders of magnitude across
        # signals (HR in bpm, steering in degrees, speed in km/h).  Without scaling,
        # L2 regularisation penalises large-scale features less, biasing LR toward
        # lower performance and making the comparison to TCN models unfair.
        # XGB is scale-invariant (tree splits) so no scaling is needed there.
        # Scaler is fit on training windows only (no val/test contamination).
        scaler_lr    = StandardScaler()
        bl_feats_tr_sc = scaler_lr.fit_transform(bl_feats_tr)
        bl_feats_te_sc = scaler_lr.transform(bl_feats_te)

        lr_model = LogisticRegression(max_iter=5000, class_weight="balanced",
                                      random_state=SEED)
        lr_model.fit(bl_feats_tr_sc, y_tr_bl.astype(int))
        lr_scores = lr_model.predict_proba(bl_feats_te_sc)[:, 1]

        xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pw_bl, eval_metric="logloss",
            random_state=SEED, verbosity=0,
        )
        xgb_model.fit(bl_feats_tr, y_tr_bl.astype(int))
        xgb_scores = xgb_model.predict_proba(bl_feats_te)[:, 1]

        # ── DB-TCN (spectral, no DANN) — fair within-script baseline ────────────
        # Re-seed before each model so DB-TCN and DANN have independent weight
        # initialisation and DataLoader sampling sequences within the fold.
        torch.manual_seed(SEED ^ seed_d)
        torch.cuda.manual_seed_all(SEED ^ seed_d)
        print(f"{d:<10} | training DB-TCN...")
        db_model = DANNDualBranchTCN(n_phys_feat, n_kin_feat).to(DEVICE)
        db_model  = train_db_tcn(db_model,
                                 Xtr_p_sc, Xtr_k_sc, Xtr_s_sc, y_tr_train,
                                 Xval_p_sc, Xval_k_sc, Xval_s_sc, y_val_d,
                                 pos_weight=pw_bl_smote)
        db_model.eval()
        with torch.no_grad():
            db_scores = torch.sigmoid(
                db_model(torch.as_tensor(Xte_p_sc).to(DEVICE),
                         torch.as_tensor(Xte_k_sc).to(DEVICE),
                         torch.as_tensor(Xte_s_sc).to(DEVICE))[0]
            ).cpu().numpy()

        # ── DANN-DB-TCN ──────────────────────────────────────────────────────────
        print(f"{d:<10} | training DANN-DB-TCN...")
        tr_subjects = np.unique(pid_tr[~vmask])
        subj2id     = {s: i for i, s in enumerate(tr_subjects)}
        subj_ids_orig = np.array([subj2id[p] for p in pid_tr[~vmask]], dtype=np.int64)
        # Synthetic windows receive a λ-weighted subject ID: the neighbour's ID
        # when λ ≥ 0.5 (window is geometrically closer to the neighbour), the
        # anchor's ID otherwise.  Ensures every training sample — real and
        # synthetic alike — receives an adversarial gradient through the GRL,
        # and is more principled than always using the anchor when λ is large.
        if n_synthetic > 0:
            assert len(_smote_anchors) == n_synthetic, (
                f"smote_raw() returned {len(_smote_anchors)} anchor indices but "
                f"{n_synthetic} synthetic samples — contract violation"
            )
            # Synthetic windows always inherit the anchor's subject ID (fix #40 /
            # fix #30 implementation correction).  The previous λ-weighted rule
            # (neighbour's ID when λ ≥ 0.5) caused two synthetics from the same
            # anchor-neighbour pair to carry different IDs based purely on λ,
            # artificially inflating cross-subject diversity in the discriminator's
            # training set.  Always using the anchor's ID is consistent with fix #30's
            # stated intent and ensures each synthetic sample is treated as an
            # augmented observation of the driver who generated its anchor window.
            syn_sids = np.array([
                subj2id[pid_tr_train[_smote_anchors[i]]]
                for i in range(n_synthetic)
            ], dtype=np.int64)
            subj_ids_tr = np.concatenate([subj_ids_orig, syn_sids])
        else:
            subj_ids_tr = subj_ids_orig

        torch.manual_seed(SEED ^ seed_d ^ 0x1)
        torch.cuda.manual_seed_all(SEED ^ seed_d ^ 0x1)
        dann_model = DANNDualBranchTCN(n_phys_feat, n_kin_feat).to(DEVICE)
        # Discriminator input = torch.cat([fusion, spec_adapted]) = 114-d (fix #21):
        # GRL gradients flow back through spec_adapter, making spectral features
        # subject-invariant alongside the fusion vector.
        disc        = SubjectDiscriminator(fusion_dim + SPECTRAL_DIM, n_subjects=len(tr_subjects)).to(DEVICE)
        dann_model, diag = train_dann_tcn(
            dann_model, disc,
            Xtr_p_sc, Xtr_k_sc, Xtr_s_sc, y_tr_train, subj_ids_tr,
            Xval_p_sc, Xval_k_sc, Xval_s_sc, y_val_d,
            pos_weight=pw_bl_smote,
        )
        dann_model.eval()
        # spec_adapter deviation diagnostic: measures how far the GRL has pushed
        # spec_adapter's weight from its identity initialisation.  A value near 0
        # means spectral features are effectively un-regularised by the adversarial
        # objective (pass-through); a larger value confirms meaningful invariance
        # learning.  Reported per fold to detect folds where adversarial training
        # converged to a near-identity solution.
        with torch.no_grad():
            _W_sa  = dann_model.spec_adapter.weight
            _I_sa  = torch.eye(SPECTRAL_DIM, device=_W_sa.device)
            _dev_sa = float((_W_sa - _I_sa).norm(p="fro").item() / SPECTRAL_DIM)
        print(f"{d:<10} | spec_adapter ||W−I||_F/dim = {_dev_sa:.4f}"
              f"{'  [near-identity: GRL may not have regularised spectral features]' if _dev_sa < 0.05 else ''}")

        with torch.no_grad():
            dann_scores = torch.sigmoid(
                dann_model(torch.as_tensor(Xte_p_sc).to(DEVICE),
                           torch.as_tensor(Xte_k_sc).to(DEVICE),
                           torch.as_tensor(Xte_s_sc).to(DEVICE))[0]
            ).cpu().numpy()

        # ── Val-weighted blend (no test leakage) ─────────────────────────────────
        with torch.no_grad():
            val_db_sc = torch.sigmoid(
                db_model(torch.as_tensor(Xval_p_sc).to(DEVICE),
                         torch.as_tensor(Xval_k_sc).to(DEVICE),
                         torch.as_tensor(Xval_s_sc).to(DEVICE))[0]
            ).cpu().numpy()
            val_dann_sc = torch.sigmoid(
                dann_model(torch.as_tensor(Xval_p_sc).to(DEVICE),
                           torch.as_tensor(Xval_k_sc).to(DEVICE),
                           torch.as_tensor(Xval_s_sc).to(DEVICE))[0]
            ).cpu().numpy()
        # Fixed equal-weight blend: using validation AUC to tune w_dann would leak
        # validation performance structure into test predictions, violating LOPO-CV.
        val_auc_db   = _nan_or(safe_auc(y_val_d, val_db_sc))
        val_auc_dann = _nan_or(safe_auc(y_val_d, val_dann_sc))
        w_dann  = 0.5
        blend_scores = w_dann * dann_scores + (1.0 - w_dann) * db_scores

        # Per-fold thresholds from validation-set Youden's J.
        # These are stored and used in the threshold metrics section to avoid
        # selecting the threshold on the test set (which would inflate F1/Prec/Rec).
        val_blend_sc     = w_dann * val_dann_sc + (1.0 - w_dann) * val_db_sc
        fold_thresh_db    = _val_threshold(y_val_d, val_db_sc)
        fold_thresh_dann  = _val_threshold(y_val_d, val_dann_sc)
        fold_thresh_blend = _val_threshold(y_val_d, val_blend_sc)
        pool_thresh_db.append(fold_thresh_db)
        pool_thresh_dann.append(fold_thresh_dann)
        pool_thresh_blend.append(fold_thresh_blend)

        # Adversarial accuracy on held-in training windows — convergence diagnostic.
        # Val subjects are NOT in subj2id (held out from discriminator training).
        # SMOTE appends synthetic rows; [:n_orig_tr] are the real training windows.
        n_orig_tr = len(subj_ids_orig)
        adv_acc   = _adv_accuracy(dann_model, disc,
                                  Xtr_p_sc[:n_orig_tr], Xtr_k_sc[:n_orig_tr],
                                  Xtr_s_sc[:n_orig_tr], subj_ids_orig)
        chance    = 1.0 / len(tr_subjects)

        # Held-out invariance diagnostic on val windows (subjects unseen by disc).
        # High entropy / low max-confidence → discriminator is genuinely confused
        # about unseen subjects, indicating cross-subject feature invariance beyond
        # the training distribution (stronger than the in-training adv_acc alone).
        val_entropy, val_max_conf, max_ent = _adv_entropy_val(
            dann_model, disc, Xval_p_sc, Xval_k_sc, Xval_s_sc)

        dann_diagnostics.append({
            "driver": d, "adv_acc": adv_acc, "chance": chance,
            "n_tr_subjects": int(len(tr_subjects)),
            "val_entropy": val_entropy, "val_max_conf": val_max_conf,
            "max_entropy": max_ent,
            "final_class_loss":  diag["class_loss"][-1],
            "final_adv_loss":    diag["adv_loss"][-1],
            "final_lambda":      diag["lambda"][-1],
        })
        print(f"{d:<10} | adv_acc={adv_acc:.3f}  chance={chance:.3f}  ratio={adv_acc/chance:.1f}x"
              f"  val_entropy={val_entropy:.3f}/{max_ent:.3f}  val_max_conf={val_max_conf:.3f}")

        # ── Gate adaptation ──────────────────────────────────────────────────────
        gate_scores, _ = gate_adapt(dann_model, Xte_p_sc, Xte_k_sc, Xte_s_sc, y_te)

        # ── Platt scaling personalisation ────────────────────────────────────────
        platt_scores = platt_adapt(dann_scores, y_te)

        # ── Head fine-tuning personalisation ─────────────────────────────────────
        head_scores = head_adapt(dann_model, Xte_p_sc, Xte_k_sc, Xte_s_sc, y_te)

        # ── Gate values ──────────────────────────────────────────────────────────
        gate_vals = dann_model.gate_values(torch.as_tensor(Xte_p_sc),
                                           torch.as_tensor(Xte_k_sc))
        mean_gate = float(gate_vals.mean())

        # ── Per-driver AUC ───────────────────────────────────────────────────────
        auc_lr    = _nan_or(safe_auc(y_te, lr_scores))
        auc_xgb   = _nan_or(safe_auc(y_te, xgb_scores))
        auc_db    = _nan_or(safe_auc(y_te, db_scores))
        auc_dann  = _nan_or(safe_auc(y_te, dann_scores))
        auc_blend = _nan_or(safe_auc(y_te, blend_scores))
        gain      = auc_dann - auc_db

        # Personalisation AUC: evaluated on eval-only windows (fix #20).
        # The first GATE_ADAPT_K windows were used to fit/adapt the personalisation
        # model; including them in AUC would inflate it because the model has already
        # seen their labels.  Score arrays still cover all windows (single consistent
        # scale) but only y_te[K:] / scores[K:] enter the metric computation.
        _K = GATE_ADAPT_K
        _has_eval = len(y_te) > _K
        if gate_scores is not None and _has_eval:
            auc_ga = _nan_or(safe_auc(y_te[_K:], gate_scores[_K:]))
        else:
            auc_ga = float("nan")
        if _has_eval:
            auc_platt = _nan_or(safe_auc(y_te[_K:], platt_scores[_K:]))
        else:
            auc_platt = float("nan")
        if head_scores is not None and _has_eval:
            auc_head = _nan_or(safe_auc(y_te[_K:], head_scores[_K:]))
        else:
            auc_head = float("nan")

        print(f"{d:<10} | {len(y_te):>5}  {100*y_te.mean():>5.1f}% | "
              f"{auc_lr:>7.4f} {auc_xgb:>7.4f} {auc_db:>7.4f} | "
              f"{auc_dann:>8.4f} {auc_ga:>8.4f} {auc_platt:>8.4f} {auc_head:>8.4f} {gain:>+6.4f} | "
              f"{auc_blend:>7.4f} | {mean_gate:>7.3f}")

        pool_y.append(y_te);    pool_db.append(db_scores)
        pool_dann.append(dann_scores)
        # gate_scores is None when gate adaptation was skipped (e.g. < GATE_ADAPT_K
        # support windows available for this driver). Fall back to population DANN
        # scores so pool_ga is always populated and pooled metrics remain comparable.
        pool_ga.append(gate_scores)  # None when gate adaptation was skipped
        pool_blend.append(blend_scores)
        pool_lr.append(lr_scores); pool_xgb.append(xgb_scores)
        pool_Xte_p.append(Xte_p_sc); pool_Xte_k.append(Xte_k_sc)
        pool_Xte_s.append(Xte_s_sc)
        pool_models_dann.append(copy.deepcopy(dann_model))
        pool_etypes.append(etypes_d); pool_gates.append(mean_gate)
        pool_gate_arrays.append(gate_vals)   # per-window α cached here

        per_driver_results.append({
            "driver": d, "n_windows": int(len(y_te)),
            "pos_rate": float(y_te.mean()),
            "auc_lr": auc_lr, "auc_xgb": auc_xgb, "auc_db": auc_db,
            "auc_dann": auc_dann, "auc_blend": auc_blend,
            "val_auc_db": val_auc_db, "val_auc_dann": val_auc_dann,
            "dann_gain": gain, "mean_gate": mean_gate, "adv_acc": adv_acc,
            # val-set thresholds used for threshold-dependent metrics (fix: avoids
            # selecting threshold on test data which inflates F1/Prec/Rec).
            "fold_thresh_db":    fold_thresh_db,
            "fold_thresh_dann":  fold_thresh_dann,
            "fold_thresh_blend": fold_thresh_blend,
            # personalisation: these methods observe y_te[:GATE_ADAPT_K] (test-participant
            # labels) and are NOT held-out generalisation estimates. Keep separate from
            # population metrics above to prevent conflation in downstream analysis.
            "personalisation": {
                "auc_gate":  auc_ga,
                "auc_platt": auc_platt,
                "auc_head":  auc_head,
            },
        })

        # ── Ablation single-branch TCNs ──────────────────────────────────────────
        for abl_i, (abl_name, abl_idx) in enumerate(ABLATION.items()):
            n_abl_feat = len(abl_idx) * 5
            # SMOTE: k-NN in raw ablation-signal space, interpolation in scaled
            # feature space (same treatment as the main dual-branch pipeline).
            if USE_SMOTE:
                _abl_smote_rng = np.random.default_rng(
                    SEED ^ seed_d ^ SMOTE_SEED_SALT ^ (0xAB00 + abl_i))
                _, y_tr_abl, _abl_smote_anchors, _abl_smote_neighbors, _abl_smote_lambdas = smote_raw(
                    X_tr_train_raw[:, :, abl_idx], y_tr_orig, rng=_abl_smote_rng,
                    pids=pid_tr_train, routes=routes_tr_train, t_starts=ts_tr_train)
            else:
                y_tr_abl             = y_tr_orig
                _abl_smote_anchors   = np.empty(0, dtype=np.int64)
                _abl_smote_neighbors = np.empty(0, dtype=np.int64)
                _abl_smote_lambdas   = np.empty(0, dtype=np.float32)
            n_abl_real = len(y_tr_orig)
            n_abl_syn  = len(_abl_smote_anchors)

            Xtr_abl_real = apply_features_all(X_tr_train_raw[:, :, abl_idx])
            Xval_abl     = apply_features_branch(X_val_raw, abl_idx)
            Xte_abl      = apply_features_branch(X_te,      abl_idx)
            sc_abl       = StandardScaler()
            sc_abl.fit(Xtr_abl_real.reshape(-1, n_abl_feat))
            Xtr_abl_real_sc = sc_abl.transform(
                Xtr_abl_real.reshape(-1, n_abl_feat)).reshape(-1, LOOKBACK_S, n_abl_feat)
            Xval_abl_sc = sc_abl.transform(
                Xval_abl.reshape(-1, n_abl_feat)).reshape(-1, LOOKBACK_S, n_abl_feat)
            Xte_abl_sc  = sc_abl.transform(
                Xte_abl.reshape(-1, n_abl_feat)).reshape(-1, LOOKBACK_S, n_abl_feat)
            if n_abl_syn > 0:
                lams_abl = _abl_smote_lambdas[:, None, None]
                Xtr_abl_syn_sc = ((1.0 - lams_abl) * Xtr_abl_real_sc[_abl_smote_anchors]
                                  + lams_abl        * Xtr_abl_real_sc[_abl_smote_neighbors])
                Xtr_abl_sc = np.concatenate([Xtr_abl_real_sc, Xtr_abl_syn_sc.astype(np.float32)])
            else:
                Xtr_abl_sc = Xtr_abl_real_sc

            # Per-branch spectral features: λ-weighted interpolation in scaled
            # spectral space, consistent with the main dual-branch treatment.
            _abl_spec_real = Xtr_sp_raw_real if abl_name == "Physiology" else Xtr_sk_raw_real
            _abl_spec_val  = Xval_sp_raw     if abl_name == "Physiology" else Xval_sk_raw
            _abl_spec_te   = Xte_sp_raw      if abl_name == "Physiology" else Xte_sk_raw
            _abl_spec_dim  = _abl_spec_real.shape[1]

            sc_abl_s = StandardScaler()
            sc_abl_s.fit(_abl_spec_real)
            Xtr_abl_s_real_sc = sc_abl_s.transform(_abl_spec_real).astype(np.float32)
            if n_abl_syn > 0:
                lams_abl_s = _abl_smote_lambdas[:, None]
                Xtr_abl_s_syn_sc = ((1.0 - lams_abl_s) * Xtr_abl_s_real_sc[_abl_smote_anchors]
                                    + lams_abl_s        * Xtr_abl_s_real_sc[_abl_smote_neighbors])
                Xtr_abl_s_sc = np.concatenate([Xtr_abl_s_real_sc,
                                               Xtr_abl_s_syn_sc.astype(np.float32)])
            else:
                Xtr_abl_s_sc = Xtr_abl_s_real_sc
            Xval_abl_s_sc = sc_abl_s.transform(_abl_spec_val).astype(np.float32)
            Xte_abl_s_sc  = sc_abl_s.transform(_abl_spec_te).astype(np.float32)

            torch.manual_seed(SEED ^ seed_d ^ (0x2 + abl_i))
            torch.cuda.manual_seed_all(SEED ^ seed_d ^ (0x2 + abl_i))
            # Select branch architecture to match the corresponding branch in DANNDualBranchTCN.
            _branch_type = "phys" if abl_name == "Physiology" else "kin"
            abl_m = SingleBranchTCN(n_abl_feat, spectral_dim=_abl_spec_dim, branch=_branch_type).to(DEVICE)
            abl_m = train_single_tcn(abl_m, Xtr_abl_sc, y_tr_abl, Xval_abl_sc, y_val_d,
                                     Xtr_spec=Xtr_abl_s_sc, Xval_spec=Xval_abl_s_sc,
                                     pos_weight=pw_bl_smote)
            abl_m.eval()
            with torch.no_grad():
                abl_sc = torch.sigmoid(
                    abl_m(torch.as_tensor(Xte_abl_sc).to(DEVICE),
                          torch.as_tensor(Xte_abl_s_sc).to(DEVICE))).cpu().numpy()
            pool_ablation[abl_name].append(
                {"y": y_te, "scores": abl_sc, "auc": safe_auc(y_te, abl_sc)})

    # ════════════════════════════════════════════════════════════════════════════
    # POOLED EVALUATION
    # ════════════════════════════════════════════════════════════════════════════
    # Global Youden's J threshold: pooled across all val folds so the same
    # threshold is applied to every test fold, removing the bias that arises
    # when a per-fold val threshold (multi-driver pool) is transferred to a
    # single test driver with a different class rate.
    # Per-fold thresholds are already in pool_thresh_db/dann/blend (collected during
    # the LOPO loop).  No pooled threshold computation needed.

    if not pool_y:
        print("\n[ERROR] All LOPO folds were skipped (single-class validation). "
              "No evaluation possible. Aborting.")
        return

    if singleclass_val_folds:
        print(f"\n[WARN] {len(singleclass_val_folds)}/{len(drivers)} folds had a single-class "
              f"validation set — AUC-based early stopping was replaced by training-loss "
              f"patience and threshold fell back to 0.5 for: {singleclass_val_folds}")
        print(f"  These folds may still be slightly overfit (loss-patience is a weaker "
              f"stopping signal than val-AUC). Interpret their per-driver results with caution.")

    all_y     = np.concatenate(pool_y)
    all_db    = np.concatenate(pool_db)
    all_dann  = np.concatenate(pool_dann)
    all_blend = np.concatenate(pool_blend)
    all_lr    = np.concatenate(pool_lr)
    all_xgb   = np.concatenate(pool_xgb)

    # GateAdapt: exclude folds where adaptation was skipped (gate_scores is None)
    # and slice to eval-only windows [K:] so the support set used for adaptation
    # is excluded from metric computation (fix #20).
    _K = GATE_ADAPT_K
    # Track which fold indices are valid for gate adaptation so we can compute
    # a matched DANN reference on the same subset for a fair comparison.
    ga_valid_idx = [i for i, (y, s) in enumerate(zip(pool_y, pool_ga))
                    if s is not None and len(y) > _K]
    ga_valid = [(pool_y[i][_K:], pool_ga[i][_K:]) for i in ga_valid_idx]
    if ga_valid:
        ga_y_parts, ga_s_parts = zip(*ga_valid)
        all_ga_y    = np.concatenate(ga_y_parts)
        all_ga      = np.concatenate(ga_s_parts)
        drv_aucs_ga = [v for v in (safe_auc(y, s) for y, s in ga_valid) if v is not None]
        # DANN population scores on the same folds and eval-only windows — fair reference.
        dann_eval_valid = [(pool_y[i][_K:], pool_dann[i][_K:]) for i in ga_valid_idx]
        all_dann_eval_y = np.concatenate([y for y, _ in dann_eval_valid])
        all_dann_eval   = np.concatenate([s for _, s in dann_eval_valid])
        drv_aucs_dann_eval = [v for v in (safe_auc(y, s) for y, s in dann_eval_valid)
                              if v is not None]
    else:
        all_ga_y = all_ga = np.array([])
        drv_aucs_ga = []
        all_dann_eval_y = all_dann_eval = np.array([])
        drv_aucs_dann_eval = []

    drv_aucs_lr    = [v for v in (safe_auc(y, s) for y, s in zip(pool_y, pool_lr))    if v is not None]
    drv_aucs_xgb   = [v for v in (safe_auc(y, s) for y, s in zip(pool_y, pool_xgb))   if v is not None]
    drv_aucs_db    = [v for v in (safe_auc(y, s) for y, s in zip(pool_y, pool_db))    if v is not None]
    drv_aucs_dann  = [v for v in (safe_auc(y, s) for y, s in zip(pool_y, pool_dann))  if v is not None]
    drv_aucs_blend = [v for v in (safe_auc(y, s) for y, s in zip(pool_y, pool_blend)) if v is not None]

    ci_dann_win = bootstrap_auc_ci_windows(all_y, all_dann)
    ci_dann_drv = bootstrap_auc_ci_drivers(drv_aucs_dann)

    def _print_pooled(name, all_s, drv_aucs, y_override=None):
        y_ref  = y_override if y_override is not None else all_y
        auc_w  = _nan_or(safe_auc(y_ref, all_s))
        ci_d   = bootstrap_auc_ci_drivers(drv_aucs)
        auprc  = _nan_or(safe_auprc(y_ref, all_s))
        brier  = brier_score_loss(y_ref, all_s)
        ece    = compute_ece(y_ref, all_s)
        mean_d = np.mean(drv_aucs) if drv_aucs else float("nan")
        std_d  = np.std(drv_aucs)  if drv_aucs else float("nan")
        print(f"\n  {name}:")
        print(f"    Driver  AUROC : {mean_d:.4f} ± {std_d:.4f}  [{ci_d[0]:.4f}, {ci_d[1]:.4f}]  ← primary metric")
        print(f"    Pooled  AUROC : {auc_w:.4f}  (point estimate only — window-level CI omitted: ~94% window overlap makes it anticonservative)")
        print(f"    AUPRC         : {auprc:.4f}")
        print(f"    Brier Score   : {brier:.4f}")
        print(f"    ECE           : {ece:.4f}")

    print(f"\n{'='*72}")
    print("POOLED EVALUATION — POPULATION MODELS (zero-shot generalisation)")
    print(f"{'='*72}")
    print(f"  These models never see test-participant labels. Results are comparable.")
    print(f"  DANN-DB-TCN is INDUCTIVE (fix #24): the test subject's windows are never")
    print(f"  seen during training. Cross-subject invariance is learned solely from")
    print(f"  training subjects via SubjectDiscriminator + GRL.")
    _print_pooled("LR baseline",      all_lr,    drv_aucs_lr)
    _print_pooled("XGB baseline",     all_xgb,   drv_aucs_xgb)
    _print_pooled("DB-TCN (no DANN)", all_db,    drv_aucs_db)
    _print_pooled("DANN-DB-TCN (inductive: test subject unseen during training)", all_dann,  drv_aucs_dann)
    _print_pooled("EqualBlend (DB + DANN, fixed 50/50)", all_blend, drv_aucs_blend)

    # Wilcoxon signed-rank: DANN-DB-TCN > DB-TCN (paired per driver)
    paired = [(r["auc_db"], r["auc_dann"]) for r in per_driver_results
              if not (np.isnan(r["auc_db"]) or np.isnan(r["auc_dann"]))]
    if len(paired) >= 5:
        db_p, da_p = zip(*paired)
        try:
            wstat, wpval = wilcoxon(list(da_p), list(db_p), alternative="greater")
            sig = "***" if wpval < 0.001 else ("**" if wpval < 0.01 else ("*" if wpval < 0.05 else "ns"))
            mean_gain = np.mean(np.array(da_p) - np.array(db_p))
            print(f"\n  Wilcoxon (DANN > DB-TCN) : W={wstat:.1f}  p={wpval:.4f}  {sig}"
                  f"  [pre-specified primary comparison; no correction for multiple tests]")
            print(f"  Mean per-driver gain     : {mean_gain:+.4f}")
        except Exception as e:
            print(f"  [WARN] Wilcoxon test failed: {e}")
    print(f"{'='*72}")

    print(f"\n{'='*72}")
    print("POOLED EVALUATION — ONLINE PERSONALISATION (uses test-participant labels)")
    print(f"{'='*72}")
    print(f"  WARNING: these methods observe y_te[:GATE_ADAPT_K={GATE_ADAPT_K}] from the test")
    print(f"  participant. They are NOT held-out generalisation estimates and MUST NOT")
    print(f"  be compared directly to the population models above.")
    print(f"  A matched DANN reference evaluated on the same folds and eval-only")
    print(f"  windows [K:] is shown alongside each personalisation method for a fair")
    print(f"  within-subset comparison.")
    if ga_valid:
        _print_pooled(
            f"DANN population (matched: same {len(ga_valid)} folds, eval windows [K:])",
            all_dann_eval, drv_aucs_dann_eval, y_override=all_dann_eval_y)
        _print_pooled(
            f"DANN + GateAdapt (online, {len(ga_valid)}/{len(pool_y)} folds)",
            all_ga, drv_aucs_ga, y_override=all_ga_y)
    else:
        print("\n  DANN + GateAdapt (online): no folds with valid adaptation.")
    print(f"{'='*72}")

    # ── THRESHOLD METRICS ────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("THRESHOLD-DEPENDENT METRICS  (per-fold Youden's J from fold validation set)")
    print(f"{'='*72}")
    print(f"  Each fold's threshold is selected by Youden's J on that fold's own val")
    print(f"  predictions, then applied to that fold's test predictions only.  Avoids")
    print(f"  pooling scores from different fold-specific model instances (which have")
    print(f"  potentially different marginal distributions) into a single threshold.")
    thresh_summary_db   = f"{np.mean(pool_thresh_db):.3f}±{np.std(pool_thresh_db):.3f}"
    thresh_summary_dann = f"{np.mean(pool_thresh_dann):.3f}±{np.std(pool_thresh_dann):.3f}"
    thresh_summary_bld  = f"{np.mean(pool_thresh_blend):.3f}±{np.std(pool_thresh_blend):.3f}"
    print(f"  Per-fold threshold mean±std — DB: {thresh_summary_db}"
          f"  DANN: {thresh_summary_dann}  Blend: {thresh_summary_bld}")
    if len(np.unique(all_y)) >= 2 and per_driver_results:
        pool_pred_db    = [(db_f >= th).astype(int)  for db_f, th in zip(pool_db,    pool_thresh_db)]
        pool_pred_dann  = [(dn_f >= th).astype(int)  for dn_f, th in zip(pool_dann,  pool_thresh_dann)]
        pool_pred_blend = [(bl_f >= th).astype(int)  for bl_f, th in zip(pool_blend, pool_thresh_blend)]
        # Pooled window-level F1/Prec/Rec are NOT reported: adjacent windows share
        # 94% of timesteps (WINDOW_STEP=5, LOOKBACK_S=90), so pooled metrics treat
        # ~17× fewer independent observations as independent, producing
        # anticonservative (over-optimistic) estimates.  Driver-level metrics
        # (each driver = one independent observation) are the sole reported values.
        for name, pool_preds in [
                ("DB-TCN (no DANN)",   pool_pred_db),
                ("DANN-DB-TCN",        pool_pred_dann),
                ("Ensemble (DB+DANN)", pool_pred_blend)]:
            drv_f1, drv_prec, drv_rec = [], [], []
            for fy, fp in zip(pool_y, pool_preds):
                if len(np.unique(fy)) >= 2:
                    drv_f1.append(float(f1_score(fy,   fp, zero_division=0)))
                    drv_prec.append(float(precision_score(fy, fp, zero_division=0)))
                    drv_rec.append(float(recall_score(fy,  fp, zero_division=0)))
            df1  = f"{np.mean(drv_f1):.3f}±{np.std(drv_f1):.3f}"   if drv_f1   else "n/a"
            dpr  = f"{np.mean(drv_prec):.3f}±{np.std(drv_prec):.3f}" if drv_prec else "n/a"
            drc  = f"{np.mean(drv_rec):.3f}±{np.std(drv_rec):.3f}"  if drv_rec  else "n/a"
            print(f"  {name:<22}  driver F1={df1} Prec={dpr} Rec={drc}")

    # ── GATE DISTRIBUTION ────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("MODALITY GATE DISTRIBUTION  (DANN-DB-TCN population model)")
    print(f"{'='*72}")
    all_gates = np.array(pool_gates)
    print(f"  α (physiology weight): mean={all_gates.mean():.3f}  std={all_gates.std():.3f}")
    print(f"  α < 0.3  (kin-dominant)  : {(all_gates < 0.3).sum():>3d} drivers "
          f"({100*(all_gates < 0.3).mean():.1f}%)")
    print(f"  α ∈ [0.3, 0.5)           : {((all_gates>=0.3)&(all_gates<0.5)).sum():>3d} drivers "
          f"({100*((all_gates>=0.3)&(all_gates<0.5)).mean():.1f}%)")
    print(f"  α ≥ 0.5  (phys-dominant) : {(all_gates >= 0.5).sum():>3d} drivers "
          f"({100*(all_gates >= 0.5).mean():.1f}%)")

    # Per-window gate values vs predicted risk (already computed during LOPO — no recomputation)
    gate_per_driver = pool_gate_arrays
    gate_vals_all   = np.concatenate(gate_per_driver)
    if len(gate_vals_all) == len(all_dann):
        r_gs, p_gs = pearsonr(gate_vals_all, all_dann)
        print(f"\n  Corr(α, predicted risk): r={r_gs:.3f}  p={p_gs:.3e}")

    # Within-driver gate variance (does α adapt across windows or stay static?)
    within_vars = [g.var() for g in gate_per_driver if len(g) > 1]
    if within_vars:
        print(f"  α within-driver variance: mean={np.mean(within_vars):.4f}  "
              f"std={np.std(within_vars):.4f}")

    # ── PERMUTATION FEATURE IMPORTANCE ──────────────────────────────────────────
    print(f"\n{'='*72}")
    print("PERMUTATION FEATURE IMPORTANCE — DANN-DB-TCN")
    print(f"{'='*72}")

    n_perm_drivers = len(pool_models_dann)   # use all available folds

    sig_importance  = {col: [] for col in SIGNAL_COLS}
    spec_importance = []
    band_importance = {f"[{lo:.2f},{hi:.2f})Hz": [] for lo, hi in SPECTRAL_BANDS}
    n_kin_sigs      = len(KIN_COLS) + len(PHYS_COLS)  # total signals per spectral band

    for i in range(n_perm_drivers):
        # Per-driver RNG: seeded from both the global seed and the driver index so
        # that permutation results are independently reproducible per fold.
        seed_d_imp = int(hashlib.md5(str(per_driver_results[i]["driver"]).encode()).hexdigest(), 16) & 0xFFFFFFFF
        rng_imp = np.random.default_rng(SEED + 2 + seed_d_imp)
        m   = pool_models_dann[i]; m.eval()
        Xp  = torch.as_tensor(pool_Xte_p[i]).to(DEVICE)
        Xk  = torch.as_tensor(pool_Xte_k[i]).to(DEVICE)
        Xs  = torch.as_tensor(pool_Xte_s[i]).to(DEVICE)
        yt  = pool_y[i]
        with torch.no_grad():
            base_s = torch.sigmoid(m(Xp, Xk, Xs)[0]).cpu().numpy()
        base_auc = safe_auc(yt, base_s)
        if base_auc is None: continue

        # Per-signal permutation — averaged over N_PERM_REPEATS shuffles to reduce
        # variance (single-permutation estimates have high noise with N≈30 drivers).
        # Spectral features are co-permuted: each signal occupies one column per band
        # in Xs (layout: [kin0..kin3, phys0, phys1] × 3 bands).  Permuting only the
        # temporal branch while leaving Xs intact would let the model recover the
        # signal's frequency information from Xs, underestimating signal importance.
        for ci, col in enumerate(SIGNAL_COLS):
            is_phys = ci in PHYS_IDX
            branch_tensor = Xp if is_phys else Xk
            n_branch_sigs = len(PHYS_COLS) if is_phys else len(KIN_COLS)
            local_i       = PHYS_IDX.index(ci) if is_phys else KIN_IDX.index(ci)
            feat_cols     = _signal_feat_cols(local_i, n_branch_sigs)  # 5 columns
            # Spectral columns for this signal: one per frequency band.
            # Layout per band b: columns [b*n_kin_sigs .. b*n_kin_sigs+n_kin_sigs).
            # Kin signals occupy the first len(KIN_COLS) slots; phys follow.
            _spec_offset  = len(KIN_COLS) if is_phys else 0
            spec_cols_sig = [b * n_kin_sigs + _spec_offset + local_i
                             for b in range(len(SPECTRAL_BANDS))]

            rep_drops = []
            for _rep in range(N_PERM_REPEATS):
                bt_perm  = branch_tensor.cpu().numpy().copy()
                xs_perm  = Xs.cpu().numpy().copy()
                idx_perm = rng_imp.permutation(len(yt))
                for fc in feat_cols:
                    bt_perm[:, :, fc] = bt_perm[idx_perm, :, fc]
                for sc in spec_cols_sig:
                    xs_perm[:, sc] = xs_perm[idx_perm, sc]
                bt_perm = torch.as_tensor(bt_perm).to(DEVICE)
                xs_perm = torch.as_tensor(xs_perm).to(DEVICE)
                with torch.no_grad():
                    perm_s = torch.sigmoid(
                        m(bt_perm if is_phys else Xp,
                          Xk if is_phys else bt_perm,
                          xs_perm)[0]).cpu().numpy()
                perm_auc = safe_auc(yt, perm_s)
                if perm_auc is not None:
                    rep_drops.append(base_auc - perm_auc)
            if rep_drops:
                sig_importance[col].append(float(np.mean(rep_drops)))

        # All spectral features — averaged over N_PERM_REPEATS
        rep_sp_drops = []
        for _rep in range(N_PERM_REPEATS):
            perm_idx = rng_imp.permutation(len(yt))
            Xs_perm  = torch.as_tensor(Xs.cpu().numpy()[perm_idx]).to(DEVICE)
            with torch.no_grad():
                perm_sp = torch.sigmoid(m(Xp, Xk, Xs_perm)[0]).cpu().numpy()
            perm_sp_auc = safe_auc(yt, perm_sp)
            if perm_sp_auc is not None:
                rep_sp_drops.append(base_auc - perm_sp_auc)
        if rep_sp_drops:
            spec_importance.append(float(np.mean(rep_sp_drops)))

        # Per-band spectral importance — averaged over N_PERM_REPEATS
        for b_i, lbl in enumerate(band_importance.keys()):
            rep_b_drops = []
            for _rep in range(N_PERM_REPEATS):
                Xs_b   = Xs.cpu().numpy().copy()
                cols_b = list(range(b_i * n_kin_sigs, (b_i + 1) * n_kin_sigs))
                perm_b = rng_imp.permutation(len(yt))
                Xs_b[:, cols_b] = Xs_b[perm_b][:, cols_b]
                Xs_b = torch.as_tensor(Xs_b).to(DEVICE)
                with torch.no_grad():
                    perm_b_s = torch.sigmoid(m(Xp, Xk, Xs_b)[0]).cpu().numpy()
                perm_b_auc = safe_auc(yt, perm_b_s)
                if perm_b_auc is not None:
                    rep_b_drops.append(base_auc - perm_b_auc)
            if rep_b_drops:
                band_importance[lbl].append(float(np.mean(rep_b_drops)))

    print(f"\n  Signal importance (mean ΔAUC when permuted, all {n_perm_drivers} drivers):")
    for col, drops in sorted(sig_importance.items(),
                             key=lambda x: -np.mean(x[1]) if x[1] else 0):
        if not drops: continue
        tag = "PHYS" if col in PHYS_COLS else " KIN"
        print(f"    [{tag}] {col:<28}  Δ={np.mean(drops):+.4f} ± {np.std(drops):.4f}")

    if spec_importance:
        print(f"\n  All spectral features permuted:  Δ={np.mean(spec_importance):+.4f} "
              f"± {np.std(spec_importance):.4f}")

    print(f"\n  Per-band spectral importance:")
    for lbl, drops in band_importance.items():
        if not drops: continue
        print(f"    {lbl:<22}  Δ={np.mean(drops):+.4f} ± {np.std(drops):.4f}")

    # ── DANN ADVERSARIAL DIAGNOSTIC ──────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("DANN ADVERSARIAL DIAGNOSTIC")
    print(f"{'='*72}")
    print(f"  (A) Convergence diagnostic — discriminator accuracy on HELD-IN training windows.")
    print(f"      Near-chance score → GRL confused the discriminator on the training distribution.")
    print(f"      Windows were seen during training; this measures convergence, NOT generalisation.")
    print(f"  (B) Held-out invariance — discriminator entropy on VAL windows (unseen subjects).")
    print(f"      Val subjects are not in the discriminator's training set.")
    print(f"      H near H_max=log(N_subj) → discriminator is maximally confused on unseen subjects,")
    print(f"      indicating genuine cross-subject feature invariance beyond the training fold.\n")
    hdr_d = (f"  {'Driver':<10}  {'N_subj':>6}  {'Chance':>7}  "
             f"{'AdvAcc':>7}  {'Ratio':>7}  "
             f"{'ValH':>7}  {'H_max':>7}  {'ValMaxC':>8}  Conv?")
    print(hdr_d)
    print(f"  {'-'*(len(hdr_d)-2)}")

    n_effective = 0
    all_ratios  = []
    for diag in dann_diagnostics:
        ratio     = diag["adv_acc"] / max(diag["chance"], 1e-9)
        effective = ratio < 2.0    # meaningful confusion: acc < 2× chance
        if effective: n_effective += 1
        all_ratios.append(ratio)
        flag = "YES" if effective else "no"
        print(f"  {diag['driver']:<10}  {diag['n_tr_subjects']:>6}  "
              f"{diag['chance']:>7.3f}  {diag['adv_acc']:>7.3f}  "
              f"{ratio:>7.2f}×  "
              f"{diag.get('val_entropy', float('nan')):>7.3f}  "
              f"{diag.get('max_entropy', float('nan')):>7.3f}  "
              f"{diag.get('val_max_conf', float('nan')):>8.3f}  {flag}")

    mean_val_entropy = np.mean([d.get("val_entropy", float("nan")) for d in dann_diagnostics])
    mean_max_entropy = np.mean([d.get("max_entropy", float("nan")) for d in dann_diagnostics])
    print(f"\n  Convergence (ratio < 2×)       : "
          f"{n_effective}/{len(dann_diagnostics)} drivers "
          f"({100*n_effective/max(len(dann_diagnostics),1):.1f}%)")
    print(f"  Mean in-training ratio         : {np.mean(all_ratios):.2f}× chance")
    print(f"  Mean val entropy / H_max       : {mean_val_entropy:.3f} / {mean_max_entropy:.3f}"
          f"  ({100*mean_val_entropy/max(mean_max_entropy, 1e-9):.1f}% of maximum)")

    # ── MODALITY ABLATION ────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("MODALITY ABLATION — Single-Branch TCN vs Dual-Branch variants")
    print(f"{'='*72}")
    for abl_name, results in pool_ablation.items():
        valid = [r for r in results if r["auc"] is not None]
        if not valid: continue
        aucs   = [r["auc"] for r in valid]
        all_s  = np.concatenate([r["scores"] for r in valid])
        all_yt = np.concatenate([r["y"]      for r in valid])
        auc_p  = _nan_or(safe_auc(all_yt, all_s))
        print(f"  {abl_name + ' only':<25}  Pooled={auc_p:.4f}  "
              f"Driver={np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    auc_db_p   = _nan_or(safe_auc(all_y, all_db))
    auc_dann_p = _nan_or(safe_auc(all_y, all_dann))
    print(f"  {'DB-TCN (no DANN)':<25}  Pooled={auc_db_p:.4f}  "
          f"Driver={np.mean(drv_aucs_db):.4f} ± {np.std(drv_aucs_db):.4f}")
    print(f"  {'DANN-DB-TCN':<25}  Pooled={auc_dann_p:.4f}  "
          f"Driver={np.mean(drv_aucs_dann):.4f} ± {np.std(drv_aucs_dann):.4f}")

    # ── STRATIFIED EVALUATION ────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("STRATIFIED EVALUATION — DANN-DB-TCN")
    print(f"{'='*72}")
    all_etypes = np.concatenate(pool_etypes)
    clc_mask   = np.array(["center_line_crossing" in et for et in all_etypes])

    for label, pos_mask in [("CLC events",     clc_mask & (all_y == 1)),
                             ("Non-CLC events", ~clc_mask & (all_y == 1))]:
        if pos_mask.sum() < 5: continue
        emask  = (all_y == 0) | pos_mask
        auc_d  = _nan_or(safe_auc(all_y[emask], all_db[emask]))
        auc_da = _nan_or(safe_auc(all_y[emask], all_dann[emask]))
        print(f"  {label:<20}  n_pos={pos_mask.sum():>4}  "
              f"DB-TCN={auc_d:.3f}  DANN={auc_da:.3f}  "
              f"Gain={auc_da - auc_d:+.3f}")

    # ── PER-DRIVER SUMMARY ────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("PER-DRIVER SUMMARY — DANN-DB-TCN")
    print(f"{'='*72}")
    dann_aucs = [r["auc_dann"] for r in per_driver_results if not np.isnan(r["auc_dann"])]
    db_aucs   = [r["auc_db"]   for r in per_driver_results if not np.isnan(r["auc_db"])]
    gains     = [r["dann_gain"] for r in per_driver_results]

    print(f"  N drivers evaluated  : {len(dann_aucs)}")
    print(f"  DANN-DB-TCN  AUROC   : {np.mean(dann_aucs):.4f} ± {np.std(dann_aucs):.4f}"
          f"  [{np.min(dann_aucs):.4f}, {np.max(dann_aucs):.4f}]")
    print(f"  DB-TCN       AUROC   : {np.mean(db_aucs):.4f} ± {np.std(db_aucs):.4f}")
    print(f"  Per-driver DANN gain : {np.mean(gains):+.4f} ± {np.std(gains):.4f}"
          f"  [{np.min(gains):+.4f}, {np.max(gains):+.4f}]")
    print(f"  Drivers DANN > DB    : {sum(g > 0 for g in gains)}/{len(gains)}"
          f"  ({100*sum(g>0 for g in gains)/max(len(gains),1):.1f}%)")

    # ── JSON ARTEFACTS ────────────────────────────────────────────────────────────
    results_out = {
        "model": "DANN-DB-TCN",
        "config": {
            "LAMBDA_MAX": LAMBDA_MAX, "ADV_WEIGHT": ADV_WEIGHT,
            "SPECTRAL_DIM": SPECTRAL_DIM,
            "SPECTRAL_BANDS": [[lo, hi] for lo, hi in SPECTRAL_BANDS],
            "EPOCHS": EPOCHS, "LR": LR,
            "LOOKBACK_S": LOOKBACK_S, "GAP": GAP, "HORIZON": HORIZON,
        },
        "pooled": {
            "lr":     {"auc": _nan_or(safe_auc(all_y, all_lr)),
                       "driver_mean": float(np.mean(drv_aucs_lr)),
                       "driver_std":  float(np.std(drv_aucs_lr))},
            "xgb":    {"auc": _nan_or(safe_auc(all_y, all_xgb)),
                       "driver_mean": float(np.mean(drv_aucs_xgb)),
                       "driver_std":  float(np.std(drv_aucs_xgb))},
            "db_tcn": {"auc": _nan_or(safe_auc(all_y, all_db)),
                       "driver_mean": float(np.mean(drv_aucs_db)),
                       "driver_std":  float(np.std(drv_aucs_db))},
            "dann_db": {
                "auc":         _nan_or(safe_auc(all_y, all_dann)),
                "driver_mean": float(np.mean(drv_aucs_dann)),
                "driver_std":  float(np.std(drv_aucs_dann)),
                "ci_window":   [float(ci_dann_win[0]), float(ci_dann_win[1])],
                "ci_driver":   [float(ci_dann_drv[0]), float(ci_dann_drv[1])],
            },
        },
        "per_driver":       per_driver_results,
        "dann_diagnostics": dann_diagnostics,
        "gate_values":      [float(g) for g in pool_gates],
    }

    out_path = OUT_DIR / "dann_db_tcn_results.json"
    with open(out_path, "w") as f:
        json.dump(results_out, f, indent=2, default=float)
    print(f"\n  Results saved → {out_path}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
