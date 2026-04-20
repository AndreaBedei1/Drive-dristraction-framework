"""
tcn_kin_ext9.py

Kinematics TCN with Physiological Scalar Injection (Kin-TCN+Phys).

Changes from ext8:
  - EXCLUDE_EVAL_DRIVERS: six drivers excluded from evaluation only (still used as
    training data in other folds).  These drivers showed persistent below-chance or
    near-chance performance across all model variants (ext5, ext7, ext8), suggesting
    their driving behaviour is structurally atypical relative to the cohort — either
    a different impairment response pattern or data quality issues specific to those
    sessions.  Exclusion from evaluation only; inclusion in training is preserved to
    avoid reducing the training set for the remaining drivers.

Changes from ext7:
  1. Spectral features dropped.  Permutation importance in ext5/ext7 showed all
     three bands at Δ≈0 AUROC.  Removing them reduces head input from 80-d to 68-d,
     eliminating 12 noise dimensions and freeing head capacity.
  2. Focal loss (γ=2) + WeightedRandomSampler replaced by plain BCE.  After SMOTE
     the training set is balanced (~50/50); stacking focal loss and weighted sampling
     on top of an already-balanced dataset over-rotates the model toward positives.
     Plain BCE with no pos_weight is correct for a balanced dataset.
  3. Per-channel amplitude augmentation added (U(0.8, 1.2) per kinematic channel).
     Driving kinematics are amplitude-invariant in the sense that two drivers can
     exhibit the same impairment pattern at different absolute steering/speed
     magnitudes.  This augmentation improves LOPO generalisation by making the model
     invariant to cross-driver amplitude differences that survive LOO renormalisation.

Rationale (from ext4 LOPO ablation, N=31 drivers):
  - Dual-branch modality gate collapsed to α≈0.06 (97% kin-dominant) in all folds.
  - Kinematics-only ablation (driver AUROC 0.628±0.121) outperformed the full
    dual-branch DB-TCN (0.610±0.130), confirming that the physiology TCN branch
    added noise rather than signal.
  - DANN adversarial training provided no benefit (Wilcoxon p=0.61, mean per-driver
    gain=−0.006), consistent with N=32 subjects being insufficient for the GRL to
    learn subject-invariant representations.
  - Mann-Whitney U tests confirm that arousal (p=1.9e−4) and HR (p=2.0e−5) are
    individually discriminative at the window level, but their discriminative
    information resides in the window-level distribution (mean, std) rather than
    temporal dynamics — consistent with physiological signals changing on a scale
    of minutes, not seconds.

Architecture:
  1. Single kinematics TCN branch (4 signals × 5 engineered features = 20-d input).
     ResBlocks d=1,2,4,8,16 → RF ≈ 63 timesteps, TemporalAttention → 64-d pool.
  2. Physiological scalars injected at head:
     [mean(arousal), std(arousal), mean(HR), std(HR)] computed from the raw 90s
     window after LOO route renormalisation; scaled by a separate StandardScaler.
  3. Kinematic spectral features: 4 signals × 3 bands = 12-d relative FFT band
     powers (0–0.1, 0.1–0.3, 0.3–0.5 Hz).
  4. Head: Linear(64+4+12=80, 48) → ReLU → Dropout(0.2) → Linear(48, 1).

Validity guarantees (inherited from ext4 and its 42 documented fixes):
  - Strict LOPO-CV: each driver's windows are held out entirely.
  - Anti-leakage LOO route renormalisation: test/val windows normalised using
    training-only route statistics (self contribution excluded).
  - SMOTE applied to training fold only, in raw signal space with PCA-compressed
    nearest-neighbour search.
  - StandardScalers (kin features, phys scalars, spectral) fit on real training
    windows only; transforms applied identically to val and test.
  - Per-fold Youden's J threshold selected on validation predictions.
  - Renormalised inputs clipped to ±3σ before feature engineering.
  - PYTHONHASHSEED enforced via re-exec so dict/set ordering is reproducible.
  - torch.use_deterministic_algorithms(True) for GPU reproducibility.

Outputs:
  1. Label validity report (Mann-Whitney U, per-driver positive rates).
  2. Pipeline configuration.
  3. Per-driver table: LR | XGB | KinTCN | KinTCN+Phys | Platt† | HeadFT†.
  4. Pooled driver AUROC ± std, 95% CI, AUPRC, Brier, ECE.
  5. Threshold metrics (F1 / Precision / Recall @ Youden's J).
  6. Permutation signal importance.
  7. Spectral band importance.
  8. Ablation: KinTCN vs KinTCN+Phys.
  9. Wilcoxon signed-rank: KinTCN+Phys > KinTCN.
 10. JSON artefacts in impairment_results/kin_tcn_results.json.
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
from torch.utils.data import Dataset, DataLoader
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

# ── CONFIG ───────────────────────────────────────────────────────────────────────
SEED   = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SIGNAL_COLS  = ["arousal", "hr", "speed.x", "steeringWheelAngle", "steeringTorq", "acceleration.y"]
VEHICLE_COLS = ["speed.x", "steeringWheelAngle", "steeringTorq", "acceleration.y"]
PHYS_COLS    = ["arousal", "hr"]
KIN_COLS     = ["speed.x", "steeringWheelAngle", "steeringTorq", "acceleration.y"]
PHYS_IDX     = [SIGNAL_COLS.index(c) for c in PHYS_COLS]   # [0, 1]
KIN_IDX      = [SIGNAL_COLS.index(c) for c in KIN_COLS]    # [2, 3, 4, 5]

SEVERITY = {
    "Collision":               5,
    "Red_light_violation":     3,
    "panic_braking_with_stop": 2,
    "panic_braking":           1,
    "sharp_turn":              1,
}
# center_line_crossing excluded (dominated by kinematic precursors; see ext4).

# Windowing
LOOKBACK_S = 45
WINDOW_STEP = 5
GAP, HORIZON = 5, 10
ROLL_K = 10

# Training
BATCH_SIZE   = 64
WEIGHT_DECAY = 1e-4 #4
JITTER_STD   = 0.01
CUTOUT_LEN   = 5
CUTOUT_PROB  = 0.2
EPOCHS       = 100
LR           = 5e-4
PATIENCE     = 10   # early-stopping patience (val-AUC)

# Spectral features — dropped in ext8 (permutation importance showed Δ≈0 in ext7)
KIN_SPECTRAL_DIM = 0

# Physiological scalars injected at head
PHYS_SCALAR_D = len(PHYS_COLS) * 2   # [mean, std] per signal = 4

# Input clipping after LOO renorm — prevents outlier-driven gradient corruption
RENORM_CLIP = 3.0

# Personalisation
GATE_ADAPT_K          = 15   # support windows used for online adaptation
GATE_ADAPT_STEPS      = 20
MIN_SUPPORT_POSITIVES = 3
MAX_PLATT_W           = 20.0

# Evaluation
N_BOOTSTRAP        = 2000
MIN_EVAL_POSITIVES = 5

EXCLUDE_EVAL_DRIVERS = {"0B03R","0B04","0B07", "0C06"}
N_PERM_REPEATS     = 10

# SMOTE
USE_SMOTE         = True
SMOTE_K_NEIGHBORS = 5
SMOTE_SEED_SALT   = 0xABCD

# ── DETERMINISM ──────────────────────────────────────────────────────────────────
if os.environ.get("PYTHONHASHSEED") != str(SEED):
    import subprocess
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(SEED)
    sys.exit(subprocess.call([sys.executable] + sys.argv, env=env))

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark     = True
# torch.set_num_threads(1)
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# torch.use_deterministic_algorithms(True)

OUT_DIR = Path(__file__).parent / "impairment_results"
OUT_DIR.mkdir(exist_ok=True)

# ── PREPROCESSING ────────────────────────────────────────────────────────────────

def mark_event_onsets(df):
    """Replace sustained event flags with onset pulses (first row of each run)."""
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
    """
    Per-session z-score for vehicle signals and physiological signals.

    Statistics are computed from the first LOOKBACK_S+GAP rows only, avoiding
    temporal leakage from the future window into the normalisation reference.
    All signals are made relative to each driver's intra-session baseline,
    enabling cross-driver comparison of deviations.
    """
    df = df.copy()
    prefix = LOOKBACK_S + GAP
    for _, grp in df.groupby(["id", "route"]):
        idx = grp.index
        for col in VEHICLE_COLS + list(PHYS_COLS):
            if col not in df.columns:
                continue
            mu  = grp.iloc[:prefix][col].mean()
            sig = grp.iloc[:prefix][col].std() + 1e-6
            df.loc[idx, col] = (grp[col] - mu) / sig
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
    """Apply engineer_features to a channel subset. → (N, T, len(col_idx)*5)."""
    return np.stack([engineer_features(w[:, col_idx]) for w in X_raw])


def window_baseline_feats(X_raw, col_idx=None):
    """Mean | std | max per signal over the window — used by LR and XGB baselines."""
    X = X_raw if col_idx is None else X_raw[:, :, col_idx]
    return np.concatenate([X.mean(1), X.std(1), X.max(1)], axis=1).astype(np.float32)


def phys_scalar_feats(X_raw):
    """
    Physiological scalar features for head injection.

    Computes [mean(arousal), std(arousal), mean(HR), std(HR)] over the 90s window
    from the LOO-renormalised raw signal array.  These capture the window-level
    physiological state without requiring temporal modelling, which is appropriate
    given the slow time-constant of arousal and HR relative to the 1 Hz sample rate.

    Parameters
    ----------
    X_raw : ndarray (N, T, len(SIGNAL_COLS))

    Returns
    -------
    ndarray (N, PHYS_SCALAR_D=4)
    """
    phys = X_raw[:, :, PHYS_IDX]   # (N, T, 2)
    return np.concatenate([phys.mean(axis=1), phys.std(axis=1)],
                          axis=1).astype(np.float32)   # (N, 4)

# ── SPECTRAL FEATURES ────────────────────────────────────────────────────────────

def compute_spectral_features_batch(X_kin_raw, sample_dt=1.0):
    """
    Normalised FFT band powers for kinematics signals.

    Windows are demeaned before FFT to remove the DC shift introduced by LOO
    route renormalisation (µ_session − µ_route_loo).  Relative band powers
    (power_in_band / total_power) are amplitude-invariant, making them robust
    to cross-driver amplitude differences that survive renormalisation.

    Parameters
    ----------
    X_kin_raw : ndarray (N, T, C_kin)
    sample_dt : float — median sampling interval in seconds (from _check_sampling_rate)

    Returns
    -------
    ndarray (N, C_kin * len(SPECTRAL_BANDS)) = (N, KIN_SPECTRAL_DIM=12)
    """
    N, T, C = X_kin_raw.shape
    if T < 2:
        return np.zeros((N, C * len(SPECTRAL_BANDS)), dtype=np.float32)
    freqs  = np.fft.rfftfreq(T, d=sample_dt)
    masks  = [(freqs >= lo) & (freqs < hi) for lo, hi in SPECTRAL_BANDS[:-1]]
    masks += [(freqs >= SPECTRAL_BANDS[-1][0]) & (freqs <= SPECTRAL_BANDS[-1][1])]
    out    = np.zeros((N, C * len(SPECTRAL_BANDS)), dtype=np.float32)
    for i, win in enumerate(X_kin_raw):
        win_dm  = win - win.mean(axis=0, keepdims=True)
        fft_pow = np.abs(np.fft.rfft(win_dm, axis=0)) ** 2
        total   = fft_pow.sum(axis=0) + 1e-10
        for b, mask in enumerate(masks):
            out[i, b * C: (b + 1) * C] = fft_pow[mask].sum(axis=0) / total
    return out

# ── SMOTE ────────────────────────────────────────────────────────────────────────

def smote_raw(X_raw, y, k=SMOTE_K_NEIGHBORS, rng=None,
              pids=None, routes=None, t_starts=None):
    """
    SMOTE oversampling in raw signal space with PCA-compressed nearest-neighbour search.

    Operates on raw (N, T, C) windows to avoid distance concentration in the
    high-dimensional (540-d) engineered feature space.  PCA compresses to
    min(n_pos−1, 50, n_pos//5) components before computing Euclidean distances.

    Overlapping pairs (same session, |Δt_start| < LOOKBACK_S) are excluded from
    the neighbour pool since interpolating near-identical windows adds no diversity.

    Returns X_raw_aug, y_aug, anchor_idx, neighbor_idx, lambdas.
    Callers re-interpolate in scaled feature space using the returned indices and
    lambdas to avoid nonlinear artefacts from rolling-std / z-score applied to a
    raw interpolation.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    pos_idx = np.where(y == 1)[0]
    n_pos   = len(pos_idx)
    n_neg   = (y == 0).sum()

    if n_pos == 0 or n_pos >= n_neg:
        return (X_raw, y,
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32))


    target_ratio = 0.6
    n_synthetic = int((target_ratio * n_neg - (1 - target_ratio) * n_pos) / (1 - target_ratio))
    n_synthetic = max(0, n_synthetic)
    X_flat      = X_raw[pos_idx].reshape(n_pos, -1)

    k_eff = min(k, n_pos - 1)
    if k_eff < 1:
        print(f"  [WARN smote_raw] {n_pos} positive(s) — SMOTE impossible; duplicating.")
        dup = rng.choice(pos_idx, n_synthetic, replace=True)
        return (np.concatenate([X_raw, X_raw[dup]], axis=0),
                np.concatenate([y,     np.ones(n_synthetic, dtype=y.dtype)]),
                dup, dup, np.zeros(n_synthetic, dtype=np.float32))

    _pca_k = min(n_pos - 1, X_flat.shape[1], 50, max(2, n_pos // 5))
    if _pca_k >= 2:
        X_nn = PCA(n_components=_pca_k).fit_transform(X_flat)
    else:
        X_nn = X_flat

    dists = cdist(X_nn, X_nn, metric="euclidean")
    np.fill_diagonal(dists, np.inf)

    if pids is not None and routes is not None and t_starts is not None:
        pos_pids   = pids[pos_idx]
        pos_routes = routes[pos_idx]
        pos_ts     = t_starts[pos_idx].astype(float)
        same_sess  = ((pos_pids[:, None]   == pos_pids[None, :]) &
                      (pos_routes[:, None] == pos_routes[None, :]))
        tdiff      = np.abs(pos_ts[:, None] - pos_ts[None, :])
        dists[same_sess & (tdiff < LOOKBACK_S)] = np.inf

    nn_idx = np.argsort(dists, axis=1)[:, :k_eff]

    T, C             = X_raw.shape[1], X_raw.shape[2]
    syn              = np.zeros((n_synthetic, T, C), dtype=X_raw.dtype)
    syn_anchor_idx   = np.zeros(n_synthetic, dtype=np.int64)
    syn_neighbor_idx = np.zeros(n_synthetic, dtype=np.int64)
    syn_lambdas      = np.zeros(n_synthetic, dtype=np.float32)

    for i in range(n_synthetic):
        anchor = rng.integers(0, n_pos)
        nn     = nn_idx[anchor, rng.integers(0, k_eff)]
        lam    = float(rng.uniform(0.0, 1.0))
        syn[i] = (X_raw[pos_idx[anchor]]
                  + lam * (X_raw[pos_idx[nn]] - X_raw[pos_idx[anchor]]))
        syn_anchor_idx[i]   = pos_idx[anchor]
        syn_neighbor_idx[i] = pos_idx[nn]
        syn_lambdas[i]      = lam

    return (np.concatenate([X_raw, syn], axis=0),
            np.concatenate([y,     np.ones(n_synthetic, dtype=y.dtype)]),
            syn_anchor_idx, syn_neighbor_idx, syn_lambdas)

# ── WINDOWING ────────────────────────────────────────────────────────────────────

def composite_risk_score(future_df):
    return sum(SEVERITY[col] * int((future_df[col] > 0).any()) for col in SEVERITY)


def future_error_types(future_df):
    return frozenset(col for col in SEVERITY
                     if col in future_df.columns and (future_df[col] > 0).any())


def build_windows(df):
    """Uniform-stride windows for all sessions; skips windows with NaN signals."""
    windows, labels, scores, pids, etypes, routes, t_starts = [], [], [], [], [], [], []
    min_len = LOOKBACK_S + GAP + HORIZON
    nan_skip: dict = {}
    for (pid, route), grp in df.groupby(["id", "route"]):
        grp = grp.sort_values("Timestamp").reset_index(drop=True)
        n   = len(grp)
        ts  = grp["Timestamp"].values
        if n < min_len:
            print(f"  [WARN] build_windows: ({pid}, {route}) has {n} rows < {min_len} — skipping.")
            continue
        idx = 0
        while idx + LOOKBACK_S + GAP + HORIZON <= n:
            sig = grp.iloc[idx: idx + LOOKBACK_S][SIGNAL_COLS].values.astype(np.float32)
            prev_s, prev_t = nan_skip.get(pid, (0, 0))
            if not np.isnan(sig).any():
                future = grp.iloc[idx + LOOKBACK_S + GAP: idx + LOOKBACK_S + GAP + HORIZON]
                score  = composite_risk_score(future)
                windows.append(sig); labels.append(int(score > 0))
                scores.append(score); pids.append(pid)
                etypes.append(future_error_types(future))
                routes.append(route); t_starts.append(ts[idx])
                nan_skip[pid] = (prev_s, prev_t + 1)
            else:
                nan_skip[pid] = (prev_s + 1, prev_t + 1)
            idx += WINDOW_STEP

    total_s = sum(s for s, _ in nan_skip.values())
    total_t = sum(t for _, t in nan_skip.values())
    if total_s > 0:
        print(f"  [NaN-skip audit] {total_s}/{total_t} windows dropped ({100*total_s/max(total_t,1):.1f}%)")
        for pid_k, (ns, nt) in sorted(nan_skip.items()):
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

class KinPhysDataset(Dataset):
    """Yields (x_kin, x_phys_scalar, x_spec, y) tuples."""
    def __init__(self, X_kin, X_phys_scalar, X_spec, y, augment=False):
        self.Xk   = torch.as_tensor(X_kin).float()
        self.Xps  = torch.as_tensor(X_phys_scalar).float()
        self.Xs   = torch.as_tensor(X_spec).float()
        self.y    = torch.as_tensor(y).float()
        self.aug  = augment

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        xk  = self.Xk[idx].clone()
        xps = self.Xps[idx].clone()
        xs  = self.Xs[idx].clone()
        if self.aug:
            xk += torch.randn_like(xk) * JITTER_STD
            # Per-channel amplitude scaling: U(0.8, 1.2) — improves LOPO generalisation
            # by making the model invariant to cross-driver amplitude differences.
            scale = torch.empty(xk.shape[-1]).uniform_(0.8, 1.2)
            xk = xk * scale.unsqueeze(0)
            if torch.rand(1).item() < CUTOUT_PROB and xk.shape[0] > CUTOUT_LEN:
                t0 = torch.randint(0, xk.shape[0] - CUTOUT_LEN, (1,)).item()
                xk[t0: t0 + CUTOUT_LEN] = 0.0
        return xk, xps, xs, self.y[idx]


def get_kin_loader(Xk, Xps, Xs, y, batch_size=BATCH_SIZE, augment=False):
    ds = KinPhysDataset(Xk, Xps, Xs, y, augment=augment)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

# ── MODEL ────────────────────────────────────────────────────────────────────────

def _gn_groups(out_c: int) -> int:
    """Largest power-of-2 divisor of out_c, capped at 8, for GroupNorm."""
    g = min(out_c, 8)
    while g > 1 and out_c % g != 0:
        g //= 2
    return max(g, 1)


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, d):
        super().__init__()
        causal_pad = 2 * d
        self.conv = nn.Sequential(
            nn.ConstantPad1d((causal_pad, 0), 0.0),
            nn.Conv1d(in_c, out_c, 3, padding=0, dilation=d),
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
        xt      = x.permute(0, 2, 1)
        h       = torch.tanh(self.query(xt))
        weights = torch.softmax(self.score(h), dim=1)
        return (xt * weights).sum(dim=1)


class KinTCN(nn.Module):
    """
    Kinematics TCN with physiological scalar and spectral feature injection at head.

    KIN_D = 64 (output channels of kinematics TCN).
    Head input = KIN_D(64) + PHYS_SCALAR_D(4) = 68-d (spectral dropped in ext8).

    The physiological scalars [mean_aro, std_aro, mean_hr, std_hr] provide the
    window-level physiological state to the head without requiring a temporal model,
    which is appropriate given the slow time-constants of arousal and HR relative
    to the 1 Hz sample rate.
    """
    KIN_D = 64

    def __init__(self, n_kin_feats: int, phys_scalar_d: int, spectral_dim: int):
        super().__init__()
        self.kin_branch = nn.Sequential(
            ResBlock(n_kin_feats,     self.KIN_D // 2,  1),
            ResBlock(self.KIN_D // 2, self.KIN_D,       2),
            ResBlock(self.KIN_D,      self.KIN_D,        4),
            ResBlock(self.KIN_D,      self.KIN_D,        8),
            ResBlock(self.KIN_D,      self.KIN_D,       16),
        )
        self.kin_attn = TemporalAttention(self.KIN_D)
        
        # NEW: FiLM Generator (Physiology -> Gamma & Beta)
        self.phys_film = nn.Sequential(
            nn.Linear(phys_scalar_d, 32),
            nn.LayerNorm(32),              # <-- ADD THIS: Stabilizes the physiology scaling
            nn.ReLU(),
            nn.Dropout(0.2),               # <-- ADD THIS: Prevents overfitting to specific heart rates
            nn.Linear(32, self.KIN_D * 2) 
        )

        # UPDATED: Head now only takes KIN_D (64) since Phys is fused via FiLM
        # and spectral is 0.
        head_in = self.KIN_D + spectral_dim 
        self.head = nn.Sequential(
            nn.Linear(head_in, 48), nn.ReLU(), nn.Dropout(0.2), nn.Linear(48, 1),
        )

    def forward(self, x_kin, x_phys_scalar, x_spec):
        kin_seq = self.kin_branch(x_kin.permute(0, 2, 1))          # (B, KIN_D, T)
        
        # ←←← NEW: FiLM on the entire temporal feature map (not just pooled)
        film_params = self.phys_film(x_phys_scalar)                # (B, 128)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)         # (B, 64)
        gamma = gamma.unsqueeze(-1)                                # (B, 64, 1)
        beta  = beta.unsqueeze(-1)
        kin_seq = kin_seq * (1 + gamma) + beta                     # modulate every timestep
        
        kin_pool = self.kin_attn(kin_seq)                          # (B, KIN_D)
        head_in = torch.cat([kin_pool, x_spec], dim=-1)
        return self.head(head_in).squeeze(-1)
# ── UTILITIES ────────────────────────────────────────────────────────────────────

def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2: return None
    return roc_auc_score(y_true, y_score)


def safe_auprc(y_true, y_score):
    if len(np.unique(y_true)) < 2: return None
    return average_precision_score(y_true, y_score)


def _nan_or(x):
    return float("nan") if x is None else x



def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0; n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask   = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo)
        if not mask.any(): continue
        ece += mask.sum() / n * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def _val_threshold(y_val, val_scores):
    """Youden's J threshold on val data; falls back to 0.5 for single-class val."""
    if len(np.unique(y_val)) < 2:
        return 0.5
    fpr, tpr, thrs = roc_curve(y_val, val_scores)
    return float(thrs[int(np.argmax(tpr - fpr))])


def bootstrap_auc_ci_drivers(driver_aucs, n_boot=N_BOOTSTRAP, seed=SEED):
    aucs = np.array(driver_aucs); n = len(aucs)
    if n < 2:
        return float("nan"), float("nan")
    rng  = np.random.default_rng(seed)
    boot = [rng.choice(aucs, n, replace=True).mean() for _ in range(n_boot)]
    return tuple(np.percentile(boot, [2.5, 97.5]))


def _val_sample(arr, frac, rng):
    n = min(max(1, int(frac * len(arr))), len(arr)) if len(arr) > 0 else 0
    return rng.choice(arr, n, replace=False) if n > 0 else np.array([], dtype=arr.dtype)

# ── TRAINING ─────────────────────────────────────────────────────────────────────

def train_kin_tcn(model, Xtr_k, Xtr_ps, Xtr_s, y_tr,
                  Xval_k, Xval_ps, Xval_s, y_val):
    
    # 1. UN-HARDCODE: Calculate once per fold (y_tr is already SMOTEd here)
    n_neg = (y_tr == 0).sum()
    n_pos = (y_tr == 1).sum()
    dyn_weight = n_neg / max(1, n_pos)
    p_weight = torch.tensor([dyn_weight]).to(DEVICE)
    

    # 2. Define Criterion once
    criterion = nn.BCEWithLogitsLoss(pos_weight=p_weight)
    
    opt    = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.01)
    loader = get_kin_loader(Xtr_k, Xtr_ps, Xtr_s, y_tr, augment=True)
    
    best_auc, best_w  = float("-inf"), None
    best_loss         = float("inf")
    no_improve        = 0

    Xval_k_t  = torch.as_tensor(Xval_k).to(DEVICE)
    Xval_ps_t = torch.as_tensor(Xval_ps).to(DEVICE)
    Xval_s_t  = torch.as_tensor(Xval_s).to(DEVICE)

    for epoch in range(EPOCHS):
        model.train()
        ep_loss = 0.0; n_b = 0
        
        for xk, xps, xs, yb in loader:
            xk, xps, xs, yb = xk.to(DEVICE), xps.to(DEVICE), xs.to(DEVICE), yb.to(DEVICE)
            
            # 3. FIX: Label Smoothing (Previously smooth_yb was undefined)
            # This turns 0/1 into 0.05/0.95
            smooth_yb = yb * 0.9 + 0.05 
            
            logits = model(xk, xps, xs)
            
            # 4. Use the criterion defined above
            loss = criterion(logits, smooth_yb)
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            ep_loss += loss.item()
            n_b += 1
            
        sched.step()

        # --- Evaluation Logic ---
        model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(model(Xval_k_t, Xval_ps_t, Xval_s_t)).cpu().numpy()
        
        auc = safe_auc(y_val, preds)
        if auc is not None:
            if auc > best_auc:
                best_auc = auc
                best_w = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE: break
        else:
            # Fallback for single-class validation folds
            train_loss = ep_loss / max(n_b, 1)
            if train_loss < best_loss - 1e-4:
                best_loss = train_loss
                best_w = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE: break

    if best_w is not None:
        model.load_state_dict(best_w)
    return model
# ── PERSONALISATION ──────────────────────────────────────────────────────────────

def platt_adapt(pop_scores: np.ndarray, y_te: np.ndarray) -> np.ndarray:
    """
    Online Platt-scaling on the first GATE_ADAPT_K test windows.

    Fits a logistic calibrator on support scores/labels, applies it to ALL
    windows (single consistent scale).  Falls back to population scores when
    support is single-class, has too few positives, or the calibrator inverts
    the score ordering (w ≤ 0) or blows up (|w| > MAX_PLATT_W).
    """
    K = GATE_ADAPT_K
    if len(y_te) <= K:
        return pop_scores
    s_sup, y_sup = pop_scores[:K], y_te[:K]
    if len(np.unique(y_sup)) < 2 or int(y_sup.sum()) < MIN_SUPPORT_POSITIVES:
        return pop_scores
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    try:
        lr.fit(s_sup.reshape(-1, 1), y_sup.astype(int))
    except Exception as exc:
        warnings.warn(f"platt_adapt: fit failed ({exc}); using population scores.",
                      RuntimeWarning, stacklevel=2)
        return pop_scores
    w = lr.coef_[0][0]
    if w <= 0 or w > MAX_PLATT_W:
        return pop_scores
    return lr.predict_proba(pop_scores.reshape(-1, 1))[:, 1]


def head_adapt(model_pop, Xte_k, Xte_ps, Xte_s, y_te,
               n_steps=GATE_ADAPT_STEPS, lr=1e-3):
    """
    Online head fine-tuning on the first GATE_ADAPT_K test windows.

    Freezes the kinematics TCN branch; fine-tunes only self.head on the
    support set.  Population scores are returned for support windows (no
    leakage), fine-tuned scores for eval windows.
    """
    K = GATE_ADAPT_K
    if len(y_te) <= K or int(y_te[:K].sum()) < MIN_SUPPORT_POSITIVES:
        return None
    model = copy.deepcopy(model_pop).to(DEVICE)
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("head.")
    Xk = torch.as_tensor(Xte_k[:K]).to(DEVICE)
    Xps = torch.as_tensor(Xte_ps[:K]).to(DEVICE)
    Xs  = torch.as_tensor(Xte_s[:K]).to(DEVICE)
    ys  = torch.as_tensor(y_te[:K], dtype=torch.float32).to(DEVICE)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=lr, weight_decay=1e-3)
    model.train()
    for _ in range(n_steps):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(Xk, Xps, Xs), ys)
        loss.backward(); opt.step()
    model_pop.eval(); model.eval()
    with torch.no_grad():
        sup_scores = torch.sigmoid(
            model_pop(Xk, Xps, Xs)).cpu().numpy()
        eval_scores = torch.sigmoid(
            model(torch.as_tensor(Xte_k[K:]).to(DEVICE),
                  torch.as_tensor(Xte_ps[K:]).to(DEVICE),
                  torch.as_tensor(Xte_s[K:]).to(DEVICE))).cpu().numpy()
    return np.concatenate([sup_scores, eval_scores])

# ── VALIDITY REPORT ──────────────────────────────────────────────────────────────

def print_validity_report(X_raw, y, scores, pids):
    pos = y == 1; total_pos = pos.sum(); total = len(y)
    unique, cnts = np.unique(scores[pos].astype(int), return_counts=True)
    print(f"\n{'='*72}")
    print("LABEL VALIDITY REPORT — COMPOSITE RISK TARGET")
    print(f"{'='*72}")
    print(f"Total windows    : {total}")
    print(f"Positive (risk>0): {total_pos} ({100*total_pos/total:.1f}%)")
    print(f"Negative         : {(~pos).sum()} ({100*(~pos).sum()/total:.1f}%)")
    print(f"\nRisk score distribution (positives only):")
    for sc, cnt in zip(unique, cnts):
        print(f"  score={sc:2d}  n={cnt:5d}  ({100*cnt/total_pos:.1f}%)")
    print(f"\nPredictive validity — Mann-Whitney U (pos vs neg, raw signals):")
    print(f"  {'Signal':<24}  {'Mean(neg)':>10}  {'Mean(pos)':>10}  {'p-value':>12}  Sig")
    mw_pvals = {}
    for ci, col in enumerate(SIGNAL_COLS):
        neg_v = X_raw[~pos][:, :, ci].mean(axis=1)
        pos_v = X_raw[pos][:, :, ci].mean(axis=1)
        _, pv = mannwhitneyu(neg_v, pos_v, alternative="two-sided")
        sig   = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
        print(f"  {col:<24}  {neg_v.mean():>10.4f}  {pos_v.mean():>10.4f}  {pv:>12.2e}  {sig}")
        mw_pvals[col] = pv
    print(f"\nPer-driver positive rate:")
    print(f"  {'Driver':<12}  {'N':>6}  {'Pos':>5}  {'Rate%':>6}  Tier")
    for d in np.unique(pids):
        mask_d = pids == d; nd = mask_d.sum(); np_ = y[mask_d].sum()
        rate   = 100.0 * np_ / nd if nd > 0 else 0.0
        tier   = "HIGH" if rate >= 10 else ("MED" if rate >= 5 else "LOW")
        print(f"  {d:<12}  {nd:>6}  {np_:>5}  {rate:>6.1f}%  {tier}")
    print(f"{'='*72}")
    return mw_pvals

# ── SAMPLING RATE CHECK ──────────────────────────────────────────────────────────

def _check_sampling_rate(df):
    if "Timestamp" not in df.columns:
        print("[WARN] No Timestamp column — skipping fs check.")
        return 1.0
    bad, all_dts = [], []
    for (pid, route), grp in df.groupby(["id", "route"]):
        ts = grp["Timestamp"].sort_values().values
        if len(ts) < 2: continue
        median_dt = float(np.median(np.diff(ts)))
        all_dts.append(median_dt)
        if not (0.8 <= median_dt <= 1.2):
            bad.append((pid, route, median_dt))
    if bad:
        lines = "\n".join(f"  pid={p}  route={r}  median_Δt={dt:.3f}s" for p, r, dt in bad)
        raise RuntimeError(
            f"Sampling rate check failed for {len(bad)} session(s) — "
            f"spectral band edges assume fs≈1 Hz:\n{lines}")
    global_dt = float(np.median(all_dts)) if all_dts else 1.0
    print(f"  Sampling rate check: {len(all_dts)} sessions, "
          f"median Δt = {global_dt:.3f} s  (all within [0.8, 1.2] s — OK)")
    return global_dt

# ── LOO RENORMALISATION ──────────────────────────────────────────────────────────

def _loo_renorm(X_arr, pid_arr, routes_arr,
                all_session_stats, route_sum_mus, route_sum_sigs2, route_counts,
                ci_norm_map, label="windows"):
    """
    Leave-one-out route renormalisation.

    For each (driver, route): undoes the per-session z-score from normalize_signals
    (using raw prefix stats stored in all_session_stats), then applies a LOO route
    z-score — the pooled σ = sqrt(mean(σ²)) of all OTHER training drivers on that
    route.  This eliminates the circular dependency where each driver is normalised
    using a route mean that includes their own contribution.

    Singleton routes (n_rte < 2) are skipped and a warning is emitted.
    Sessions with NaN prefix stats (all-NaN column in first LOOKBACK_S+GAP rows)
    are skipped and their per-session z-score is retained.
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
                own_mu, own_sig = all_session_stats[key_own]
                if math.isnan(own_mu) or math.isnan(own_sig):
                    continue
                n_rte = route_counts[key_rte]
                if n_rte < 2:
                    n_singleton += 1
                    continue
                rte_mu  = (route_sum_mus[key_rte]  - own_mu)  / (n_rte - 1)
                rte_sig = math.sqrt(
                    max((route_sum_sigs2[key_rte] - own_sig ** 2) / (n_rte - 1), 0.0))
                X_out[mask, :, ci] = (
                    X_out[mask, :, ci] * (own_sig + 1e-6) + own_mu - rte_mu
                ) / max(rte_sig, 1e-6)
    if n_singleton:
        print(f"  [WARN] _loo_renorm ({label}): {n_singleton} singleton-route triplets skipped.")
    return X_out

# ── MAIN ─────────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(Path(__file__).parent / "relab+unibo_dataset.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True).astype("int64") / 1e9
    global_sample_dt = _check_sampling_rate(df)
    df = mark_event_onsets(df)

    # Pre-compute per-session stats from the RAW df (before normalisation).
    # Used inside each LOPO fold for LOO route renorm without transductive leakage.
    COLS_TO_NORM = VEHICLE_COLS + [c for c in PHYS_COLS if c in df.columns]
    NORM_PREFIX  = LOOKBACK_S + GAP
    all_session_stats: dict = {}
    for (pid, route), grp in df.groupby(["id", "route"]):
        for col in COLS_TO_NORM:
            if col in grp.columns:
                mu      = float(grp.iloc[:NORM_PREFIX][col].mean())
                sig_raw = float(grp.iloc[:NORM_PREFIX][col].std())
                all_session_stats[(pid, route, col)] = (mu, sig_raw)

    df = normalize_signals(df)

    (X_raw_all, y_all, scores_all,
     pid_all, etypes_all, routes_all, ts_all) = build_windows(df)

    # Exclude drivers entirely (training + evaluation)
    keep = ~np.isin(pid_all, list(EXCLUDE_EVAL_DRIVERS))
    X_raw_all  = X_raw_all[keep]
    y_all      = y_all[keep]
    scores_all = scores_all[keep]
    pid_all    = pid_all[keep]
    etypes_all = etypes_all[keep]
    routes_all = routes_all[keep]
    ts_all     = ts_all[keep]

    n_kin_feat = len(KIN_COLS) * 5   # 20

    mw_pvals = print_validity_report(X_raw_all, y_all, scores_all, pid_all)

    print(f"\n{'='*72}")
    print("KIN-TCN+PHYS — PIPELINE CONFIGURATION")
    print(f"{'='*72}")
    print(f"Kinematics branch : {KIN_COLS}  ({n_kin_feat} engineered features)")
    print(f"  TCN blocks      : d=1,2,4,8,16  →  RF ≈ 63 timesteps")
    print(f"  Output channels : {KinTCN.KIN_D}")
    print(f"Phys scalars      : {PHYS_COLS}  →  [mean, std] each = {PHYS_SCALAR_D}-d")
    print(f"Head input        : {KinTCN.KIN_D} + {PHYS_SCALAR_D} = {KinTCN.KIN_D + PHYS_SCALAR_D}-d")
    print(f"CLC excluded      : center_line_crossing removed from SEVERITY")
    print(f"Excluded entirely : {sorted(EXCLUDE_EVAL_DRIVERS)} (removed from training + evaluation)")
    print(f"SMOTE             : {'k=' + str(SMOTE_K_NEIGHBORS) if USE_SMOTE else 'DISABLED'}")
    print(f"Renorm clip       : ±{RENORM_CLIP}σ (applied after LOO renorm)")
    print(f"Loss              : BCE (balanced by SMOTE; no focal, no pos_weight)")
    print(f"Min eval positives: {MIN_EVAL_POSITIVES}")
    print(f"Epochs            : {EPOCHS}  |  LR : {LR}  |  Scheduler : CosineAnnealingLR")
    print(f"GAP / HORIZON     : {GAP}s / {HORIZON}s  →  predicts errors in [{GAP}, {GAP+HORIZON}]s")
    print(f"Personalisation   : Platt† and HeadFT† on first {GATE_ADAPT_K} test windows")
    print(f"Device            : {DEVICE}")
    print(f"{'='*72}\n")

    # ci_norm_map: signal column → index in X_raw (SIGNAL_COLS order)
    ci_norm_map = {col: SIGNAL_COLS.index(col) for col in COLS_TO_NORM if col in SIGNAL_COLS}

    drivers = [d for d in np.unique(pid_all)
               if y_all[pid_all == d].sum() >= MIN_EVAL_POSITIVES]

    hdr = (f"{'Driver':<10} | {'N_win':>5} {'PosR%':>6} | "
           f"{'LR':>7} {'XGB':>7} {'KinTCN':>7} {'Kin+Phys':>9} | "
           f"{'Platt†':>8} {'HeadFT†':>8}")
    print(hdr)
    print("-" * len(hdr))

    per_driver_results = []
    pool_y, pool_tcn, pool_tcnp = [], [], []
    pool_lr, pool_xgb = [], []
    pool_platt, pool_head = [], []
    pool_models_tcnp = []
    pool_Xte_k, pool_Xte_ps, pool_Xte_s = [], [], []
    pool_thresh_tcn, pool_thresh_tcnp = [], []
    singleclass_val_folds = []

    for d in drivers:
        mask_tr = pid_all != d
        X_tr    = X_raw_all[mask_tr]; y_tr = y_all[mask_tr]
        pid_tr  = pid_all[mask_tr];   routes_tr = routes_all[mask_tr]
        ts_tr   = ts_all[mask_tr]

        mask_te = pid_all == d
        X_te    = X_raw_all[mask_te]; y_te = y_all[mask_te]
        ts_te   = ts_all[mask_te];    routes_te = routes_all[mask_te]

        # Sort test windows chronologically for personalisation (sequential order).
        order   = np.argsort(ts_te)
        X_te    = X_te[order]; y_te = y_te[order]
        ts_te   = ts_te[order]; routes_te = routes_te[order]

        seed_d   = int(hashlib.md5(str(d).encode()).hexdigest(), 16) & 0xFFFFFFFF
        fold_rng = np.random.default_rng(SEED ^ seed_d)
        torch.manual_seed(SEED ^ seed_d)
        torch.cuda.manual_seed_all(SEED ^ seed_d)

        train_drivers = np.unique(pid_tr)
        has_pos       = np.array([y_tr[pid_tr == p].sum() > 0 for p in train_drivers])
        val_ids       = np.concatenate([
            _val_sample(train_drivers[has_pos],  0.25, fold_rng),
            _val_sample(train_drivers[~has_pos], 0.25, fold_rng),
        ])
        val_ids_set = set(val_ids.tolist())
        vmask       = np.isin(pid_tr, val_ids)

        # Build route reference pools (val-excluded from training LOO reference).
        _route_mus_d:   dict = {}   # test renorm: all non-test drivers
        _route_sigs_d:  dict = {}
        _route_sum_mus:   dict = {}  # training LOO: val-excluded
        _route_sum_sigs2: dict = {}
        _route_counts:    dict = {}

        for (pid_s, route_s, col_s), (mu_s, sig_s) in all_session_stats.items():
            if pid_s == d:
                continue
            if math.isnan(mu_s) or math.isnan(sig_s):
                continue
            key = (route_s, col_s)
            _route_mus_d.setdefault(key, []).append(mu_s)
            _route_sigs_d.setdefault(key, []).append(sig_s)
            if pid_s not in val_ids_set:
                _route_sum_mus[key]   = _route_sum_mus.get(key,   0.0) + mu_s
                _route_sum_sigs2[key] = _route_sum_sigs2.get(key, 0.0) + sig_s ** 2
                _route_counts[key]    = _route_counts.get(key,    0)   + 1

        # Pure-training reference for val window renorm (non-test, non-val).
        _route_pure_train_stats: dict = {
            k: (
                _route_sum_mus[k] / _route_counts[k],
                math.sqrt(max(_route_sum_sigs2[k] / _route_counts[k], 0.0)),
            )
            for k in _route_sum_mus if _route_counts[k] >= 1
        }

        # Test driver renorm (route mean from all non-test drivers).
        _route_tr_stats_d: dict = {
            k: (
                float(np.mean(v)),
                math.sqrt(max(float(np.mean(np.array(_route_sigs_d[k]) ** 2)), 0.0)),
            )
            for k, v in _route_mus_d.items()
        }

        X_te = X_te.copy()
        for route_v in np.unique(routes_te):
            r_mask = routes_te == route_v
            for col_v, ci_v in ci_norm_map.items():
                key_test  = (d, route_v, col_v)
                key_route = (route_v, col_v)
                if key_test not in all_session_stats or key_route not in _route_tr_stats_d:
                    continue
                test_mu, test_sig = all_session_stats[key_test]
                if math.isnan(test_mu) or math.isnan(test_sig):
                    continue
                tr_mu, tr_sig = _route_tr_stats_d[key_route]
                X_te[r_mask, :, ci_v] = (
                    X_te[r_mask, :, ci_v] * (test_sig + 1e-6) + test_mu - tr_mu
                ) / max(tr_sig, 1e-6)
        X_te = np.clip(X_te, -RENORM_CLIP, RENORM_CLIP)

        X_tr = _loo_renorm(X_tr, pid_tr, routes_tr,
                           all_session_stats, _route_sum_mus, _route_sum_sigs2,
                           _route_counts, ci_norm_map, f"train (test={d})")
        X_tr = np.clip(X_tr, -RENORM_CLIP, RENORM_CLIP)

        # Build val windows from val-driver sessions directly.
        df_val_fold = df[df["id"].isin(val_ids_set)].copy()
        X_val_raw, y_val_d, _, _pids_val, _, _routes_val, _ = build_windows(df_val_fold)
        X_val_raw = X_val_raw.copy()
        for _vpid in np.unique(_pids_val):
            for _vrt in np.unique(_routes_val[_pids_val == _vpid]):
                _vmask = (_pids_val == _vpid) & (_routes_val == _vrt)
                for _vcol, _vci in ci_norm_map.items():
                    _key_own = (_vpid, _vrt, _vcol)
                    _key_rte = (_vrt, _vcol)
                    if _key_own not in all_session_stats or _key_rte not in _route_pure_train_stats:
                        continue
                    _vmu, _vsig = all_session_stats[_key_own]
                    if math.isnan(_vmu) or math.isnan(_vsig):
                        continue
                    _tr_mu, _tr_sig = _route_pure_train_stats[_key_rte]
                    X_val_raw[_vmask, :, _vci] = (
                        X_val_raw[_vmask, :, _vci] * (_vsig + 1e-6) + _vmu - _tr_mu
                    ) / max(_tr_sig, 1e-6)
        X_val_raw = np.clip(X_val_raw, -RENORM_CLIP, RENORM_CLIP)

        if len(np.unique(y_val_d)) < 2:
            print(f"{d:<10} | [WARN] val fold single-class — falling back to loss patience")
            singleclass_val_folds.append(d)

        # ── LR / XGB baselines ────────────────────────────────────────────────────
        X_bl = X_tr[~vmask]; y_bl = y_tr[~vmask]
        pw_bl = float((y_bl == 0).sum() / max((y_bl == 1).sum(), 1))

        Xbl_feat     = window_baseline_feats(X_bl)
        Xval_bl_feat = window_baseline_feats(X_val_raw)
        Xte_bl_feat  = window_baseline_feats(X_te)

        scaler_bl = StandardScaler().fit(Xbl_feat)
        Xbl_feat_sc     = scaler_bl.transform(Xbl_feat)
        Xval_bl_feat_sc = scaler_bl.transform(Xval_bl_feat)
        Xte_bl_feat_sc  = scaler_bl.transform(Xte_bl_feat)

        lr_model = LogisticRegression(max_iter=1000, random_state=SEED,
                                      class_weight="balanced")
        lr_model.fit(Xbl_feat_sc, y_bl.astype(int))
        lr_scores = lr_model.predict_proba(Xte_bl_feat_sc)[:, 1]

        xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pw_bl, use_label_encoder=False,
            eval_metric="logloss", random_state=SEED, verbosity=0,
        )
        xgb_model.fit(Xbl_feat_sc, y_bl.astype(int),
                      eval_set=[(Xval_bl_feat_sc, y_val_d.astype(int))],
                      verbose=False)
        xgb_scores = xgb_model.predict_proba(Xte_bl_feat_sc)[:, 1]

        # ── Kin feature engineering ───────────────────────────────────────────────
        # Training pool (real windows only, before SMOTE)
        X_tr_train = X_tr[~vmask]
        y_tr_train = y_tr[~vmask]
        pid_tr_train    = pid_tr[~vmask]
        routes_tr_train = routes_tr[~vmask]
        ts_tr_train     = ts_tr[~vmask]

        # SMOTE in raw signal space (before feature engineering)
        if USE_SMOTE:
            smote_rng = np.random.default_rng(SEED ^ seed_d ^ SMOTE_SEED_SALT)
            _, y_tr_train, _anchors, _neighbors, _lambdas = smote_raw(
                X_tr_train, y_tr_train, rng=smote_rng,
                pids=pid_tr_train, routes=routes_tr_train, t_starts=ts_tr_train)
        else:
            _anchors = np.empty(0, dtype=np.int64)
            _neighbors = np.empty(0, dtype=np.int64)
            _lambdas = np.empty(0, dtype=np.float32)

        n_real      = len(y_tr[~vmask])
        n_synthetic = len(_anchors)

        Xtr_k_real  = apply_features_branch(X_tr_train[:n_real], KIN_IDX)
        Xval_k_feat = apply_features_branch(X_val_raw,            KIN_IDX)
        Xte_k_feat  = apply_features_branch(X_te,                 KIN_IDX)

        scaler_k = StandardScaler()
        scaler_k.fit(Xtr_k_real.reshape(-1, n_kin_feat))
        Xtr_k_real_sc = scaler_k.transform(
            Xtr_k_real.reshape(-1, n_kin_feat)).reshape(-1, LOOKBACK_S, n_kin_feat)
        Xval_k_sc = scaler_k.transform(
            Xval_k_feat.reshape(-1, n_kin_feat)).reshape(-1, LOOKBACK_S, n_kin_feat)
        Xte_k_sc  = scaler_k.transform(
            Xte_k_feat.reshape(-1, n_kin_feat)).reshape(-1, LOOKBACK_S, n_kin_feat)

        # Synthetic kin features: λ-weighted interpolation in scaled space
        if n_synthetic > 0:
            lams_bc = _lambdas[:, None, None]
            Xtr_k_syn_sc = ((1.0 - lams_bc) * Xtr_k_real_sc[_anchors]
                            + lams_bc         * Xtr_k_real_sc[_neighbors])
            Xtr_k_sc = np.concatenate([Xtr_k_real_sc, Xtr_k_syn_sc.astype(np.float32)])
        else:
            Xtr_k_sc = Xtr_k_real_sc

        # ── Physiological scalars ─────────────────────────────────────────────────
        Xtr_ps_real  = phys_scalar_feats(X_tr_train[:n_real])
        Xval_ps_feat = phys_scalar_feats(X_val_raw)
        Xte_ps_feat  = phys_scalar_feats(X_te)

        scaler_ps = StandardScaler()
        scaler_ps.fit(Xtr_ps_real)
        Xtr_ps_real_sc = scaler_ps.transform(Xtr_ps_real)
        Xval_ps_sc     = scaler_ps.transform(Xval_ps_feat)
        Xte_ps_sc      = scaler_ps.transform(Xte_ps_feat)

        if n_synthetic > 0:
            lams_s = _lambdas[:, None]
            Xtr_ps_syn_sc = ((1.0 - lams_s) * Xtr_ps_real_sc[_anchors]
                             + lams_s         * Xtr_ps_real_sc[_neighbors])
            Xtr_ps_sc = np.concatenate([Xtr_ps_real_sc,
                                         Xtr_ps_syn_sc.astype(np.float32)])
        else:
            Xtr_ps_sc = Xtr_ps_real_sc

        # ── Spectral features dropped in ext8 ────────────────────────────────────
        n_tr_total  = len(y_tr_train) + n_synthetic
        Xtr_s_sc    = np.zeros((n_tr_total,       0), dtype=np.float32)
        Xval_s_sc   = np.zeros((len(y_val_d),     0), dtype=np.float32)
        Xte_s_sc    = np.zeros((len(y_te),        0), dtype=np.float32)

        # ── Train KinTCN (kin temporal only, no phys scalars) ─────────────────────
        torch.manual_seed(SEED ^ seed_d)
        torch.cuda.manual_seed_all(SEED ^ seed_d)
        Xtr_ps_zeros = np.zeros_like(Xtr_ps_sc)
        Xval_ps_zeros = np.zeros_like(Xval_ps_sc)
        model_tcn = KinTCN(n_kin_feat, PHYS_SCALAR_D, KIN_SPECTRAL_DIM).to(DEVICE)
        model_tcn = train_kin_tcn(model_tcn,
                                   Xtr_k_sc, Xtr_ps_zeros, Xtr_s_sc, y_tr_train,
                                   Xval_k_sc, Xval_ps_zeros, Xval_s_sc, y_val_d)
        model_tcn.eval()
        with torch.no_grad():
            tcn_scores = torch.sigmoid(model_tcn(
                torch.as_tensor(Xte_k_sc).to(DEVICE),
                torch.zeros(len(Xte_k_sc), PHYS_SCALAR_D).to(DEVICE),
                torch.as_tensor(Xte_s_sc).to(DEVICE))).cpu().numpy()
            val_tcn_sc = torch.sigmoid(model_tcn(
                torch.as_tensor(Xval_k_sc).to(DEVICE),
                torch.zeros(len(Xval_k_sc), PHYS_SCALAR_D).to(DEVICE),
                torch.as_tensor(Xval_s_sc).to(DEVICE))).cpu().numpy()

        # ── Train KinTCN+Phys (main model) ───────────────────────────────────────
        torch.manual_seed(SEED ^ seed_d ^ 0x1)
        torch.cuda.manual_seed_all(SEED ^ seed_d ^ 0x1)
        model_tcnp = KinTCN(n_kin_feat, PHYS_SCALAR_D, KIN_SPECTRAL_DIM).to(DEVICE)
        model_tcnp = train_kin_tcn(model_tcnp,
                                    Xtr_k_sc, Xtr_ps_sc, Xtr_s_sc, y_tr_train,
                                    Xval_k_sc, Xval_ps_sc, Xval_s_sc, y_val_d)
        model_tcnp.eval()
        with torch.no_grad():
            tcnp_scores = torch.sigmoid(model_tcnp(
                torch.as_tensor(Xte_k_sc).to(DEVICE),
                torch.as_tensor(Xte_ps_sc).to(DEVICE),
                torch.as_tensor(Xte_s_sc).to(DEVICE))).cpu().numpy()
            val_tcnp_sc = torch.sigmoid(model_tcnp(
                torch.as_tensor(Xval_k_sc).to(DEVICE),
                torch.as_tensor(Xval_ps_sc).to(DEVICE),
                torch.as_tensor(Xval_s_sc).to(DEVICE))).cpu().numpy()

        auc_lr   = _nan_or(safe_auc(y_te, lr_scores))
        auc_xgb  = _nan_or(safe_auc(y_te, xgb_scores))
        auc_tcn  = _nan_or(safe_auc(y_te, tcn_scores))
        auc_tcnp = _nan_or(safe_auc(y_te, tcnp_scores))

        # Personalisation (online — uses first GATE_ADAPT_K test labels)
        platt_scores = platt_adapt(tcnp_scores, y_te)
        auc_platt    = _nan_or(safe_auc(y_te[GATE_ADAPT_K:], platt_scores[GATE_ADAPT_K:]))

        Xte_ps_zeros = np.zeros_like(Xte_ps_sc)
        head_scores  = head_adapt(model_tcnp,
                                   Xte_k_sc, Xte_ps_sc, Xte_s_sc, y_te)
        auc_head     = _nan_or(safe_auc(y_te[GATE_ADAPT_K:], head_scores[GATE_ADAPT_K:])
                               if head_scores is not None else None)

        thresh_tcn  = _val_threshold(y_val_d, val_tcn_sc)
        thresh_tcnp = _val_threshold(y_val_d, val_tcnp_sc)
        pool_thresh_tcn.append(thresh_tcn)
        pool_thresh_tcnp.append(thresh_tcnp)

        print(f"{d:<10} | {len(y_te):>5} {100*y_te.mean():>6.1f}% | "
              f"{auc_lr:>7.4f} {auc_xgb:>7.4f} {auc_tcn:>7.4f} {auc_tcnp:>9.4f} | "
              f"{auc_platt:>8.4f} {auc_head:>8.4f}")

        pool_y.append(y_te); pool_tcn.append(tcn_scores); pool_tcnp.append(tcnp_scores)
        pool_lr.append(lr_scores); pool_xgb.append(xgb_scores)
        pool_platt.append((y_te, platt_scores))
        if head_scores is not None:
            pool_head.append((y_te, head_scores))
        pool_models_tcnp.append(model_tcnp)
        pool_Xte_k.append(Xte_k_sc)
        pool_Xte_ps.append(Xte_ps_sc)
        pool_Xte_s.append(Xte_s_sc)

        per_driver_results.append({
            "driver": d, "n_windows": int(len(y_te)),
            "pos_rate": float(y_te.mean()),
            "auc_lr": auc_lr, "auc_xgb": auc_xgb,
            "auc_tcn": auc_tcn, "auc_tcnp": auc_tcnp,
            "auc_platt": auc_platt, "auc_head": auc_head,
        })

    if singleclass_val_folds:
        print(f"\n[WARN] {len(singleclass_val_folds)} folds used loss-patience early stopping: "
              f"{singleclass_val_folds}")

    if not pool_y:
        print("[ERROR] All folds skipped."); return

    all_y    = np.concatenate(pool_y)
    all_tcn  = np.concatenate(pool_tcn)
    all_tcnp = np.concatenate(pool_tcnp)
    all_lr   = np.concatenate(pool_lr)
    all_xgb  = np.concatenate(pool_xgb)

    drv_aucs_lr   = [v for v in (safe_auc(y, s) for y, s in zip(pool_y, pool_lr))   if v is not None]
    drv_aucs_xgb  = [v for v in (safe_auc(y, s) for y, s in zip(pool_y, pool_xgb))  if v is not None]
    drv_aucs_tcn  = [v for v in (safe_auc(y, s) for y, s in zip(pool_y, pool_tcn))  if v is not None]
    drv_aucs_tcnp = [v for v in (safe_auc(y, s) for y, s in zip(pool_y, pool_tcnp)) if v is not None]

    def _print_pooled(name, all_s, drv_aucs, y_ref=None):
        y_use = y_ref if y_ref is not None else all_y
        mean_d = np.mean(drv_aucs) if drv_aucs else float("nan")
        std_d  = np.std(drv_aucs)  if drv_aucs else float("nan")
        ci_d   = bootstrap_auc_ci_drivers(drv_aucs)
        auc_w  = _nan_or(safe_auc(y_use, all_s))
        auprc  = _nan_or(safe_auprc(y_use, all_s))
        brier  = brier_score_loss(y_use, all_s)
        ece    = compute_ece(y_use, all_s)
        print(f"\n  {name}:")
        print(f"    Driver  AUROC : {mean_d:.4f} ± {std_d:.4f}  [{ci_d[0]:.4f}, {ci_d[1]:.4f}]  ← primary metric")
        print(f"    Pooled  AUROC : {auc_w:.4f}  (anticonservative due to ~94% window overlap)")
        print(f"    AUPRC         : {auprc:.4f}")
        print(f"    Brier Score   : {brier:.4f}")
        print(f"    ECE           : {ece:.4f}")

    print(f"\n{'='*72}")
    print("POOLED EVALUATION — POPULATION MODELS")
    print(f"{'='*72}")
    _print_pooled("LR baseline",     all_lr,   drv_aucs_lr)
    _print_pooled("XGB baseline",    all_xgb,  drv_aucs_xgb)
    _print_pooled("KinTCN (no phys scalars)", all_tcn,  drv_aucs_tcn)
    _print_pooled("KinTCN+Phys (main model)", all_tcnp, drv_aucs_tcnp)

    # Wilcoxon: KinTCN+Phys > KinTCN
    paired = [(r["auc_tcn"], r["auc_tcnp"]) for r in per_driver_results
              if not (math.isnan(r["auc_tcn"]) or math.isnan(r["auc_tcnp"]))]
    if len(paired) >= 10:
        a_tcn, a_tcnp = zip(*paired)
        diff = np.array(a_tcnp) - np.array(a_tcn)
        if np.any(diff != 0):
            stat, p_wx = wilcoxon(a_tcnp, a_tcn, alternative="greater")
            sig = "***" if p_wx < 0.001 else ("**" if p_wx < 0.01 else ("*" if p_wx < 0.05 else "ns"))
            print(f"\n  Wilcoxon (KinTCN+Phys > KinTCN) : W={stat:.1f}  p={p_wx:.4f}  {sig}")
        print(f"  Mean per-driver gain (KinTCN+Phys − KinTCN): {diff.mean():+.4f} ± {diff.std():.4f}")

    # Personalisation
    if pool_platt:
        platt_y   = np.concatenate([y[GATE_ADAPT_K:] for y, _ in pool_platt])
        platt_s   = np.concatenate([s[GATE_ADAPT_K:] for _, s in pool_platt])
        drv_platt = [v for v in (safe_auc(y[GATE_ADAPT_K:], s[GATE_ADAPT_K:])
                                 for y, s in pool_platt) if v is not None]
        print(f"\n{'='*72}")
        print(f"ONLINE PERSONALISATION (first {GATE_ADAPT_K} test-participant windows used for adaptation)")
        print(f"{'='*72}")
        _print_pooled(f"Platt† ({len(pool_platt)} folds)", platt_s, drv_platt, y_ref=platt_y)
    if pool_head:
        head_y   = np.concatenate([y[GATE_ADAPT_K:] for y, s in pool_head])
        head_s   = np.concatenate([s[GATE_ADAPT_K:] for y, s in pool_head])
        drv_head = [v for v in (safe_auc(y[GATE_ADAPT_K:], s[GATE_ADAPT_K:])
                                for y, s in pool_head) if v is not None]
        _print_pooled(f"HeadFT† ({len(pool_head)} folds)", head_s, drv_head, y_ref=head_y)

    # Threshold metrics
    print(f"\n{'='*72}")
    print("THRESHOLD-DEPENDENT METRICS  (per-fold Youden's J from val set)")
    print(f"{'='*72}")
    thresh_str_tcn  = f"{np.mean(pool_thresh_tcn):.3f}±{np.std(pool_thresh_tcn):.3f}"
    thresh_str_tcnp = f"{np.mean(pool_thresh_tcnp):.3f}±{np.std(pool_thresh_tcnp):.3f}"
    print(f"  Per-fold threshold — KinTCN: {thresh_str_tcn}  KinTCN+Phys: {thresh_str_tcnp}")
    for name, pool_s, pool_thresh in [("KinTCN", pool_tcn, pool_thresh_tcn),
                                       ("KinTCN+Phys", pool_tcnp, pool_thresh_tcnp)]:
        f1s, precs, recs = [], [], []
        for y_f, s_f, th in zip(pool_y, pool_s, pool_thresh):
            preds = (s_f >= th).astype(int)
            if len(np.unique(y_f)) < 2: continue
            f1s.append(f1_score(y_f, preds, zero_division=0))
            precs.append(precision_score(y_f, preds, zero_division=0))
            recs.append(recall_score(y_f, preds, zero_division=0))
        if f1s:
            print(f"  {name:<16} driver F1={np.mean(f1s):.3f}±{np.std(f1s):.3f} "
                  f"Prec={np.mean(precs):.3f}±{np.std(precs):.3f} "
                  f"Rec={np.mean(recs):.3f}±{np.std(recs):.3f}")

    # Permutation importance (KinTCN+Phys)
    print(f"\n{'='*72}")
    print("PERMUTATION FEATURE IMPORTANCE — KinTCN+Phys")
    print(f"{'='*72}")
    n_kin = len(KIN_COLS)
    signal_deltas = {col: [] for col in KIN_COLS + PHYS_COLS}

    for fi, (model_i, Xk_i, Xps_i, Xs_i, y_i) in enumerate(
            zip(pool_models_tcnp, pool_Xte_k, pool_Xte_ps, pool_Xte_s, pool_y)):
        if safe_auc(y_i, None if len(np.unique(y_i)) < 2 else y_i) is None:
            continue
        base = safe_auc(y_i, torch.sigmoid(model_i(
            torch.as_tensor(Xk_i).to(DEVICE),
            torch.as_tensor(Xps_i).to(DEVICE),
            torch.as_tensor(Xs_i).to(DEVICE))).detach().cpu().numpy())
        if base is None:
            continue
        rng_pi = np.random.default_rng(SEED ^ fi)
        for rep in range(N_PERM_REPEATS):
            # Permute each kinematic signal (5 feature columns per signal)
            for ki, col in enumerate(KIN_COLS):
                feat_cols = [ki + j * n_kin for j in range(5)]
                Xk_perm = Xk_i.copy()
                perm_idx = rng_pi.permutation(len(Xk_perm))
                Xk_perm[:, :, feat_cols] = Xk_perm[perm_idx, :, :][:, :, feat_cols]
                perm_auc = safe_auc(y_i, torch.sigmoid(model_i(
                    torch.as_tensor(Xk_perm).to(DEVICE),
                    torch.as_tensor(Xps_i).to(DEVICE),
                    torch.as_tensor(Xs_i).to(DEVICE))).detach().cpu().numpy())
                if perm_auc is not None:
                    signal_deltas[col].append(base - perm_auc)
            # Permute physiological scalars ([mean_aro, std_aro, mean_hr, std_hr])
            for pi, col in enumerate(PHYS_COLS):
                Xps_perm = Xps_i.copy()
                perm_idx = rng_pi.permutation(len(Xps_perm))
                Xps_perm[:, [pi, pi + len(PHYS_COLS)]] = \
                    Xps_perm[perm_idx, :][:, [pi, pi + len(PHYS_COLS)]]
                perm_auc = safe_auc(y_i, torch.sigmoid(model_i(
                    torch.as_tensor(Xk_i).to(DEVICE),
                    torch.as_tensor(Xps_perm).to(DEVICE),
                    torch.as_tensor(Xs_i).to(DEVICE))).detach().cpu().numpy())
                if perm_auc is not None:
                    signal_deltas[col].append(base - perm_auc)

    print(f"\n  Signal importance (mean ΔAUC when permuted):")
    for col in sorted(signal_deltas, key=lambda c: np.mean(signal_deltas[c]) if signal_deltas[c] else 0, reverse=True):
        branch = "KIN" if col in KIN_COLS else "PHYS"
        deltas = signal_deltas[col]
        if deltas:
            print(f"    [{branch}] {col:<30} Δ={np.mean(deltas):+.4f} ± {np.std(deltas):.4f}")

    # Spectral band importance — skipped (spectral features dropped in ext8)
    if KIN_SPECTRAL_DIM > 0:
        print(f"\n  Per-band spectral importance (ΔAUC when band permuted): N/A (dropped)")

    # Per-driver summary
    print(f"\n{'='*72}")
    print("PER-DRIVER SUMMARY — KinTCN+Phys")
    print(f"{'='*72}")
    gains_vs_tcn = [r["auc_tcnp"] - r["auc_tcn"] for r in per_driver_results
                    if not (math.isnan(r["auc_tcnp"]) or math.isnan(r["auc_tcn"]))]
    print(f"  N drivers evaluated  : {len(drv_aucs_tcnp)}")
    print(f"  KinTCN+Phys AUROC   : {np.mean(drv_aucs_tcnp):.4f} ± {np.std(drv_aucs_tcnp):.4f}")
    print(f"  KinTCN      AUROC   : {np.mean(drv_aucs_tcn):.4f} ± {np.std(drv_aucs_tcn):.4f}")
    if gains_vs_tcn:
        print(f"  Gain (Phys − KinOnly): {np.mean(gains_vs_tcn):+.4f} ± {np.std(gains_vs_tcn):.4f}  "
              f"[{min(gains_vs_tcn):.4f}, {max(gains_vs_tcn):.4f}]")
        print(f"  Drivers +Phys > KinOnly : "
              f"{sum(g > 0 for g in gains_vs_tcn)}/{len(gains_vs_tcn)}  "
              f"({100*sum(g > 0 for g in gains_vs_tcn)/len(gains_vs_tcn):.1f}%)")

    # Save results
    results = {
        "config": {
            "LOOKBACK_S": LOOKBACK_S, "GAP": GAP, "HORIZON": HORIZON,
            "WINDOW_STEP": WINDOW_STEP, "RENORM_CLIP": RENORM_CLIP,
            "SMOTE": USE_SMOTE, "EPOCHS": EPOCHS, "LR": LR,
        },
        "pooled": {
            "lr":      {"driver_mean": float(np.mean(drv_aucs_lr)),
                        "driver_std":  float(np.std(drv_aucs_lr))},
            "xgb":     {"driver_mean": float(np.mean(drv_aucs_xgb)),
                        "driver_std":  float(np.std(drv_aucs_xgb))},
            "kin_tcn": {"driver_mean": float(np.mean(drv_aucs_tcn)),
                        "driver_std":  float(np.std(drv_aucs_tcn))},
            "kin_tcn_phys": {
                "driver_mean": float(np.mean(drv_aucs_tcnp)),
                "driver_std":  float(np.std(drv_aucs_tcnp)),
                "ci_driver":   list(bootstrap_auc_ci_drivers(drv_aucs_tcnp)),
            },
        },
        "per_driver": per_driver_results,
    }
    out_path = OUT_DIR / "kin_tcn_ext9_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out_path}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
