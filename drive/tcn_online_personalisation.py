"""
tcn_online_personalisation.py

Online (prequential) personalisation for driving impairment detection.

Motivation
----------
tcn_impairment_detect.py uses a fixed temporal split:
  pers (early session) → [buffer] → eval (late session)
This creates concept drift: the model adapts to early-session behaviour but
evaluates on late-session behaviour, which may differ (fatigue, track familiarity).

This file replaces that fixed split with a predict-then-update loop:

  For each window t (in temporal order):
    1. Predict with the current personalised model → record score.
    2. Label for window t becomes available (t + GAP + HORIZON seconds later;
       immediately available in simulation).
    3. Add (window_t, label_t) to a rolling replay buffer.
    4. Take ONLINE_STEPS gradient steps on the replay buffer.

Evaluation: prequential AUC — all predictions are made *before* the model
sees the label, so there is zero label leakage.

Output mirrors tcn_impairment_detect.py:
  - Label validity report (Mann-Whitney U, per-driver tier)
  - Pipeline configuration summary
  - Per-driver table: LR / XGBoost / TCN-Population / TCN-Online AUC
  - Pooled evaluation: AUC + 95% CI + AUPRC + Brier + ECE for all four models
  - Threshold-dependent metrics: F1 / Precision / Recall at Youden's-J threshold
  - Online learning dynamics: per-quartile AUC + Wilcoxon test on gains
  - Permutation feature importance (TCN-Population, pooled)
  - Modality ablation (Physiology+speed / Kinematics / Combined)
  - Stratified evaluation (CLC-only vs Non-CLC positive windows)
  - Per-driver summary statistics (mean / median / std / min / max)

Architecture and data processing are identical to tcn_impairment_detect.py.
"""

import copy
import hashlib
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                              average_precision_score, f1_score,
                              precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu, wilcoxon
import xgboost as xgb
import random

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEED   = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SIGNAL_COLS = [
    "arousal", "hr",
    "steeringWheelAngle",
    "steeringTorq",
    "acceleration.y",
    "speed.x",
]

VEHICLE_COLS = ["steeringWheelAngle", "steeringTorq", "acceleration.y", "speed.x"]

ABLATION_CONDITIONS = {
    "Physiology + speed": ["arousal", "hr", "speed.x"],
    "Kinematics":         ["steeringWheelAngle", "steeringTorq", "acceleration.y"],
    "Combined":           SIGNAL_COLS,
}

SEVERITY = {
    "Collision":               5,
    "Red_light_violation":     3,
    "panic_braking_with_stop": 2,
    "center_line_crossing":    2,
    "panic_braking":           1,
    "sharp_turn":              1,
}

# Population model training (identical to tcn_impairment_detect.py)
EPOCHS, LR    = 100, 1e-3
LOOKBACK_S    = 60
WINDOW_STEP   = 5
GAP, HORIZON  = 15, 5
BATCH_SIZE    = 64
WEIGHT_DECAY  = 1e-4
ROLL_K        = 10
JITTER_STD    = 0.01
CUTOUT_LEN    = 5
CUTOUT_PROB   = 0.2
EVENT_VICINITY = 10
N_BOOTSTRAP   = 2000
MIN_POSITIVES      = 1
MIN_EVAL_POSITIVES = 3
N_LC_BINS          = 4     # quartile groups for the online learning curve

# Online personalisation
ONLINE_LR          = 1e-4   # small LR — fine adjustments only
ONLINE_STEPS       = 5      # gradient steps after each window
REPLAY_BUFFER_SIZE = 20     # rolling window of recent (x, y) pairs
ONLINE_LAYERS      = ["head"]  # layers updated online; head-only for stability
GRAD_CLIP_NORM     = 1.0    # max gradient norm for online updates
# L2 tether: penalises drift from the population head weights.
# Adaptive: lambda * (REPLAY_BUFFER_SIZE / buf_size) so regularisation is
# strongest when the buffer is barely filled and relaxes as data accumulates.
# Mirror of LAMBDA_TETHER in tcn_impairment_detect.py, tuned for per-window SGD.
LAMBDA_TETHER_ONLINE = 0.1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark     = True

OUT_DIR = Path(__file__).parent / "impairment_results"
OUT_DIR.mkdir(exist_ok=True)

# ── NORMALISATION ──────────────────────────────────────────────────────────────

def mark_event_onsets(df):
    """Convert contiguous error runs to single onset rows per (driver, route)."""
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
    """Per-(driver, route) z-score for vehicle signals; physiology left absolute."""
    df = df.copy()
    for _, grp in df.groupby(["id", "route"]):
        idx = grp.index
        for col in VEHICLE_COLS:
            mu  = grp[col].mean()
            sig = grp[col].std() + 1e-6
            df.loc[idx, col] = (grp[col] - mu) / sig
    return df

# ── FEATURE ENGINEERING ────────────────────────────────────────────────────────

def engineer_features(window):
    """(T, C) → (T, C*5): raw | diff1 | roll_mean | roll_std | z-score"""
    T, C   = window.shape
    diff1  = np.diff(window, axis=0, prepend=window[:1])
    cs     = np.cumsum(window,      axis=0)
    cs_sq  = np.cumsum(window ** 2, axis=0)
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


def apply_features(X_raw):
    return np.stack([engineer_features(w) for w in X_raw])


def window_baseline_feats(X_raw, col_idx=None):
    """(N, T, C) → (N, C_sel*3): per-signal [mean, std, max] over the lookback window.
    Used as tabular features for LR and XGBoost baselines."""
    X = X_raw if col_idx is None else X_raw[:, :, col_idx]
    return np.concatenate([
        X.mean(axis=1),
        X.std(axis=1),
        X.max(axis=1),
    ], axis=1).astype(np.float32)

# ── WINDOWING ──────────────────────────────────────────────────────────────────

def composite_risk_score(future_df):
    return sum(
        SEVERITY[col] * int((future_df[col] > 0).any())
        for col in SEVERITY
    )


def future_error_types(future_df):
    """Return frozenset of error-type column names present in future_df."""
    return frozenset(
        col for col in SEVERITY
        if col in future_df.columns and (future_df[col] > 0).any()
    )


def build_windows(df, fine_stride=True):
    """Returns (windows, labels, scores, pids, etypes, routes, t_starts).

    fine_stride=True  (training): EVENT_VICINITY densification around positive events.
    fine_stride=False (eval)    : fixed WINDOW_STEP, unbiased AUC.

    routes and t_starts allow prequential temporal ordering:
        np.lexsort((t_starts, routes))  — sorts by route first, then timestamp.
    Safe even when Timestamps reset to zero at the start of each route.
    """
    windows, labels, scores, pids, etypes, routes, t_starts = [], [], [], [], [], [], []

    for (pid, route), grp in df.groupby(["id", "route"]):
        grp = grp.sort_values("Timestamp").reset_index(drop=True)
        n   = len(grp)
        ts  = grp["Timestamp"].values

        if fine_stride:
            # First pass: find positive event starts at normal stride.
            pos_starts = set()
            idx = 0
            while idx + LOOKBACK_S + GAP + HORIZON <= n:
                sig    = grp.iloc[idx: idx + LOOKBACK_S][SIGNAL_COLS].values.astype(np.float32)
                future = grp.iloc[idx + LOOKBACK_S + GAP: idx + LOOKBACK_S + GAP + HORIZON]
                if not np.isnan(sig).any() and composite_risk_score(future) > 0:
                    pos_starts.add(idx)
                idx += WINDOW_STEP

            # Build vicinity: up to EVENT_VICINITY rows before each positive start.
            vicinity = set()
            for ps in pos_starts:
                for offset in range(-EVENT_VICINITY, 1):
                    t = ps + offset
                    if t >= 0:
                        vicinity.add(t)

            # Second pass: step=1 in vicinity, WINDOW_STEP elsewhere.
            idx = 0
            while idx + LOOKBACK_S + GAP + HORIZON <= n:
                step = 1 if idx in vicinity else WINDOW_STEP
                sig  = grp.iloc[idx: idx + LOOKBACK_S][SIGNAL_COLS].values.astype(np.float32)
                if not np.isnan(sig).any():
                    future = grp.iloc[idx + LOOKBACK_S + GAP: idx + LOOKBACK_S + GAP + HORIZON]
                    score  = composite_risk_score(future)
                    windows.append(sig)
                    labels.append(int(score > 0))
                    scores.append(score)
                    pids.append(pid)
                    etypes.append(future_error_types(future))
                    routes.append(route)
                    t_starts.append(ts[idx])
                idx += step
        else:
            idx = 0
            while idx + LOOKBACK_S + GAP + HORIZON <= n:
                sig = grp.iloc[idx: idx + LOOKBACK_S][SIGNAL_COLS].values.astype(np.float32)
                if not np.isnan(sig).any():
                    future = grp.iloc[idx + LOOKBACK_S + GAP: idx + LOOKBACK_S + GAP + HORIZON]
                    score  = composite_risk_score(future)
                    windows.append(sig)
                    labels.append(int(score > 0))
                    scores.append(score)
                    pids.append(pid)
                    etypes.append(future_error_types(future))
                    routes.append(route)
                    t_starts.append(ts[idx])
                idx += WINDOW_STEP

    return (np.array(windows,   dtype=np.float32),
            np.array(labels,    dtype=np.float32),
            np.array(scores,    dtype=np.float32),
            np.array(pids),
            np.array(etypes, dtype=object),
            np.array(routes),
            np.array(t_starts))

# ── DATASET & UTILS ────────────────────────────────────────────────────────────

class DrivingDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.as_tensor(X).float()
        self.y = torch.as_tensor(y).float()
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]
        if self.augment:
            x += torch.randn_like(x) * JITTER_STD
            if random.random() < CUTOUT_PROB:
                t0 = random.randint(0, x.shape[0] - CUTOUT_LEN)
                x[t0: t0 + CUTOUT_LEN, :] = 0.0
        return x, y


def get_balanced_loader(X, y, batch_size=BATCH_SIZE, augment=False):
    y_int   = y.astype(int)
    counts  = np.bincount(y_int, minlength=2)
    weights = 1.0 / (counts + 1e-6)
    sw      = torch.from_numpy(weights[y_int])
    sampler = WeightedRandomSampler(sw, len(sw))
    return DataLoader(DrivingDataset(X, y, augment=augment),
                      batch_size=batch_size, sampler=sampler)


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return None
    return roc_auc_score(y_true, y_score)


def safe_auprc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return None
    return average_precision_score(y_true, y_score)


def bootstrap_auc_ci_windows(y_true, y_score, n_boot=N_BOOTSTRAP, seed=SEED):
    """Window-level bootstrap 95% CI for pooled AUC-ROC."""
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    if len(aucs) < 10:
        return float("nan"), float("nan")
    return tuple(np.percentile(aucs, [2.5, 97.5]))


def bootstrap_auc_ci_drivers(driver_aucs, n_boot=N_BOOTSTRAP, seed=SEED):
    """Driver-level bootstrap CI over per-driver AUC scores.

    Resampling at the driver level respects the fact that windows from the
    same driver share a model checkpoint and are not independent.
    """
    aucs = np.array(driver_aucs)
    n    = len(aucs)
    if n < 2:
        return float("nan"), float("nan")
    rng  = np.random.default_rng(seed)
    boot = [rng.choice(aucs, n, replace=True).mean() for _ in range(n_boot)]
    return tuple(np.percentile(boot, [2.5, 97.5]))


def focal_loss(logits, targets, gamma=2.0, pos_weight=None):
    """Focal loss with optional positive-class weight for class imbalance."""
    pw = (torch.tensor(pos_weight, dtype=logits.dtype, device=logits.device)
          if pos_weight is not None else None)
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pw, reduction="none"
    )
    pt  = torch.exp(-bce)
    return ((1 - pt) ** gamma * bce).mean()


def _val_sample(arr, frac, rng):
    """Sample a stratified fraction of arr without replacement (min 1 if non-empty)."""
    n = min(max(1, int(frac * len(arr))), len(arr)) if len(arr) > 0 else 0
    return rng.choice(arr, n, replace=False) if n > 0 else np.array([], dtype=arr.dtype)


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error: mean |accuracy - confidence| weighted by bin size."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    n    = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask   = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo)
        if mask.sum() == 0:
            continue
        acc  = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += mask.sum() / n * abs(acc - conf)
    return ece


def threshold_metrics(y_true, y_prob):
    """F1 / Precision / Recall at the Youden's-J optimal threshold.

    Youden's J = TPR + TNR - 1 = TPR - FPR, maximised over the ROC curve.
    This is threshold-free in the sense that no held-out calibration set is
    needed — the threshold is derived analytically from the ROC curve itself.
    Returns (f1, precision, recall, threshold).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_idx  = int(np.argmax(tpr - fpr))
    thresh = float(thresholds[j_idx])
    preds  = (y_prob >= thresh).astype(int)
    f1   = float(f1_score(y_true,        preds, zero_division=0))
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec  = float(recall_score(y_true,    preds, zero_division=0))
    return f1, prec, rec, thresh

# ── MODEL (identical to tcn_impairment_detect.py) ─────────────────────────────

class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Linear(channels, channels // 2)
        self.score = nn.Linear(channels // 2, 1)

    def forward(self, x):
        x_t     = x.permute(0, 2, 1)
        h       = torch.tanh(self.query(x_t))
        weights = torch.softmax(self.score(h), dim=1)
        return torch.sum(x_t * weights, dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, d):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, out_c, 3, padding=d, dilation=d),
            nn.BatchNorm1d(out_c),
            nn.ReLU(),
            nn.Dropout1d(0.1),
        )
        self.res = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.res(x)


class TCN_Attention_Net(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 5 blocks: dilations 1,2,4,8,16 → RF = 1 + 2*(1+2+4+8+16) = 63
        self.network = nn.Sequential(
            ResBlock(in_channels, 32,  1),
            ResBlock(32,          64,  2),
            ResBlock(64,          64,  4),
            ResBlock(64,          64,  8),
            ResBlock(64,          64, 16),
        )
        self.attention = TemporalAttention(64)
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.network(x.permute(0, 2, 1))
        return self.head(self.attention(x)).squeeze(-1)

# ── VALIDITY REPORT ────────────────────────────────────────────────────────────

def print_validity_report(X_raw, y, scores, pid):
    print("\n" + "=" * 70)
    print("LABEL VALIDITY REPORT — COMPOSITE RISK TARGET")
    print("=" * 70)
    n_total, n_pos = len(y), int(y.sum())
    print(f"Total windows    : {n_total}")
    print(f"Positive (risk>0): {n_pos} ({n_pos/n_total*100:.1f}%)")
    print(f"Negative         : {n_total - n_pos} ({(n_total-n_pos)/n_total*100:.1f}%)")

    print(f"\nRisk score distribution (positives only):")
    for v in sorted(np.unique(scores[y == 1])):
        c = (scores[y == 1] == v).sum()
        print(f"  score={int(v):>2}  n={c:>5}  ({c/n_pos*100:.1f}%)")

    print(f"\nPredictive validity — Mann-Whitney U (pos vs neg, raw signals):")
    print(f"  {'Signal':<25}  {'Mean(neg)':>10}  {'Mean(pos)':>10}  {'p-value':>12}  Sig")
    for i, col in enumerate(SIGNAL_COLS):
        neg_v = X_raw[y == 0, :, i].mean(axis=1)
        pos_v = X_raw[y == 1, :, i].mean(axis=1)
        if len(pos_v) < 2:
            continue
        _, p = mannwhitneyu(pos_v, neg_v, alternative="two-sided")
        sig  = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"  {col:<25}  {neg_v.mean():>10.4f}  {pos_v.mean():>10.4f}  {p:>12.2e}  {sig}")

    print(f"\nPer-driver positive rate:")
    print(f"  {'Driver':<12}  {'N':>6}  {'Pos':>5}  {'Rate%':>7}  Tier")
    for d in np.unique(pid):
        m    = pid == d
        nd   = int(m.sum())
        np_  = int(y[m].sum())
        rate = np_ / nd * 100
        tier = "HIGH" if rate >= 10 else ("MED" if rate >= 5 else "LOW")
        print(f"  {d:<12}  {nd:>6}  {np_:>5}  {rate:>6.1f}%  {tier}")
    print("=" * 70)

# ── ONLINE PERSONALISATION ─────────────────────────────────────────────────────

def online_evaluate_driver(model_pop, Xte_sc, y_te):
    """
    Prequential predict-then-update loop over all test windows.

    Xte_sc and y_te MUST be sorted in strict temporal order so that
    predictions are always made before the corresponding label is seen.

    Steps per window t:
      1. Predict with current model (before update) → no label leakage.
      2. Add (x_t, y_t) to rolling replay buffer (circular, O(1) writes).
      3. If buffer contains both classes, take ONLINE_STEPS gradient steps
         on the full replay buffer with class-weighted focal loss.

    The backbone is set to train() so Dropout is active during online updates
    (regularization against overfitting on the tiny replay buffer).
    BatchNorm layers are explicitly kept in eval() to preserve the running
    statistics learned during population training.

    An ExponentialLR scheduler decays the online learning rate after each
    gradient update window to reduce oscillation as the model adapts.

    Population model scores are pre-computed in a single batched forward pass
    before the loop begins (model_pop never changes during online eval).

    model_pop must already be in eval() mode when passed in.

    Returns
    -------
    pop_scores    : (N,) predictions from the frozen population model
    online_scores : (N,) prequential predictions from the online model
    """
    model_online = copy.deepcopy(model_pop)

    # Only the head is updated online — exact name prefix match avoids
    # accidentally matching layers whose names contain "head" as a substring.
    for name, param in model_online.named_parameters():
        param.requires_grad = any(
            name == layer or name.startswith(layer + ".")
            for layer in ONLINE_LAYERS
        )

    online_params = [p for p in model_online.parameters() if p.requires_grad]
    opt = torch.optim.Adam(online_params, lr=ONLINE_LR, weight_decay=WEIGHT_DECAY)
    # ExponentialLR: decay by ~0.5% per update window for gradual stabilisation.
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.995)

    # Snapshot of the population head weights — used for the adaptive L2 tether.
    # Detached so they never receive gradients.
    pop_head = {
        name: p.clone().detach()
        for name, p in model_online.named_parameters()
        if p.requires_grad
    }

    # Pre-compute frozen population scores in one batch.
    Xte_t = torch.as_tensor(Xte_sc, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pop_scores = torch.sigmoid(model_pop(Xte_t)).cpu().numpy()

    # Preallocated circular buffer — avoids repeated copies on every window.
    buf_X_np = np.zeros((REPLAY_BUFFER_SIZE, *Xte_sc.shape[1:]), dtype=np.float32)
    buf_y_np = np.zeros(REPLAY_BUFFER_SIZE, dtype=np.float32)
    buf_head = 0   # next write position (wraps around)
    buf_size = 0   # number of valid entries (≤ REPLAY_BUFFER_SIZE)

    online_scores = []

    for t in range(len(Xte_sc)):
        x_t = Xte_t[t:t + 1]

        # ── 1. Predict (before any update on this window) ──────────────────
        model_online.eval()
        with torch.no_grad():
            online_scores.append(torch.sigmoid(model_online(x_t)).item())

        # ── 2. Update replay buffer ────────────────────────────────────────
        buf_X_np[buf_head] = Xte_sc[t]
        buf_y_np[buf_head] = y_te[t]
        buf_head = (buf_head + 1) % REPLAY_BUFFER_SIZE
        buf_size = min(buf_size + 1, REPLAY_BUFFER_SIZE)

        buf_y_arr = buf_y_np[:buf_size]

        # ── 3. Gradient steps on full replay buffer ────────────────────────
        if len(np.unique(buf_y_arr)) >= 2:
            n_pos = int(buf_y_arr.sum())
            n_neg = buf_size - n_pos
            pos_weight = n_neg / max(n_pos, 1)

            # Build tensors once per window (not once per gradient step).
            buf_X_t = torch.as_tensor(buf_X_np[:buf_size], dtype=torch.float32).to(DEVICE)
            buf_y_t = torch.as_tensor(buf_y_arr,           dtype=torch.float32).to(DEVICE)

            # Enable train() so Dropout regularizes the small-batch updates.
            # Explicitly freeze BatchNorm to preserve running statistics.
            model_online.train()
            for m in model_online.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()

            # Adaptive tether: stronger when buffer is sparse, relaxes when full.
            lam = LAMBDA_TETHER_ONLINE * (REPLAY_BUFFER_SIZE / buf_size)

            for _ in range(ONLINE_STEPS):
                logits  = model_online(buf_X_t)
                task    = focal_loss(logits, buf_y_t, pos_weight=pos_weight)
                tether  = sum(
                    ((p - pop_head[name]) ** 2).sum()
                    for name, p in model_online.named_parameters()
                    if p.requires_grad
                )
                loss = task + lam * tether
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online_params, max_norm=GRAD_CLIP_NORM)
                opt.step()

            # Step the scheduler only after an optimizer step to avoid
            # PyTorch's "scheduler before optimizer" warning.
            scheduler.step()

    return pop_scores, np.array(online_scores)

# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(Path(__file__).parent / "relab+unibo_dataset.csv")
    df = mark_event_onsets(df)
    df = normalize_signals(df)

    X_raw_tr, y_tr_all, _,          pid_tr_all, _,          _,             _          = build_windows(df, fine_stride=True)
    X_raw_te, y_te_all, scores_te,  pid_te_all, etypes_te,  routes_te_all, ts_te_all  = build_windows(df, fine_stride=False)

    n_raw   = len(SIGNAL_COLS)
    n_feats = n_raw * 5

    print_validity_report(X_raw_te, y_te_all, scores_te, pid_te_all)

    print(f"\n{'='*70}")
    print("ONLINE PERSONALISATION — prequential predict-then-update")
    print(f"{'='*70}")
    print(f"Signals           : {SIGNAL_COLS}")
    print(f"Normalisation     : vehicle z-scored per route; physiology absolute")
    print(f"Feature channels  : {n_raw} raw → {n_feats} engineered (TCN)")
    print(f"Baseline features : {n_raw} signals × 3 stats = {n_raw*3} (mean, std, max)")
    print(f"Loss              : focal (gamma=2)")
    print(f"TCN blocks        : 5 (d=1,2,4,8,16 — RF=63) with residual connections")
    print(f"Epochs            : {EPOCHS} with CosineAnnealingLR")
    print(f"GAP / HORIZON     : {GAP}s / {HORIZON}s → predicts errors in [{GAP}, {GAP+HORIZON}]s")
    print(f"Online LR         : {ONLINE_LR}  |  Steps/window : {ONLINE_STEPS}")
    print(f"Replay buffer     : {REPLAY_BUFFER_SIZE} windows  |  Layers : {ONLINE_LAYERS}")
    print(f"Evaluation        : prequential AUC (predict before update — no leakage)")
    print(f"Bootstrap CI      : {N_BOOTSTRAP} resamples (window-level pooled; driver-level summary)")
    print(f"{'='*70}\n")

    drivers = [d for d in np.unique(pid_te_all)
               if y_te_all[pid_te_all == d].sum() >= MIN_POSITIVES]
    rng_perm = np.random.default_rng(SEED + 1)    # permutation importance RNG

    hdr = (f"{'Driver':<10} | {'N_win':>5} {'PosR%':>6} | "
           f"{'LR':>7} {'XGB':>7} | "
           f"{'PopAUC':>7} {'OnlAUC':>7} {'Gain':>7}")
    print(hdr)
    print("-" * len(hdr))

    per_driver_results = []

    # Pooled arrays (aligned: one entry per driver)
    pool_y          = []
    pool_pop_scores = []
    pool_onl_scores = []
    pool_lr_probs   = []
    pool_xgb_probs  = []
    pool_Xte_sc     = []   # scaled engineered features — for permutation importance
    pool_models     = []
    pool_etypes     = []
    pool_ablation   = {name: [] for name in ABLATION_CONDITIONS}

    for d in drivers:
        mask_tr = pid_tr_all != d
        X_tr    = X_raw_tr[mask_tr]
        y_tr    = y_tr_all[mask_tr]
        pid_tr  = pid_tr_all[mask_tr]

        mask_te   = pid_te_all == d
        X_te      = X_raw_te[mask_te]
        y_te      = y_te_all[mask_te]
        ts_te     = ts_te_all[mask_te]
        routes_te = routes_te_all[mask_te]
        etypes_d  = etypes_te[mask_te]

        # Sort test windows in strict temporal order (prequential guarantee).
        # np.lexsort: last key = primary sort → route first, timestamp within route.
        order     = np.lexsort((ts_te, routes_te))
        X_te      = X_te[order]
        y_te      = y_te[order]
        etypes_d  = etypes_d[order]

        # Validation split — stratified by driver positive-rate.
        seed_d   = int(hashlib.md5(str(d).encode()).hexdigest(), 16) & 0xFFFFFFFF
        fold_rng = np.random.default_rng(SEED ^ seed_d)

        train_drivers = np.unique(pid_tr)
        has_pos = np.array([y_tr[pid_tr == p].sum() > 0 for p in train_drivers])
        pos_d   = train_drivers[has_pos]
        neg_d   = train_drivers[~has_pos]

        val_ids = np.concatenate([
            _val_sample(pos_d, 0.20, fold_rng),
            _val_sample(neg_d, 0.20, fold_rng),
        ])
        vmask = np.isin(pid_tr, val_ids)

        if len(np.unique(y_tr[vmask])) < 2:
            print(f"{d:<10} | SKIP — val fold single-class")
            continue

        # ── TCN features & scaling ─────────────────────────────────────────
        Xtr_feat  = apply_features(X_tr[~vmask])
        Xval_feat = apply_features(X_tr[vmask])
        Xte_feat  = apply_features(X_te)

        scaler  = StandardScaler()
        Xtr_sc  = scaler.fit_transform(
            Xtr_feat.reshape(-1, n_feats)).reshape(-1, LOOKBACK_S, n_feats)
        Xval_sc = scaler.transform(
            Xval_feat.reshape(-1, n_feats)).reshape(-1, LOOKBACK_S, n_feats)
        Xte_sc  = scaler.transform(
            Xte_feat.reshape(-1, n_feats)).reshape(-1, LOOKBACK_S, n_feats)

        # ── Baseline features (window stats, uniform-stride train distribution) ─
        # Uses X_raw_te from all other drivers — same protocol as tcn_impairment_detect.py.
        mask_bl_tr  = pid_te_all != d
        X_bl_base   = X_raw_te[mask_bl_tr]
        y_bl_base   = y_te_all[mask_bl_tr]
        pid_bl_base = pid_te_all[mask_bl_tr]
        vmask_bl    = np.isin(pid_bl_base, val_ids)

        X_bl_tr_f  = window_baseline_feats(X_bl_base[~vmask_bl])
        X_bl_te_f  = window_baseline_feats(X_te)
        bl_scaler  = StandardScaler()
        X_bl_tr_sc = bl_scaler.fit_transform(X_bl_tr_f)
        X_bl_te_sc = bl_scaler.transform(X_bl_te_f)

        y_bl_tr = y_bl_base[~vmask_bl]
        pos_w   = (y_bl_tr == 0).sum() / max(1, (y_bl_tr == 1).sum())

        lr_clf  = LogisticRegression(C=1.0, class_weight="balanced",
                                     max_iter=1000, random_state=SEED)
        lr_clf.fit(X_bl_tr_sc, y_bl_tr)

        xgb_clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            scale_pos_weight=pos_w, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, verbosity=0,
        )
        xgb_clf.fit(X_bl_tr_sc, y_bl_tr)

        lr_probs  = lr_clf.predict_proba(X_bl_te_sc)[:, 1]
        xgb_probs = xgb_clf.predict_proba(X_bl_te_sc)[:, 1]

        # ── Train population TCN ───────────────────────────────────────────
        model     = TCN_Attention_Net(n_feats).to(DEVICE)
        opt       = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        loader    = get_balanced_loader(Xtr_sc, y_tr[~vmask], augment=True)

        best_auc, best_w = float("-inf"), None
        for _ in range(EPOCHS):
            model.train()
            for xb, yb in loader:
                loss = focal_loss(model(xb.to(DEVICE)), yb.to(DEVICE))
                opt.zero_grad(); loss.backward(); opt.step()
            scheduler.step()
            model.eval()
            with torch.no_grad():
                preds = torch.sigmoid(
                    model(torch.as_tensor(Xval_sc).to(DEVICE))).cpu().numpy()
            auc = safe_auc(y_tr[vmask], preds)
            if auc is not None and auc > best_auc:
                best_auc = auc
                best_w   = copy.deepcopy(model.state_dict())

        if best_w is None:
            print(f"{d:<10} | SKIP — no valid checkpoint")
            continue
        model.load_state_dict(best_w)
        model.eval()

        # ── Modality ablation — one extra population TCN per non-Combined condition ──
        ab_probs = {}
        for cond_name, cond_cols in ABLATION_CONDITIONS.items():
            if cond_name == "Combined":
                ab_probs[cond_name] = None   # filled after online eval
                continue

            cidx      = [SIGNAL_COLS.index(c) for c in cond_cols]
            n_raw_ab  = len(cidx)
            n_feat_ab = n_raw_ab * 5

            Xtr_ab  = apply_features(X_tr[~vmask][:, :, cidx])
            Xval_ab = apply_features(X_tr[ vmask][:, :, cidx])
            Xte_ab  = apply_features(X_te[:, :, cidx])

            sc_ab      = StandardScaler()
            Xtr_ab_sc  = sc_ab.fit_transform(
                Xtr_ab.reshape(-1, n_feat_ab)).reshape(-1, LOOKBACK_S, n_feat_ab)
            Xval_ab_sc = sc_ab.transform(
                Xval_ab.reshape(-1, n_feat_ab)).reshape(-1, LOOKBACK_S, n_feat_ab)
            Xte_ab_sc  = sc_ab.transform(
                Xte_ab.reshape(-1, n_feat_ab)).reshape(-1, LOOKBACK_S, n_feat_ab)

            mdl_ab   = TCN_Attention_Net(n_feat_ab).to(DEVICE)
            opt_ab   = torch.optim.Adam(mdl_ab.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            sched_ab = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ab, T_max=EPOCHS)
            ld_ab    = get_balanced_loader(Xtr_ab_sc, y_tr[~vmask], augment=True)

            best_ab, wt_ab = float("-inf"), None
            for _ in range(EPOCHS):
                mdl_ab.train()
                for xb, yb in ld_ab:
                    loss = focal_loss(mdl_ab(xb.to(DEVICE)), yb.to(DEVICE))
                    opt_ab.zero_grad(); loss.backward(); opt_ab.step()
                sched_ab.step()
                mdl_ab.eval()
                with torch.no_grad():
                    pv = torch.sigmoid(
                        mdl_ab(torch.as_tensor(Xval_ab_sc).to(DEVICE))).cpu().numpy()
                a = safe_auc(y_tr[vmask], pv)
                if a is not None and a > best_ab:
                    best_ab, wt_ab = a, copy.deepcopy(mdl_ab.state_dict())

            if wt_ab is None:
                ab_probs[cond_name] = None
                continue

            mdl_ab.load_state_dict(wt_ab)
            mdl_ab.eval()
            with torch.no_grad():
                logits_ab = mdl_ab(
                    torch.as_tensor(Xte_ab_sc, dtype=torch.float32).to(DEVICE)
                ).cpu().numpy()
            ab_probs[cond_name] = torch.sigmoid(torch.tensor(logits_ab)).numpy()

        # ── Online prequential evaluation ──────────────────────────────────
        pop_scores, onl_scores = online_evaluate_driver(model, Xte_sc, y_te)

        # Combined ablation reuses population scores — no extra training.
        ab_probs["Combined"] = pop_scores

        auc_pop = safe_auc(y_te, pop_scores)
        auc_onl = safe_auc(y_te, onl_scores)

        if auc_pop is None or auc_onl is None:
            print(f"{d:<10} | SKIP — degenerate eval labels")
            continue

        auc_lr  = safe_auc(y_te, lr_probs)
        auc_xgb = safe_auc(y_te, xgb_probs)
        gain    = auc_onl - auc_pop
        pos_r   = y_te.mean() * 100

        reportable = int(y_te.sum()) >= MIN_EVAL_POSITIVES
        row = (f"{d:<10} | {len(y_te):>5} {pos_r:>5.1f}% | "
               f"{auc_lr or 0:>7.4f} {auc_xgb or 0:>7.4f} | "
               f"{auc_pop:>7.4f} {auc_onl:>7.4f} {gain:>+7.4f}")
        if reportable:
            print(row)
        else:
            print(row + f"  ← only {int(y_te.sum())} pos")

        # Accumulate pool arrays (only after all SKIPs are cleared).
        pool_y.append(y_te)
        pool_pop_scores.append(pop_scores)
        pool_onl_scores.append(onl_scores)
        pool_lr_probs.append(lr_probs)
        pool_xgb_probs.append(xgb_probs)
        pool_Xte_sc.append(Xte_sc.copy())
        pool_models.append(copy.deepcopy(model))
        pool_etypes.append(etypes_d)
        for cond_name in ABLATION_CONDITIONS:
            pool_ablation[cond_name].append(ab_probs[cond_name])

        if reportable:
            per_driver_results.append({
                "driver":   str(d),
                "n_win":    int(len(y_te)),
                "pos_rate": float(pos_r),
                "auc_lr":   float(auc_lr)  if auc_lr  is not None else None,
                "auc_xgb":  float(auc_xgb) if auc_xgb is not None else None,
                "auc_pop":  float(auc_pop),
                "auc_onl":  float(auc_onl),
                "gain":     float(gain),
            })

    # ══════════════════════════════════════════════════════════════════════════
    # POOLED EVALUATION  (primary result)
    # ══════════════════════════════════════════════════════════════════════════
    if not pool_y:
        print("No drivers with valid results.")
        return

    print("-" * len(hdr))

    y_pool   = np.concatenate(pool_y)
    pp_all   = np.concatenate(pool_pop_scores)
    po_all   = np.concatenate(pool_onl_scores)
    plr_all  = np.concatenate(pool_lr_probs)
    pxgb_all = np.concatenate(pool_xgb_probs)
    n_pool   = len(y_pool)
    pos_pool = int(y_pool.sum())

    print("\n" + "=" * 70)
    print("POOLED EVALUATION  (primary result)")
    print("=" * 70)
    print(f"Windows: {n_pool}  |  Positives: {pos_pool} ({pos_pool/n_pool*100:.1f}%)  "
          f"|  Drivers: {len(pool_y)}\n")
    print(f"AUC: 95% CI via window-level bootstrap ({N_BOOTSTRAP} resamples).")
    print("Brier / ECE: raw model probabilities (no post-hoc calibration).\n")

    models_eval = [
        ("LR (baseline)",      plr_all),
        ("XGBoost (baseline)", pxgb_all),
        ("TCN-Population",     pp_all),
        ("TCN-Online",         po_all),
    ]

    # ── Table 1: discrimination + calibration ──────────────────────────────
    print(f"  {'Model':<22}  {'AUC':>6}  {'95% CI':^17}  {'AUPRC':>6}  {'Brier':>6}  {'ECE':>6}")
    print(f"  {'-'*22}  {'-'*6}  {'-'*17}  {'-'*6}  {'-'*6}  {'-'*6}")

    pooled_metrics = {}
    for name, probs in models_eval:
        auc   = safe_auc(y_pool,   probs) or float("nan")
        auprc = safe_auprc(y_pool, probs) or float("nan")
        brier = brier_score_loss(y_pool, probs)
        ece   = compute_ece(y_pool, probs)
        lo, hi = bootstrap_auc_ci_windows(y_pool, probs)
        ci_str = f"[{lo:.3f} – {hi:.3f}]"
        print(f"  {name:<22}  {auc:.4f}  {ci_str:^17}  {auprc:.4f}  {brier:.4f}  {ece:.4f}")
        pooled_metrics[name] = {"auc": auc, "auprc": auprc, "brier": brier, "ece": ece,
                                "ci_lo": lo, "ci_hi": hi}

    best_bl_auc = max(pooled_metrics["LR (baseline)"]["auc"],
                      pooled_metrics["XGBoost (baseline)"]["auc"])
    tcn_pop_auc = pooled_metrics["TCN-Population"]["auc"]
    tcn_onl_auc = pooled_metrics["TCN-Online"]["auc"]
    print(f"\n  Best baseline AUC   : {best_bl_auc:.4f}")
    print(f"  TCN-Pop   gain      : {tcn_pop_auc - best_bl_auc:+.4f}")
    print(f"  TCN-Online gain     : {tcn_onl_auc - best_bl_auc:+.4f}")
    print(f"  Online vs Pop       : {tcn_onl_auc - tcn_pop_auc:+.4f}")

    # ── Table 2: threshold-dependent metrics at Youden's-J threshold ───────
    print(f"\n  Threshold-dependent metrics (Youden's-J optimal threshold per model):")
    print(f"  {'Model':<22}  {'Thresh':>7}  {'F1':>6}  {'Prec':>6}  {'Recall':>6}")
    print(f"  {'-'*22}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*6}")
    for name, probs in models_eval:
        if len(np.unique(y_pool)) < 2:
            continue
        f1, prec, rec, thr = threshold_metrics(y_pool, probs)
        print(f"  {name:<22}  {thr:>7.3f}  {f1:>6.4f}  {prec:>6.4f}  {rec:>6.4f}")
        pooled_metrics[name].update({"f1": f1, "precision": prec,
                                     "recall": rec, "threshold": thr})

    # Per-driver AUC summary (driver-level bootstrap CI).
    pop_auc_list = [r["auc_pop"] for r in per_driver_results]
    onl_auc_list = [r["auc_onl"] for r in per_driver_results]
    lo_pop, hi_pop = bootstrap_auc_ci_drivers(pop_auc_list)
    lo_onl, hi_onl = bootstrap_auc_ci_drivers(onl_auc_list)
    print(f"\n  Mean driver AUC — Population : {np.mean(pop_auc_list):.4f}  [{lo_pop:.3f}–{hi_pop:.3f}]")
    print(f"  Mean driver AUC — Online     : {np.mean(onl_auc_list):.4f}  [{lo_onl:.3f}–{hi_onl:.3f}]")
    print(f"  Net driver gain              : {np.mean(onl_auc_list) - np.mean(pop_auc_list):+.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # ONLINE LEARNING DYNAMICS
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ONLINE LEARNING DYNAMICS")
    print("=" * 70)

    # ── Wilcoxon signed-rank test on per-driver gains ──────────────────────
    gains = np.array([r["gain"] for r in per_driver_results])
    print(f"Per-driver AUC gain  (Online − Population):")
    print(f"  n={len(gains)}  mean={gains.mean():+.4f}  median={np.median(gains):+.4f}  "
          f"std={gains.std():.4f}")
    if len(gains) >= 10:
        w_stat, p_two = wilcoxon(gains)
        _, p_greater  = wilcoxon(gains, alternative="greater")
        sig = "***" if p_two < 0.001 else ("**" if p_two < 0.01 else ("*" if p_two < 0.05 else "ns"))
        print(f"  Wilcoxon signed-rank: W={w_stat:.1f}  p(two-sided)={p_two:.4f} {sig}"
              f"  p(Online>Pop)={p_greater:.4f}")
    else:
        print(f"  Wilcoxon: insufficient drivers (n={len(gains)} < 10), skipped.")
        w_stat, p_two, p_greater = float("nan"), float("nan"), float("nan")

    # ── Quartile learning curve ────────────────────────────────────────────
    # For each driver, split windows into N_LC_BINS equal-size quantile groups
    # by temporal position. Compute AUC(pop) and AUC(online) per group.
    # Average across drivers that have both classes in the group.
    print(f"\n  Quartile learning curve  (Q={N_LC_BINS}, averaged over drivers with ≥2 classes per bin):")
    print(f"  {'Quartile':<10}  {'N_drivers':>9}  {'AUC(Pop)':>9}  {'AUC(Online)':>11}  {'Gain':>7}")
    print(f"  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*11}  {'-'*7}")

    lc_pop_bins = [[] for _ in range(N_LC_BINS)]
    lc_onl_bins = [[] for _ in range(N_LC_BINS)]
    lc_curve_data = []

    for y_d, pop_d, onl_d in zip(pool_y, pool_pop_scores, pool_onl_scores):
        n     = len(y_d)
        edges = np.linspace(0, n, N_LC_BINS + 1).astype(int)
        for b in range(N_LC_BINS):
            sl = slice(edges[b], edges[b + 1])
            y_b, pp_b, po_b = y_d[sl], pop_d[sl], onl_d[sl]
            if len(np.unique(y_b)) < 2:
                continue
            lc_pop_bins[b].append(roc_auc_score(y_b, pp_b))
            lc_onl_bins[b].append(roc_auc_score(y_b, po_b))

    for b in range(N_LC_BINS):
        if not lc_pop_bins[b]:
            print(f"  Q{b+1:<9}  SKIP (no driver with both classes in this bin)")
            lc_curve_data.append(None)
            continue
        m_pop = float(np.mean(lc_pop_bins[b]))
        m_onl = float(np.mean(lc_onl_bins[b]))
        nd    = len(lc_pop_bins[b])
        gain_b = m_onl - m_pop
        print(f"  Q{b+1:<9}  {nd:>9}  {m_pop:>9.4f}  {m_onl:>11.4f}  {gain_b:>+7.4f}")
        lc_curve_data.append({"q": b + 1, "n_drivers": nd,
                               "auc_pop": m_pop, "auc_onl": m_onl, "gain": gain_b})

    # ══════════════════════════════════════════════════════════════════════════
    # PERMUTATION FEATURE IMPORTANCE  (TCN-Population, pooled)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PERMUTATION FEATURE IMPORTANCE  (TCN-Population, pooled)")
    print("=" * 70)
    print("Columns permuted per signal: [raw, diff1, roll_mean, roll_std, z-score]")
    print("Permutation: sample-axis shuffle (breaks label correlation).\n")

    baseline_auc = pooled_metrics["TCN-Population"]["auc"]
    perm_results = []

    for sig_idx, sig_name in enumerate(SIGNAL_COLS):
        feat_cols = [sig_idx + k * n_raw for k in range(5)]
        perm_probs_all = []

        for X_sc, mdl in zip(pool_Xte_sc, pool_models):
            X_perm   = X_sc.copy()
            perm_idx = rng_perm.permutation(len(X_perm))
            for c in feat_cols:
                X_perm[:, :, c] = X_perm[perm_idx, :, c]

            mdl.eval()
            with torch.no_grad():
                lp = mdl(torch.as_tensor(X_perm, dtype=torch.float32).to(DEVICE)).cpu().numpy()
            perm_probs_all.append(torch.sigmoid(torch.tensor(lp)).numpy())

        perm_probs = np.concatenate(perm_probs_all)
        auc_perm   = safe_auc(y_pool, perm_probs) or float("nan")
        drop       = baseline_auc - auc_perm
        perm_results.append((sig_name, auc_perm, drop))

    perm_results.sort(key=lambda x: -x[2])
    print(f"  {'Signal':<25}  {'AUC (shuffled)':>14}  {'Drop':>8}")
    print(f"  {'-'*25}  {'-'*14}  {'-'*8}")
    for name, auc_p, drop in perm_results:
        print(f"  {name:<25}  {auc_p:>14.4f}  {drop:>+8.4f}")
    print(f"\n  Baseline (unshuffled) AUC = {baseline_auc:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # MODALITY ABLATION  (TCN-Population per signal subset, pooled)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODALITY ABLATION  (TCN-Population, pooled LOPO-CV)")
    print("=" * 70)
    print("Each condition trains a separate population TCN on the same folds.")
    print("'Combined' reuses the main-loop TCN-Population result.\n")
    print(f"  {'Condition':<20}  {'Signals':>7}  {'AUC':>6}  {'95% CI':^17}  {'AUPRC':>6}  {'Brier':>6}")
    print(f"  {'-'*20}  {'-'*7}  {'-'*6}  {'-'*17}  {'-'*6}  {'-'*6}")

    for cond_name, cond_cols in ABLATION_CONDITIONS.items():
        probs_list = pool_ablation[cond_name]
        valid_pairs = [(p, y) for p, y in zip(probs_list, pool_y) if p is not None]
        n_failed = len(probs_list) - len(valid_pairs)
        if not valid_pairs:
            print(f"  {cond_name:<20}  SKIP (all folds failed)")
            continue
        probs = np.concatenate([p for p, _ in valid_pairs])
        y_ab  = np.concatenate([y for _, y in valid_pairs])
        warn  = f"  [{n_failed} fold(s) skipped]" if n_failed else ""
        auc   = safe_auc(y_ab,   probs) or float("nan")
        auprc = safe_auprc(y_ab, probs) or float("nan")
        brier = brier_score_loss(y_ab, probs)
        lo, hi = bootstrap_auc_ci_windows(y_ab, probs)
        ci_str = f"[{lo:.3f} – {hi:.3f}]"
        n_sig  = len(cond_cols)
        print(f"  {cond_name:<20}  {n_sig:>7}  {auc:.4f}  {ci_str:^17}  {auprc:.4f}  {brier:.4f}{warn}")

    # ══════════════════════════════════════════════════════════════════════════
    # STRATIFIED EVALUATION  (CLC vs non-CLC positive windows)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STRATIFIED EVALUATION  (TCN-Population & Kinematics-only, pooled)")
    print("=" * 70)
    print("Diagnoses whether lateral-signal importance is driven by")
    print("center_line_crossing (CLC) or generalises to other error types.\n")
    print("  CLC-only : future slice contains CLC and no other SEVERITY error.")
    print("  Non-CLC  : future slice contains at least one non-CLC SEVERITY error.")
    print("  Each stratum evaluated against all negatives.\n")

    etypes_pool = np.concatenate(pool_etypes)

    kin_raw   = pool_ablation.get("Kinematics", [])
    kin_valid = [(p, y, e) for p, y, e in zip(kin_raw, pool_y, pool_etypes) if p is not None]
    n_kin_failed = len(kin_raw) - len(kin_valid)

    strat_models = [("TCN-Combined", pp_all, y_pool, etypes_pool)]
    if kin_valid:
        kin_probs  = np.concatenate([p for p, _, _ in kin_valid])
        kin_y      = np.concatenate([y for _, y, _ in kin_valid])
        kin_etypes = np.concatenate([e for _, _, e in kin_valid])
        strat_models.append(("TCN-Kinematics", kin_probs, kin_y, kin_etypes))
        if n_kin_failed:
            print(f"  Note: {n_kin_failed} fold(s) excluded from Kinematics stratum (training failed)\n")

    print(f"  {'Stratum':<12}  {'Model':<18}  {'N_pos':>5}  {'AUC':>6}  {'95% CI':^17}")
    print(f"  {'-'*12}  {'-'*18}  {'-'*5}  {'-'*6}  {'-'*17}")

    for stratum_name in ["CLC-only", "Non-CLC"]:
        for model_name, probs_all, y_strat_pool, etypes_strat in strat_models:
            clc_only_m = np.array([
                "center_line_crossing" in e and not (e - {"center_line_crossing"})
                for e in etypes_strat
            ])
            non_clc_m = np.array([bool(e - {"center_line_crossing"}) for e in etypes_strat])
            neg_m     = (y_strat_pool == 0)
            pos_m     = clc_only_m if stratum_name == "CLC-only" else non_clc_m
            eval_mask = pos_m | neg_m
            y_strat   = pos_m[eval_mask].astype(int)
            n_pos     = int(y_strat.sum())
            if n_pos < 5 or len(np.unique(y_strat)) < 2:
                if model_name == strat_models[0][0]:
                    print(f"  {stratum_name:<12}  {'SKIP (too few positives)'}")
                continue
            p_strat    = probs_all[eval_mask]
            auc_s      = safe_auc(y_strat, p_strat) or float("nan")
            lo_s, hi_s = bootstrap_auc_ci_windows(y_strat, p_strat)
            ci_s       = f"[{lo_s:.3f} – {hi_s:.3f}]"
            print(f"  {stratum_name:<12}  {model_name:<18}  {n_pos:>5}  {auc_s:.4f}  {ci_s:^17}")

    print(f"\n  Reference — TCN-Combined overall AUC = {baseline_auc:.4f}")
    print(f"  If AUC(CLC-only) >> AUC(Non-CLC): lateral signals exploit CLC correlation.")
    print(f"  If AUC(CLC-only) ≈  AUC(Non-CLC): model captures general impairment.")

    # ══════════════════════════════════════════════════════════════════════════
    # PER-DRIVER SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    if per_driver_results:
        rdf = pd.DataFrame(per_driver_results)
        print("\n" + "=" * 70)
        print(f"PER-DRIVER SUMMARY  ({len(rdf)} reportable drivers, ≥{MIN_EVAL_POSITIVES} eval pos)")
        print("=" * 70)

        with pd.option_context("display.float_format", "{:.4f}".format):
            print(rdf[["auc_pop", "auc_onl", "gain", "pos_rate", "n_win"]]
                  .agg(["mean", "median", "std", "min", "max"]).T.to_string())

        n_imp  = (rdf["gain"] > 0).sum()
        n_hurt = (rdf["gain"] < 0).sum()
        print(f"\nOnline improved AUC for {n_imp}/{len(rdf)}, hurt {n_hurt}/{len(rdf)}.")
    else:
        print("\nNo reportable drivers. Rely on pooled evaluation.")

    # ── Save results ──────────────────────────────────────────────────────────
    print(f"\nDrivers evaluated : {len(per_driver_results)}")

    results = {
        "pooled": {m: pooled_metrics[m] for m in pooled_metrics},
        "mean_driver": {
            "auc_population":  float(np.mean(pop_auc_list)) if pop_auc_list else None,
            "auc_online":      float(np.mean(onl_auc_list)) if onl_auc_list else None,
            "net_gain":        float(np.mean(onl_auc_list) - np.mean(pop_auc_list)) if pop_auc_list else None,
            "ci_population":   [float(lo_pop), float(hi_pop)],
            "ci_online":       [float(lo_onl), float(hi_onl)],
            "n_drivers":       len(per_driver_results),
        },
        "wilcoxon": {
            "n":         int(len(gains)),
            "mean_gain": float(gains.mean()),
            "W":         float(w_stat),
            "p_two":     float(p_two),
            "p_greater": float(p_greater),
        },
        "learning_curve": [x for x in lc_curve_data if x is not None],
        "per_driver": per_driver_results,
        "config": {
            "online_lr":            ONLINE_LR,
            "online_steps":         ONLINE_STEPS,
            "replay_buffer_size":   REPLAY_BUFFER_SIZE,
            "online_layers":        ONLINE_LAYERS,
            "lambda_tether_online": LAMBDA_TETHER_ONLINE,
        },
    }
    out_path = OUT_DIR / "tcn_online_personalisation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
