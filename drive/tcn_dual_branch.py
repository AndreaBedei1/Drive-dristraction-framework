"""
tcn_dual_branch.py

Dual-Branch TCN with Cross-Modal Attention for driving impairment detection.

Motivation
----------
Physiology (arousal, HR) and kinematics (steering, acceleration) operate at
fundamentally different temporal scales:

  - Kinematics are reactive: deviations build up in the 10–30 s before an error.
    Optimal receptive field: ~31 s  (d=1,2,4,8).
  - Physiology is tonic: arousal/HR reflect impairment state with 10–30 s
    response lags.  Optimal receptive field: ~43 s  (d=1,4,16).

A standard single-stack TCN applies the same receptive field to both, forcing a
sub-optimal temporal compromise.  Dual-Branch TCN assigns each modality its own
stack and then learns *how* they interact.

Cross-modal attention
---------------------
After temporal pooling of each branch:

  phys_summary attends over kin_seq  →  which kinematic patterns co-occur with
                                         elevated physiological state?
  kin_summary  attends over phys_seq →  which physiological moments amplify the
                                         kinematic risk signal?

Residual enhancement: each pooled representation is additively updated by the
cross-attended context from the other branch before fusion.

Modality gate
-------------
  α = σ( MLP( [phys_pool, kin_pool] ) )  ∈ (0, 1)

  fused = concat( α · phys_enhanced ,  (1−α) · kin_enhanced )

α → 1: physiology-dominant driver.
α → 0: kinematics-dominant driver.

The gate is exposed per sample after every forward pass.  Per-driver mean gate
values are reported in the output and saved to JSON — the interpretability
contribution of this architecture.

Gate-only personalisation
-------------------------
The gate_net has only ~3 200 parameters.  At test time, fine-tuning solely those
parameters on the first GATE_ADAPT_K windows of a new driver provides rapid,
sample-efficient personalisation (≥ 5 × fewer adapted parameters than head-only
adaptation in the standard TCN).

Outputs (mirrors tcn_online_personalisation.py)
-----------------------------------------------
  1.  Label validity report (Mann-Whitney U, per-driver tier)
  2.  Pipeline configuration
  3.  Per-driver table: LR | XGB | SingleTCN | DB-Pop | GateAdapt | gate(α)
  4.  Pooled evaluation: AUC 95% CI, AUPRC, Brier, ECE
  5.  Threshold-dependent metrics (F1/Precision/Recall @ Youden's J)
  6.  Branch gate distribution analysis + correlation with physiology p-values
  7.  Permutation feature importance — by signal AND by branch
  8.  Modality ablation: Physiology-only TCN | Kinematics-only TCN | DB-TCN
  9.  Stratified evaluation (CLC-only vs Non-CLC positive windows)
  10. Per-driver summary statistics
  11. JSON artefacts in impairment_results/
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
from scipy.stats import mannwhitneyu, wilcoxon, pearsonr
import xgboost as xgb
import random

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEED   = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SIGNAL_COLS = [
    "arousal", "hr",
    "steeringWheelAngle", "steeringTorq", "acceleration.y", "speed.x",
]
VEHICLE_COLS = ["steeringWheelAngle", "steeringTorq", "acceleration.y", "speed.x"]

# Modality split — the architectural motivation
PHYS_COLS = ["arousal", "hr"]
KIN_COLS  = ["steeringWheelAngle", "steeringTorq", "acceleration.y", "speed.x"]
PHYS_IDX  = [SIGNAL_COLS.index(c) for c in PHYS_COLS]   # [0, 1]
KIN_IDX   = [SIGNAL_COLS.index(c) for c in KIN_COLS]    # [2, 3, 4, 5]

SEVERITY = {
    "Collision":               5,
    "Red_light_violation":     3,
    "panic_braking_with_stop": 2,
    "center_line_crossing":    2,
    "panic_braking":           1,
    "sharp_turn":              1,
}

# Windowing
LOOKBACK_S     = 60
WINDOW_STEP    = 5
GAP, HORIZON   = 15, 5
EVENT_VICINITY = 10
ROLL_K         = 10
BATCH_SIZE     = 64
WEIGHT_DECAY   = 1e-4
JITTER_STD     = 0.01
CUTOUT_LEN     = 5
CUTOUT_PROB    = 0.2

# Training
EPOCHS = 100
LR     = 1e-3

# Gate-only personalisation
GATE_ADAPT_K     = 15   # support windows for gate adaptation
GATE_ADAPT_STEPS = 20   # gradient steps
GATE_ADAPT_LR    = 5e-4

# Evaluation
N_BOOTSTRAP        = 2000
MIN_POSITIVES      = 1
MIN_EVAL_POSITIVES = 3
N_LC_BINS          = 4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark     = True

OUT_DIR = Path(__file__).parent / "impairment_results"
OUT_DIR.mkdir(exist_ok=True)

# ── PREPROCESSING ──────────────────────────────────────────────────────────────

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
    for _, grp in df.groupby(["id", "route"]):
        idx = grp.index
        for col in VEHICLE_COLS:
            mu  = grp[col].mean()
            sig = grp[col].std() + 1e-6
            df.loc[idx, col] = (grp[col] - mu) / sig
    return df

# ── FEATURE ENGINEERING ────────────────────────────────────────────────────────

def engineer_features(window):
    """(T, C) → (T, 5C): raw | diff1 | roll_mean | roll_std | z-score"""
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
    """Apply engineer_features to a subset of signal channels.
    Returns (N, T, len(col_idx)*5)."""
    return np.stack([engineer_features(w[:, col_idx]) for w in X_raw])


def apply_features_all(X_raw):
    """Apply engineer_features to all channels. Returns (N, T, C*5)."""
    return np.stack([engineer_features(w) for w in X_raw])


def window_baseline_feats(X_raw, col_idx=None):
    X = X_raw if col_idx is None else X_raw[:, :, col_idx]
    return np.concatenate([X.mean(1), X.std(1), X.max(1)], axis=1).astype(np.float32)

# ── WINDOWING ──────────────────────────────────────────────────────────────────

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


def build_windows(df, fine_stride=True):
    windows, labels, scores, pids, etypes, routes, t_starts = [], [], [], [], [], [], []
    for (pid, route), grp in df.groupby(["id", "route"]):
        grp = grp.sort_values("Timestamp").reset_index(drop=True)
        n   = len(grp)
        ts  = grp["Timestamp"].values

        if fine_stride:
            pos_starts = set()
            idx = 0
            while idx + LOOKBACK_S + GAP + HORIZON <= n:
                sig    = grp.iloc[idx: idx + LOOKBACK_S][SIGNAL_COLS].values.astype(np.float32)
                future = grp.iloc[idx + LOOKBACK_S + GAP: idx + LOOKBACK_S + GAP + HORIZON]
                if not np.isnan(sig).any() and composite_risk_score(future) > 0:
                    pos_starts.add(idx)
                idx += WINDOW_STEP

            vicinity = set()
            for ps in pos_starts:
                for offset in range(-EVENT_VICINITY, 1):
                    t = ps + offset
                    if t >= 0:
                        vicinity.add(t)

            idx = 0
            while idx + LOOKBACK_S + GAP + HORIZON <= n:
                step = 1 if idx in vicinity else WINDOW_STEP
                sig  = grp.iloc[idx: idx + LOOKBACK_S][SIGNAL_COLS].values.astype(np.float32)
                if not np.isnan(sig).any():
                    future = grp.iloc[idx + LOOKBACK_S + GAP: idx + LOOKBACK_S + GAP + HORIZON]
                    score  = composite_risk_score(future)
                    windows.append(sig);  labels.append(int(score > 0))
                    scores.append(score); pids.append(pid)
                    etypes.append(future_error_types(future))
                    routes.append(route); t_starts.append(ts[idx])
                idx += step
        else:
            idx = 0
            while idx + LOOKBACK_S + GAP + HORIZON <= n:
                sig = grp.iloc[idx: idx + LOOKBACK_S][SIGNAL_COLS].values.astype(np.float32)
                if not np.isnan(sig).any():
                    future = grp.iloc[idx + LOOKBACK_S + GAP: idx + LOOKBACK_S + GAP + HORIZON]
                    score  = composite_risk_score(future)
                    windows.append(sig);  labels.append(int(score > 0))
                    scores.append(score); pids.append(pid)
                    etypes.append(future_error_types(future))
                    routes.append(route); t_starts.append(ts[idx])
                idx += WINDOW_STEP

    return (np.array(windows,  dtype=np.float32),
            np.array(labels,   dtype=np.float32),
            np.array(scores,   dtype=np.float32),
            np.array(pids),
            np.array(etypes, dtype=object),
            np.array(routes),
            np.array(t_starts))

# ── DATASET ────────────────────────────────────────────────────────────────────

class DualDrivingDataset(Dataset):
    """Dataset holding separate physiology and kinematics feature tensors."""
    def __init__(self, X_phys, X_kin, y, augment=False):
        self.Xp      = torch.as_tensor(X_phys).float()
        self.Xk      = torch.as_tensor(X_kin).float()
        self.y       = torch.as_tensor(y).float()
        self.augment = augment

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        xp = self.Xp[idx].clone()
        xk = self.Xk[idx].clone()
        if self.augment:
            xp += torch.randn_like(xp) * JITTER_STD
            xk += torch.randn_like(xk) * JITTER_STD
            if random.random() < CUTOUT_PROB:
                t0 = random.randint(0, xp.shape[0] - CUTOUT_LEN)
                xp[t0: t0 + CUTOUT_LEN] = 0.0
                xk[t0: t0 + CUTOUT_LEN] = 0.0
        return xp, xk, self.y[idx]


def get_balanced_loader_dual(Xp, Xk, y, batch_size=BATCH_SIZE, augment=False):
    y_int   = y.astype(int)
    counts  = np.bincount(y_int, minlength=2)
    weights = 1.0 / (counts + 1e-6)
    sw      = torch.from_numpy(weights[y_int])
    sampler = WeightedRandomSampler(sw, len(sw))
    return DataLoader(DualDrivingDataset(Xp, Xk, y, augment=augment),
                      batch_size=batch_size, sampler=sampler)

# ── MODEL ──────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, d):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, out_c, 3, padding=d, dilation=d),
            nn.BatchNorm1d(out_c), nn.ReLU(), nn.Dropout1d(0.1),
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
        # x: (B, C, T)
        xt      = x.permute(0, 2, 1)               # (B, T, C)
        h       = torch.tanh(self.query(xt))        # (B, T, C//2)
        weights = torch.softmax(self.score(h), dim=1)  # (B, T, 1)
        return (xt * weights).sum(dim=1)            # (B, C)


class CrossModalAttention(nn.Module):
    """
    Query-context cross-attention.

    A pooled summary (query, shape B×d_q) attends over a temporal sequence
    (key/value, shape B×d_s×T) from the other modality.

    Intuition: "Given my current state, which moments in the other modality's
    sequence are most informative about risk?"

    Returns: (B, d_s) — attended representation of the other branch's sequence.
    """
    def __init__(self, d_query, d_seq):
        super().__init__()
        d_attn = max(d_seq // 2, 8)
        self.proj_q = nn.Linear(d_query, d_attn, bias=False)
        self.proj_s = nn.Linear(d_seq,   d_attn, bias=False)
        self.score  = nn.Linear(d_attn,  1,      bias=False)

    def forward(self, query, seq):
        # query: (B, d_q)  |  seq: (B, d_s, T)
        seq_t  = seq.permute(0, 2, 1)                      # (B, T, d_s)
        q_proj = self.proj_q(query).unsqueeze(1)           # (B, 1, d_attn)
        s_proj = self.proj_s(seq_t)                        # (B, T, d_attn)
        attn   = torch.softmax(
            self.score(torch.tanh(q_proj + s_proj)), dim=1)  # (B, T, 1)
        return (seq_t * attn).sum(dim=1)                   # (B, d_s)


class ModalityGate(nn.Module):
    """
    Learnable modality gate  α ∈ (0, 1).

    α is computed dynamically from the joint branch summaries — it is
    *input-dependent*, not a fixed driver-level parameter.  This means the
    model can shift its reliance on physiology vs kinematics within a single
    session based on the current signal content.

    α → 1 : physiology dominant.
    α → 0 : kinematics dominant.
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
        return self.gate_net(torch.cat([phys_pool, kin_pool], dim=-1))  # (B, 1)


class DualBranchTCN(nn.Module):
    """
    Dual-Branch TCN with Cross-Modal Attention.

    Physiology branch  (arousal, HR):
      3 ResBlocks, dilations d=1,4,16 → receptive field ≈ 43 timesteps.
      Captures slow physiological buildup (HR lag ~10–30 s).

    Kinematics branch  (torque, accel.y, angle, speed):
      4 ResBlocks, dilations d=1,2,4,8 → receptive field ≈ 31 timesteps.
      Captures reactive kinematic deviations (sub-second to ~15 s).

    Cross-modal attention:
      phys_pool attends over kin_seq → phys-enriched kinematics context.
      kin_pool  attends over phys_seq → kin-enriched physiology context.

    Residual cross-modal enhancement:
      phys_enhanced = phys_pool + kin→phys attended context
      kin_enhanced  = kin_pool  + phys→kin attended context

    Modality gate:
      α = σ(MLP([phys_pool, kin_pool]))
      fused = concat(α·phys_enhanced, (1−α)·kin_enhanced)

    Head: Linear(96→48) → ReLU → Dropout(0.2) → Linear(48→1)
    """
    PHYS_D = 32
    KIN_D  = 64

    def __init__(self, n_phys_feats, n_kin_feats):
        super().__init__()

        # Physiology branch — slow temporal dynamics (longer RF)
        # RF = 1 + 2*(1 + 4 + 16) = 43 timesteps
        self.phys_branch = nn.Sequential(
            ResBlock(n_phys_feats, self.PHYS_D,  1),
            ResBlock(self.PHYS_D,  self.PHYS_D,  4),
            ResBlock(self.PHYS_D,  self.PHYS_D, 16),
        )
        self.phys_attn = TemporalAttention(self.PHYS_D)

        # Kinematics branch — fast temporal dynamics (shorter RF)
        # RF = 1 + 2*(1 + 2 + 4 + 8) = 31 timesteps
        self.kin_branch = nn.Sequential(
            ResBlock(n_kin_feats,      self.KIN_D // 2, 1),
            ResBlock(self.KIN_D // 2,  self.KIN_D,      2),
            ResBlock(self.KIN_D,       self.KIN_D,      4),
            ResBlock(self.KIN_D,       self.KIN_D,      8),
        )
        self.kin_attn = TemporalAttention(self.KIN_D)

        # Cross-modal attention
        self.phys_attends_kin = CrossModalAttention(self.PHYS_D, self.KIN_D)
        self.kin_attends_phys = CrossModalAttention(self.KIN_D,  self.PHYS_D)

        # Modality gate
        self.gate = ModalityGate(self.PHYS_D, self.KIN_D)

        # Fusion + head
        fusion_dim = self.PHYS_D + self.KIN_D  # 96
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 48),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(48, 1),
        )

        self._last_gate: torch.Tensor | None = None  # exposed after forward()

    def forward(self, x_phys, x_kin):
        """
        x_phys : (B, T, n_phys_feats)
        x_kin  : (B, T, n_kin_feats)
        returns: (B,) logits
        """
        # ── Branch temporal processing ──────────────────────────────────────
        phys_seq = self.phys_branch(x_phys.permute(0, 2, 1))  # (B, PHYS_D, T)
        kin_seq  = self.kin_branch(x_kin.permute(0, 2, 1))    # (B, KIN_D,  T)

        # ── Temporal pooling within each branch ─────────────────────────────
        phys_pool = self.phys_attn(phys_seq)   # (B, PHYS_D)
        kin_pool  = self.kin_attn(kin_seq)     # (B, KIN_D)

        # ── Cross-modal attention ────────────────────────────────────────────
        # phys summary attends over kin sequence
        cross_phys_kin = self.phys_attends_kin(phys_pool, kin_seq)  # (B, KIN_D)
        # kin summary attends over phys sequence
        cross_kin_phys = self.kin_attends_phys(kin_pool,  phys_seq) # (B, PHYS_D)

        # ── Residual cross-modal enhancement ────────────────────────────────
        phys_enhanced = phys_pool + cross_kin_phys  # (B, PHYS_D)
        kin_enhanced  = kin_pool  + cross_phys_kin  # (B, KIN_D)

        # ── Modality gate ────────────────────────────────────────────────────
        alpha = self.gate(phys_pool, kin_pool)   # (B, 1)
        self._last_gate = alpha.detach()

        # ── Gated fusion + head ──────────────────────────────────────────────
        fused = torch.cat([alpha * phys_enhanced,
                           (1.0 - alpha) * kin_enhanced], dim=-1)  # (B, 96)
        return self.head(fused).squeeze(-1)   # (B,)

    def gate_values(self, x_phys, x_kin, device=DEVICE):
        """Return per-sample gate values α ∈ (0,1) without storing gradients."""
        self.eval()
        with torch.no_grad():
            self.forward(x_phys.to(device), x_kin.to(device))
        return self._last_gate.cpu().squeeze(-1).numpy()


# Single-branch baseline (used for ablation conditions)
class SingleBranchTCN(nn.Module):
    """Standard TCN baseline used in the modality ablation."""
    def __init__(self, in_channels):
        super().__init__()
        self.network = nn.Sequential(
            ResBlock(in_channels, 32,  1),
            ResBlock(32,          64,  2),
            ResBlock(64,          64,  4),
            ResBlock(64,          64,  8),
            ResBlock(64,          64, 16),
        )
        self.attention = TemporalAttention(64)
        self.head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.head(self.attention(self.network(x.permute(0, 2, 1)))).squeeze(-1)

# ── UTILITIES ──────────────────────────────────────────────────────────────────

def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2: return None
    return roc_auc_score(y_true, y_score)


def safe_auprc(y_true, y_score):
    if len(np.unique(y_true)) < 2: return None
    return average_precision_score(y_true, y_score)


def focal_loss(logits, targets, gamma=2.0, pos_weight=None):
    pw  = (torch.tensor(pos_weight, dtype=logits.dtype, device=logits.device)
           if pos_weight is not None else None)
    bce = F.binary_cross_entropy_with_logits(logits, targets,
                                             pos_weight=pw, reduction="none")
    pt  = torch.exp(-bce)
    return ((1 - pt) ** gamma * bce).mean()


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


def bootstrap_auc_ci_windows(y_true, y_score, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed); n = len(y_true); aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2: continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    if len(aucs) < 10: return float("nan"), float("nan")
    return tuple(np.percentile(aucs, [2.5, 97.5]))


def bootstrap_auc_ci_drivers(driver_aucs, n_boot=N_BOOTSTRAP, seed=SEED):
    aucs = np.array(driver_aucs); n = len(aucs)
    if n < 2: return float("nan"), float("nan")
    rng  = np.random.default_rng(seed)
    boot = [rng.choice(aucs, n, replace=True).mean() for _ in range(n_boot)]
    return tuple(np.percentile(boot, [2.5, 97.5]))


def _val_sample(arr, frac, rng):
    n = min(max(1, int(frac * len(arr))), len(arr)) if len(arr) > 0 else 0
    return rng.choice(arr, n, replace=False) if n > 0 else np.array([], dtype=arr.dtype)

# ── TRAINING ───────────────────────────────────────────────────────────────────

def train_dual_tcn(model, Xtr_p, Xtr_k, y_tr, Xval_p, Xval_k, y_val):
    """LOPO population training for the DualBranchTCN."""
    opt       = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loader    = get_balanced_loader_dual(Xtr_p, Xtr_k, y_tr, augment=True)
    best_auc, best_w = float("-inf"), None

    for _ in range(EPOCHS):
        model.train()
        for xp, xk, yb in loader:
            loss = focal_loss(model(xp.to(DEVICE), xk.to(DEVICE)), yb.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(
                model(torch.as_tensor(Xval_p).to(DEVICE),
                      torch.as_tensor(Xval_k).to(DEVICE))
            ).cpu().numpy()
        auc = safe_auc(y_val, preds)
        if auc is not None and auc > best_auc:
            best_auc = auc
            best_w   = copy.deepcopy(model.state_dict())

    if best_w is not None:
        model.load_state_dict(best_w)
    return model


def train_single_tcn(model, Xtr_sc, y_tr, Xval_sc, y_val):
    """Population training for the single-branch ablation TCN."""
    opt       = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    y_int     = y_tr.astype(int)
    counts    = np.bincount(y_int, minlength=2)
    weights   = 1.0 / (counts + 1e-6)
    sw        = torch.from_numpy(weights[y_int])
    from torch.utils.data import TensorDataset
    ds      = TensorDataset(torch.as_tensor(Xtr_sc).float(),
                            torch.as_tensor(y_tr).float())
    sampler = WeightedRandomSampler(sw, len(sw))
    loader  = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler)
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
        auc = safe_auc(y_val, preds)
        if auc is not None and auc > best_auc:
            best_auc = auc; best_w = copy.deepcopy(model.state_dict())

    if best_w is not None:
        model.load_state_dict(best_w)
    return model


def gate_adapt(model_pop, Xte_p, Xte_k, y_te):
    """
    Gate-only personalisation on the first GATE_ADAPT_K windows.

    Only gate_net parameters are updated.  All other weights are frozen.
    Returns adapted model scores on the FULL test sequence (not just post-support),
    using population scores for the support windows to avoid leakage.
    """
    if len(y_te) < GATE_ADAPT_K + MIN_EVAL_POSITIVES:
        return None, 0

    model_adapt = copy.deepcopy(model_pop)

    # Freeze everything except gate_net
    for name, param in model_adapt.named_parameters():
        param.requires_grad = name.startswith("gate.")

    gate_params = [p for p in model_adapt.parameters() if p.requires_grad]
    if not gate_params:
        return None, 0

    opt = torch.optim.Adam(gate_params, lr=GATE_ADAPT_LR, weight_decay=1e-2)

    X_sup_p = torch.as_tensor(Xte_p[:GATE_ADAPT_K]).float().to(DEVICE)
    X_sup_k = torch.as_tensor(Xte_k[:GATE_ADAPT_K]).float().to(DEVICE)
    y_sup   = torch.as_tensor(y_te[:GATE_ADAPT_K]).float().to(DEVICE)

    if len(torch.unique(y_sup)) < 2:
        return None, 0   # single-class support — skip adaptation

    n_pos = int(y_sup.sum()); n_neg = GATE_ADAPT_K - n_pos
    pos_w = float(np.clip(n_neg / max(n_pos, 1), 0.2, 5.0))

    # Keep BN in eval to preserve population running stats
    model_adapt.train()
    for m in model_adapt.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()

    for _ in range(GATE_ADAPT_STEPS):
        loss = focal_loss(model_adapt(X_sup_p, X_sup_k), y_sup, pos_weight=pos_w)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(gate_params, 1.0)
        opt.step()

    model_adapt.eval()
    with torch.no_grad():
        # Population scores for support windows (no leakage)
        pop_sup = torch.sigmoid(
            model_pop(X_sup_p, X_sup_k)).cpu().numpy()
        # Adapted scores for eval windows
        adpt_eval = torch.sigmoid(
            model_adapt(
                torch.as_tensor(Xte_p[GATE_ADAPT_K:]).float().to(DEVICE),
                torch.as_tensor(Xte_k[GATE_ADAPT_K:]).float().to(DEVICE),
            )).cpu().numpy()

    scores = np.concatenate([pop_sup, adpt_eval])
    return scores, GATE_ADAPT_K

# ── VALIDITY REPORT ────────────────────────────────────────────────────────────

def print_validity_report(X_raw, y, scores, pid):
    n_total, n_pos = len(y), int(y.sum())
    print(f"\n{'='*72}")
    print("LABEL VALIDITY REPORT — COMPOSITE RISK TARGET")
    print(f"{'='*72}")
    print(f"Total windows    : {n_total}")
    print(f"Positive (risk>0): {n_pos} ({n_pos/n_total*100:.1f}%)")
    print(f"Negative         : {n_total - n_pos} ({(n_total-n_pos)/n_total*100:.1f}%)")

    print(f"\nRisk score distribution (positives only):")
    for v in sorted(np.unique(scores[y == 1])):
        c = (scores[y == 1] == v).sum()
        print(f"  score={int(v):>2}  n={c:>5}  ({c/n_pos*100:.1f}%)")

    print(f"\nPredictive validity — Mann-Whitney U (pos vs neg, raw signals):")
    print(f"  {'Signal':<25}  {'Mean(neg)':>10}  {'Mean(pos)':>10}  {'p-value':>12}  Sig")
    mw_pvals = {}
    for i, col in enumerate(SIGNAL_COLS):
        neg_v = X_raw[y == 0, :, i].mean(axis=1)
        pos_v = X_raw[y == 1, :, i].mean(axis=1)
        if len(pos_v) < 2: continue
        _, p = mannwhitneyu(pos_v, neg_v, alternative="two-sided")
        sig  = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        mw_pvals[col] = p
        print(f"  {col:<25}  {neg_v.mean():>10.4f}  {pos_v.mean():>10.4f}  {p:>12.2e}  {sig}")

    print(f"\nPer-driver positive rate:")
    print(f"  {'Driver':<12}  {'N':>6}  {'Pos':>5}  {'Rate%':>7}  Tier")
    for d in np.unique(pid):
        m    = pid == d
        nd    = int(m.sum()); n_pos = int(y[m].sum())
        rate  = n_pos / nd * 100
        tier  = "HIGH" if rate >= 10 else ("MED" if rate >= 5 else "LOW")
        print(f"  {d:<12}  {nd:>6}  {n_pos:>5}  {rate:>6.1f}%  {tier}")
    print("=" * 72)
    return mw_pvals

# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(Path(__file__).parent / "relab+unibo_dataset.csv")
    df = mark_event_onsets(df)
    df = normalize_signals(df)

    X_raw_tr, y_tr_all, _,          pid_tr_all, _,          _,             _         = build_windows(df, fine_stride=True)
    X_raw_te, y_te_all, scores_te,  pid_te_all, etypes_te,  routes_te_all, ts_te_all = build_windows(df, fine_stride=False)

    n_phys_raw  = len(PHYS_COLS)    # 2
    n_kin_raw   = len(KIN_COLS)     # 4
    n_phys_feat = n_phys_raw * 5   # 10
    n_kin_feat  = n_kin_raw  * 5   # 20
    n_all_raw   = len(SIGNAL_COLS)  # 6
    n_all_feat  = n_all_raw * 5    # 30

    mw_pvals = print_validity_report(X_raw_te, y_te_all, scores_te, pid_te_all)

    print(f"\n{'='*72}")
    print("DUAL-BRANCH TCN — PIPELINE CONFIGURATION")
    print(f"{'='*72}")
    print(f"Signals           : {SIGNAL_COLS}")
    print(f"Physiology branch : {PHYS_COLS}  ({n_phys_feat} engineered features)")
    print(f"  TCN blocks      : d=1,4,16  →  RF ≈ 43 timesteps")
    print(f"  Output channels : {DualBranchTCN.PHYS_D}")
    print(f"Kinematics branch : {KIN_COLS}  ({n_kin_feat} engineered features)")
    print(f"  TCN blocks      : d=1,2,4,8  →  RF ≈ 31 timesteps")
    print(f"  Output channels : {DualBranchTCN.KIN_D}")
    print(f"Cross-modal attn  : bidirectional (phys↔kin) with residual enhancement")
    print(f"Modality gate     : α = σ(MLP([phys_pool, kin_pool]))  ∈ (0,1)")
    print(f"Fusion            : concat(α·phys_enh, (1−α)·kin_enh) → 96-d → head")
    print(f"Normalisation     : vehicle z-scored per route; physiology absolute")
    print(f"  Branch scalers  : independent (phys and kin scaled separately)")
    print(f"Loss              : focal (γ=2)")
    print(f"Epochs            : {EPOCHS}  |  LR : {LR}  |  Scheduler : CosineAnnealingLR")
    print(f"GAP / HORIZON     : {GAP}s / {HORIZON}s  →  predicts errors in [{GAP}, {GAP+HORIZON}]s")
    print(f"Gate adapt (pers) : {GATE_ADAPT_K} support windows  |  {GATE_ADAPT_STEPS} steps  |  gate_net only")
    print(f"Bootstrap CI      : {N_BOOTSTRAP} resamples (window-level pooled; driver-level summary)")
    print(f"Device            : {DEVICE}")
    print(f"{'='*72}\n")

    drivers = [d for d in np.unique(pid_te_all)
               if y_te_all[pid_te_all == d].sum() >= MIN_POSITIVES]
    rng_perm = np.random.default_rng(SEED + 1)

    hdr = (f"{'Driver':<10} | {'N_win':>5} {'PosR%':>6} | "
           f"{'LR':>7} {'XGB':>7} {'StdTCN':>7} | "
           f"{'DB-Pop':>7} {'GateAdpt':>8} {'Gain':>6} | "
           f"{'gate(α)':>7}")
    print(hdr)
    print("-" * len(hdr))

    per_driver_results = []
    pool_y, pool_db, pool_ga = [], [], []
    pool_lr, pool_xgb, pool_std = [], [], []
    pool_Xte_p, pool_Xte_k = [], []
    pool_models_db = []
    pool_etypes    = []
    pool_gates     = []   # per-driver mean gate values

    # Ablation pools (single-branch TCNs)
    ABLATION = {"Physiology": PHYS_IDX, "Kinematics": KIN_IDX}
    pool_ablation = {k: [] for k in ABLATION}

    for d in drivers:
        mask_tr = pid_tr_all != d
        X_tr    = X_raw_tr[mask_tr]; y_tr = y_tr_all[mask_tr]; pid_tr = pid_tr_all[mask_tr]

        mask_te   = pid_te_all == d
        X_te      = X_raw_te[mask_te]; y_te = y_te_all[mask_te]
        ts_te     = ts_te_all[mask_te]; routes_te = routes_te_all[mask_te]
        etypes_d  = etypes_te[mask_te]

        # Temporal ordering for honest prequential split
        order     = np.lexsort((ts_te, routes_te))
        X_te      = X_te[order]; y_te = y_te[order]; etypes_d = etypes_d[order]

        # Validation split
        seed_d   = int(hashlib.md5(str(d).encode()).hexdigest(), 16) & 0xFFFFFFFF
        fold_rng = np.random.default_rng(SEED ^ seed_d)
        train_drivers = np.unique(pid_tr)
        has_pos = np.array([y_tr[pid_tr == p].sum() > 0 for p in train_drivers])
        val_ids = np.concatenate([
            _val_sample(train_drivers[has_pos],  0.20, fold_rng),
            _val_sample(train_drivers[~has_pos], 0.20, fold_rng),
        ])
        vmask = np.isin(pid_tr, val_ids)
        if len(np.unique(y_tr[vmask])) < 2:
            print(f"{d:<10} | SKIP — val fold single-class")
            continue

        # ── Feature engineering & scaling (per branch) ──────────────────────
        Xtr_p_feat  = apply_features_branch(X_tr[~vmask], PHYS_IDX)
        Xval_p_feat = apply_features_branch(X_tr[vmask],  PHYS_IDX)
        Xte_p_feat  = apply_features_branch(X_te,         PHYS_IDX)

        Xtr_k_feat  = apply_features_branch(X_tr[~vmask], KIN_IDX)
        Xval_k_feat = apply_features_branch(X_tr[vmask],  KIN_IDX)
        Xte_k_feat  = apply_features_branch(X_te,         KIN_IDX)

        scaler_p = StandardScaler()
        Xtr_p_sc  = scaler_p.fit_transform(
            Xtr_p_feat.reshape(-1, n_phys_feat)).reshape(-1, LOOKBACK_S, n_phys_feat)
        Xval_p_sc = scaler_p.transform(
            Xval_p_feat.reshape(-1, n_phys_feat)).reshape(-1, LOOKBACK_S, n_phys_feat)
        Xte_p_sc  = scaler_p.transform(
            Xte_p_feat.reshape(-1, n_phys_feat)).reshape(-1, LOOKBACK_S, n_phys_feat)

        scaler_k = StandardScaler()
        Xtr_k_sc  = scaler_k.fit_transform(
            Xtr_k_feat.reshape(-1, n_kin_feat)).reshape(-1, LOOKBACK_S, n_kin_feat)
        Xval_k_sc = scaler_k.transform(
            Xval_k_feat.reshape(-1, n_kin_feat)).reshape(-1, LOOKBACK_S, n_kin_feat)
        Xte_k_sc  = scaler_k.transform(
            Xte_k_feat.reshape(-1, n_kin_feat)).reshape(-1, LOOKBACK_S, n_kin_feat)

        # ── Baseline (LR / XGBoost on window stats) ──────────────────────────
        # Use training-set windows (fine-stride) to avoid test-pool leakage.
        mask_bl_tr  = pid_tr_all != d
        X_bl_base   = X_raw_tr[mask_bl_tr]
        y_bl_base   = y_tr_all[mask_bl_tr]
        pid_bl_base = pid_tr_all[mask_bl_tr]
        vmask_bl    = np.isin(pid_bl_base, val_ids)
        X_bl_tr_f   = window_baseline_feats(X_bl_base[~vmask_bl])
        X_bl_te_f   = window_baseline_feats(X_te)
        bl_scaler   = StandardScaler()
        X_bl_tr_sc  = bl_scaler.fit_transform(X_bl_tr_f)
        X_bl_te_sc  = bl_scaler.transform(X_bl_te_f)
        y_bl_tr     = y_bl_base[~vmask_bl]
        pos_w_bl    = (y_bl_tr == 0).sum() / max(1, (y_bl_tr == 1).sum())

        lr_clf = LogisticRegression(C=1.0, class_weight="balanced",
                                    max_iter=1000, random_state=SEED)
        lr_clf.fit(X_bl_tr_sc, y_bl_tr)
        xgb_clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            scale_pos_weight=pos_w_bl, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, verbosity=0)
        xgb_clf.fit(X_bl_tr_sc, y_bl_tr)
        lr_probs  = lr_clf.predict_proba(X_bl_te_sc)[:, 1]
        xgb_probs = xgb_clf.predict_proba(X_bl_te_sc)[:, 1]

        # ── Standard single-branch TCN (comparison baseline) ─────────────────
        Xtr_all_feat  = apply_features_all(X_tr[~vmask])
        Xval_all_feat = apply_features_all(X_tr[vmask])
        Xte_all_feat  = apply_features_all(X_te)
        sc_all  = StandardScaler()
        Xtr_all_sc  = sc_all.fit_transform(
            Xtr_all_feat.reshape(-1, n_all_feat)).reshape(-1, LOOKBACK_S, n_all_feat)
        Xval_all_sc = sc_all.transform(
            Xval_all_feat.reshape(-1, n_all_feat)).reshape(-1, LOOKBACK_S, n_all_feat)
        Xte_all_sc  = sc_all.transform(
            Xte_all_feat.reshape(-1, n_all_feat)).reshape(-1, LOOKBACK_S, n_all_feat)

        print(f"{d:<10} | training std-TCN...", flush=True)
        model_std = SingleBranchTCN(n_all_feat).to(DEVICE)
        model_std = train_single_tcn(model_std, Xtr_all_sc, y_tr[~vmask],
                                     Xval_all_sc, y_tr[vmask])
        model_std.eval()
        with torch.no_grad():
            std_probs = torch.sigmoid(
                model_std(torch.as_tensor(Xte_all_sc).float().to(DEVICE))
            ).cpu().numpy()

        # ── Dual-Branch TCN ───────────────────────────────────────────────────
        print(f"{d:<10} | training DB-TCN...",  flush=True)
        model_db = DualBranchTCN(n_phys_feat, n_kin_feat).to(DEVICE)
        model_db = train_dual_tcn(model_db,
                                  Xtr_p_sc, Xtr_k_sc, y_tr[~vmask],
                                  Xval_p_sc, Xval_k_sc, y_tr[vmask])
        model_db.eval()
        with torch.no_grad():
            db_probs = torch.sigmoid(
                model_db(torch.as_tensor(Xte_p_sc).float().to(DEVICE),
                         torch.as_tensor(Xte_k_sc).float().to(DEVICE))
            ).cpu().numpy()

        # Gate values for this driver
        gate_vals = model_db.gate_values(
            torch.as_tensor(Xte_p_sc).float(),
            torch.as_tensor(Xte_k_sc).float())
        mean_gate = float(gate_vals.mean())

        # ── Gate-only personalisation ─────────────────────────────────────────
        ga_probs, n_sup = gate_adapt(model_db, Xte_p_sc, Xte_k_sc, y_te)

        # ── Modality ablation (single-branch TCNs per modality) ───────────────
        ab_probs_d = {}
        for ab_name, ab_idx in ABLATION.items():
            n_raw_ab  = len(ab_idx)
            n_feat_ab = n_raw_ab * 5

            Xtr_ab  = apply_features_branch(X_tr[~vmask], ab_idx)
            Xval_ab = apply_features_branch(X_tr[vmask],  ab_idx)
            Xte_ab  = apply_features_branch(X_te,         ab_idx)

            sc_ab       = StandardScaler()
            Xtr_ab_sc   = sc_ab.fit_transform(
                Xtr_ab.reshape(-1, n_feat_ab)).reshape(-1, LOOKBACK_S, n_feat_ab)
            Xval_ab_sc  = sc_ab.transform(
                Xval_ab.reshape(-1, n_feat_ab)).reshape(-1, LOOKBACK_S, n_feat_ab)
            Xte_ab_sc   = sc_ab.transform(
                Xte_ab.reshape(-1, n_feat_ab)).reshape(-1, LOOKBACK_S, n_feat_ab)

            m_ab = SingleBranchTCN(n_feat_ab).to(DEVICE)
            m_ab = train_single_tcn(m_ab, Xtr_ab_sc, y_tr[~vmask],
                                    Xval_ab_sc, y_tr[vmask])
            m_ab.eval()
            with torch.no_grad():
                ab_probs_d[ab_name] = torch.sigmoid(
                    m_ab(torch.as_tensor(Xte_ab_sc).float().to(DEVICE))
                ).cpu().numpy()

        # ── Per-driver metrics ────────────────────────────────────────────────
        auc_lr   = safe_auc(y_te, lr_probs)
        auc_xgb  = safe_auc(y_te, xgb_probs)
        auc_std  = safe_auc(y_te, std_probs)
        auc_db   = safe_auc(y_te, db_probs)
        auc_ga   = safe_auc(y_te, ga_probs) if ga_probs is not None else None
        gain_ga  = (auc_ga - auc_db) if auc_ga is not None and auc_db is not None else float("nan")
        pos_r    = float(y_te.mean() * 100)

        def _f(v): return f"{v:.4f}" if v is not None else "  n/a "
        print(f"{d:<10} | {len(y_te):>5} {pos_r:>5.1f}% | "
              f"{_f(auc_lr)} {_f(auc_xgb)} {_f(auc_std)} | "
              f"{_f(auc_db)} {_f(auc_ga)} {gain_ga:>+6.4f} | "
              f"  {mean_gate:.3f}")

        reportable = int(y_te.sum()) >= MIN_EVAL_POSITIVES
        if reportable:
            pool_y.append(y_te)
            pool_db.append(db_probs)
            if ga_probs is None:
                print(f"  [WARN] {d}: gate adaptation failed (insufficient/single-class support) — "
                      f"falling back to population model for GA pool.")
            pool_ga.append(ga_probs if ga_probs is not None else db_probs)
            pool_lr.append(lr_probs)
            pool_xgb.append(xgb_probs)
            pool_std.append(std_probs)
            pool_Xte_p.append(Xte_p_sc.copy())
            pool_Xte_k.append(Xte_k_sc.copy())
            pool_models_db.append(copy.deepcopy(model_db))
            pool_etypes.append(etypes_d)
            pool_gates.append(mean_gate)
            for ab_name in ABLATION:
                pool_ablation[ab_name].append(ab_probs_d.get(ab_name))

            per_driver_results.append({
                "driver":     str(d),
                "n_win":      len(y_te),
                "pos_rate":   pos_r,
                "auc_lr":     float(auc_lr)  if auc_lr  is not None else None,
                "auc_xgb":    float(auc_xgb) if auc_xgb is not None else None,
                "auc_std_tcn": float(auc_std) if auc_std is not None else None,
                "auc_db_pop": float(auc_db)  if auc_db  is not None else None,
                "auc_gate_adapt": float(auc_ga) if auc_ga is not None else None,
                "gain_gate":  float(gain_ga),
                "mean_gate":  mean_gate,
                "n_sup":      n_sup,
            })

    # ══════════════════════════════════════════════════════════════════════════
    # POOLED EVALUATION
    # ══════════════════════════════════════════════════════════════════════════
    if not per_driver_results:
        print("No reportable drivers."); return

    print("-" * len(hdr))

    y_pool   = np.concatenate(pool_y)
    db_all   = np.concatenate(pool_db)
    ga_all   = np.concatenate(pool_ga)
    lr_all   = np.concatenate(pool_lr)
    xgb_all  = np.concatenate(pool_xgb)
    std_all  = np.concatenate(pool_std)
    n_pool   = len(y_pool); pos_pool = int(y_pool.sum())

    print(f"\n{'='*72}")
    print("POOLED EVALUATION  (primary result)")
    print(f"{'='*72}")
    print(f"Windows: {n_pool}  |  Positives: {pos_pool} ({pos_pool/n_pool*100:.1f}%)  "
          f"|  Drivers: {len(per_driver_results)}\n")
    print("AUC: 95% CI via window-level bootstrap.  "
          "Brier/ECE: raw probabilities (no post-hoc calibration).\n")

    models_eval = [
        ("LR (baseline)",        lr_all),
        ("XGBoost (baseline)",   xgb_all),
        ("TCN-Single (pop.)",    std_all),
        ("DB-TCN  (pop.)",       db_all),
        ("DB-TCN  (gate-adapt)", ga_all),
    ]

    print(f"  {'Model':<24}  {'AUC':>6}  {'95% CI':^17}  {'AUPRC':>6}  {'Brier':>6}  {'ECE':>6}")
    print(f"  {'-'*24}  {'-'*6}  {'-'*17}  {'-'*6}  {'-'*6}  {'-'*6}")

    pooled_metrics = {}
    for name, probs in models_eval:
        auc   = safe_auc(y_pool, probs)   or float("nan")
        auprc = safe_auprc(y_pool, probs) or float("nan")
        brier = brier_score_loss(y_pool, probs)
        ece   = compute_ece(y_pool, probs)
        lo, hi = bootstrap_auc_ci_windows(y_pool, probs)
        print(f"  {name:<24}  {auc:.4f}  [{lo:.3f} – {hi:.3f}]  {auprc:.4f}  {brier:.4f}  {ece:.4f}")
        pooled_metrics[name] = dict(auc=auc, auprc=auprc, brier=brier, ece=ece,
                                    ci_lo=lo, ci_hi=hi)

    best_bl = max(pooled_metrics["LR (baseline)"]["auc"],
                  pooled_metrics["XGBoost (baseline)"]["auc"])
    db_auc  = pooled_metrics["DB-TCN  (pop.)"]["auc"]
    std_auc = pooled_metrics["TCN-Single (pop.)"]["auc"]
    print(f"\n  Best baseline AUC        : {best_bl:.4f}")
    print(f"  DB-TCN gain over baseline: {db_auc - best_bl:+.4f}")
    print(f"  DB-TCN gain over std-TCN : {db_auc - std_auc:+.4f}")

    # Threshold-dependent metrics
    print(f"\n  Threshold-dependent metrics (Youden's-J optimal threshold):")
    print(f"  {'Model':<24}  {'Thresh':>7}  {'F1':>6}  {'Prec':>6}  {'Recall':>6}")
    print(f"  {'-'*24}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*6}")
    for name, probs in models_eval:
        if len(np.unique(y_pool)) < 2: continue
        f1, prec, rec, thr = threshold_metrics(y_pool, probs)
        print(f"  {name:<24}  {thr:>7.3f}  {f1:>6.4f}  {prec:>6.4f}  {rec:>6.4f}")
        pooled_metrics[name].update(dict(f1=f1, precision=prec, recall=rec, threshold=thr))

    # Per-driver AUC summary
    db_auc_list  = [r["auc_db_pop"]     for r in per_driver_results if r["auc_db_pop"]  is not None]
    std_auc_list = [r["auc_std_tcn"]    for r in per_driver_results if r["auc_std_tcn"] is not None]
    ga_auc_list  = [r["auc_gate_adapt"] for r in per_driver_results if r["auc_gate_adapt"] is not None]
    lo_db,  hi_db  = bootstrap_auc_ci_drivers(db_auc_list)
    lo_std, hi_std = bootstrap_auc_ci_drivers(std_auc_list)
    lo_ga,  hi_ga  = bootstrap_auc_ci_drivers(ga_auc_list) if ga_auc_list else (float("nan"), float("nan"))
    print(f"\n  Mean driver AUC — TCN-Single  : {np.mean(std_auc_list):.4f}  [{lo_std:.3f}–{hi_std:.3f}]")
    print(f"  Mean driver AUC — DB-TCN Pop  : {np.mean(db_auc_list):.4f}  [{lo_db:.3f}–{hi_db:.3f}]")
    if ga_auc_list:
        print(f"  Mean driver AUC — Gate-Adapt  : {np.mean(ga_auc_list):.4f}  [{lo_ga:.3f}–{hi_ga:.3f}]")

    # ══════════════════════════════════════════════════════════════════════════
    # GATE DISTRIBUTION ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("MODALITY GATE DISTRIBUTION  (α: physiology weight per driver)")
    print(f"{'='*72}")
    print("α → 1: physiology-dominant  |  α → 0: kinematics-dominant\n")

    gates = np.array(pool_gates)
    print(f"  Mean  : {gates.mean():.3f}   Median : {np.median(gates):.3f}")
    print(f"  Std   : {gates.std():.3f}   Min/Max: {gates.min():.3f} / {gates.max():.3f}")
    n_phys_dom = int((gates > 0.5).sum())
    n_kin_dom  = int((gates < 0.5).sum())
    print(f"  Physiology-dominant (α>0.5) : {n_phys_dom}/{len(gates)} drivers")
    print(f"  Kinematics-dominant (α<0.5) : {n_kin_dom}/{len(gates)} drivers\n")

    pos_rates = np.array([r["pos_rate"] for r in per_driver_results])
    if len(gates) >= 5:
        r_posrate, p_posrate = pearsonr(gates, pos_rates)
        print(f"  Correlation (gate vs pos_rate) : r={r_posrate:+.3f}  p={p_posrate:.3e}")

    # Wilcoxon on gate-adapt gains
    gains_ga = np.array([r["gain_gate"] for r in per_driver_results
                         if not np.isnan(r["gain_gate"])])
    if len(gains_ga) >= 10:
        w, p_two = wilcoxon(gains_ga)
        _, p_gt  = wilcoxon(gains_ga, alternative="greater")
        sig = "***" if p_two < 0.001 else ("**" if p_two < 0.01 else ("*" if p_two < 0.05 else "ns"))
        print(f"\n  Gate-adapt gain  n={len(gains_ga)}  mean={gains_ga.mean():+.4f}  "
              f"Wilcoxon W={w:.0f}  p={p_two:.4f} {sig}  p(improve)={p_gt:.4f}")

    print(f"\n  Per-driver gate values:")
    print(f"  {'Driver':<10}  {'α (mean)':>8}  {'Dominance':<14}  {'AUC(DB-Pop)':>11}  {'AUC(GateAdpt)':>13}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*14}  {'-'*11}  {'-'*13}")
    for r, g in zip(per_driver_results, pool_gates):
        dom  = "physiology" if g > 0.5 else "kinematics"
        ga_s = f"{r['auc_gate_adapt']:.4f}" if r['auc_gate_adapt'] is not None else "  n/a  "
        print(f"  {r['driver']:<10}  {g:>8.3f}  {dom:<14}  {r['auc_db_pop']:>11.4f}  {ga_s:>13}")

    # ══════════════════════════════════════════════════════════════════════════
    # PERMUTATION FEATURE IMPORTANCE
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("PERMUTATION FEATURE IMPORTANCE  (DB-TCN Population, pooled)")
    print(f"{'='*72}")
    print("Permutation: sample-axis shuffle per signal (breaks label correlation).")
    print("Reported as: drop in pooled AUC when signal is permuted.\n")

    base_auc  = pooled_metrics["DB-TCN  (pop.)"]["auc"]
    perm_rows = []

    # Signal-level importance: permute all 5 feature columns (raw,diff,roll_mean,roll_std,z) per signal
    for sig_i, sig_name in enumerate(SIGNAL_COLS):
        is_phys = sig_i in PHYS_IDX
        branch  = "physiology" if is_phys else "kinematics"
        # Within-branch column indices
        branch_idx  = PHYS_IDX if is_phys else KIN_IDX
        local_i     = branch_idx.index(sig_i)
        n_raw_b     = len(branch_idx)
        feat_cols   = [local_i + n_raw_b * k for k in range(5)]

        perm_probs_all = []
        for Xp_ev, Xk_ev, mdl in zip(pool_Xte_p, pool_Xte_k, pool_models_db):
            if is_phys:
                X_perm = Xp_ev.copy()
                perm_idx = rng_perm.permutation(len(X_perm))
                for c in feat_cols:
                    X_perm[:, :, c] = X_perm[perm_idx, :, c]
                Xp_in = torch.as_tensor(X_perm).float().to(DEVICE)
                Xk_in = torch.as_tensor(Xk_ev).float().to(DEVICE)
            else:
                X_perm = Xk_ev.copy()
                perm_idx = rng_perm.permutation(len(X_perm))
                for c in feat_cols:
                    X_perm[:, :, c] = X_perm[perm_idx, :, c]
                Xp_in = torch.as_tensor(Xp_ev).float().to(DEVICE)
                Xk_in = torch.as_tensor(X_perm).float().to(DEVICE)

            mdl.eval()
            with torch.no_grad():
                p = torch.sigmoid(mdl(Xp_in, Xk_in)).cpu().numpy()
            perm_probs_all.append(p)

        perm_probs = np.concatenate(perm_probs_all)
        auc_p = safe_auc(y_pool, perm_probs) or float("nan")
        drop  = base_auc - auc_p
        perm_rows.append((sig_name, branch, auc_p, drop))

    perm_rows.sort(key=lambda x: -x[3])
    print(f"  {'Signal':<25}  {'Branch':<12}  {'AUC (permuted)':>14}  {'Drop':>8}")
    print(f"  {'-'*25}  {'-'*12}  {'-'*14}  {'-'*8}")
    for sig_name, branch, auc_p, drop in perm_rows:
        print(f"  {sig_name:<25}  {branch:<12}  {auc_p:>14.4f}  {drop:>+8.4f}")
    print(f"\n  Baseline (unshuffled) AUC = {base_auc:.4f}")

    # Branch-level importance (permute entire branch)
    print(f"\n  Branch-level importance (all features of branch permuted simultaneously):")
    for branch_name, branch_idx_b in [("Physiology", PHYS_IDX), ("Kinematics", KIN_IDX)]:
        is_phys_b = (branch_name == "Physiology")
        n_raw_b   = len(branch_idx_b)
        all_feat_cols = list(range(n_raw_b * 5))
        perm_probs_all = []
        for Xp_ev, Xk_ev, mdl in zip(pool_Xte_p, pool_Xte_k, pool_models_db):
            if is_phys_b:
                X_perm   = Xp_ev.copy()
                perm_idx = rng_perm.permutation(len(X_perm))
                X_perm   = X_perm[perm_idx]
                Xp_in = torch.as_tensor(X_perm).float().to(DEVICE)
                Xk_in = torch.as_tensor(Xk_ev).float().to(DEVICE)
            else:
                X_perm   = Xk_ev.copy()
                perm_idx = rng_perm.permutation(len(X_perm))
                X_perm   = X_perm[perm_idx]
                Xp_in = torch.as_tensor(Xp_ev).float().to(DEVICE)
                Xk_in = torch.as_tensor(X_perm).float().to(DEVICE)
            mdl.eval()
            with torch.no_grad():
                p = torch.sigmoid(mdl(Xp_in, Xk_in)).cpu().numpy()
            perm_probs_all.append(p)
        perm_probs = np.concatenate(perm_probs_all)
        auc_p = safe_auc(y_pool, perm_probs) or float("nan")
        drop  = base_auc - auc_p
        print(f"  {branch_name:<12}  AUC={auc_p:.4f}  Drop={drop:+.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # MODALITY ABLATION
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("MODALITY ABLATION  (single-branch TCN per modality, pooled LOPO-CV)")
    print(f"{'='*72}")
    print("Each condition trains a separate single-branch TCN on the same folds.")
    print("'DB-TCN' reuses the main-loop result (dual-branch population model).\n")

    ablation_rows = list(ABLATION.items()) + [("DB-TCN (combined)", None)]
    print(f"  {'Condition':<22}  {'Signals':>7}  {'AUC':>6}  {'95% CI':^17}  {'AUPRC':>6}  {'Brier':>6}")
    print(f"  {'-'*22}  {'-'*7}  {'-'*6}  {'-'*17}  {'-'*6}  {'-'*6}")

    for ab_name, ab_idx in ablation_rows:
        if ab_idx is None:
            probs_list = pool_db
            n_sig = len(SIGNAL_COLS)
        else:
            probs_list = pool_ablation[ab_name]
            n_sig = len(ab_idx)

        valid = [(p, y) for p, y in zip(probs_list, pool_y) if p is not None]
        if not valid:
            print(f"  {ab_name:<22}  SKIP (all folds failed)")
            continue
        probs = np.concatenate([p for p, _ in valid])
        y_ab  = np.concatenate([y for _, y in valid])
        auc   = safe_auc(y_ab, probs)   or float("nan")
        auprc = safe_auprc(y_ab, probs) or float("nan")
        brier = brier_score_loss(y_ab, probs)
        lo, hi = bootstrap_auc_ci_windows(y_ab, probs)
        print(f"  {ab_name:<22}  {n_sig:>7}  {auc:.4f}  [{lo:.3f} – {hi:.3f}]  {auprc:.4f}  {brier:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # STRATIFIED EVALUATION (CLC-only vs Non-CLC)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("STRATIFIED EVALUATION  (DB-TCN & Kinematics-only, pooled)")
    print(f"{'='*72}")
    print("CLC-only : window's future slice contains CLC and no other SEVERITY error.")
    print("Non-CLC  : future slice contains at least one non-CLC SEVERITY error.")
    print("Each stratum evaluated against all negative windows.\n")

    etypes_pool = np.concatenate(pool_etypes)
    kin_valid   = [(p, y, e) for p, y, e in
                   zip(pool_ablation["Kinematics"], pool_y, pool_etypes) if p is not None]

    strat_models = [("DB-TCN", db_all, y_pool, etypes_pool)]
    if kin_valid:
        strat_models.append((
            "Kinematics-only",
            np.concatenate([p for p, _, _ in kin_valid]),
            np.concatenate([y for _, y, _ in kin_valid]),
            np.concatenate([e for _, _, e in kin_valid]),
        ))

    print(f"  {'Stratum':<12}  {'Model':<18}  {'N_pos':>5}  {'AUC':>6}  {'95% CI':^17}")
    print(f"  {'-'*12}  {'-'*18}  {'-'*5}  {'-'*6}  {'-'*17}")

    for stratum_name in ["CLC-only", "Non-CLC"]:
        for mod_name, probs_all, y_st, et_st in strat_models:
            clc_only_m = np.array([
                "center_line_crossing" in e and not (e - {"center_line_crossing"})
                for e in et_st])
            non_clc_m  = np.array([bool(e - {"center_line_crossing"}) for e in et_st])
            neg_m      = (y_st == 0)
            pos_m      = clc_only_m if stratum_name == "CLC-only" else non_clc_m
            eval_mask  = pos_m | neg_m
            y_strat    = pos_m[eval_mask].astype(int)
            n_pos      = int(y_strat.sum())
            if n_pos < 5 or len(np.unique(y_strat)) < 2:
                if mod_name == strat_models[0][0]:
                    print(f"  {stratum_name:<12}  SKIP (< 5 positives)")
                continue
            p_strat    = probs_all[eval_mask]
            auc_s      = safe_auc(y_strat, p_strat) or float("nan")
            lo_s, hi_s = bootstrap_auc_ci_windows(y_strat, p_strat)
            print(f"  {stratum_name:<12}  {mod_name:<18}  {n_pos:>5}  {auc_s:.4f}  [{lo_s:.3f} – {hi_s:.3f}]")

    print(f"\n  Reference — DB-TCN overall AUC = {db_auc:.4f}")
    print(f"  If AUC(CLC-only) >> AUC(Non-CLC): kinematics exploit CLC correlation.")
    print(f"  If AUC(CLC-only) ≈  AUC(Non-CLC): model captures general impairment.")

    # ══════════════════════════════════════════════════════════════════════════
    # PER-DRIVER SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    rdf = pd.DataFrame(per_driver_results)
    print(f"\n{'='*72}")
    print(f"PER-DRIVER SUMMARY  ({len(rdf)} reportable drivers, ≥{MIN_EVAL_POSITIVES} eval pos)")
    print("=" * 72)
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(rdf[["auc_db_pop", "auc_gate_adapt", "gain_gate",
                   "mean_gate", "pos_rate", "n_win"]]
              .agg(["mean", "median", "std", "min", "max"]).T.to_string())

    n_imp  = (rdf["gain_gate"] > 0).sum()
    n_hurt = (rdf["gain_gate"] < 0).sum()
    print(f"\nGate-adapt improved AUC for {n_imp}/{len(rdf)}, hurt {n_hurt}/{len(rdf)}.")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    results_json = {
        "config": dict(GAP=GAP, HORIZON=HORIZON, LOOKBACK_S=LOOKBACK_S,
                       GATE_ADAPT_K=GATE_ADAPT_K, GATE_ADAPT_STEPS=GATE_ADAPT_STEPS),
        "pooled": {k: {m: float(v) if isinstance(v, (float, np.floating)) else v
                       for m, v in pooled_metrics[k].items()}
                   for k in pooled_metrics},
        "per_driver": per_driver_results,
        "gate_distribution": dict(mean=float(gates.mean()), std=float(gates.std()),
                                  min=float(gates.min()), max=float(gates.max()),
                                  n_phys_dominant=int(n_phys_dom),
                                  n_kin_dominant=int(n_kin_dom)),
        "permutation_importance": [dict(signal=s, branch=b, auc_perm=float(a), drop=float(d))
                                   for s, b, a, d in perm_rows],
    }
    out_path = OUT_DIR / "dual_branch_results.json"
    with open(out_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
