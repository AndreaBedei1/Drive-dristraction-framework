"""
tcn_dann.py

Domain-Adversarial Dual-Branch TCN (DANN-DB-TCN) for driving impairment prediction.

Extension of tcn_dual_branch.py with two complementary improvements:

1.  Domain-Adversarial Training (DANN)
    ───────────────────────────────────
    In LOPO evaluation the test subject is completely unseen during training.
    Standard ERM implicitly encodes subject-specific patterns in the feature
    extractor, degrading cross-subject generalisation.

    DANN (Ganin et al., 2016) attaches a subject discriminator to the shared
    fusion vector via a Gradient Reversal Layer (GRL):

        fused_96d ──GRL(λ)──► SubjectDiscriminator ──► P(subject | x)

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

    4 kinematic signals × 3 bands = 12 spectral features per window.
    Band powers are normalised by total window power → amplitude-invariant,
    cross-subject relative spectral energy.

    Spectral features are injected at the head level (post-fusion), keeping the
    TCN branch architecture unchanged and maintaining interpretability of α.

Architecture changes relative to tcn_dual_branch.py:
    • Head input: 96 (fusion) + 12 (spectral) = 108-d
    • GRL + SubjectDiscriminator attached to the 96-d fusion vector during training
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
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu, wilcoxon, pearsonr
import xgboost as xgb
import random

# ── CONFIG ──────────────────────────────────────────────────────────────────────
SEED   = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SIGNAL_COLS  = ["arousal", "hr",
                "steeringWheelAngle", "steeringTorq", "acceleration.y", "speed.x"]
VEHICLE_COLS = ["steeringWheelAngle", "steeringTorq", "acceleration.y", "speed.x"]
PHYS_COLS    = ["arousal", "hr"]
KIN_COLS     = ["steeringWheelAngle", "steeringTorq", "acceleration.y", "speed.x"]
PHYS_IDX     = [SIGNAL_COLS.index(c) for c in PHYS_COLS]   # [0, 1]
KIN_IDX      = [SIGNAL_COLS.index(c) for c in KIN_COLS]    # [2, 3, 4, 5]

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
EPOCHS  = 100
LR      = 1e-3
PATIENCE      = 10  # early-stopping patience for DB-TCN (epochs without val-AUC improvement)
DANN_PATIENCE = 15  # longer patience for DANN: adversarial + GRL annealing makes val-AUC noisier

# ── DANN hyperparameters ─────────────────────────────────────────────────────────
LAMBDA_MAX      = 1.0   # maximum GRL reversal strength
ADV_WEIGHT      = 0.1   # weight of adversarial loss in total loss
LAMBDA_WARMUP_P = 0.05  # floor on p → λ ≈ 0.24 at epoch 0 (avoids cold-start instability)

# ── Spectral features ─────────────────────────────────────────────────────────────
# At fs = 1 Hz (arousal/HR/vehicle logged at ~1 Hz), Nyquist = 0.5 Hz.
# Three bands chosen to isolate known driving-distraction frequency signatures.
SPECTRAL_BANDS = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.501)]   # 0.501 to include 0.5 Hz cleanly
SPECTRAL_DIM   = (len(KIN_COLS) + len(PHYS_COLS)) * len(SPECTRAL_BANDS)   # 6 × 3 = 18

# Gate personalisation
GATE_ADAPT_K     = 15
GATE_ADAPT_STEPS = 20
GATE_ADAPT_LR    = 5e-4

# Evaluation
N_BOOTSTRAP        = 2000
MIN_POSITIVES      = 1
MIN_EVAL_POSITIVES = 3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark     = True

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
    for _, grp in df.groupby(["id", "route"]):
        idx = grp.index
        for col in VEHICLE_COLS:
            mu  = grp[col].mean()
            sig = grp[col].std() + 1e-6
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
    """Apply engineer_features to a subset of channels. → (N, T, len(col_idx)*5)."""
    return np.stack([engineer_features(w[:, col_idx]) for w in X_raw])


def apply_features_all(X_raw):
    return np.stack([engineer_features(w) for w in X_raw])


def window_baseline_feats(X_raw, col_idx=None):
    X = X_raw if col_idx is None else X_raw[:, :, col_idx]
    return np.concatenate([X.mean(1), X.std(1), X.max(1)], axis=1).astype(np.float32)

# ── SPECTRAL FEATURES ────────────────────────────────────────────────────────────

def compute_spectral_features_batch(X_kin_raw):
    """
    Compute normalised spectral band power features for a batch of windows.

    Parameters
    ----------
    X_kin_raw : ndarray, shape (N, T, 4)
        Raw (pre-feature-engineering) kinematic signals.
        Signals: steeringWheelAngle, steeringTorq, acceleration.y, speed.x.

    Returns
    -------
    spectral : ndarray, shape (N, SPECTRAL_DIM=12)
        Relative band powers (signal_power_in_band / total_signal_power).
        Layout: [band0_sig0, band0_sig1, ..., band0_sig3,
                 band1_sig0, ..., band2_sig3]

    Physiological motivation
    ─────────────────────────
    At 1 Hz sampling (T=60), freq resolution = 1/60 ≈ 0.017 Hz.
    The 0.1–0.5 Hz steering band contains micro-correction oscillations
    whose frequency increases under secondary-task cognitive load
    (Verwey & Zaidel, 1999; Patten et al., 2004).
    Normalised band power is amplitude-invariant → cross-subject robust.
    """
    N, T, C = X_kin_raw.shape
    freqs   = np.fft.rfftfreq(T, d=1.0)      # (T//2 + 1,) at fs=1 Hz
    masks   = [(freqs >= lo) & (freqs < hi) for lo, hi in SPECTRAL_BANDS]

    out = np.zeros((N, C * len(SPECTRAL_BANDS)), dtype=np.float32)
    for i, win in enumerate(X_kin_raw):           # win: (T, C)
        fft_pow = np.abs(np.fft.rfft(win, axis=0)) ** 2   # (F, C)
        total   = fft_pow.sum(axis=0) + 1e-10              # (C,)
        for b, mask in enumerate(masks):
            out[i, b * C : (b + 1) * C] = fft_pow[mask].sum(axis=0) / total
    return out   # (N, 12)

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


def build_windows(df, fine_stride=True):
    windows, labels, scores, pids, etypes, routes, t_starts = [], [], [], [], [], [], []
    min_session_len = LOOKBACK_S + GAP + HORIZON
    for (pid, route), grp in df.groupby(["id", "route"]):
        grp = grp.sort_values("Timestamp").reset_index(drop=True)
        n   = len(grp)
        ts  = grp["Timestamp"].values

        if n < min_session_len:
            print(f"  [WARN] build_windows: session (pid={pid}, route={route}) has {n} rows "
                  f"< minimum {min_session_len} — skipping.")
            continue

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
            if random.random() < CUTOUT_PROB:
                t0 = random.randint(0, xp.shape[0] - CUTOUT_LEN)
                xp[t0: t0 + CUTOUT_LEN] = 0.0
                xk[t0: t0 + CUTOUT_LEN] = 0.0
        return xp, xk, xs, self.y[idx], self.sids[idx]


def get_dann_loader(Xp, Xk, Xs, y, subj_ids=None, batch_size=BATCH_SIZE, augment=False):
    y_int   = y.astype(int)
    counts  = np.bincount(y_int, minlength=2)
    weights = 1.0 / (counts + 1e-6)
    sw      = torch.from_numpy(weights[y_int])
    sampler = WeightedRandomSampler(sw, len(sw))
    return DataLoader(
        DANNDataset(Xp, Xk, Xs, y, subj_ids, augment=augment),
        batch_size=batch_size, sampler=sampler,
    )

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

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, d):
        super().__init__()
        # Causal convolution: pad (kernel_size-1)*dilation = 2*d timesteps on the
        # left only, so position t can only attend to positions [t-2d, t-d, t].
        causal_pad = 2 * d
        self.conv = nn.Sequential(
            nn.ConstantPad1d((causal_pad, 0), 0.0),
            nn.Conv1d(in_c, out_c, 3, padding=0, dilation=d),
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

    (a) Spectral injection — 12-d normalised FFT band powers concatenated
        to the 96-d fusion vector before the head (→ 108-d).

    (b) DANN branch — 96-d fusion vector passed through SubjectDiscriminator
        via GRL during training, making features subject-invariant.
        Not used at inference.

    PHYS_D = 32  (physiology branch output channels)
    KIN_D  = 64  (kinematics branch output channels)
    Fusion = 96-d; Head input = 108-d (96 + 12 spectral).
    """
    PHYS_D = 32
    KIN_D  = 64

    def __init__(self, n_phys_feats: int, n_kin_feats: int):
        super().__init__()

        # Physiology branch — RF ≈ 43 ts (d=1,4,16), captures tonic arousal/HR lag
        self.phys_branch = nn.Sequential(
            ResBlock(n_phys_feats, self.PHYS_D,  1),
            ResBlock(self.PHYS_D,  self.PHYS_D,  4),
            ResBlock(self.PHYS_D,  self.PHYS_D, 16),
        )
        self.phys_attn = TemporalAttention(self.PHYS_D)

        # Kinematics branch — RF ≈ 31 ts (d=1,2,4,8), captures reactive deviations
        self.kin_branch = nn.Sequential(
            ResBlock(n_kin_feats,      self.KIN_D // 2, 1),
            ResBlock(self.KIN_D // 2,  self.KIN_D,      2),
            ResBlock(self.KIN_D,       self.KIN_D,      4),
            ResBlock(self.KIN_D,       self.KIN_D,      8),
        )
        self.kin_attn = TemporalAttention(self.KIN_D)

        # Modality gate
        self.gate = ModalityGate(self.PHYS_D, self.KIN_D)

        # Head: fusion(96) + spectral(12) → 108 → 48 → 1
        fusion_dim = self.PHYS_D + self.KIN_D   # 96
        self.head = nn.Sequential(
            nn.Linear(fusion_dim + SPECTRAL_DIM, 48),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(48, 1),
        )

        self._last_gate: torch.Tensor | None = None

    def _encode(self, x_phys, x_kin):
        """Shared encoder returning fusion vector and gate value."""
        phys_seq  = self.phys_branch(x_phys.permute(0, 2, 1))   # (B, PHYS_D, T)
        kin_seq   = self.kin_branch(x_kin.permute(0, 2, 1))     # (B, KIN_D,  T)
        phys_pool = self.phys_attn(phys_seq)                     # (B, PHYS_D)
        kin_pool  = self.kin_attn(kin_seq)                       # (B, KIN_D)

        alpha = self.gate(phys_pool, kin_pool)   # (B, 1)
        self._last_gate = alpha.detach()

        fused = torch.cat([alpha * phys_pool,
                           (1.0 - alpha) * kin_pool], dim=-1)   # (B, 96)
        return fused, alpha

    def forward(self, x_phys, x_kin, x_spec=None):
        """
        Parameters
        ----------
        x_phys : (B, T, n_phys_feats)
        x_kin  : (B, T, n_kin_feats)
        x_spec : (B, SPECTRAL_DIM) or None → zeros

        Returns
        -------
        logits : (B,)
        fusion : (B, 96)   — exposed for SubjectDiscriminator
        """
        fused, _ = self._encode(x_phys, x_kin)
        if x_spec is None:
            x_spec = torch.zeros(fused.shape[0], SPECTRAL_DIM,
                                 dtype=fused.dtype, device=fused.device)
        head_in = torch.cat([fused, x_spec], dim=-1)    # (B, 108)
        return self.head(head_in).squeeze(-1), fused     # (B,), (B, 96)

    def gate_values(self, x_phys, x_kin, device=DEVICE):
        """Per-sample gate α without storing gradients."""
        self.eval()
        with torch.no_grad():
            self.forward(x_phys.to(device), x_kin.to(device))
        return self._last_gate.cpu().squeeze(-1).numpy()


class SingleBranchTCN(nn.Module):
    """Standard single-branch TCN for modality ablation."""
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

# ── UTILITIES ────────────────────────────────────────────────────────────────────

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
    assert n_feat_types == 5, (
        "_signal_feat_cols: n_feat_types must equal the number of concatenated "
        "blocks in engineer_features() [raw, diff1, roll_mean, roll_std, z_score]. "
        f"Got {n_feat_types}."
    )
    assert 0 <= local_signal_idx < n_branch_signals, (
        f"_signal_feat_cols: local_signal_idx {local_signal_idx} out of range "
        f"[0, {n_branch_signals})."
    )
    C = n_branch_signals
    return [local_signal_idx + k * C for k in range(n_feat_types)]

# ── TRAINING ─────────────────────────────────────────────────────────────────────

def train_dann_tcn(model, discriminator,
                   Xtr_p, Xtr_k, Xtr_s, y_tr, subj_ids_tr,
                   Xval_p, Xval_k, Xval_s, y_val,
                   pos_weight=None):
    """
    DANN training for DANNDualBranchTCN.

    Jointly optimises:
        L = L_focal(y_pred, y)  +  ADV_WEIGHT · L_CE(subj_pred, subj_id)

    The GRL in SubjectDiscriminator reverses gradients into the feature
    extractor, driving it toward subject-invariant representations.
    Lambda is annealed from 0 → LAMBDA_MAX using the standard DANN schedule.

    Validation AUC (classification only) is used for early stopping.

    Parameters
    ----------
    pos_weight : float or None
        Positive-class weight for focal_loss (neg_count / pos_count).
        Pass the same value used for LR/XGB baselines for a fair comparison.

    Returns trained model and per-epoch diagnostic dict.
    """
    all_params = list(model.parameters()) + list(discriminator.parameters())
    opt        = torch.optim.Adam(all_params, lr=LR, weight_decay=WEIGHT_DECAY)
    sched      = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loader     = get_dann_loader(Xtr_p, Xtr_k, Xtr_s, y_tr, subj_ids_tr, augment=True)
    best_auc, best_w = float("-inf"), None
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

            logits, fusion = model(xp, xk, xs)
            cls_loss       = focal_loss(logits, yb, pos_weight=pos_weight)
            subj_logits    = discriminator(fusion, lambda_)
            adv_loss       = F.cross_entropy(subj_logits, sid)

            loss = cls_loss + ADV_WEIGHT * adv_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()

            ep_cls += cls_loss.item(); ep_adv += adv_loss.item(); n_b += 1

        sched.step()
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
            # Single-class val fold edge case: still advance patience so early
            # stopping eventually triggers rather than running all EPOCHS silently.
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
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    dummy_ids = np.zeros(len(y_tr), dtype=np.int64)
    loader    = get_dann_loader(Xtr_p, Xtr_k, Xtr_s, y_tr, dummy_ids, augment=True)
    best_auc, best_w = float("-inf"), None
    no_improve = 0

    for _ in range(EPOCHS):
        model.train()
        for xp, xk, xs, yb, _ in loader:
            logits, _ = model(xp.to(DEVICE), xk.to(DEVICE), xs.to(DEVICE))
            loss = focal_loss(logits, yb.to(DEVICE), pos_weight=pos_weight)
            opt.zero_grad(); loss.backward(); opt.step()
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
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    if best_w is not None:
        model.load_state_dict(best_w)
    return model


def train_single_tcn(model, Xtr_sc, y_tr, Xval_sc, y_val):
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    y_int   = y_tr.astype(int)
    counts  = np.bincount(y_int, minlength=2)
    weights = 1.0 / (counts + 1e-6)
    sw      = torch.from_numpy(weights[y_int])
    ds      = TensorDataset(torch.as_tensor(Xtr_sc).float(), torch.as_tensor(y_tr).float())
    loader  = DataLoader(ds, batch_size=BATCH_SIZE, sampler=WeightedRandomSampler(sw, len(sw)))
    best_auc, best_w = float("-inf"), None
    no_improve = 0

    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            # Apply the same augmentation as DB-TCN/DANN-DB-TCN for fair comparison.
            xb = xb + torch.randn_like(xb) * JITTER_STD
            if random.random() < CUTOUT_PROB:
                t0 = random.randint(0, xb.shape[1] - CUTOUT_LEN)
                xb[:, t0: t0 + CUTOUT_LEN, :] = 0.0
            loss = focal_loss(model(xb), yb.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(model(torch.as_tensor(Xval_sc).to(DEVICE))).cpu().numpy()
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
    then applies it to the remaining evaluation windows.

    - Support set  : pop_scores[:GATE_ADAPT_K] / y_te[:GATE_ADAPT_K]
    - Evaluation   : pop_scores[GATE_ADAPT_K:]  (Platt-rescaled)
    - Support scores come from the frozen population model (no leakage)
    - Returns full-length score array: [pop_support | platt_eval]
    """
    K = GATE_ADAPT_K
    if len(y_te) <= K:
        return pop_scores

    s_sup, y_sup = pop_scores[:K], y_te[:K]
    s_eval       = pop_scores[K:]

    if len(np.unique(y_sup)) < 2:
        # Not enough class diversity — fall back to population scores
        return pop_scores

    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(s_sup.reshape(-1, 1), y_sup.astype(int))
    platt_eval = lr.predict_proba(s_eval.reshape(-1, 1))[:, 1]

    return np.concatenate([s_sup, platt_eval])


def head_adapt(model_pop, Xte_p, Xte_k, Xte_s, y_te,
               n_steps: int = 25, lr: float = 1e-3):
    """
    Online head fine-tuning personalisation on the first GATE_ADAPT_K test windows.

    Freezes everything except self.head; overfits the classification head to
    this driver's support set so it learns their specific decision boundary.

    - Support set  : y_te[:GATE_ADAPT_K]  — used for head fine-tuning
    - Evaluation   : y_te[GATE_ADAPT_K:]  — fine-tuned model scores only
    - Support scores come from the frozen population model (no leakage)
    - Returns full-length score array: [pop_support | head_adapted_eval]
    """
    K = GATE_ADAPT_K
    if len(y_te) <= K:
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
        logits, _ = model(Xp, Xk, Xs)
        loss = focal_loss(logits, ys, pos_weight=pos_w)
        loss.backward()
        opt.step()

    # Population scores on support set (no leakage), fine-tuned on eval set
    model_pop.eval()
    model.eval()
    with torch.no_grad():
        sup_scores = torch.sigmoid(
            model_pop(Xp, Xk, Xs)[0]).cpu().numpy()
        Xp_e = torch.as_tensor(Xte_p[K:]).to(DEVICE)
        Xk_e = torch.as_tensor(Xte_k[K:]).to(DEVICE)
        Xs_e = torch.as_tensor(Xte_s[K:]).to(DEVICE)
        eval_scores = torch.sigmoid(model(Xp_e, Xk_e, Xs_e)[0]).cpu().numpy()

    return np.concatenate([sup_scores, eval_scores])


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

    for bn_m in model_adapt.modules():
        if isinstance(bn_m, nn.BatchNorm1d):
            bn_m.eval()
    model_adapt.train()

    for _ in range(GATE_ADAPT_STEPS):
        logits, _ = model_adapt(X_sup_p, X_sup_k, X_sup_s)
        loss = focal_loss(logits, y_sup, pos_weight=pos_w)
        opt_g.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(gate_params, 1.0)
        opt_g.step()

    model_adapt.eval()
    with torch.no_grad():
        pop_sup = torch.sigmoid(
            model_pop(X_sup_p, X_sup_k, X_sup_s)[0]
        ).cpu().numpy()
        ada_eval = torch.sigmoid(
            model_adapt(
                torch.as_tensor(Xte_p[GATE_ADAPT_K:]).to(DEVICE),
                torch.as_tensor(Xte_k[GATE_ADAPT_K:]).to(DEVICE),
                torch.as_tensor(Xte_s[GATE_ADAPT_K:]).to(DEVICE),
            )[0]
        ).cpu().numpy()

    return np.concatenate([pop_sup, ada_eval]), GATE_ADAPT_K


def _adv_accuracy(model, discriminator, Xval_p, Xval_k, Xval_s, subj_ids_val):
    """
    Subject discriminator accuracy on validation set after training.

    Lower accuracy → stronger domain invariance in the feature extractor.
    Chance level ≈ 1 / N_subjects.

    Note: lambda_=0 so GRL is transparent (pure forward inference).
    """
    model.eval(); discriminator.eval()
    with torch.no_grad():
        _, fusion = model(
            torch.as_tensor(Xval_p).to(DEVICE),
            torch.as_tensor(Xval_k).to(DEVICE),
            torch.as_tensor(Xval_s).to(DEVICE),
        )
        subj_logits = discriminator.net(fusion)   # bypass GRL — pure forward inference
        preds       = subj_logits.argmax(dim=-1).cpu().numpy()
    return float((preds == subj_ids_val).mean())

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

def main():
    df = pd.read_csv(Path(__file__).parent / "relab+unibo_dataset.csv")
    df = mark_event_onsets(df)
    df = normalize_signals(df)

    X_raw_tr, y_tr_all, _,         pid_tr_all, _,         _,            _        = build_windows(df, fine_stride=True)
    X_raw_te, y_te_all, scores_te, pid_te_all, etypes_te, routes_te_all, ts_te_all = build_windows(df, fine_stride=False)

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
    print(f"  TCN blocks      : d=1,4,16  →  RF ≈ 43 timesteps")
    print(f"  Output channels : {DANNDualBranchTCN.PHYS_D}")
    print(f"Kinematics branch : {KIN_COLS}  ({n_kin_feat} engineered features)")
    print(f"  TCN blocks      : d=1,2,4,8  →  RF ≈ 31 timesteps")
    print(f"  Output channels : {DANNDualBranchTCN.KIN_D}")
    print(f"Cross-modal attn  : DISABLED — branches fused via gate only")
    print(f"Modality gate     : α = σ(MLP([phys_pool, kin_pool]))  ∈ (0,1)")
    print(f"Fusion            : concat(α·phys_enh, (1−α)·kin_enh) → {fusion_dim}-d")
    print(f"Spectral features : {len(SIGNAL_COLS)} signals × {len(SPECTRAL_BANDS)} bands = {SPECTRAL_DIM}-d")
    print(f"  Bands (Hz)      : [0.00,0.10)  [0.10,0.30)  [0.30,0.50]")
    print(f"  Normalisation   : relative band power (÷ total window power)")
    print(f"Head input        : {fusion_dim} (fusion) + {SPECTRAL_DIM} (spectral) = {head_dim}-d")
    print(f"DANN              : GRL + SubjectDiscriminator ({fusion_dim}-d → 32 → N_subj)")
    print(f"  λ schedule      : 0 → {LAMBDA_MAX}  (annealed, Ganin et al. schedule)")
    print(f"  Adversarial wt  : {ADV_WEIGHT}  (total loss = focal + {ADV_WEIGHT}·CE_subj)")
    print(f"Normalisation     : vehicle z-scored per route; physiology absolute")
    print(f"  Spectral scaler : StandardScaler fit on training fold only")
    print(f"Loss              : focal (γ=2) + {ADV_WEIGHT}×CE(subject)")
    print(f"Epochs            : {EPOCHS}  |  LR : {LR}  |  Scheduler : CosineAnnealingLR")
    print(f"GAP / HORIZON     : {GAP}s / {HORIZON}s  →  predicts errors in [{GAP}, {GAP+HORIZON}]s")
    print(f"Gate adapt (pers) : {GATE_ADAPT_K} support windows  |  {GATE_ADAPT_STEPS} steps  |  gate_net only")
    print(f"Bootstrap CI      : {N_BOOTSTRAP} resamples (window-level pooled; driver-level summary)")
    print(f"Device            : {DEVICE}")
    print(f"{'='*72}\n")

    drivers  = [d for d in np.unique(pid_te_all)
                if y_te_all[pid_te_all == d].sum() >= MIN_POSITIVES]

    hdr = (f"{'Driver':<10} | {'N_win':>5} {'PosR%':>6} | "
           f"{'LR':>7} {'XGB':>7} {'DB-TCN':>7} | "
           f"{'DANN-Pop':>8} {'GateAdpt†':>9} {'Platt†':>8} {'HeadFT†':>8} {'Gain':>6} | "
           f"{'gate(α)':>7}")
    print(hdr)
    print("-" * len(hdr))

    per_driver_results = []
    pool_y, pool_db, pool_dann, pool_ga = [], [], [], []
    pool_lr, pool_xgb = [], []
    pool_Xte_p, pool_Xte_k, pool_Xte_s = [], [], []
    pool_models_dann  = []
    pool_etypes, pool_gates = [], []
    dann_diagnostics  = []

    ABLATION      = {"Physiology": PHYS_IDX, "Kinematics": KIN_IDX}
    pool_ablation = {k: [] for k in ABLATION}

    for d in drivers:
        mask_tr = pid_tr_all != d
        X_tr    = X_raw_tr[mask_tr]; y_tr = y_tr_all[mask_tr]; pid_tr = pid_tr_all[mask_tr]

        mask_te  = pid_te_all == d
        X_te     = X_raw_te[mask_te]; y_te = y_te_all[mask_te]
        ts_te    = ts_te_all[mask_te]; routes_te = routes_te_all[mask_te]
        etypes_d = etypes_te[mask_te]

        order    = np.lexsort((ts_te, routes_te))
        X_te     = X_te[order]; y_te = y_te[order]; etypes_d = etypes_d[order]

        seed_d   = int(hashlib.md5(str(d).encode()).hexdigest(), 16) & 0xFFFFFFFF
        fold_rng = np.random.default_rng(SEED ^ seed_d)
        train_drivers = np.unique(pid_tr)
        has_pos  = np.array([y_tr[pid_tr == p].sum() > 0 for p in train_drivers])
        val_ids  = np.concatenate([
            _val_sample(train_drivers[has_pos],  0.20, fold_rng),
            _val_sample(train_drivers[~has_pos], 0.20, fold_rng),
        ])
        vmask = np.isin(pid_tr, val_ids)
        if len(np.unique(y_tr[vmask])) < 2:
            print(f"{d:<10} | SKIP — val fold single-class")
            continue

        # ── Feature engineering & branch scaling ────────────────────────────────
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

        # ── Spectral features — scaler fit on training fold only ─────────────────
        # Kinematics spectral (existing)
        Xtr_sk_raw  = compute_spectral_features_batch(X_tr[~vmask][:, :, KIN_IDX])
        Xval_sk_raw = compute_spectral_features_batch(X_tr[vmask][:,  :, KIN_IDX])
        Xte_sk_raw  = compute_spectral_features_batch(X_te[:,         :, KIN_IDX])
        # Physiology spectral (HR/arousal slow-frequency structure)
        Xtr_sp_raw  = compute_spectral_features_batch(X_tr[~vmask][:, :, PHYS_IDX])
        Xval_sp_raw = compute_spectral_features_batch(X_tr[vmask][:,  :, PHYS_IDX])
        Xte_sp_raw  = compute_spectral_features_batch(X_te[:,         :, PHYS_IDX])

        # Interleave kin+phys per band: [kin_b0|phys_b0, kin_b1|phys_b1, kin_b2|phys_b2]
        _n_kin  = len(KIN_COLS)
        _n_phys = len(PHYS_COLS)
        _n_spec = _n_kin + _n_phys

        def _concat_spec(sk, sp):
            return np.hstack([
                np.hstack([sk[:, b*_n_kin:(b+1)*_n_kin],
                           sp[:, b*_n_phys:(b+1)*_n_phys]])
                for b in range(len(SPECTRAL_BANDS))
            ])

        Xtr_s_raw  = _concat_spec(Xtr_sk_raw,  Xtr_sp_raw)
        Xval_s_raw = _concat_spec(Xval_sk_raw, Xval_sp_raw)
        Xte_s_raw  = _concat_spec(Xte_sk_raw,  Xte_sp_raw)

        # Per-band normalization: fit a separate StandardScaler for each frequency
        # band so that low-freq power (typically larger) does not dominate.
        _scaler_s_bands = [StandardScaler() for _ in SPECTRAL_BANDS]
        Xtr_s_sc  = np.hstack([
            _scaler_s_bands[b].fit_transform(Xtr_s_raw[:, b*_n_spec:(b+1)*_n_spec])
            for b in range(len(SPECTRAL_BANDS))
        ])
        Xval_s_sc = np.hstack([
            _scaler_s_bands[b].transform(Xval_s_raw[:, b*_n_spec:(b+1)*_n_spec])
            for b in range(len(SPECTRAL_BANDS))
        ])
        Xte_s_sc  = np.hstack([
            _scaler_s_bands[b].transform(Xte_s_raw[:, b*_n_spec:(b+1)*_n_spec])
            for b in range(len(SPECTRAL_BANDS))
        ])

        y_tr_train = y_tr[~vmask]; y_val_d = y_tr[vmask]

        # ── LR & XGB baselines ──────────────────────────────────────────────────
        bl_feats_tr = window_baseline_feats(X_tr[~vmask])
        bl_feats_te = window_baseline_feats(X_te)
        pw_bl = (y_tr_train == 0).sum() / max((y_tr_train == 1).sum(), 1)

        lr_model = LogisticRegression(max_iter=5000, class_weight="balanced",
                                      random_state=SEED)
        lr_model.fit(bl_feats_tr, y_tr_train.astype(int))
        lr_scores = lr_model.predict_proba(bl_feats_te)[:, 1]

        xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pw_bl, eval_metric="logloss",
            random_state=SEED, verbosity=0,
        )
        xgb_model.fit(bl_feats_tr, y_tr_train.astype(int))
        xgb_scores = xgb_model.predict_proba(bl_feats_te)[:, 1]

        # ── DB-TCN (spectral, no DANN) — fair within-script baseline ────────────
        print(f"{d:<10} | training DB-TCN...")
        db_model = DANNDualBranchTCN(n_phys_feat, n_kin_feat).to(DEVICE)
        db_model  = train_db_tcn(db_model,
                                 Xtr_p_sc, Xtr_k_sc, Xtr_s_sc, y_tr_train,
                                 Xval_p_sc, Xval_k_sc, Xval_s_sc, y_val_d,
                                 pos_weight=pw_bl)
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
        subj_ids_tr = np.array([subj2id[p] for p in pid_tr[~vmask]], dtype=np.int64)

        dann_model = DANNDualBranchTCN(n_phys_feat, n_kin_feat).to(DEVICE)
        disc       = SubjectDiscriminator(fusion_dim, n_subjects=len(tr_subjects)).to(DEVICE)
        dann_model, diag = train_dann_tcn(
            dann_model, disc,
            Xtr_p_sc, Xtr_k_sc, Xtr_s_sc, y_tr_train, subj_ids_tr,
            Xval_p_sc, Xval_k_sc, Xval_s_sc, y_val_d,
            pos_weight=pw_bl,
        )
        dann_model.eval()
        with torch.no_grad():
            dann_scores = torch.sigmoid(
                dann_model(torch.as_tensor(Xte_p_sc).to(DEVICE),
                           torch.as_tensor(Xte_k_sc).to(DEVICE),
                           torch.as_tensor(Xte_s_sc).to(DEVICE))[0]
            ).cpu().numpy()

        # Adversarial accuracy on a subsample of training data (diagnostic).
        # Val drivers are held out at driver-level so they are never seen by the
        # discriminator; evaluate on training windows instead.
        rng_adv  = np.random.default_rng(seed_d + 1)
        n_adv    = min(512, len(y_tr_train))
        adv_idx  = rng_adv.choice(len(y_tr_train), n_adv, replace=False)
        adv_acc  = _adv_accuracy(dann_model, disc,
                                 Xtr_p_sc[adv_idx], Xtr_k_sc[adv_idx],
                                 Xtr_s_sc[adv_idx], subj_ids_tr[adv_idx])
        chance   = 1.0 / len(tr_subjects)
        dann_diagnostics.append({
            "driver": d, "adv_acc": adv_acc, "chance": chance,
            "n_tr_subjects": int(len(tr_subjects)),
            "final_class_loss": diag["class_loss"][-1],
            "final_adv_loss":   diag["adv_loss"][-1],
            "final_lambda":     diag["lambda"][-1],
        })
        print(f"{d:<10} | adv_acc={adv_acc:.3f}  chance={chance:.3f}  ratio={adv_acc/chance:.1f}x")

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
        auc_lr    = safe_auc(y_te, lr_scores)    or float("nan")
        auc_xgb   = safe_auc(y_te, xgb_scores)  or float("nan")
        auc_db    = safe_auc(y_te, db_scores)    or float("nan")
        auc_dann  = safe_auc(y_te, dann_scores)  or float("nan")
        auc_ga    = (safe_auc(y_te, gate_scores) or float("nan")) if gate_scores is not None else float("nan")
        auc_platt = safe_auc(y_te, platt_scores) or float("nan")
        auc_head  = (safe_auc(y_te, head_scores) or float("nan")) if head_scores is not None else float("nan")
        gain      = auc_dann - auc_db

        print(f"{d:<10} | {len(y_te):>5}  {100*y_te.mean():>5.1f}% | "
              f"{auc_lr:>7.4f} {auc_xgb:>7.4f} {auc_db:>7.4f} | "
              f"{auc_dann:>8.4f} {auc_ga:>8.4f} {auc_platt:>8.4f} {auc_head:>8.4f} {gain:>+6.4f} | "
              f"{mean_gate:>7.3f}")

        pool_y.append(y_te);    pool_db.append(db_scores)
        pool_dann.append(dann_scores)
        pool_ga.append(gate_scores if gate_scores is not None else dann_scores)
        pool_lr.append(lr_scores); pool_xgb.append(xgb_scores)
        pool_Xte_p.append(Xte_p_sc); pool_Xte_k.append(Xte_k_sc)
        pool_Xte_s.append(Xte_s_sc)
        pool_models_dann.append(copy.deepcopy(dann_model))
        pool_etypes.append(etypes_d); pool_gates.append(mean_gate)

        per_driver_results.append({
            "driver": d, "n_windows": int(len(y_te)),
            "pos_rate": float(y_te.mean()),
            "auc_lr": auc_lr, "auc_xgb": auc_xgb, "auc_db": auc_db,
            "auc_dann": auc_dann, "auc_gate": auc_ga, "auc_platt": auc_platt,
            "auc_head": auc_head,
            "dann_gain": gain, "mean_gate": mean_gate, "adv_acc": adv_acc,
        })

        # ── Ablation single-branch TCNs ──────────────────────────────────────────
        for abl_name, abl_idx in ABLATION.items():
            n_abl_feat = len(abl_idx) * 5
            Xtr_abl    = apply_features_branch(X_tr[~vmask], abl_idx)
            Xval_abl   = apply_features_branch(X_tr[vmask],  abl_idx)
            Xte_abl    = apply_features_branch(X_te,         abl_idx)
            sc_abl     = StandardScaler()
            Xtr_abl_sc  = sc_abl.fit_transform(
                Xtr_abl.reshape(-1, n_abl_feat)).reshape(-1, LOOKBACK_S, n_abl_feat)
            Xval_abl_sc = sc_abl.transform(
                Xval_abl.reshape(-1, n_abl_feat)).reshape(-1, LOOKBACK_S, n_abl_feat)
            Xte_abl_sc  = sc_abl.transform(
                Xte_abl.reshape(-1, n_abl_feat)).reshape(-1, LOOKBACK_S, n_abl_feat)
            abl_m = SingleBranchTCN(n_abl_feat).to(DEVICE)
            abl_m = train_single_tcn(abl_m, Xtr_abl_sc, y_tr_train, Xval_abl_sc, y_val_d)
            abl_m.eval()
            with torch.no_grad():
                abl_sc = torch.sigmoid(
                    abl_m(torch.as_tensor(Xte_abl_sc).to(DEVICE))).cpu().numpy()
            pool_ablation[abl_name].append(
                {"y": y_te, "scores": abl_sc, "auc": safe_auc(y_te, abl_sc)})

    # ════════════════════════════════════════════════════════════════════════════
    # POOLED EVALUATION
    # ════════════════════════════════════════════════════════════════════════════
    all_y    = np.concatenate(pool_y)
    all_db   = np.concatenate(pool_db)
    all_dann = np.concatenate(pool_dann)
    all_ga   = np.concatenate(pool_ga)
    all_lr   = np.concatenate(pool_lr)
    all_xgb  = np.concatenate(pool_xgb)

    drv_aucs_lr   = [safe_auc(y, s) for y, s in zip(pool_y, pool_lr)   if safe_auc(y, s) is not None]
    drv_aucs_xgb  = [safe_auc(y, s) for y, s in zip(pool_y, pool_xgb)  if safe_auc(y, s) is not None]
    drv_aucs_db   = [safe_auc(y, s) for y, s in zip(pool_y, pool_db)   if safe_auc(y, s) is not None]
    drv_aucs_dann = [safe_auc(y, s) for y, s in zip(pool_y, pool_dann) if safe_auc(y, s) is not None]
    drv_aucs_ga   = [safe_auc(y, s) for y, s in zip(pool_y, pool_ga)   if safe_auc(y, s) is not None]

    ci_dann_win = bootstrap_auc_ci_windows(all_y, all_dann)
    ci_dann_drv = bootstrap_auc_ci_drivers(drv_aucs_dann)

    def _print_pooled(name, all_s, drv_aucs):
        auc_w  = safe_auc(all_y, all_s) or float("nan")
        ci_w   = bootstrap_auc_ci_windows(all_y, all_s)
        ci_d   = bootstrap_auc_ci_drivers(drv_aucs)
        auprc  = safe_auprc(all_y, all_s) or float("nan")
        brier  = brier_score_loss(all_y, all_s)
        ece    = compute_ece(all_y, all_s)
        mean_d = np.mean(drv_aucs) if drv_aucs else float("nan")
        std_d  = np.std(drv_aucs)  if drv_aucs else float("nan")
        print(f"\n  {name}:")
        print(f"    Pooled  AUROC : {auc_w:.4f}  [{ci_w[0]:.4f}, {ci_w[1]:.4f}]  (window bootstrap CI)")
        print(f"    Driver  AUROC : {mean_d:.4f} ± {std_d:.4f}  [{ci_d[0]:.4f}, {ci_d[1]:.4f}]")
        print(f"    AUPRC         : {auprc:.4f}")
        print(f"    Brier Score   : {brier:.4f}")
        print(f"    ECE           : {ece:.4f}")

    print(f"\n{'='*72}")
    print("POOLED EVALUATION — POPULATION MODELS (zero-shot generalisation)")
    print(f"{'='*72}")
    print(f"  These models never see test-participant labels. Results are comparable.")
    _print_pooled("LR baseline",      all_lr,   drv_aucs_lr)
    _print_pooled("XGB baseline",     all_xgb,  drv_aucs_xgb)
    _print_pooled("DB-TCN (no DANN)", all_db,   drv_aucs_db)
    _print_pooled("DANN-DB-TCN",      all_dann, drv_aucs_dann)

    # Wilcoxon signed-rank: DANN-DB-TCN > DB-TCN (paired per driver)
    paired = [(r["auc_db"], r["auc_dann"]) for r in per_driver_results
              if not (np.isnan(r["auc_db"]) or np.isnan(r["auc_dann"]))]
    if len(paired) >= 5:
        db_p, da_p = zip(*paired)
        try:
            wstat, wpval = wilcoxon(list(da_p), list(db_p), alternative="greater")
            sig = "***" if wpval < 0.001 else ("**" if wpval < 0.01 else ("*" if wpval < 0.05 else "ns"))
            mean_gain = np.mean(np.array(da_p) - np.array(db_p))
            print(f"\n  Wilcoxon (DANN > DB-TCN) : W={wstat:.1f}  p={wpval:.4f}  {sig}")
            print(f"  Mean per-driver gain     : {mean_gain:+.4f}")
        except Exception:
            pass
    print(f"{'='*72}")

    print(f"\n{'='*72}")
    print("POOLED EVALUATION — ONLINE PERSONALISATION (uses test-participant labels)")
    print(f"{'='*72}")
    print(f"  WARNING: these methods observe y_te[:GATE_ADAPT_K={GATE_ADAPT_K}] from the test")
    print(f"  participant. They are NOT held-out generalisation estimates and MUST NOT")
    print(f"  be compared directly to the population models above.")
    _print_pooled("DANN + GateAdapt (online)", all_ga, drv_aucs_ga)
    print(f"{'='*72}")

    # ── THRESHOLD METRICS ────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("THRESHOLD-DEPENDENT METRICS  (Youden's J threshold)")
    print(f"{'='*72}")
    for name, scores in [("DB-TCN (no DANN)", all_db), ("DANN-DB-TCN", all_dann)]:
        if len(np.unique(all_y)) < 2: continue
        f1, prec, rec, thresh = threshold_metrics(all_y, scores)
        print(f"  {name:<20}  F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  @thresh={thresh:.3f}")

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

    # Per-window gate values vs predicted risk
    gate_per_driver = [
        m.gate_values(torch.as_tensor(Xp), torch.as_tensor(Xk))
        for m, Xp, Xk in zip(pool_models_dann, pool_Xte_p, pool_Xte_k)
    ]
    gate_vals_all = np.concatenate(gate_per_driver)
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

    N_PERM_DRIVERS = min(10, len(pool_models_dann))
    rng_imp = np.random.default_rng(SEED + 2)

    sig_importance  = {col: [] for col in SIGNAL_COLS}
    spec_importance = []
    band_importance = {f"[{lo:.2f},{hi:.2f})Hz": [] for lo, hi in SPECTRAL_BANDS}
    n_kin_sigs      = len(KIN_COLS) + len(PHYS_COLS)  # total signals per spectral band

    for i in range(N_PERM_DRIVERS):
        m   = pool_models_dann[i]; m.eval()
        Xp  = torch.as_tensor(pool_Xte_p[i]).to(DEVICE)
        Xk  = torch.as_tensor(pool_Xte_k[i]).to(DEVICE)
        Xs  = torch.as_tensor(pool_Xte_s[i]).to(DEVICE)
        yt  = pool_y[i]
        with torch.no_grad():
            base_s = torch.sigmoid(m(Xp, Xk, Xs)[0]).cpu().numpy()
        base_auc = safe_auc(yt, base_s)
        if base_auc is None: continue

        # Per-signal permutation
        for ci, col in enumerate(SIGNAL_COLS):
            is_phys = ci in PHYS_IDX
            branch_tensor = Xp if is_phys else Xk
            n_branch_sigs = len(PHYS_COLS) if is_phys else len(KIN_COLS)
            local_i       = PHYS_IDX.index(ci) if is_phys else KIN_IDX.index(ci)
            feat_cols     = _signal_feat_cols(local_i, n_branch_sigs)  # 5 columns

            bt_perm  = branch_tensor.clone()
            idx_perm = torch.from_numpy(rng_imp.permutation(len(yt))).to(DEVICE)
            for fc in feat_cols:
                bt_perm[:, :, fc] = bt_perm[idx_perm, :, fc]

            with torch.no_grad():
                perm_s = torch.sigmoid(
                    m(bt_perm if is_phys else Xp,
                      Xk if is_phys else bt_perm,
                      Xs)[0]).cpu().numpy()
            perm_auc = safe_auc(yt, perm_s)
            if perm_auc is not None:
                sig_importance[col].append(base_auc - perm_auc)

        # All spectral features
        perm_idx = torch.from_numpy(rng_imp.permutation(len(yt))).to(DEVICE)
        Xs_perm  = Xs[perm_idx]
        with torch.no_grad():
            perm_sp = torch.sigmoid(m(Xp, Xk, Xs_perm)[0]).cpu().numpy()
        perm_sp_auc = safe_auc(yt, perm_sp)
        if perm_sp_auc is not None:
            spec_importance.append(base_auc - perm_sp_auc)

        # Per-band spectral importance
        for b_i, lbl in enumerate(band_importance.keys()):
            Xs_b = Xs.clone()
            cols_b = list(range(b_i * n_kin_sigs, (b_i + 1) * n_kin_sigs))
            perm_b = torch.from_numpy(rng_imp.permutation(len(yt))).to(DEVICE)
            Xs_b[:, cols_b] = Xs_b[perm_b][:, cols_b]
            with torch.no_grad():
                perm_b_s = torch.sigmoid(m(Xp, Xk, Xs_b)[0]).cpu().numpy()
            perm_b_auc = safe_auc(yt, perm_b_s)
            if perm_b_auc is not None:
                band_importance[lbl].append(base_auc - perm_b_auc)

    print(f"\n  Signal importance (mean ΔAUC when permuted, {N_PERM_DRIVERS} drivers):")
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
    print(f"  Discriminator accuracy on validation set after training.")
    print(f"  Lower accuracy → stronger subject invariance in feature extractor.")
    print(f"  Chance level ≈ 1/N_subjects.  Ratio = AdvAcc/Chance.\n")
    print(f"  {'Driver':<10}  {'N_subj':>6}  {'Chance':>7}  {'AdvAcc':>7}  {'Ratio':>7}  {'Invariant?'}")
    print(f"  {'-'*58}")

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
              f"{ratio:>7.2f}×  {flag}")

    print(f"\n  Domain invariance (ratio < 2×) : "
          f"{n_effective}/{len(dann_diagnostics)} drivers "
          f"({100*n_effective/max(len(dann_diagnostics),1):.1f}%)")
    print(f"  Mean ratio across drivers      : {np.mean(all_ratios):.2f}× chance")

    # ── MODALITY ABLATION ────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("MODALITY ABLATION — Single-Branch TCN vs Dual-Branch variants")
    print(f"{'='*72}")
    for abl_name, results in pool_ablation.items():
        valid = [r for r in results if r["auc"] is not None]
        if not valid: continue
        aucs   = [r["auc"] for r in valid]
        all_s  = np.concatenate([r["scores"] for r in valid])
        all_yt = np.concatenate([pool_y[j] for j in range(len(valid))])
        auc_p  = safe_auc(all_yt, all_s) or float("nan")
        print(f"  {abl_name + ' only':<25}  Pooled={auc_p:.4f}  "
              f"Driver={np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    auc_db_p   = safe_auc(all_y, all_db)   or float("nan")
    auc_dann_p = safe_auc(all_y, all_dann) or float("nan")
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
        auc_d  = safe_auc(all_y[emask], all_db[emask])
        auc_da = safe_auc(all_y[emask], all_dann[emask])
        print(f"  {label:<20}  n_pos={pos_mask.sum():>4}  "
              f"DB-TCN={auc_d:.3f}  DANN={auc_da:.3f}  "
              f"Gain={((auc_da or 0) - (auc_d or 0)):+.3f}")

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
            "lr":     {"auc": float(safe_auc(all_y, all_lr)  or float("nan")),
                       "driver_mean": float(np.mean(drv_aucs_lr)),
                       "driver_std":  float(np.std(drv_aucs_lr))},
            "xgb":    {"auc": float(safe_auc(all_y, all_xgb) or float("nan")),
                       "driver_mean": float(np.mean(drv_aucs_xgb)),
                       "driver_std":  float(np.std(drv_aucs_xgb))},
            "db_tcn": {"auc": float(safe_auc(all_y, all_db)  or float("nan")),
                       "driver_mean": float(np.mean(drv_aucs_db)),
                       "driver_std":  float(np.std(drv_aucs_db))},
            "dann_db": {
                "auc":         float(safe_auc(all_y, all_dann) or float("nan")),
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
