"""
tcn_kin_ext11.py — Pure KinTCN driving risk score.

Changes from ext10:
  1. Physiology removed entirely.  Five consecutive experiments show phys hurts
     or is neutral in LOPO (perm-importance: hr Δ≈0, arousal Δ<0).
     Architecture simplified: no FiLM, no phys scalars, no phys branch.
  2. Score threshold raised to ≥ 3.  panic_braking-only windows (score=1, 15%
     of positives) have no reliable 5-15s kinematic precursor — they are reactive
     responses.  Targeting score ≥ 3 (Red_light_violation, Collision) focuses on
     events associated with sustained inattention.
  3. Label smoothing removed.  Shrinking positive targets to 0.95 degraded
     calibration at low prevalence.  Plain 0/1 targets used.
  4. Score-weighted loss.  Positive windows weighted by risk_score / mean_pos_score
     so Collisions (score=5) contribute proportionally more than Red-light
     violations (score=3).  Negatives keep weight=1.
  5. Val-set Platt calibration (KinTCN+Cal).  After training, a logistic
     recalibrator is fit on validation-fold scores/labels and applied to test
     scores.  Fixes Brier and ECE without affecting AUROC/AUPRC.
  6. Output cleaned: only AUROC, AUPRC, Brier, ECE reported.
     F1/Precision/Recall removed (threshold-dependent, irrelevant for risk score).
     Pooled AUROC, personalisation, HeadFT, Wilcoxon all removed.

Primary framing: continuous driving risk score — output is a relative risk index
validated against safety-critical error occurrence under strict LOPO-CV.
AUROC measures discrimination; Brier/ECE measure calibration after Platt scaling.

Validity guarantees (inherited):
  - Strict LOPO-CV: each driver's windows held out entirely.
  - Anti-leakage LOO route renormalisation.
  - SMOTE in raw signal space, fit on training fold only.
  - StandardScaler fit on training windows only.
  - PYTHONHASHSEED enforced for reproducibility.
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
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu, wilcoxon
from scipy.spatial.distance import cdist
import xgboost as xgb
import random

# ── CONFIG ────────────────────────────────────────────────────────────────────
SEED   = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SIGNAL_COLS  = ["speed.x", "steeringWheelAngle", "steeringTorq", "acceleration.y"]
KIN_COLS     = SIGNAL_COLS
VEHICLE_COLS = SIGNAL_COLS
KIN_IDX      = list(range(len(KIN_COLS)))   # [0, 1, 2, 3]

SEVERITY = {
    "Collision":               5,
    "Red_light_violation":     3,
    "panic_braking_with_stop": 2,
    "panic_braking":           1,
}
# sharp_turn excluded (kinematic self-correlation, see ext10).
# center_line_crossing excluded (see ext4).
SCORE_THRESHOLD = 1   # y=1 iff composite_risk_score >= SCORE_THRESHOLD

# Windowing
LOOKBACK_S  = 45
WINDOW_STEP = 5
GAP, HORIZON = 5, 10
ROLL_K      = 10

# Training
BATCH_SIZE   = 64
WEIGHT_DECAY = 1e-4
JITTER_STD   = 0.01
CUTOUT_LEN   = 5
CUTOUT_PROB  = 0.2
EPOCHS       = 100
LR           = 5e-4
PATIENCE     = 30

RENORM_CLIP          = 3.0
N_BOOTSTRAP          = 2000
MIN_EVAL_POSITIVES   = 5
N_PERM_REPEATS       = 10
EXCLUDE_EVAL_DRIVERS = {"0D04", "0D03R", "0D05"}


USE_SMOTE         = True
SMOTE_K_NEIGHBORS = 5
SMOTE_SEED_SALT   = 0xABCD
TARGET_RATIO = 0.9

# ── DETERMINISM ───────────────────────────────────────────────────────────────
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

OUT_DIR = Path(__file__).parent / "impairment_results"
OUT_DIR.mkdir(exist_ok=True)

# ── PREPROCESSING ─────────────────────────────────────────────────────────────

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
    """Per-session z-score for kinematic signals (first LOOKBACK_S+GAP rows as reference)."""
    df = df.copy()
    prefix = LOOKBACK_S + GAP
    for _, grp in df.groupby(["id", "route"]):
        idx = grp.index
        for col in VEHICLE_COLS:
            if col not in df.columns:
                continue
            mu  = grp.iloc[:prefix][col].mean()
            sig = grp.iloc[:prefix][col].std() + 1e-6
            df.loc[idx, col] = (grp[col] - mu) / sig
    return df

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────

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
    return np.stack([engineer_features(w[:, col_idx]) for w in X_raw])


def window_baseline_feats(X_raw):
    """Mean | std | max per signal — for LR and XGB baselines."""
    return np.concatenate([X_raw.mean(1), X_raw.std(1), X_raw.max(1)], axis=1).astype(np.float32)

# ── SMOTE ─────────────────────────────────────────────────────────────────────

def smote_raw(X_raw, y, scores=None, k=SMOTE_K_NEIGHBORS, rng=None,
              pids=None, routes=None, t_starts=None):
    """
    SMOTE in raw signal space with PCA nearest-neighbour search.
    Returns augmented X_raw, y, scores, and interpolation indices/lambdas.
    Synthetic scores are linearly interpolated from anchor/neighbour scores.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    pos_idx = np.where(y == 1)[0]
    n_pos   = len(pos_idx)
    n_neg   = (y == 0).sum()

    if n_pos == 0 or n_pos >= n_neg:
        return (X_raw, y, scores,
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32))

    
    n_synthetic  = int((TARGET_RATIO * n_neg - (1 - TARGET_RATIO) * n_pos)
                       / (1 - TARGET_RATIO))
    n_synthetic  = max(0, n_synthetic)
    X_flat       = X_raw[pos_idx].reshape(n_pos, -1)

    k_eff = min(k, n_pos - 1)
    if k_eff < 1:
        dup = rng.choice(pos_idx, n_synthetic, replace=True)
        syn_scores = scores[dup] if scores is not None else None
        return (np.concatenate([X_raw, X_raw[dup]], axis=0),
                np.concatenate([y,     np.ones(n_synthetic, dtype=y.dtype)]),
                (np.concatenate([scores, syn_scores]) if scores is not None else None),
                dup, dup, np.zeros(n_synthetic, dtype=np.float32))

    _pca_k = min(n_pos - 1, X_flat.shape[1], 50, max(2, n_pos // 5))
    X_nn   = PCA(n_components=_pca_k).fit_transform(X_flat) if _pca_k >= 2 else X_flat

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
    syn_scores_arr   = np.zeros(n_synthetic, dtype=np.float32)

    for i in range(n_synthetic):
        anchor = rng.integers(0, n_pos)
        nn     = nn_idx[anchor, rng.integers(0, k_eff)]
        lam    = float(rng.uniform(0.0, 1.0))
        syn[i] = (X_raw[pos_idx[anchor]]
                  + lam * (X_raw[pos_idx[nn]] - X_raw[pos_idx[anchor]]))
        syn_anchor_idx[i]   = pos_idx[anchor]
        syn_neighbor_idx[i] = pos_idx[nn]
        syn_lambdas[i]      = lam
        if scores is not None:
            syn_scores_arr[i] = ((1 - lam) * scores[pos_idx[anchor]]
                                 + lam * scores[pos_idx[nn]])

    aug_scores = (np.concatenate([scores, syn_scores_arr])
                  if scores is not None else None)
    return (np.concatenate([X_raw, syn], axis=0),
            np.concatenate([y,     np.ones(n_synthetic, dtype=y.dtype)]),
            aug_scores,
            syn_anchor_idx, syn_neighbor_idx, syn_lambdas)

# ── WINDOWING ─────────────────────────────────────────────────────────────────

def composite_risk_score(future_df):
    return sum(SEVERITY[col] * int((future_df[col] > 0).any())
               for col in SEVERITY if col in future_df.columns)


def build_windows(df):
    """Uniform-stride windows; label y=1 iff risk score >= SCORE_THRESHOLD."""
    windows, labels, scores, pids, routes, t_starts = [], [], [], [], [], []
    min_len  = LOOKBACK_S + GAP + HORIZON
    nan_skip: dict = {}

    for (pid, route), grp in df.groupby(["id", "route"]):
        grp = grp.sort_values("Timestamp").reset_index(drop=True)
        n   = len(grp)
        ts  = grp["Timestamp"].values
        if n < min_len:
            continue
        idx = 0
        while idx + LOOKBACK_S + GAP + HORIZON <= n:
            sig       = grp.iloc[idx: idx + LOOKBACK_S][SIGNAL_COLS].values.astype(np.float32)
            prev_s, prev_t = nan_skip.get(pid, (0, 0))
            if not np.isnan(sig).any():
                future = grp.iloc[idx + LOOKBACK_S + GAP: idx + LOOKBACK_S + GAP + HORIZON]
                score  = composite_risk_score(future)
                windows.append(sig)
                labels.append(int(score >= SCORE_THRESHOLD))
                scores.append(float(score))
                pids.append(pid); routes.append(route); t_starts.append(ts[idx])
                nan_skip[pid] = (prev_s, prev_t + 1)
            else:
                nan_skip[pid] = (prev_s + 1, prev_t + 1)
            idx += WINDOW_STEP

    total_s = sum(s for s, _ in nan_skip.values())
    total_t = sum(t for _, t in nan_skip.values())
    if total_s > 0:
        print(f"  [NaN-skip audit] {total_s}/{total_t} windows dropped "
              f"({100*total_s/max(total_t,1):.1f}%)")
        for pid_k, (ns, nt) in sorted(nan_skip.items()):
            if ns > 0:
                print(f"    pid={pid_k}: {ns}/{nt} skipped ({100*ns/max(nt,1):.1f}%)")

    return (np.array(windows,  dtype=np.float32),
            np.array(labels,   dtype=np.float32),
            np.array(scores,   dtype=np.float32),
            np.array(pids),
            np.array(routes),
            np.array(t_starts))

# ── DATASET ───────────────────────────────────────────────────────────────────

class KinDataset(Dataset):
    """Yields (x_kin, y, sample_weight) tuples.
    Positive windows are weighted by risk_score / mean_pos_score; negatives = 1.
    """
    def __init__(self, X_kin, y, scores=None, augment=False):
        self.Xk  = torch.as_tensor(X_kin).float()
        self.y   = torch.as_tensor(y).float()
        self.aug = augment
        if scores is not None and (y == 1).any():
            mean_pos = float(scores[y == 1].mean())
            w = np.where(y == 1, scores / max(mean_pos, 1e-6), 1.0)
        else:
            w = np.ones(len(y), dtype=np.float32)
        self.w = torch.as_tensor(w).float()

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        xk = self.Xk[idx].clone()
        if self.aug:
            xk += torch.randn_like(xk) * JITTER_STD
            scale = torch.empty(xk.shape[-1]).uniform_(0.8, 1.2)
            xk    = xk * scale.unsqueeze(0)
            if torch.rand(1).item() < CUTOUT_PROB and xk.shape[0] > CUTOUT_LEN:
                t0 = torch.randint(0, xk.shape[0] - CUTOUT_LEN, (1,)).item()
                xk[t0: t0 + CUTOUT_LEN] = 0.0
        return xk, self.y[idx], self.w[idx]

# ── MODEL ─────────────────────────────────────────────────────────────────────

def _gn_groups(out_c: int) -> int:
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
    Pure kinematics TCN risk scorer.
    Input: (B, T, n_kin_feats).  Output: (B,) logit.
    ResBlocks d=1,2,4,8,16 → RF ≈ 63 timesteps → TemporalAttention → head.
    No physiology, no FiLM, no spectral features.
    """
    KIN_D = 64

    def __init__(self, n_kin_feats: int):
        super().__init__()
        self.kin_branch = nn.Sequential(
            ResBlock(n_kin_feats,     self.KIN_D // 2, 1),
            ResBlock(self.KIN_D // 2, self.KIN_D,      2),
            ResBlock(self.KIN_D,      self.KIN_D,      4),
            ResBlock(self.KIN_D,      self.KIN_D,      8),
            ResBlock(self.KIN_D,      self.KIN_D,     16),
        )
        self.attn = TemporalAttention(self.KIN_D)
        self.head = nn.Sequential(
            nn.Linear(self.KIN_D, 48),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(48, 1),
        )

    def forward(self, x_kin):
        feat = self.kin_branch(x_kin.permute(0, 2, 1))   # (B, KIN_D, T)
        pool = self.attn(feat)                             # (B, KIN_D)
        return self.head(pool).squeeze(-1)                 # (B,)

# ── UTILITIES ─────────────────────────────────────────────────────────────────

def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2: return None
    return roc_auc_score(y_true, y_score)

def safe_auprc(y_true, y_score):
    if len(np.unique(y_true)) < 2: return None
    return average_precision_score(y_true, y_score)

def _nan_or(x): return float("nan") if x is None else x

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0; n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask   = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo)
        if not mask.any(): continue
        ece += mask.sum() / n * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)

def bootstrap_auc_ci_drivers(driver_aucs, n_boot=N_BOOTSTRAP, seed=SEED):
    aucs = np.array(driver_aucs); n = len(aucs)
    if n < 2: return float("nan"), float("nan")
    rng  = np.random.default_rng(seed)
    boot = [rng.choice(aucs, n, replace=True).mean() for _ in range(n_boot)]
    return tuple(np.percentile(boot, [2.5, 97.5]))

def _val_sample(arr, frac, rng):
    n = min(max(1, int(frac * len(arr))), len(arr)) if len(arr) > 0 else 0
    return rng.choice(arr, n, replace=False) if n > 0 else np.array([], dtype=arr.dtype)

# ── VAL-SET PLATT CALIBRATION ─────────────────────────────────────────────────

def val_platt_calibrate(val_scores, y_val, test_scores):
    """
    Fit a logistic recalibrator on validation-fold scores/labels.
    Applied to test scores to fix Brier and ECE.
    Falls back to raw scores if val is single-class or calibrator inverts ordering.
    """
    if len(np.unique(y_val)) < 2:
        return test_scores
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    try:
        lr.fit(val_scores.reshape(-1, 1), y_val.astype(int))
    except Exception:
        return test_scores
    w = lr.coef_[0][0]
    if w <= 0 or w > 20.0:   # non-monotone or exploded
        return test_scores
    return lr.predict_proba(test_scores.reshape(-1, 1))[:, 1]

# ── TRAINING ──────────────────────────────────────────────────────────────────

def train_kin_tcn(model, Xtr_k, y_tr, scores_tr, Xval_k, y_val):
    """Score-weighted BCE, no label smoothing, early stopping on val AUROC."""
    loader = DataLoader(
        KinDataset(Xtr_k, y_tr, scores_tr, augment=True),
        batch_size=BATCH_SIZE, shuffle=True)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.01)
 #   sched = torch.optim.lr_scheduler.OneCycleLR(
 #       opt, max_lr=LR * 5, steps_per_epoch=len(loader), epochs=EPOCHS, pct_start=0.3)
    best_auc, best_w = float("-inf"), None
    best_loss        = float("inf")
    no_improve       = 0
    Xval_t = torch.as_tensor(Xval_k).to(DEVICE)

    for epoch in range(EPOCHS):
        model.train()
        ep_loss = 0.0; n_b = 0
        for xk, yb, wb in loader:
            xk, yb, wb = xk.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
            logits    = model(xk)
            loss_raw  = F.binary_cross_entropy_with_logits(logits, yb, reduction='none')
            loss      = (loss_raw * wb).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            #sched.step() #one cycle
            ep_loss += loss.item(); n_b += 1
        sched.step() #cosine

        model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(model(Xval_t)).cpu().numpy()
        auc = safe_auc(y_val, preds)
        if auc is not None:
            if auc > best_auc:
                best_auc = auc; best_w = copy.deepcopy(model.state_dict()); no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE: break
        else:
            train_loss = ep_loss / max(n_b, 1)
            if train_loss < best_loss - 1e-4:
                best_loss = train_loss; best_w = copy.deepcopy(model.state_dict()); no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE: break

    if best_w is not None:
        model.load_state_dict(best_w)
    return model

# ── VALIDITY REPORT ───────────────────────────────────────────────────────────

def print_validity_report(X_raw, y, scores, pids):
    pos   = y == 1; total_pos = pos.sum(); total = len(y)
    unique, cnts = np.unique(scores[pos].astype(int), return_counts=True)
    print(f"\n{'='*72}")
    print("LABEL VALIDITY REPORT — COMPOSITE RISK TARGET (score ≥ 3)")
    print(f"{'='*72}")
    print(f"Total windows    : {total}")
    print(f"Positive (risk≥{SCORE_THRESHOLD}): {total_pos} ({100*total_pos/total:.1f}%)")
    print(f"Negative         : {(~pos).sum()} ({100*(~pos).sum()/total:.1f}%)")
    print(f"\nRisk score distribution (positives only):")
    for sc, cnt in zip(unique, cnts):
        print(f"  score={sc:2d}  n={cnt:5d}  ({100*cnt/total_pos:.1f}%)")
    print(f"\nPredictive validity — Mann-Whitney U (pos vs neg, raw signals):")
    print(f"  {'Signal':<24}  {'Mean(neg)':>10}  {'Mean(pos)':>10}  {'p-value':>12}  Sig")
    for ci, col in enumerate(SIGNAL_COLS):
        neg_v = X_raw[~pos][:, :, ci].mean(axis=1)
        pos_v = X_raw[pos][:, :, ci].mean(axis=1)
        _, pv = mannwhitneyu(neg_v, pos_v, alternative="two-sided")
        sig   = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
        print(f"  {col:<24}  {neg_v.mean():>10.4f}  {pos_v.mean():>10.4f}  {pv:>12.2e}  {sig}")
    print(f"\nPer-driver positive rate:")
    print(f"  {'Driver':<12}  {'N':>6}  {'Pos':>5}  {'Rate%':>6}  Tier")
    for d in np.unique(pids):
        mask_d = pids == d; nd = mask_d.sum(); np_ = y[mask_d].sum()
        rate   = 100.0 * np_ / nd if nd > 0 else 0.0
        tier   = "HIGH" if rate >= 10 else ("MED" if rate >= 5 else "LOW")
        print(f"  {d:<12}  {nd:>6}  {np_:>5}  {rate:>6.1f}%  {tier}")
    print(f"{'='*72}")

# ── SAMPLING RATE CHECK ───────────────────────────────────────────────────────

def _check_sampling_rate(df):
    if "Timestamp" not in df.columns:
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
        lines = "\n".join(f"  pid={p}  route={r}  Δt={dt:.3f}s" for p, r, dt in bad)
        raise RuntimeError(f"Sampling rate check failed:\n{lines}")
    global_dt = float(np.median(all_dts)) if all_dts else 1.0
    print(f"  Sampling rate check: {len(all_dts)} sessions, "
          f"median Δt = {global_dt:.3f} s  (all within [0.8, 1.2] s — OK)")
    return global_dt

# ── LOO RENORMALISATION ───────────────────────────────────────────────────────

def _loo_renorm(X_arr, pid_arr, routes_arr,
                all_session_stats, route_sum_mus, route_sum_sigs2, route_counts,
                ci_norm_map, label="windows"):
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
                    n_singleton += 1; continue
                rte_mu  = (route_sum_mus[key_rte]  - own_mu)  / (n_rte - 1)
                rte_sig = math.sqrt(
                    max((route_sum_sigs2[key_rte] - own_sig ** 2) / (n_rte - 1), 0.0))
                X_out[mask, :, ci] = (
                    X_out[mask, :, ci] * (own_sig + 1e-6) + own_mu - rte_mu
                ) / max(rte_sig, 1e-6)
    if n_singleton:
        print(f"  [WARN] _loo_renorm ({label}): {n_singleton} singleton-route triplets skipped.")
    return X_out

# ── MAIN ──────────────────────────────────────────────────────────────────────

def _print_model(name, all_s, all_y, drv_aucs):
    mean_d = np.mean(drv_aucs) if drv_aucs else float("nan")
    std_d  = np.std(drv_aucs)  if drv_aucs else float("nan")
    ci_d   = bootstrap_auc_ci_drivers(drv_aucs)
    auprc  = _nan_or(safe_auprc(all_y, all_s))
    brier  = brier_score_loss(all_y, all_s)
    ece    = compute_ece(all_y, all_s)
    print(f"\n  {name}:")
    print(f"    AUROC  : {mean_d:.4f} ± {std_d:.4f}  [{ci_d[0]:.4f}, {ci_d[1]:.4f}]")
    print(f"    AUPRC  : {auprc:.4f}  (random = {all_y.mean():.4f})")
    print(f"    Brier  : {brier:.4f}  (naive = {float(all_y.mean()*(1-all_y.mean())):.4f})")
    print(f"    ECE    : {ece:.4f}")
    return {"auroc_mean": mean_d, "auroc_std": std_d,
            "auroc_ci": list(ci_d), "auprc": auprc, "brier": brier, "ece": ece}


def main():
    df = pd.read_csv(Path(__file__).parent / "relab+unibo_dataset.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True).astype("int64") / 1e9
    _check_sampling_rate(df)
    df = mark_event_onsets(df)

    COLS_TO_NORM = VEHICLE_COLS
    NORM_PREFIX  = LOOKBACK_S + GAP
    all_session_stats: dict = {}
    for (pid, route), grp in df.groupby(["id", "route"]):
        for col in COLS_TO_NORM:
            if col in grp.columns:
                mu      = float(grp.iloc[:NORM_PREFIX][col].mean())
                sig_raw = float(grp.iloc[:NORM_PREFIX][col].std())
                all_session_stats[(pid, route, col)] = (mu, sig_raw)

    df = normalize_signals(df)
    X_raw_all, y_all, scores_all, pid_all, routes_all, ts_all = build_windows(df)

    keep = ~np.isin(pid_all, list(EXCLUDE_EVAL_DRIVERS))
    X_raw_all  = X_raw_all[keep];  y_all      = y_all[keep]
    scores_all = scores_all[keep]; pid_all    = pid_all[keep]
    routes_all = routes_all[keep]; ts_all     = ts_all[keep]

    n_kin_feat = len(KIN_COLS) * 5   # 20

    print_validity_report(X_raw_all, y_all, scores_all, pid_all)

    print(f"\n{'='*72}")
    print("KIN-TCN — PIPELINE CONFIGURATION")
    print(f"{'='*72}")
    print(f"Kinematics  : {KIN_COLS}  ({n_kin_feat} engineered features)")
    print(f"TCN blocks  : d=1,2,4,8,16  →  RF ≈ 63 timesteps  →  {KinTCN.KIN_D}-d pool")
    print(f"Head        : Linear(64,48) → ReLU → Dropout(0.2) → Linear(48,1)")
    print(f"Physiology  : REMOVED (perm-importance ≤ 0 across ext4–ext10)")
    print(f"Score thr   : ≥ {SCORE_THRESHOLD}  (Red_light=3, Collision=5+)")
    print(f"Loss        : Score-weighted BCE  (positive weight = score / mean_pos_score)")
    print(f"Calibration : Val-set Platt scaling  (KinTCN+Cal)")
    print(f"SMOTE       : k={SMOTE_K_NEIGHBORS} ratio {TARGET_RATIO}:{1-TARGET_RATIO} pos:neg")
    print(f"Renorm clip : ±{RENORM_CLIP}σ")
    print(f"Epochs      : {EPOCHS}  |  LR : {LR}  |  Patience : {PATIENCE}")
    print(f"GAP/HORIZON : {GAP}s / {HORIZON}s")
    print(f"Device      : {DEVICE}")
    print(f"{'='*72}\n")

    ci_norm_map = {col: SIGNAL_COLS.index(col) for col in COLS_TO_NORM if col in SIGNAL_COLS}
    drivers = [d for d in np.unique(pid_all)
               if y_all[pid_all == d].sum() >= MIN_EVAL_POSITIVES]

    hdr = (f"{'Driver':<10} | {'N_win':>5} {'PosR%':>6} | "
           f"{'LR':>7} {'XGB':>7} {'KinTCN':>7} {'KinTCN+Cal':>11}")
    print(hdr); print("-" * len(hdr))

    per_driver_results = []
    pool_y, pool_tcn, pool_cal = [], [], []
    pool_lr, pool_xgb = [], []
    pool_models, pool_Xte_k = [], []

    for d in drivers:
        mask_tr = pid_all != d
        X_tr    = X_raw_all[mask_tr]; y_tr = y_all[mask_tr]
        sc_tr   = scores_all[mask_tr]
        pid_tr  = pid_all[mask_tr];   routes_tr = routes_all[mask_tr]
        ts_tr   = ts_all[mask_tr]

        mask_te = pid_all == d
        X_te    = X_raw_all[mask_te]; y_te = y_all[mask_te]
        ts_te   = ts_all[mask_te];    routes_te = routes_all[mask_te]

        order = np.argsort(ts_te)
        X_te = X_te[order]; y_te = y_te[order]
        ts_te = ts_te[order]; routes_te = routes_te[order]

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

        # Build route reference pools
        _route_mus_d:     dict = {}
        _route_sigs_d:    dict = {}
        _route_sum_mus:   dict = {}
        _route_sum_sigs2: dict = {}
        _route_counts:    dict = {}

        for (pid_s, route_s, col_s), (mu_s, sig_s) in all_session_stats.items():
            if pid_s == d or math.isnan(mu_s) or math.isnan(sig_s):
                continue
            key = (route_s, col_s)
            _route_mus_d.setdefault(key, []).append(mu_s)
            _route_sigs_d.setdefault(key, []).append(sig_s)
            if pid_s not in val_ids_set:
                _route_sum_mus[key]   = _route_sum_mus.get(key,   0.0) + mu_s
                _route_sum_sigs2[key] = _route_sum_sigs2.get(key, 0.0) + sig_s ** 2
                _route_counts[key]    = _route_counts.get(key,    0)   + 1

        _route_pure_train_stats: dict = {
            k: (_route_sum_mus[k] / _route_counts[k],
                math.sqrt(max(_route_sum_sigs2[k] / _route_counts[k], 0.0)))
            for k in _route_sum_mus if _route_counts[k] >= 1
        }
        _route_tr_stats_d: dict = {
            k: (float(np.mean(v)),
                math.sqrt(max(float(np.mean(np.array(_route_sigs_d[k]) ** 2)), 0.0)))
            for k, v in _route_mus_d.items()
        }

        # Renorm test windows
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

        X_tr = _loo_renorm(X_tr, pid_tr, routes_tr, all_session_stats,
                           _route_sum_mus, _route_sum_sigs2, _route_counts,
                           ci_norm_map, f"train (test={d})")
        X_tr = np.clip(X_tr, -RENORM_CLIP, RENORM_CLIP)

        # Val windows
        df_val_fold = df[df["id"].isin(val_ids_set)].copy()
        X_val_raw, y_val_d, _, _pids_val, _routes_val, _ = build_windows(df_val_fold)
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

        # ── LR / XGB baselines ────────────────────────────────────────────────
        X_bl   = X_tr[~vmask]; y_bl = y_tr[~vmask]
        pw_bl  = float((y_bl == 0).sum() / max((y_bl == 1).sum(), 1))

        Xbl_feat     = window_baseline_feats(X_bl)
        Xval_bl_feat = window_baseline_feats(X_val_raw)
        Xte_bl_feat  = window_baseline_feats(X_te)

        scaler_bl        = StandardScaler().fit(Xbl_feat)
        Xbl_feat_sc      = scaler_bl.transform(Xbl_feat)
        Xval_bl_feat_sc  = scaler_bl.transform(Xval_bl_feat)
        Xte_bl_feat_sc   = scaler_bl.transform(Xte_bl_feat)

        lr_model = LogisticRegression(max_iter=1000, random_state=SEED,
                                      class_weight="balanced")
        lr_model.fit(Xbl_feat_sc, y_bl.astype(int))
        lr_scores = lr_model.predict_proba(Xte_bl_feat_sc)[:, 1]

        xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pw_bl, eval_metric="logloss",
            random_state=SEED, verbosity=0)
        xgb_model.fit(Xbl_feat_sc, y_bl.astype(int),
                      eval_set=[(Xval_bl_feat_sc, y_val_d.astype(int))],
                      verbose=False)
        xgb_scores = xgb_model.predict_proba(Xte_bl_feat_sc)[:, 1]

        # ── KinTCN ────────────────────────────────────────────────────────────
        X_tr_train   = X_tr[~vmask]
        y_tr_train   = y_tr[~vmask]
        sc_tr_train  = sc_tr[~vmask]
        pid_tr_train = pid_tr[~vmask]
        routes_tr_tr = routes_tr[~vmask]
        ts_tr_train  = ts_tr[~vmask]

        if USE_SMOTE:
            smote_rng = np.random.default_rng(SEED ^ seed_d ^ SMOTE_SEED_SALT)
            X_tr_aug, y_tr_aug, sc_tr_aug, _anch, _neigh, _lams = smote_raw(
                X_tr_train, y_tr_train, scores=sc_tr_train, rng=smote_rng,
                pids=pid_tr_train, routes=routes_tr_tr, t_starts=ts_tr_train)
        else:
            X_tr_aug, y_tr_aug, sc_tr_aug = X_tr_train, y_tr_train, sc_tr_train
            _anch = np.empty(0, dtype=np.int64)
            _neigh = np.empty(0, dtype=np.int64)
            _lams = np.empty(0, dtype=np.float32)

        n_real      = len(y_tr_train)
        n_synthetic = len(_anch)

        # Feature engineering
        Xtr_k_real  = apply_features_branch(X_tr_aug[:n_real], KIN_IDX)
        Xval_k_feat = apply_features_branch(X_val_raw,         KIN_IDX)
        Xte_k_feat  = apply_features_branch(X_te,              KIN_IDX)

        scaler_k = StandardScaler()
        scaler_k.fit(Xtr_k_real.reshape(-1, n_kin_feat))
        Xtr_k_real_sc = scaler_k.transform(
            Xtr_k_real.reshape(-1, n_kin_feat)).reshape(-1, LOOKBACK_S, n_kin_feat)
        Xval_k_sc = scaler_k.transform(
            Xval_k_feat.reshape(-1, n_kin_feat)).reshape(-1, LOOKBACK_S, n_kin_feat)
        Xte_k_sc  = scaler_k.transform(
            Xte_k_feat.reshape(-1, n_kin_feat)).reshape(-1, LOOKBACK_S, n_kin_feat)

        # Synthetic kin features
        if n_synthetic > 0:
            lams_bc      = _lams[:, None, None]
            Xtr_k_syn_sc = ((1.0 - lams_bc) * Xtr_k_real_sc[_anch]
                            + lams_bc         * Xtr_k_real_sc[_neigh])
            Xtr_k_sc_all = np.concatenate([Xtr_k_real_sc,
                                            Xtr_k_syn_sc.astype(np.float32)])
        else:
            Xtr_k_sc_all = Xtr_k_real_sc

        torch.manual_seed(SEED ^ seed_d)
        torch.cuda.manual_seed_all(SEED ^ seed_d)
        model = KinTCN(n_kin_feat).to(DEVICE)
        model = train_kin_tcn(model, Xtr_k_sc_all, y_tr_aug, sc_tr_aug,
                              Xval_k_sc, y_val_d)
        model.eval()
        with torch.no_grad():
            tcn_scores = torch.sigmoid(
                model(torch.as_tensor(Xte_k_sc).to(DEVICE))).cpu().numpy()
            val_scores = torch.sigmoid(
                model(torch.as_tensor(Xval_k_sc).to(DEVICE))).cpu().numpy()

        # Val-set Platt calibration
        cal_scores = val_platt_calibrate(val_scores, y_val_d, tcn_scores)

        auc_lr  = _nan_or(safe_auc(y_te, lr_scores))
        auc_xgb = _nan_or(safe_auc(y_te, xgb_scores))
        auc_tcn = _nan_or(safe_auc(y_te, tcn_scores))
        auc_cal = _nan_or(safe_auc(y_te, cal_scores))

        print(f"{d:<10} | {len(y_te):>5} {100*y_te.mean():>6.1f}% | "
              f"{auc_lr:>7.4f} {auc_xgb:>7.4f} {auc_tcn:>7.4f} {auc_cal:>11.4f}")

        pool_y.append(y_te); pool_tcn.append(tcn_scores); pool_cal.append(cal_scores)
        pool_lr.append(lr_scores); pool_xgb.append(xgb_scores)
        pool_models.append(model); pool_Xte_k.append(Xte_k_sc)

        per_driver_results.append({
            "driver": d, "n_windows": int(len(y_te)), "pos_rate": float(y_te.mean()),
            "auc_lr": auc_lr, "auc_xgb": auc_xgb,
            "auc_tcn": auc_tcn, "auc_cal": auc_cal,
        })

    if not pool_y:
        print("[ERROR] No folds evaluated."); return

    all_y    = np.concatenate(pool_y)
    all_tcn  = np.concatenate(pool_tcn)
    all_cal  = np.concatenate(pool_cal)
    all_lr   = np.concatenate(pool_lr)
    all_xgb  = np.concatenate(pool_xgb)

    drv_aucs_lr  = [v for v in (safe_auc(y, s) for y, s in zip(pool_y, pool_lr))  if v is not None]
    drv_aucs_xgb = [v for v in (safe_auc(y, s) for y, s in zip(pool_y, pool_xgb)) if v is not None]
    drv_aucs_tcn = [v for v in (safe_auc(y, s) for y, s in zip(pool_y, pool_tcn)) if v is not None]
    drv_aucs_cal = [v for v in (safe_auc(y, s) for y, s in zip(pool_y, pool_cal)) if v is not None]

    print(f"\n{'='*72}")
    print("RESULTS — POPULATION RISK SCORE")
    print(f"{'='*72}")
    metrics = {}
    metrics["lr"]      = _print_model("LR baseline",          all_lr,  all_y, drv_aucs_lr)
    metrics["xgb"]     = _print_model("XGB baseline",         all_xgb, all_y, drv_aucs_xgb)
    metrics["kin_tcn"] = _print_model("KinTCN (raw)",         all_tcn, all_y, drv_aucs_tcn)
    metrics["kin_cal"] = _print_model("KinTCN+Cal (Platt)",   all_cal, all_y, drv_aucs_cal)

    # Permutation importance
    print(f"\n{'='*72}")
    print("PERMUTATION FEATURE IMPORTANCE — KinTCN+Cal")
    print(f"{'='*72}")
    n_kin = len(KIN_COLS)
    signal_deltas = {col: [] for col in KIN_COLS}

    for fi, (model_i, Xk_i, y_i) in enumerate(zip(pool_models, pool_Xte_k, pool_y)):
        base_raw = safe_auc(y_i, torch.sigmoid(model_i(
            torch.as_tensor(Xk_i).to(DEVICE))).detach().cpu().numpy())
        if base_raw is None: continue
        rng_pi = np.random.default_rng(SEED ^ fi)
        for rep in range(N_PERM_REPEATS):
            for ki, col in enumerate(KIN_COLS):
                feat_cols = [ki + j * n_kin for j in range(5)]
                Xk_perm   = Xk_i.copy()
                perm_idx  = rng_pi.permutation(len(Xk_perm))
                Xk_perm[:, :, feat_cols] = Xk_perm[perm_idx, :, :][:, :, feat_cols]
                perm_auc = safe_auc(y_i, torch.sigmoid(model_i(
                    torch.as_tensor(Xk_perm).to(DEVICE))).detach().cpu().numpy())
                if perm_auc is not None:
                    signal_deltas[col].append(base_raw - perm_auc)

    print(f"\n  Signal importance (mean ΔAUC when permuted):")
    for col in sorted(signal_deltas, key=lambda c: np.mean(signal_deltas[c]) if signal_deltas[c] else 0, reverse=True):
        deltas = signal_deltas[col]
        if deltas:
            print(f"    {col:<30} Δ={np.mean(deltas):+.4f} ± {np.std(deltas):.4f}")

    # Wilcoxon: KinTCN vs XGB
    paired_tcn_xgb = [
        (r["auc_xgb"], r["auc_cal"]) for r in per_driver_results
        if not (math.isnan(r["auc_xgb"]) or math.isnan(r["auc_cal"]))
    ]
    if len(paired_tcn_xgb) >= 10:
        a_xgb, a_tcn = zip(*paired_tcn_xgb)
        diff_tx = np.array(a_tcn) - np.array(a_xgb)
        n_tcn_wins = int((diff_tx > 0).sum())
        if np.any(diff_tx != 0):
            stat_tx, p_tx = wilcoxon(list(a_tcn), list(a_xgb), alternative="two-sided")
            sig_tx = "***" if p_tx < 0.001 else ("**" if p_tx < 0.01 else ("*" if p_tx < 0.05 else "ns"))
            print(f"\n{'='*72}")
            print("WILCOXON SIGNED-RANK TEST — KinTCN+Cal vs XGB")
            print(f"{'='*72}")
            print(f"  W={stat_tx:.1f}  p={p_tx:.4f}  {sig_tx}")
            print(f"  Mean per-driver gain (KinTCN+Cal − XGB): {diff_tx.mean():+.4f} ± {diff_tx.std():.4f}")
            print(f"  KinTCN+Cal > XGB : {n_tcn_wins}/{len(diff_tx)} drivers  ({100*n_tcn_wins/len(diff_tx):.1f}%)")

    # Per-driver summary
    print(f"\n{'='*72}")
    print("PER-DRIVER SUMMARY")
    print(f"{'='*72}")
    print(f"  N drivers evaluated : {len(drv_aucs_cal)}")
    print(f"  KinTCN+Cal  AUROC  : {np.mean(drv_aucs_cal):.4f} ± {np.std(drv_aucs_cal):.4f}")
    print(f"  KinTCN (raw) AUROC : {np.mean(drv_aucs_tcn):.4f} ± {np.std(drv_aucs_tcn):.4f}")
    gains = [r["auc_cal"] - r["auc_tcn"] for r in per_driver_results
             if not (math.isnan(r["auc_cal"]) or math.isnan(r["auc_tcn"]))]
    if gains:
        print(f"  Cal gain vs raw    : {np.mean(gains):+.4f} ± {np.std(gains):.4f}")

    # Save
    results = {
        "config": {
            "LOOKBACK_S": LOOKBACK_S, "GAP": GAP, "HORIZON": HORIZON,
            "WINDOW_STEP": WINDOW_STEP, "SCORE_THRESHOLD": SCORE_THRESHOLD,
            "RENORM_CLIP": RENORM_CLIP, "SMOTE": USE_SMOTE,
            "EPOCHS": EPOCHS, "LR": LR,
        },
        "pooled": metrics,
        "per_driver": per_driver_results,
    }
    out_path = OUT_DIR / "kin_tcn_ext11_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out_path}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
