"""
tcn.py  ·  Temporal Convolutional Network — driver impairment detection
===========================================================================
Q1-journal LOPO-CV pipeline.

Architecture
  · 4-block dilated TCN + temporal attention (dilations 1/2/4/8; RF = 30 s)
  · Focal loss (γ=2), CosineAnnealingLR, WeightedRandomSampler
  · Feature engineering per window: raw | Δ1 | rolling-mean | rolling-std | z-score
    → C × 5 channels into the TCN
  · Temporal augmentation during training: Gaussian jitter + random cutout

Labels
  · Composite risk score: severity-weighted sum over 6 error types
    (Collision=5, Red_light_violation=3, panic_braking_with_stop=2,
     center_line_crossing=2, panic_braking=1, sharp_turn=1)
  · Onset detection: contiguous non-zero runs collapse to a single onset row,
    preventing positive-rate inflation from multi-second events

Pre-processing
  · Vehicle dynamics z-scored per (driver, route)
  · Physiological signals kept on absolute scale (predictive signal is in level)

Temporal split — per test participant
  pers → BUFFER → cal → BUFFER → eval
  · Personalisation : L2-tethered fine-tune on last TCN block + head
  · Calibration     : Platt scaling on held-out cal slice (no leakage)
  · Evaluation      : calibrated probabilities → AUC, AUPRC, Brier, BSS

Baselines
  · Logistic Regression + XGBoost (window statistics: mean, std, max per signal)

Pooled evaluation
  · AUC-ROC + 95 % CI (bootstrap, N=2000), AUPRC, Brier score, BSS
  · Permutation feature importance (sample-axis shuffle, TCN-Population, pooled)

Label validity report
  · Positive rate, risk-score distribution, Mann-Whitney U per signal
"""

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu
import xgboost as xgb
import random

# ─── CONFIG ───────────────────────────────────────────────────────────────────

SEED   = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SIGNAL_COLS = [
    "arousal", "hr",              # physiological — absolute level is predictive
    "brake", "accelerator",       # vehicle dynamics — within-session deviation predictive
    "steeringWheelAngle",
    "speed.x",
]

# Only these are z-scored per (driver, route).
# Physiological signals are left on absolute scale.
VEHICLE_COLS = ["brake", "accelerator", "steeringWheelAngle"]

SEVERITY = {
    "Collision":               5,
    "Red_light_violation":     3,
    "panic_braking_with_stop": 2,
    "center_line_crossing":    2,   # contiguous non-zero rows = one onset event
    "panic_braking":           1,
    "sharp_turn":              1,
}

EPOCHS, LR    = 100, 1e-3
LOOKBACK_S    = 60
WINDOW_STEP   = 5
GAP, HORIZON  = 3, 5
BATCH_SIZE    = 64
WEIGHT_DECAY  = 1e-4

MIN_POSITIVES          = 1     # minimum positives in test set to enter the loop
MIN_EVAL_POSITIVES     = 3     # minimum positives in eval slice to be "reportable"
PERS_MIN_DRIVER_RATE   = 0.05  # skip personalisation if driver overall rate < 5 %

HYBRID_LR            = 5e-4
HYBRID_EPOCHS        = 30
HYBRID_MIN_POSITIVES = 5       # pers slice must contain at least this many positives
CAL_MIN_POSITIVES    = 2       # cal slice must contain at least this many positives
LAMBDA_TETHER        = 1e-1

ROLL_K         = 10
JITTER_STD     = 0.01
CUTOUT_LEN     = 5
CUTOUT_PROB    = 0.2
EVENT_VICINITY = 10
N_BOOTSTRAP    = 2000

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False


# ─── PRE-PROCESSING ───────────────────────────────────────────────────────────

def mark_event_onsets(df):
    """
    Replace each contiguous non-zero run in every SEVERITY column with a
    single 1 at the first row of the run; set all subsequent rows to 0.
    Applied per (driver, route) to prevent cross-session boundary artefacts.

    Motivation: a 21-second center_line_crossing generates ~4 positive windows
    at 5 s stride; onset detection ensures at most 1 window is labeled positive
    per physical event.
    """
    df = df.copy()
    for (_, _), grp in df.groupby(["id", "route"]):
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
    Per-(driver, route) z-score for VEHICLE_COLS only.
    Physiological signals are left on absolute scale: their predictive signal
    is in absolute arousal / heart-rate level, not within-session deviation.
    """
    df = df.copy()
    for (_, _), grp in df.groupby(["id", "route"]):
        idx = grp.index
        for col in VEHICLE_COLS:
            mu  = grp[col].mean()
            sig = grp[col].std() + 1e-6
            df.loc[idx, col] = (grp[col] - mu) / sig
    return df


# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────

def engineer_features(window):
    """
    (T, C) → (T, C×5): raw | first-difference | rolling-mean | rolling-std | z-score.
    Gives the TCN access to both signal level and local dynamics simultaneously.
    The five groups are concatenated along the channel axis; the per-signal
    permutation importance section assumes this ordering.
    """
    T, C      = window.shape
    diff1     = np.diff(window, axis=0, prepend=window[:1])
    roll_mean = np.zeros_like(window)
    roll_std  = np.zeros_like(window)
    for t in range(T):
        seg          = window[max(0, t - ROLL_K + 1): t + 1]
        roll_mean[t] = seg.mean(axis=0)
        roll_std[t]  = seg.std(axis=0) + 1e-6
    w_mean  = window.mean(axis=0, keepdims=True)
    w_std   = window.std(axis=0,  keepdims=True) + 1e-6
    z_score = (window - w_mean) / w_std
    return np.concatenate([window, diff1, roll_mean, roll_std, z_score],
                          axis=1).astype(np.float32)


def apply_features(X_raw):
    """(N, T, C) → (N, T, C×5)."""
    return np.stack([engineer_features(w) for w in X_raw])


def window_baseline_feats(X_raw):
    """
    (N, T, C) → (N, C×3): per-signal [mean, std, max] over the lookback window.
    Tabular feature vector for LR and XGBoost baselines.
    """
    return np.concatenate([
        X_raw.mean(axis=1),
        X_raw.std(axis=1),
        X_raw.max(axis=1),
    ], axis=1).astype(np.float32)


# ─── DATASET & UTILS ──────────────────────────────────────────────────────────

class DrivingDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X       = torch.as_tensor(X).float()
        self.y       = torch.as_tensor(y).float()
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]
        if self.augment:
            x += torch.randn_like(x) * JITTER_STD
            if random.random() < CUTOUT_PROB:
                t_start = random.randint(0, x.shape[0] - CUTOUT_LEN)
                x[t_start: t_start + CUTOUT_LEN, :] = 0.0
        return x, y


def get_balanced_loader(X, y, batch_size=BATCH_SIZE, augment=False):
    """Balanced DataLoader via WeightedRandomSampler."""
    y_int          = y.astype(int)
    counts         = np.bincount(y_int, minlength=2)
    weights        = 1.0 / (counts + 1e-6)
    sample_weights = torch.from_numpy(weights[y_int])
    sampler        = WeightedRandomSampler(sample_weights, len(sample_weights))
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


def bootstrap_auc_ci(y_true, y_score, n_boot=N_BOOTSTRAP, seed=SEED):
    """Non-parametric bootstrap 95 % CI for AUC-ROC (percentile method)."""
    rng_b = np.random.default_rng(seed)
    n     = len(y_true)
    aucs  = []
    for _ in range(n_boot):
        idx = rng_b.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    if len(aucs) < 10:
        return float("nan"), float("nan")
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return lo, hi


def fit_calibrator(logits_tensor, y_array, device, lr=0.01, steps=500):
    """Platt-scale calibrator fitted on pre-computed (detached) logits."""
    cal      = LogitCalibrator().to(device)
    opt      = torch.optim.Adam(cal.parameters(), lr=lr)
    y_tensor = torch.as_tensor(y_array).float().to(device)
    for _ in range(steps):
        loss = F.binary_cross_entropy_with_logits(cal(logits_tensor), y_tensor)
        opt.zero_grad(); loss.backward(); opt.step()
    return cal


def min_count_boundary(y, min_positives, start=0):
    """
    Return the first index i (exclusive) such that y[start:i] contains
    at least `min_positives` positive labels.  Returns None if exhausted.
    """
    count = 0
    for i in range(start, len(y)):
        if y[i] == 1:
            count += 1
        if count >= min_positives:
            return i + 1
    return None


# ─── MODEL ────────────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Linear(channels, channels // 2)
        self.score = nn.Linear(channels // 2, 1)

    def forward(self, x):
        x_t     = x.permute(0, 2, 1)           # (B, T, C)
        h       = torch.tanh(self.query(x_t))
        weights = torch.softmax(self.score(h), dim=1)
        return torch.sum(x_t * weights, dim=1)  # (B, C)


class TCN_Attention_Net(nn.Module):
    """
    4-block dilated TCN + temporal attention.

    Dilations 1/2/4/8 give a receptive field of 2*(1+2+4+8) = 30 s over
    the 60-second lookback window; the attention mechanism then re-weights
    which timesteps contribute most.  Deeper than the 3-block variant in
    tcn_impairment_detect.py — justified by the longer lookback.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.network = nn.Sequential(
            self._block(in_channels,  64, 1),
            self._block(64,          128, 2),
            self._block(128,         128, 4),
            self._block(128,         128, 8),
        )
        self.attention = TemporalAttention(128)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def _block(self, in_c, out_c, d):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, 3, padding=d, dilation=d),
            nn.BatchNorm1d(out_c),
            nn.ReLU(),
            nn.Dropout1d(0.1),
        )

    def forward(self, x):
        x = self.network(x.permute(0, 2, 1))   # (B, C, T) conv, then attention
        return self.head(self.attention(x)).squeeze(-1)


class LogitCalibrator(nn.Module):
    """Platt scaling: |a| · z + b (|a| keeps the transform monotone)."""
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, z):
        return torch.abs(self.a) * z + self.b


# ─── WINDOWING ────────────────────────────────────────────────────────────────

def composite_risk_score(future_df):
    """Sum of severity weights for each error type present in the future slice."""
    return sum(
        SEVERITY[col] * int((future_df[col] > 0).any())
        for col in SEVERITY
        if col in future_df.columns
    )


def build_windows(df, fine_stride=True):
    """
    Slide a LOOKBACK_S window over each (driver, route) session.

    fine_stride=True  (training): vicinity densification around positive events
                                  to give the model more examples near risk onset.
    fine_stride=False (evaluation): fixed WINDOW_STEP, unbiased AUC / metric set.

    Returns
    -------
    X      : (N, T, C) float32  raw signal values
    y      : (N,)      int      binary label (composite risk > 0)
    scores : (N,)      int      composite risk score value
    pids   : (N,)      object   participant ID
    """
    windows, labels, scores, pids = [], [], [], []

    for (pid, route), grp in df.groupby(["id", "route"]):
        grp = grp.sort_values("Timestamp").reset_index(drop=True)
        n   = len(grp)

        if fine_stride:
            # Pass 1: locate all positive-onset starts at WINDOW_STEP
            pos_starts = set()
            idx = 0
            while idx + LOOKBACK_S + GAP + HORIZON <= n:
                sig    = grp.iloc[idx: idx + LOOKBACK_S][SIGNAL_COLS].values.astype(np.float32)
                future = grp.iloc[idx + LOOKBACK_S + GAP: idx + LOOKBACK_S + GAP + HORIZON]
                if not np.isnan(sig).any() and composite_risk_score(future) > 0:
                    pos_starts.add(idx)
                idx += WINDOW_STEP

            # Pass 2: densify EVENT_VICINITY rows before each positive start
            vicinity = set()
            for ps in pos_starts:
                for offset in range(-EVENT_VICINITY, 1):
                    t = ps + offset
                    if t >= 0:
                        vicinity.add(t)

            seen = set()
            idx  = 0
            while idx + LOOKBACK_S + GAP + HORIZON <= n:
                step = 1 if idx in vicinity else WINDOW_STEP
                if idx not in seen:
                    sig = grp.iloc[idx: idx + LOOKBACK_S][SIGNAL_COLS].values.astype(np.float32)
                    if not np.isnan(sig).any():
                        future = grp.iloc[idx + LOOKBACK_S + GAP: idx + LOOKBACK_S + GAP + HORIZON]
                        score  = composite_risk_score(future)
                        windows.append(sig)
                        scores.append(score)
                        labels.append(int(score > 0))
                        pids.append(pid)
                        seen.add(idx)
                idx += step

        else:
            idx = 0
            while idx + LOOKBACK_S + GAP + HORIZON <= n:
                sig = grp.iloc[idx: idx + LOOKBACK_S][SIGNAL_COLS].values.astype(np.float32)
                if not np.isnan(sig).any():
                    future = grp.iloc[idx + LOOKBACK_S + GAP: idx + LOOKBACK_S + GAP + HORIZON]
                    score  = composite_risk_score(future)
                    windows.append(sig)
                    scores.append(score)
                    labels.append(int(score > 0))
                    pids.append(pid)
                idx += WINDOW_STEP

    return (np.stack(windows), np.array(labels),
            np.array(scores),  np.array(pids))


# ─── VALIDITY REPORT ──────────────────────────────────────────────────────────

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

    print(f"\nPredictive validity — Mann-Whitney U (pos vs neg, window-mean per signal):")
    print(f"  {'Signal':<25}  {'Mean(neg)':>10}  {'Mean(pos)':>10}  {'p-value':>12}  Sig")
    for i, col in enumerate(SIGNAL_COLS):
        neg_v = X_raw[y == 0, :, i].mean(axis=1)
        pos_v = X_raw[y == 1, :, i].mean(axis=1)
        if len(pos_v) < 2:
            continue
        _, p = mannwhitneyu(pos_v, neg_v, alternative="two-sided")
        sig  = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"  {col:<25}  {neg_v.mean():>10.4f}  {pos_v.mean():>10.4f}  {p:>12.2e}  {sig}")

    print(f"\nPer-driver positive rate (eval-stride windows):")
    print(f"  {'Driver':<12}  {'N':>6}  {'Pos':>5}  {'Rate%':>7}  Tier")
    for d in np.unique(pid):
        m    = pid == d
        nd   = int(m.sum())
        np_  = int(y[m].sum())
        rate = np_ / nd * 100
        tier = "HIGH" if rate >= 10 else ("MED" if rate >= 5 else "LOW")
        print(f"  {d:<12}  {nd:>6}  {np_:>5}  {rate:>6.1f}%  {tier}")
    print("=" * 70)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv("relab+unibo_dataset.csv")
    df = mark_event_onsets(df)   # collapse contiguous error runs → single onset
    df = normalize_signals(df)   # z-score vehicle dynamics per (driver, route)

    X_raw_tr, y_tr_all, scores_tr, pid_tr_all = build_windows(df, fine_stride=True)
    X_raw_te, y_te_all, scores_te, pid_te_all = build_windows(df, fine_stride=False)

    # Buffer: number of eval-stride (WINDOW_STEP) rows to skip between consecutive
    # slices so the lookback window of the first window in a slice cannot overlap
    # with the preceding slice.
    BUFFER  = (LOOKBACK_S + GAP + HORIZON + WINDOW_STEP - 1) // WINDOW_STEP
    n_raw   = len(SIGNAL_COLS)
    n_feats = n_raw * 5          # after engineer_features

    print_validity_report(X_raw_te, y_te_all, scores_te, pid_te_all)

    print(f"\n{'='*70}")
    print("PIPELINE CONFIGURATION")
    print(f"{'='*70}")
    print(f"Signals           : {SIGNAL_COLS}")
    print(f"Normalisation     : vehicle z-scored per (driver, route); physiology absolute")
    print(f"Feature channels  : {n_raw} raw → {n_feats} engineered (TCN)")
    print(f"Baseline features : {n_raw} signals × 3 stats = {n_raw*3} (mean, std, max)")
    print(f"Loss              : focal (gamma=2)")
    print(f"TCN architecture  : 4 blocks (d=1,2,4,8); RF = 30 s; 128 final channels")
    print(f"Epochs            : {EPOCHS} with CosineAnnealingLR")
    print(f"Personalisation   : L2-tethered fine-tuning (last TCN block + head)")
    print(f"Calibration       : Platt scaling on held-out cal slice (no leakage)")
    print(f"Bootstrap CI      : {N_BOOTSTRAP} resamples (95 % CI on pooled AUC)")
    print(f"{'='*70}\n")

    drivers = [d for d in np.unique(pid_te_all)
               if y_te_all[pid_te_all == d].sum() >= MIN_POSITIVES]
    rng = np.random.default_rng(SEED)

    hdr = (f"{'Driver':<10} | {'N_eval':>6} {'PosR%':>6} | "
           f"{'PopAUC':>7} {'HybAUC':>7} {'Gain':>7} | "
           f"{'BSS_pop':>7} {'BSS_hyb':>7} | "
           f"{'Pers':>5} {'PPosR%':>6} | Notes")
    print(hdr)
    print("-" * len(hdr))

    pool_y                          = []
    pool_probs_pop, pool_probs_hyb  = [], []   # calibrated, for Brier/BSS
    pool_probs_lr,  pool_probs_xgb  = [], []   # baselines
    pool_X_eval,    pool_models     = [], []   # uncalibrated models for permutation importance
    per_driver_results              = []

    for d in drivers:
        # ── LOPO split ────────────────────────────────────────────────────────
        mask_tr    = pid_tr_all != d
        X_tr, y_tr = X_raw_tr[mask_tr], y_tr_all[mask_tr]
        pid_tr     = pid_tr_all[mask_tr]

        mask_te    = pid_te_all == d
        X_te, y_te = X_raw_te[mask_te], y_te_all[mask_te]

        # Validation fold: 15 % of training participants (for early stopping)
        val_ids = rng.choice(
            np.unique(pid_tr),
            max(1, int(0.15 * len(np.unique(pid_tr)))),
            replace=False,
        )
        vmask = np.isin(pid_tr, val_ids)

        if len(np.unique(y_tr[vmask])) < 2:
            print(f"{d:<10} | SKIP — val fold single-class")
            continue

        # ── Feature engineering + StandardScaler (TCN) ────────────────────────
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

        # ── Baseline features (window stats, StandardScaler) ─────────────────
        X_bl_tr_f  = window_baseline_feats(X_tr[~vmask])
        X_bl_te_f  = window_baseline_feats(X_te)
        bl_scaler  = StandardScaler()
        X_bl_tr_sc = bl_scaler.fit_transform(X_bl_tr_f)
        X_bl_te_sc = bl_scaler.transform(X_bl_te_f)

        # ── Fit baselines on training fold ────────────────────────────────────
        pos_w   = (y_tr[~vmask] == 0).sum() / max(1, (y_tr[~vmask] == 1).sum())
        lr_clf  = LogisticRegression(C=1.0, class_weight="balanced",
                                     max_iter=1000, random_state=SEED)
        lr_clf.fit(X_bl_tr_sc, y_tr[~vmask])

        xgb_clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            scale_pos_weight=pos_w, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, verbosity=0,
        )
        xgb_clf.fit(X_bl_tr_sc, y_tr[~vmask])

        # ── Train TCN population model ─────────────────────────────────────────
        model     = TCN_Attention_Net(n_feats).to(DEVICE)
        opt       = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        loader    = get_balanced_loader(Xtr_sc, y_tr[~vmask], augment=True)

        best_auc, best_w = float("-inf"), None
        for _ in range(EPOCHS):
            model.train()
            for xb, yb in loader:
                bce  = F.binary_cross_entropy_with_logits(
                    model(xb.to(DEVICE)), yb.to(DEVICE), reduction="none")
                pt   = torch.exp(-bce)
                loss = ((1 - pt) ** 2 * bce).mean()
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
        if best_auc < 0.5:
            print(f"{d:<10} | WARN — best val AUC {best_auc:.3f} < 0.5")

        # ── Temporal split: pers → BUFFER → cal → BUFFER → eval ──────────────
        #
        #  pers : first contiguous block with ≥ HYBRID_MIN_POSITIVES positives
        #  cal  : next block (after buffer) with ≥ CAL_MIN_POSITIVES positives
        #  eval : remainder (after second buffer) — primary metric set
        #
        end_pers = min_count_boundary(y_te, HYBRID_MIN_POSITIVES)
        if end_pers is None:
            end_pers, pers_skipped = len(y_te), True
        else:
            pers_skipped = False

        X_pers, y_pers = Xte_sc[:end_pers], y_te[:end_pers]

        cal_start = end_pers + BUFFER
        if cal_start >= len(y_te):
            print(f"{d:<10} | SKIP — no room for cal after pers+buffer")
            continue

        end_cal = min_count_boundary(y_te, CAL_MIN_POSITIVES, start=cal_start)
        if end_cal is None:
            print(f"{d:<10} | SKIP — not enough positives for cal slice")
            continue

        X_cal, y_cal = Xte_sc[cal_start:end_cal], y_te[cal_start:end_cal]

        eval_start = end_cal + BUFFER
        if eval_start >= len(y_te):
            print(f"{d:<10} | SKIP — no room for eval after cal+buffer")
            continue

        X_eval, y_eval = Xte_sc[eval_start:], y_te[eval_start:]

        if len(X_eval) < 10 or y_eval.sum() == 0 or y_eval.sum() == len(y_eval):
            print(f"{d:<10} | SKIP — eval degenerate ({int(y_eval.sum())}/{len(y_eval)} pos)")
            continue

        reportable   = int(y_eval.sum()) >= MIN_EVAL_POSITIVES
        driver_rate  = y_te.mean()
        dense_enough = driver_rate >= PERS_MIN_DRIVER_RATE

        # ── L2-tethered personalisation ───────────────────────────────────────
        model_hyb = copy.deepcopy(model)
        for name, param in model_hyb.named_parameters():
            param.requires_grad = any(x in name for x in ["head", "network.3"])

        personalised = False
        if not pers_skipped and dense_enough and y_pers.sum() >= HYBRID_MIN_POSITIVES:
            pop_params  = {pname: p.clone().detach() for pname, p in model.named_parameters()}
            pers_loader = get_balanced_loader(
                X_pers, y_pers,
                batch_size=max(1, min(BATCH_SIZE, len(X_pers))),
                augment=False,
            )
            opt_h = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model_hyb.parameters()),
                lr=HYBRID_LR, weight_decay=WEIGHT_DECAY,
            )
            n_batches = max(1, len(pers_loader))
            prev_loss, patience, pat_count = float("inf"), 5, 0
            model_hyb.train()
            for _ in range(HYBRID_EPOCHS):
                ep_loss = 0.0
                for xb, yb in pers_loader:
                    logits = model_hyb(xb.to(DEVICE))
                    bce    = F.binary_cross_entropy_with_logits(
                                logits, yb.to(DEVICE), reduction="none")
                    pt     = torch.exp(-bce)
                    task   = ((1 - pt) ** 2 * bce).mean()
                    tether = sum(
                        ((p - pop_params[pname]) ** 2).sum()
                        for pname, p in model_hyb.named_parameters() if p.requires_grad
                    )
                    batch_loss = task + LAMBDA_TETHER * tether
                    opt_h.zero_grad(); batch_loss.backward(); opt_h.step()
                    ep_loss += batch_loss.item()
                ep_loss /= n_batches
                if abs(prev_loss - ep_loss) < 1e-5:
                    pat_count += 1
                    if pat_count >= patience:
                        break
                else:
                    pat_count = 0
                prev_loss = ep_loss
            personalised = True

        # ── Platt calibration on held-out cal slice ───────────────────────────
        model.eval(); model_hyb.eval()
        with torch.no_grad():
            X_cal_t   = torch.as_tensor(X_cal, dtype=torch.float32).to(DEVICE)
            z_cal_pop = model(X_cal_t)
            z_cal_hyb = model_hyb(X_cal_t)

        cal_pop = fit_calibrator(z_cal_pop, y_cal, DEVICE)
        cal_hyb = fit_calibrator(z_cal_hyb, y_cal, DEVICE)

        # ── Inference on eval slice ───────────────────────────────────────────
        with torch.no_grad():
            X_eval_t = torch.as_tensor(X_eval, dtype=torch.float32).to(DEVICE)
            p_pop    = torch.sigmoid(cal_pop(model(X_eval_t))).cpu().numpy()
            p_hyb    = torch.sigmoid(cal_hyb(model_hyb(X_eval_t))).cpu().numpy()
            # Uncalibrated logits kept for permutation importance (AUC = rank-invariant)
            logits_pop_eval = model(X_eval_t).cpu().numpy()

        X_bl_eval = X_bl_te_sc[eval_start:]
        lr_probs  = lr_clf.predict_proba(X_bl_eval)[:, 1]
        xgb_probs = xgb_clf.predict_proba(X_bl_eval)[:, 1]

        # BSS — valid here because calibrator was fitted on a separate cal slice
        bs_ref  = brier_score_loss(y_eval, np.full_like(y_eval, y_eval.mean(), dtype=float))
        bss_pop = 1 - brier_score_loss(y_eval, p_pop) / (bs_ref + 1e-12)
        bss_hyb = 1 - brier_score_loss(y_eval, p_hyb) / (bs_ref + 1e-12)

        # Accumulate
        pool_y.append(y_eval)
        pool_probs_pop.append(p_pop)
        pool_probs_hyb.append(p_hyb)
        pool_probs_lr.append(lr_probs)
        pool_probs_xgb.append(xgb_probs)
        pool_X_eval.append(X_eval.copy())
        pool_models.append(copy.deepcopy(model))   # population model, pre-personalisation

        auc_pop  = safe_auc(y_eval, p_pop)
        auc_hyb  = safe_auc(y_eval, p_hyb)
        gain_auc = (auc_hyb - auc_pop) if (auc_pop is not None and auc_hyb is not None) \
                   else float("nan")
        pos_rate = y_eval.mean() * 100
        ppers_r  = y_pers.mean() * 100 if len(y_pers) else 0.0

        notes = []
        if not dense_enough:
            notes.append(f"NO_PERS(rate={driver_rate*100:.1f}%<{PERS_MIN_DRIVER_RATE*100:.0f}%)")
        elif not personalised:
            notes.append(f"NO_PERS(pers_pos={int(y_pers.sum())}<{HYBRID_MIN_POSITIVES})")
        if pos_rate < 10:
            notes.append("LOW_RATE")

        row = (
            f"{d:<10} | {len(y_eval):>6} {pos_rate:>5.1f}% | "
            f"{auc_pop or 0:>7.4f} {auc_hyb or 0:>7.4f} {gain_auc:>+7.4f} | "
            f"{bss_pop:>7.4f} {bss_hyb:>7.4f} | "
            f"{len(y_pers):>5} {ppers_r:>5.1f}% | "
            + ", ".join(notes)
        )
        if reportable:
            print(row)
        else:
            print(row + f"  ← only {int(y_eval.sum())} eval pos")

        if reportable:
            per_driver_results.append({
                "driver":       d,
                "n_eval":       len(y_eval),
                "pos_rate_%":   round(pos_rate, 1),
                "auc_pop":      auc_pop,
                "auc_hyb":      auc_hyb,
                "gain_auc":     gain_auc,
                "bss_pop":      bss_pop,
                "bss_hyb":      bss_hyb,
                "gain_bss":     bss_hyb - bss_pop,
                "n_pers":       len(y_pers),
                "pos_pers":     int(y_pers.sum()),
                "n_cal":        len(y_cal),
                "pos_cal":      int(y_cal.sum()),
                "personalised": personalised,
                "dense_enough": dense_enough,
            })

    # ================================================================
    # POOLED EVALUATION  (primary result)
    # ================================================================
    print("\n" + "=" * 70)
    print("POOLED EVALUATION  (primary result)")
    print("=" * 70)

    if not pool_y:
        print("No drivers produced a valid eval slice.")
        return

    y_pool    = np.concatenate(pool_y)
    pp_cal    = np.concatenate(pool_probs_pop)   # Platt-calibrated
    ph_cal    = np.concatenate(pool_probs_hyb)   # Platt-calibrated
    plr_all   = np.concatenate(pool_probs_lr)
    pxgb_all  = np.concatenate(pool_probs_xgb)
    n_pool    = len(y_pool)
    pos_pool  = int(y_pool.sum())

    print(f"Windows: {n_pool}  |  Positives: {pos_pool} ({pos_pool/n_pool*100:.1f}%)  "
          f"|  Drivers: {len(pool_y)}\n")
    print(f"AUC: 95 % CI via bootstrap ({N_BOOTSTRAP} resamples).")
    print("Brier / BSS: TCN probabilities Platt-calibrated on per-driver held-out "
          "cal slice; baselines use raw model output.\n")

    bs_ref_pool = brier_score_loss(
        y_pool, np.full_like(y_pool, y_pool.mean(), dtype=float))

    models_eval = [
        ("LR (baseline)",       plr_all),
        ("XGBoost (baseline)",  pxgb_all),
        ("TCN-Population",      pp_cal),
        ("TCN-Hybrid",          ph_cal),
    ]

    print(f"  {'Model':<22}  {'AUC':>6}  {'95% CI':^17}  {'AUPRC':>6}  "
          f"{'Brier':>6}  {'BSS':>7}")
    print(f"  {'-'*22}  {'-'*6}  {'-'*17}  {'-'*6}  {'-'*6}  {'-'*7}")

    pooled_metrics = {}
    for name, probs in models_eval:
        auc   = safe_auc(y_pool, probs) or float("nan")
        auprc = safe_auprc(y_pool, probs) or float("nan")
        brier = brier_score_loss(y_pool, probs)
        bss   = 1 - brier / (bs_ref_pool + 1e-12)
        lo, hi = bootstrap_auc_ci(y_pool, probs)
        ci_str = f"[{lo:.3f} – {hi:.3f}]"
        print(f"  {name:<22}  {auc:.4f}  {ci_str:^17}  {auprc:.4f}  "
              f"{brier:.4f}  {bss:>+7.4f}")
        pooled_metrics[name] = {"auc": auc, "auprc": auprc, "brier": brier,
                                 "bss": bss, "ci_lo": lo, "ci_hi": hi}

    best_bl_auc = max(pooled_metrics["LR (baseline)"]["auc"],
                      pooled_metrics["XGBoost (baseline)"]["auc"])
    tcn_pop_auc = pooled_metrics["TCN-Population"]["auc"]
    tcn_hyb_auc = pooled_metrics["TCN-Hybrid"]["auc"]
    print(f"\n  Best baseline AUC : {best_bl_auc:.4f}")
    print(f"  TCN-Pop  gain     : {tcn_pop_auc - best_bl_auc:+.4f}")
    print(f"  TCN-Hyb  gain     : {tcn_hyb_auc - best_bl_auc:+.4f}")
    print(f"  Hybrid vs Pop     : {tcn_hyb_auc - tcn_pop_auc:+.4f}")

    # ================================================================
    # PERMUTATION FEATURE IMPORTANCE  (TCN-Population, pooled)
    # ================================================================
    print("\n" + "=" * 70)
    print("PERMUTATION FEATURE IMPORTANCE  (TCN-Population, pooled)")
    print("=" * 70)
    print("Each signal's 5 feature groups (raw, Δ1, roll_mean, roll_std, z-score)")
    print("are jointly permuted along the sample axis.  AUC is rank-invariant so")
    print("calibration has no effect on these values.\n")

    # Baseline AUC from uncalibrated sigmoid (rank-invariant = same as calibrated)
    perm_baseline_probs = np.concatenate([
        torch.sigmoid(
            mdl(torch.as_tensor(X_e, dtype=torch.float32).to(DEVICE))
        ).detach().cpu().numpy()
        for X_e, mdl in zip(pool_X_eval, pool_models)
    ])
    baseline_auc = safe_auc(y_pool, perm_baseline_probs) or float("nan")

    perm_results = []
    for sig_idx, sig_name in enumerate(SIGNAL_COLS):
        feat_cols = [sig_idx + k * n_raw for k in range(5)]
        perm_probs_all = []

        for X_e, mdl in zip(pool_X_eval, pool_models):
            X_perm   = X_e.copy()
            perm_idx = rng.permutation(len(X_perm))
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

    # ================================================================
    # PER-DRIVER SUMMARY
    # ================================================================
    if per_driver_results:
        rdf = pd.DataFrame(per_driver_results)
        print("\n" + "=" * 70)
        print(f"PER-DRIVER SUMMARY  ({len(rdf)} reportable drivers, "
              f"≥{MIN_EVAL_POSITIVES} eval positives)")
        print("=" * 70)
        with pd.option_context("display.float_format", "{:.4f}".format):
            print(rdf[["auc_pop", "auc_hyb", "gain_auc",
                        "bss_pop", "bss_hyb", "gain_bss",
                        "pos_rate_%", "n_eval"]]
                  .agg(["mean", "median", "std", "min", "max"]).T.to_string())

        n_imp      = (rdf["gain_auc"] > 0).sum()
        n_hurt     = (rdf["gain_auc"] < 0).sum()
        n_pers_ran = int(rdf["personalised"].sum())
        print(f"\nPersonalisation ran for {n_pers_ran}/{len(rdf)} reportable drivers.")
        print(f"Hybrid improved AUC for {n_imp}/{len(rdf)}, hurt {n_hurt}/{len(rdf)}.")

        pers_df  = rdf[rdf["personalised"]]
        npers_df = rdf[~rdf["personalised"]]
        if len(pers_df):
            print(f"\nPersonalised     (n={len(pers_df)}): "
                  f"mean gain AUC = {pers_df['gain_auc'].mean():+.4f}  "
                  f"mean gain BSS = {pers_df['gain_bss'].mean():+.4f}")
        if len(npers_df):
            print(f"Non-personalised (n={len(npers_df)}): "
                  f"mean gain AUC = {npers_df['gain_auc'].mean():+.4f}  (sanity: ~0)")
    else:
        print("\nNo reportable drivers.  Rely on pooled evaluation above.")


if __name__ == "__main__":
    main()
