"""
tcn_fomaml.py

First-Order Model-Agnostic Meta-Learning (FOMAML) for rapid driver personalisation.

Standard training (tcn_impairment_detect.py) optimises the population objective:
    θ* = argmin_θ  Σ_{d ∈ train}  L(θ, D_d)

FOMAML optimises the meta-objective — minimise loss AFTER adaptation:
    θ* = argmin_θ  Σ_{d ∈ train}  L(θ_d', Q_d)
    where  θ_d' = SGD^{N_INNER}(θ, S_d)   ← inner loop on support set S_d
           Q_d                              ← query set for the same driver

The first-order approximation (FOMAML, Finn et al. 2017) drops second-order
terms: the meta-gradient is computed w.r.t. the adapted params θ_d' rather
than tracing back through the inner loop. This makes the implementation
equivalent to standard backprop — no higher-order autodiff required.

At test time (LOPO held-out driver):
    1. Run N_INNER_STEPS of SGD on the pers windows → θ_adapted.
    2. Evaluate on the remaining eval windows.

Output mirrors tcn_impairment_detect.py:
  - Label validity report (Mann-Whitney U, per-driver tier)
  - Pipeline configuration summary
  - Per-driver table: LR / XGBoost / TCN-Meta / TCN-FOMAML AUC
  - Pooled evaluation: AUC + 95% CI + AUPRC + Brier for all models
  - Permutation feature importance (TCN-Meta, pooled)
  - Modality ablation (standard TCN training for each signal subset)
  - Stratified evaluation (CLC-only vs Non-CLC positive windows)
  - Per-driver summary statistics
"""

import copy
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss)
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu
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

SEVERITY = {
    "Collision":               5,
    "Red_light_violation":     3,
    "panic_braking_with_stop": 2,
    "center_line_crossing":    2,
    "panic_braking":           1,
    "sharp_turn":              1,
}

# Sentinel must be defined before ABLATION_CONDITIONS so the dict key and the
# sentinel are the same object — no risk of a string typo breaking the comparison.
_META_POP_KEY = "FOMAML-pop. (no adapt)"
ABLATION_CONDITIONS = {
    # The first two use standard TCN training on subsets of signals.
    "Physiology + speed":  ["arousal", "hr", "speed.x"],
    "Kinematics":          ["steeringWheelAngle", "steeringTorq", "acceleration.y"],
    # Reference entry: FOMAML-trained population model, zero test-time adaptation.
    # Training objective differs from the two entries above (meta vs standard).
    # Reported in a separate column of the ablation table to avoid conflation.
    _META_POP_KEY:         SIGNAL_COLS,
}

# Windowing — identical to tcn_impairment_detect.py
LOOKBACK_S    = 60
WINDOW_STEP   = 5
GAP, HORIZON  = 15, 5
ROLL_K        = 10
JITTER_STD    = 0.01
CUTOUT_LEN    = 5
CUTOUT_PROB   = 0.2
BATCH_SIZE    = 64
WEIGHT_DECAY  = 1e-4
N_BOOTSTRAP   = 2000
N_PERM        = 30    # repetitions for permutation importance (averaged)
# engineer_features concatenates: raw, diff, roll_mean, roll_std, z_score
N_FEAT_GROUPS = 5
MIN_POSITIVES      = 1
MIN_EVAL_POSITIVES = 3
PERS_MIN_DRIVER_RATE = 0.05

# Standard TCN training (for baselines and ablation)
EPOCHS_STD = 100
LR_STD     = 1e-3

# ── FOMAML hyperparameters ─────────────────────────────────────────────────────
META_STEPS      = 500   # outer gradient steps per LOPO fold
META_BATCH_SIZE = 4     # tasks (drivers) sampled per meta-step
META_LR         = 1e-3  # outer (meta) learning rate  β
INNER_LR        = 1e-2  # inner (adaptation) LR       α  (typically 10× outer)
N_INNER_STEPS   = 5     # gradient steps in inner loop — same at train and test
N_SUPPORT       = 20    # support windows per meta-training task

# Test-time: None = all layers (most faithful to FOMAML theory).
# Restrict to ["head", "network.4"] for more conservative adaptation
# that reduces the risk of catastrophic forgetting on small support sets.
TEST_ADAPT_LAYERS = None

HYBRID_MIN_POSITIVES = 5
# Minimum query windows required per meta-training task (guarantees a
# meaningful gradient even after the support set is carved out).
MIN_QUERY_WINDOWS = 10

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
    T, C   = window.shape
    diff1  = np.diff(window, axis=0, prepend=window[:1])
    cs     = np.cumsum(window,       axis=0)
    cs_sq  = np.cumsum(window ** 2,  axis=0)
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


def apply_features(X_raw, col_idx=None):
    if col_idx is not None:
        X_raw = X_raw[:, :, col_idx]
    return np.stack([engineer_features(w) for w in X_raw])


def window_baseline_feats(X_raw, col_idx=None):
    X = X_raw if col_idx is None else X_raw[:, :, col_idx]
    return np.concatenate([X.mean(1), X.std(1), X.max(1)], axis=1).astype(np.float32)

# ── WINDOWING ──────────────────────────────────────────────────────────────────

def composite_risk_score(future_df):
    return sum(
        SEVERITY[col] * int((future_df[col] > 0).any())
        for col in SEVERITY
        if col in future_df.columns
    )


def build_windows(df, fine_stride=False):
    windows, labels, pids, etypes = [], [], [], []
    for (pid, route), grp in df.groupby(["id", "route"]):
        grp = grp.sort_values("Timestamp").reset_index(drop=True)
        n   = len(grp)
        if fine_stride:
            pos_starts = set()
            idx = 0
            while idx + LOOKBACK_S + GAP + HORIZON <= n:
                sig    = grp.iloc[idx: idx + LOOKBACK_S][SIGNAL_COLS].values.astype(np.float32)
                future = grp.iloc[idx + LOOKBACK_S + GAP: idx + LOOKBACK_S + GAP + HORIZON]
                if not np.isnan(sig).any() and composite_risk_score(future) > 0:
                    for offset in range(-5, 6):
                        s = max(0, idx + offset)
                        if s not in pos_starts and s + LOOKBACK_S + GAP + HORIZON <= n:
                            s2 = grp.iloc[s: s + LOOKBACK_S][SIGNAL_COLS].values.astype(np.float32)
                            if not np.isnan(s2).any():
                                pos_starts.add(s)
                idx += 1
            added = set(pos_starts)
            idx = 0
            while idx + LOOKBACK_S + GAP + HORIZON <= n:
                sig    = grp.iloc[idx: idx + LOOKBACK_S][SIGNAL_COLS].values.astype(np.float32)
                future = grp.iloc[idx + LOOKBACK_S + GAP: idx + LOOKBACK_S + GAP + HORIZON]
                if not np.isnan(sig).any() and composite_risk_score(future) == 0 and idx not in added:
                    windows.append(sig); labels.append(0); pids.append(pid)
                    etypes.append(frozenset())
                idx += WINDOW_STEP
            for s in sorted(pos_starts):
                sig    = grp.iloc[s: s + LOOKBACK_S][SIGNAL_COLS].values.astype(np.float32)
                future = grp.iloc[s + LOOKBACK_S + GAP: s + LOOKBACK_S + GAP + HORIZON]
                if not np.isnan(sig).any():
                    label = 1 if composite_risk_score(future) > 0 else 0
                    et    = frozenset(c for c in SEVERITY if (future[c] > 0).any())
                    windows.append(sig); labels.append(label)
                    pids.append(pid); etypes.append(et)
        else:
            idx = 0
            while idx + LOOKBACK_S + GAP + HORIZON <= n:
                sig    = grp.iloc[idx: idx + LOOKBACK_S][SIGNAL_COLS].values.astype(np.float32)
                future = grp.iloc[idx + LOOKBACK_S + GAP: idx + LOOKBACK_S + GAP + HORIZON]
                if not np.isnan(sig).any():
                    label = 1 if composite_risk_score(future) > 0 else 0
                    et    = frozenset(c for c in SEVERITY if (future[c] > 0).any())
                    windows.append(sig); labels.append(label)
                    pids.append(pid); etypes.append(et)
                idx += WINDOW_STEP
    return (np.array(windows, dtype=np.float32),
            np.array(labels,  dtype=np.float32),
            np.array(pids),
            np.array(etypes, dtype=object))

# ── DATASET & UTILS ────────────────────────────────────────────────────────────

class DrivingDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.as_tensor(X).float()
        self.y = torch.as_tensor(y).float()
        self.augment = augment

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            x += torch.randn_like(x) * JITTER_STD
            if random.random() < CUTOUT_PROB:
                t0 = random.randint(0, x.shape[0] - CUTOUT_LEN)
                x[t0: t0 + CUTOUT_LEN, :] = 0.0
        return x, self.y[idx]


def get_balanced_loader(X, y, batch_size=BATCH_SIZE, augment=False):
    y_int   = y.astype(int)
    counts  = np.bincount(y_int, minlength=2)
    weights = 1.0 / (counts + 1e-6)
    sw      = torch.from_numpy(weights[y_int])
    sampler = WeightedRandomSampler(sw, len(sw))
    return DataLoader(DrivingDataset(X, y, augment=augment),
                      batch_size=batch_size, sampler=sampler)


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2: return None
    return roc_auc_score(y_true, y_score)


def safe_auprc(y_true, y_score):
    if len(np.unique(y_true)) < 2: return None
    return average_precision_score(y_true, y_score)


def _nan(x):
    """Convert None → float('nan'); pass all other values through.
    Safe replacement for `safe_auc(...) or float('nan')`, which incorrectly
    converts a valid AUC of 0.0 to nan due to Python's truthiness rules."""
    return x if x is not None else float("nan")


def bootstrap_auc_ci(driver_aucs, n_boot=N_BOOTSTRAP, seed=SEED):
    aucs = np.array(driver_aucs)
    n    = len(aucs)
    if n < 2: return float("nan"), float("nan")
    rng  = np.random.default_rng(seed)
    boot = [rng.choice(aucs, n, replace=True).mean() for _ in range(n_boot)]
    return tuple(np.percentile(boot, [2.5, 97.5]))


def min_count_boundary(y, min_positives):
    count = 0
    for i in range(len(y)):
        if y[i] == 1:
            count += 1
        if count >= min_positives:
            return i + 1
    return None


def focal_loss(logits, targets, gamma=2.0):
    """Standard focal loss (Lin et al. 2017).

    pt = P(y|x) = sigmoid(logit) for y=1, 1-sigmoid(logit) for y=0.
    Class balance is handled upstream via balanced_batch / WeightedRandomSampler,
    so no pos_weight is applied here — mixing pos_weight into BCE changes pt and
    breaks the standard focal modulation.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt  = torch.exp(-bce)
    return ((1 - pt) ** gamma * bce).mean()


def balanced_batch(X, y, n_each, rng=None):
    """Sample n_each positives and n_each negatives.
    rng: a numpy Generator (np.random.default_rng). Falls back to a fresh
    default_rng when None — always pass an explicit rng for reproducibility."""
    _rng  = rng if rng is not None else np.random.default_rng()
    pos   = np.where(y == 1)[0]
    neg   = np.where(y == 0)[0]
    # Cap n_each to the available minority count so sampling is always
    # without replacement, preventing degenerate batches when one class
    # has very few examples (e.g. repeating the same window N times).
    n_each = min(n_each, len(pos), len(neg))
    p     = _rng.choice(pos, n_each, replace=False)
    n     = _rng.choice(neg, n_each, replace=False)
    idx   = np.concatenate([p, n])
    _rng.shuffle(idx)
    return (torch.as_tensor(X[idx], dtype=torch.float32).to(DEVICE),
            torch.as_tensor(y[idx], dtype=torch.float32).to(DEVICE))


def freeze_norm_layers(model):
    """Set model to train mode, then freeze all normalisation layers to eval.
    Keeps Dropout in train mode while preventing running-stat corruption from
    small support/query batches.  Handles BN1d/2d/3d, LayerNorm, GroupNorm,
    and InstanceNorm so this stays correct if new norm types are added later."""
    model.train()
    _NORM_TYPES = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.LayerNorm, nn.GroupNorm,
        nn.InstanceNorm1d, nn.InstanceNorm2d,
    )
    for m in model.modules():
        if isinstance(m, _NORM_TYPES):
            m.eval()

# ── MODEL — identical to tcn_impairment_detect.py ─────────────────────────────

class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Linear(channels, channels // 2)
        self.score = nn.Linear(channels // 2, 1)

    def forward(self, x):
        x_t     = x.permute(0, 2, 1)
        h       = torch.tanh(self.query(x_t))
        weights = torch.softmax(self.score(h), dim=1)
        return (x_t * weights).sum(dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, d):
        super().__init__()
        self.d    = d
        # padding=0 here; causal left-padding applied in forward so the model
        # cannot attend to future timesteps within the receptive field.
        self.conv = nn.Conv1d(in_c, out_c, 3, padding=0, dilation=d)
        self.bn   = nn.BatchNorm1d(out_c)
        self.drop = nn.Dropout1d(0.1)
        self.res  = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        # Left-only padding = (kernel_size - 1) * dilation = 2 * d
        out = F.pad(x, (2 * self.d, 0))
        out = self.drop(F.relu(self.bn(self.conv(out))))
        return out + self.res(x)


class TCN_Attention_Net(nn.Module):
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

# ── STANDARD TCN TRAINING (for baselines and ablation) ────────────────────────

def train_standard_tcn(model, Xtr_sc, y_tr, Xval_sc, y_val):
    opt       = torch.optim.Adam(model.parameters(), lr=LR_STD, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_STD)
    loader    = get_balanced_loader(Xtr_sc, y_tr, augment=True)
    best_auc, best_w = float("-inf"), None
    for _ in range(EPOCHS_STD):
        model.train()
        for xb, yb in loader:
            loss = focal_loss(model(xb.to(DEVICE)), yb.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(model(
                torch.as_tensor(Xval_sc, dtype=torch.float32).to(DEVICE))).cpu().numpy()
        auc = safe_auc(y_val, preds)
        if auc is not None and auc > best_auc:
            best_auc = auc
            best_w   = copy.deepcopy(model.state_dict())
    if best_w is not None:
        model.load_state_dict(best_w)
    return model

# ── FOMAML CORE ────────────────────────────────────────────────────────────────

def inner_adapt(model, X_sup, y_sup, n_steps=N_INNER_STEPS,
                lr=INNER_LR, layers=None, rng=None):
    """
    Adapt a deep copy of model on (X_sup, y_sup) using n_steps of SGD.
    layers=None updates all parameters (FOMAML training convention).
    Original model is unchanged.
    NOTE: if the support set is single-class (n_each==0), the model is
    returned unchanged — callers should guard with label-diversity checks.
    """
    model_inner = copy.deepcopy(model)
    if layers is not None:
        for name, param in model_inner.named_parameters():
            param.requires_grad = any(
                name == l or name.startswith(l + ".") for l in layers)
        unmatched = [l for l in layers if not any(
            n == l or n.startswith(l + ".") for n, _ in model_inner.named_parameters())]
        if unmatched:
            warnings.warn(
                f"inner_adapt: TEST_ADAPT_LAYERS entries match no parameters: "
                f"{unmatched}. Check layer names against model.named_parameters().")
    params = [p for p in model_inner.parameters() if p.requires_grad]
    inner_opt = torch.optim.SGD(params, lr=lr)
    n_each = min(int((y_sup == 1).sum()), int((y_sup == 0).sum()))
    if n_each == 0:
        # Single-class support set: no balanced adaptation possible.
        return model_inner
    freeze_norm_layers(model_inner)
    for _ in range(n_steps):
        xb, yb = balanced_batch(X_sup, y_sup, n_each, rng=rng)
        loss = focal_loss(model_inner(xb), yb)
        inner_opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        inner_opt.step()
    return model_inner


def fomaml_meta_train(model, Xtr_sc, ytr, pid_tr,
                      Xval_sc=None, yval=None, patience=5, check_every=50):
    """
    META_STEPS outer gradient steps using FOMAML.

    Per meta-step:
      1. Sample META_BATCH_SIZE drivers.
      2. For each: support = first N_SUPPORT windows, query = rest.
      3. Inner loop (all layers, SGD) on support → θ_d'.
      4. Query loss w.r.t. θ_d' → backprop.
      5. Transfer gradients to original θ (FOMAML approximation).
      6. Outer Adam step on accumulated gradients.

    Early stopping: if Xval_sc / yval are supplied, val AUC is computed every
    check_every steps. The best-weight checkpoint is restored on exit;
    training halts when val AUC has not improved for `patience` consecutive
    checks (i.e. patience * check_every outer steps).
    """
    model.train()
    outer_opt = torch.optim.Adam(model.parameters(),
                                 lr=META_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        outer_opt, T_max=META_STEPS)
    training_drivers = list(np.unique(pid_tr))
    meta_rng  = random.Random(SEED)   # deterministic task sampling
    inner_rng = np.random.default_rng(SEED)  # deterministic mini-batch draws

    n_skip_size = 0   # tasks skipped: too few windows
    n_skip_cls  = 0   # tasks skipped: single-class support or query

    use_val      = (Xval_sc is not None and yval is not None
                    and len(np.unique(yval)) >= 2)
    best_val_auc = float("-inf")
    best_weights = copy.deepcopy(model.state_dict())
    no_improve   = 0

    for step in range(META_STEPS):
        batch   = meta_rng.sample(training_drivers,
                                  min(META_BATCH_SIZE, len(training_drivers)))
        outer_opt.zero_grad()
        n_valid = 0

        for d in batch:
            mask = pid_tr == d
            X_d, y_d = Xtr_sc[mask], ytr[mask]
            if len(X_d) < N_SUPPORT + MIN_QUERY_WINDOWS:
                n_skip_size += 1
                continue
            # Stratified support sampling: guarantees both classes appear in
            # both support and query regardless of event clustering in time.
            pos_idx = np.where(y_d == 1)[0]
            neg_idx = np.where(y_d == 0)[0]
            # Reserve at least MIN_QRY_EACH of each class for the query set so
            # the meta-gradient sees a reasonably balanced query loss even for
            # drivers with skewed class distributions.
            MIN_QRY_EACH = 2
            n_sup_each = min(N_SUPPORT // 2,
                             len(pos_idx) - MIN_QRY_EACH,
                             len(neg_idx) - MIN_QRY_EACH)
            if n_sup_each < 1:
                n_skip_cls += 1
                continue
            sup_pos = inner_rng.choice(pos_idx, n_sup_each, replace=False)
            sup_neg = inner_rng.choice(neg_idx, n_sup_each, replace=False)
            sup_idx = np.concatenate([sup_pos, sup_neg])
            qry_idx = np.setdiff1d(np.arange(len(y_d)), sup_idx)
            X_sup, y_sup = X_d[sup_idx], y_d[sup_idx]
            X_qry, y_qry = X_d[qry_idx], y_d[qry_idx]
            if len(np.unique(y_qry)) < 2:
                n_skip_cls += 1
                continue

            model_inner = inner_adapt(model, X_sup, y_sup,
                                      n_steps=N_INNER_STEPS,
                                      lr=INNER_LR, layers=None, rng=inner_rng)

            # Balanced query batch: consistent with inner-loop sampling and uses
            # the standard focal loss formula without pos_weight distortion.
            # n_each capped to min class size so sampling is without replacement.
            n_qry_each = min(int((y_qry == 1).sum()), int((y_qry == 0).sum()),
                             N_SUPPORT // 2)
            if n_qry_each == 0:
                n_skip_cls += 1
                continue
            xb_q, yb_q = balanced_batch(X_qry, y_qry, n_qry_each, rng=inner_rng)

            # Clear inner-loop residual gradients so q_loss.backward() computes
            # only the query gradient — the correct FOMAML meta-gradient.
            # freeze_norm_layers was already called inside inner_adapt; zero_grad
            # here replaces the redundant second call that was previously here.
            model_inner.zero_grad()

            q_loss = focal_loss(model_inner(xb_q), yb_q)
            q_loss.backward()

            # FOMAML: copy gradients from adapted params to original params.
            # outer_opt.zero_grad() was called above, so p_orig.grad starts as
            # None. We accumulate only over valid tasks (skipped tasks contribute
            # neither gradient nor count), then divide by n_valid to get the
            # exact mean gradient — mathematically equivalent to averaging
            # per-task meta-gradients before the outer Adam step.
            for p_orig, p_inner in zip(model.parameters(),
                                       model_inner.parameters()):
                if p_inner.grad is None: continue
                if p_orig.grad is None:
                    # .detach() is defensive: grad tensors are already detached
                    # from the autograd graph, but makes intent explicit.
                    p_orig.grad = p_inner.grad.detach().clone()
                else:
                    p_orig.grad.add_(p_inner.grad)
            n_valid += 1

        if n_valid > 0:
            for p in model.parameters():
                if p.grad is not None: p.grad.div_(n_valid)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            outer_opt.step()
            scheduler.step()  # only step when a gradient was applied

        if use_val and (step + 1) % check_every == 0:
            model.eval()
            with torch.no_grad():
                val_preds = torch.sigmoid(
                    model(torch.as_tensor(Xval_sc, dtype=torch.float32).to(DEVICE))
                ).cpu().numpy()
            model.train()
            val_auc = safe_auc(yval, val_preds)
            if val_auc is not None:
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_weights = copy.deepcopy(model.state_dict())
                    no_improve   = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    print(f"  [meta-train] early stop at step {step+1} "
                          f"(best val AUC={best_val_auc:.4f})", flush=True)
                    break

        if (step + 1) % 100 == 0:
            print(f"  meta-step {step+1}/{META_STEPS}", flush=True)

    if use_val:
        model.load_state_dict(best_weights)
        print(f"  [meta-train] restored best weights "
              f"(val AUC={best_val_auc:.4f})", flush=True)

    print(f"  [meta-train] skip summary — size: {n_skip_size}, "
          f"single-class: {n_skip_cls} (out of "
          f"{META_STEPS * META_BATCH_SIZE} task slots)", flush=True)
    return model

# ── STRATIFICATION HELPERS ────────────────────────────────────────────────────

def _clc_only_mask(et, y):
    """True for positive windows whose only event type is center_line_crossing."""
    return (y == 1) & np.array([
        "center_line_crossing" in e
        and not any(c for c in e if c != "center_line_crossing")
        for e in et
    ])


def _non_clc_mask(et, y):
    """True for positive windows that contain at least one non-CLC event."""
    return (y == 1) & np.array([
        len(e - {"center_line_crossing"}) > 0 for e in et
    ])

# ── VALIDITY REPORT ────────────────────────────────────────────────────────────

def print_validity_report(X_raw, y, scores, pid):
    n_total = len(y); n_pos = int(y.sum())
    print(f"\n{'='*70}")
    print("LABEL VALIDITY REPORT — COMPOSITE RISK TARGET")
    print(f"{'='*70}")
    print(f"Total windows    : {n_total}")
    print(f"Positive (risk>0): {n_pos} ({n_pos/n_total*100:.1f}%)")
    print(f"Negative         : {n_total-n_pos} ({(n_total-n_pos)/n_total*100:.1f}%)")
    print(f"\nRisk score distribution (positives only):")
    for v in sorted(np.unique(scores[y == 1])):
        c = (scores[y == 1] == v).sum()
        print(f"  score={int(v):>2}  n={c:>5}  ({c/n_pos*100:.1f}%)")
    print(f"\nPredictive validity — Mann-Whitney U (pos vs neg, raw signals):")
    print(f"  {'Signal':<25}  {'Mean(neg)':>10}  {'Mean(pos)':>10}  {'p-value':>12}  Sig")
    for i, col in enumerate(SIGNAL_COLS):
        neg_v = X_raw[y == 0, :, i].mean(axis=1)
        pos_v = X_raw[y == 1, :, i].mean(axis=1)
        if len(pos_v) < 2: continue
        _, p = mannwhitneyu(pos_v, neg_v, alternative="two-sided")
        sig  = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"  {col:<25}  {neg_v.mean():>10.4f}  {pos_v.mean():>10.4f}  {p:>12.2e}  {sig}")
    print(f"\nPer-driver positive rate:")
    print(f"  {'Driver':<12}  {'N':>6}  {'Pos':>5}  {'Rate%':>7}  Tier")
    for d in np.unique(pid):
        m    = pid == d
        nd      = int(m.sum()); n_pos_d = int(y[m].sum())
        rate    = n_pos_d / nd * 100
        tier    = "HIGH" if rate >= 10 else ("MED" if rate >= 5 else "LOW")
        print(f"  {d:<12}  {nd:>6}  {n_pos_d:>5}  {rate:>6.1f}%  {tier}")
    print("=" * 70)

# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(Path(__file__).parent / "relab+unibo_dataset.csv")
    df = mark_event_onsets(df)
    df = normalize_signals(df)
    # Sort globally so groupby iterates sessions in a consistent temporal order.
    # Within each (id, route) group build_windows also sorts by Timestamp, but
    # the cross-route ordering within a driver's test set must be deterministic
    # for min_count_boundary (temporal adaptation boundary) to be meaningful.
    df = df.sort_values(["id", "route", "Timestamp"]).reset_index(drop=True)

    # fine_stride=True for training oversamples near positive events (±5 offsets
    # per event onset).  WeightedRandomSampler then rebalances class counts on
    # top.  The two effects are intentional: fine-stride gives the model more
    # distinct pre-event contexts to learn from; the sampler ensures class
    # balance within each mini-batch.  Test windows use regular stride only.
    X_raw_tr, y_tr_all, pid_tr_all, _           = build_windows(df, fine_stride=True)
    X_raw_te, y_te_all, pid_te_all, etypes_te   = build_windows(df, fine_stride=False)

    n_raw   = len(SIGNAL_COLS)
    n_feats = apply_features(X_raw_tr[:1]).shape[2]  # engineer_features expands C → 5C

    # Composite risk scores (for validity report)
    scores_te = np.array([
        sum(SEVERITY[c] for c in et) for et in etypes_te
    ], dtype=float)

    print_validity_report(X_raw_te, y_te_all, scores_te, pid_te_all)

    print(f"\n{'='*70}")
    print("PIPELINE CONFIGURATION")
    print(f"{'='*70}")
    print(f"Signals          : {SIGNAL_COLS}")
    print(f"GAP / HORIZON    : {GAP}s / {HORIZON}s")
    print(f"TCN blocks       : 5 (d=1,2,4,8,16 — RF=63) with residual connections")
    print(f"Meta-steps       : {META_STEPS}  |  Batch : {META_BATCH_SIZE} tasks/step")
    print(f"Outer LR (β)     : {META_LR}  |  Inner LR (α) : {INNER_LR}")
    print(f"Inner steps      : {N_INNER_STEPS}  |  Support : {N_SUPPORT} windows/task")
    print(f"Test adapt layers: {'all' if TEST_ADAPT_LAYERS is None else TEST_ADAPT_LAYERS}")
    print(f"Bootstrap CI     : {N_BOOTSTRAP} resamples (driver-level)")
    print(f"Device           : {DEVICE}")
    print(f"{'='*70}\n")

    drivers = [d for d in np.unique(pid_te_all)
               if y_te_all[pid_te_all == d].sum() >= MIN_POSITIVES]

    hdr = (f"{'Driver':<10} | {'N_eval':>6} {'PosR%':>6} | "
           f"{'LR':>6} {'XGB':>6} {'StdTCN':>7} {'MetaAUC':>8} {'AdaptAUC':>9} {'Gain':>7} | "
           f"{'N_pers':>6} {'PPosR%':>7}")
    print(hdr)
    print("-" * len(hdr))

    # Accumulate for pooled evaluation
    pool_y                = []
    pool_lr, pool_xgb     = [], []
    pool_std              = []
    pool_meta, pool_adapt = [], []
    pool_ablation         = {name: [] for name in ABLATION_CONDITIONS}
    pool_etypes           = []
    pool_X_eval           = []
    pool_model_states     = []  # CPU state dicts — avoids holding 30 live models in GPU memory
    per_driver_results    = []

    for d in drivers:
        # Canonical string for seed encoding: coerce numpy floats (e.g. "1.0")
        # to int strings ("1") so the seed is stable regardless of CSV dtype.
        d_key = str(int(d)) if isinstance(d, (float, np.floating)) else str(d)

        # Per-driver seed: adaptation RNG is independent of driver iteration
        # order, so reordering drivers or changing thresholds cannot shift
        # the random sequence for unrelated drivers.
        # Use deterministic encoding (not hash()) to be reproducible across
        # Python sessions without relying on PYTHONHASHSEED.
        _seed_d_adapt = int.from_bytes(d_key.encode(), "little") & 0xFFFF_FFFF
        test_adapt_rng = np.random.default_rng(SEED + 2 + _seed_d_adapt)

        mask_tr = pid_tr_all != d
        X_tr    = X_raw_tr[mask_tr]; y_tr = y_tr_all[mask_tr]
        pid_tr  = pid_tr_all[mask_tr]

        mask_te = pid_te_all == d
        X_te    = X_raw_te[mask_te]; y_te = y_te_all[mask_te]
        et_te   = etypes_te[mask_te]

        if y_te.sum() < MIN_POSITIVES:
            continue

        # Validation split
        seed_d  = int.from_bytes(d_key.encode(), "little") & 0xFFFFFFFF
        val_ids = np.random.default_rng(SEED ^ seed_d).choice(
            np.unique(pid_tr), max(1, int(0.15 * len(np.unique(pid_tr)))),
            replace=False)
        vmask = np.isin(pid_tr, val_ids)
        if len(np.unique(y_tr[vmask])) < 2:
            print(f"{d:<10} | SKIP — val fold single-class")
            continue

        # ── TCN features & scaling ────────────────────────────────────────
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

        # ── Baseline features (window stats) ─────────────────────────────
        # LR/XGB are trained on fine-stride windows (same as TCN) but evaluated
        # on regular-stride test windows.  This density asymmetry is intentional:
        # the training set is enriched near positive events for all models alike,
        # while the test set reflects natural driving at regular intervals.
        X_bl_tr_f  = window_baseline_feats(X_tr[~vmask])
        y_bl_tr    = y_tr[~vmask]
        X_bl_te_f  = window_baseline_feats(X_te)
        bl_scaler  = StandardScaler()
        X_bl_tr_sc = bl_scaler.fit_transform(X_bl_tr_f)
        X_bl_te_sc = bl_scaler.transform(X_bl_te_f)

        neg_pos_ratio = (y_bl_tr == 0).sum() / max(1, (y_bl_tr == 1).sum())
        lr_clf = LogisticRegression(C=1.0, class_weight="balanced",
                                    max_iter=1000, random_state=SEED)
        lr_clf.fit(X_bl_tr_sc, y_bl_tr)
        xgb_clf = xgb.XGBClassifier(n_estimators=200, max_depth=4,
                                     learning_rate=0.05, scale_pos_weight=neg_pos_ratio,
                                     subsample=0.8, colsample_bytree=0.8,
                                     random_state=SEED, verbosity=0)
        xgb_clf.fit(X_bl_tr_sc, y_bl_tr)

        # ── FOMAML meta-training ──────────────────────────────────────────
        print(f"{d:<10} | meta-training...", flush=True)
        model = TCN_Attention_Net(n_feats).to(DEVICE)
        model = fomaml_meta_train(model, Xtr_sc, y_tr[~vmask], pid_tr[~vmask],
                                  Xval_sc=Xval_sc, yval=y_tr[vmask])
        model.eval()

        # ── Modality ablation (standard TCN training) ─────────────────────
        ablation_probs = {}
        for ab_name, ab_cols in ABLATION_CONDITIONS.items():
            if ab_name == _META_POP_KEY:
                continue  # filled from meta_scores below
            ab_idx    = [SIGNAL_COLS.index(c) for c in ab_cols]
            # Derive feature count from training data, not test, to avoid any
            # accidental dependency on the test driver's window count.
            ab_feats  = apply_features(X_tr[:1], col_idx=ab_idx).shape[2]
            Xtr_ab    = apply_features(X_tr[~vmask], col_idx=ab_idx)
            Xval_ab   = apply_features(X_tr[vmask],  col_idx=ab_idx)
            Xte_ab    = apply_features(X_te,          col_idx=ab_idx)
            sc_ab     = StandardScaler()
            Xtr_ab_sc = sc_ab.fit_transform(
                Xtr_ab.reshape(-1, ab_feats)).reshape(-1, LOOKBACK_S, ab_feats)
            Xval_ab_sc = sc_ab.transform(
                Xval_ab.reshape(-1, ab_feats)).reshape(-1, LOOKBACK_S, ab_feats)
            Xte_ab_sc  = sc_ab.transform(
                Xte_ab.reshape(-1, ab_feats)).reshape(-1, LOOKBACK_S, ab_feats)
            m_ab = TCN_Attention_Net(ab_feats).to(DEVICE)
            m_ab = train_standard_tcn(m_ab, Xtr_ab_sc, y_tr[~vmask],
                                      Xval_ab_sc, y_tr[vmask])
            m_ab.eval()
            with torch.no_grad():
                ablation_probs[ab_name] = torch.sigmoid(
                    m_ab(torch.as_tensor(Xte_ab_sc, dtype=torch.float32).to(DEVICE))
                ).cpu().numpy()

        # ── Meta (no adaptation) scores ───────────────────────────────────
        with torch.no_grad():
            meta_scores = torch.sigmoid(
                model(torch.as_tensor(Xte_sc, dtype=torch.float32).to(DEVICE))
            ).cpu().numpy()
        ablation_probs[_META_POP_KEY] = meta_scores

        # ── Standard TCN (full signals, population-trained) baseline ──────
        print(f"{d:<10} | std-TCN training...", flush=True)
        model_std = TCN_Attention_Net(n_feats).to(DEVICE)
        model_std = train_standard_tcn(model_std, Xtr_sc, y_tr[~vmask],
                                       Xval_sc, y_tr[vmask])
        model_std.eval()
        with torch.no_grad():
            std_scores = torch.sigmoid(
                model_std(torch.as_tensor(Xte_sc, dtype=torch.float32).to(DEVICE))
            ).cpu().numpy()

        # ── Test-time adaptation ──────────────────────────────────────────
        end_pers = min_count_boundary(y_te, HYBRID_MIN_POSITIVES)
        adapt_scores = meta_scores.copy()
        n_pers = 0; ppos_r = 0.0; adapted = False

        # Gate on the support-set positive rate.
        # NOTE: this is an offline / retrospective analysis.  We use ground-truth
        # labels from the support windows (y_te[:end_pers]) to decide whether the
        # driver's error rate is high enough to warrant adaptation.  In a real-time
        # deployment the adaptation decision must be made without knowing whether
        # future events occur; this gate would need to be replaced with a proxy
        # (e.g. model confidence or an external distraction signal).
        sup_rate = y_te[:end_pers].mean() if end_pers is not None else 0.0
        if (end_pers is not None and sup_rate >= PERS_MIN_DRIVER_RATE):
            X_pers, y_pers = Xte_sc[:end_pers], y_te[:end_pers]
            n_pers  = end_pers
            ppos_r  = float(y_pers.mean() * 100)
            if (y_pers == 0).sum() >= 1 and (y_pers == 1).sum() >= 1:
                model_adapted = inner_adapt(model, X_pers, y_pers,
                                            n_steps=N_INNER_STEPS,
                                            lr=INNER_LR,
                                            layers=TEST_ADAPT_LAYERS,
                                            rng=test_adapt_rng)
                model_adapted.eval()
                # Evaluate only on windows NOT seen during adaptation to
                # avoid AUC inflation from support-set leakage.
                # Support windows retain meta (unadapted) scores; post-support
                # windows use adapted scores — giving a fair hybrid signal.
                with torch.no_grad():
                    post_scores = torch.sigmoid(
                        model_adapted(torch.as_tensor(
                            Xte_sc[end_pers:], dtype=torch.float32).to(DEVICE))
                    ).cpu().numpy()
                adapt_scores = meta_scores.copy()
                adapt_scores[end_pers:] = post_scores
                adapted = True

        # Exclude support windows from all per-driver and pooled metrics to
        # prevent AUC inflation from support-set memorisation and to ensure
        # that pooled metrics are always computed on the same post-support
        # window slice regardless of whether the driver was adapted.
        # Using eval_start=0 for non-adapted drivers would mean the pooled
        # comparison mixes full-session windows (non-adapted) with
        # post-support windows (adapted), biasing the aggregate AUC in favour
        # of adapted drivers.
        eval_start = end_pers if end_pers is not None else 0

        # ── Per-driver metrics ────────────────────────────────────────────
        lr_probs  = lr_clf.predict_proba(X_bl_te_sc)[:, 1]
        xgb_probs = xgb_clf.predict_proba(X_bl_te_sc)[:, 1]

        y_ev = y_te[eval_start:]
        auc_lr   = _nan(safe_auc(y_ev, lr_probs[eval_start:]))
        auc_xgb  = _nan(safe_auc(y_ev, xgb_probs[eval_start:]))
        auc_std  = _nan(safe_auc(y_ev, std_scores[eval_start:]))
        auc_meta = _nan(safe_auc(y_ev, meta_scores[eval_start:]))
        auc_adpt = _nan(safe_auc(y_ev, adapt_scores[eval_start:]))
        # Guard against nan: if either side is nan, gain is nan and the driver
        # is excluded from helps/hurts counts (see n_helped/n_hurt below).
        gain  = (auc_adpt - auc_meta
                 if not (np.isnan(auc_adpt) or np.isnan(auc_meta))
                 else float("nan"))
        pos_r = float(y_ev.mean() * 100)

        reportable = int(y_ev.sum()) >= MIN_EVAL_POSITIVES

        # eval_start = end_pers (or 0) for all drivers, so show the fraction
        # so per-driver sample sizes are transparent.
        n_total_te = len(y_te)
        adapt_tag = "ADAPTED" if adapted else "NO_ADAPT"
        if eval_start > 0:
            flags = f" ({adapt_tag}, eval {len(y_ev)}/{n_total_te}, excl. {eval_start} supp.)"
        else:
            flags = f" ({adapt_tag})"
        print(f"{d:<10} | {len(y_ev):>6} {pos_r:>5.1f}% | "
              f"{auc_lr:>6.4f} {auc_xgb:>6.4f} {auc_std:>7.4f} "
              f"{auc_meta:>8.4f} {auc_adpt:>9.4f} {gain:>+7.4f} | "
              f"{n_pers:>6} {ppos_r:>6.1f}%{flags}")

        if reportable:
            pool_y.append(y_ev)
            pool_lr.append(lr_probs[eval_start:])
            pool_xgb.append(xgb_probs[eval_start:])
            pool_std.append(std_scores[eval_start:])
            pool_meta.append(meta_scores[eval_start:])
            pool_adapt.append(adapt_scores[eval_start:])
            pool_etypes.append(et_te[eval_start:])
            pool_X_eval.append(Xte_sc[eval_start:])
            pool_model_states.append({k: v.cpu() for k, v in model.state_dict().items()})
            for ab_name in ABLATION_CONDITIONS:
                ab_p = ablation_probs[ab_name]
                assert ab_p is not None, f"ablation_probs[{ab_name!r}] was never assigned"
                pool_ablation[ab_name].append(ab_p[eval_start:])

            per_driver_results.append({
                "driver": str(d), "n_eval": len(y_ev),
                "pos_rate": pos_r,
                "auc_lr": float(auc_lr), "auc_xgb": float(auc_xgb),
                "auc_std_tcn": float(auc_std),
                "auc_meta": float(auc_meta), "auc_adapted": float(auc_adpt),
                "gain": float(gain), "n_pers": n_pers, "ppos_rate": ppos_r,
                "adapted": adapted,
            })

    # ── Pooled evaluation ──────────────────────────────────────────────────────
    if not per_driver_results:
        print("No reportable drivers."); return

    print("-" * len(hdr))

    all_y     = np.concatenate(pool_y)
    all_lr    = np.concatenate(pool_lr)
    all_xgb   = np.concatenate(pool_xgb)
    all_std   = np.concatenate(pool_std)
    all_meta  = np.concatenate(pool_meta)
    all_adapt = np.concatenate(pool_adapt)

    meta_auc_list  = [r["auc_meta"]    for r in per_driver_results]
    adapt_auc_list = [r["auc_adapted"] for r in per_driver_results]
    std_auc_list   = [r["auc_std_tcn"] for r in per_driver_results]

    lo_m, hi_m     = bootstrap_auc_ci(meta_auc_list)
    lo_a, hi_a     = bootstrap_auc_ci(adapt_auc_list)
    lo_lr, hi_lr   = bootstrap_auc_ci([r["auc_lr"]      for r in per_driver_results])
    lo_xgb, hi_xgb = bootstrap_auc_ci([r["auc_xgb"]    for r in per_driver_results])
    lo_std, hi_std = bootstrap_auc_ci(std_auc_list)

    # Count/stats restricted to drivers that actually adapted — non-adapted
    # drivers always have gain==0 and would dilute the mean gain toward zero.
    adapted_results = [r for r in per_driver_results if r["adapted"]]
    n_helped = sum(1 for r in adapted_results
                   if not np.isnan(r["gain"]) and r["gain"] > 0)
    n_hurt   = sum(1 for r in adapted_results
                   if not np.isnan(r["gain"]) and r["gain"] < 0)
    adapted_gains = [r["gain"] for r in adapted_results if not np.isnan(r["gain"])]
    mean_gain = float(np.mean(adapted_gains)) if adapted_gains else float("nan")

    print(f"\n{'─'*70}")
    print("POOLED EVALUATION (driver-level mean AUC, 95% CI)")
    print(f"{'─'*70}")
    for name, aucs, lo, hi, probs in [
        ("LR",           [r["auc_lr"]  for r in per_driver_results], lo_lr,  hi_lr,  all_lr),
        ("XGBoost",      [r["auc_xgb"] for r in per_driver_results], lo_xgb, hi_xgb, all_xgb),
        ("TCN-Std",      std_auc_list,                                lo_std, hi_std, all_std),
        ("TCN-Meta",     meta_auc_list,                               lo_m,   hi_m,   all_meta),
        ("TCN-FOMAML",   adapt_auc_list,                              lo_a,   hi_a,   all_adapt),
    ]:
        mean_auc  = float(np.mean(aucs))
        auprc     = _nan(safe_auprc(all_y, probs))
        brier     = brier_score_loss(all_y, probs)
        print(f"  {name:<14}: AUC={mean_auc:.4f} [{lo:.3f}–{hi:.3f}]  "
              f"AUPRC={auprc:.3f}  Brier={brier:.3f}")

    print(f"\n  FOMAML net gain  : {mean_gain:+.4f} (adapted drivers only, n={len(adapted_gains)})")
    print(f"  Helps / hurts    : {n_helped} / {n_hurt} of {len(adapted_results)} adapted drivers")

    # ── Permutation feature importance (TCN-Meta, driver-level mean AUC) ─────
    # Baseline and permuted scores both use driver-level mean AUC, matching the
    # primary evaluation metric.  Previously used pooled (concatenated) AUC,
    # which is a different quantity and inconsistent with the reported numbers.
    print(f"\n{'─'*70}")
    print("PERMUTATION FEATURE IMPORTANCE — TCN-Meta (driver-level mean AUC)")
    print(f"{'─'*70}")
    print(f"  {'Signal':<25}  {'Drop':>8}  {'AUC_perm':>10}")

    base_perm = float(np.mean(meta_auc_list))   # driver-level mean, consistent with table
    perm_results = {}
    # Single model instance reused across all (signal, rep, driver) triples to
    # avoid 6 × N_PERM × n_drivers allocations.  state_dict is reloaded in-place
    # before each driver evaluation; load_state_dict handles cross-device copies.
    perm_mdl = TCN_Attention_Net(n_feats).to(DEVICE)
    for sig_i, sig in enumerate(SIGNAL_COLS):
        feat_cols = [sig_i + n_raw * k for k in range(N_FEAT_GROUPS)]
        drop_runs = []
        for rep_i in range(N_PERM):
            # Seed per (signal, repetition) so each run is independent of driver
            # pool size, N_PERM, and other signals.  Changing one signal or
            # dropping a driver cannot shift the permutation sequence for others.
            rng_perm = np.random.default_rng(SEED + 1 + sig_i * N_PERM + rep_i)
            per_driver_perm_aucs = []
            for X_eval, y_ev_d, sd in zip(pool_X_eval, pool_y, pool_model_states):
                X_perm = X_eval.copy()
                perm_idx = rng_perm.permutation(len(X_perm))
                X_perm[:, :, feat_cols] = X_perm[perm_idx][:, :, feat_cols]
                perm_mdl.load_state_dict(sd)  # in-place reload; handles CPU→GPU copies
                perm_mdl.eval()
                with torch.no_grad():
                    p = torch.sigmoid(
                        perm_mdl(torch.as_tensor(X_perm, dtype=torch.float32).to(DEVICE))
                    ).cpu().numpy()
                a = safe_auc(y_ev_d, p)
                if a is not None:
                    per_driver_perm_aucs.append(a)
            if per_driver_perm_aucs:
                drop_runs.append(base_perm - float(np.mean(per_driver_perm_aucs)))
        drop = float(np.mean(drop_runs)) if drop_runs else float("nan")
        perm_results[sig] = drop
        print(f"  {sig:<25}  {drop:>+8.4f}  {base_perm - drop:>10.4f}")

    # ── Modality ablation ─────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("MODALITY ABLATION")
    print("  Standard-TCN conditions: Physiology+speed, Kinematics")
    print(f"  Reference condition ({_META_POP_KEY}): FOMAML population model, zero adaptation")
    print(f"{'─'*70}")
    print(f"  {'Condition':<25}  {'Mean AUC':>9}  {'AUPRC':>7}  {'Brier':>7}")
    ablation_summary = {}
    for ab_name, ab_list in pool_ablation.items():
        if not ab_list: continue
        ab_all  = np.concatenate(ab_list)
        ab_aucs = []
        for i in range(len(ab_list)):
            a = safe_auc(pool_y[i], ab_list[i])
            if a is not None:
                ab_aucs.append(a)
        mean_ab = float(np.mean(ab_aucs)) if ab_aucs else float("nan")
        auprc   = _nan(safe_auprc(all_y, ab_all))
        brier   = brier_score_loss(all_y, ab_all)
        ablation_summary[ab_name] = {
            "mean_auc": mean_ab, "auprc": float(auprc), "brier": float(brier),
        }
        print(f"  {ab_name:<25}  {mean_ab:>9.4f}  {auprc:>7.3f}  {brier:>7.3f}")

    # ── Stratified evaluation ─────────────────────────────────────────────────
    # Uses driver-level mean AUC to be consistent with the primary evaluation
    # metric. Each driver's stratum AUC is computed on their windows only.
    print(f"\n{'─'*70}")
    print("STRATIFIED EVALUATION — TCN-Meta (driver-level mean AUC)")
    print(f"{'─'*70}")
    for stratum_name, mask_fn in [
        ("CLC-only positives",  _clc_only_mask),
        ("Non-CLC positives",   _non_clc_mask),
    ]:
        strat_aucs = []
        n_pos_total = 0
        for y_d, p_d, et_d in zip(pool_y, pool_meta, pool_etypes):
            pos_mask = mask_fn(et_d, y_d)
            neg_mask = y_d == 0
            idx      = np.where(pos_mask | neg_mask)[0]
            if pos_mask.sum() < 1: continue
            n_pos_total += int(pos_mask.sum())
            a = safe_auc(y_d[idx], p_d[idx])
            if a is not None:
                strat_aucs.append(a)
        if n_pos_total < 5 or not strat_aucs: continue
        print(f"  {stratum_name:<30}: n_pos={n_pos_total:>4}  "
              f"AUC={np.mean(strat_aucs):.4f} (n_drivers={len(strat_aucs)})")

    # ── Per-driver summary stats ───────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("PER-DRIVER AUC SUMMARY — TCN-FOMAML adapted")
    print(f"{'─'*70}")
    adapt_arr = np.array(adapt_auc_list)
    print(f"  Mean={adapt_arr.mean():.4f}  Median={np.median(adapt_arr):.4f}  "
          f"Std={adapt_arr.std():.4f}  Min={adapt_arr.min():.4f}  Max={adapt_arr.max():.4f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    results = {
        "summary": {
            "mean_auc_std_tcn": float(np.mean(std_auc_list)),
            "mean_auc_meta":    float(np.mean(meta_auc_list)),
            "mean_auc_adapted": float(np.mean(adapt_auc_list)),
            "net_gain":         mean_gain,  # mean over adapted drivers only
            "ci_std_tcn": [float(lo_std), float(hi_std)],
            "ci_meta":    [float(lo_m),   float(hi_m)],
            "ci_adapted": [float(lo_a),   float(hi_a)],
            "n_helped": n_helped, "n_hurt": n_hurt,
            "n_drivers": len(per_driver_results),
        },
        "permutation_importance": perm_results,
        "ablation": ablation_summary,
        "per_driver": per_driver_results,
        "config": {
            "meta_steps": META_STEPS, "meta_batch_size": META_BATCH_SIZE,
            "meta_lr": META_LR, "inner_lr": INNER_LR,
            "n_inner_steps": N_INNER_STEPS, "n_support": N_SUPPORT,
            "test_adapt_layers": TEST_ADAPT_LAYERS,
            # Thresholds that gate which drivers contribute to pooled results
            "hybrid_min_positives": HYBRID_MIN_POSITIVES,
            "pers_min_driver_rate": PERS_MIN_DRIVER_RATE,
            "min_eval_positives": MIN_EVAL_POSITIVES,
            "min_positives": MIN_POSITIVES,
        },
    }
    out_path = OUT_DIR / "tcn_fomaml.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
