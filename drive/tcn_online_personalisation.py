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

Comparison: frozen population model (same LOPO fold) vs online-personalised.

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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import random

# ── CONFIG (mirrors tcn_impairment_detect.py) ─────────────────────────────────
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

# Population model training (identical to main file)
EPOCHS, LR   = 100, 1e-3
LOOKBACK_S   = 60
WINDOW_STEP  = 5
GAP, HORIZON = 3, 10
BATCH_SIZE   = 64
WEIGHT_DECAY = 1e-4
ROLL_K       = 10
JITTER_STD   = 0.01
CUTOUT_LEN   = 5
CUTOUT_PROB  = 0.2
N_BOOTSTRAP  = 2000
MIN_POSITIVES = 1

# Online personalisation
ONLINE_LR          = 1e-4   # small LR — we're making fine adjustments
ONLINE_STEPS       = 5      # gradient steps after each window
REPLAY_BUFFER_SIZE = 20     # rolling window of recent (x, y) pairs
ONLINE_LAYERS      = ["head"]  # layers updated online; head-only for stability
GRAD_CLIP_NORM     = 1.0    # max gradient norm for online updates

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

# ── WINDOWING ──────────────────────────────────────────────────────────────────

def composite_risk_score(future_df):
    return sum(
        SEVERITY[col] * int((future_df[col] > 0).any())
        for col in SEVERITY
    )


def build_windows(df, fine_stride=True):
    """Returns (windows, labels, pids, routes, t_starts).

    routes and t_starts together allow prequential temporal ordering via
    np.lexsort((t_starts, routes)), which sorts by route first and then by
    timestamp within each route.  This is safe even when Timestamps reset
    to zero at the start of each route (i.e. are not globally unique).

    Note: in fine_stride mode, windows added to pos_starts via ±5-row jitter
    whose futures turn out to be risk-free (composite_risk_score == 0) are
    excluded from both the positive and negative lists to avoid mislabelling.
    """
    windows, labels, pids, routes, t_starts = [], [], [], [], []
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
                    for offset in range(-5, 6):
                        start = max(0, idx + offset)
                        if start not in pos_starts and start + LOOKBACK_S + GAP + HORIZON <= n:
                            s2 = grp.iloc[start: start + LOOKBACK_S][SIGNAL_COLS].values.astype(np.float32)
                            if not np.isnan(s2).any():
                                pos_starts.add(start)
                idx += 1
            # Negative windows at normal stride.
            idx = 0
            added = set(pos_starts)
            while idx + LOOKBACK_S + GAP + HORIZON <= n:
                sig    = grp.iloc[idx: idx + LOOKBACK_S][SIGNAL_COLS].values.astype(np.float32)
                future = grp.iloc[idx + LOOKBACK_S + GAP: idx + LOOKBACK_S + GAP + HORIZON]
                if not np.isnan(sig).any() and composite_risk_score(future) == 0 and idx not in added:
                    windows.append(sig); labels.append(0); pids.append(pid)
                    routes.append(route); t_starts.append(ts[idx])
                idx += WINDOW_STEP
            for s in sorted(pos_starts):
                sig    = grp.iloc[s: s + LOOKBACK_S][SIGNAL_COLS].values.astype(np.float32)
                future = grp.iloc[s + LOOKBACK_S + GAP: s + LOOKBACK_S + GAP + HORIZON]
                if not np.isnan(sig).any():
                    windows.append(sig)
                    labels.append(1 if composite_risk_score(future) > 0 else 0)
                    pids.append(pid)
                    routes.append(route)
                    t_starts.append(ts[s])
        else:
            idx = 0
            while idx + LOOKBACK_S + GAP + HORIZON <= n:
                sig    = grp.iloc[idx: idx + LOOKBACK_S][SIGNAL_COLS].values.astype(np.float32)
                future = grp.iloc[idx + LOOKBACK_S + GAP: idx + LOOKBACK_S + GAP + HORIZON]
                if not np.isnan(sig).any():
                    windows.append(sig)
                    labels.append(1 if composite_risk_score(future) > 0 else 0)
                    pids.append(pid)
                    routes.append(route)
                    t_starts.append(ts[idx])
                idx += WINDOW_STEP

    return (np.array(windows,   dtype=np.float32),
            np.array(labels,    dtype=np.float32),
            np.array(pids),
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


def bootstrap_auc_ci(driver_aucs, n_boot=N_BOOTSTRAP, seed=SEED):
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

# ── ONLINE PERSONALISATION ─────────────────────────────────────────────────────

def online_evaluate_driver(model_pop, Xte_sc, y_te):
    """
    Prequential predict-then-update loop over all test windows.

    Xte_sc and y_te MUST be sorted in strict temporal order so that
    predictions are always made before the corresponding label is seen.

    Steps per window t:
      1. Predict with current model (before update) → no label leakage.
      2. Add (x_t, y_t) to rolling replay buffer (deque, O(1) appends).
      3. If buffer contains both classes, take ONLINE_STEPS gradient steps
         on the full replay buffer with class-weighted focal loss.
         Using the full buffer (rather than hard balanced subsampling) avoids
         discarding valid majority-class examples; the pos_weight compensates
         for imbalance instead.

    The backbone is set to train() so Dropout is active during online updates
    (regularization against overfitting on the tiny replay buffer).
    BatchNorm layers are explicitly kept in eval() to preserve the running
    statistics learned during population training — only head gradients flow.

    An ExponentialLR scheduler decays the online learning rate across windows
    to reduce oscillation once the model has adapted.

    Population model scores are pre-computed in a single batched forward pass
    before the loop begins (model_pop never changes during online eval).

    model_pop must already be in eval() mode when passed in.

    Returns
    -------
    pop_scores    : (N,) predictions from the frozen population model
    online_scores : (N,) prequential predictions from the online model
    """
    model_online = copy.deepcopy(model_pop)

    # Only the head is updated online — use exact name prefix match to avoid
    # accidentally matching layers whose names happen to contain "head".
    for name, param in model_online.named_parameters():
        param.requires_grad = any(
            name == layer or name.startswith(layer + ".")
            for layer in ONLINE_LAYERS
        )

    online_params = [p for p in model_online.parameters() if p.requires_grad]
    opt = torch.optim.Adam(online_params, lr=ONLINE_LR, weight_decay=WEIGHT_DECAY)
    # Decay LR by ~0.5% per window to reduce oscillation as the model adapts.
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.995)

    # ── Pre-compute frozen population scores in one batch ─────────────────
    Xte_t = torch.as_tensor(Xte_sc, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pop_scores = torch.sigmoid(model_pop(Xte_t)).cpu().numpy()

    # Preallocated circular buffer — avoids repeated np.array(deque) copies
    # on every window.  Once full, buf_X_np[:REPLAY_BUFFER_SIZE] always
    # contains all current elements (order irrelevant for SGD).
    buf_X_np = np.zeros((REPLAY_BUFFER_SIZE, *Xte_sc.shape[1:]), dtype=np.float32)
    buf_y_np = np.zeros(REPLAY_BUFFER_SIZE, dtype=np.float32)
    buf_head = 0   # next write position (wraps around)
    buf_size = 0   # number of valid entries (≤ REPLAY_BUFFER_SIZE)

    online_scores = []

    for t in range(len(Xte_sc)):
        x_t = Xte_t[t:t + 1]

        # ── 1. Predict (before any update on this window) ─────────────────
        model_online.eval()
        with torch.no_grad():
            online_scores.append(torch.sigmoid(model_online(x_t)).item())

        # ── 2. Update replay buffer ───────────────────────────────────────
        buf_X_np[buf_head] = Xte_sc[t]
        buf_y_np[buf_head] = y_te[t]
        buf_head = (buf_head + 1) % REPLAY_BUFFER_SIZE
        buf_size = min(buf_size + 1, REPLAY_BUFFER_SIZE)

        buf_y_arr = buf_y_np[:buf_size]

        # ── 3. Gradient steps on full replay buffer ───────────────────────
        # Scheduler advances every window so the "~0.5% per window" decay
        # matches window count regardless of whether both classes are present.
        if len(np.unique(buf_y_arr)) >= 2:
            # Class-weighted focal loss over the full buffer; no hard subsampling.
            n_pos = int(buf_y_arr.sum())
            n_neg = buf_size - n_pos
            pos_weight = n_neg / max(n_pos, 1)

            # Build tensors once per window (not once per gradient step).
            buf_X_t = torch.as_tensor(buf_X_np[:buf_size], dtype=torch.float32).to(DEVICE)
            buf_y_t = torch.as_tensor(buf_y_arr,           dtype=torch.float32).to(DEVICE)

            # Enable train() so Dropout regularizes the small-batch updates.
            # Explicitly freeze BatchNorm layers to preserve running statistics.
            model_online.train()
            for m in model_online.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()

            for _ in range(ONLINE_STEPS):
                logits = model_online(buf_X_t)
                loss   = focal_loss(logits, buf_y_t, pos_weight=pos_weight)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online_params, max_norm=GRAD_CLIP_NORM)
                opt.step()

        scheduler.step()

    return pop_scores, np.array(online_scores)

# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(Path(__file__).parent / "relab+unibo_dataset.csv")
    df = mark_event_onsets(df)
    df = normalize_signals(df)

    X_raw_tr, y_tr_all, pid_tr_all, _, _                     = build_windows(df, fine_stride=True)
    X_raw_te, y_te_all, pid_te_all, routes_te_all, ts_te_all = build_windows(df, fine_stride=False)

    n_raw   = len(SIGNAL_COLS)
    n_feats = n_raw * 5

    print(f"\n{'='*65}")
    print("ONLINE PERSONALISATION — prequential predict-then-update")
    print(f"{'='*65}")
    print(f"Signals        : {SIGNAL_COLS}")
    print(f"GAP / HORIZON  : {GAP}s / {HORIZON}s  →  predicts errors in [{GAP}, {GAP+HORIZON}]s")
    print(f"Online LR      : {ONLINE_LR}  |  Steps/window : {ONLINE_STEPS}")
    print(f"Replay buffer  : {REPLAY_BUFFER_SIZE} windows  |  Layers : {ONLINE_LAYERS}")
    print(f"Evaluation     : prequential AUC (predict before update — no leakage)")
    print(f"{'='*65}\n")

    drivers = [d for d in np.unique(pid_te_all)
               if y_te_all[pid_te_all == d].sum() >= MIN_POSITIVES]

    hdr = (f"{'Driver':<10} | {'N_win':>5} {'PosR%':>6} | "
           f"{'PopAUC':>7} {'OnlAUC':>7} {'Gain':>7}")
    print(hdr)
    print("-" * len(hdr))

    per_driver_results = []

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

        # ── Sort test windows in strict temporal order (prequential guarantee) ─
        # np.lexsort: last key = primary sort.  Sort by route first, then by
        # timestamp within each route — safe when Timestamps are not globally
        # unique across routes (e.g. reset to 0 at the start of each route).
        order     = np.lexsort((ts_te, routes_te))
        X_te      = X_te[order]
        y_te      = y_te[order]
        routes_te = routes_te[order]

        # ── Validation split — stratified by driver positive-rate ─────────
        # Ensures the val fold contains at least one driver with positives,
        # reducing degenerate single-class folds that would force a SKIP.
        seed_d   = int(hashlib.md5(str(d).encode()).hexdigest(), 16) & 0xFFFFFFFF
        fold_rng = np.random.default_rng(SEED ^ seed_d)

        train_drivers = np.unique(pid_tr)
        has_pos = np.array([y_tr[pid_tr == p].sum() > 0 for p in train_drivers])
        pos_d   = train_drivers[has_pos]
        neg_d   = train_drivers[~has_pos]

        def _val_sample(arr, frac, rng):
            n = min(max(1, int(frac * len(arr))), len(arr)) if len(arr) > 0 else 0
            return rng.choice(arr, n, replace=False) if n > 0 else np.array([], dtype=arr.dtype)

        val_ids = np.concatenate([
            _val_sample(pos_d, 0.20, fold_rng),
            _val_sample(neg_d, 0.20, fold_rng),
        ])
        vmask = np.isin(pid_tr, val_ids)

        if len(np.unique(y_tr[vmask])) < 2:
            print(f"{d:<10} | SKIP — val fold single-class")
            continue

        # ── Features & scaling ────────────────────────────────────────────
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

        # ── Train population model (identical to main file) ───────────────
        model     = TCN_Attention_Net(n_feats).to(DEVICE)
        opt       = torch.optim.Adam(model.parameters(), lr=LR,
                                     weight_decay=WEIGHT_DECAY)
        # CosineAnnealing for population training (smooth warm-up + decay over
        # fixed epochs); ExponentialLR in online_evaluate_driver for gradual
        # stabilisation across an unknown number of incoming windows.
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

        # ── Online prequential evaluation ─────────────────────────────────
        pop_scores, onl_scores = online_evaluate_driver(model, Xte_sc, y_te)

        auc_pop = safe_auc(y_te, pop_scores)
        auc_onl = safe_auc(y_te, onl_scores)

        if auc_pop is None or auc_onl is None:
            print(f"{d:<10} | SKIP — degenerate eval labels")
            continue

        gain  = auc_onl - auc_pop
        g_str = f"{gain:+.4f}"
        pos_r = y_te.mean() * 100

        print(f"{d:<10} | {len(y_te):>5} {pos_r:>5.1f}% | "
              f"{auc_pop:>7.4f} {auc_onl:>7.4f} {g_str:>7}")

        per_driver_results.append({
            "driver":   str(d),
            "n_win":    int(len(y_te)),
            "pos_rate": float(pos_r),
            "auc_pop":  float(auc_pop),
            "auc_onl":  float(auc_onl),
            "gain":     float(gain),
        })

    # ── Pooled results ────────────────────────────────────────────────────────
    if not per_driver_results:
        print("No drivers with valid results.")
        return

    print("-" * len(hdr))

    # Per-driver mean AUC — consistent with the driver-level bootstrap CI.
    # (Pooled AUC over concatenated windows is dominated by high-window drivers
    # and estimates a different quantity than the bootstrap resampling.)
    pop_auc_list = [r["auc_pop"] for r in per_driver_results]
    onl_auc_list = [r["auc_onl"] for r in per_driver_results]
    auc_pop_mean = float(np.mean(pop_auc_list))
    auc_onl_mean = float(np.mean(onl_auc_list))

    lo_pop, hi_pop = bootstrap_auc_ci(pop_auc_list)
    lo_onl, hi_onl = bootstrap_auc_ci(onl_auc_list)

    print(f"\nMean AUC — Population : {auc_pop_mean:.4f}  "
          f"[{lo_pop:.3f}–{hi_pop:.3f}]")
    print(f"Mean AUC — Online     : {auc_onl_mean:.4f}  "
          f"[{lo_onl:.3f}–{hi_onl:.3f}]")
    print(f"Net gain              : {auc_onl_mean - auc_pop_mean:+.4f}")
    print(f"\nDrivers evaluated : {len(per_driver_results)}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "mean_driver": {
            "auc_population": auc_pop_mean,
            "auc_online":     auc_onl_mean,
            "net_gain":       auc_onl_mean - auc_pop_mean,
            "ci_population":  [float(lo_pop), float(hi_pop)],
            "ci_online":      [float(lo_onl), float(hi_onl)],
            "n_drivers":      len(per_driver_results),
        },
        "per_driver": per_driver_results,
        "config": {
            "online_lr":          ONLINE_LR,
            "online_steps":       ONLINE_STEPS,
            "replay_buffer_size": REPLAY_BUFFER_SIZE,
            "online_layers":      ONLINE_LAYERS,
        },
    }
    out_path = OUT_DIR / "tcn_online_personalisation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
