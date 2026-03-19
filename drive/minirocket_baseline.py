"""
minirocket_baseline.py

MiniRocket baseline for driving impairment detection.
Identical data pipeline to tcn_dual_branch.py — same LOPO-CV, same metrics.

Usage:
    source venv/bin/activate
    pip install sktime   # if not already installed
    python drive/minirocket_baseline.py

MiniRocket pipeline:
    raw signals (N, T, C)
    → MiniRocketMultivariate transform  (N, 9_996 features)
    → StandardScaler
    → RidgeClassifierCV  (standard MiniRocket recipe)

Outputs metrics side-by-side with the TCN results from MEMORY:
    G4 (DB-TCN best): AUROC 0.610, AP 0.712, F1 0.793, Brier 0.232, ECE 0.125
"""

import warnings
from itertools import combinations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    f1_score, roc_curve,
)


# ── PURE NUMPY ROCKET TRANSFORM ─────────────────────────────────────────────────
# Drop-in replacement for sktime MiniRocketMultivariate — no numba/llvmlite needed.
# Implements the MiniRocket fixed kernel set (84 kernels × dilations) with PPV features.

class MiniRocketNP:
    """Pure numpy MiniRocket for multivariate time series.

    Follows the MiniRocket paper (Dempster et al., 2021):
      - 84 base kernels: all C(9,3) ways to place three +2 weights in a length-9
        kernel (rest = -1), so each kernel sums to 0.
      - Random dilations geometrically spaced up to max_dilation.
      - Random channel subset per kernel (multivariate extension).
      - Feature: PPV (proportion of positive values after convolution).

    X expected shape: (N, C, T)  — channels-first, matching sktime convention.
    """

    _BASE_WEIGHTS = np.array(
        [np.array([-1.0 if i not in idx3 else 2.0 for i in range(9)])
         for idx3 in combinations(range(9), 3)],
        dtype=np.float32,
    )  # (84, 9)

    def __init__(self, num_kernels=2_000, random_state=42):
        self.num_kernels   = num_kernels
        self.random_state  = random_state
        self.params_       = None

    def _build_params(self, C, T):
        rng = np.random.default_rng(self.random_state)
        K   = 9
        max_dil = max(1, (T - 1) // (K - 1))
        n_dil   = max(1, int(np.log2(max_dil)) + 1)
        dilations = np.unique(
            np.round(np.geomspace(1, max_dil, n_dil)).astype(int)
        )
        params = []
        for w in self._BASE_WEIGHTS:
            for d in dilations:
                n_ch = int(rng.integers(1, C + 1))
                chs  = rng.choice(C, n_ch, replace=False)
                params.append((w, d, chs))

        rng.shuffle(params)
        self.params_ = params[:self.num_kernels]

    def _apply(self, X):
        N, C, T = X.shape
        K = 9
        feats = np.empty((N, len(self.params_)), dtype=np.float32)
        for fi, (w, d, chs) in enumerate(self.params_):
            L = T - (K - 1) * d
            if L <= 0:
                feats[:, fi] = 0.0
                continue
            # Stack dilated slices: shape (K, N, L)
            slices = np.stack(
                [X[:, chs, ki * d: ki * d + L].sum(axis=1) for ki in range(K)]
            )                         # (K, N, L)
            out = (w[:, None, None] * slices).sum(axis=0)  # (N, L)
            feats[:, fi] = (out > 0).mean(axis=1)           # PPV
        return feats

    def fit_transform(self, X):
        self._build_params(X.shape[1], X.shape[2])
        return self._apply(X)

    def transform(self, X):
        if self.params_ is None:
            raise RuntimeError("Call fit_transform first.")
        return self._apply(X)

warnings.filterwarnings("ignore")

# ── CONFIG (must match tcn_dual_branch.py) ─────────────────────────────────────
SEED = 42
np.random.seed(SEED)

SIGNAL_COLS = [
    "arousal", "hr",
    "steeringWheelAngle", "steeringTorq", "acceleration.y", "speed.x",
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

LOOKBACK_S     = 60
WINDOW_STEP    = 5
GAP, HORIZON   = 15, 5
EVENT_VICINITY = 10
N_BOOTSTRAP    = 2000
MIN_POSITIVES  = 1


# ── PREPROCESSING (identical to tcn_dual_branch.py) ────────────────────────────

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


def composite_risk_score(future_df):
    return sum(
        SEVERITY[col] * int((future_df[col] > 0).any())
        for col in SEVERITY
    )


def build_windows(df, fine_stride=True):
    windows, labels, pids = [], [], []
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
                    windows.append(sig); labels.append(int(score > 0)); pids.append(pid)
                idx += step
        else:
            idx = 0
            while idx + LOOKBACK_S + GAP + HORIZON <= n:
                sig = grp.iloc[idx: idx + LOOKBACK_S][SIGNAL_COLS].values.astype(np.float32)
                if not np.isnan(sig).any():
                    future = grp.iloc[idx + LOOKBACK_S + GAP: idx + LOOKBACK_S + GAP + HORIZON]
                    score  = composite_risk_score(future)
                    windows.append(sig); labels.append(int(score > 0)); pids.append(pid)
                idx += WINDOW_STEP

    return (np.array(windows, dtype=np.float32),
            np.array(labels,  dtype=np.float32),
            np.array(pids))


# ── METRICS ────────────────────────────────────────────────────────────────────

def ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    total, err = len(y_true), 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        err += mask.sum() * abs(acc - conf)
    return err / total


def youden_f1(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j_idx = np.argmax(tpr - fpr)
    t     = thr[j_idx]
    return f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)


def bootstrap_auc(y_true, y_prob, n=N_BOOTSTRAP, seed=SEED):
    rng  = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].sum() == 0 or y_true[idx].sum() == len(idx):
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return np.mean(aucs), lo, hi


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    def make_transform():
        return MiniRocketNP(num_kernels=2_000, random_state=SEED)
    print("Using MiniRocketNP (pure numpy, no numba)")

    # ── Data ──────────────────────────────────────────────────────────────────
    csv = Path(__file__).parent / "relab+unibo_dataset.csv"
    df  = pd.read_csv(csv)
    df  = mark_event_onsets(df)
    df  = normalize_signals(df)

    X_tr_raw, y_tr, pid_tr = build_windows(df, fine_stride=True)
    X_te_raw, y_te, pid_te = build_windows(df, fine_stride=False)

    # MiniRocket expects (N, C, T)
    X_tr_rocket = X_tr_raw.transpose(0, 2, 1)  # (N, 6, 60)
    X_te_rocket = X_te_raw.transpose(0, 2, 1)

    drivers = [d for d in np.unique(pid_te) if y_te[pid_te == d].sum() >= MIN_POSITIVES]

    print(f"\n{'='*60}")
    print(f"MiniRocket LOPO-CV  —  {len(drivers)} drivers")
    print(f"{'='*60}")
    hdr = f"{'Driver':<12} {'N':>5} {'PosR%':>6} {'AUC':>7} {'AP':>7} {'F1':>6} {'Brier':>7}"
    print(hdr); print("-" * len(hdr))

    pool_y, pool_prob = [], []

    for d in drivers:
        tr_mask = pid_tr != d
        te_mask = pid_te == d

        X_tr_d = X_tr_rocket[tr_mask]; y_tr_d = y_tr[tr_mask]
        X_te_d = X_te_rocket[te_mask]; y_te_d = y_te[te_mask]

        if len(np.unique(y_tr_d)) < 2 or len(np.unique(y_te_d)) < 2:
            print(f"{str(d):<12} SKIP — single class in train or test")
            continue

        # Fit transform on train only (anti-leakage)
        rocket = make_transform()
        Xtr_feat = rocket.fit_transform(X_tr_d)
        Xte_feat = rocket.transform(X_te_d)

        scaler   = StandardScaler(with_mean=False)
        Xtr_sc   = scaler.fit_transform(Xtr_feat)
        Xte_sc   = scaler.transform(Xte_feat)

        clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        clf.fit(Xtr_sc, y_tr_d)

        # RidgeClassifier has decision_function, not predict_proba
        # Convert to [0,1] via sigmoid
        scores = clf.decision_function(Xte_sc)
        prob   = 1 / (1 + np.exp(-scores))

        auc   = roc_auc_score(y_te_d, prob)
        ap    = average_precision_score(y_te_d, prob)
        f1    = youden_f1(y_te_d, prob)
        brier = brier_score_loss(y_te_d, prob)
        pos_r = y_te_d.mean() * 100

        print(f"{str(d):<12} {len(y_te_d):>5} {pos_r:>5.1f}% {auc:>7.4f} {ap:>7.4f} {f1:>6.4f} {brier:>7.4f}")

        pool_y.append(y_te_d)
        pool_prob.append(prob)

    # ── Pooled results ─────────────────────────────────────────────────────────
    pool_y    = np.concatenate(pool_y)
    pool_prob = np.concatenate(pool_prob)

    auc_mean, auc_lo, auc_hi = bootstrap_auc(pool_y, pool_prob)
    ap    = average_precision_score(pool_y, pool_prob)
    brier = brier_score_loss(pool_y, pool_prob)
    ece_v = ece(pool_y, pool_prob)
    f1    = youden_f1(pool_y, pool_prob)

    print(f"\n{'='*60}")
    print("POOLED RESULTS")
    print(f"{'='*60}")
    print(f"  AUC-ROC : {auc_mean:.4f}  95% CI [{auc_lo:.4f}–{auc_hi:.4f}]")
    print(f"  AUPRC   : {ap:.4f}")
    print(f"  F1      : {f1:.4f}  (@ Youden's threshold)")
    print(f"  Brier   : {brier:.4f}")
    print(f"  ECE     : {ece_v:.4f}")
    print(f"\n{'─'*60}")
    print("REFERENCE  (DB-TCN G4 from tcn_dual_branch.py)")
    print(f"  AUC-ROC : 0.6100  95% CI [0.5890–0.6310]")
    print(f"  AUPRC   : 0.7120")
    print(f"  F1      : 0.7930")
    print(f"  Brier   : 0.2320")
    print(f"  ECE     : 0.1250")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
