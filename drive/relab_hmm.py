#!/usr/bin/env python3
"""
HMM-based arousal state encoding for driving-error prediction.
Compares against XGBoost baseline (feature set A from relab_ml.py).

Approach
────────
For each LOPO fold (test = one participant):
  1. Fit GaussianHMM(n_states) on arousal sequences of *training* participants.
  2. Decode every session (train + test) via Viterbi → per-timestep state.
  3. For each row i, roll a lookback window over the decoded states:
       state_frac_k = fraction of timesteps in state k within [t-25s, t)
  4. Classify with XGBoost:
       HMM features : [state_frac_0..K-1, hr_mean, ar_slope, dist_frac, is_distracted]
       Baseline     : [ar_mean, hr_mean, ar_slope, dist_frac, is_distracted]

States are reordered by mean arousal (state_0 = lowest arousal).
Metrics: AUROC, Average Precision, Brier score.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from pathlib import Path
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA       = Path(__file__).parent / "relab+unibo_dataset.csv"
LOOKBACK_S = 25
N_STATES   = 3

ERR_COLS = [
    "Collision", "Red_light_violation", "center_line_crossing",
    "panic_braking", "panic_braking_with_stop", "sharp_turn",
]

XGB_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric="logloss",
    random_state=42, verbosity=0, n_jobs=-1,
)

STATE_COLS = [f"state_frac_{k}" for k in range(N_STATES)]
HMM_FEATS  = STATE_COLS + ["hr_mean", "ar_slope", "dist_frac", "is_distracted"]
BASE_FEATS = ["ar_mean", "hr_mean", "ar_slope", "dist_frac", "is_distracted"]


# ── data loading ──────────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.sort_values(["id", "route", "Timestamp"]).reset_index(drop=True)
    df["has_error"]     = (df[ERR_COLS] > 0).any(axis=1).astype(int)
    df["is_distracted"] = (df["safe_driving"] < 0.5).astype(int)
    return df


# ── per-session feature extraction (vectorized) ───────────────────────────────

def extract_session_features(grp: pd.DataFrame, lb: pd.Timedelta) -> pd.DataFrame:
    """
    For every row i in grp, compute lookback features from [t-lb, t).
    Returns DataFrame with one row per valid timestep.
    """
    grp = grp.sort_values("Timestamp").reset_index(drop=True)
    ts  = grp["Timestamp"]
    records = []
    for i in range(len(grp)):
        t0   = ts.iloc[i]
        mask = (ts >= t0 - lb) & (ts < t0)
        win  = grp[mask]
        if len(win) < 2:
            continue
        ar_v = win["arousal"].dropna().values
        hr_v = win["hr"].dropna().values
        if len(ar_v) < 2:
            continue
        records.append({
            "orig_idx":      i,
            "has_error":     int(grp["has_error"].iloc[i]),
            "is_distracted": int(grp["is_distracted"].iloc[i]),
            "ar_mean":       float(np.mean(ar_v)),
            "ar_slope":      float(np.polyfit(range(len(ar_v)), ar_v, 1)[0]),
            "hr_mean":       float(np.mean(hr_v)) if len(hr_v) >= 2 else np.nan,
            "dist_frac":     float(win["is_distracted"].mean()),
        })
    return pd.DataFrame(records)


def extract_windows(df: pd.DataFrame) -> pd.DataFrame:
    lb = pd.Timedelta(seconds=LOOKBACK_S)
    parts = []
    for (pid, route), grp in df.groupby(["id", "route"]):
        feat = extract_session_features(grp, lb)
        feat["id"]    = pid
        feat["route"] = route
        parts.append(feat)
    feat = pd.concat(parts, ignore_index=True)
    feat["hr_mean"] = feat["hr_mean"].fillna(feat["hr_mean"].median())
    return feat


# ── HMM helpers ───────────────────────────────────────────────────────────────

def _fit_hmm(df: pd.DataFrame, pids) -> GaussianHMM:
    """Fit GaussianHMM on arousal sequences of given participants."""
    seqs, lengths = [], []
    for pid in pids:
        for _, grp in df[df["id"] == pid].groupby("route"):
            ar = grp["arousal"].ffill().fillna(0).values.reshape(-1, 1)
            if len(ar) >= 5:
                seqs.append(ar)
                lengths.append(len(ar))
    X = np.vstack(seqs) if seqs else np.zeros((N_STATES * 2, 1))
    lengths = lengths if seqs else [N_STATES * 2]
    model = GaussianHMM(
        n_components=N_STATES, covariance_type="diag",
        n_iter=100, random_state=42, verbose=False,
    )
    model.fit(X, lengths)
    return model


def _state_perm(model: GaussianHMM) -> np.ndarray:
    """argsort by mean arousal → perm[0] = index of lowest-arousal state."""
    return np.argsort(model.means_[:, 0])


def _decode_session(model: GaussianHMM, ar: np.ndarray, inv_perm: np.ndarray) -> np.ndarray:
    """Viterbi decode; remap raw state indices via inv_perm."""
    _, raw = model.decode(ar.reshape(-1, 1), algorithm="viterbi")
    return inv_perm[raw]


def _hmm_window_fracs(grp_df: pd.DataFrame, model: GaussianHMM,
                      inv_perm: np.ndarray, feat_rows: pd.DataFrame,
                      lb: pd.Timedelta) -> pd.DataFrame:
    """
    Vectorized: decode full session, then for each valid window compute
    state occupancy fractions.
    Returns feat_rows augmented with state_frac_0..K-1 columns.
    """
    grp_df = grp_df.sort_values("Timestamp").reset_index(drop=True)
    ar_full = grp_df["arousal"].ffill().fillna(0).values
    states  = _decode_session(model, ar_full, inv_perm)   # shape (n,)
    ts      = grp_df["Timestamp"].values  # numpy datetime64

    out = feat_rows.copy()
    fracs = np.zeros((len(feat_rows), N_STATES))

    lb_ns = int(lb.total_seconds() * 1e9)   # nanoseconds

    for j, (_, row) in enumerate(feat_rows.iterrows()):
        i  = int(row["orig_idx"])
        t0 = ts[i]
        mask = (ts >= t0 - np.timedelta64(lb_ns, "ns")) & (ts < t0)
        win_states = states[mask]
        if len(win_states) > 0:
            for k in range(N_STATES):
                fracs[j, k] = (win_states == k).mean()

    for k in range(N_STATES):
        out[f"state_frac_{k}"] = fracs[:, k]
    return out


# ── add HMM features to a feature DataFrame ──────────────────────────────────

def add_hmm_features(df_raw: pd.DataFrame, feat: pd.DataFrame,
                     pids, model: GaussianHMM, inv_perm: np.ndarray) -> pd.DataFrame:
    lb   = pd.Timedelta(seconds=LOOKBACK_S)
    parts = []
    for pid in pids:
        for route, grp in df_raw[df_raw["id"] == pid].groupby("route"):
            sub = feat[(feat["id"] == pid) & (feat["route"] == route)]
            if len(sub) == 0:
                continue
            augmented = _hmm_window_fracs(grp, model, inv_perm, sub, lb)
            parts.append(augmented)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ── LOPO-CV ──────────────────────────────────────────────────────────────────

def lopo_cv(df_raw: pd.DataFrame, feat: pd.DataFrame) -> dict:
    participants = feat["id"].unique()
    res = {
        "Baseline A": {"aucs": [], "aps": [], "briers": []},
        "HMM+XGB":    {"aucs": [], "aps": [], "briers": []},
    }

    for fold_i, pid in enumerate(participants, 1):
        train_pids = [p for p in participants if p != pid]
        test_feat  = feat[feat["id"] == pid]
        train_feat = feat[feat["id"] != pid]

        if test_feat["has_error"].nunique() < 2:
            continue

        y_tr = train_feat["has_error"].values
        y_te = test_feat["has_error"].values
        pos_rate  = y_tr.mean()
        scale_pos = (1 - pos_rate) / max(pos_rate, 1e-6)

        # ── baseline ─────────────────────────────────────────────────────────
        sc = StandardScaler()
        X_tr = sc.fit_transform(train_feat[BASE_FEATS].values)
        X_te = sc.transform(test_feat[BASE_FEATS].values)
        clf = XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos)
        clf.fit(X_tr, y_tr)
        p_b = clf.predict_proba(X_te)[:, 1]
        res["Baseline A"]["aucs"].append(roc_auc_score(y_te, p_b))
        res["Baseline A"]["aps"].append(average_precision_score(y_te, p_b))
        res["Baseline A"]["briers"].append(brier_score_loss(y_te, p_b))

        # ── HMM ──────────────────────────────────────────────────────────────
        try:
            model    = _fit_hmm(df_raw, train_pids)
            perm     = _state_perm(model)
            inv_perm = np.argsort(perm)

            train_h = add_hmm_features(df_raw, train_feat, train_pids, model, inv_perm)
            test_h  = add_hmm_features(df_raw, test_feat,  [pid],       model, inv_perm)

            if len(train_h) == 0 or len(test_h) == 0:
                continue
            if test_h["has_error"].nunique() < 2:
                continue

            y_tr_h = train_h["has_error"].values
            y_te_h = test_h["has_error"].values
            pos_h  = y_tr_h.mean()
            sp_h   = (1 - pos_h) / max(pos_h, 1e-6)

            sc_h  = StandardScaler()
            X_tr_h = sc_h.fit_transform(train_h[HMM_FEATS].values)
            X_te_h = sc_h.transform(test_h[HMM_FEATS].values)
            clf_h  = XGBClassifier(**XGB_PARAMS, scale_pos_weight=sp_h)
            clf_h.fit(X_tr_h, y_tr_h)
            p_h = clf_h.predict_proba(X_te_h)[:, 1]

            res["HMM+XGB"]["aucs"].append(roc_auc_score(y_te_h, p_h))
            res["HMM+XGB"]["aps"].append(average_precision_score(y_te_h, p_h))
            res["HMM+XGB"]["briers"].append(brier_score_loss(y_te_h, p_h))
        except Exception as e:
            print(f"  fold {fold_i} HMM error: {e}")

        if fold_i % 10 == 0:
            print(f"  fold {fold_i}/{len(participants)} done")

    return res


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    print(f"Loading data (lookback={LOOKBACK_S}s, HMM states={N_STATES})…")
    df_raw = load_raw()
    feat   = extract_windows(df_raw)
    print(f"  {len(feat)} windows, {feat['id'].nunique()} participants, "
          f"error_rate={feat['has_error'].mean()*100:.1f}%\n")

    print("Running LOPO-CV …")
    res = lopo_cv(df_raw, feat)

    print(f"\n{'Method':<15}  {'AUROC':>14}  {'AP':>14}  {'Brier':>10}  folds")
    print("-" * 65)
    for name, r in res.items():
        if not r["aucs"]:
            print(f"{name:<15}  — no folds")
            continue
        a, p, b = r["aucs"], r["aps"], r["briers"]
        print(f"{name:<15}  {np.mean(a):.3f} ± {np.std(a):.3f}  "
              f"{np.mean(p):.3f} ± {np.std(p):.3f}  "
              f"{np.mean(b):.3f} ± {np.std(b):.3f}  {len(a)}")

    if res["HMM+XGB"]["aucs"] and res["Baseline A"]["aucs"]:
        delta = np.mean(res["HMM+XGB"]["aucs"]) - np.mean(res["Baseline A"]["aucs"])
        print(f"\nΔAUROC (HMM − Baseline) = {delta:+.4f}")


if __name__ == "__main__":
    main()
