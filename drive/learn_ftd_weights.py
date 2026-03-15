#!/usr/bin/env python3
"""
Cross-dataset transfer learning personalisation.

Population model: MLP trained on ALL of relab+unibo_dataset.csv (43 participants).
Personalisation : linear probe (freeze MLP backbone, fit new logistic head)
                  on K labeled samples from a participant in original/.
Evaluation      : remaining samples of the same participant in original/.

Feature alignment (shared 9-dim space):
  ar_mean, ar_slope, ar_std, hr_mean, steer_mean, brake_mean,
  speed_mean, dist_frac, is_distracted

Both datasets use a 25 s lookback window.
K_LIST = [10, 20, 50, 100]
Participants in original/ with enough data: participant_01, participant_02.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

BASE_DIR   = Path(__file__).parent
RELAB_DATA = BASE_DIR / "relab+unibo_dataset.csv"
ORIG_DATA  = BASE_DIR / "data" / "original" / "Dataset Timeline_distraction.csv"

LOOKBACK_S = 25
K_LIST     = [10, 20, 50, 100]
TARGET     = "has_error"

FEAT_COLS = [
    "ar_mean", "ar_slope", "ar_std",
    "hr_mean", "steer_mean", "brake_mean",
    "speed_mean", "dist_frac", "is_distracted",
]

MLP_PARAMS = dict(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
)

XGB_PARAMS = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric="logloss",
    random_state=42, verbosity=0, n_jobs=-1,
)

ERR_ALL = [
    "Collision", "Red_light_violation", "center_line_crossing",
    "panic_braking", "panic_braking_with_stop", "sharp_turn",
]


# ── feature extraction: relab+unibo ──────────────────────────────────────────

def load_relab() -> pd.DataFrame:
    df = pd.read_csv(RELAB_DATA)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.sort_values(["id", "route", "Timestamp"]).reset_index(drop=True)
    df[TARGET]         = (df[ERR_ALL] > 0).any(axis=1).astype(int)
    df["is_distracted"] = (df["safe_driving"] < 0.5).astype(int)
    df["speed_x"]      = df["speed.x"].abs()

    lb = pd.Timedelta(seconds=LOOKBACK_S)
    records = []
    for (pid, route), grp in df.groupby(["id", "route"]):
        grp = grp.sort_values("Timestamp").reset_index(drop=True)
        for i in range(len(grp)):
            t0  = grp["Timestamp"].iloc[i]
            win = grp[(grp["Timestamp"] >= t0 - lb) & (grp["Timestamp"] < t0)]
            if len(win) < 2:
                continue
            ar_v = win["arousal"].dropna().values
            hr_v = win["hr"].dropna().values
            st_v = win["steeringWheelAngle"].dropna().values
            br_v = win["brake"].dropna().values
            sp_v = win["speed_x"].dropna().values
            if len(ar_v) < 2:
                continue
            records.append({
                TARGET:          int(grp[TARGET].iloc[i]),
                "is_distracted": int(grp["is_distracted"].iloc[i]),
                "ar_mean":       float(np.mean(ar_v)),
                "ar_slope":      float(np.polyfit(range(len(ar_v)), ar_v, 1)[0]),
                "ar_std":        float(np.std(ar_v)),
                "hr_mean":       float(np.mean(hr_v)) if len(hr_v) >= 2 else np.nan,
                "steer_mean":    float(np.mean(np.abs(st_v))) if len(st_v) >= 1 else 0.0,
                "brake_mean":    float(np.mean(br_v)) if len(br_v) >= 1 else 0.0,
                "speed_mean":    float(np.mean(sp_v)) if len(sp_v) >= 1 else 0.0,
                "dist_frac":     float(win["is_distracted"].mean()),
            })

    feat = pd.DataFrame(records)
    feat["hr_mean"] = feat["hr_mean"].fillna(feat["hr_mean"].median())
    return feat


# ── feature extraction: original/ ────────────────────────────────────────────

def load_original() -> pd.DataFrame:
    df = pd.read_csv(ORIG_DATA)
    df = df[df["user_id"].str.startswith("participant", na=False)].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed",
                                     utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["user_id", "run_id", "timestamp"]).reset_index(drop=True)
    df[TARGET]          = df["error_types"].notna().astype(int)
    df["is_distracted"] = df["model_pred"].notna().astype(int)

    lb = pd.Timedelta(seconds=LOOKBACK_S)
    records = []
    for (pid, run), grp in df.groupby(["user_id", "run_id"]):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        for i in range(len(grp)):
            t0  = grp["timestamp"].iloc[i]
            win = grp[(grp["timestamp"] >= t0 - lb) & (grp["timestamp"] < t0)]
            if len(win) < 2:
                continue
            ar_v = win["arousal"].dropna().values
            hr_v = win["hr_bpm"].dropna().values
            st_v = win["steer_angle_deg"].dropna().values
            br_v = win["brake"].dropna().values
            sp_v = win["speed_kmh"].dropna().values
            if len(ar_v) < 2:
                continue
            records.append({
                "id":            pid,
                "run":           run,
                TARGET:          int(grp[TARGET].iloc[i]),
                "is_distracted": int(grp["is_distracted"].iloc[i]),
                "ar_mean":       float(np.mean(ar_v)),
                "ar_slope":      float(np.polyfit(range(len(ar_v)), ar_v, 1)[0]),
                "ar_std":        float(np.std(ar_v)),
                "hr_mean":       float(np.mean(hr_v)) if len(hr_v) >= 2 else np.nan,
                "steer_mean":    float(np.mean(np.abs(st_v))) if len(st_v) >= 1 else 0.0,
                "brake_mean":    float(np.mean(br_v)) if len(br_v) >= 1 else 0.0,
                "speed_mean":    float(np.mean(sp_v)) if len(sp_v) >= 1 else 0.0,
                "dist_frac":     float(win["is_distracted"].mean()),
            })

    feat = pd.DataFrame(records)
    feat["hr_mean"] = feat["hr_mean"].fillna(feat["hr_mean"].median())
    return feat


# ── MLP hidden activations ────────────────────────────────────────────────────

def get_hidden(mlp: MLPClassifier, X: np.ndarray) -> np.ndarray:
    h = X
    for W, b in zip(mlp.coefs_[:-1], mlp.intercepts_[:-1]):
        h = h @ W + b
        h = np.maximum(0, h)
    return h


# ── cross-dataset evaluation ─────────────────────────────────────────────────

def evaluate_transfer(relab: pd.DataFrame, orig: pd.DataFrame) -> None:
    """
    1. Train population MLP + XGBoost on all of relab (no LOPO — full population).
    2. For each participant in orig:
         a. population model score on all their data
         b. linear probe using K samples → eval on remaining
    """
    print("Training population models on relab+unibo (full dataset)…")
    X_pop = relab[FEAT_COLS].values
    y_pop = relab[TARGET].values

    sc = StandardScaler()
    X_pop_s = sc.fit_transform(X_pop)

    pos_rate  = y_pop.mean()
    scale_pos = (1 - pos_rate) / max(pos_rate, 1e-6)

    # XGBoost population
    xgb = XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos)
    xgb.fit(X_pop_s, y_pop)
    print(f"  XGBoost trained — relab error rate: {pos_rate*100:.1f}%")

    # MLP population
    pos_idx = np.where(y_pop == 1)[0]
    neg_idx = np.where(y_pop == 0)[0]
    ratio   = max(1, len(neg_idx) // len(pos_idx))
    idx_bal = np.concatenate([neg_idx, np.tile(pos_idx, ratio)])
    mlp = MLPClassifier(**MLP_PARAMS)
    mlp.fit(X_pop_s[idx_bal], y_pop[idx_bal])
    print(f"  MLP trained — hidden layers: {mlp.hidden_layer_sizes}")

    # evaluate on each participant in original/
    print(f"\n{'='*76}")
    print(f"  Cross-dataset evaluation — original/ participants")
    print(f"{'='*76}")

    for pid, pdata in orig.groupby("id"):
        X_te_raw = pdata[FEAT_COLS].values
        y_te     = pdata[TARGET].values

        if y_te.sum() < 3 or len(np.unique(y_te)) < 2:
            print(f"\n  {pid}: skip (only {y_te.sum()} errors)")
            continue

        X_te_s = sc.transform(X_te_raw)

        # population scores (no personalisation)
        p_xgb = xgb.predict_proba(X_te_s)[:, 1]
        p_mlp = mlp.predict_proba(X_te_s)[:, 1]

        auc_xgb = roc_auc_score(y_te, p_xgb)
        auc_mlp = roc_auc_score(y_te, p_mlp)
        ap_xgb  = average_precision_score(y_te, p_xgb)
        ap_mlp  = average_precision_score(y_te, p_mlp)

        print(f"\n  {pid}  ({len(pdata)} windows, {y_te.sum()} errors, "
              f"{y_te.mean()*100:.1f}% error rate)")
        print(f"  {'Model':<22}  {'AUROC':>8}  {'AP':>8}  folds/K")
        print(f"  {'-'*46}")
        print(f"  {'XGB-pop':<22}  {auc_xgb:.3f}     {ap_xgb:.3f}    —")
        print(f"  {'MLP-pop':<22}  {auc_mlp:.3f}     {ap_mlp:.3f}    —")

        # hidden repr for entire test set
        h_te = get_hidden(mlp, X_te_s)

        for k in K_LIST:
            if len(pdata) < k + 10:
                continue
            rng = np.random.default_rng(42)
            cal_idx  = rng.choice(len(pdata), size=k, replace=False)
            eval_idx = np.setdiff1d(np.arange(len(pdata)), cal_idx)

            y_cal  = y_te[cal_idx]
            y_eval = y_te[eval_idx]
            if len(np.unique(y_cal)) < 2 or len(np.unique(y_eval)) < 2:
                continue

            probe = LogisticRegression(max_iter=500, random_state=42,
                                       class_weight="balanced")
            try:
                probe.fit(h_te[cal_idx], y_cal)
                p_probe = probe.predict_proba(h_te[eval_idx])[:, 1]
            except Exception:
                continue

            auc_pr = roc_auc_score(y_eval, p_probe)
            ap_pr  = average_precision_score(y_eval, p_probe)
            delta  = auc_pr - auc_mlp
            print(f"  {'MLP-probe@'+str(k):<22}  {auc_pr:.3f}     {ap_pr:.3f}    "
                  f"K={k}  Δ={delta:+.3f} vs MLP-pop")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    print(f"Loading relab+unibo dataset (lookback={LOOKBACK_S}s)…")
    relab = load_relab()
    print(f"  {len(relab)} windows, {relab[TARGET].mean()*100:.1f}% error rate")

    print(f"\nLoading original/ dataset…")
    orig = load_original()
    print(f"  {len(orig)} windows, {orig['id'].nunique()} participants")
    for pid, g in orig.groupby("id"):
        print(f"    {pid}: {len(g)} windows, {g[TARGET].sum()} errors "
              f"({g[TARGET].mean()*100:.1f}%)")

    print()
    evaluate_transfer(relab, orig)


if __name__ == "__main__":
    main()
