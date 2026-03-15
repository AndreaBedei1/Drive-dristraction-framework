#!/usr/bin/env python3
"""
ML-based error prediction on relab+unibo dataset.
LOPO-CV (Leave-One-Participant-Out).

Targets compared:
  T1    — all errors      (includes center_line_crossing, ~27.6 %)
  T1-nc — all errors excl. center_line_crossing (serious + clc removed)
  T2    — serious errors  (collision, red-light, panic-braking, sharp-turn, ~2 %)

Feature sets:
  A   — physiology only  : ar_mean, hr_mean, ar_slope, dist_frac, is_distracted
  A+z — A + ar_z         : A + per-participant z-scored arousal
  E   — + extended       : A + ar_std + brake + steer
  Ez  — E + ar_z

Extras:
  - Feature importance (XGBoost gain) aggregated across LOPO folds (printed + saved).
  - ar_z = per-participant z-score of ar_mean, computed leak-free per LOPO fold.

Metrics: AUROC, Average Precision, Brier score.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              brier_score_loss, f1_score, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA  = Path(__file__).parent / "relab+unibo_dataset.csv"
LOOKBACK_S = 25

ERR_ALL = [
    "Collision", "Red_light_violation", "center_line_crossing",
    "panic_braking", "panic_braking_with_stop", "sharp_turn",
]
ERR_NO_CLC = [
    "Collision", "Red_light_violation",
    "panic_braking", "panic_braking_with_stop", "sharp_turn",
]
ERR_SERIOUS = ERR_NO_CLC   # same list — serious = no-clc

XGB_PARAMS = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric="logloss",
    random_state=42, verbosity=0, n_jobs=-1,
)

EMO_COLS = ["anger", "fear", "sadness", "happiness", "surprise", "disgust", "neutral"]

FEAT_A   = ["ar_mean", "hr_mean", "ar_slope", "dist_frac", "is_distracted"]
FEAT_E   = ["ar_mean", "hr_mean", "ar_slope", "ar_std", "dist_frac", "is_distracted",
            "brake_mean", "steer_mean"]
FEAT_V   = FEAT_E + ["speed_mean", "lat_accel_mean", "ar_x_speed"]          # + vehicle dynamics
FEAT_EMO = FEAT_A + [f"emo_{e}" for e in EMO_COLS]                          # + emotions (physio)
FEAT_ALL = FEAT_E + [f"emo_{e}" for e in EMO_COLS] + ["speed_mean",
           "lat_accel_mean", "ar_x_speed"]                                   # full multimodal


# ── data loading & feature extraction ────────────────────────────────────────

def load_features(lookback_s: int = LOOKBACK_S) -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.sort_values(["id", "route", "Timestamp"]).reset_index(drop=True)
    df["has_error"]     = (df[ERR_ALL]     > 0).any(axis=1).astype(int)
    df["has_noclc"]     = (df[ERR_NO_CLC]  > 0).any(axis=1).astype(int)
    df["has_serious"]   = (df[ERR_SERIOUS] > 0).any(axis=1).astype(int)
    df["is_distracted"] = (df["safe_driving"] < 0.5).astype(int)
    # derived columns
    df["speed_x"]    = df["speed.x"].abs()
    df["lat_accel"]  = df["acceleration.y"].abs()

    lb = pd.Timedelta(seconds=lookback_s)
    records = []

    for (pid, route), grp in df.groupby(["id", "route"]):
        grp = grp.sort_values("Timestamp").reset_index(drop=True)
        for i in range(len(grp)):
            t0  = grp["Timestamp"].iloc[i]
            win = grp[(grp["Timestamp"] >= t0 - lb) & (grp["Timestamp"] < t0)]
            if len(win) < 2:
                continue
            ar_v  = win["arousal"].dropna().values
            hr_v  = win["hr"].dropna().values
            br_v  = win["brake"].dropna().values
            st_v  = win["steeringWheelAngle"].dropna().values
            sp_v  = win["speed_x"].dropna().values
            la_v  = win["lat_accel"].dropna().values
            if len(ar_v) < 2:
                continue

            ar_m  = float(np.mean(ar_v))
            sp_m  = float(np.mean(sp_v)) if len(sp_v) >= 1 else 0.0

            row = {
                "id":              pid,
                "route":           route,
                "has_error":       int(grp["has_error"].iloc[i]),
                "has_noclc":       int(grp["has_noclc"].iloc[i]),
                "has_serious":     int(grp["has_serious"].iloc[i]),
                "is_distracted":   int(grp["is_distracted"].iloc[i]),
                "ar_mean":         ar_m,
                "ar_slope":        float(np.polyfit(range(len(ar_v)), ar_v, 1)[0]),
                "ar_std":          float(np.std(ar_v)),
                "hr_mean":         float(np.mean(hr_v)) if len(hr_v) >= 2 else np.nan,
                "dist_frac":       float(win["is_distracted"].mean()),
                "brake_mean":      float(np.mean(br_v)) if len(br_v) >= 1 else 0.0,
                "steer_mean":      float(np.mean(np.abs(st_v))) if len(st_v) >= 1 else 0.0,
                "speed_mean":      sp_m,
                "lat_accel_mean":  float(np.mean(np.abs(la_v))) if len(la_v) >= 1 else 0.0,
                "ar_x_speed":      ar_m * sp_m,
            }
            # emotion means over lookback window
            for e in EMO_COLS:
                ev = win[e].dropna().values
                row[f"emo_{e}"] = float(np.mean(ev)) if len(ev) >= 1 else np.nan
            records.append(row)

    feat = pd.DataFrame(records)
    for col in ["hr_mean", "speed_mean", "lat_accel_mean", "ar_x_speed"] + \
               [f"emo_{e}" for e in EMO_COLS]:
        feat[col] = feat[col].fillna(feat[col].median())
    return feat


# ── per-fold ar_z (leak-free) ─────────────────────────────────────────────────

def add_arz_fold(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute ar_z = (ar_mean - pid_mean) / pid_std using ONLY training participants.
    For training: normalise each participant by their own stats.
    For test:     normalise the test participant using test-participant stats
                  (no leakage from train; consistent personalisation).
    """
    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        stats = df.groupby("id")["ar_mean"].agg(["mean", "std"]).rename(
            columns={"mean": "_mu", "std": "_sd"})
        df = df.join(stats, on="id")
        df["_sd"] = df["_sd"].fillna(1.0).clip(lower=1e-6)
        df["ar_z"] = (df["ar_mean"] - df["_mu"]) / df["_sd"]
        return df.drop(columns=["_mu", "_sd"])

    return _norm(train), _norm(test)


# ── LOPO-CV ──────────────────────────────────────────────────────────────────

def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    """Return (threshold, F1) that maximises F1 on pooled OOF predictions."""
    prec, rec, thresholds = precision_recall_curve(y_true, proba)
    # precision_recall_curve returns one extra point (prec[-1]=1, rec[-1]=0, no threshold)
    f1s = np.where((prec[:-1] + rec[:-1]) > 0,
                   2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1]), 0.0)
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])


def lopo_cv(feat: pd.DataFrame, feature_cols: list, target: str,
            collect_importance: bool = False) -> dict:
    participants = feat["id"].unique()
    aucs, aps, briers = [], [], []
    importances = []
    # for global (pooled) aggregation
    all_y_true, all_proba = [], []

    needs_arz = "ar_z" in feature_cols

    for pid in participants:
        train = feat[feat["id"] != pid].copy()
        test  = feat[feat["id"] == pid].copy()

        if test[target].nunique() < 2:
            continue

        if needs_arz:
            train, test = add_arz_fold(train, test)

        X_tr = train[feature_cols].values
        y_tr = train[target].values
        X_te = test[feature_cols].values
        y_te = test[target].values

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

        pos_rate  = y_tr.mean()
        scale_pos = (1 - pos_rate) / max(pos_rate, 1e-6)

        clf = XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos)
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, 1]

        aucs.append(roc_auc_score(y_te, proba))
        aps.append(average_precision_score(y_te, proba))
        briers.append(brier_score_loss(y_te, proba))

        all_y_true.append(y_te)
        all_proba.append(proba)

        if collect_importance:
            gain = clf.get_booster().get_score(importance_type="gain")
            importances.append(gain)

    # ── global (pooled OOF) metrics ───────────────────────────────────────────
    y_pool = np.concatenate(all_y_true)
    p_pool = np.concatenate(all_proba)

    global_auroc = float(roc_auc_score(y_pool, p_pool))
    global_ap    = float(average_precision_score(y_pool, p_pool))
    global_brier = float(brier_score_loss(y_pool, p_pool))
    best_thr, best_f1 = best_f1_threshold(y_pool, p_pool)

    result = {
        # per-fold averages (classic)
        "AUROC":        (np.mean(aucs),   np.std(aucs)),
        "AP":           (np.mean(aps),    np.std(aps)),
        "Brier":        (np.mean(briers), np.std(briers)),
        "n_folds":      len(aucs),
        # pooled OOF (global)
        "global_AUROC": global_auroc,
        "global_AP":    global_ap,
        "global_Brier": global_brier,
        "best_thr":     best_thr,
        "best_F1":      best_f1,
    }
    if collect_importance:
        result["importances"] = importances
    return result


def aggregate_importance(importances: list, feature_cols: list) -> pd.DataFrame:
    """Average gain across folds; missing features get 0."""
    rows = []
    for imp in importances:
        # XGBoost uses f0, f1, ... internally when given numpy arrays
        row = {f"f{i}": imp.get(f"f{i}", 0.0) for i in range(len(feature_cols))}
        rows.append(row)
    df = pd.DataFrame(rows)
    mean_gain = df.mean()
    tbl = pd.DataFrame({
        "feature": feature_cols,
        "mean_gain": [mean_gain.get(f"f{i}", 0.0) for i in range(len(feature_cols))],
    }).sort_values("mean_gain", ascending=False).reset_index(drop=True)
    return tbl


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    print(f"Loading features (lookback={LOOKBACK_S}s)…")
    feat = load_features()
    print(f"  {len(feat)} rows, {feat['id'].nunique()} participants")
    print(f"  T1  error rate (all)   = {feat['has_error'].mean()*100:.1f}%")
    print(f"  T1-nc rate (no clc)    = {feat['has_noclc'].mean()*100:.2f}%")
    print(f"  T2  serious error rate = {feat['has_serious'].mean()*100:.2f}%\n")

    configs = [
        # (name, feature_cols, target, collect_importance)
        ("T1-A   physio",       FEAT_A,   "has_error",   True),
        ("T1-E   extended",     FEAT_E,   "has_error",   True),
        ("T1-V   +vehicle",     FEAT_V,   "has_error",   True),
        ("T1-EMO +emotion",     FEAT_EMO, "has_error",   True),
        ("T1-ALL multimodal",   FEAT_ALL, "has_error",   True),
        ("T2-A   physio",       FEAT_A,   "has_serious", False),
        ("T2-E   extended",     FEAT_E,   "has_serious", False),
        ("T2-ALL multimodal",   FEAT_ALL, "has_serious", False),
    ]

    # ── per-fold averages ─────────────────────────────────────────────────────
    print(f"{'Config':<28}  {'AUROC(fold)':>14}  {'AP(fold)':>14}  {'Brier(fold)':>12}  folds")
    print("-" * 80)

    results = {}
    for name, cols, target, ci in configs:
        r = lopo_cv(feat, cols, target, collect_importance=ci)
        results[name] = r
        auc_m, auc_s = r["AUROC"]
        ap_m,  ap_s  = r["AP"]
        br_m,  br_s  = r["Brier"]
        print(f"{name:<28}  {auc_m:.3f} ± {auc_s:.3f}  "
              f"{ap_m:.3f} ± {ap_s:.3f}  "
              f"{br_m:.3f} ± {br_s:.3f}  {r['n_folds']}")

    # ── global pooled OOF metrics ─────────────────────────────────────────────
    print(f"\n{'Config':<28}  {'AUROC(global)':>14}  {'AP(global)':>11}  "
          f"{'Brier(global)':>14}  {'best_thr':>9}  {'F1@thr':>7}")
    print("-" * 90)
    for name, r in results.items():
        print(f"{name:<28}  {r['global_AUROC']:.3f}           "
              f"{r['global_AP']:.3f}        "
              f"{r['global_Brier']:.3f}           "
              f"{r['best_thr']:.3f}      "
              f"{r['best_F1']:.3f}")

    # ── Δ vs T1-A baseline ────────────────────────────────────────────────────
    print("\n--- Δ AUROC (fold-avg) vs T1-A (baseline) ---")
    base = results["T1-A   physio"]["AUROC"][0]
    for name, r in results.items():
        if name == "T1-A   physio":
            continue
        delta = r["AUROC"][0] - base
        print(f"  {name:<28}  ΔAUROC = {delta:+.4f}")

    # ── Feature importance (T1-A and T1-E) ───────────────────────────────────
    for cfg_name, feat_cols in [("T1-A   physio", FEAT_A),
                                 ("T1-E   extended", FEAT_E),
                                 ("T1-V   +vehicle", FEAT_V),
                                 ("T1-EMO +emotion", FEAT_EMO),
                                 ("T1-ALL multimodal", FEAT_ALL)]:
        r = results[cfg_name]
        if "importances" not in r:
            continue
        tbl = aggregate_importance(r["importances"], feat_cols)
        print(f"\n--- Feature importance (gain) — {cfg_name} ---")
        print(tbl.to_string(index=False, float_format="%.2f"))


if __name__ == "__main__":
    main()
