#!/usr/bin/env python3
"""
MLP transfer-learning personalisation on the P02 (original/) dataset.

Dataset: drive/data/original/Dataset Timeline_distraction.csv
         9 participants (1 corrupted row filtered), ~6223 rows, ~1 Hz sampling.

Approach:
  - Feature extraction: 25 s lookback window → ar_mean, ar_slope, ar_std,
    hr_mean, steer_mean, brake_mean, speed_mean, dist_frac, is_distracted
  - LOPO-CV (8 folds): population XGBoost vs population MLP vs MLP-probe@K
  - K_LIST = [10, 20, 50]
  - Focus participant: participant_02 (7 runs, 1935 rows, 67 errors)

Metrics: AUROC (fold-avg ± std), AUROC (pooled OOF), AP (pooled).
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

DATA = Path(__file__).parent / "data" / "original" / "Dataset Timeline_distraction.csv"
LOOKBACK_S = 25
K_LIST = [10, 20, 50]
TARGET = "has_error"

MLP_PARAMS = dict(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=400,
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

FEAT_COLS = [
    "ar_mean", "ar_slope", "ar_std",
    "hr_mean", "steer_mean", "brake_mean",
    "speed_mean", "dist_frac", "is_distracted",
]


# ── data loading & feature extraction ────────────────────────────────────────

def load_features() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    # drop corrupted merge-conflict row
    df = df[df["user_id"].str.startswith("participant", na=False)].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True,
                                     errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["user_id", "run_id", "timestamp"]).reset_index(drop=True)

    df[TARGET] = df["error_types"].notna().astype(int)
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


# ── LOPO-CV ───────────────────────────────────────────────────────────────────

def _record(res, name, y, p):
    if len(np.unique(y)) < 2:
        return
    res[name]["aucs"].append(roc_auc_score(y, p))
    res[name]["y_true"].append(y)
    res[name]["proba"].append(p)


def lopo_transfer(feat: pd.DataFrame) -> dict:
    participants = feat["id"].unique()
    names = ["XGB-pop", "MLP-pop"] + [f"MLP-probe@{k}" for k in K_LIST]
    res = {n: {"aucs": [], "y_true": [], "proba": [], "folds_pid": []}
           for n in names}

    for fold_i, pid in enumerate(participants, 1):
        train = feat[feat["id"] != pid].copy()
        test  = feat[feat["id"] == pid].copy()

        if test[TARGET].nunique() < 2:
            print(f"  fold {fold_i} ({pid}): skip — no both classes in test")
            continue

        X_tr = train[FEAT_COLS].values
        y_tr = train[TARGET].values
        X_te = test[FEAT_COLS].values
        y_te = test[TARGET].values

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s  = sc.transform(X_te)

        pos_rate  = y_tr.mean()
        scale_pos = (1 - pos_rate) / max(pos_rate, 1e-6)

        # XGBoost population
        xgb = XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos)
        xgb.fit(X_tr_s, y_tr)
        _record(res, "XGB-pop", y_te, xgb.predict_proba(X_te_s)[:, 1])
        res["XGB-pop"]["folds_pid"].append(pid)

        # MLP population (oversample positives)
        pos_idx = np.where(y_tr == 1)[0]
        neg_idx = np.where(y_tr == 0)[0]
        if len(pos_idx) == 0:
            continue
        ratio = max(1, len(neg_idx) // len(pos_idx))
        idx_bal = np.concatenate([neg_idx, np.tile(pos_idx, ratio)])
        mlp = MLPClassifier(**MLP_PARAMS)
        mlp.fit(X_tr_s[idx_bal], y_tr[idx_bal])
        p_mlp = mlp.predict_proba(X_te_s)[:, 1]
        _record(res, "MLP-pop", y_te, p_mlp)

        # linear probes
        h_te = get_hidden(mlp, X_te_s)
        for k in K_LIST:
            key = f"MLP-probe@{k}"
            if len(test) < k + 5:
                continue
            rng = np.random.default_rng(42)
            cal_idx  = rng.choice(len(test), size=k, replace=False)
            eval_idx = np.setdiff1d(np.arange(len(test)), cal_idx)
            if len(np.unique(y_te[cal_idx])) < 2:
                continue
            if len(np.unique(y_te[eval_idx])) < 2:
                continue
            probe = LogisticRegression(max_iter=500, random_state=42,
                                       class_weight="balanced")
            try:
                probe.fit(h_te[cal_idx], y_te[cal_idx])
                p_probe = probe.predict_proba(h_te[eval_idx])[:, 1]
            except Exception:
                continue
            _record(res, key, y_te[eval_idx], p_probe)

        print(f"  fold {fold_i}/{len(participants)} ({pid}): "
              f"n_test={len(test)}, errors={y_te.sum()}, "
              f"xgb={res['XGB-pop']['aucs'][-1]:.3f}, "
              f"mlp={res['MLP-pop']['aucs'][-1]:.3f}")

    return res


# ── reporting ─────────────────────────────────────────────────────────────────

def print_results(res: dict) -> None:
    print(f"\n{'='*72}")
    print(f"  Original dataset — MLP transfer learning (LOPO-CV)")
    print(f"{'='*72}")
    print(f"{'Model':<20}  {'AUROC(fold)':>14}  {'AUROC(pool)':>12}  "
          f"{'AP(pool)':>9}  folds")
    print("-" * 66)
    order = ["XGB-pop", "MLP-pop"] + [f"MLP-probe@{k}" for k in K_LIST]
    for name in order:
        r = res[name]
        if not r["aucs"]:
            print(f"{name:<20}  — (no folds)")
            continue
        y_p = np.concatenate(r["y_true"])
        p_p = np.concatenate(r["proba"])
        print(f"{name:<20}  {np.mean(r['aucs']):.3f} ± {np.std(r['aucs']):.3f}  "
              f"{roc_auc_score(y_p, p_p):.3f}        "
              f"{average_precision_score(y_p, p_p):.3f}    {len(r['aucs'])}")

    # Δ vs XGB-pop
    if res["XGB-pop"]["aucs"]:
        base = np.mean(res["XGB-pop"]["aucs"])
        print("\n  Δ AUROC(fold-avg) vs XGB-pop:")
        for name in order[1:]:
            r = res[name]
            if r["aucs"]:
                print(f"    {name:<18}  {np.mean(r['aucs']) - base:+.4f}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    print(f"Loading features (lookback={LOOKBACK_S}s)…")
    feat = load_features()
    n_pid = feat["id"].nunique()
    er = feat[TARGET].mean() * 100
    print(f"  {len(feat)} windows, {n_pid} participants, error_rate={er:.1f}%")
    print(f"  Rows per participant:")
    for pid, g in feat.groupby("id"):
        print(f"    {pid}: {len(g)} windows, {g[TARGET].sum()} errors "
              f"({g[TARGET].mean()*100:.1f}%)")
    print()
    print("Running LOPO-CV…")
    res = lopo_transfer(feat)
    print_results(res)


if __name__ == "__main__":
    main()
