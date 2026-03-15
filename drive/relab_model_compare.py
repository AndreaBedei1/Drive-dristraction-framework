#!/usr/bin/env python3
"""
Model comparison on relab+unibo dataset (LOPO-CV).

Compares classifiers on the best feature set (T1-V) and baseline (T1-A):
  - Logistic Regression      (interpretable linear baseline)
  - Random Forest            (ensemble, handles non-linearity without boosting)
  - XGBoost                  (current baseline)
  - LightGBM                 (faster boosting, sometimes better on small data)
  - SVM (RBF)                (kernel method, good with small N)

Also tests a simple personalisation layer:
  XGBoost + Platt calibration using K=10 samples from the test participant.

Feature sets: T1-A (physio), T1-V (best full).
Target: has_error (T1, 27.6% positive).
Metrics: AUROC (fold-avg ± std), AUROC (pooled OOF), AP (pooled).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# ── import feature loading from relab_ml ─────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from relab_ml import load_features, FEAT_A, FEAT_V

TARGET = "has_error"
K_PERSONALISE = 10   # samples from test participant for Platt layer

# ── classifiers ───────────────────────────────────────────────────────────────

def make_classifiers(scale_pos: float) -> dict:
    return {
        "LogReg":   LogisticRegression(
                        class_weight="balanced", max_iter=1000,
                        solver="lbfgs", C=1.0, random_state=42),
        "RF":       RandomForestClassifier(
                        n_estimators=300, max_depth=6, class_weight="balanced",
                        random_state=42, n_jobs=-1),
        "XGBoost":  XGBClassifier(
                        n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        use_label_encoder=False, eval_metric="logloss",
                        scale_pos_weight=scale_pos,
                        random_state=42, verbosity=0, n_jobs=-1),
        "LightGBM": None,   # filled conditionally below
    }


def _try_lgbm(scale_pos: float):
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            random_state=42, verbosity=-1, n_jobs=-1)
    except ImportError:
        return None


# ── LOPO-CV (multi-model) ─────────────────────────────────────────────────────

def lopo_multi(feat: pd.DataFrame, feature_cols: list, target: str) -> dict:
    participants = feat["id"].unique()
    # accumulate per-fold aucs and pooled predictions per model
    results = {}   # name -> {"aucs": [], "y_true": [], "proba": []}

    for pid in participants:
        train = feat[feat["id"] != pid].copy()
        test  = feat[feat["id"] == pid].copy()

        if test[target].nunique() < 2:
            continue

        X_tr = train[feature_cols].values
        y_tr = train[target].values
        X_te = test[feature_cols].values
        y_te = test[target].values

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

        pos_rate  = y_tr.mean()
        scale_pos = (1 - pos_rate) / max(pos_rate, 1e-6)

        clfs = make_classifiers(scale_pos)
        lgbm = _try_lgbm(scale_pos)
        if lgbm is not None:
            clfs["LightGBM"] = lgbm
        else:
            clfs.pop("LightGBM")

        for name, clf in clfs.items():
            clf.fit(X_tr, y_tr)
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X_te)[:, 1]
            else:
                proba = clf.decision_function(X_te)
                proba = (proba - proba.min()) / (proba.max() - proba.min() + 1e-9)

            if name not in results:
                results[name] = {"aucs": [], "y_true": [], "proba": []}
            results[name]["aucs"].append(roc_auc_score(y_te, proba))
            results[name]["y_true"].append(y_te)
            results[name]["proba"].append(proba)

    # aggregate
    out = {}
    for name, r in results.items():
        y_pool = np.concatenate(r["y_true"])
        p_pool = np.concatenate(r["proba"])
        out[name] = {
            "auroc_mean": float(np.mean(r["aucs"])),
            "auroc_std":  float(np.std(r["aucs"])),
            "auroc_pool": float(roc_auc_score(y_pool, p_pool)),
            "ap_pool":    float(average_precision_score(y_pool, p_pool)),
            "n_folds":    len(r["aucs"]),
        }
    return out


# ── Personalisation: XGBoost + Platt layer using K test samples ───────────────

def lopo_platt_personalise(feat: pd.DataFrame, feature_cols: list,
                            target: str, k: int = K_PERSONALISE) -> dict:
    """
    For each fold:
      1. Train XGBoost on training participants (population model).
      2. Take k samples from the test participant (randomly chosen),
         get population model's log-odds, fit a 1-D logistic (Platt) on them.
      3. Evaluate on remaining test samples.
    Skips folds where the test participant has < k+5 samples or < 2 classes.
    """
    from sklearn.linear_model import LogisticRegression as LR
    participants = feat["id"].unique()
    aucs_pop, aucs_pers = [], []
    y_pool_pop, p_pool_pop = [], []
    y_pool_per, p_pool_per = [], []

    for pid in participants:
        train = feat[feat["id"] != pid].copy()
        test  = feat[feat["id"] == pid].copy()

        if test[target].nunique() < 2 or len(test) < k + 5:
            continue

        X_tr = train[feature_cols].values
        y_tr = train[target].values

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)

        pos_rate  = y_tr.mean()
        scale_pos = (1 - pos_rate) / max(pos_rate, 1e-6)

        clf = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            scale_pos_weight=scale_pos,
            random_state=42, verbosity=0, n_jobs=-1)
        clf.fit(X_tr, y_tr)

        # split test into calibration (k) and evaluation (rest)
        idx_cal  = test.sample(n=k, random_state=42).index
        idx_eval = test.index.difference(idx_cal)
        cal  = test.loc[idx_cal]
        ev   = test.loc[idx_eval]

        if ev[target].nunique() < 2:
            continue

        X_cal  = sc.transform(cal[feature_cols].values)
        X_eval = sc.transform(ev[feature_cols].values)
        y_cal  = cal[target].values
        y_eval = ev[target].values

        # population probabilities
        p_cal_pop  = clf.predict_proba(X_cal)[:, 1]
        p_eval_pop = clf.predict_proba(X_eval)[:, 1]

        # Platt: fit logistic on log-odds of population model
        def logit(p):
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return np.log(p / (1 - p))

        platt = LR(max_iter=200)
        try:
            platt.fit(logit(p_cal_pop).reshape(-1, 1), y_cal)
            p_eval_pers = platt.predict_proba(
                logit(p_eval_pop).reshape(-1, 1))[:, 1]
        except Exception:
            continue

        aucs_pop.append(roc_auc_score(y_eval, p_eval_pop))
        aucs_pers.append(roc_auc_score(y_eval, p_eval_pers))
        y_pool_pop.append(y_eval);  p_pool_pop.append(p_eval_pop)
        y_pool_per.append(y_eval);  p_pool_per.append(p_eval_pers)

    def _agg(aucs, ys, ps):
        y = np.concatenate(ys); p = np.concatenate(ps)
        return {
            "auroc_mean": float(np.mean(aucs)),
            "auroc_std":  float(np.std(aucs)),
            "auroc_pool": float(roc_auc_score(y, p)),
            "ap_pool":    float(average_precision_score(y, p)),
            "n_folds":    len(aucs),
        }

    return {
        f"XGB-pop (eval on {k}-shot remainder)": _agg(aucs_pop,  y_pool_pop, p_pool_pop),
        f"XGB+Platt K={k}":                      _agg(aucs_pers, y_pool_per, p_pool_per),
    }


# ── entry point ───────────────────────────────────────────────────────────────

def print_table(title: str, results: dict) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"{'Model':<28}  {'AUROC(fold)':>14}  {'AUROC(pool)':>12}  {'AP(pool)':>9}  folds")
    print("-" * 72)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["auroc_pool"]):
        print(f"{name:<28}  {r['auroc_mean']:.3f} ± {r['auroc_std']:.3f}  "
              f"{r['auroc_pool']:.3f}        "
              f"{r['ap_pool']:.3f}    {r['n_folds']}")


def main():
    print("Loading features…")
    feat = load_features()
    print(f"  {len(feat)} rows, {feat['id'].nunique()} participants, "
          f"error_rate={feat[TARGET].mean()*100:.1f}%\n")

    for feat_name, feat_cols in [("T1-A (physio)", FEAT_A),
                                  ("T1-V (vehicle+physio)", FEAT_V)]:
        print(f"Running model comparison — {feat_name}…")
        res = lopo_multi(feat, feat_cols, TARGET)
        print_table(feat_name, res)

    print("\nRunning personalisation experiment (T1-V, XGBoost+Platt)…")
    res_pers = lopo_platt_personalise(feat, FEAT_V, TARGET, k=K_PERSONALISE)
    print_table(f"Personalisation K={K_PERSONALISE} (T1-V)", res_pers)


if __name__ == "__main__":
    main()
