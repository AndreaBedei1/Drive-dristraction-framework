#!/usr/bin/env python3
"""
Transfer learning personalisation for driving-error prediction.

Approach (linear probe / last-layer fine-tuning):
  Population phase  : train a MLP on N-1 participants → shared representation.
  Personalisation   : freeze hidden layers, fit a new logistic head on K labeled
                      samples from the test participant (linear probe).
  Evaluation        : test on the remaining (unlabeled-at-fit-time) samples.

Compared against:
  - XGBoost population model (from relab_ml.py)
  - MLP population model (no personalisation)
  - MLP + linear-probe at K = 5 / 10 / 20 / 50

LOPO-CV throughout.
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

import sys
sys.path.insert(0, str(Path(__file__).parent))
from relab_ml import load_features, FEAT_V

TARGET = "has_error"
K_LIST = [5, 10, 20, 50]

MLP_PARAMS = dict(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=300,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,
)

XGB_PARAMS = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric="logloss",
    random_state=42, verbosity=0, n_jobs=-1,
)


# ── hidden-layer activations ──────────────────────────────────────────────────

def get_hidden(mlp: MLPClassifier, X: np.ndarray) -> np.ndarray:
    """Forward pass through all hidden layers (relu), return last hidden repr."""
    h = X
    for W, b in zip(mlp.coefs_[:-1], mlp.intercepts_[:-1]):
        h = h @ W + b
        h = np.maximum(0, h)   # ReLU (matches MLPClassifier default)
    return h


# ── LOPO-CV ───────────────────────────────────────────────────────────────────

def lopo_transfer(feat: pd.DataFrame, feature_cols: list, target: str,
                  k_list: list) -> dict:
    """
    Returns dict: model_name -> {aucs, y_true, proba}
    Models: XGB-pop, MLP-pop, MLP-probe@K for each K in k_list.
    """
    participants = feat["id"].unique()
    names = ["XGB-pop", "MLP-pop"] + [f"MLP-probe@{k}" for k in k_list]
    res = {n: {"aucs": [], "y_true": [], "proba": []} for n in names}

    for fold_i, pid in enumerate(participants, 1):
        train = feat[feat["id"] != pid].copy()
        test  = feat[feat["id"] == pid].copy()

        if test[target].nunique() < 2:
            continue

        X_tr = train[feature_cols].values
        y_tr = train[target].values
        X_te = test[feature_cols].values
        y_te = test[target].values

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s  = sc.transform(X_te)

        pos_rate  = y_tr.mean()
        scale_pos = (1 - pos_rate) / max(pos_rate, 1e-6)

        # ── XGBoost population ────────────────────────────────────────────────
        xgb = XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos)
        xgb.fit(X_tr_s, y_tr)
        p_xgb = xgb.predict_proba(X_te_s)[:, 1]
        _record(res, "XGB-pop", y_te, p_xgb)

        # ── MLP population ────────────────────────────────────────────────────
        # class_weight not supported directly; oversample minority
        pos_idx = np.where(y_tr == 1)[0]
        neg_idx = np.where(y_tr == 0)[0]
        if len(pos_idx) == 0:
            continue
        # replicate positives to match negatives (simple oversampling)
        ratio = len(neg_idx) // len(pos_idx)
        idx_bal = np.concatenate([neg_idx, np.tile(pos_idx, ratio)])
        X_bal = X_tr_s[idx_bal]
        y_bal = y_tr[idx_bal]

        mlp = MLPClassifier(**MLP_PARAMS)
        mlp.fit(X_bal, y_bal)
        p_mlp = mlp.predict_proba(X_te_s)[:, 1]
        _record(res, "MLP-pop", y_te, p_mlp)

        # ── linear probe at various K ─────────────────────────────────────────
        h_te = get_hidden(mlp, X_te_s)   # shape (n_test, 32)

        for k in k_list:
            key = f"MLP-probe@{k}"
            if len(test) < k + 5:
                continue

            # sample K from test participant for calibration
            rng = np.random.default_rng(42)
            cal_idx  = rng.choice(len(test), size=k, replace=False)
            eval_idx = np.setdiff1d(np.arange(len(test)), cal_idx)

            if len(np.unique(y_te[cal_idx])) < 2:
                continue
            if len(np.unique(y_te[eval_idx])) < 2:
                continue

            h_cal  = h_te[cal_idx]
            h_eval = h_te[eval_idx]
            y_cal  = y_te[cal_idx]
            y_eval = y_te[eval_idx]

            probe = LogisticRegression(max_iter=500, random_state=42,
                                       class_weight="balanced")
            try:
                probe.fit(h_cal, y_cal)
                p_probe = probe.predict_proba(h_eval)[:, 1]
            except Exception:
                continue

            _record(res, key, y_eval, p_probe)

        if fold_i % 10 == 0:
            print(f"  fold {fold_i}/{len(participants)} done")

    return res


def _record(res: dict, name: str, y: np.ndarray, p: np.ndarray) -> None:
    if len(np.unique(y)) < 2:
        return
    res[name]["aucs"].append(roc_auc_score(y, p))
    res[name]["y_true"].append(y)
    res[name]["proba"].append(p)


# ── reporting ─────────────────────────────────────────────────────────────────

def print_results(res: dict) -> None:
    print(f"\n{'='*72}")
    print(f"  Transfer learning personalisation — T1-V (LOPO-CV)")
    print(f"{'='*72}")
    print(f"{'Model':<22}  {'AUROC(fold)':>14}  {'AUROC(pool)':>12}  "
          f"{'AP(pool)':>9}  folds")
    print("-" * 68)

    # order: XGB-pop first, then MLP-pop, then probes by K
    order = ["XGB-pop", "MLP-pop"] + [f"MLP-probe@{k}" for k in K_LIST]
    for name in order:
        r = res.get(name)
        if r is None or not r["aucs"]:
            print(f"{name:<22}  — (no folds)")
            continue
        aucs = r["aucs"]
        y_p  = np.concatenate(r["y_true"])
        p_p  = np.concatenate(r["proba"])
        auroc_pool = roc_auc_score(y_p, p_p)
        ap_pool    = average_precision_score(y_p, p_p)
        print(f"{name:<22}  {np.mean(aucs):.3f} ± {np.std(aucs):.3f}  "
              f"{auroc_pool:.3f}        "
              f"{ap_pool:.3f}    {len(aucs)}")

    # delta vs XGB-pop
    if "XGB-pop" in res and res["XGB-pop"]["aucs"]:
        base = np.mean(res["XGB-pop"]["aucs"])
        print("\n  Δ AUROC(fold-avg) vs XGB-pop:")
        for name in order[1:]:
            r = res.get(name)
            if r and r["aucs"]:
                delta = np.mean(r["aucs"]) - base
                print(f"    {name:<20}  {delta:+.4f}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    print("Loading features…")
    feat = load_features()
    print(f"  {len(feat)} rows, {feat['id'].nunique()} participants, "
          f"error_rate={feat[TARGET].mean()*100:.1f}%\n")

    print("Running LOPO-CV (XGB-pop + MLP-pop + linear probes)…")
    res = lopo_transfer(feat, FEAT_V, TARGET, K_LIST)
    print_results(res)


if __name__ == "__main__":
    main()
