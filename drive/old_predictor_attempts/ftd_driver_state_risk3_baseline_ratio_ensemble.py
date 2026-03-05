"""
Ensemble 3-class risk model (baseline-relative ratio) with confusion matrix plot.

Goal:
- Improve multiclass metrics by blending binary probabilities from main + derived features.
- Keep target scientific: r = p_error(second_x) / p_baseline_global(T).
- Save confusion matrix image in evaluation output.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, f1_score

from ftd_driver_state_lookahead import (
    BASE_FEATURE_COLS,
    FEATURE_COLS,
    RANDOM_SEED,
    add_causal_session_features,
    apply_imputation_stats,
    bootstrap_ci,
    build_baselines,
    build_encoding_map,
    build_lookups,
    eval_metrics,
    fit_label_encoders,
    fit_train_imputation_stats,
    generate_samples,
    load_data,
    resolve_xgb_device,
    run_integrity_checks,
    select_threshold_f1,
    split_users,
)
from ftd_driver_state_risk3_baseline_ratio import (
    add_phase_second,
    baseline_prob_from_rate,
    build_phase_curve_train,
    choose_ratio_thresholds,
    map_phase_prob,
    parse_float_pair,
    ratio_to_class,
)
from ftd_driver_state_risk3_from_binary import (
    RISK_LABELS,
    add_future_error_count,
    apply_feature_postprocess_custom,
    build_risk_relation,
    class_distribution,
    estimate_h_from_training,
    fit_feature_postprocess_custom,
    multiclass_metrics,
    normalize_label_cols,
    train_binary_xgb,
)


LOG = logging.getLogger("ftd_driver_state_risk3_ratio_ensemble")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def parse_float_list(raw: str) -> List[float]:
    vals = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected non-empty comma-separated float list")
    out = sorted(set(vals))
    for v in out:
        if v < 0.0 or v > 1.0:
            raise ValueError("Blend weights must be in [0,1]")
    return out


def save_confusion_matrix_png(cm: np.ndarray, labels: Sequence[str], out_path: Path, title: str) -> None:
    cm = np.asarray(cm, dtype=float)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.maximum(row_sum, 1.0))

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    thresh = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{int(cm[i, j])}\\n{cm_norm[i, j]*100:.1f}%"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=9,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Row-normalized rate", rotation=90)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)



def class_details_3c(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    f1_vec = f1_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).astype(float)
    row_sum = cm.sum(axis=1)
    rec = np.divide(np.diag(cm), np.maximum(row_sum, 1.0))
    return {
        "f1_low": float(f1_vec[0]),
        "f1_medium": float(f1_vec[1]),
        "f1_high": float(f1_vec[2]),
        "recall_low": float(rec[0]),
        "recall_medium": float(rec[1]),
        "recall_high": float(rec[2]),
    }


def cal_objective_score(metrics: Dict[str, float], details: Dict[str, float], args) -> float:
    return float(
        args.obj_w_macro * metrics["MacroF1"]
        + args.obj_w_qwk * metrics["QWK"]
        + args.obj_w_bal * metrics["BalancedAcc"]
        + args.obj_w_medium_f1 * details["f1_medium"]
        + args.obj_w_medium_recall * details["recall_medium"]
    )


def tune_ratio_bands_on_cal_weighted(
    ratio_pred_cal: np.ndarray,
    y_cal_cls: np.ndarray,
    args,
) -> Dict:
    rr = np.asarray(ratio_pred_cal, dtype=float)
    y = np.asarray(y_cal_cls, dtype=int)

    q_grid = np.linspace(0.03, 0.97, 65)
    cand = np.unique(np.quantile(rr, q_grid))
    min_frac = float(max(0.0, args.ratio_band_min_frac))
    min_med_recall = float(max(0.0, args.min_medium_recall))

    best = None
    best_key = None
    for i in range(len(cand) - 1):
        t1 = float(cand[i])
        for j in range(i + 1, len(cand)):
            t2 = float(cand[j])
            if t2 <= t1:
                continue
            y_hat = ratio_to_class(rr, t1, t2)
            dist = class_distribution(y_hat)
            if min(dist.values()) < min_frac:
                continue
            m = multiclass_metrics(y, y_hat)
            d = class_details_3c(y, y_hat)
            if d["recall_medium"] < min_med_recall:
                continue
            score = cal_objective_score(m, d, args)
            key = (score, m["MacroF1"], d["f1_medium"], m["QWK"], m["BalancedAcc"])
            if best is None or key > best_key:
                best = {
                    "t1": t1,
                    "t2": t2,
                    "metrics": m,
                    "details": d,
                    "pred_dist": dist,
                    "mode": "grid_quantiles_weighted",
                    "objective_score": float(score),
                }
                best_key = key

    if best is None:
        t1 = float(np.quantile(rr, 1.0 / 3.0))
        t2 = float(np.quantile(rr, 2.0 / 3.0))
        if t2 <= t1:
            t2 = t1 + 1e-3
        y_hat = ratio_to_class(rr, t1, t2)
        m = multiclass_metrics(y, y_hat)
        d = class_details_3c(y, y_hat)
        best = {
            "t1": t1,
            "t2": t2,
            "metrics": m,
            "details": d,
            "pred_dist": class_distribution(y_hat),
            "mode": "terciles_fallback",
            "objective_score": float(cal_objective_score(m, d, args)),
        }
    return best

def prepare_feature_set(
    df_tr: pd.DataFrame,
    df_ca: pd.DataFrame,
    df_te: pd.DataFrame,
    feature_set: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], Dict]:
    if feature_set not in {"main", "derived"}:
        raise ValueError(f"Unsupported feature_set: {feature_set}")

    tr = df_tr.copy()
    ca = df_ca.copy()
    te = df_te.copy()

    feature_cols = list(BASE_FEATURE_COLS if feature_set == "main" else FEATURE_COLS)
    if feature_set == "derived":
        tr = add_causal_session_features(tr)
        ca = add_causal_session_features(ca)
        te = add_causal_session_features(te)

    pp = fit_feature_postprocess_custom(tr, feature_cols)
    tr = apply_feature_postprocess_custom(tr, pp, feature_cols)
    ca = apply_feature_postprocess_custom(ca, pp, feature_cols)
    te = apply_feature_postprocess_custom(te, pp, feature_cols)
    return tr, ca, te, feature_cols, pp


def run_pipeline(args) -> Dict:
    d_raw, e_raw, eb_raw, db_raw = load_data(args.data_path)
    run_integrity_checks(d_raw, e_raw)
    d_raw, e_raw = normalize_label_cols(d_raw, e_raw)

    users = sorted(set(d_raw["user_id"].unique()) | set(e_raw["user_id"].unique()))
    train_users, cal_users, test_users = split_users(users, seed=args.seed)

    impute_stats = fit_train_imputation_stats(train_users, d_raw)
    d = apply_imputation_stats(d_raw, impute_stats)
    d, e_raw = normalize_label_cols(d, e_raw)

    le_pred, le_emo = fit_label_encoders(train_users, d, e_raw)
    pred_enc = build_encoding_map(le_pred)
    emo_enc = build_encoding_map(le_emo)
    baselines = build_baselines(train_users, eb_raw, db_raw)

    d_train = d[d["user_id"].isin(set(train_users))]
    global_model_prob = (
        float(d_train[["model_prob_start", "model_prob_end"]].stack().median())
        if len(d_train)
        else 0.5
    )
    global_emotion_prob = (
        float(d_train[["emotion_prob_start", "emotion_prob_end"]].stack().median())
        if len(d_train)
        else 0.5
    )
    if not np.isfinite(global_model_prob):
        global_model_prob = 0.5
    if not np.isfinite(global_emotion_prob):
        global_emotion_prob = 0.5

    wbs, webs, errs = build_lookups(d, e_raw, users)

    h_info = estimate_h_from_training(
        train_users=train_users,
        distractions=d,
        errors_dist=e_raw,
        quantile=args.h_quantile,
        scale=args.h_scale,
        h_min=args.h_min,
        h_max=args.h_max,
        delay_cap_s=args.h_delay_cap,
    )
    H = int(h_info["H"])
    T = int(args.t_horizon)
    LOG.info("config: H=%s (train-derived) T=%s", H, T)

    df_tr = generate_samples(
        H,
        T,
        train_users,
        wbs,
        webs,
        errs,
        baselines,
        pred_enc,
        emo_enc,
        global_model_prob,
        global_emotion_prob,
    )
    df_ca = generate_samples(
        H,
        T,
        cal_users,
        wbs,
        webs,
        errs,
        baselines,
        pred_enc,
        emo_enc,
        global_model_prob,
        global_emotion_prob,
    )
    df_te = generate_samples(
        H,
        T,
        test_users,
        wbs,
        webs,
        errs,
        baselines,
        pred_enc,
        emo_enc,
        global_model_prob,
        global_emotion_prob,
    )

    df_tr = add_future_error_count(df_tr, errs, horizon_s=T)
    df_ca = add_future_error_count(df_ca, errs, horizon_s=T)
    df_te = add_future_error_count(df_te, errs, horizon_s=T)

    df_tr = add_phase_second(df_tr)
    df_ca = add_phase_second(df_ca)
    df_te = add_phase_second(df_te)

    for name, df in [("train", df_tr), ("cal", df_ca), ("test", df_te)]:
        if df.empty:
            raise RuntimeError(f"{name} split has zero samples")
        LOG.info("%s samples=%s pos_rate=%.4f", name, len(df), float(df["target_bin"].mean()))

    phase_curve = build_phase_curve_train(
        df_train=df_tr,
        smooth_window=args.phase_smooth_window,
        prior_weight=args.phase_prior_weight,
    )
    p_phase_tr = map_phase_prob(df_tr, phase_curve)
    p_phase_ca = map_phase_prob(df_ca, phase_curve)
    p_phase_te = map_phase_prob(df_te, phase_curve)

    global_base_rate = float(np.median(df_tr["baseline_error_rate"].values))
    global_base_p = float(
        baseline_prob_from_rate(np.asarray([global_base_rate], dtype=float), T)[0]
    )

    ratio_true_tr = p_phase_tr / max(global_base_p, 1e-6)
    ratio_true_ca = p_phase_ca / max(global_base_p, 1e-6)
    ratio_true_te = p_phase_te / max(global_base_p, 1e-6)

    fixed_thr = parse_float_pair(args.ratio_thresholds)
    q_thr = parse_float_pair(args.ratio_quantiles)
    thr_info = choose_ratio_thresholds(
        ratio_train=ratio_true_tr,
        mode=args.ratio_threshold_mode,
        fixed_thresholds=fixed_thr,
        quantiles=q_thr,
        min_high_frac=args.min_high_frac,
    )
    t1_train = float(thr_info["t1"])
    t2_train = float(thr_info["t2"])

    y_tr_cls = ratio_to_class(ratio_true_tr, t1_train, t2_train)
    y_ca_cls = ratio_to_class(ratio_true_ca, t1_train, t2_train)
    y_te_cls = ratio_to_class(ratio_true_te, t1_train, t2_train)

    y_tr_bin = df_tr["target_bin"].values.astype(int)
    y_ca_bin = df_ca["target_bin"].values.astype(int)
    y_te_bin = df_te["target_bin"].values.astype(int)

    # Train two base models: main and derived.
    tr_m, ca_m, te_m, feat_main, pp_main = prepare_feature_set(df_tr, df_ca, df_te, "main")
    X_tr_m = tr_m[feat_main].values.astype(float)
    X_ca_m = ca_m[feat_main].values.astype(float)
    X_te_m = te_m[feat_main].values.astype(float)

    tr_d, ca_d, te_d, feat_derived, pp_derived = prepare_feature_set(df_tr, df_ca, df_te, "derived")
    X_tr_d = tr_d[feat_derived].values.astype(float)
    X_ca_d = ca_d[feat_derived].values.astype(float)
    X_te_d = te_d[feat_derived].values.astype(float)

    g_tr = df_tr["user_id"].values
    model_main, info_main = train_binary_xgb(X_tr_m, y_tr_bin, g_tr, args)
    model_derived, info_derived = train_binary_xgb(X_tr_d, y_tr_bin, g_tr, args)

    p_ca_main = model_main.predict_proba(X_ca_m)[:, 1]
    p_te_main = model_main.predict_proba(X_te_m)[:, 1]
    p_ca_derived = model_derived.predict_proba(X_ca_d)[:, 1]
    p_te_derived = model_derived.predict_proba(X_te_d)[:, 1]

    candidates: List[Dict] = []
    # Singles
    candidates.append({"name": "main", "w_main": 1.0, "w_derived": 0.0, "p_cal": p_ca_main, "p_test": p_te_main})
    candidates.append({"name": "derived", "w_main": 0.0, "w_derived": 1.0, "p_cal": p_ca_derived, "p_test": p_te_derived})

    for w in parse_float_list(args.blend_weights):
        if w <= 0.0 or w >= 1.0:
            continue
        p_cal = w * p_ca_main + (1.0 - w) * p_ca_derived
        p_te = w * p_te_main + (1.0 - w) * p_te_derived
        candidates.append(
            {
                "name": f"blend_wmain_{w:.2f}",
                "w_main": float(w),
                "w_derived": float(1.0 - w),
                "p_cal": p_cal,
                "p_test": p_te,
            }
        )

    eval_rows: List[Dict] = []
    best = None
    best_key = None

    for cand in candidates:
        p_cal = np.asarray(cand["p_cal"], dtype=float)
        p_te = np.asarray(cand["p_test"], dtype=float)

        ratio_pred_cal = p_cal / max(global_base_p, 1e-6)
        ratio_pred_te = p_te / max(global_base_p, 1e-6)

        y_hat_cal_fixed = ratio_to_class(ratio_pred_cal, t1_train, t2_train)
        y_hat_te_fixed = ratio_to_class(ratio_pred_te, t1_train, t2_train)
        metrics_cal_fixed = multiclass_metrics(y_ca_cls, y_hat_cal_fixed)
        metrics_test_fixed = multiclass_metrics(y_te_cls, y_hat_te_fixed)

        if np.unique(y_ca_cls).size >= 3:
            band = tune_ratio_bands_on_cal_weighted(
                ratio_pred_cal=ratio_pred_cal,
                y_cal_cls=y_ca_cls,
                args=args,
            )
        else:
            details_fixed = class_details_3c(y_ca_cls, y_hat_cal_fixed)
            band = {
                "t1": float(t1_train),
                "t2": float(t2_train),
                "metrics": metrics_cal_fixed,
                "details": details_fixed,
                "pred_dist": class_distribution(y_hat_cal_fixed),
                "mode": "skipped_insufficient_cal_classes",
                "objective_score": float(cal_objective_score(metrics_cal_fixed, details_fixed, args)),
            }

        y_hat_cal_tuned = ratio_to_class(ratio_pred_cal, band["t1"], band["t2"])
        y_hat_te_tuned = ratio_to_class(ratio_pred_te, band["t1"], band["t2"])
        metrics_cal_tuned = multiclass_metrics(y_ca_cls, y_hat_cal_tuned)
        metrics_test_tuned = multiclass_metrics(y_te_cls, y_hat_te_tuned)
        details_cal_tuned = class_details_3c(y_ca_cls, y_hat_cal_tuned)
        objective_cal = float(cal_objective_score(metrics_cal_tuned, details_cal_tuned, args))

        row = {
            "name": cand["name"],
            "w_main": float(cand["w_main"]),
            "w_derived": float(cand["w_derived"]),
            "band_t1": float(band["t1"]),
            "band_t2": float(band["t2"]),
            "band_mode": str(band.get("mode", "unknown")),
            "metrics_cal_fixed": metrics_cal_fixed,
            "metrics_test_fixed": metrics_test_fixed,
            "metrics_cal_tuned": metrics_cal_tuned,
            "metrics_test_tuned": metrics_test_tuned,
            "cal_details_tuned": details_cal_tuned,
            "cal_objective_score": objective_cal,
            "pred_dist_test_tuned": class_distribution(y_hat_te_tuned),
            "p_cal": p_cal,
            "p_test": p_te,
            "y_hat_te_tuned": y_hat_te_tuned,
            "y_hat_ca_tuned": y_hat_cal_tuned,
            "ratio_pred_te": ratio_pred_te,
        }
        eval_rows.append(row)

        key = (
            row["cal_objective_score"],
            metrics_cal_tuned["MacroF1"],
            details_cal_tuned["f1_medium"],
            metrics_cal_tuned["QWK"],
            metrics_cal_tuned["BalancedAcc"],
        )
        if best is None or key > best_key:
            best = row
            best_key = key

    assert best is not None

    p_ca_best = np.asarray(best["p_cal"], dtype=float)
    p_te_best = np.asarray(best["p_test"], dtype=float)
    y_hat_ca = np.asarray(best["y_hat_ca_tuned"], dtype=int)
    y_hat_te = np.asarray(best["y_hat_te_tuned"], dtype=int)

    multi_cal = multiclass_metrics(y_ca_cls, y_hat_ca)
    multi_test = multiclass_metrics(y_te_cls, y_hat_te)

    cm_test = confusion_matrix(y_te_cls, y_hat_te, labels=[0, 1, 2]).tolist()
    report_test = classification_report(
        y_te_cls,
        y_hat_te,
        labels=[0, 1, 2],
        target_names=RISK_LABELS,
        output_dict=True,
        zero_division=0,
    )

    thr_bin = select_threshold_f1(y_ca_bin, p_ca_best)
    bin_metrics_test = eval_metrics(y_te_bin, p_te_best, thr_bin)

    ci_qwk = bootstrap_ci(
        y_te_cls,
        y_hat_te.astype(float),
        lambda y, yhatf: cohen_kappa_score(y, yhatf.astype(int), weights="quadratic"),
        n_boot=args.bootstrap,
    )
    ci_macrof1 = bootstrap_ci(
        y_te_cls,
        y_hat_te.astype(float),
        lambda y, yhatf: f1_score(y, yhatf.astype(int), average="macro"),
        n_boot=args.bootstrap,
    )

    relation_pred_test = build_risk_relation(df_te["future_error_count"].values, y_hat_te)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    cm_png = outdir / "confusion_matrix_test.png"
    save_confusion_matrix_png(
        np.asarray(cm_test, dtype=float),
        labels=RISK_LABELS,
        out_path=cm_png,
        title=f"Confusion Matrix (test) - {best['name']}",
    )

    phase_out = phase_curve.copy()
    phase_out["ratio_vs_global_baseline"] = phase_out["p_error"] / max(global_base_p, 1e-6)

    artifact = {
        "model_main": model_main,
        "model_derived": model_derived,
        "best_candidate": {
            "name": best["name"],
            "w_main": best["w_main"],
            "w_derived": best["w_derived"],
            "band_t1": best["band_t1"],
            "band_t2": best["band_t2"],
            "band_mode": best["band_mode"],
        },
        "feature_cols_main": feat_main,
        "feature_cols_derived": feat_derived,
        "feature_postprocess_main": pp_main,
        "feature_postprocess_derived": pp_derived,
        "best_config": {"H": H, "T": T},
        "h_estimation": h_info,
        "phase_curve_train": phase_out,
        "ratio_thresholds_train": {"t1": t1_train, "t2": t2_train},
        "global_baseline_error_rate_train": float(global_base_rate),
        "global_baseline_probability_T_train": float(global_base_p),
        "label_encoders": {"model_pred": le_pred, "emotion": le_emo},
        "encoding_maps": {"model_pred": pred_enc, "emotion": emo_enc},
        "impute_stats": impute_stats,
        "baselines": baselines,
        "global_driver_state_defaults": {
            "model_prob": float(global_model_prob),
            "emotion_prob": float(global_emotion_prob),
        },
    }
    joblib.dump(artifact, outdir / "driver_state_risk3_ratio_ensemble_model.joblib")

    result = {
        "task": "3-class risk from baseline-relative ratio with main+derived ensemble",
        "constraints": {
            "driver_state_only": True,
            "leakage_controls": [
                "user-level train/cal/test split",
                "train-only imputation",
                "train-only encoders",
                "train-only feature clipping",
                "H estimated from train only",
                "phase risk curve estimated on train only",
                "blend and class bands tuned on calibration only",
                "single locked test evaluation",
            ],
        },
        "device": {
            "xgb_requested": args.xgb_device,
            "xgb_resolved": args.xgb_device_resolved,
            "xgb_n_jobs": int(args.xgb_n_jobs),
        },
        "users": {"train": train_users, "cal": cal_users, "test": test_users},
        "best_config": {"H": H, "T": T},
        "h_estimation": h_info,
        "target_definition": {
            "ratio_formula": "r = p_error(second_x) / p_baseline_global(T)",
            "p_error_second_x_source": "training phase curve over generated samples",
            "p_baseline_T_formula": "1 - exp(-median_train_baseline_error_rate * T)",
            "ratio_threshold_mode_requested": args.ratio_threshold_mode,
            "ratio_thresholds_fixed_requested": [fixed_thr[0], fixed_thr[1]],
            "ratio_quantiles_requested": [q_thr[0], q_thr[1]],
            "thresholds_from_train": {
                "t1": t1_train,
                "t2": t2_train,
                "mode_used": thr_info.get("mode_used", "unknown"),
                "train_dist": thr_info.get("train_dist", {}),
            },
            "global_baseline_error_rate_train": float(global_base_rate),
            "global_baseline_probability_T_train": float(global_base_p),
            "cal_objective_weights": {
                "macro": float(args.obj_w_macro),
                "qwk": float(args.obj_w_qwk),
                "balanced_acc": float(args.obj_w_bal),
                "medium_f1": float(args.obj_w_medium_f1),
                "medium_recall": float(args.obj_w_medium_recall),
                "min_medium_recall": float(args.min_medium_recall),
            },
        },
        "samples": {"train": int(len(df_tr)), "cal": int(len(df_ca)), "test": int(len(df_te))},
        "class_distribution_true": {
            "train": class_distribution(y_tr_cls),
            "cal": class_distribution(y_ca_cls),
            "test": class_distribution(y_te_cls),
        },
        "ensemble_candidates": [
            {
                "name": c["name"],
                "w_main": c["w_main"],
                "w_derived": c["w_derived"],
                "band_t1": c["band_t1"],
                "band_t2": c["band_t2"],
                "band_mode": c["band_mode"],
                "metrics_cal_tuned": c["metrics_cal_tuned"],
                "metrics_test_tuned": c["metrics_test_tuned"],
                "pred_dist_test_tuned": c["pred_dist_test_tuned"],
                "cal_details_tuned": c["cal_details_tuned"],
                "cal_objective_score": c["cal_objective_score"],
            }
            for c in eval_rows
        ],
        "best_candidate": {
            "name": best["name"],
            "w_main": float(best["w_main"]),
            "w_derived": float(best["w_derived"]),
            "band_t1": float(best["band_t1"]),
            "band_t2": float(best["band_t2"]),
            "band_mode": str(best["band_mode"]),
            "cal_details_tuned": best["cal_details_tuned"],
            "cal_objective_score": float(best["cal_objective_score"]),
        },
        "class_distribution_pred_test": class_distribution(y_hat_te),
        "metrics_cal_multiclass_selected": multi_cal,
        "metrics_test_multiclass_selected": multi_test,
        "metrics_test_binary_proxy": bin_metrics_test,
        "ci_test_multiclass": {
            "QWK": {"mean": ci_qwk[0], "low": ci_qwk[1], "high": ci_qwk[2]},
            "MacroF1": {"mean": ci_macrof1[0], "low": ci_macrof1[1], "high": ci_macrof1[2]},
        },
        "per_class_report_test": report_test,
        "confusion_matrix_test": cm_test,
        "confusion_matrix_test_png": str(cm_png),
        "risk_relation_test_by_predicted_class": relation_pred_test,
        "phase_curve_train": phase_out.to_dict(orient="records"),
        "train_info_binary": {
            "main": info_main,
            "derived": info_derived,
        },
    }

    with open(outdir / "driver_state_risk3_ratio_ensemble_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Baseline-ratio 3-class ensemble (main+derived) with confusion matrix image"
    )
    ap.add_argument("--data-path", default="data")
    ap.add_argument("--output-dir", default="evaluation/driver_state_risk3_ratio_ensemble")
    ap.add_argument("--t-horizon", type=int, default=5)

    ap.add_argument("--h-scale", type=float, default=0.8)
    ap.add_argument("--h-quantile", type=float, default=0.75)
    ap.add_argument("--h-min", type=int, default=3)
    ap.add_argument("--h-max", type=int, default=24)
    ap.add_argument("--h-delay-cap", type=float, default=40.0)

    ap.add_argument("--ratio-threshold-mode", choices=["auto", "quantile", "fixed"], default="auto")
    ap.add_argument("--ratio-thresholds", default="7,10")
    ap.add_argument("--ratio-quantiles", default="0.70,0.90")
    ap.add_argument("--min-high-frac", type=float, default=0.10)

    ap.add_argument("--phase-smooth-window", type=int, default=3)
    ap.add_argument("--phase-prior-weight", type=float, default=30.0)

    ap.add_argument("--ratio-band-min-frac", type=float, default=0.05)
    ap.add_argument("--blend-weights", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    ap.add_argument("--min-medium-recall", type=float, default=0.18)
    ap.add_argument("--obj-w-macro", type=float, default=0.35)
    ap.add_argument("--obj-w-qwk", type=float, default=0.20)
    ap.add_argument("--obj-w-bal", type=float, default=0.10)
    ap.add_argument("--obj-w-medium-f1", type=float, default=0.20)
    ap.add_argument("--obj-w-medium-recall", type=float, default=0.15)
    ap.add_argument("--search-iter", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=120)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--xgb-device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--xgb-n-jobs", type=int, default=-1)
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    args.xgb_device_resolved = resolve_xgb_device(args.xgb_device)
    if args.xgb_device_resolved == "cuda" and int(args.xgb_n_jobs) <= 0:
        args.xgb_n_jobs = 1

    LOG.info(
        "run config: T=%s ratio_mode=%s xgb=%s/%s",
        args.t_horizon,
        args.ratio_threshold_mode,
        args.xgb_device,
        args.xgb_device_resolved,
    )
    result = run_pipeline(args)
    m = result["metrics_test_multiclass_selected"]
    LOG.info(
        "FINAL TEST MULTI: MacroF1=%.4f WeightedF1=%.4f BalAcc=%.4f QWK=%.4f | H=%s T=%s best=%s",
        m["MacroF1"],
        m["WeightedF1"],
        m["BalancedAcc"],
        m["QWK"],
        result["best_config"]["H"],
        result["best_config"]["T"],
        result["best_candidate"]["name"],
    )
    LOG.info("confusion matrix image: %s", result["confusion_matrix_test_png"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())







