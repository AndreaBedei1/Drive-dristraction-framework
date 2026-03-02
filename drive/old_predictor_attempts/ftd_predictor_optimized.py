"""
Optimized Fitness-to-Drive predictor.

Improves on legacy scripts with:
- stronger preprocessing and missing-value handling,
- richer per-second feature engineering,
- quick H/T configuration search,
- tuned XGBoost + logistic baseline,
- proper post-hoc calibration on held-out calibration users.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from bisect import bisect_left
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


RANDOM_SEED = 42
UNKNOWN_LABEL = "unknown"
RECALL_LEVELS = [0.80, 0.85, 0.90, 0.95]

LOG = logging.getLogger("ftd_opt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


FEATURE_COLS = [
    "time_in_current_dist",
    "time_since_dist_end",
    "cognitive_load_decay",
    "model_prob",
    "model_prob_sq",
    "model_pred_enc",
    "emotion_prob",
    "emotion_prob_sq",
    "emotion_label_enc",
    "arousal",
    "arousal_decay",
    "arousal_dev",
    "arousal_dev_sq",
    "hr_bpm",
    "hr_dev",
    "hr_dev_sq",
    "dist_density_30",
    "dist_density_60",
    "dist_density_120",
    "err_density_10",
    "err_density_30",
    "err_density_60",
    "errors_so_far",
    "user_arousal_baseline",
    "user_hr_baseline",
    "baseline_error_rate",
    "sensor_missing_flag",
]


def parse_int_list(raw: str) -> List[int]:
    vals = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one integer")
    return vals


def norm_label(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    return out.replace(
        {
            "": UNKNOWN_LABEL,
            "nan": UNKNOWN_LABEL,
            "NaN": UNKNOWN_LABEL,
            "None": UNKNOWN_LABEL,
            "none": UNKNOWN_LABEL,
            "<NA>": UNKNOWN_LABEL,
        }
    )


def to_numeric(df: pd.DataFrame, cols: Sequence[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def fill_user_global(df: pd.DataFrame, col: str, fallback: float = 0.0) -> None:
    med = df.groupby("user_id")[col].transform("median")
    df[col] = df[col].fillna(med)
    g = df[col].median()
    if pd.isna(g):
        g = float(fallback)
    df[col] = df[col].fillna(float(g))


def load_data(data_path: str, max_window_seconds: float = 0.0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(os.path.join(data_path, "Dataset Distractions_distraction.csv"))
    e = pd.read_csv(os.path.join(data_path, "Dataset Errors_distraction.csv"))
    eb = pd.read_csv(os.path.join(data_path, "Dataset Errors_baseline.csv"))
    db = pd.read_csv(os.path.join(data_path, "Dataset Driving Time_baseline.csv"))

    d["timestamp_start"] = pd.to_datetime(d["timestamp_start"], errors="coerce", format="ISO8601")
    d["timestamp_end"] = pd.to_datetime(d["timestamp_end"], errors="coerce", format="ISO8601")
    e["timestamp"] = pd.to_datetime(e["timestamp"], errors="coerce", format="ISO8601")

    d = d.dropna(subset=["timestamp_start", "timestamp_end"]).copy()
    d = d[d["timestamp_end"] >= d["timestamp_start"]].copy()
    if float(max_window_seconds) > 0:
        duration_s = (d["timestamp_end"] - d["timestamp_start"]).dt.total_seconds()
        d = d[duration_s <= float(max_window_seconds)].copy()
    e = e.dropna(subset=["timestamp"]).copy()

    for col in ["model_pred_start", "model_pred_end", "emotion_label_start", "emotion_label_end", "details"]:
        if col in d.columns:
            d[col] = norm_label(d[col])
    for col in ["model_pred", "emotion_label", "error_type", "details"]:
        if col in e.columns:
            e[col] = norm_label(e[col])
    if "error_type" in eb.columns:
        eb["error_type"] = norm_label(eb["error_type"])

    d_num = [
        "arousal_start",
        "arousal_end",
        "hr_bpm_start",
        "hr_bpm_end",
        "model_prob_start",
        "model_prob_end",
        "emotion_prob_start",
        "emotion_prob_end",
        "speed_kmh_start",
        "speed_kmh_end",
        "steer_angle_deg_start",
        "steer_angle_deg_end",
    ]
    e_num = ["model_prob", "emotion_prob", "speed_kmh", "steer_angle_deg", "frame", "sim_time_seconds"]
    db_num = ["run_duration_seconds", "hr_baseline", "arousal_baseline"]

    to_numeric(d, d_num)
    to_numeric(e, e_num)
    to_numeric(db, db_num)

    # Preserve missingness information before imputation.
    d["flag_arousal_missing"] = ((d["arousal_start"].isna()) | (d["arousal_end"].isna())).astype(float)
    d["flag_hr_missing"] = ((d["hr_bpm_start"].isna()) | (d["hr_bpm_end"].isna())).astype(float)
    d["flag_sensor_missing"] = ((d["flag_arousal_missing"] > 0.0) | (d["flag_hr_missing"] > 0.0)).astype(float)

    for a, b in [
        ("arousal_start", "arousal_end"),
        ("hr_bpm_start", "hr_bpm_end"),
        ("model_prob_start", "model_prob_end"),
        ("emotion_prob_start", "emotion_prob_end"),
        ("speed_kmh_start", "speed_kmh_end"),
        ("steer_angle_deg_start", "steer_angle_deg_end"),
    ]:
        d[a] = d[a].fillna(d[b])
        d[b] = d[b].fillna(d[a])

    for col in d_num:
        fb = 0.0
        if "hr_bpm" in col:
            fb = 70.0
        elif "arousal" in col:
            fb = 0.5
        fill_user_global(d, col, fallback=fb)

    for col in e_num:
        fb = 0.5 if "prob" in col else 0.0
        fill_user_global(e, col, fallback=fb)

    for col in db_num:
        fb = 70.0 if col == "hr_baseline" else 0.5 if col == "arousal_baseline" else 0.0
        fill_user_global(db, col, fallback=fb)

    for col in ["model_prob_start", "model_prob_end", "emotion_prob_start", "emotion_prob_end", "arousal_start", "arousal_end"]:
        d[col] = d[col].clip(0.0, 1.0)
    for col in ["model_prob", "emotion_prob"]:
        e[col] = e[col].clip(0.0, 1.0)
    for col in ["hr_bpm_start", "hr_bpm_end"]:
        d[col] = d[col].clip(35.0, 220.0)
    for col in ["speed_kmh_start", "speed_kmh_end", "speed_kmh"]:
        if col in d.columns:
            d[col] = d[col].clip(lower=0.0)
        if col in e.columns:
            e[col] = e[col].clip(lower=0.0)

    db = db[db["run_duration_seconds"] > 0].copy()
    return d.reset_index(drop=True), e.reset_index(drop=True), eb.reset_index(drop=True), db.reset_index(drop=True)


def split_users(users: Sequence[str], seed: int = RANDOM_SEED) -> Tuple[List[str], List[str], List[str]]:
    arr = np.array(sorted(set(users)))
    rng = np.random.RandomState(seed)
    rng.shuffle(arr)
    n_total = len(arr)
    n_test = max(1, int(0.2 * n_total))
    n_cal = max(1, int(0.2 * (n_total - n_test)))
    test_users = arr[:n_test].tolist()
    cal_users = arr[n_test : n_test + n_cal].tolist()
    train_users = arr[n_test + n_cal :].tolist()
    return train_users, cal_users, test_users


def build_lookups(distractions: pd.DataFrame, errors_dist: pd.DataFrame, users: Iterable[str]):
    users_set = set(users)
    dsel = distractions[distractions["user_id"].isin(users_set)]
    esel = errors_dist[errors_dist["user_id"].isin(users_set)]

    wbs: Dict[Tuple[str, int], pd.DataFrame] = {}
    webs: Dict[Tuple[str, int], List[pd.Timestamp]] = {}
    errs: Dict[Tuple[str, int], List[pd.Timestamp]] = {}

    for (uid, rid), grp in dsel.groupby(["user_id", "run_id"], sort=False):
        key = (str(uid), int(rid))
        g = grp.sort_values("timestamp_start").reset_index(drop=True)
        wbs[key] = g
        webs[key] = sorted(g["timestamp_end"].tolist())

    for (uid, rid), grp in esel.groupby(["user_id", "run_id"], sort=False):
        errs[(str(uid), int(rid))] = sorted(grp["timestamp"].tolist())

    return wbs, webs, errs


def fit_encoders(train_users: Iterable[str], d: pd.DataFrame, e: pd.DataFrame) -> Tuple[LabelEncoder, LabelEncoder]:
    users = set(train_users)
    dtr = d[d["user_id"].isin(users)]
    etr = e[e["user_id"].isin(users)]

    pred_vocab = sorted(
        set(dtr["model_pred_start"].tolist())
        | set(dtr["model_pred_end"].tolist())
        | set(etr["model_pred"].tolist())
        | {UNKNOWN_LABEL}
    )
    emo_vocab = sorted(
        set(dtr["emotion_label_start"].tolist())
        | set(dtr["emotion_label_end"].tolist())
        | set(etr["emotion_label"].tolist())
        | {UNKNOWN_LABEL}
    )
    return LabelEncoder().fit(pred_vocab), LabelEncoder().fit(emo_vocab)


def safe_encode(value: str, enc: LabelEncoder) -> int:
    v = str(value).strip() if value is not None else UNKNOWN_LABEL
    if not v:
        v = UNKNOWN_LABEL
    if v not in enc.classes_:
        v = UNKNOWN_LABEL
    return int(enc.transform([v])[0])


def build_baselines(train_users: Iterable[str], d: pd.DataFrame, eb: pd.DataFrame, db: pd.DataFrame):
    users = set(train_users)
    dtr = d[d["user_id"].isin(users)]
    ebtr = eb[eb["user_id"].isin(users)]
    dbtr = db[db["user_id"].isin(users)]

    total_s = float(dbtr["run_duration_seconds"].sum()) if not dbtr.empty else 0.0
    global_err_rate = float(len(ebtr) / total_s) if total_s > 0 else 0.0
    user_err = ebtr.groupby("user_id").size().astype(float)
    user_secs = dbtr.groupby("user_id")["run_duration_seconds"].sum().astype(float)
    user_err_rate = (user_err / user_secs).replace([np.inf, -np.inf], np.nan).to_dict()

    user_hr = dbtr.groupby("user_id")["hr_baseline"].median().to_dict() if "hr_baseline" in dbtr.columns else {}
    user_arousal = dbtr.groupby("user_id")["arousal_baseline"].median().to_dict() if "arousal_baseline" in dbtr.columns else {}

    hr_global = float(pd.concat([dtr["hr_bpm_start"], dtr["hr_bpm_end"]]).median()) if not dtr.empty else 70.0
    ar_global = float(pd.concat([dtr["arousal_start"], dtr["arousal_end"]]).median()) if not dtr.empty else 0.5
    if np.isnan(hr_global):
        hr_global = 70.0
    if np.isnan(ar_global):
        ar_global = 0.5

    return {
        "user_err_rate": {str(k): float(v) for k, v in user_err_rate.items() if pd.notna(v)},
        "user_hr": {str(k): float(v) for k, v in user_hr.items() if pd.notna(v)},
        "user_arousal": {str(k): float(v) for k, v in user_arousal.items() if pd.notna(v)},
        "global_err_rate": float(global_err_rate),
        "global_hr": float(hr_global),
        "global_arousal": float(ar_global),
    }


def density(items: List[pd.Timestamp], ts: pd.Timestamp, lookback_s: int) -> int:
    if not items:
        return 0
    lo = ts - pd.Timedelta(seconds=lookback_s)
    return bisect_left(items, ts) - bisect_left(items, lo)


def prev_duration(wins: pd.DataFrame, ts: pd.Timestamp) -> float:
    prev = wins[wins["timestamp_end"] < ts]
    if prev.empty:
        return 0.0
    row = prev.loc[prev["timestamp_end"].idxmax()]
    return float((row["timestamp_end"] - row["timestamp_start"]).total_seconds())


def pick_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    p, r, t = precision_recall_curve(y_true, y_prob)
    if len(t) == 0:
        return 0.5
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-9)
    return float(t[int(np.argmax(f1))])


def eval_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, float]:
    y_bin = (y_prob >= thr).astype(int)
    p, r, _ = precision_recall_curve(y_true, y_prob)
    out = {
        "AUC-PR": float(average_precision_score(y_true, y_prob)),
        "AUC-ROC": float(roc_auc_score(y_true, y_prob)),
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "Log-Loss": float(log_loss(y_true, y_prob)),
        "MCC": float(matthews_corrcoef(y_true, y_bin)),
        "Kappa": float(cohen_kappa_score(y_true, y_bin)),
        "F1": float(f1_score(y_true, y_bin)),
        "Precision": float(precision_score(y_true, y_bin)),
        "Recall": float(recall_score(y_true, y_bin)),
        "Threshold": float(thr),
    }
    for lv in RECALL_LEVELS:
        mask = r >= lv
        out[f"P@{int(lv*100)}"] = float(p[mask].max()) if mask.any() else 0.0
    return out


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, metric_fn, n_bootstrap: int):
    rng = np.random.RandomState(RANDOM_SEED)
    idx = np.arange(len(y_true))
    vals: List[float] = []
    for _ in range(int(n_bootstrap)):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_true[b])) < 2:
            continue
        vals.append(float(metric_fn(y_true[b], y_prob[b])))
    if not vals:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _xgb_cuda_available() -> bool:
    # Fast path from build metadata when available.
    try:
        info = xgb.build_info()
        use_cuda = str(info.get("USE_CUDA", "")).strip().lower()
        if use_cuda in {"1", "true", "yes", "on"}:
            return True
        if use_cuda in {"0", "false", "no", "off"}:
            return False
    except Exception:
        pass

    # Runtime probe fallback.
    try:
        probe = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=1,
            max_depth=1,
            learning_rate=0.1,
            tree_method="hist",
            device="cuda",
            n_jobs=1,
        )
        Xp = np.array([[0.0], [1.0]], dtype=np.float32)
        yp = np.array([0, 1], dtype=np.int32)
        probe.fit(Xp, yp)
        return True
    except Exception:
        return False


def resolve_xgb_device(requested: str) -> str:
    req = str(requested).strip().lower()
    if req not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported xgb device '{requested}'. Use auto|cpu|cuda.")
    if req == "cpu":
        return "cpu"
    if req == "cuda":
        if not _xgb_cuda_available():
            raise RuntimeError("XGBoost CUDA requested but not available in this environment.")
        return "cuda"
    return "cuda" if _xgb_cuda_available() else "cpu"


def gen_samples(
    mode: str,
    H: int,
    T: int,
    users: Iterable[str],
    wbs: Dict[Tuple[str, int], pd.DataFrame],
    webs: Dict[Tuple[str, int], List[pd.Timestamp]],
    errs: Dict[Tuple[str, int], List[pd.Timestamp]],
    baselines,
    le_pred: LabelEncoder,
    le_emo: LabelEncoder,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    users_set = set(users)
    H = max(1, int(H))
    T = max(1, int(T))

    for key, wins in wbs.items():
        uid, rid = key
        if uid not in users_set or wins.empty:
            continue
        err_ts = errs.get(key, [])
        usr_ar = baselines["user_arousal"].get(uid, baselines["global_arousal"])
        usr_hr = baselines["user_hr"].get(uid, baselines["global_hr"])
        usr_er = baselines["user_err_rate"].get(uid, baselines["global_err_rate"])

        for i in range(len(wins)):
            row = wins.iloc[i]
            start_ts = row["timestamp_start"]
            end_ts = row["timestamp_end"]
            win_dur = float((end_ts - start_ts).total_seconds())
            if win_dur <= 0:
                continue

            next_start = wins.iloc[i + 1]["timestamp_start"] if i + 1 < len(wins) else None
            hang = float(H)
            if next_start is not None:
                hang = min(hang, float((next_start - end_ts).total_seconds()))
            hang = max(0.0, hang)
            upper_ts = end_ts + pd.Timedelta(seconds=hang)
            span_s = float((upper_ts - start_ts).total_seconds())
            n_bins = int(np.ceil(span_s))
            if n_bins <= 0:
                continue

            s_ar, e_ar = float(row["arousal_start"]), float(row["arousal_end"])
            s_hr, e_hr = float(row["hr_bpm_start"]), float(row["hr_bpm_end"])
            s_mp, e_mp = float(row["model_prob_start"]), float(row["model_prob_end"])
            s_ep, e_ep = float(row["emotion_prob_start"]), float(row["emotion_prob_end"])
            s_pr, e_pr = float(safe_encode(row["model_pred_start"], le_pred)), float(safe_encode(row["model_pred_end"], le_pred))
            s_em, e_em = float(safe_encode(row["emotion_label_start"], le_emo)), float(safe_encode(row["emotion_label_end"], le_emo))
            sensor_missing_flag = float(row.get("flag_sensor_missing", 0.0))

            for off in range(n_bins):
                ts = start_ts + pd.Timedelta(seconds=off)
                inside = float(off < win_dur)
                if inside:
                    a = float(np.clip(off / max(win_dur, 1e-6), 0.0, 1.0))
                    cur_ar = s_ar + (e_ar - s_ar) * a
                    cur_hr = s_hr + (e_hr - s_hr) * a
                    cur_mp = s_mp + (e_mp - s_mp) * a
                    cur_ep = s_ep + (e_ep - s_ep) * a
                    cur_pr = s_pr + (e_pr - s_pr) * a
                    cur_em = s_em + (e_em - s_em) * a
                    t_in = float(off)
                    t_after = 0.0
                    cld = cur_mp
                    ar_decay = cur_ar
                else:
                    t_after = float(off - win_dur)
                    dec = float(np.exp(-t_after / max(float(H), 1e-6)))
                    cur_ar, cur_hr, cur_mp, cur_ep = e_ar, e_hr, e_mp, e_ep
                    cur_pr, cur_em = e_pr, e_em
                    t_in = 0.0
                    cld = cur_mp * dec
                    ar_decay = usr_ar + (cur_ar - usr_ar) * dec

                hz = 1 if mode == "second_per_second" else T
                j = bisect_left(err_ts, ts)
                errors_so_far = float(j)
                target = 1 if j < len(err_ts) and err_ts[j] < ts + pd.Timedelta(seconds=hz) else 0
                ar_dev = cur_ar - usr_ar
                hr_dev = cur_hr - usr_hr

                rows.append(
                    {
                        "user_id": uid,
                        "run_id": int(rid),
                        "target": int(target),
                        "time_in_current_dist": t_in,
                        "time_since_dist_end": t_after,
                        "cognitive_load_decay": cld,
                        "model_prob": cur_mp,
                        "model_prob_sq": cur_mp * cur_mp,
                        "model_pred_enc": cur_pr,
                        "emotion_prob": cur_ep,
                        "emotion_prob_sq": cur_ep * cur_ep,
                        "emotion_label_enc": cur_em,
                        "arousal": cur_ar,
                        "arousal_decay": ar_decay,
                        "arousal_dev": ar_dev,
                        "arousal_dev_sq": ar_dev * ar_dev,
                        "hr_bpm": cur_hr,
                        "hr_dev": hr_dev,
                        "hr_dev_sq": hr_dev * hr_dev,
                        "dist_density_30": float(density(webs.get(key, []), ts, 30)),
                        "dist_density_60": float(density(webs.get(key, []), ts, 60)),
                        "dist_density_120": float(density(webs.get(key, []), ts, 120)),
                        "err_density_10": float(density(err_ts, ts, 10)),
                        "err_density_30": float(density(err_ts, ts, 30)),
                        "err_density_60": float(density(err_ts, ts, 60)),
                        "errors_so_far": errors_so_far,
                        "user_arousal_baseline": usr_ar,
                        "user_hr_baseline": usr_hr,
                        "baseline_error_rate": usr_er,
                        "sensor_missing_flag": sensor_missing_flag,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in FEATURE_COLS:
        fill_user_global(df, c, fallback=0.0)
    return df


def default_xgb(pos_rate: float, xgb_device: str, xgb_n_jobs: int) -> Pipeline:
    spw = max((1.0 - pos_rate) / max(pos_rate, 1e-6), 1.0)
    mdl = xgb.XGBClassifier(
        objective="binary:logistic",
        random_state=RANDOM_SEED,
        eval_metric="logloss",
        tree_method="hist",
        device=str(xgb_device),
        n_jobs=int(xgb_n_jobs),
        n_estimators=320,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=2.0,
        reg_alpha=0.1,
        scale_pos_weight=spw,
    )
    return Pipeline([("scaler", StandardScaler()), ("xgb", mdl)])


def quick_config_search(
    mode: str,
    H_vals: Sequence[int],
    T_vals: Sequence[int],
    train_users: Sequence[str],
    wbs,
    webs,
    errs,
    baselines,
    le_pred,
    le_emo,
    xgb_device: str,
    xgb_n_jobs: int,
) -> Tuple[int, int]:
    if mode == "second_per_second":
        cfgs = [(h, 1) for h in H_vals]
    else:
        cfgs = [(h, t) for h in H_vals for t in T_vals]

    best = cfgs[0]
    best_score = -1.0
    for H, T in cfgs:
        df = gen_samples(mode, H, T, train_users, wbs, webs, errs, baselines, le_pred, le_emo)
        if df.empty or df["target"].nunique() < 2:
            continue
        X = df[FEATURE_COLS].values.astype(float)
        y = df["target"].values.astype(int)
        g = df["user_id"].values
        scores: List[float] = []
        for tr, va in GroupKFold(n_splits=3).split(X, y, groups=g):
            m = default_xgb(float(y[tr].mean()), xgb_device=xgb_device, xgb_n_jobs=xgb_n_jobs)
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[va])[:, 1]
            scores.append(float(average_precision_score(y[va], p)))
        if not scores:
            continue
        sc = float(np.mean(scores))
        LOG.info("  quick cfg %s H=%s T=%s ap=%.4f", mode, H, T, sc)
        if sc > best_score:
            best_score = sc
            best = (H, T)
    LOG.info("Best cfg %s -> H=%s T=%s ap=%.4f", mode, best[0], best[1], best_score)
    return best


def tune_models(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    groups: np.ndarray,
    search_iter: int,
    xgb_device: str,
    xgb_n_jobs: int,
):
    pos_rate = float(y_tr.mean())
    xgb_base = default_xgb(pos_rate, xgb_device=xgb_device, xgb_n_jobs=xgb_n_jobs)
    xgb_dist = {
        "xgb__n_estimators": [220, 320, 420, 520, 720],
        "xgb__max_depth": [3, 4, 5, 6],
        "xgb__learning_rate": [0.02, 0.03, 0.05, 0.08],
        "xgb__subsample": [0.7, 0.85, 1.0],
        "xgb__colsample_bytree": [0.6, 0.8, 1.0],
        "xgb__min_child_weight": [1, 3, 5, 7],
        "xgb__gamma": [0.0, 0.5, 1.0, 2.0],
        "xgb__reg_lambda": [1.0, 2.0, 5.0, 10.0],
        "xgb__reg_alpha": [0.0, 0.1, 0.5, 1.0],
    }
    xgb_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=xgb_dist,
        n_iter=int(search_iter),
        scoring="average_precision",
        cv=GroupKFold(n_splits=3),
        random_state=RANDOM_SEED,
        n_jobs=1,
        refit=True,
    )
    xgb_search.fit(X_tr, y_tr, groups=groups)
    LOG.info("  tuned xgb cv_ap=%.4f", float(xgb_search.best_score_))

    lr_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    max_iter=3000,
                    solver="liblinear",
                ),
            ),
        ]
    )
    lr_grid = GridSearchCV(
        estimator=lr_pipe,
        param_grid={"lr__C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]},
        scoring="average_precision",
        cv=GroupKFold(n_splits=3),
        n_jobs=1,
        refit=True,
    )
    lr_grid.fit(X_tr, y_tr, groups=groups)
    LOG.info("  tuned logreg cv_ap=%.4f", float(lr_grid.best_score_))
    return {"xgb_tuned": xgb_search.best_estimator_, "logreg_tuned": lr_grid.best_estimator_}


def calibrate_pick(model: Pipeline, X_cal: np.ndarray, y_cal: np.ndarray):
    p_raw = model.predict_proba(X_cal)[:, 1]
    best_name = "none"
    best_model = model
    best_prob = p_raw
    best_ap = float(average_precision_score(y_cal, p_raw))

    for method in ["sigmoid", "isotonic"]:
        cal = CalibratedClassifierCV(model, method=method, cv="prefit")
        cal.fit(X_cal, y_cal)
        p = cal.predict_proba(X_cal)[:, 1]
        ap = float(average_precision_score(y_cal, p))
        if ap > best_ap:
            best_ap = ap
            best_name = method
            best_model = cal
            best_prob = p

    return best_name, best_model, best_prob


def parse_baseline(log_path: str) -> Dict[str, float]:
    def _parse_lines(lines: List[str]) -> Dict[str, float]:
        out_local: Dict[str, float] = {}
        in_cal_local = False
        for line in lines:
            if "Calibrated XGBoost:" in line:
                in_cal_local = True
                continue
            if in_cal_local and (not line.strip() or "Bootstrap" in line):
                break
            if in_cal_local and ":" in line:
                chunk = line.split("INFO]")[-1].strip()
                if ":" not in chunk:
                    continue
                k, v = chunk.split(":", 1)
                try:
                    out_local[k.strip()] = float(v.strip())
                except Exception:
                    continue
        return out_local

    out: Dict[str, float] = {}
    if not os.path.exists(log_path):
        return out

    for enc in ["utf-8", "utf-8-sig", "utf-16", "utf-16le", "latin-1"]:
        try:
            with open(log_path, "r", encoding=enc, errors="ignore") as f:
                lines = [x.rstrip() for x in f]
        except Exception:
            continue
        out = _parse_lines(lines)
        if out:
            return out
    return out


def run_mode(mode: str, args, d: pd.DataFrame, e: pd.DataFrame, eb: pd.DataFrame, db: pd.DataFrame):
    users = sorted(set(d["user_id"].unique()) | set(e["user_id"].unique()))
    train_users, cal_users, test_users = split_users(users)
    LOG.info("[%s] users train=%s cal=%s test=%s", mode, len(train_users), len(cal_users), len(test_users))

    wbs, webs, errs = build_lookups(d, e, users)
    le_pred, le_emo = fit_encoders(train_users, d, e)
    baselines = build_baselines(train_users, d, eb, db)

    h_vals = args.h_second if mode == "second_per_second" else args.h_lookahead
    t_vals = [1] if mode == "second_per_second" else args.t_lookahead
    best_h, best_t = quick_config_search(
        mode,
        h_vals,
        t_vals,
        train_users,
        wbs,
        webs,
        errs,
        baselines,
        le_pred,
        le_emo,
        xgb_device=args.xgb_device_resolved,
        xgb_n_jobs=args.xgb_n_jobs,
    )

    df_tr = gen_samples(mode, best_h, best_t, train_users, wbs, webs, errs, baselines, le_pred, le_emo)
    df_ca = gen_samples(mode, best_h, best_t, cal_users, wbs, webs, errs, baselines, le_pred, le_emo)
    df_te = gen_samples(mode, best_h, best_t, test_users, wbs, webs, errs, baselines, le_pred, le_emo)
    for nm, df in [("train", df_tr), ("cal", df_ca), ("test", df_te)]:
        if df.empty or df["target"].nunique() < 2:
            raise RuntimeError(f"{mode}: invalid {nm} split")
        LOG.info("[%s] %s n=%s pos=%.4f", mode, nm, len(df), float(df["target"].mean()))

    q_low = df_tr[FEATURE_COLS].quantile(0.01)
    q_hi = df_tr[FEATURE_COLS].quantile(0.99)
    df_tr[FEATURE_COLS] = df_tr[FEATURE_COLS].clip(lower=q_low, upper=q_hi, axis=1)
    df_ca[FEATURE_COLS] = df_ca[FEATURE_COLS].clip(lower=q_low, upper=q_hi, axis=1)
    df_te[FEATURE_COLS] = df_te[FEATURE_COLS].clip(lower=q_low, upper=q_hi, axis=1)

    X_tr = df_tr[FEATURE_COLS].values.astype(float)
    y_tr = df_tr["target"].values.astype(int)
    g_tr = df_tr["user_id"].values
    X_ca = df_ca[FEATURE_COLS].values.astype(float)
    y_ca = df_ca["target"].values.astype(int)
    X_te = df_te[FEATURE_COLS].values.astype(float)
    y_te = df_te["target"].values.astype(int)

    candidates = tune_models(
        X_tr,
        y_tr,
        g_tr,
        search_iter=args.search_iter,
        xgb_device=args.xgb_device_resolved,
        xgb_n_jobs=args.xgb_n_jobs,
    )
    best_name = ""
    best_model = None
    best_prob_cal = None
    best_cal_name = ""
    best_cal_ap = -1.0
    for nm, mdl in candidates.items():
        cal_name, cal_model, p_cal = calibrate_pick(mdl, X_ca, y_ca)
        ap = float(average_precision_score(y_ca, p_cal))
        LOG.info("[%s] candidate=%s calib=%s cal_ap=%.4f", mode, nm, cal_name, ap)
        if ap > best_cal_ap:
            best_cal_ap = ap
            best_name = nm
            best_model = cal_model
            best_cal_name = cal_name
            best_prob_cal = p_cal

    thr = pick_threshold_f1(y_ca, best_prob_cal)
    p_te = best_model.predict_proba(X_te)[:, 1]
    met = eval_metrics(y_te, p_te, thr)
    ci_ap = bootstrap_ci(y_te, p_te, average_precision_score, args.bootstrap)
    ci_roc = bootstrap_ci(y_te, p_te, roc_auc_score, args.bootstrap)
    ci_br = bootstrap_ci(y_te, p_te, brier_score_loss, args.bootstrap)

    return {
        "best_config": {"H": int(best_h), "T": int(best_t)},
        "best_candidate": best_name,
        "best_calibration": best_cal_name,
        "threshold": float(thr),
        "metrics_test": met,
        "ci_test": {
            "AUC-PR": {"mean": ci_ap[0], "low": ci_ap[1], "high": ci_ap[2]},
            "AUC-ROC": {"mean": ci_roc[0], "low": ci_roc[1], "high": ci_roc[2]},
            "Brier": {"mean": ci_br[0], "low": ci_br[1], "high": ci_br[2]},
        },
        "samples": {"train": int(len(df_tr)), "cal": int(len(df_ca)), "test": int(len(df_te))},
        "pos_rate": {"train": float(y_tr.mean()), "cal": float(y_ca.mean()), "test": float(y_te.mean())},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="data")
    ap.add_argument("--output-dir", default="evaluation/optimized")
    ap.add_argument("--mode", choices=["second_per_second", "lookahead", "both"], default="both")
    ap.add_argument("--h-second", default="5,6,7,8,9,10,11,12,13,14,15")
    ap.add_argument("--h-lookahead", default="5,6,7,8,9,10,11,12,13,14,15")
    ap.add_argument("--t-lookahead", default="1,2,3,4,5")
    ap.add_argument("--search-iter", type=int, default=24)
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--max-window-seconds", type=float, default=0.0, help="If > 0, drop distraction windows longer than this value (outlier filter).")
    ap.add_argument("--xgb-device", choices=["auto", "cpu", "cuda"], default="auto", help="XGBoost compute device.")
    ap.add_argument("--xgb-n-jobs", type=int, default=-1, help="XGBoost CPU threads (-1 = all).")
    ap.add_argument("--baseline-log-second", default="evaluation/second_per_second_baseline_run.log")
    ap.add_argument("--baseline-log-lookahead", default="evaluation/lookahead_baseline_run.log")
    args = ap.parse_args()

    args.h_second = parse_int_list(args.h_second)
    args.h_lookahead = parse_int_list(args.h_lookahead)
    args.t_lookahead = parse_int_list(args.t_lookahead)
    args.xgb_device_resolved = resolve_xgb_device(args.xgb_device)
    if args.xgb_device_resolved == "cuda" and int(args.xgb_n_jobs) <= 0:
        args.xgb_n_jobs = 1
    LOG.info("XGBoost device: requested=%s resolved=%s n_jobs=%s", args.xgb_device, args.xgb_device_resolved, args.xgb_n_jobs)
    os.makedirs(args.output_dir, exist_ok=True)

    d, e, eb, db = load_data(args.data_path, max_window_seconds=args.max_window_seconds)
    LOG.info("Loaded rows: distractions=%s errors_dist=%s errors_base=%s driving_base=%s", len(d), len(e), len(eb), len(db))

    modes = ["second_per_second", "lookahead"] if args.mode == "both" else [args.mode]
    out = {"runs": {}, "baseline_reference": {}}
    out["baseline_reference"]["second_per_second"] = parse_baseline(args.baseline_log_second)
    out["baseline_reference"]["lookahead"] = parse_baseline(args.baseline_log_lookahead)

    for mode in modes:
        LOG.info("=" * 76)
        LOG.info("RUN MODE: %s", mode)
        out["runs"][mode] = run_mode(mode, args, d, e, eb, db)

    for mode in modes:
        base = out["baseline_reference"]["second_per_second" if mode == "second_per_second" else "lookahead"]
        if base:
            m = out["runs"][mode]["metrics_test"]
            out["runs"][mode]["delta_vs_baseline"] = {
                "AUC-PR": float(m["AUC-PR"] - base.get("AUC-PR", 0.0)),
                "AUC-ROC": float(m["AUC-ROC"] - base.get("AUC-ROC", 0.0)),
                "Brier": float(m["Brier"] - base.get("Brier", 0.0)),
                "F1": float(m["F1"] - base.get("F1", 0.0)),
            }

    out_path = os.path.join(args.output_dir, "optimized_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    LOG.info("Saved %s", out_path)

    for mode in modes:
        m = out["runs"][mode]["metrics_test"]
        LOG.info("[%s] TEST AUC-PR=%.4f AUC-ROC=%.4f Brier=%.4f F1=%.4f", mode, m["AUC-PR"], m["AUC-ROC"], m["Brier"], m["F1"])
        dlt = out["runs"][mode].get("delta_vs_baseline")
        if dlt:
            LOG.info("[%s] DELTA dAUC-PR=%+.4f dAUC-ROC=%+.4f dBrier=%+.4f dF1=%+.4f", mode, dlt["AUC-PR"], dlt["AUC-ROC"], dlt["Brier"], dlt["F1"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
