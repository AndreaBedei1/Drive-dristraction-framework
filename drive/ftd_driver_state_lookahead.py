"""
Driver-state-only Fitness-to-Drive impairment pipeline.

Goal:
Predict P(error in next T seconds) using only driver-state signals around
distraction windows, with hangover horizon H and no data leakage.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from bisect import bisect_left
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
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

LOG = logging.getLogger("ftd_driver_state")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


DRIVER_STATE_DIST_NUMERIC = [
    "arousal_start",
    "arousal_end",
    "hr_bpm_start",
    "hr_bpm_end",
    "model_prob_start",
    "model_prob_end",
    "emotion_prob_start",
    "emotion_prob_end",
]

DRIVER_STATE_LABEL_COLS = [
    "model_pred_start",
    "model_pred_end",
    "emotion_label_start",
    "emotion_label_end",
]

FEATURE_COLS = [
    "time_in_distraction",
    "time_since_distraction_end",
    "within_distraction",
    "hangover_decay",
    "window_progress",
    "window_duration",
    "prev_window_duration",
    "model_prob",
    "model_prob_sq",
    "model_prob_slope_decay",
    "model_pred_enc",
    "emotion_prob",
    "emotion_prob_sq",
    "emotion_prob_slope_decay",
    "emotion_label_enc",
    "arousal",
    "arousal_sq",
    "arousal_delta_baseline",
    "arousal_delta_sq",
    "arousal_slope_decay",
    "hr_bpm",
    "hr_delta_baseline",
    "hr_delta_sq",
    "hr_slope_decay",
    "dist_density_30",
    "dist_density_60",
    "dist_density_120",
    "sensor_missing_flag",
    "baseline_error_rate",
]


def parse_int_list(raw: str) -> List[int]:
    vals = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one integer")
    return sorted(set(vals))


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
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _xgb_cuda_available() -> bool:
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


def load_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = Path(data_path)
    d = pd.read_csv(path / "Dataset Distractions_distraction.csv")
    e = pd.read_csv(path / "Dataset Errors_distraction.csv")
    eb = pd.read_csv(path / "Dataset Errors_baseline.csv")
    db = pd.read_csv(path / "Dataset Driving Time_baseline.csv")

    d["timestamp_start"] = pd.to_datetime(d["timestamp_start"], errors="coerce", format="ISO8601")
    d["timestamp_end"] = pd.to_datetime(d["timestamp_end"], errors="coerce", format="ISO8601")
    e["timestamp"] = pd.to_datetime(e["timestamp"], errors="coerce", format="ISO8601")
    if "timestamp" in db.columns:
        db["timestamp"] = pd.to_datetime(db["timestamp"], errors="coerce", format="ISO8601")

    d = d.dropna(subset=["timestamp_start", "timestamp_end"]).copy()
    d = d[d["timestamp_end"] >= d["timestamp_start"]].copy()
    e = e.dropna(subset=["timestamp"]).copy()

    for col in DRIVER_STATE_LABEL_COLS:
        if col not in d.columns:
            d[col] = UNKNOWN_LABEL
    for col in ["model_pred", "emotion_label"]:
        if col not in e.columns:
            e[col] = UNKNOWN_LABEL
    for col in DRIVER_STATE_LABEL_COLS:
        d[col] = norm_label(d[col])
    for col in ["model_pred", "emotion_label", "error_type"]:
        if col in e.columns:
            e[col] = norm_label(e[col])
    if "error_type" in eb.columns:
        eb["error_type"] = norm_label(eb["error_type"])

    to_numeric(d, DRIVER_STATE_DIST_NUMERIC)
    to_numeric(e, ["model_prob", "emotion_prob"])
    to_numeric(db, ["run_duration_seconds", "hr_baseline", "arousal_baseline"])

    d["model_pred_start"] = d["model_pred_start"].replace(UNKNOWN_LABEL, np.nan)
    d["model_pred_start"] = d["model_pred_start"].fillna(d["model_pred_end"])
    d["model_pred_start"] = norm_label(d["model_pred_start"])

    d["sensor_missing_flag"] = (
        d["arousal_start"].isna()
        | d["arousal_end"].isna()
        | d["hr_bpm_start"].isna()
        | d["hr_bpm_end"].isna()
    ).astype(float)

    for col in ["model_prob_start", "model_prob_end", "emotion_prob_start", "emotion_prob_end", "arousal_start", "arousal_end"]:
        d[col] = d[col].clip(0.0, 1.0)
    for col in ["hr_bpm_start", "hr_bpm_end"]:
        d[col] = d[col].clip(35.0, 220.0)
    for col in ["model_prob", "emotion_prob"]:
        if col in e.columns:
            e[col] = e[col].clip(0.0, 1.0)

    return d.reset_index(drop=True), e.reset_index(drop=True), eb.reset_index(drop=True), db.reset_index(drop=True)


def run_integrity_checks(distractions: pd.DataFrame, errors_dist: pd.DataFrame) -> None:
    issues: List[str] = []
    error_sessions = set(errors_dist[["user_id", "run_id"]].drop_duplicates().itertuples(index=False, name=None))
    dist_sessions = set(distractions[["user_id", "run_id"]].drop_duplicates().itertuples(index=False, name=None))
    missing = error_sessions - dist_sessions
    if missing:
        issues.append(f"errors in sessions without distractions: {len(missing)}")

    for (uid, rid), grp in distractions.groupby(["user_id", "run_id"], sort=False):
        g = grp.sort_values("timestamp_start").reset_index(drop=True)
        if not g["timestamp_start"].is_monotonic_increasing:
            issues.append(f"non-monotonic timestamp_start for {uid}/{rid}")
        overlap = (g["timestamp_start"].iloc[1:].values < g["timestamp_end"].iloc[:-1].values).any()
        if overlap:
            issues.append(f"overlap distraction windows for {uid}/{rid}")

    dup = errors_dist.duplicated(subset=["user_id", "run_id", "timestamp"]).sum()
    if dup:
        issues.append(f"duplicate error timestamps: {dup}")

    if issues:
        for msg in issues:
            LOG.error("integrity_check: %s", msg)
        raise RuntimeError("Data integrity checks failed. Fix data first.")
    LOG.info("integrity_check: PASS")


def split_users(users: Sequence[str], seed: int = RANDOM_SEED) -> Tuple[List[str], List[str], List[str]]:
    arr = np.array(sorted(set(users)))
    if len(arr) < 3:
        raise RuntimeError(f"Need at least 3 users, found {len(arr)}")
    rng = np.random.RandomState(seed)
    rng.shuffle(arr)

    n_total = len(arr)
    n_test = max(1, int(round(0.2 * n_total)))
    n_cal = max(1, int(round(0.2 * (n_total - n_test))))
    if n_test + n_cal >= n_total:
        n_cal = max(1, n_total - n_test - 1)

    test_users = arr[:n_test].tolist()
    cal_users = arr[n_test : n_test + n_cal].tolist()
    train_users = arr[n_test + n_cal :].tolist()
    return train_users, cal_users, test_users


def fit_train_imputation_stats(train_users: Iterable[str], d: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    users = set(train_users)
    dtr = d[d["user_id"].isin(users)].copy()
    stats: Dict[str, Dict[str, float]] = {}
    for col in DRIVER_STATE_DIST_NUMERIC:
        per_user = dtr.groupby("user_id")[col].median().to_dict()
        global_med = float(dtr[col].median()) if len(dtr) else np.nan
        if np.isnan(global_med):
            if "hr" in col:
                global_med = 70.0
            elif "prob" in col or "arousal" in col:
                global_med = 0.5
            else:
                global_med = 0.0
        stats[col] = {"global": float(global_med), "per_user": {str(k): float(v) for k, v in per_user.items() if pd.notna(v)}}
    return stats


def apply_imputation_stats(df: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    out = df.copy()
    for col, st in stats.items():
        if col not in out.columns:
            continue
        mapped = out["user_id"].map(st["per_user"])
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].fillna(mapped).fillna(float(st["global"]))
    for col in DRIVER_STATE_LABEL_COLS:
        if col in out.columns:
            out[col] = norm_label(out[col])
    return out


def fit_label_encoders(train_users: Iterable[str], d: pd.DataFrame, e: pd.DataFrame) -> Tuple[LabelEncoder, LabelEncoder]:
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

    le_pred = LabelEncoder()
    le_pred.fit(pred_vocab)
    le_emo = LabelEncoder()
    le_emo.fit(emo_vocab)
    return le_pred, le_emo


def build_baselines(train_users: Iterable[str], eb: pd.DataFrame, db: pd.DataFrame) -> Dict:
    users = set(train_users)
    eb_tr = eb[eb["user_id"].isin(users)].copy()
    db_tr = db[db["user_id"].isin(users)].copy()

    ar_user = db_tr.groupby("user_id")["arousal_baseline"].median().to_dict() if "arousal_baseline" in db_tr.columns else {}
    hr_user = db_tr.groupby("user_id")["hr_baseline"].median().to_dict() if "hr_baseline" in db_tr.columns else {}
    ar_global = float(db_tr["arousal_baseline"].median()) if "arousal_baseline" in db_tr.columns and len(db_tr) else 0.5
    hr_global = float(db_tr["hr_baseline"].median()) if "hr_baseline" in db_tr.columns and len(db_tr) else 70.0
    if not np.isfinite(ar_global):
        ar_global = 0.5
    if not np.isfinite(hr_global):
        hr_global = 70.0

    if "run_duration_seconds" in db_tr.columns and len(db_tr):
        sec_user = db_tr.groupby("user_id")["run_duration_seconds"].sum().to_dict()
    else:
        sec_user = {}
    err_user = eb_tr.groupby("user_id").size().to_dict() if len(eb_tr) else {}

    rate_user: Dict[str, float] = {}
    for uid in users:
        ecount = float(err_user.get(uid, 0.0))
        secs = float(sec_user.get(uid, 0.0))
        rate_user[str(uid)] = float((ecount + 1.0) / (secs + 60.0))

    if rate_user:
        rate_global = float(np.median(list(rate_user.values())))
    else:
        rate_global = 1.0 / 600.0

    return {
        "user_arousal": {str(k): float(v) for k, v in ar_user.items() if pd.notna(v)},
        "user_hr": {str(k): float(v) for k, v in hr_user.items() if pd.notna(v)},
        "user_err_rate": rate_user,
        "global_arousal": float(ar_global),
        "global_hr": float(hr_global),
        "global_err_rate": float(rate_global),
    }


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


def density(events: List[pd.Timestamp], ts: pd.Timestamp, lookback_s: int) -> int:
    lo = ts - pd.Timedelta(seconds=int(lookback_s))
    i = bisect_left(events, lo)
    j = bisect_left(events, ts)
    return int(max(0, j - i))


def build_encoding_map(le: LabelEncoder) -> Dict[str, int]:
    return {str(c): int(i) for i, c in enumerate(le.classes_)}


def safe_encode(label: str, enc_map: Dict[str, int]) -> float:
    k = str(label).strip()
    if not k:
        k = UNKNOWN_LABEL
    return float(enc_map.get(k, enc_map.get(UNKNOWN_LABEL, 0)))


def generate_samples(
    H: int,
    T: int,
    users: Iterable[str],
    wbs: Dict[Tuple[str, int], pd.DataFrame],
    webs: Dict[Tuple[str, int], List[pd.Timestamp]],
    errs: Dict[Tuple[str, int], List[pd.Timestamp]],
    baselines: Dict,
    pred_enc: Dict[str, int],
    emo_enc: Dict[str, int],
    global_model_prob: float,
    global_emotion_prob: float,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    users_set = set(users)
    H = max(1, int(H))
    T = max(1, int(T))
    horizon = pd.Timedelta(seconds=T)

    for key, wins in wbs.items():
        uid, rid = key
        if uid not in users_set or wins.empty:
            continue

        err_ts = errs.get(key, [])
        usr_ar = float(baselines["user_arousal"].get(uid, baselines["global_arousal"]))
        usr_hr = float(baselines["user_hr"].get(uid, baselines["global_hr"]))
        usr_er = float(baselines["user_err_rate"].get(uid, baselines["global_err_rate"]))

        for i in range(len(wins)):
            row = wins.iloc[i]
            start_ts = row["timestamp_start"]
            end_ts = row["timestamp_end"]
            win_dur = float((end_ts - start_ts).total_seconds())
            if not np.isfinite(win_dur) or win_dur <= 0:
                continue

            next_start = wins.iloc[i + 1]["timestamp_start"] if i + 1 < len(wins) else None
            hang = float(H)
            if next_start is not None:
                hang = min(hang, float((next_start - end_ts).total_seconds()))
            hang = max(0.0, hang)

            upper_ts = end_ts + pd.Timedelta(seconds=hang)
            span_s = float((upper_ts - start_ts).total_seconds())
            n_bins = int(math.ceil(max(0.0, span_s)))
            if n_bins <= 0:
                continue

            s_ar, e_ar = float(row["arousal_start"]), float(row["arousal_end"])
            s_hr, e_hr = float(row["hr_bpm_start"]), float(row["hr_bpm_end"])
            s_mp, e_mp = float(row["model_prob_start"]), float(row["model_prob_end"])
            s_ep, e_ep = float(row["emotion_prob_start"]), float(row["emotion_prob_end"])

            s_pr = safe_encode(row["model_pred_start"], pred_enc)
            e_pr = safe_encode(row["model_pred_end"], pred_enc)
            s_em = safe_encode(row["emotion_label_start"], emo_enc)
            e_em = safe_encode(row["emotion_label_end"], emo_enc)

            ar_slope = (e_ar - s_ar) / max(win_dur, 1e-6)
            hr_slope = (e_hr - s_hr) / max(win_dur, 1e-6)
            mp_slope = (e_mp - s_mp) / max(win_dur, 1e-6)
            ep_slope = (e_ep - s_ep) / max(win_dur, 1e-6)

            prev_win_dur = 0.0
            if i > 0:
                p = wins.iloc[i - 1]
                prev_win_dur = float((p["timestamp_end"] - p["timestamp_start"]).total_seconds())
                if not np.isfinite(prev_win_dur) or prev_win_dur < 0:
                    prev_win_dur = 0.0

            sensor_missing = float(row.get("sensor_missing_flag", 0.0))

            for off in range(n_bins):
                ts = start_ts + pd.Timedelta(seconds=off)
                inside = float(off < win_dur)
                if inside:
                    alpha = float(np.clip(off / max(win_dur, 1e-6), 0.0, 1.0))
                    cur_ar = s_ar + (e_ar - s_ar) * alpha
                    cur_hr = s_hr + (e_hr - s_hr) * alpha
                    cur_mp = s_mp + (e_mp - s_mp) * alpha
                    cur_ep = s_ep + (e_ep - s_ep) * alpha
                    cur_pr = s_pr + (e_pr - s_pr) * alpha
                    cur_em = s_em + (e_em - s_em) * alpha
                    t_in = float(off)
                    t_after = 0.0
                    dec = 1.0
                    progress = alpha
                    phase_sec = float(off - win_dur)
                else:
                    t_after = float(off - win_dur)
                    dec = float(np.exp(-t_after / max(float(H), 1e-6)))
                    cur_ar = usr_ar + (e_ar - usr_ar) * dec
                    cur_hr = usr_hr + (e_hr - usr_hr) * dec
                    cur_mp = global_model_prob + (e_mp - global_model_prob) * dec
                    cur_ep = global_emotion_prob + (e_ep - global_emotion_prob) * dec
                    cur_pr = e_pr
                    cur_em = e_em
                    t_in = 0.0
                    progress = 1.0
                    phase_sec = t_after

                j = bisect_left(err_ts, ts)
                target = 1 if j < len(err_ts) and err_ts[j] < ts + horizon else 0

                ar_dev = cur_ar - usr_ar
                hr_dev = cur_hr - usr_hr

                rows.append(
                    {
                        "user_id": uid,
                        "run_id": int(rid),
                        "target": int(target),
                        "phase_sec": float(phase_sec),
                        "time_in_distraction": t_in,
                        "time_since_distraction_end": t_after,
                        "within_distraction": inside,
                        "hangover_decay": float(dec),
                        "window_progress": float(progress),
                        "window_duration": float(win_dur),
                        "prev_window_duration": float(prev_win_dur),
                        "model_prob": float(cur_mp),
                        "model_prob_sq": float(cur_mp * cur_mp),
                        "model_prob_slope_decay": float(mp_slope * dec),
                        "model_pred_enc": float(cur_pr),
                        "emotion_prob": float(cur_ep),
                        "emotion_prob_sq": float(cur_ep * cur_ep),
                        "emotion_prob_slope_decay": float(ep_slope * dec),
                        "emotion_label_enc": float(cur_em),
                        "arousal": float(cur_ar),
                        "arousal_sq": float(cur_ar * cur_ar),
                        "arousal_delta_baseline": float(ar_dev),
                        "arousal_delta_sq": float(ar_dev * ar_dev),
                        "arousal_slope_decay": float(ar_slope * dec),
                        "hr_bpm": float(cur_hr),
                        "hr_delta_baseline": float(hr_dev),
                        "hr_delta_sq": float(hr_dev * hr_dev),
                        "hr_slope_decay": float(hr_slope * dec),
                        "dist_density_30": float(density(webs.get(key, []), ts, 30)),
                        "dist_density_60": float(density(webs.get(key, []), ts, 60)),
                        "dist_density_120": float(density(webs.get(key, []), ts, 120)),
                        "sensor_missing_flag": float(sensor_missing),
                        "baseline_error_rate": float(usr_er),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def fit_feature_postprocess(df_train: pd.DataFrame) -> Dict[str, pd.Series]:
    med = df_train[FEATURE_COLS].median()
    q_lo = df_train[FEATURE_COLS].quantile(0.01)
    q_hi = df_train[FEATURE_COLS].quantile(0.99)
    return {"median": med, "q_lo": q_lo, "q_hi": q_hi}


def apply_feature_postprocess(df: pd.DataFrame, pp: Dict[str, pd.Series]) -> pd.DataFrame:
    out = df.copy()
    out[FEATURE_COLS] = out[FEATURE_COLS].fillna(pp["median"])
    out[FEATURE_COLS] = out[FEATURE_COLS].clip(lower=pp["q_lo"], upper=pp["q_hi"], axis=1)
    out[FEATURE_COLS] = out[FEATURE_COLS].fillna(pp["median"])
    return out


def min_group_splits(groups: np.ndarray, max_splits: int = 4) -> int:
    return max(2, min(int(max_splits), int(len(np.unique(groups)))))


def default_xgb(pos_rate: float, xgb_device: str, xgb_n_jobs: int, lite: bool = False):
    spw = max((1.0 - pos_rate) / max(pos_rate, 1e-6), 1.0)
    params = dict(
        objective="binary:logistic",
        random_state=RANDOM_SEED,
        eval_metric="logloss",
        tree_method="hist",
        device=str(xgb_device),
        n_jobs=int(xgb_n_jobs),
        scale_pos_weight=spw,
        n_estimators=120 if lite else 320,
        max_depth=3 if lite else 4,
        learning_rate=0.07 if lite else 0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=2.0,
        reg_alpha=0.1,
    )
    return xgb.XGBClassifier(**params)


def quick_config_search(
    H_vals: Sequence[int],
    T_vals: Sequence[int],
    train_users: Sequence[str],
    wbs,
    webs,
    errs,
    baselines,
    pred_enc,
    emo_enc,
    global_model_prob: float,
    global_emotion_prob: float,
    xgb_device: str,
    xgb_n_jobs: int,
    tolerance_ap: float,
) -> Tuple[int, int, pd.DataFrame]:
    records: List[Dict[str, float]] = []
    for H in H_vals:
        for T in T_vals:
            df = generate_samples(H, T, train_users, wbs, webs, errs, baselines, pred_enc, emo_enc, global_model_prob, global_emotion_prob)
            if df.empty or df["target"].nunique() < 2:
                records.append({"H": int(H), "T": int(T), "ap_mean": np.nan, "ap_std": np.nan, "brier_mean": np.nan, "n": int(len(df))})
                continue

            X = df[FEATURE_COLS].values.astype(float)
            y = df["target"].values.astype(int)
            g = df["user_id"].values

            ap_scores: List[float] = []
            br_scores: List[float] = []
            n_splits = min_group_splits(g, max_splits=4)
            for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups=g):
                if np.unique(y[tr]).size < 2 or np.unique(y[va]).size < 2:
                    continue
                tr_df = df.iloc[tr].copy()
                va_df = df.iloc[va].copy()
                pp = fit_feature_postprocess(tr_df)
                tr_df = apply_feature_postprocess(tr_df, pp)
                va_df = apply_feature_postprocess(va_df, pp)
                X_tr = tr_df[FEATURE_COLS].values.astype(float)
                y_tr = tr_df["target"].values.astype(int)
                X_va = va_df[FEATURE_COLS].values.astype(float)
                y_va = va_df["target"].values.astype(int)

                mdl = default_xgb(float(y_tr.mean()), xgb_device=xgb_device, xgb_n_jobs=xgb_n_jobs, lite=True)
                mdl.fit(X_tr, y_tr)
                p = mdl.predict_proba(X_va)[:, 1]
                ap_scores.append(float(average_precision_score(y_va, p)))
                br_scores.append(float(brier_score_loss(y_va, p)))

            if ap_scores:
                ap_mean = float(np.mean(ap_scores))
                ap_std = float(np.std(ap_scores))
                br_mean = float(np.mean(br_scores))
            else:
                ap_mean = np.nan
                ap_std = np.nan
                br_mean = np.nan

            records.append({"H": int(H), "T": int(T), "ap_mean": ap_mean, "ap_std": ap_std, "brier_mean": br_mean, "n": int(len(df))})
            LOG.info("quick_cfg H=%s T=%s n=%s ap=%.4f brier=%.4f", H, T, len(df), ap_mean, br_mean)

    results = pd.DataFrame(records)
    valid = results[np.isfinite(results["ap_mean"])].copy()
    if valid.empty:
        raise RuntimeError("No valid (H, T) configuration produced trainable data.")

    best_ap = float(valid["ap_mean"].max())
    candidates = valid[valid["ap_mean"] >= best_ap - float(tolerance_ap)].copy()
    chosen = candidates.sort_values(["T", "ap_mean", "H"], ascending=[True, False, True]).iloc[0]
    best_h = int(chosen["H"])
    best_t = int(chosen["T"])
    LOG.info("chosen_config H=%s T=%s ap=%.4f (best_ap=%.4f tol=%.4f)", best_h, best_t, float(chosen["ap_mean"]), best_ap, float(tolerance_ap))
    return best_h, best_t, results.sort_values(["H", "T"]).reset_index(drop=True)


def tune_models(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    groups: np.ndarray,
    search_iter: int,
    xgb_device: str,
    xgb_n_jobs: int,
):
    pos_rate = float(y_tr.mean())
    n_splits = min_group_splits(groups, max_splits=4)

    xgb_base = default_xgb(pos_rate=pos_rate, xgb_device=xgb_device, xgb_n_jobs=xgb_n_jobs, lite=False)
    xgb_dist = {
        "n_estimators": [220, 320, 420, 520, 720],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.02, 0.03, 0.05, 0.08],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0.0, 0.5, 1.0, 2.0],
        "reg_lambda": [1.0, 2.0, 5.0, 10.0],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
    }
    xgb_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=xgb_dist,
        n_iter=int(search_iter),
        scoring="average_precision",
        cv=GroupKFold(n_splits=n_splits),
        random_state=RANDOM_SEED,
        n_jobs=1,
        refit=True,
    )
    xgb_search.fit(X_tr, y_tr, groups=groups)
    LOG.info("tuned_xgb cv_ap=%.4f", float(xgb_search.best_score_))

    lr_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    max_iter=5000,
                    solver="liblinear",
                ),
            ),
        ]
    )
    lr_grid = GridSearchCV(
        estimator=lr_pipe,
        param_grid={"model__C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]},
        scoring="average_precision",
        cv=GroupKFold(n_splits=n_splits),
        n_jobs=1,
        refit=True,
    )
    lr_grid.fit(X_tr, y_tr, groups=groups)
    LOG.info("tuned_logreg cv_ap=%.4f", float(lr_grid.best_score_))
    return {"xgb_tuned": xgb_search.best_estimator_, "logreg_tuned": lr_grid.best_estimator_}


def select_threshold_f1(y_true: np.ndarray, prob: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y_true, prob)
    if len(thr) == 0:
        return 0.5
    f1 = 2.0 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    idx = int(np.nanargmax(f1))
    return float(thr[idx])


def eval_metrics(y_true: np.ndarray, prob: np.ndarray, thr: float) -> Dict[str, float]:
    y_hat = (prob >= float(thr)).astype(int)
    out = {
        "AUC-PR": float(average_precision_score(y_true, prob)),
        "AUC-ROC": float(roc_auc_score(y_true, prob)),
        "Brier": float(brier_score_loss(y_true, prob)),
        "LogLoss": float(log_loss(y_true, np.clip(prob, 1e-12, 1 - 1e-12))),
        "F1": float(f1_score(y_true, y_hat)),
        "Precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "Recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_hat)),
        "Kappa": float(cohen_kappa_score(y_true, y_hat)),
    }
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, prob)
    for level in RECALL_LEVELS:
        valid = prec_curve[:-1][rec_curve[:-1] >= level]
        out[f"Precision@Recall>={level:.2f}"] = float(valid.max()) if len(valid) else 0.0
    return out


def bootstrap_ci(y_true: np.ndarray, prob: np.ndarray, metric_fn, n_boot: int, seed: int = RANDOM_SEED) -> Tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    idx = np.arange(len(y_true))
    vals = []
    for _ in range(int(n_boot)):
        b = rng.choice(idx, size=len(idx), replace=True)
        yb = y_true[b]
        if len(np.unique(yb)) < 2:
            continue
        try:
            vals.append(float(metric_fn(yb, prob[b])))
        except Exception:
            continue
    if not vals:
        return (float("nan"), float("nan"), float("nan"))
    arr = np.array(vals, dtype=float)
    return float(np.mean(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def calibrate_candidate(model, X_cal: np.ndarray, y_cal: np.ndarray):
    candidates = []
    p_raw = model.predict_proba(X_cal)[:, 1]
    candidates.append(("none", model, p_raw))
    for method in ["sigmoid", "isotonic"]:
        try:
            cal = CalibratedClassifierCV(estimator=model, method=method, cv="prefit")
            cal.fit(X_cal, y_cal)
            p = cal.predict_proba(X_cal)[:, 1]
            candidates.append((method, cal, p))
        except Exception as exc:
            LOG.warning("calibration_failed method=%s reason=%s", method, exc)

    best = None
    best_key = None
    for name, mdl, p in candidates:
        ap = float(average_precision_score(y_cal, p))
        brier = float(brier_score_loss(y_cal, p))
        key = (ap, -brier)
        if best is None or key > best_key:
            best = (name, mdl, p, ap, brier)
            best_key = key
    assert best is not None
    return best


def unwrap_estimator(model):
    if isinstance(model, CalibratedClassifierCV):
        return model.estimator
    return model


def plot_ht_heatmaps(results: pd.DataFrame, best_h: int, best_t: int, out_path: Path) -> None:
    valid = results.copy()
    ap_pivot = valid.pivot(index="H", columns="T", values="ap_mean")
    br_pivot = valid.pivot(index="H", columns="T", values="brier_mean")
    if ap_pivot.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title, cmap in [
        (axes[0], ap_pivot, "AUC-PR (CV mean)", "viridis"),
        (axes[1], br_pivot, "Brier (CV mean)", "magma_r"),
    ]:
        arr = data.values.astype(float)
        im = ax.imshow(arr, aspect="auto", cmap=cmap)
        ax.set_xticks(np.arange(len(data.columns)))
        ax.set_xticklabels([str(x) for x in data.columns])
        ax.set_yticks(np.arange(len(data.index)))
        ax.set_yticklabels([str(x) for x in data.index])
        ax.set_xlabel("T (seconds)")
        ax.set_ylabel("H (seconds)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.85)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, color="white")

    if best_h in ap_pivot.index and best_t in ap_pivot.columns:
        bi = int(np.where(ap_pivot.index.values == best_h)[0][0])
        bj = int(np.where(ap_pivot.columns.values == best_t)[0][0])
        axes[0].plot([bj], [bi], marker="o", markersize=12, markerfacecolor="none", markeredgecolor="cyan", markeredgewidth=2)
        axes[1].plot([bj], [bi], marker="o", markersize=12, markerfacecolor="none", markeredgecolor="cyan", markeredgewidth=2)

    fig.suptitle(f"H/T search heatmaps (selected H={best_h}, T={best_t})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_roc_pr(y_true: np.ndarray, prob: np.ndarray, out_path: Path, title_suffix: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    RocCurveDisplay.from_predictions(y_true, prob, ax=ax1, name="Model")
    ax1.plot([0, 1], [0, 1], "k--", label="chance")
    ax1.set_title(f"ROC {title_suffix}")
    ax1.legend()
    PrecisionRecallDisplay.from_predictions(y_true, prob, ax=ax2)
    ax2.set_title(f"Precision-Recall {title_suffix}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_calibration(y_true: np.ndarray, prob_raw: np.ndarray, prob_cal: np.ndarray, out_path: Path, title_suffix: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for prob, label in [(prob_raw, "raw"), (prob_cal, "calibrated")]:
        fop, mpv = calibration_curve(y_true, prob, n_bins=10)
        ax.plot(mpv, fop, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title(f"Calibration {title_suffix}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_risk_profile(df_test: pd.DataFrame, prob_test: np.ndarray, best_h: int, best_t: int, out_path: Path) -> None:
    dfp = df_test.copy()
    dfp["pred"] = prob_test
    dfp["phase_bin"] = np.round(dfp["phase_sec"]).astype(int)
    lo = int(max(-10, np.floor(dfp["phase_bin"].min())))
    hi = int(min(best_h, np.ceil(dfp["phase_bin"].max())))
    dfp = dfp[(dfp["phase_bin"] >= lo) & (dfp["phase_bin"] <= hi)].copy()
    if dfp.empty:
        return

    grp = dfp.groupby("phase_bin").agg(target_rate=("target", "mean"), pred_rate=("pred", "mean"), n=("target", "size")).reset_index()
    grp = grp[grp["n"] >= 25]
    if grp.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grp["phase_bin"], grp["target_rate"], "o-", label=f"Empirical P(error in next {best_t}s)")
    ax.plot(grp["phase_bin"], grp["pred_rate"], "s-", label="Predicted impairment probability")
    ax.axvline(0, color="k", linestyle="--", linewidth=1, label="Distraction end")
    ax.set_xlabel("Seconds relative to distraction end (negative: during distraction)")
    ax.set_ylabel("Probability")
    ax.set_title(f"Risk profile around distraction end (H={best_h}, T={best_t})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_feature_importance(model, out_path: Path) -> None:
    est = unwrap_estimator(model)
    names = FEATURE_COLS
    values = None

    if hasattr(est, "feature_importances_"):
        values = np.asarray(est.feature_importances_, dtype=float)
    elif isinstance(est, Pipeline) and "model" in est.named_steps:
        inner = est.named_steps["model"]
        if hasattr(inner, "coef_"):
            values = np.abs(np.asarray(inner.coef_, dtype=float).reshape(-1))
    if values is None or len(values) != len(names):
        return

    order = np.argsort(values)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(np.arange(len(order)), values[order])
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([names[i] for i in order])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Feature importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def run_pipeline(args) -> Dict:
    d_raw, e_raw, eb_raw, db_raw = load_data(args.data_path)
    run_integrity_checks(d_raw, e_raw)

    users = sorted(set(d_raw["user_id"].unique()) | set(e_raw["user_id"].unique()))
    train_users, cal_users, test_users = split_users(users, seed=args.seed)
    LOG.info("users: train=%s cal=%s test=%s", len(train_users), len(cal_users), len(test_users))

    impute_stats = fit_train_imputation_stats(train_users, d_raw)
    d = apply_imputation_stats(d_raw, impute_stats)

    le_pred, le_emo = fit_label_encoders(train_users, d, e_raw)
    pred_enc = build_encoding_map(le_pred)
    emo_enc = build_encoding_map(le_emo)
    baselines = build_baselines(train_users, eb_raw, db_raw)

    d_train = d[d["user_id"].isin(set(train_users))]
    global_model_prob = float(d_train[["model_prob_start", "model_prob_end"]].stack().median()) if len(d_train) else 0.5
    global_emotion_prob = float(d_train[["emotion_prob_start", "emotion_prob_end"]].stack().median()) if len(d_train) else 0.5
    if not np.isfinite(global_model_prob):
        global_model_prob = 0.5
    if not np.isfinite(global_emotion_prob):
        global_emotion_prob = 0.5

    all_users = sorted(set(users))
    wbs, webs, errs = build_lookups(d, e_raw, all_users)

    best_h, best_t, search_df = quick_config_search(
        H_vals=args.h_values,
        T_vals=args.t_values,
        train_users=train_users,
        wbs=wbs,
        webs=webs,
        errs=errs,
        baselines=baselines,
        pred_enc=pred_enc,
        emo_enc=emo_enc,
        global_model_prob=global_model_prob,
        global_emotion_prob=global_emotion_prob,
        xgb_device=args.xgb_device_resolved,
        xgb_n_jobs=args.xgb_n_jobs,
        tolerance_ap=args.t_parsimony_tol,
    )

    df_tr = generate_samples(best_h, best_t, train_users, wbs, webs, errs, baselines, pred_enc, emo_enc, global_model_prob, global_emotion_prob)
    df_ca = generate_samples(best_h, best_t, cal_users, wbs, webs, errs, baselines, pred_enc, emo_enc, global_model_prob, global_emotion_prob)
    df_te = generate_samples(best_h, best_t, test_users, wbs, webs, errs, baselines, pred_enc, emo_enc, global_model_prob, global_emotion_prob)

    for name, df in [("train", df_tr), ("cal", df_ca), ("test", df_te)]:
        if df.empty or df["target"].nunique() < 2:
            raise RuntimeError(f"{name} split has insufficient data/class diversity.")
        LOG.info("%s samples=%s pos_rate=%.4f", name, len(df), float(df["target"].mean()))

    pp = fit_feature_postprocess(df_tr)
    df_tr = apply_feature_postprocess(df_tr, pp)
    df_ca = apply_feature_postprocess(df_ca, pp)
    df_te = apply_feature_postprocess(df_te, pp)

    X_tr = df_tr[FEATURE_COLS].values.astype(float)
    y_tr = df_tr["target"].values.astype(int)
    g_tr = df_tr["user_id"].values
    X_ca = df_ca[FEATURE_COLS].values.astype(float)
    y_ca = df_ca["target"].values.astype(int)
    X_te = df_te[FEATURE_COLS].values.astype(float)
    y_te = df_te["target"].values.astype(int)

    models = tune_models(
        X_tr=X_tr,
        y_tr=y_tr,
        groups=g_tr,
        search_iter=args.search_iter,
        xgb_device=args.xgb_device_resolved,
        xgb_n_jobs=args.xgb_n_jobs,
    )

    model_selection = []
    best_pack = None
    for name, model in models.items():
        cal_name, cal_model, p_cal, ap_cal, brier_cal = calibrate_candidate(model, X_ca, y_ca)
        rec = {
            "model": name,
            "calibration": cal_name,
            "ap_cal": float(ap_cal),
            "brier_cal": float(brier_cal),
        }
        model_selection.append(rec)
        key = (ap_cal, -brier_cal)
        if best_pack is None or key > best_pack["key"]:
            best_pack = {
                "key": key,
                "name": name,
                "calibration": cal_name,
                "model": cal_model,
                "p_cal": p_cal,
            }

    assert best_pack is not None
    chosen_model = best_pack["model"]
    chosen_name = str(best_pack["name"])
    chosen_cal = str(best_pack["calibration"])
    p_cal = np.asarray(best_pack["p_cal"], dtype=float)
    threshold = select_threshold_f1(y_ca, p_cal)

    chosen_base = models[chosen_name]
    p_te_raw = chosen_base.predict_proba(X_te)[:, 1]
    p_te = chosen_model.predict_proba(X_te)[:, 1]

    metrics_test = eval_metrics(y_te, p_te, threshold)
    ci_ap = bootstrap_ci(y_te, p_te, average_precision_score, n_boot=args.bootstrap)
    ci_roc = bootstrap_ci(y_te, p_te, roc_auc_score, n_boot=args.bootstrap)
    ci_brier = bootstrap_ci(y_te, p_te, brier_score_loss, n_boot=args.bootstrap)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_ht_heatmaps(search_df, best_h, best_t, output_dir / "ht_search_heatmaps.png")
    plot_roc_pr(y_te, p_te, output_dir / "roc_pr_curves.png", f"(H={best_h}, T={best_t})")
    plot_calibration(y_te, p_te_raw, p_te, output_dir / "calibration_curve.png", f"(H={best_h}, T={best_t})")
    plot_risk_profile(df_te, p_te, best_h, best_t, output_dir / "risk_profile.png")
    plot_feature_importance(chosen_model, output_dir / "feature_importance.png")

    artifact = {
        "model": chosen_model,
        "base_model": chosen_base,
        "feature_postprocess": pp,
        "label_encoders": {"model_pred": le_pred, "emotion": le_emo},
        "encoding_maps": {"model_pred": pred_enc, "emotion": emo_enc},
        "impute_stats": impute_stats,
        "baselines": baselines,
        "feature_cols": FEATURE_COLS,
        "best_config": {"H": int(best_h), "T": int(best_t)},
        "threshold": float(threshold),
        "global_driver_state_defaults": {
            "model_prob": float(global_model_prob),
            "emotion_prob": float(global_emotion_prob),
        },
    }
    joblib.dump(artifact, output_dir / "driver_state_model.joblib")

    search_records = []
    for rec in search_df.to_dict(orient="records"):
        out_rec = {}
        for k, v in rec.items():
            if isinstance(v, (np.floating, float)):
                out_rec[k] = float(v) if np.isfinite(v) else None
            elif isinstance(v, (np.integer, int)):
                out_rec[k] = int(v)
            else:
                out_rec[k] = v
        search_records.append(out_rec)

    result = {
        "constraints": {
            "driver_state_only": True,
            "excluded_sources": [
                "speed_kmh",
                "steer_angle_deg",
                "map_name",
                "weather",
                "frame",
                "sim_time_seconds",
                "x/y/z position",
                "road_id",
                "lane_id",
            ],
            "leakage_controls": [
                "user-level train/cal/test split",
                "train-only imputation",
                "train-only encoders",
                "train-only feature clipping",
                "H/T search on train only",
                "calibration + threshold on calibration only",
                "single locked test evaluation",
            ],
        },
        "device": {
            "xgb_requested": args.xgb_device,
            "xgb_resolved": args.xgb_device_resolved,
            "xgb_n_jobs": int(args.xgb_n_jobs),
        },
        "users": {
            "train": train_users,
            "cal": cal_users,
            "test": test_users,
        },
        "best_config": {"H": int(best_h), "T": int(best_t)},
        "best_model": {
            "name": chosen_name,
            "calibration": chosen_cal,
            "threshold_f1_on_cal": float(threshold),
        },
        "model_selection_on_cal": model_selection,
        "samples": {"train": int(len(df_tr)), "cal": int(len(df_ca)), "test": int(len(df_te))},
        "positive_rate": {
            "train": float(y_tr.mean()),
            "cal": float(y_ca.mean()),
            "test": float(y_te.mean()),
        },
        "metrics_test": metrics_test,
        "ci_test": {
            "AUC-PR": {"mean": ci_ap[0], "low": ci_ap[1], "high": ci_ap[2]},
            "AUC-ROC": {"mean": ci_roc[0], "low": ci_roc[1], "high": ci_roc[2]},
            "Brier": {"mean": ci_brier[0], "low": ci_brier[1], "high": ci_brier[2]},
        },
        "search_results": search_records,
        "feature_cols": FEATURE_COLS,
    }

    with open(output_dir / "driver_state_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Driver-state-only lookahead impairment predictor")
    ap.add_argument("--data-path", default="data")
    ap.add_argument("--output-dir", default="evaluation/driver_state_lookahead")
    ap.add_argument("--h-values", default="4,6,8,10,12,15,18")
    ap.add_argument("--t-values", default="1,2,3,4,5")
    ap.add_argument("--t-parsimony-tol", type=float, default=0.05, help="Allow AP drop up to this value to prefer smaller T.")
    ap.add_argument("--search-iter", type=int, default=20, help="RandomizedSearchCV iterations for final model tuning.")
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--xgb-device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--xgb-n-jobs", type=int, default=-1, help="XGBoost CPU threads (-1 = all).")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    args.h_values = parse_int_list(args.h_values)
    args.t_values = parse_int_list(args.t_values)
    args.xgb_device_resolved = resolve_xgb_device(args.xgb_device)
    if args.xgb_device_resolved == "cuda" and int(args.xgb_n_jobs) <= 0:
        args.xgb_n_jobs = 1
    LOG.info(
        "xgb device: requested=%s resolved=%s n_jobs=%s",
        args.xgb_device,
        args.xgb_device_resolved,
        args.xgb_n_jobs,
    )

    result = run_pipeline(args)
    m = result["metrics_test"]
    LOG.info(
        "FINAL TEST: AUC-PR=%.4f AUC-ROC=%.4f Brier=%.4f F1=%.4f | H=%s T=%s model=%s/%s",
        m["AUC-PR"],
        m["AUC-ROC"],
        m["Brier"],
        m["F1"],
        result["best_config"]["H"],
        result["best_config"]["T"],
        result["best_model"]["name"],
        result["best_model"]["calibration"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
