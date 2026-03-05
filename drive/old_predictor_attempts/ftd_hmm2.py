import argparse
import json
import logging
import math
from bisect import bisect_left
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils import * 

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
LOG = logging.getLogger("impairment_hmm")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ------------------------------------------------------------------------------
# Constants (defaults)
# ------------------------------------------------------------------------------
RANDOM_SEED = 42
UNKNOWN_LABEL = "unknown"
BWD_FLOOR = 1e-10
FTD_IMPAIRED = 0.50
FTD_CAUTION = 0.25

# ------------------------------------------------------------------------------
# Feature set for production: physiological signals only (no direct time)
# ------------------------------------------------------------------------------
BASE_PHYSIO_FEATURES = [
    "hr_bpm", 
    "arousal", "model_prob", "emotion_prob",
    "hr_delta_baseline", "arousal_delta_baseline", "hr_delta_sq",
    "arousal_hr_coupling", "model_arousal_coupling", #"emotion_hr_coupling",
    "state_energy", "state_imbalance",
    "model_emotion_gap_abs", "model_emotion_product",
    "model_pred_enc", "emotion_label_enc",
    "signal_diff_abs_energy",
    "dist_density_5",
    "baseline_error_rate", "sensor_missing_flag",
    "distraction_intensity",
]

DRIVER_STATE_DIST_NUMERIC = [
    "arousal_start", "arousal_end", "hr_bpm_start", "hr_bpm_end",
    "model_prob_start", "model_prob_end", "emotion_prob_start", "emotion_prob_end",
]
DRIVER_STATE_LABEL_COLS = [
    "model_pred_start", "model_pred_end", "emotion_label_start", "emotion_label_end",
]

BASE_FEATURE_COLS = [
    "model_prob", "model_pred_enc",
    "emotion_prob", "emotion_label_enc",
    "arousal", "arousal_delta_baseline",
    "hr_bpm", "hr_delta_baseline", "hr_delta_sq",
    "dist_density_5",
    "sensor_missing_flag", "baseline_error_rate",
    "distraction_intensity",
]
ROLL_SOURCE_COLS = ["model_prob", "emotion_prob", "arousal", "hr_bpm"]
ROLL_FEATURE_COLS = (
    [f"{col}_diff1_abs" for col in ROLL_SOURCE_COLS]
    + ["state_energy", "state_imbalance", "model_emotion_gap_abs",
       "model_emotion_product", "arousal_hr_coupling", "model_arousal_coupling",
       #"emotion_hr_coupling", 
       "signal_diff_abs_energy"]
)
FEATURE_COLS = BASE_FEATURE_COLS + ROLL_FEATURE_COLS

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------
def parse_int_list(raw: str) -> List[int]:
    vals = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one integer")
    return sorted(set(vals))

def norm_label(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    return out.replace({
        "": UNKNOWN_LABEL, "nan": UNKNOWN_LABEL, "NaN": UNKNOWN_LABEL,
        "None": UNKNOWN_LABEL, "none": UNKNOWN_LABEL, "<NA>": UNKNOWN_LABEL,
    })

def to_numeric(df: pd.DataFrame, cols: Sequence[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

def load_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = Path(data_path)
    if not path.exists():
        alt = Path(__file__).resolve().parent / str(data_path)
        if alt.exists():
            path = alt
    if not path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    d = pd.read_csv(path / "Dataset Distractions_distraction_old.csv")
    e = pd.read_csv(path / "Dataset Errors_distraction_old.csv")
    eb = pd.read_csv(path / "Dataset Errors_baseline_old.csv")
    db = pd.read_csv(path / "Dataset Driving Time_baseline_old.csv")

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

    d["model_pred_start"] = d["model_pred_start"].where(d["model_pred_start"] != UNKNOWN_LABEL, np.nan)
    d["model_pred_start"] = d["model_pred_start"].fillna(d["model_pred_end"])
    d["model_pred_start"] = norm_label(d["model_pred_start"])

    d["sensor_missing_flag"] = (
        d["arousal_start"].isna() | d["arousal_end"].isna() |
        d["hr_bpm_start"].isna() | d["hr_bpm_end"].isna()
    ).astype(float)

    for col in ["model_prob_start", "model_prob_end", "emotion_prob_start",
                "emotion_prob_end", "arousal_start", "arousal_end"]:
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
    cal_users = arr[n_test:n_test + n_cal].tolist()
    train_users = arr[n_test + n_cal:].tolist()
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
        set(dtr["model_pred_start"].tolist()) | set(dtr["model_pred_end"].tolist()) |
        set(etr["model_pred"].tolist()) | {UNKNOWN_LABEL}
    )
    emo_vocab = sorted(
        set(dtr["emotion_label_start"].tolist()) | set(dtr["emotion_label_end"].tolist()) |
        set(etr["emotion_label"].tolist()) | {UNKNOWN_LABEL}
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
    rows: List[Dict] = []
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

            s_pred_raw = str(row["model_pred_start"]).strip()
            e_pred_raw = str(row["model_pred_end"]).strip()
            s_pred_present = 0.0 if s_pred_raw in ["", "nan", "None", "unknown", UNKNOWN_LABEL] else 1.0
            e_pred_present = 0.0 if e_pred_raw in ["", "nan", "None", "unknown", UNKNOWN_LABEL] else 1.0

            s_pr = safe_encode(row["model_pred_start"], pred_enc)
            e_pr = safe_encode(row["model_pred_end"], pred_enc)
            s_em = safe_encode(row["emotion_label_start"], emo_enc)
            e_em = safe_encode(row["emotion_label_end"], emo_enc)

            ar_slope = (e_ar - s_ar) / max(win_dur, 1e-6)
            hr_slope = (e_hr - s_hr) / max(win_dur, 1e-6)
            mp_slope = (e_mp - s_mp) / max(win_dur, 1e-6)
            ep_slope = (e_ep - s_ep) / max(win_dur, 1e-6)

            sensor_missing = float(row.get("sensor_missing_flag", 0.0))
            hr_peak = max(s_hr, e_hr)
            ar_peak = max(s_ar, e_ar)
            hr_peak_delta = hr_peak - usr_hr
            ar_peak_delta = ar_peak - usr_ar

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
                    cur_pred_present = s_pred_present + (e_pred_present - s_pred_present) * alpha
                    distraction_intensity = cur_mp * cur_pred_present
                    t_in = float(off)
                    t_after = 0.0
                else:
                    t_after = float(off - win_dur)
                    cur_ar = usr_ar + (e_ar - usr_ar) * np.exp(-t_after / max(float(H), 1e-6))
                    cur_hr = usr_hr + (e_hr - usr_hr) * np.exp(-t_after / max(float(H), 1e-6))
                    cur_mp = global_model_prob + (e_mp - global_model_prob) * np.exp(-t_after / max(float(H), 1e-6))
                    cur_ep = global_emotion_prob + (e_ep - global_emotion_prob) * np.exp(-t_after / max(float(H), 1e-6))
                    cur_pred_present = e_pred_present
                    cur_pr = e_pr
                    cur_em = e_em
                    t_in = 0.0
                    distraction_intensity = 0.0

                j = bisect_left(err_ts, ts)
                target = 1 if j < len(err_ts) and err_ts[j] < ts + horizon else 0

                ar_dev = cur_ar - usr_ar
                hr_dev = cur_hr - usr_hr

                rows.append({
                    "user_id": uid,
                    "run_id": int(rid),
                    "distraction_idx": i,
                    "target": int(target),
                    "sample_ts": ts,
                    "time_since_distraction_end": t_after,
                    "time_in_distraction": t_in,
                    # within_distraction kept as metadata only — NOT in FEATURE_COLS
                    "within_distraction": inside,
                    "model_prob": float(cur_mp),
                    "model_pred_enc": float(cur_pr),
                    "emotion_prob": float(cur_ep),
                    "emotion_label_enc": float(cur_em),
                    "arousal": float(cur_ar),
                    "arousal_delta_baseline": float(ar_dev),
                    "hr_bpm": float(cur_hr),
                    "hr_delta_baseline": float(hr_dev),
                    "hr_delta_sq": float(hr_dev * hr_dev),
                    "dist_density_5": float(density(webs.get(key, []), ts, 5)),
                    "sensor_missing_flag": float(sensor_missing),
                    "baseline_error_rate": float(usr_er),
                    "distraction_intensity": float(distraction_intensity),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def add_causal_session_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["_ord"] = np.arange(len(out))
    out = out.sort_values(["user_id", "run_id", "sample_ts", "_ord"]).reset_index(drop=True)
    grp = out.groupby(["user_id", "run_id"], sort=False)

    for col in ROLL_SOURCE_COLS:
        out[f"{col}_diff1_abs"] = np.abs(grp[col].diff().fillna(0.0))

    out["state_energy"] = (
        np.abs(out["arousal_delta_baseline"]) + np.abs(out["hr_delta_baseline"]) +
        out["model_prob"] + out["emotion_prob"]
    ) / 4.0
    out["state_imbalance"] = np.abs(out["arousal_delta_baseline"]) + np.abs(out["hr_delta_baseline"])
    out["model_emotion_gap_abs"] = np.abs(out["model_prob"] - out["emotion_prob"])
    out["model_emotion_product"] = out["model_prob"] * out["emotion_prob"]
    out["arousal_hr_coupling"] = out["arousal_delta_baseline"] * out["hr_delta_baseline"]
    out["model_arousal_coupling"] = out["model_prob"] * out["arousal_delta_baseline"]
    #out["emotion_hr_coupling"] = out["emotion_prob"] * out["hr_delta_baseline"]
    out["signal_diff_abs_energy"] = (
        out["model_prob_diff1_abs"] + out["emotion_prob_diff1_abs"] +
        out["arousal_diff1_abs"] + out["hr_bpm_diff1_abs"]
    ) / 4.0

    out = out.sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)
    return out

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

# ------------------------------------------------------------------------------
# HMM class
# ------------------------------------------------------------------------------
class ImpairmentHMM:
    def __init__(
        self,
        n_states: int = 3,
        covariance_type: str = "diag",
        n_iter: int = 150,
        tol: float = 1e-4,
        random_state: int = RANDOM_SEED,
        constrain_backward: bool = False,
        transmat_prior_scale: float = 15.0,
        variance_floor: float = 1e-3,
    ) -> None:
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.constrain_backward = constrain_backward
        self.transmat_prior_scale = transmat_prior_scale
        self.variance_floor = variance_floor
        self.model_ = None
        self.train_loglik_ = []
        self.risk_order_ = None

    def _build_transition_prior(self, scale: float) -> np.ndarray:
        K = self.n_states
        P = np.ones((K, K), dtype=float) * 1e-6
        for i in range(K):
            P[i, i] = scale * 6.0
            if i + 1 < K:
                P[i, i+1] = scale * 2.5
            if i + 2 < K:
                P[i, i+2] = scale * 0.5
            if i - 1 >= 0:
                P[i, i-1] = scale * 0.2
            if i - 2 >= 0:
                P[i, i-2] = scale * 0.1
        return P

    def _zero_backward_arcs(self, A: np.ndarray) -> np.ndarray:
        A = A.copy()
        for i in range(1, self.n_states):
            A[i, :i] = BWD_FLOOR
        return A / A.sum(axis=1, keepdims=True).clip(1e-12)

    def _supervised_init(self, X: np.ndarray, labels: np.ndarray):
        K, d = self.n_states, X.shape[1]
        counts = np.bincount(labels, minlength=K)
        LOG.info("Supervised init counts: %s",
                 {f"state{s}": int(counts[s]) for s in range(K-1, -1, -1)})

        means = np.zeros((K, d))
        covars = []
        gvar = np.var(X, axis=0) + self.variance_floor * 2

        for raw_idx in range(K):
            sem_label = (K - 1) - raw_idx
            mask = labels == sem_label
            if mask.sum() > 3:
                means[raw_idx] = X[mask].mean(axis=0)
                v = np.var(X[mask], axis=0) + self.variance_floor
            else:
                means[raw_idx] = X.mean(axis=0)
                v = gvar
            covars.append(v)

        startprob = np.ones(K) / K
        transmat_prior = self._build_transition_prior(self.transmat_prior_scale)

        hmm = GaussianHMM(
            n_components=K,
            covariance_type=self.covariance_type,
            n_iter=0,
            tol=1e-12,
            random_state=self.random_state,
            init_params="",
            params="stmc",
            transmat_prior=transmat_prior,
        )
        hmm.startprob_ = startprob
        hmm.transmat_ = transmat_prior / transmat_prior.sum(axis=1, keepdims=True)
        hmm.means_ = means
        hmm.covars_ = np.array(covars)
        return hmm

    def fit(self, X: np.ndarray, bounds: List[Tuple[int, int]], labels: np.ndarray):
        lengths = [e - s for s, e in bounds if e > s]
        if not lengths:
            raise ValueError("No valid sequences")
        LOG.info("HMM fit: %d seqs, %d samples, %d feats, backward_constraint=%s",
                 len(lengths), sum(lengths), X.shape[1],
                 "ON" if self.constrain_backward else "OFF")

        hmm = self._supervised_init(X, labels)

        prev_ll = None
        for it in range(self.n_iter):
            hmm.n_iter = 1
            hmm.init_params = ""
            hmm.tol = 0.0
            hmm.fit(X, lengths=lengths)
            if self.constrain_backward:
                hmm.transmat_ = self._zero_backward_arcs(hmm.transmat_)
            ll = float(hmm.score(X, lengths=lengths))
            delta = ll - prev_ll if prev_ll is not None else float("nan")
            self.train_loglik_.append(ll)
            LOG.info("EM iter %3d | ll = %.2f  Δ = %.4f", it+1, ll, delta)
            if prev_ll is not None and abs(delta) < self.tol * (abs(prev_ll) + 1.0):
                LOG.info("Converged after %d iterations", it+1)
                break
            prev_ll = ll

        self.model_ = hmm
        return self

    def predict_posteriors(self, X: np.ndarray, bounds: List[Tuple[int, int]]) -> Tuple[np.ndarray, float]:
        if self.model_ is None:
            raise RuntimeError("Not fitted")
        gamma = np.zeros((len(X), self.n_states))
        ll = 0.0
        for s, e in bounds:
            if e > s:
                gamma[s:e] = self.model_.predict_proba(X[s:e])
                ll += float(self.model_.score(X[s:e]))
        return gamma, ll

    def predict_viterbi(self, X: np.ndarray, bounds: List[Tuple[int, int]]) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Not fitted")
        path = np.full(len(X), -1, dtype=int)
        for s, e in bounds:
            if e > s:
                path[s:e] = self.model_.predict(X[s:e])
        return path

    def to_dict(self) -> dict:
        if self.model_ is None:
            raise RuntimeError("Not fitted")
        m = self.model_
        return {
            "n_states": self.n_states,
            "covariance_type": self.covariance_type,
            "startprob": m.startprob_.tolist(),
            "transmat": m.transmat_.tolist(),
            "means": m.means_.tolist(),
            "covars": m.covars_.tolist(),
            "train_loglik": self.train_loglik_,
        }

# ------------------------------------------------------------------------------
# PIC Labeler
# ------------------------------------------------------------------------------
class PICLabeler:

    def __init__(self, n_bins: int = 30, random_state: int = RANDOM_SEED) -> None:
        self.n_bins = n_bins
        self.random_state = random_state
        self.scaler_ = None
        self.coef_ = None
        self.intercept_ = None
        self.pic_thresholds_ = None
        self.feature_cols_ = None
        self.n_states_ = None

    def fit(
        self,
        df_ca: pd.DataFrame,
        train_p_error_baseline: float,
        lookahead_t: int,
        n_states: int,
        feature_cols: List[str] = None,
        min_samples_per_bin: int = 10,
    ) -> "PICLabeler":
        if feature_cols is None:
            feature_cols = [c for c in BASE_PHYSIO_FEATURES if c in df_ca.columns]
        if not feature_cols:
            raise ValueError("No physiological features found in calibration DataFrame.")
        self.feature_cols_ = feature_cols
        self.n_states_ = n_states

        baseline_target_prob = 1.0 - np.exp(-train_p_error_baseline * lookahead_t)
        LOG.info(
            "PICLabeler: baseline_target_prob=%.4f  (λ=%.4f /s, T=%d s)",
            baseline_target_prob, train_p_error_baseline, lookahead_t,
        )

        if "target" not in df_ca.columns:
            raise ValueError("Calibration DataFrame must contain a 'target' column.")

        X_ca_raw = df_ca[feature_cols].astype(float).values
        y_ca = df_ca["target"].astype(int).values

        self.scaler_ = StandardScaler()
        X_ca_scaled = self.scaler_.fit_transform(X_ca_raw)

        lr = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=self.random_state,
        )
        lr.fit(X_ca_scaled, y_ca)

        self.coef_ = lr.coef_.ravel()
        self.intercept_ = float(lr.intercept_[0])

        pos_rate = y_ca.mean()
        LOG.info(
            "PICLabeler: fitted on %d CAL samples (%.1f%% positive), %d physio features",
            len(y_ca), 100.0 * pos_rate, len(feature_cols),
        )
        LOG.info(
            "PICLabeler top-3 weights: %s",
            sorted(zip(feature_cols, self.coef_), key=lambda x: abs(x[1]), reverse=True)[:3],
        )

        # Phase 2: thresholds from calibration PIC distribution
        pic_ca = self._pic(X_ca_scaled)

        bins = np.percentile(pic_ca, np.linspace(0, 100, self.n_bins + 1))
        bins = np.unique(bins)
        bin_indices = np.digitize(pic_ca, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

        bin_error_rates, bin_centers, bin_counts = [], [], []
        for b in range(len(bins) - 1):
            mask = bin_indices == b
            cnt = mask.sum()
            if cnt >= min_samples_per_bin:
                bin_error_rates.append(float(y_ca[mask].mean()))
                bin_centers.append(float(pic_ca[mask].mean()))
                bin_counts.append(cnt)

        bin_error_rates = np.array(bin_error_rates)
        bin_centers = np.array(bin_centers)

        if len(bin_error_rates) < 2:
            raise RuntimeError("Too few bins with sufficient samples to estimate PIC thresholds.")

        sort_idx = np.argsort(bin_centers)
        bin_centers = bin_centers[sort_idx]
        bin_error_rates = bin_error_rates[sort_idx]

        percentiles = np.linspace(100 / n_states, 100 * (n_states - 1) / n_states, n_states - 1)
        thresholds = np.percentile(pic_ca, percentiles)
        self.pic_thresholds_ = thresholds[::-1].tolist()

        LOG.info(
            "PICLabeler thresholds (high → low): %s",
            [round(t, 4) for t in self.pic_thresholds_],
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("PICLabeler not fitted.")
        X_raw = df[self.feature_cols_].astype(float).values
        X_scaled = self.scaler_.transform(X_raw)
        pic = self._pic(X_scaled)
        return self._labels_from_pic(pic)

    def predict_pic(self, df: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("PICLabeler not fitted.")
        X_scaled = self.scaler_.transform(df[self.feature_cols_].astype(float).values)
        return self._pic(X_scaled)

    def _pic(self, X_scaled: np.ndarray) -> np.ndarray:
        logit = X_scaled @ self.coef_ + self.intercept_
        return 1.0 / (1.0 + np.exp(-logit))

    def _labels_from_pic(self, pic: np.ndarray) -> np.ndarray:
        thresholds = self.pic_thresholds_
        K = self.n_states_
        labels = np.full(len(pic), K - 1, dtype=int)
        for level, thr in enumerate(thresholds[:-1]):
            labels[pic <= thr] = K - 2 - level
        labels[pic <= thresholds[-1]] = 0
        return labels

    def to_dict(self) -> dict:
        return {
            "n_states": self.n_states_,
            "feature_cols": self.feature_cols_,
            "coef": self.coef_.tolist(),
            "intercept": self.intercept_,
            "pic_thresholds": list(self.pic_thresholds_),
            "n_bins": self.n_bins,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PICLabeler":
        obj = cls(n_bins=d["n_bins"])
        obj.n_states_ = d["n_states"]
        obj.feature_cols_ = d["feature_cols"]
        obj.coef_ = np.array(d["coef"])
        obj.intercept_ = float(d["intercept"])
        obj.pic_thresholds_ = d["pic_thresholds"]
        return obj

# ------------------------------------------------------------------------------
# Helper functions for label derivation and state profiling
# ------------------------------------------------------------------------------
def log_label_distribution(labels: np.ndarray, tag: str, state_names: List[str] = None):
    if state_names is None:
        max_label = labels.max()
        state_names = [f"I{i}" for i in range(max_label, -1, -1)]
    counts = np.bincount(labels, minlength=len(state_names))
    total = counts.sum()
    parts = [f"{state_names[k]}={counts[k]} ({100*counts[k]/total:.1f}%)"
             for k in range(len(state_names)-1, -1, -1)]
    LOG.info("Labels [%s]  %s", tag, "  ".join(parts))

def compute_risk_score(gamma: np.ndarray, risk_order: np.ndarray,
                       weights: np.ndarray, intercept: float) -> np.ndarray:
    gamma_ordered = gamma[:, risk_order]
    logit = gamma_ordered @ weights + intercept
    return 1.0 / (1.0 + np.exp(-logit))


def _bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray,
                      n_boot: int = 1000, seed: int = RANDOM_SEED,
                      ci: float = 0.95) -> Tuple[float, float]:
    """Return (lower, upper) bootstrap confidence interval for AUC-ROC."""
    rng = np.random.RandomState(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if y_true[idx].sum() == 0 or y_true[idx].sum() == n:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(aucs, 100 * alpha))
    hi = float(np.percentile(aucs, 100 * (1 - alpha)))
    return lo, hi


def evaluate_risk_score(score: np.ndarray, target: np.ndarray,
                        df: pd.DataFrame,
                        tag: str, out_dir: Path, prefix: str = "") -> dict:
    valid = ~(np.isnan(score) | np.isnan(target))
    score_v = score[valid]
    target_v = target[valid].astype(int)

    if len(score_v) == 0:
        LOG.warning("No valid samples for %s evaluation", tag)
        return {}

    # Per-sample metrics (inflated — reported for reference only)
    auc_sample = roc_auc_score(target_v, score_v)
    ap_sample = average_precision_score(target_v, score_v)
    LOG.info("%s [per-sample, biased]:  AUC-ROC = %.4f  AP = %.4f  n = %d",
             tag, auc_sample, ap_sample, len(score_v))

    # Distraction-event-level metrics (primary / unbiased)
    agg_df = df.loc[valid, ["user_id", "run_id", "distraction_idx"]].copy()
    agg_df["score"] = score_v
    agg_df["target"] = target_v

    event_df = (
        agg_df
        .groupby(["user_id", "run_id", "distraction_idx"], sort=False)
        .agg(score_mean=("score", "mean"), target_max=("target", "max"))
        .reset_index()
    )

    y_ev = event_df["target_max"].values.astype(int)
    s_ev = event_df["score_mean"].values

    if y_ev.sum() == 0 or y_ev.sum() == len(y_ev):
        LOG.warning("%s: event-level labels are constant – skipping event AUC", tag)
        auc_event, ap_event = float("nan"), float("nan")
        ci_lo, ci_hi = float("nan"), float("nan")
    else:
        auc_event = roc_auc_score(y_ev, s_ev)
        ap_event = average_precision_score(y_ev, s_ev)
        ci_lo, ci_hi = _bootstrap_auc_ci(y_ev, s_ev)
        LOG.info(
            "%s [event-level, primary]:  AUC-ROC = %.4f  95%%CI [%.4f, %.4f]"
            "  AP = %.4f  n_events = %d",
            tag, auc_event, ci_lo, ci_hi, ap_event, len(y_ev),
        )

    # Calibration plot (event level)
    if not np.isnan(auc_event):
        prob_true, prob_pred = calibration_curve(y_ev, s_ev, n_bins=10, strategy="uniform")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(prob_pred, prob_true, marker="o",
                label=f"{tag} AUC={auc_event:.3f} AP={ap_event:.3f}")
        ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"Calibration plot (event level) – {tag}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}calibration_{tag.replace(' ', '_')}.png", dpi=150)
        plt.close(fig)

    return {
        "auc_roc": auc_event,
        "auc_roc_ci_lo": ci_lo,
        "auc_roc_ci_hi": ci_hi,
        "avg_precision": ap_event,
        "auc_roc_sample_biased": auc_sample,
        "avg_precision_sample_biased": ap_sample,
        "n_events": int(len(y_ev)),
        "n_samples": int(len(score_v)),
    }


def derive_state_profile(gamma: np.ndarray, true_labels: np.ndarray,
                         viterbi_path: np.ndarray) -> Dict:
    K = gamma.shape[1]
    mass = gamma.sum(axis=0).clip(1e-8)
    mean_label = (gamma.T @ true_labels.astype(float)) / mass
    risk_order = np.argsort(mean_label)[::-1].astype(int)
    semantics = {int(raw): f"I{rank}" for rank, raw in enumerate(risk_order)}
    LOG.info("State semantics: %s", semantics)
    LOG.info("Mean impairment per raw state: %s",
             {int(i): round(float(mean_label[i]), 3) for i in range(K)})
    p = viterbi_path
    rank_map = {raw: r for r, raw in enumerate(risk_order)}
    ranks = np.array([rank_map[s] for s in p])
    forward = np.sum(ranks[:-1] > ranks[1:])
    backward = np.sum(ranks[:-1] < ranks[1:])
    total = max(len(p)-1, 1)
    traj = {
        "forward_rate": round(forward / total, 4),
        "backward_rate": round(backward / total, 4),
        "fwd_vs_bwd_ratio": round(forward / max(backward, 1e-8), 4),
    }
    LOG.info("Trajectory: fwd=%.3f  bwd=%.3f  fwd/bwd=%.2f",
             traj["forward_rate"], traj["backward_rate"], traj["fwd_vs_bwd_ratio"])
    return {"risk_order": risk_order.tolist(), "trajectory_stats": traj}

# ------------------------------------------------------------------------------
# Plotting functions
# ------------------------------------------------------------------------------
def compute_empirical_rates_for_users(
    distractions: pd.DataFrame,
    errors_dist: pd.DataFrame,
    users: List[str],
    max_seconds: int = 15,
) -> pd.Series:
    from collections import defaultdict
    d_sel = distractions[distractions["user_id"].isin(users)].copy()
    e_sel = errors_dist[errors_dist["user_id"].isin(users)].copy()

    exposure = defaultdict(float)
    error_counts = defaultdict(int)

    for (uid, rid), grp in d_sel.groupby(["user_id", "run_id"]):
        err_ts = e_sel[(e_sel["user_id"] == uid) & (e_sel["run_id"] == rid)]["timestamp"].tolist()
        for _, row in grp.iterrows():
            end_ts = row["timestamp_end"]
            next_start = grp[grp["timestamp_start"] > end_ts]["timestamp_start"].min()
            h = min(max_seconds,
                    (next_start - end_ts).total_seconds() if pd.notna(next_start) else max_seconds)
            if h <= 0:
                continue
            for off in range(int(h) + 1):
                ts = end_ts + pd.Timedelta(seconds=off)
                exposure[off] += 1.0
                j = bisect_left(err_ts, ts)
                if j < len(err_ts) and err_ts[j] < ts + pd.Timedelta(seconds=1):
                    error_counts[off] += 1

    rates = {}
    for sec in range(max_seconds + 1):
        rates[sec] = error_counts[sec] / exposure[sec] if exposure[sec] > 0 else np.nan
    return pd.Series(rates, name="error_rate")

def plot_risk_vs_error_rate(risk, ts, out, empirical_rates):
    seconds = np.arange(0, 16)
    emp_rate = np.array([empirical_rates.get(s, np.nan) for s in seconds])
    mean_risk = []
    for s in seconds:
        mask = (ts >= s) & (ts < s+1)
        mean_risk.append(risk[mask].mean() if mask.sum() > 0 else np.nan)
    mean_risk = np.array(mean_risk)
    valid = ~(np.isnan(emp_rate) | np.isnan(mean_risk))
    corr = np.corrcoef(emp_rate[valid], mean_risk[valid])[0, 1] if valid.sum() > 2 else np.nan

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax2 = ax1.twinx()
    ax1.bar(seconds[valid], emp_rate[valid], color="#3498db", alpha=0.45,
            label="Empirical error rate")
    ax2.plot(seconds[valid], mean_risk[valid], "o-", color="#c0392b",
             linewidth=2.5, markersize=6, label="Risk Score")
    ax1.set_xlabel("Seconds since distraction ended", fontsize=12)
    ax1.set_ylabel("Error rate (events/s)", color="#3498db", fontsize=12)
    ax2.set_ylabel("Risk Score\n(calibrated probability)", color="#c0392b", fontsize=12)
    ax1.set_title(
        f"Post-distraction recovery curve (test users)\ncorrelation r = {corr:.3f}",
        fontsize=13, pad=20,
    )
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    LOG.info("Risk vs error rate → %s (r=%.3f)", out, corr)
    return float(corr)

def plot_hmm_means(model, feat_names, risk_order, out, top_n=20):
    means = model.means_
    mu = means.mean(axis=0)
    std = means.std(axis=0).clip(1e-8)
    z = (means - mu) / std
    z_ordered = z[risk_order, :]
    var_across = z_ordered.var(axis=0)
    top_idx = np.argsort(var_across)[::-1][:top_n]
    z_plot = z_ordered[:, top_idx]
    feat_plot = [feat_names[i] for i in top_idx]
    n_sel = len(feat_plot)
    fig, ax = plt.subplots(figsize=(max(8, n_sel*0.5), 4))
    im = ax.imshow(z_plot, aspect="auto", cmap="RdYlGn_r", vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(n_sel))
    ax.set_xticklabels(feat_plot, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(risk_order)))
    ax.set_yticklabels([f"I{i}" for i in range(len(risk_order)-1, -1, -1)], fontsize=9)
    plt.colorbar(im, ax=ax, label="z-score across states")
    ax.set_title("HMM emission means (z-scored)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("HMM means heatmap → %s", out)

def plot_feature_importance(means, feat_names, risk_order, out, top_n=20):
    means_ordered = means[risk_order, :]
    var_across = means_ordered.var(axis=0)
    sorted_idx = np.argsort(var_across)[::-1]
    top_idx = sorted_idx[:top_n]
    top_var = var_across[top_idx]
    total = top_var.sum()
    percentages = (top_var / total) * 100 if total > 0 else np.zeros_like(top_var)
    top_features = [feat_names[i] for i in top_idx]
    fig, ax = plt.subplots(figsize=(10, max(6, top_n*0.3)))
    y_pos = np.arange(len(top_features))
    bars = ax.barh(y_pos, percentages[::-1], color="#3498db")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features[::-1], fontsize=8)
    ax.set_xlabel("Importance (%)")
    ax.set_title(f"Top {top_n} features by variance across impairment states")
    for bar, pct in zip(bars, percentages[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=7)
    ax.set_xlim(0, percentages.max() * 1.15)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("Feature importance → %s", out)

# ------------------------------------------------------------------------------
# Sequence helpers
# ------------------------------------------------------------------------------
def sort_for_hmm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_ord"] = np.arange(len(df))
    df = df.sort_values(["user_id", "run_id", "sample_ts", "_ord"]).reset_index(drop=True)
    return df.drop(columns=["_ord"])


def sequence_bounds(df: pd.DataFrame) -> List[Tuple[int, int]]:
    df = df.reset_index(drop=True)
    bounds = []
    for (uid, rid), grp in df.groupby(["user_id", "run_id"], sort=True):
        idx = grp.index
        s, e = int(idx.min()), int(idx.max()) + 1
        if (e - s) != len(grp):
            raise RuntimeError(
                f"Non-contiguous index for user={uid} run={rid} "
                f"(span {e-s} ≠ count {len(grp)}). "
                "Ensure df is sorted by (user_id, run_id, sample_ts) "
                "and reset_index(drop=True) before calling sequence_bounds."
            )
        if e > s:
            bounds.append((s, e))
    return bounds


def last_index_above(series: pd.Series, threshold: float):
    mask = series > threshold
    if mask.any():
        return int(mask[mask].index[-1])
    return None

# ------------------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------------------
def run_pipeline(args):
    d_raw, e_raw, eb_raw, db_raw = load_data(args.data_path)
    run_integrity_checks(d_raw, e_raw)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    users = sorted(set(d_raw["user_id"].unique()) | set(e_raw["user_id"].unique()))
    train_users, cal_users, test_users = split_users(users, seed=args.seed)

    # Estimate hangover horizon H from training users only
    subset_e = e_raw[e_raw["user_id"].isin(train_users)].copy()
    subset_d = d_raw[d_raw["user_id"].isin(train_users)].copy()
    subset_eb = eb_raw[eb_raw["user_id"].isin(train_users)].copy()
    subset_db = db_raw[db_raw["user_id"].isin(train_users)].copy()

    subset_d["duration_sec"] = (
        subset_d["timestamp_end"] - subset_d["timestamp_start"]
    ).dt.total_seconds()
    train_base_duration = subset_db["run_duration_seconds"].sum()
    train_p_error_baseline = len(subset_eb) / train_base_duration
    LOG.info("Baseline error rate on training users: %.4f errors/s", train_p_error_baseline)

    global_results = compute_global_probabilities(
        subset_d, subset_e, subset_eb, subset_db, rolling_window=ROLLING_WINDOW
    )
    rates = global_results["rates_series"]
    H = last_index_above(rates, train_p_error_baseline)
    LOG.info(
        "Estimated hangover horizon H = %d seconds "
        "(last point where error rate exceeds baseline)", H
    )

    # Prepare data — imputation stats from train users only
    impute_stats = fit_train_imputation_stats(train_users, d_raw)
    d = apply_imputation_stats(d_raw, impute_stats)

    le_pred, le_emo = fit_label_encoders(train_users, d, e_raw)
    pred_enc = build_encoding_map(le_pred)
    emo_enc = build_encoding_map(le_emo)
    baselines = build_baselines(train_users, eb_raw, db_raw)

    d_train = d[d["user_id"].isin(train_users)]
    gmp = float(d_train[["model_prob_start", "model_prob_end"]].stack().median()) if len(d_train) else 0.5
    gep = float(d_train[["emotion_prob_start", "emotion_prob_end"]].stack().median()) if len(d_train) else 0.5
    if not np.isfinite(gmp): gmp = 0.5
    if not np.isfinite(gep): gep = 0.5

    wbs, webs, errs = build_lookups(d_raw, e_raw, users)

    T = args.lookahead_t
    LOG.info("Generating samples: H=%d s, T=%d s", H, T)

    def gen(user_set):
        df_out = generate_samples(
            H, T, user_set, wbs, webs, errs, baselines,
            pred_enc, emo_enc, gmp, gep,
        )
        df_out = add_causal_session_features(df_out)
        return df_out

    df_tr = gen(train_users)
    df_ca = gen(cal_users)
    df_te = gen(test_users)

    pp = fit_feature_postprocess(df_tr)
    df_tr = apply_feature_postprocess(df_tr, pp)
    df_ca = apply_feature_postprocess(df_ca, pp)
    df_te = apply_feature_postprocess(df_te, pp)

    df_tr = sort_for_hmm(df_tr)
    df_ca = sort_for_hmm(df_ca)
    df_te = sort_for_hmm(df_te)

    LOG.info("Label mode: PIC (physiological impairment composite, cal-fitted)")
    pic_labeler = PICLabeler(
        n_bins=getattr(args, "pic_n_bins", 30),
        random_state=args.seed,
    )
    pic_labeler.fit(
        df_ca,                              # ← cal only
        train_p_error_baseline=train_p_error_baseline,
        lookahead_t=T,
        n_states=args.n_states,
        min_samples_per_bin=10,
    )
    lbl_tr = pic_labeler.predict(df_tr)    # strictly out-of-sample supervision
    lbl_ca = pic_labeler.predict(df_ca)
    lbl_te = pic_labeler.predict(df_te)

    state_names = [f"I{i}" for i in range(args.n_states - 1, -1, -1)]
    log_label_distribution(lbl_tr, "train", state_names)
    log_label_distribution(lbl_ca, "cal", state_names)
    log_label_distribution(lbl_te, "test", state_names)

    available = [c for c in BASE_PHYSIO_FEATURES if c in df_tr.columns]
    LOG.info("Using %d physiological features (time-free): %s", len(available), available)

    if len(available) < 5:
        LOG.error("Too few features available. Check your data.")
        return

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(df_tr[available].astype(float))
    X_ca = scaler.transform(df_ca[available].astype(float))
    X_te = scaler.transform(df_te[available].astype(float))

    b_tr = sequence_bounds(df_tr)
    b_ca = sequence_bounds(df_ca)
    b_te = sequence_bounds(df_te)

    hmm = ImpairmentHMM(
        n_states=args.n_states,
        covariance_type=args.hmm_cov,
        n_iter=args.hmm_max_iter,
        tol=args.hmm_tol,
        random_state=args.seed,
        constrain_backward=args.constrain_backward,
        transmat_prior_scale=args.transmat_prior_scale,
        variance_floor=1e-3,
    )
    hmm.fit(X_tr, b_tr, lbl_tr)

    gamma_tr, ll_tr = hmm.predict_posteriors(X_tr, b_tr)
    gamma_ca, ll_ca = hmm.predict_posteriors(X_ca, b_ca)
    gamma_te, ll_te = hmm.predict_posteriors(X_te, b_te)

    LOG.info("HMM ll  train=%.1f  cal=%.1f  test=%.1f", ll_tr, ll_ca, ll_te)

    viterbi_tr = hmm.predict_viterbi(X_tr, b_tr)
    profile = derive_state_profile(gamma_tr, lbl_tr, viterbi_tr)
    risk_order = np.array(profile["risk_order"])
    hmm.risk_order_ = risk_order

    target_ca = df_ca["target"].values.astype(int)
    target_te = df_te["target"].values.astype(int)

    LOG.info("Learning risk weights from calibration set posteriors...")
    gamma_ordered_ca = gamma_ca[:, risk_order]
    lr = LogisticRegression(C=1.0, class_weight="balanced", random_state=args.seed)
    lr.fit(gamma_ordered_ca, target_ca)
    learned_weights = lr.coef_.ravel()
    learned_intercept = lr.intercept_[0]
    LOG.info("Learned weights: %s  intercept: %.4f", learned_weights, learned_intercept)

    risk_tr = compute_risk_score(gamma_tr, risk_order, learned_weights, learned_intercept)
    risk_ca = compute_risk_score(gamma_ca, risk_order, learned_weights, learned_intercept)
    risk_te = compute_risk_score(gamma_te, risk_order, learned_weights, learned_intercept)

    eval_risk = evaluate_risk_score(
        risk_te, target_te, df_te,
        "Risk (test)", out_dir, prefix="risk_",
    )
    LOG.info(
        "Test set (event-level): AUC-ROC = %.4f  95%%CI [%.4f, %.4f]  AP = %.4f",
        eval_risk["auc_roc"],
        eval_risk["auc_roc_ci_lo"],
        eval_risk["auc_roc_ci_hi"],
        eval_risk["avg_precision"],
    )

    test_rates = compute_empirical_rates_for_users(d_raw, e_raw, test_users, max_seconds=15)
    ts_te = df_te["time_since_distraction_end"].fillna(0.0).values.astype(float)
    corr = plot_risk_vs_error_rate(risk_te, ts_te, out_dir / "risk_vs_error_rate.png", test_rates)
    plot_feature_importance(hmm.model_.means_, available, risk_order,
                            out_dir / "feature_importance.png", top_n=20)

    artifact = {
        "hmm": hmm.to_dict(),
        "state_profile": profile,
        "scaler": scaler,
        "feature_cols": available,
        "risk_weights": learned_weights.tolist(),
        "risk_intercept": learned_intercept,
        "ftd_thresholds": {"impaired": FTD_IMPAIRED, "caution": FTD_CAUTION},
        "config": {"H": H, "T": T, "n_states": args.n_states},
        "postprocess": pp,
        "pic_labeler": pic_labeler.to_dict(),
    }
    joblib.dump(artifact, out_dir / "impairment_hmm.joblib")

    result = {
        "config": {"H": H, "T": T, "n_states": args.n_states},
        "n_features": len(available),
        "trajectory_stats": profile["trajectory_stats"],
        "risk_vs_error_corr": float(corr) if corr is not None else None,
        # Event-level (primary, unbiased)
        "test_auc_roc": eval_risk["auc_roc"],
        "test_auc_roc_ci_lo": eval_risk["auc_roc_ci_lo"],
        "test_auc_roc_ci_hi": eval_risk["auc_roc_ci_hi"],
        "test_avg_precision": eval_risk["avg_precision"],
        "test_n_events": eval_risk["n_events"],
        # Per-sample (biased, retained for comparison)
        "test_auc_roc_sample_biased": eval_risk["auc_roc_sample_biased"],
        "test_avg_precision_sample_biased": eval_risk["avg_precision_sample_biased"],
        "test_n_samples": eval_risk["n_samples"],
        "pic_labeler": pic_labeler.to_dict(),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    LOG.info("DONE  fwd/bwd=%.2f  risk_corr=%.3f",
             profile["trajectory_stats"]["fwd_vs_bwd_ratio"], corr)
    return result

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Driver Impairment HMM with PIC labels (physiological only, no time features)"
    )
    p.add_argument("--data-path", default="data", help="Path to data directory")
    p.add_argument("--output-dir", default="result/")
    p.add_argument("--lookahead-t", type=int, default=1, help="T (seconds)")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--n-states", type=int, default=3, help="Number of impairment states")
    p.add_argument("--hmm-cov", default="diag", choices=["diag", "spherical", "tied"])
    p.add_argument("--hmm-max-iter", type=int, default=150)
    p.add_argument("--hmm-tol", type=float, default=1e-4)
    p.add_argument("--transmat-prior-scale", type=float, default=15.0)
    p.add_argument("--constrain-backward", action="store_true", default=True)
    p.add_argument(
        "--pic-n-bins", type=int, default=30,
        help="Number of quantile bins used when fitting PIC thresholds.",
    )
    return p

def main():
    args = build_arg_parser().parse_args()
    run_pipeline(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())