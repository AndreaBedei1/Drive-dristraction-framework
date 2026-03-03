import argparse
import json
import logging
import math
from bisect import bisect_left
from itertools import islice
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from utils import *   # assumed to contain compute_global_probabilities and ROLLING_WINDOW

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
LOG = logging.getLogger("impairment_hmm")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ------------------------------------------------------------------------------
# Constants (defaults)
# ------------------------------------------------------------------------------
RANDOM_SEED = 42
DEFAULT_BENCHMARK_DATA_PATH = str(Path(__file__).resolve().parent / "data_hmm_synth_v2")
UNKNOWN_LABEL = "unknown"
BWD_FLOOR = 1e-10
FTD_IMPAIRED = 0.50
FTD_CAUTION = 0.25

# ------------------------------------------------------------------------------
# Rolling window sizes – change as desired (commented out in no‑time version)
# ------------------------------------------------------------------------------
# ROLL_SPANS = [3, 5, 10]

# ------------------------------------------------------------------------------
# Feature set for production: physiological signals only (no direct time)
# ------------------------------------------------------------------------------
BASE_PHYSIO_FEATURES = [
    # Physiological levels (current state, no direct time encoding)
    "hr_bpm", "arousal", "model_prob", "emotion_prob",
    # Deviations from personal baseline
    "hr_delta_baseline", "arousal_delta_baseline", "hr_delta_sq",
    # Cross-signal interactions (physiological coupling)
    "arousal_hr_coupling", "model_arousal_coupling", "emotion_hr_coupling",
    # Composite physiological activation
    "state_energy", "state_imbalance",
    "model_emotion_gap_abs", "model_emotion_product",
    # Categorical state encodings
    "model_pred_enc", "emotion_label_enc",
    # Signal dynamics (rate of change, not cumulative time)
    "signal_diff_abs_energy",
    # Distraction history density (event counts, not direct time)
    "dist_density_5", 
    # Per-user constants
    "baseline_error_rate", "sensor_missing_flag",
    #"within_distraction",
    "distraction_intensity"
]

# Rolling features are disabled in no‑time version, but we keep the feature lists
# for internal processing (they are still used in add_causal_session_features).
# These lists are unchanged from the original no‑time script.
DRIVER_STATE_DIST_NUMERIC = [
    "arousal_start", "arousal_end", "hr_bpm_start", "hr_bpm_end",
    "model_prob_start", "model_prob_end", "emotion_prob_start", "emotion_prob_end",
]
DRIVER_STATE_LABEL_COLS = [
    "model_pred_start", "model_pred_end", "emotion_label_start", "emotion_label_end",
]
BASE_FEATURE_COLS = [
    # Phase indicator (binary, not a time measurement)
    "within_distraction",
    # Physiological levels
    "model_prob", "model_pred_enc",
    "emotion_prob", "emotion_label_enc",
    "arousal", "arousal_delta_baseline",
    "hr_bpm", "hr_delta_baseline", "hr_delta_sq",
    # Distraction history (event counts, not direct time)
    "dist_density_5",
    # Per-user constants
    "sensor_missing_flag", "baseline_error_rate",
    "distraction_intensity"
]
ROLL_SOURCE_COLS = ["model_prob", "emotion_prob", "arousal", "hr_bpm"]
ROLL_FEATURE_COLS = (
    [f"{col}_diff1_abs" for col in ROLL_SOURCE_COLS]
    + ["state_energy", "state_imbalance", "model_emotion_gap_abs",
       "model_emotion_product", "arousal_hr_coupling", "model_arousal_coupling",
       "emotion_hr_coupling", "signal_diff_abs_energy"]
)
FEATURE_COLS = BASE_FEATURE_COLS + ROLL_FEATURE_COLS

# ------------------------------------------------------------------------------
# Helper functions (unchanged)
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

def _has_data_rows(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            lines = list(islice(f, 2))
        return len(lines) >= 2
    except Exception:
        return False

def _resolve_dataset_file(path: Path, filename: str, use_old_files: bool) -> Path:
    stem = filename[:-4] if filename.lower().endswith(".csv") else filename
    candidates: List[str] = []
    if use_old_files:
        candidates.append(f"{stem}_old.csv")
    candidates.append(f"{stem}.csv")
    if not use_old_files:
        candidates.append(f"{stem}_old.csv")

    for candidate in candidates:
        file_path = path / candidate
        if _has_data_rows(file_path):
            return file_path

    existing = [path / candidate for candidate in candidates if (path / candidate).exists()]
    if existing:
        return existing[0]
    raise FileNotFoundError(f"Missing dataset file for {filename}. Tried: {candidates}")

def load_data(data_path: str, use_old_files: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = Path(data_path)
    if not path.exists():
        alt = Path(__file__).resolve().parent / str(data_path)
        if alt.exists():
            path = alt
    if not path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    d_path = _resolve_dataset_file(path, "Dataset Distractions_distraction.csv", use_old_files)
    e_path = _resolve_dataset_file(path, "Dataset Errors_distraction.csv", use_old_files)
    eb_path = _resolve_dataset_file(path, "Dataset Errors_baseline.csv", use_old_files)
    db_path = _resolve_dataset_file(path, "Dataset Driving Time_baseline.csv", use_old_files)
    LOG.info("Loading datasets from: %s, %s, %s, %s", d_path.name, e_path.name, eb_path.name, db_path.name)
    d = pd.read_csv(d_path)
    e = pd.read_csv(e_path)
    eb = pd.read_csv(eb_path)
    db = pd.read_csv(db_path)

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

    def _series_vocab(series: pd.Series) -> List[str]:
        values = norm_label(series).tolist() if series is not None else [UNKNOWN_LABEL]
        cleaned = {UNKNOWN_LABEL}
        for value in values:
            text = str(value).strip()
            cleaned.add(text if text else UNKNOWN_LABEL)
        return sorted(cleaned)

    pred_vocab = sorted(
        set(_series_vocab(dtr["model_pred_start"]) if "model_pred_start" in dtr.columns else [UNKNOWN_LABEL]) |
        set(_series_vocab(dtr["model_pred_end"]) if "model_pred_end" in dtr.columns else [UNKNOWN_LABEL]) |
        set(_series_vocab(etr["model_pred"]) if "model_pred" in etr.columns else [UNKNOWN_LABEL])
    )
    emo_vocab = sorted(
        set(_series_vocab(dtr["emotion_label_start"]) if "emotion_label_start" in dtr.columns else [UNKNOWN_LABEL]) |
        set(_series_vocab(dtr["emotion_label_end"]) if "emotion_label_end" in dtr.columns else [UNKNOWN_LABEL]) |
        set(_series_vocab(etr["emotion_label"]) if "emotion_label" in etr.columns else [UNKNOWN_LABEL])
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
        run_start_ts = wins.iloc[0]["timestamp_start"]

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
            hr_peak_ts = end_ts if e_hr >= s_hr else start_ts
            ar_peak_ts = end_ts if e_ar >= s_ar else start_ts
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
                    phase_sec = float(off - win_dur)
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
                    phase_sec = t_after
                    distraction_intensity = 0.0
               
                j = bisect_left(err_ts, ts)
                target = 1 if j < len(err_ts) and err_ts[j] < ts + horizon else 0

                ar_dev = cur_ar - usr_ar
                hr_dev = cur_hr - usr_hr

                rows.append({
                    "user_id": uid,
                    "run_id": int(rid),
                    "target": int(target),
                    "sample_ts": ts,
                    # ── keep for sequencing / evaluation only (not HMM features) ──
                    "time_since_distraction_end": t_after,
                    "time_in_distraction": t_in,
                    "within_distraction": inside,
                    # ── physiological levels ──────────────────────────────────────
                    "model_prob": float(cur_mp),
                    "model_pred_enc": float(cur_pr),
                    "emotion_prob": float(cur_ep),
                    "emotion_label_enc": float(cur_em),
                    "arousal": float(cur_ar),
                    "arousal_delta_baseline": float(ar_dev),
                    "hr_bpm": float(cur_hr),
                    "hr_delta_baseline": float(hr_dev),
                    "hr_delta_sq": float(hr_dev * hr_dev),
                    # ── distraction history (event counts, not direct time) ────────
                    "dist_density_5": float(density(webs.get(key, []), ts, 5)),
                    # ── per-user constants ────────────────────────────────────────
                    "sensor_missing_flag": float(sensor_missing),
                    "baseline_error_rate": float(usr_er),
                    "distraction_intensity": float(distraction_intensity)
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
        # for span in ROLL_SPANS:
        #     ema = grp[col].transform(lambda s: s.ewm(span=span, adjust=False).mean())
        #     out[f"{col}_ema{span}"] = ema
        #     out[f"{col}_dev_ema{span}"] = out[col] - ema
        #     out[f"{col}_roll_std{span}"] = grp[col].transform(
        #         lambda s: s.rolling(window=span, min_periods=1).std(ddof=0)
        #     ).fillna(0.0)
        #     out[f"{col}_roll_range{span}"] = grp[col].transform(
        #         lambda s: s.rolling(window=span, min_periods=1).max() -
        #         s.rolling(window=span, min_periods=1).min()
        #     ).fillna(0.0)
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
    out["emotion_hr_coupling"] = out["emotion_prob"] * out["hr_delta_baseline"]

    out["signal_diff_abs_energy"] = (
        out["model_prob_diff1_abs"] + out["emotion_prob_diff1_abs"] +
        out["arousal_diff1_abs"] + out["hr_bpm_diff1_abs"]
    ) / 4.0
    # Drawdowns removed: cummax - decaying_signal is a monotone function of
    # elapsed time post-distraction, equivalent to a time proxy.

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
# HMM class with tunable number of states and proper initialization
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
        self.risk_order_ = None   # will be set after fitting

    def _build_transition_prior(self, scale: float) -> np.ndarray:
        """
        Build a transition prior matrix for an arbitrary number of states.
        States are ordered from highest impairment (index 0) to recovered (index K-1).
        Prior favours self-transitions and forward moves (to lower impairment)
        more than backward moves.
        """
        K = self.n_states
        P = np.ones((K, K), dtype=float) * 1e-6   # small base to avoid zeros
        for i in range(K):
            P[i, i] = scale * 6.0
            # forward (to less impaired, j > i)
            if i + 1 < K:
                P[i, i+1] = scale * 2.5
            if i + 2 < K:
                P[i, i+2] = scale * 0.5
            # backward (to more impaired, j < i)
            if i - 1 >= 0:
                P[i, i-1] = scale * 0.2
            if i - 2 >= 0:
                P[i, i-2] = scale * 0.1
        return P

    def _zero_backward_arcs(self, A: np.ndarray) -> np.ndarray:
        """Force backward transitions to be extremely small."""
        A = A.copy()
        for i in range(1, self.n_states):
            A[i, :i] = BWD_FLOOR
        return A / A.sum(axis=1, keepdims=True).clip(1e-12)

    def _supervised_init(self, X: np.ndarray, labels: np.ndarray):
        """
        Supervised initialisation using provided discrete labels.
        Labels are assumed to be 0 (recovered), 1 (mild), ..., K-1 (high).
        Raw state indices are mapped so that raw0 = high, raw1 = next, ..., rawK-1 = recovered.
        """
        K, d = self.n_states, X.shape[1]
        counts = np.bincount(labels, minlength=K)
        LOG.info("Supervised init counts: %s",
                 {f"state{s}": int(counts[s]) for s in range(K-1, -1, -1)})  # show highest first

        means = np.zeros((K, d))
        covars = []
        gvar = np.var(X, axis=0) + self.variance_floor * 2

        for raw_idx in range(K):
            # Semantic label that this raw state should represent: highest impairment = raw0
            sem_label = (K - 1) - raw_idx   # raw0 gets label K-1 (high), rawK-1 gets label 0 (recovered)
            mask = labels == sem_label
            if mask.sum() > 3:
                means[raw_idx] = X[mask].mean(axis=0)
                v = np.var(X[mask], axis=0) + self.variance_floor
            else:
                means[raw_idx] = X.mean(axis=0)
                v = gvar
            covars.append(v)

        # Start probabilities: uniform – no strong prior on initial state
        startprob = np.ones(K) / K

        # Transition prior
        transmat_prior = self._build_transition_prior(self.transmat_prior_scale)

        # Create the underlying GaussianHMM
        hmm = GaussianHMM(
            n_components=K,
            covariance_type=self.covariance_type,
            n_iter=1,
            tol=1e-12,
            random_state=self.random_state,
            init_params="",           # we set all parameters manually
            params="stmc",             # we will update startprob, transmat, means, covars
            transmat_prior=transmat_prior,
        )
        hmm.startprob_ = startprob
        hmm.transmat_ = transmat_prior / transmat_prior.sum(axis=1, keepdims=True)  # normalised prior as initial guess
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
        # First EM step to incorporate data
        hmm.fit(X, lengths=lengths)

        if self.constrain_backward:
            hmm.transmat_ = self._zero_backward_arcs(hmm.transmat_)

        prev_ll = None
        for it in range(self.n_iter):
            hmm.n_iter = 1
            hmm.init_params = ""   # do not re‑initialise
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
# Generalized PIC Labeler for K states
# ------------------------------------------------------------------------------
class PICLabeler:
    """
    Two-phase physiological label constructor for K impairment states.

    Phase 1 — Learn risk weights from training errors
        Fits a logistic regression on training samples:
            target ~ BASE_PHYSIO_FEATURES (physio only)
        where target=1 means an error occurred in the T-second lookahead window.
        The fitted coefficients become the physiological risk weights w.

    Phase 2 — Assign labels via PIC thresholds
        PIC_i = sigmoid(w · physio_features_i)
        For K states, we need K-1 thresholds. Thresholds are found on the **calibration**
        set by binning PIC and locating where the empirical error rate crosses
        multiples of the baseline rate. Multipliers are defined as:
            mult_i = 2.0 ** i   for i = 1, 2, ..., K-1
        This yields thresholds p_1 > p_2 > ... > p_{K-1} (higher thresholds
        correspond to higher impairment).

    Samples flagged within_distraction=True are always assigned the highest state (K-1).
    """

    def __init__(self, n_bins: int = 30, random_state: int = RANDOM_SEED) -> None:
        self.n_bins       = n_bins
        self.random_state = random_state
        self.scaler_         = None
        self.coef_           = None
        self.intercept_      = None
        self.pic_thresholds_ = None   # list of length K-1, sorted descending
        self.feature_cols_   = None
        self.n_states_       = None

    def fit(
        self,
        df_tr: pd.DataFrame,
        df_ca: pd.DataFrame,
        train_p_error_baseline: float,
        lookahead_t: int,
        n_states: int,
        feature_cols: List[str] = None,
        min_samples_per_bin: int = 10,
    ) -> "PICLabeler":
        if feature_cols is None:
            # Use all BASE_PHYSIO_FEATURES present in df_tr
            feature_cols = [c for c in BASE_PHYSIO_FEATURES if c in df_tr.columns]
        if not feature_cols:
            raise ValueError("No physiological features found in training DataFrame.")
        self.feature_cols_ = feature_cols
        self.n_states_ = n_states

        baseline_target_prob = 1.0 - np.exp(-train_p_error_baseline * lookahead_t)
        LOG.info(
            "PICLabeler: baseline_target_prob=%.4f  (λ=%.4f /s, T=%d s)",
            baseline_target_prob, train_p_error_baseline, lookahead_t,
        )

        if "target" not in df_tr.columns:
            raise ValueError("Training DataFrame must contain a 'target' column.")

        # ---- Phase 1: Fit LR on training set ----
        X_tr_raw = df_tr[feature_cols].astype(float).values
        y_tr     = df_tr["target"].astype(int).values

        self.scaler_ = StandardScaler()
        X_tr_scaled  = self.scaler_.fit_transform(X_tr_raw)

        lr = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=self.random_state,
        )
        lr.fit(X_tr_scaled, y_tr)

        self.coef_      = lr.coef_.ravel()
        self.intercept_ = float(lr.intercept_[0])

        pos_rate = y_tr.mean()
        LOG.info(
            "PICLabeler Phase 1: fitted on %d samples (%.1f%% positive), "
            "%d physio features",
            len(y_tr), 100.0 * pos_rate, len(feature_cols),
        )
        LOG.info(
            "PICLabeler top-3 weights: %s",
            sorted(zip(feature_cols, self.coef_), key=lambda x: abs(x[1]), reverse=True)[:3],
        )

        # ---- Phase 2: Compute PIC scores on calibration set and find thresholds ----
        X_ca_raw = df_ca[feature_cols].astype(float).values
        X_ca_scaled = self.scaler_.transform(X_ca_raw)
        pic_ca = self._pic(X_ca_scaled)
        y_ca   = df_ca["target"].astype(int).values

        # Bin calibration samples
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
        bin_centers     = np.array(bin_centers)
        bin_counts      = np.array(bin_counts)

        if len(bin_error_rates) < 2:
            raise RuntimeError("Too few bins with sufficient samples to estimate PIC thresholds.")

        sort_idx = np.argsort(bin_centers)
        bin_centers = bin_centers[sort_idx]
        bin_error_rates = bin_error_rates[sort_idx]

        # Multipliers for thresholds (2, 4, 8, ... for states 1..K-1)
        # Instead of multipliers, compute percentiles that split the calibration set
        # into roughly equal‑sized groups (for K states, we need K-1 thresholds).
        percentiles = np.linspace(100 / n_states, 100 * (n_states - 1) / n_states, n_states - 1)
        thresholds = np.percentile(pic_ca, percentiles)          # ascending order
        self.pic_thresholds_ = thresholds[::-1].tolist()         # descending for high → low


        LOG.info(
            "PICLabeler thresholds (high → low): %s",
            [round(t, 4) for t in self.pic_thresholds_]
        )

        # Assign labels to training set for diagnostics (optional)
        X_tr_scaled = self.scaler_.transform(X_tr_raw)   # reuse scaler
        pic_tr = self._pic(X_tr_scaled)
        lbl_tr = self._labels_from_pic(pic_tr, df_tr)
        counts = np.bincount(lbl_tr, minlength=n_states)
        total  = counts.sum()
        parts  = [f"I{i}={counts[i]} ({100*counts[i]/total:.1f}%)"
                  for i in range(n_states-1, -1, -1)]
        LOG.info("PICLabeler train label distribution: %s", "  ".join(parts))

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return integer impairment labels (0..K-1) for any DataFrame."""
        if self.coef_ is None:
            raise RuntimeError("PICLabeler not fitted.")
        X_raw    = df[self.feature_cols_].astype(float).values
        X_scaled = self.scaler_.transform(X_raw)
        pic      = self._pic(X_scaled)
        return self._labels_from_pic(pic, df)

    def predict_pic(self, df: pd.DataFrame) -> np.ndarray:
        """Return raw PIC scores in [0, 1] for diagnostics."""
        if self.coef_ is None:
            raise RuntimeError("PICLabeler not fitted.")
        X_scaled = self.scaler_.transform(df[self.feature_cols_].astype(float).values)
        return self._pic(X_scaled)

    def _pic(self, X_scaled: np.ndarray) -> np.ndarray:
        logit = X_scaled @ self.coef_ + self.intercept_
        return 1.0 / (1.0 + np.exp(-logit))

    def _labels_from_pic(self, pic: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        thresholds = self.pic_thresholds_   # descending
        K = self.n_states_
        labels = np.full(len(pic), K-1, dtype=int)   # start with highest

        # For each threshold (from second highest down to lowest), lower label when pic <= threshold
        for level, thr in enumerate(thresholds[:-1]):
            labels[pic <= thr] = K - 2 - level

        # Lowest state (recovered) for those below the smallest threshold
        labels[pic <= thresholds[-1]] = 0

        # if "within_distraction" in df.columns:
        #     wd = df["within_distraction"].fillna(0).astype(bool).values
        #     labels[wd] = K - 1

        return labels

    def to_dict(self) -> dict:
        return {
            "n_states":       self.n_states_,
            "feature_cols":   self.feature_cols_,
            "coef":           self.coef_.tolist(),
            "intercept":      self.intercept_,
            "pic_thresholds": list(self.pic_thresholds_),
            "n_bins":         self.n_bins,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PICLabeler":
        obj = cls(n_bins=d["n_bins"])
        obj.n_states_       = d["n_states"]
        obj.feature_cols_   = d["feature_cols"]
        obj.coef_           = np.array(d["coef"])
        obj.intercept_      = float(d["intercept"])
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

def compute_risk_score(gamma: np.ndarray, risk_order: np.ndarray, weights: np.ndarray, intercept: float) -> np.ndarray:
    """Compute calibrated risk probability from HMM posteriors."""
    gamma_ordered = gamma[:, risk_order]
    logit = gamma_ordered @ weights + intercept
    prob = 1.0 / (1.0 + np.exp(-logit))
    return prob

def evaluate_risk_score(score: np.ndarray, target: np.ndarray,
                        tag: str, out_dir: Path, prefix: str = "") -> dict:
    """Compute AUC-ROC, AP, and calibration plot."""
    valid = ~(np.isnan(score) | np.isnan(target))
    score = score[valid]
    target = target[valid].astype(int)
    if len(score) == 0:
        LOG.warning(f"No valid samples for {tag} evaluation")
        return {}
    auc = roc_auc_score(target, score)
    ap = average_precision_score(target, score)
    prob_true, prob_pred = calibration_curve(target, score, n_bins=10, strategy='uniform')
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker='o', label=f'{tag} (AUC={auc:.3f}, AP={ap:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title(f'Calibration plot – {tag}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}calibration_{tag.replace(' ', '_')}.png", dpi=150)
    plt.close(fig)
    LOG.info(f"{tag}: AUC-ROC = {auc:.4f}, Average Precision = {ap:.4f}")
    return {"auc_roc": auc, "avg_precision": ap, "n_samples": int(len(score))}

def summarize_scores(score: np.ndarray, target: np.ndarray) -> dict:
    valid = ~(np.isnan(score) | np.isnan(target))
    score = score[valid]
    target = target[valid].astype(int)
    if len(score) == 0:
        return {"auc_roc": float("nan"), "avg_precision": float("nan"), "n_samples": 0}
    return {
        "auc_roc": float(roc_auc_score(target, score)),
        "avg_precision": float(average_precision_score(target, score)),
        "n_samples": int(len(score)),
    }

def derive_state_profile(gamma: np.ndarray, true_labels: np.ndarray, viterbi_path: np.ndarray) -> Dict:
    """Determine state semantics by mean impairment label per raw state."""
    K = gamma.shape[1]
    mass = gamma.sum(axis=0).clip(1e-8)
    mean_label = (gamma.T @ true_labels.astype(float)) / mass
    risk_order = np.argsort(mean_label)[::-1].astype(int)   # highest impairment first
    semantics = {int(raw): f"I{rank}" for rank, raw in enumerate(risk_order)}
    LOG.info("State semantics: %s", semantics)
    LOG.info("Mean impairment per raw state: %s",
             {int(i): round(float(mean_label[i]), 3) for i in range(K)})
    # Compute forward/backward rates using rank order
    p = viterbi_path
    rank_map = {raw: r for r, raw in enumerate(risk_order)}
    ranks = np.array([rank_map[s] for s in p])
    forward = np.sum(ranks[:-1] > ranks[1:])   # moving to lower rank (less impaired)
    backward = np.sum(ranks[:-1] < ranks[1:])  # moving to higher rank (more impaired)
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
    """
    Compute empirical error rate per second after distraction end for given users.
    Returns a Series indexed by second (0..max_seconds) with rate = errors / total time.
    """
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
            horizon = min(max_seconds, (next_start - end_ts).total_seconds() if pd.notna(next_start) else max_seconds)
            if horizon <= 0:
                continue
            for off in range(int(horizon) + 1):
                ts = end_ts + pd.Timedelta(seconds=off)
                exposure[off] += 1.0
                j = bisect_left(err_ts, ts)
                if j < len(err_ts) and err_ts[j] < ts + pd.Timedelta(seconds=1):
                    error_counts[off] += 1

    rates = {}
    for sec in range(max_seconds + 1):
        if exposure[sec] > 0:
            rates[sec] = error_counts[sec] / exposure[sec]
        else:
            rates[sec] = np.nan
    return pd.Series(rates, name="error_rate")

def plot_risk_vs_error_rate(risk, ts, out, empirical_rates):
    seconds = np.arange(0, 16)
    emp_rate = np.array([empirical_rates.get(s, np.nan) for s in seconds])
    mean_risk = []
    for s in seconds:
        mask = (ts >= s) & (ts < s+1)
        mean_risk.append(risk[mask].mean() if mask.sum()>0 else np.nan)
    mean_risk = np.array(mean_risk)
    valid = ~(np.isnan(emp_rate) | np.isnan(mean_risk))
    corr = np.corrcoef(emp_rate[valid], mean_risk[valid])[0,1] if valid.sum()>2 else np.nan

    fig, ax1 = plt.subplots(figsize=(11,5.5))
    ax2 = ax1.twinx()
    ax1.bar(seconds[valid], emp_rate[valid], color="#3498db", alpha=0.45, label="Empirical error rate")
    ax2.plot(seconds[valid], mean_risk[valid], "o-", color="#c0392b", linewidth=2.5, markersize=6,
             label="Risk Score")
    ax1.set_xlabel("Seconds since distraction ended", fontsize=12)
    ax1.set_ylabel("Error rate (events/s)", color="#3498db", fontsize=12)
    ax2.set_ylabel("Risk Score\n(calibrated probability)", color="#c0392b", fontsize=12)
    ax1.set_title(f"Post-distraction recovery curve (test users)\ncorrelation r = {corr:.3f}", fontsize=13, pad=20)
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
    z_ordered = z[risk_order, :]   # rows in semantic order (high to recovered)
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
    percentages = (top_var / total) * 100 if total>0 else np.zeros_like(top_var)
    top_features = [feat_names[i] for i in top_idx]
    fig, ax = plt.subplots(figsize=(10, max(6, top_n*0.3)))
    y_pos = np.arange(len(top_features))
    bars = ax.barh(y_pos, percentages[::-1], color="#3498db")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features[::-1], fontsize=8)
    ax.set_xlabel("Importance (%)")
    ax.set_title(f"Top {top_n} features by variance across impairment states")
    for i, (bar, pct) in enumerate(zip(bars, percentages[::-1])):
        ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                f"{pct:.1f}%", va='center', fontsize=7)
    ax.set_xlim(0, percentages.max()*1.15)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("Feature importance → %s", out)




# ------------------------------------------------------------------------------
# Extra importance plots
# ------------------------------------------------------------------------------
def plot_permutation_feature_importance(model, X, y, feat_names, out, top_n=20):
    perm = permutation_importance(
        model,
        X,
        y,
        n_repeats=5,
        random_state=RANDOM_SEED,
        scoring="roc_auc",
    )
    order = np.argsort(perm.importances_mean)[::-1][:top_n]
    names = [feat_names[i] for i in order]
    values = perm.importances_mean[order]
    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.35)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values[::-1], color="#2e86de")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_xlabel("Permutation importance (validation AUC drop)")
    ax.set_title(f"Top {len(names)} validation permutation importances")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("Permutation feature importance -> %s", out)
    return {names[i]: float(values[i]) for i in range(len(names))}


def _feature_family(name: str) -> str:
    if name.startswith("hr_") or "hr_" in name:
        return "heart_rate"
    if name.startswith("emotion_") or "emotion" in name:
        return "emotion"
    if name.startswith("model_") or "distraction" in name or name == "dist_density_5":
        return "distraction"
    if name.startswith("arousal") or "arousal" in name:
        return "arousal"
    if "baseline" in name or "sensor_missing" in name:
        return "baseline"
    return "other"


def plot_grouped_feature_importance(importance_map: Dict[str, float], out: Path):
    grouped: Dict[str, float] = {}
    for feat, value in importance_map.items():
        family = _feature_family(feat)
        grouped[family] = grouped.get(family, 0.0) + max(0.0, float(value))
    items = sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)
    if not items:
        return grouped
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color="#27ae60")
    ax.set_ylabel("Grouped importance")
    ax.set_title("Feature-family importance")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("Grouped feature importance -> %s", out)
    return grouped


# ------------------------------------------------------------------------------
# Sequence helpers
# ------------------------------------------------------------------------------
def sort_for_hmm(df):
    df = df.copy()
    df["_ord"] = np.arange(len(df))
    df = df.sort_values(["user_id", "run_id", "sample_ts", "_ord"]).reset_index(drop=True)
    return df.drop(columns=["_ord"])

def sequence_bounds(df):
    bounds = []
    for _, grp in df.groupby(["user_id", "run_id"], sort=False):
        s, e = grp.index.min(), grp.index.max()+1
        if e > s:
            bounds.append((s, e))
    return bounds

def last_index_above(series: pd.Series, threshold: float):
    mask = series > threshold
    if mask.any():
        return int(mask[mask].index[-1])
    else:
        return None


def _candidate_sort_key(metrics: Dict[str, float]) -> Tuple[float, float]:
    auc = float(metrics.get("auc_roc", float("-inf")))
    ap = float(metrics.get("avg_precision", float("-inf")))
    return (auc, ap)


def fit_hmm_candidate(df_tr, df_va, df_te, feature_cols, train_p_error_baseline, args) -> Dict:
    pic_labeler = PICLabeler(
        n_bins=getattr(args, "pic_n_bins", 30),
        random_state=args.seed,
    )
    pic_labeler.fit(
        df_tr,
        df_tr,
        train_p_error_baseline=train_p_error_baseline,
        lookahead_t=args.lookahead_t,
        n_states=args.n_states,
        min_samples_per_bin=10,
    )
    lbl_tr = pic_labeler.predict(df_tr)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(df_tr[feature_cols].astype(float))
    X_va = scaler.transform(df_va[feature_cols].astype(float))
    X_te = scaler.transform(df_te[feature_cols].astype(float))

    b_tr = sequence_bounds(df_tr)
    b_va = sequence_bounds(df_va)
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
    gamma_va, ll_va = hmm.predict_posteriors(X_va, b_va)
    gamma_te, ll_te = hmm.predict_posteriors(X_te, b_te)

    viterbi_tr = hmm.predict_viterbi(X_tr, b_tr)
    profile = derive_state_profile(gamma_tr, lbl_tr, viterbi_tr)
    risk_order = np.array(profile["risk_order"])
    hmm.risk_order_ = risk_order

    target_tr = df_tr["target"].values.astype(int)
    target_va = df_va["target"].values.astype(int)
    target_te = df_te["target"].values.astype(int)

    risk_lr = LogisticRegression(C=1.0, class_weight="balanced", random_state=args.seed)
    risk_lr.fit(gamma_tr[:, risk_order], target_tr)

    val_pred = risk_lr.predict_proba(gamma_va[:, risk_order])[:, 1]
    test_pred = risk_lr.predict_proba(gamma_te[:, risk_order])[:, 1]

    return {
        "family": "hmm",
        "feature_cols": list(feature_cols),
        "model": hmm,
        "pic_labeler": pic_labeler,
        "scaler": scaler,
        "risk_lr": risk_lr,
        "profile": profile,
        "risk_order": risk_order,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "val_metrics": summarize_scores(val_pred, target_va),
        "test_metrics": summarize_scores(test_pred, target_te),
        "ll": {"train": float(ll_tr), "val": float(ll_va), "test": float(ll_te)},
    }


def fit_hgb_candidate(df_tr, df_va, df_te, feature_cols, args) -> Dict:
    target_tr = df_tr["target"].values.astype(int)
    target_va = df_va["target"].values.astype(int)
    target_te = df_te["target"].values.astype(int)

    grid = [
        {"max_depth": 4, "learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 50, "l2_regularization": 0.0},
        {"max_depth": 6, "learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 50, "l2_regularization": 0.0},
        {"max_depth": 6, "learning_rate": 0.03, "max_iter": 500, "min_samples_leaf": 30, "l2_regularization": 0.0},
        {"max_depth": 8, "learning_rate": 0.03, "max_iter": 500, "min_samples_leaf": 30, "l2_regularization": 0.1},
    ]

    best = None
    X_tr = df_tr[feature_cols].astype(float).values
    X_va = df_va[feature_cols].astype(float).values
    X_te = df_te[feature_cols].astype(float).values

    for cfg in grid:
        model = HistGradientBoostingClassifier(random_state=args.seed, **cfg)
        model.fit(X_tr, target_tr)
        val_pred = model.predict_proba(X_va)[:, 1]
        metrics = summarize_scores(val_pred, target_va)
        if best is None or _candidate_sort_key(metrics) > _candidate_sort_key(best["val_metrics"]):
            best = {
                "model": model,
                "config": dict(cfg),
                "val_pred": val_pred,
                "val_metrics": metrics,
            }

    if best is None:
        raise RuntimeError("Failed to fit any HistGradientBoosting candidate.")

    test_pred = best["model"].predict_proba(X_te)[:, 1]
    return {
        "family": "hgb",
        "feature_cols": list(feature_cols),
        "model": best["model"],
        "config": best["config"],
        "val_pred": best["val_pred"],
        "test_pred": test_pred,
        "val_metrics": best["val_metrics"],
        "test_metrics": summarize_scores(test_pred, target_te),
    }

# ------------------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------------------
def run_pipeline(args):
    d_raw, e_raw, eb_raw, db_raw = load_data(args.data_path, use_old_files=args.use_old_files)
    run_integrity_checks(d_raw, e_raw)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    users = sorted(set(d_raw["user_id"].unique()) | set(e_raw["user_id"].unique()))
    train_users, val_users, test_users = split_users(users, seed=args.seed)
    LOG.info("User split: train=%d val=%d test=%d", len(train_users), len(val_users), len(test_users))
    LOG.info("Train users: %s", train_users)
    LOG.info("Val users: %s", val_users)
    LOG.info("Test users: %s", test_users)

    # Estimate hangover horizon H from training users
    subset_e = e_raw[e_raw['user_id'].isin(train_users)].copy()
    subset_d = d_raw[d_raw['user_id'].isin(train_users)].copy()
    subset_eb = eb_raw[eb_raw['user_id'].isin(train_users)].copy()
    subset_db = db_raw[db_raw['user_id'].isin(train_users)].copy()

    subset_d['duration_sec'] = (subset_d['timestamp_end'] - subset_d['timestamp_start']).dt.total_seconds()
    train_base_duration = subset_db['run_duration_seconds'].sum()
    train_p_error_baseline = len(subset_eb) / train_base_duration
    LOG.info("Baseline error rate on training users: %.4f errors/s", train_p_error_baseline)

    global_results = compute_global_probabilities(subset_d, subset_e, subset_eb, subset_db, rolling_window=ROLLING_WINDOW)
    rates = global_results['rates_series']
    H = last_index_above(rates, train_p_error_baseline)
    if H is None:
        H = max(1, int(args.lookahead_t))
    LOG.info("Estimated hangover horizon H = %d seconds (last point where error rate exceeds baseline)", H)

    # Prepare data
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

    wbs, webs, errs = build_lookups(d, e_raw, users)

    T = args.lookahead_t
    LOG.info("Generating samples: H=%d s, T=%d s", H, T)

    def gen(user_set):
        df = generate_samples(
            H, T, user_set, wbs, webs, errs, baselines,
            pred_enc, emo_enc, gmp, gep,
        )
        df = add_causal_session_features(df)
        return df

    df_tr = gen(train_users)
    df_ca = gen(val_users)
    df_te = gen(test_users)

    pp = fit_feature_postprocess(df_tr)
    df_tr = apply_feature_postprocess(df_tr, pp)
    df_ca = apply_feature_postprocess(df_ca, pp)
    df_te = apply_feature_postprocess(df_te, pp)

    df_tr = sort_for_hmm(df_tr)
    df_ca = sort_for_hmm(df_ca)
    df_te = sort_for_hmm(df_te)

    available = [c for c in BASE_PHYSIO_FEATURES if c in df_tr.columns]
    LOG.info("Using %d physiological features (time-free): %s", len(available), available)

    if len(available) < 5:
        LOG.error("Too few features available. Check your data.")
        return

    requested_family = str(getattr(args, "model_family", "auto")).strip().lower()
    candidate_families = ["hmm", "hgb"] if requested_family == "auto" else [requested_family]
    candidates: List[Dict] = []
    for family in candidate_families:
        if family == "hmm":
            LOG.info("Fitting candidate model: HMM")
            candidates.append(
                fit_hmm_candidate(
                    df_tr=df_tr,
                    df_va=df_ca,
                    df_te=df_te,
                    feature_cols=available,
                    train_p_error_baseline=train_p_error_baseline,
                    args=args,
                )
            )
        elif family == "hgb":
            LOG.info("Fitting candidate model: HistGradientBoosting")
            candidates.append(
                fit_hgb_candidate(
                    df_tr=df_tr,
                    df_va=df_ca,
                    df_te=df_te,
                    feature_cols=available,
                    args=args,
                )
            )
        else:
            raise ValueError(f"Unsupported model family: {family}")

    for cand in candidates:
        LOG.info(
            "Candidate %s  val_auc=%.4f  val_ap=%.4f  test_auc=%.4f  test_ap=%.4f",
            cand["family"],
            cand["val_metrics"]["auc_roc"],
            cand["val_metrics"]["avg_precision"],
            cand["test_metrics"]["auc_roc"],
            cand["test_metrics"]["avg_precision"],
        )

    best = max(candidates, key=lambda cand: _candidate_sort_key(cand["val_metrics"]))
    LOG.info("Selected model family: %s", best["family"])

    target_te = df_te["target"].values.astype(int)
    eval_risk = evaluate_risk_score(best["test_pred"], target_te, "Risk (test)", Path(args.output_dir), prefix="risk_")
    LOG.info(
        "Selected model test metrics: AUC-ROC = %.4f, Average Precision = %.4f",
        eval_risk["auc_roc"],
        eval_risk["avg_precision"],
    )

    test_rates = compute_empirical_rates_for_users(d_raw, e_raw, test_users, max_seconds=15)
    ts_te = df_te["time_since_distraction_end"].fillna(0.0).values.astype(float)
    corr = plot_risk_vs_error_rate(best["test_pred"], ts_te, out_dir / "risk_vs_error_rate.png", test_rates)

    feature_family_importance = {}
    if best["family"] == "hmm":
        plot_feature_importance(best["model"].model_.means_, available, best["risk_order"], out_dir / "feature_importance.png", top_n=20)
        hmm_importance = {
            available[i]: float(best["model"].model_.means_[best["risk_order"], i].var())
            for i in range(len(available))
        }
        feature_family_importance = plot_grouped_feature_importance(hmm_importance, out_dir / "feature_family_importance.png")
    else:
        importance_map = plot_permutation_feature_importance(
            best["model"],
            df_ca[available].astype(float).values,
            df_ca["target"].astype(int).values,
            available,
            out_dir / "feature_importance.png",
            top_n=min(20, len(available)),
        )
        feature_family_importance = plot_grouped_feature_importance(importance_map, out_dir / "feature_family_importance.png")

    artifact = {
        "model_family": best["family"],
        "feature_cols": available,
        "postprocess": pp,
        "config": {"H": H, "T": T, "n_states": args.n_states},
        "ftd_thresholds": {"impaired": FTD_IMPAIRED, "caution": FTD_CAUTION},
    }
    if best["family"] == "hmm":
        artifact.update(
            {
                "hmm": best["model"].to_dict(),
                "state_profile": best["profile"],
                "scaler": best["scaler"],
                "risk_lr": best["risk_lr"],
                "pic_labeler": best["pic_labeler"].to_dict(),
            }
        )
    else:
        artifact.update(
            {
                "model": best["model"],
                "model_config": best.get("config", {}),
            }
        )
    joblib.dump(artifact, out_dir / "impairment_hmm.joblib")

    result = {
        "config": {"H": H, "T": T, "n_states": args.n_states},
        "selected_model_family": best["family"],
        "candidate_metrics": {cand["family"]: {"val": cand["val_metrics"], "test": cand["test_metrics"]} for cand in candidates},
        "split_sizes": {"train_users": len(train_users), "val_users": len(val_users), "test_users": len(test_users)},
        "n_features": len(available),
        "risk_vs_error_corr": float(corr) if corr is not None else None,
        "val_auc_roc": best["val_metrics"]["auc_roc"],
        "val_avg_precision": best["val_metrics"]["avg_precision"],
        "test_auc_roc": eval_risk["auc_roc"],
        "test_avg_precision": eval_risk["avg_precision"],
        "feature_family_importance": feature_family_importance,
    }
    if best["family"] == "hmm":
        result["trajectory_stats"] = best["profile"]["trajectory_stats"]
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    if best["family"] == "hmm":
        LOG.info("DONE  family=%s  fwd/bwd=%.2f  risk_corr=%.3f", best["family"], best["profile"]["trajectory_stats"]["fwd_vs_bwd_ratio"], corr)
    else:
        LOG.info("DONE  family=%s  risk_corr=%.3f", best["family"], corr)
    return result

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def build_arg_parser():
    p = argparse.ArgumentParser(description="Driver Impairment HMM with PIC labels (physiological only, no time features)")
    p.add_argument(
        "--data-path",
        default=DEFAULT_BENCHMARK_DATA_PATH,
        help="Path to data directory. Defaults to the synthetic benchmark dataset.",
    )
    p.add_argument("--output-dir", default="result/")
    p.add_argument(
        "--use-old-files",
        action="store_true",
        default=False,
        help="Prefer *_old.csv datasets and fall back to them when the new CSVs are empty.",
    )
    p.add_argument("--lookahead-t", type=int, default=1, help="T (seconds)")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--n-states", type=int, default=3, help="Number of impairment states")
    p.add_argument("--model-family", default="auto", choices=["auto", "hmm", "hgb"])
    p.add_argument("--hmm-cov", default="diag", choices=["diag","spherical","tied"])
    p.add_argument("--hmm-max-iter", type=int, default=150)
    p.add_argument("--hmm-tol", type=float, default=1e-4)
    p.add_argument("--transmat-prior-scale", type=float, default=15.0)
    p.add_argument("--constrain-backward", action="store_true", default=False)
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
