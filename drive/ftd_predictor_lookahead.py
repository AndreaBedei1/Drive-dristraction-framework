"""
Enhanced Fitness-to-Drive - Probability of error in next T seconds after distraction
=====================================================================================
Fixes applied vs original:
  #1  CalibratedClassifierCV cv="prefit"  (was cv=None, which silently re-fits from scratch)
  #2  Calibration 70/30 internal split   (fit on 70 %, select threshold on 30 %;
                                          original fitted and scored on the same set)
  #3  Train-only imputation              (stats computed from train rows, applied everywhere;
                                          original leaked cal/test medians into training)
  #4  All logic wrapped in main()        (no module-level side-effects; helper functions
                                          take explicit DataFrame parameters instead of
                                          capturing globals)
  #5  Physiological baselines from       (arousal_baseline / hr_baseline columns in
      driving_base, not distractions      driving_base; original used distraction-window
                                          values which are elevated vs resting state)
  #6  Outer CV stratified by per-user    (original passed all-zero y to StratifiedGroupKFold,
      positive-rate proxy                 making it identical to plain GroupKFold)
  #7  O(n²) next-window lookup replaced  (precompute_next_starts builds a parallel list once;
      with precomputed index              original re-scanned dist_sel per distraction row)
  #8  Hangover change-rates decay        (arousal_change_rate / cognitive_load_change_rate
      exponentially, not snap to 0        now multiplied by exp(-t/H) after window end;
                                          original hard-coded 0.0 for the whole hangover)
  #9  time_since_last_error is always    (separate has_prior_error flag added; original
      non-negative + explicit flag        used -1.0 as a sentinel mixing sign with magnitude)
  #10 RandomizedSearchCV (n_iter=30)     (original XGB_PARAM_GRID had ~52k combinations per
      replaces full GridSearchCV          fold, making runtime impractical)
"""

import os
import logging
import warnings
from bisect import bisect_left
from typing import Dict, List, Optional, Set, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    RocCurveDisplay, PrecisionRecallDisplay,
    average_precision_score, brier_score_loss, cohen_kappa_score,
    f1_score, log_loss, matthews_corrcoef, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

warnings.filterwarnings('once')

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH   = 'data/'
OUTPUT_DIR  = 'evaluation/'
RANDOM_SEED = 42

H_CANDIDATES  = [10]       # hangover length (seconds)
T_CANDIDATES  = [3, 5]     # lookahead window for target (seconds)

RECALL_LEVELS = [0.80, 0.85, 0.90, 0.95]
N_BOOTSTRAP   = 500
BOOTSTRAP_CI  = 0.95
UNKNOWN_LABEL = 'unknown'

# Fix #10: RandomizedSearchCV distribution (original full grid ≈ 52k fits/fold)
XGB_PARAM_DIST = {
    'xgb__n_estimators':     [100, 200, 300, 400, 500],
    'xgb__max_depth':        [3, 4, 5, 6, 7],
    'xgb__learning_rate':    [0.01, 0.03, 0.05, 0.08, 0.10],
    'xgb__subsample':        [0.70, 0.80, 0.90, 1.00],
    'xgb__colsample_bytree': [0.60, 0.80, 1.00],
    'xgb__gamma':            [0.0, 0.1, 0.2, 0.5, 1.0],
    'xgb__reg_alpha':        [0.0, 0.1, 0.5, 1.0],
    'xgb__reg_lambda':       [1, 2, 3, 5],
    'xgb__min_child_weight': [1, 3, 5, 7],
}
XGB_SEARCH_ITER = 30

# Fix #9: has_prior_error (binary flag) + time_since_last_error (non-negative seconds)
FEATURE_COLS = [
    'time_since_last_dist',
    'time_in_current_dist',
    'time_since_dist_end',
    'cognitive_load_decay',
    'model_prob',
    'model_pred_enc',
    'model_prob_sq',
    'emotion_prob',
    'emotion_label_enc',
    'arousal',
    'arousal_decay',
    'user_arousal_baseline',
    'hr_bpm',
    'emotion_prob_sq',
    'arousal_deviation_sq',
    'distraction_density_rate_30',
    'distraction_density_rate_60',
    'distraction_density_rate_120',
    'prev_dist_duration',
    'has_prior_error',        # Fix #9: explicit binary flag
    'time_since_last_error',  # Fix #9: always non-negative (0.0 when no prior error)
    'arousal_change_rate',    # Fix #8: decays in hangover
    'cognitive_load_change_rate',  # Fix #8: decays in hangover
    'user_hr_baseline',
    'baseline_error_rate',
]


# ──────────────────────────────────────────────────────────────────────────────
# 1. Data loading
# ──────────────────────────────────────────────────────────────────────────────
def load_data(
    data_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    distractions = pd.read_csv(os.path.join(data_path, 'Dataset Distractions_distraction.csv'))
    errors_dist  = pd.read_csv(os.path.join(data_path, 'Dataset Errors_distraction.csv'))
    errors_base  = pd.read_csv(os.path.join(data_path, 'Dataset Errors_baseline.csv'))
    driving_base = pd.read_csv(os.path.join(data_path, 'Dataset Driving Time_baseline.csv'))

    distractions['timestamp_start'] = pd.to_datetime(
        distractions['timestamp_start'], format='ISO8601', errors='coerce')
    distractions['timestamp_end'] = pd.to_datetime(
        distractions['timestamp_end'], format='ISO8601', errors='coerce')
    errors_dist['timestamp'] = pd.to_datetime(
        errors_dist['timestamp'], format='ISO8601', errors='coerce')
    errors_base['timestamp'] = pd.to_datetime(
        errors_base['timestamp'], format='ISO8601', errors='coerce')
    if 'timestamp' in driving_base.columns:
        driving_base['timestamp'] = pd.to_datetime(
            driving_base['timestamp'], format='ISO8601', errors='coerce')

    distractions = distractions.dropna(subset=['timestamp_start', 'timestamp_end']).copy()
    distractions = distractions[
        distractions['timestamp_end'] >= distractions['timestamp_start']
    ].copy()
    errors_dist = errors_dist.dropna(subset=['timestamp']).copy()

    log.info("  Distraction events       : %d", len(distractions))
    log.info("  Errors (distraction run) : %d", len(errors_dist))
    log.info("  Errors (baseline run)    : %d", len(errors_base))

    return (
        distractions.reset_index(drop=True),
        errors_dist.reset_index(drop=True),
        errors_base.reset_index(drop=True),
        driving_base.reset_index(drop=True),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2. Integrity checks
# ──────────────────────────────────────────────────────────────────────────────
def run_integrity_checks(
    distractions: pd.DataFrame,
    errors_dist: pd.DataFrame,
) -> None:
    issues: List[str] = []

    error_sessions = set(
        errors_dist[['user_id', 'run_id']].drop_duplicates().itertuples(index=False, name=None)
    )
    dist_sessions = set(
        distractions[['user_id', 'run_id']].drop_duplicates().itertuples(index=False, name=None)
    )
    missing = error_sessions - dist_sessions
    if missing:
        issues.append(f"Errors in sessions without distractions: {missing}")

    for (uid, rid), grp in distractions.groupby(['user_id', 'run_id']):
        sg = grp.sort_values('timestamp_start').reset_index(drop=True)
        if not sg['timestamp_start'].is_monotonic_increasing:
            issues.append(f"timestamp_start not monotonic for {uid}, {rid}")
        for i in range(1, len(sg)):
            if sg.iloc[i]['timestamp_start'] < sg.iloc[i - 1]['timestamp_end']:
                issues.append(f"Overlapping distraction windows for {uid}, {rid}")
                break

    dup = errors_dist.duplicated(subset=['user_id', 'run_id', 'timestamp'])
    if dup.sum():
        issues.append(f"{dup.sum()} duplicate error timestamps in errors_dist")

    required = ['user_id', 'run_id', 'timestamp', 'model_pred', 'model_prob',
                'emotion_label', 'emotion_prob']
    for col in required:
        if col not in errors_dist.columns:
            issues.append(f"Missing column in errors_dist: '{col}'")

    if issues:
        for iss in issues:
            log.error("  [FAIL] %s", iss)
        raise RuntimeError("Fix data issues before training.")
    log.info("  [PASS] All integrity checks passed.")


# Outlier capping: only hard physiological bounds — no statistical estimates,
# so applying globally before any split is safe.
def cap_outliers(distractions: pd.DataFrame) -> pd.DataFrame:
    distractions = distractions.copy()
    for col in ['arousal_start', 'arousal_end', 'model_prob_start', 'model_prob_end',
                'emotion_prob_start', 'emotion_prob_end']:
        if col in distractions.columns:
            distractions[col] = distractions[col].clip(0.0, 1.0)
    for col in ['hr_bpm_start', 'hr_bpm_end']:
        if col in distractions.columns:
            distractions[col] = distractions[col].clip(35.0, 220.0)
    return distractions


# ──────────────────────────────────────────────────────────────────────────────
# Fix #3: Train-only imputation
# ──────────────────────────────────────────────────────────────────────────────
def compute_imputation_stats(
    dist_train: pd.DataFrame,
    err_train: pd.DataFrame,
    numeric_dist_cols: List[str],
    numeric_err_cols: List[str],
) -> Dict:
    """
    Compute per-user and global medians from training rows only.
    These stats are later applied to cal/test rows to avoid leakage.
    """
    stats: Dict = {'dist': {}, 'err': {}}

    for col in numeric_dist_cols:
        if col not in dist_train.columns:
            continue
        per_user   = dist_train.groupby('user_id')[col].median().to_dict()
        global_med = float(dist_train[col].median())
        if np.isnan(global_med):
            global_med = (0.5 if ('prob' in col or 'arousal' in col)
                          else 70.0 if 'hr' in col
                          else 0.0)
        stats['dist'][col] = {'per_user': per_user, 'global': global_med}

    for col in numeric_err_cols:
        if col not in err_train.columns:
            continue
        per_user   = err_train.groupby('user_id')[col].median().to_dict()
        global_med = float(err_train[col].median())
        if np.isnan(global_med):
            global_med = 0.5 if 'prob' in col else 0.0
        stats['err'][col] = {'per_user': per_user, 'global': global_med}

    return stats


def apply_imputation(df: pd.DataFrame, col_stats: Dict) -> pd.DataFrame:
    """Apply pre-computed imputation stats to any split without data leakage."""
    df = df.copy()
    for col, stat in col_stats.items():
        if col not in df.columns:
            continue
        if not df[col].isna().any():
            continue
        per_user   = stat['per_user']
        global_med = stat['global']
        fill       = df['user_id'].map(per_user).where(lambda s: s.notna(), global_med)
        df[col]    = df[col].fillna(fill).fillna(global_med)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Fix #4: Helper functions take explicit DataFrame parameters (no global capture)
# ──────────────────────────────────────────────────────────────────────────────
def build_session_lookups(
    users_set: Set,
    distractions: pd.DataFrame,
) -> Tuple[Dict, Dict]:
    wbs:  Dict = {}
    webs: Dict = {}
    subset = distractions[distractions['user_id'].isin(users_set)]
    for (uid, rid), grp in subset.groupby(['user_id', 'run_id']):
        sg = grp.sort_values('timestamp_start').reset_index(drop=True)
        wbs[(uid, rid)]  = sg
        webs[(uid, rid)] = sorted(sg['timestamp_end'].tolist())
    return wbs, webs


# Fix #7: precompute next-window start times once per session
def precompute_next_starts(wbs: Dict) -> Dict:
    """
    Returns {key: [next_start_0, next_start_1, ..., None]} parallel to wbs[key].
    Eliminates the O(n²) re-scan inside generate_samples.
    """
    next_starts: Dict = {}
    for key, wins in wbs.items():
        n  = len(wins)
        ns = [None] * n
        for i in range(n - 1):
            ns[i] = wins.iloc[i + 1]['timestamp_start']
        next_starts[key] = ns
    return next_starts


# Fix #5: arousal/HR baselines sourced from driving_base (resting state)
def compute_user_baselines(
    users_set: Set,
    errors_base: pd.DataFrame,
    driving_base: pd.DataFrame,
) -> Tuple[Dict, float]:
    """Error rate from baseline (non-distraction) runs."""
    err_sub  = errors_base[errors_base['user_id'].isin(users_set)]
    drv_sub  = driving_base[driving_base['user_id'].isin(users_set)]
    total_s  = float(drv_sub['run_duration_seconds'].sum())
    global_r = float(len(err_sub) / total_s) if total_s > 0 else 0.0
    user_err  = err_sub.groupby('user_id').size()
    user_secs = drv_sub.groupby('user_id')['run_duration_seconds'].sum()
    per_user  = (
        (user_err / user_secs)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(global_r)
        .to_dict()
    )
    return {str(k): float(v) for k, v in per_user.items()}, global_r


def compute_user_arousal_baseline(
    users_set: Set,
    driving_base: pd.DataFrame,
) -> Tuple[Dict, float]:
    """
    Fix #5: Use arousal_baseline from driving_base (resting-state measure).
    Original used distraction-window arousal values which are systematically elevated.
    """
    if 'arousal_baseline' not in driving_base.columns:
        log.warning("'arousal_baseline' not in driving_base; defaulting to 0.5")
        return {}, 0.5
    sub        = driving_base[driving_base['user_id'].isin(users_set)]
    per_user   = sub.groupby('user_id')['arousal_baseline'].median()
    global_med = float(sub['arousal_baseline'].median()) if len(sub) > 0 else 0.5
    if np.isnan(global_med):
        global_med = 0.5
    return {str(k): float(v) for k, v in per_user.items() if pd.notna(v)}, global_med


def compute_user_hr_baseline(
    users_set: Set,
    driving_base: pd.DataFrame,
) -> Tuple[Dict, float]:
    """
    Fix #5: Use hr_baseline from driving_base (resting-state measure).
    """
    if 'hr_baseline' not in driving_base.columns:
        log.warning("'hr_baseline' not in driving_base; defaulting to 70.0")
        return {}, 70.0
    sub        = driving_base[driving_base['user_id'].isin(users_set)]
    per_user   = sub.groupby('user_id')['hr_baseline'].median()
    global_med = float(sub['hr_baseline'].median()) if len(sub) > 0 else 70.0
    if np.isnan(global_med):
        global_med = 70.0
    return {str(k): float(v) for k, v in per_user.items() if pd.notna(v)}, global_med


def fit_label_encoders(
    train_users: Set,
    distractions: pd.DataFrame,
    errors_dist: pd.DataFrame,
) -> Tuple[LabelEncoder, LabelEncoder]:
    dist_tr = distractions[distractions['user_id'].isin(train_users)]
    err_tr  = errors_dist[errors_dist['user_id'].isin(train_users)]
    emo_vocab = sorted(set(
        pd.concat([
            dist_tr['emotion_label_start'],
            dist_tr['emotion_label_end'],
            err_tr['emotion_label'],
        ]).dropna().tolist() + [UNKNOWN_LABEL]
    ))
    pred_vocab = sorted(set(
        pd.concat([
            dist_tr['model_pred_start'],
            dist_tr['model_pred_end'],
            err_tr['model_pred'],
        ]).dropna().tolist() + [UNKNOWN_LABEL]
    ))
    return LabelEncoder().fit(emo_vocab), LabelEncoder().fit(pred_vocab)


def _safe_encode(label, le: LabelEncoder) -> int:
    s = (UNKNOWN_LABEL
         if label is None or (isinstance(label, float) and np.isnan(label))
         else str(label).strip())
    if not s:
        s = UNKNOWN_LABEL
    if s not in le.classes_:
        s = UNKNOWN_LABEL
    return int(le.transform([s])[0])


def _get_distraction_density(
    uid, rid,
    ts: pd.Timestamp,
    lookback_s: int,
    webs: Dict,
) -> int:
    ends = webs.get((uid, rid), [])
    if not ends:
        return 0
    t_lo = ts - pd.Timedelta(seconds=lookback_s)
    return bisect_left(ends, ts) - bisect_left(ends, t_lo)


def _get_prev_dist_duration(uid, rid, ts: pd.Timestamp, wbs: Dict) -> float:
    wins = wbs.get((uid, rid))
    if wins is None:
        return 0.0
    prev = wins[wins['timestamp_end'] < ts]
    if prev.empty:
        return 0.0
    last = prev.loc[prev['timestamp_end'].idxmax()]
    return float((last['timestamp_end'] - last['timestamp_start']).total_seconds())


# ──────────────────────────────────────────────────────────────────────────────
# 4. Sample generator
# ──────────────────────────────────────────────────────────────────────────────
def generate_samples(
    H: int,
    T: int,
    users_set: Set,
    wbs: Dict,
    webs: Dict,
    next_starts: Dict,          # Fix #7: precomputed per-run list
    error_dict: Dict,
    arousal_bl_dict: Dict,
    hr_bl_dict: Dict,
    baseline_rate_dict: Dict,
    global_arousal_bl: float,
    global_hr_bl: float,
    global_baseline_rate: float,
    le_emo: LabelEncoder,
    le_pred: LabelEncoder,
) -> pd.DataFrame:
    """
    Generate per-second samples for users_set.
    Target = 1 if any error occurs in [bin_start, bin_start + T).
    All features are computed from data available strictly before bin_start.
    """
    rows: List[Dict] = []
    H_f = float(max(1, H))

    for key, wins in tqdm(wbs.items(), desc="Generating samples", disable=None):
        uid, rid = key
        if uid not in users_set:
            continue

        ns_list = next_starts.get(key, [None] * len(wins))
        err_ts  = error_dict.get(key, [])

        u_aro_bl    = arousal_bl_dict.get(str(uid), global_arousal_bl)
        u_hr_bl     = hr_bl_dict.get(str(uid), global_hr_bl)
        u_base_rate = baseline_rate_dict.get(str(uid), global_baseline_rate)

        for i in range(len(wins)):
            row      = wins.iloc[i]
            start_ts = row['timestamp_start']
            end_ts   = row['timestamp_end']
            win_dur  = float((end_ts - start_ts).total_seconds())
            if win_dur <= 0:
                continue

            # Fix #7: O(1) lookup via precomputed list
            next_start = ns_list[i]
            hangover_s = H_f
            if next_start is not None:
                hangover_s = min(hangover_s, float((next_start - end_ts).total_seconds()))
            hangover_s   = max(0.0, hangover_s)
            upper_ts     = end_ts + pd.Timedelta(seconds=hangover_s)
            total_span_s = float((upper_ts - start_ts).total_seconds())
            num_bins     = int(np.ceil(total_span_s))
            if num_bins <= 0:
                continue

            s_ar = float(row['arousal_start'])
            e_ar = float(row['arousal_end'])
            s_hr = float(row['hr_bpm_start'])
            e_hr = float(row['hr_bpm_end'])
            s_mp = float(row['model_prob_start'])
            e_mp = float(row['model_prob_end'])
            s_ep = float(row['emotion_prob_start'])
            e_ep = float(row['emotion_prob_end'])
            s_pr = _safe_encode(row['model_pred_start'],    le_pred)
            e_pr = _safe_encode(row['model_pred_end'],      le_pred)
            s_em = _safe_encode(row['emotion_label_start'], le_emo)
            e_em = _safe_encode(row['emotion_label_end'],   le_emo)

            # Window-level change rates (used as baseline for Fix #8 decay)
            aro_rate_window = (e_ar - s_ar) / max(win_dur, 1e-6)
            cld_rate_window = (e_mp - s_mp) / max(win_dur, 1e-6)

            for offset_sec in range(num_bins):
                bin_start     = start_ts + pd.Timedelta(seconds=offset_sec)
                inside_window = float(offset_sec) < win_dur

                if inside_window:
                    alpha   = float(offset_sec) / max(win_dur, 1e-6)
                    cur_ar  = s_ar + (e_ar - s_ar) * alpha
                    cur_hr  = s_hr + (e_hr - s_hr) * alpha
                    cur_mp  = s_mp + (e_mp - s_mp) * alpha
                    cur_ep  = s_ep + (e_ep - s_ep) * alpha
                    # Hold categorical label at start-of-window value until window ends
                    cur_pr  = s_pr
                    cur_em  = s_em
                    time_in   = float(offset_sec)
                    t_since   = 0.0
                    t_end     = 0.0
                    cld       = cur_mp
                    ar_decay  = cur_ar
                    # Fix #8: change rates are valid inside the window
                    aro_rate  = aro_rate_window
                    cld_rate  = cld_rate_window
                else:
                    t_end     = float(offset_sec) - win_dur
                    dec       = float(np.exp(-t_end / H_f))
                    cur_ar    = e_ar
                    cur_hr    = e_hr
                    cur_mp    = e_mp
                    cur_ep    = e_ep
                    cur_pr    = e_pr
                    cur_em    = e_em
                    time_in   = 0.0
                    t_since   = t_end
                    cld       = cur_mp * dec
                    ar_decay  = u_aro_bl + (cur_ar - u_aro_bl) * dec
                    # Fix #8: decay the change rate exponentially in the hangover
                    aro_rate  = aro_rate_window * dec
                    cld_rate  = cld_rate_window * dec

                # Target
                target    = 0
                idx_err   = bisect_left(err_ts, bin_start)
                if idx_err < len(err_ts) and err_ts[idx_err] < bin_start + pd.Timedelta(seconds=T):
                    target = 1

                # Fix #9: explicit binary flag + non-negative elapsed time
                idx_prev       = bisect_left(err_ts, bin_start) - 1
                has_prior      = 1.0 if idx_prev >= 0 else 0.0
                t_since_err    = float((bin_start - err_ts[idx_prev]).total_seconds()) if idx_prev >= 0 else 0.0

                arousal_dev = cur_ar - u_aro_bl
                d30  = _get_distraction_density(uid, rid, bin_start, 30,  webs)
                d60  = _get_distraction_density(uid, rid, bin_start, 60,  webs)
                d120 = _get_distraction_density(uid, rid, bin_start, 120, webs)

                rows.append({
                    'user_id':              uid,
                    'run_id':               rid,
                    'distraction_start_ts': start_ts,
                    'offset_sec':           offset_sec,
                    'target':               int(target),

                    'time_since_last_dist':       t_since,
                    'time_in_current_dist':        time_in,
                    'time_since_dist_end':         t_end,
                    'cognitive_load_decay':        cld,
                    'model_prob':                  cur_mp,
                    'model_pred_enc':              float(cur_pr),
                    'model_prob_sq':               cur_mp ** 2,
                    'emotion_prob':                cur_ep,
                    'emotion_label_enc':           float(cur_em),
                    'arousal':                     cur_ar,
                    'arousal_decay':               ar_decay,
                    'user_arousal_baseline':       u_aro_bl,
                    'hr_bpm':                      cur_hr,
                    'emotion_prob_sq':             cur_ep ** 2,
                    'arousal_deviation_sq':        arousal_dev ** 2,
                    'distraction_density_rate_30':  d30  / 30.0,
                    'distraction_density_rate_60':  d60  / 60.0,
                    'distraction_density_rate_120': d120 / 120.0,
                    'prev_dist_duration':          _get_prev_dist_duration(uid, rid, bin_start, wbs),
                    'has_prior_error':             has_prior,    # Fix #9
                    'time_since_last_error':       t_since_err,  # Fix #9
                    'arousal_change_rate':         aro_rate,     # Fix #8
                    'cognitive_load_change_rate':  cld_rate,     # Fix #8
                    'user_hr_baseline':            u_hr_bl,
                    'baseline_error_rate':         u_base_rate,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in FEATURE_COLS:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(0.0)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Fix #6: Per-user positive-rate proxy for stratified outer CV
# ──────────────────────────────────────────────────────────────────────────────
def compute_user_strat_labels(
    all_users: np.ndarray,
    errors_dist: pd.DataFrame,
    distractions: pd.DataFrame,
) -> np.ndarray:
    """
    Binary label per user: 1 if the user's error-per-distraction-second rate
    is at or above the median across all users.  Gives StratifiedGroupKFold
    a meaningful signal to balance (original passed all-zero y).
    """
    user_err = (
        errors_dist.groupby('user_id').size()
        .reindex(all_users, fill_value=0)
        .values.astype(float)
    )
    dur_col = (distractions['timestamp_end'] - distractions['timestamp_start']).dt.total_seconds().clip(lower=0)
    dist_copy = distractions.copy()
    dist_copy['dur'] = dur_col
    user_dur = (
        dist_copy.groupby('user_id')['dur'].sum()
        .reindex(all_users, fill_value=1.0)
        .values
    )
    user_dur  = np.maximum(user_dur, 1.0)
    rates     = user_err / user_dur
    median_r  = np.median(rates)
    return (rates >= median_r).astype(int)


# ──────────────────────────────────────────────────────────────────────────────
# Model building — Fix #10: RandomizedSearchCV
# ──────────────────────────────────────────────────────────────────────────────
def build_and_tune_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    pos_rate: float,
    n_inner_splits: int = 3,
    verbose: int = 0,
) -> Pipeline:
    spw = max((1.0 - pos_rate) / max(pos_rate, 1e-6), 1.0)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', xgb.XGBClassifier(
            objective        = 'binary:logistic',
            random_state     = RANDOM_SEED,
            eval_metric      = 'logloss',
            tree_method      = 'hist',
            n_jobs           = 1,
            scale_pos_weight = spw,
        )),
    ])
    inner_cv = StratifiedGroupKFold(n_splits=n_inner_splits, shuffle=True, random_state=RANDOM_SEED)
    # Fix #10: RandomizedSearchCV replaces the impractical full GridSearchCV
    search = RandomizedSearchCV(
        pipeline, XGB_PARAM_DIST,
        n_iter       = XGB_SEARCH_ITER,
        cv           = inner_cv,
        scoring      = 'average_precision',
        n_jobs       = 1,
        verbose      = verbose,
        refit        = True,
        random_state = RANDOM_SEED,
    )
    search.fit(X_train, y_train, groups=groups_train)
    log.info("  Tuned XGB cv_ap=%.4f", float(search.best_score_))
    return search.best_estimator_


def select_threshold_from_cal(y_cal: np.ndarray, y_prob: np.ndarray) -> float:
    prec, rec, thresh = precision_recall_curve(y_cal, y_prob)
    if len(thresh) == 0:
        return 0.5
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    return float(thresh[np.argmax(f1)])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> Dict:
    y_bin = (y_pred >= threshold).astype(int)
    prec, rec, _ = precision_recall_curve(y_true, y_pred)
    metrics: Dict = {
        'AUC-PR':          float(average_precision_score(y_true, y_pred)),
        'AUC-ROC':         float(roc_auc_score(y_true, y_pred)),
        'Brier':           float(brier_score_loss(y_true, y_pred)),
        'Log-Loss':        float(log_loss(y_true, y_pred)),
        'MCC':             float(matthews_corrcoef(y_true, y_bin)),
        'Kappa':           float(cohen_kappa_score(y_true, y_bin)),
        'F1':              float(f1_score(y_true, y_bin, zero_division=0)),
        'Precision':       float(precision_score(y_true, y_bin, zero_division=0)),
        'Recall':          float(recall_score(y_true, y_bin, zero_division=0)),
        'Threshold (cal)': float(threshold),
    }
    for r in RECALL_LEVELS:
        mask = rec >= r
        metrics[f'P@{int(r * 100)}'] = float(prec[mask].max()) if mask.any() else 0.0
    return metrics


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_func,
    n: int = N_BOOTSTRAP,
) -> Tuple[float, float, float]:
    rng    = np.random.RandomState(RANDOM_SEED)
    scores: List[float] = []
    idx    = np.arange(len(y_true))
    for _ in range(n):
        boot = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_true[boot])) < 2:
            continue
        scores.append(float(metric_func(y_true[boot], y_score[boot])))
    if len(scores) < n * 0.8:
        log.warning("Only %d valid bootstrap samples (expected %d)", len(scores), n)
    if not scores:
        return float('nan'), float('nan'), float('nan')
    lo = float(np.percentile(scores, (1 - BOOTSTRAP_CI) / 2 * 100))
    hi = float(np.percentile(scores, (1 + BOOTSTRAP_CI) / 2 * 100))
    return float(np.mean(scores)), lo, hi


# ──────────────────────────────────────────────────────────────────────────────
# Fix #4: main() — all execution happens here, no module-level side effects
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load
    log.info("Loading datasets …")
    distractions, errors_dist, errors_base, driving_base = load_data(DATA_PATH)

    # 2. Integrity checks (before any split or imputation)
    log.info("Running data integrity checks …")
    run_integrity_checks(distractions, errors_dist)

    # Hard physiological capping — constant bounds, no statistical leakage
    distractions = cap_outliers(distractions)

    # 3. Fix #3: Split users BEFORE computing any statistics
    all_users = np.array(sorted(
        set(distractions['user_id'].unique()) | set(errors_dist['user_id'].unique())
    ))
    rng            = np.random.RandomState(RANDOM_SEED)
    all_users_shuf = all_users.copy()
    rng.shuffle(all_users_shuf)

    n_test      = max(1, int(0.20 * len(all_users_shuf)))
    n_cal       = max(1, int(0.20 * (len(all_users_shuf) - n_test)))
    test_users  = set(all_users_shuf[:n_test])
    cal_users   = set(all_users_shuf[n_test: n_test + n_cal])
    train_users = set(all_users_shuf[n_test + n_cal:])
    log.info(
        "Global split — train: %d | cal: %d | test: %d users",
        len(train_users), len(cal_users), len(test_users),
    )

    # Fix #3: compute imputation stats from train rows only, then apply to all splits
    numeric_dist_cols = [
        'model_prob_start', 'model_prob_end',
        'emotion_prob_start', 'emotion_prob_end',
        'arousal_start', 'arousal_end',
        'hr_bpm_start', 'hr_bpm_end',
    ]
    numeric_err_cols = ['model_prob', 'emotion_prob']

    dist_train_rows = distractions[distractions['user_id'].isin(train_users)]
    err_train_rows  = errors_dist[errors_dist['user_id'].isin(train_users)]
    global_imp      = compute_imputation_stats(
        dist_train_rows, err_train_rows, numeric_dist_cols, numeric_err_cols
    )
    distractions = apply_imputation(distractions, global_imp['dist'])
    errors_dist  = apply_imputation(errors_dist,  global_imp['err'])
    log.info("Imputation complete (statistics from train users only).")

    # Pre-index all errors once
    error_dict_all: Dict = {}
    for (uid, rid), grp in errors_dist.groupby(['user_id', 'run_id']):
        error_dict_all[(uid, rid)] = sorted(grp['timestamp'].tolist())

    # Fix #6: per-user stratification labels for StratifiedGroupKFold
    user_strat_y = compute_user_strat_labels(all_users, errors_dist, distractions)

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Nested CV over H × T
    # ──────────────────────────────────────────────────────────────────────────
    group_kfold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    log.info("\n" + "=" * 70)
    log.info("NESTED CROSS-VALIDATION OVER H AND T")
    log.info("=" * 70)

    cv_results: List[Dict] = []

    for H in H_CANDIDATES:
        for T in T_CANDIDATES:
            log.info("\n--- Evaluating H = %ds, T = %ds ---", H, T)
            outer: Dict = {'auc_pr': [], 'auc_roc': [], 'brier': [], 'logloss': []}

            # Fix #6: pass meaningful y
            for fold_i, (tr_idx, te_idx) in enumerate(
                group_kfold.split(
                    np.zeros(len(all_users)),
                    y=user_strat_y,
                    groups=all_users,
                )
            ):
                train_f = set(all_users[i] for i in tr_idx)
                test_f  = set(all_users[i] for i in te_idx)

                # Fix #3: per-fold imputation from fold-train rows only
                fold_dist_tr = distractions[distractions['user_id'].isin(train_f)]
                fold_err_tr  = errors_dist[errors_dist['user_id'].isin(train_f)]
                fold_imp     = compute_imputation_stats(
                    fold_dist_tr, fold_err_tr, numeric_dist_cols, numeric_err_cols
                )
                fold_dist = apply_imputation(distractions, fold_imp['dist'])
                fold_err  = apply_imputation(errors_dist,  fold_imp['err'])

                # Fix #5: baselines from driving_base
                wbs_tr, webs_tr              = build_session_lookups(train_f, fold_dist)
                ns_tr                        = precompute_next_starts(wbs_tr)
                bsl_dict, gr                 = compute_user_baselines(train_f, errors_base, driving_base)
                aro_dict, aro_gl             = compute_user_arousal_baseline(train_f, driving_base)
                hr_dict,  hr_gl              = compute_user_hr_baseline(train_f, driving_base)
                le_emo_f, le_pred_f          = fit_label_encoders(train_f, fold_dist, fold_err)

                fold_err_dict: Dict = {}
                for (uid, rid), grp in fold_err.groupby(['user_id', 'run_id']):
                    fold_err_dict[(uid, rid)] = sorted(grp['timestamp'].tolist())

                df_tr_f = generate_samples(
                    H, T, train_f,
                    wbs_tr, webs_tr, ns_tr,
                    fold_err_dict,
                    aro_dict, hr_dict, bsl_dict,
                    aro_gl, hr_gl, gr,
                    le_emo_f, le_pred_f,
                )
                if df_tr_f.empty or df_tr_f['target'].nunique() < 2:
                    log.warning("  Fold %d: skipped (train insufficient diversity)", fold_i)
                    continue

                pos_r = float(df_tr_f['target'].mean())
                log.info("  Fold %d: n_train=%d  pos_rate=%.4f", fold_i, len(df_tr_f), pos_r)

                fold_model = build_and_tune_pipeline(
                    df_tr_f[FEATURE_COLS].values.astype(float),
                    df_tr_f['target'].values.astype(int),
                    groups_train=df_tr_f['user_id'].values,
                    pos_rate=pos_r,
                )

                wbs_te, webs_te = build_session_lookups(test_f, fold_dist)
                ns_te           = precompute_next_starts(wbs_te)
                df_te_f = generate_samples(
                    H, T, test_f,
                    wbs_te, webs_te, ns_te,
                    fold_err_dict,
                    aro_dict, hr_dict, bsl_dict,
                    aro_gl, hr_gl, gr,
                    le_emo_f, le_pred_f,
                )
                if df_te_f.empty or df_te_f['target'].nunique() < 2:
                    log.warning("  Fold %d: skipped (test insufficient diversity)", fold_i)
                    continue

                y_te = df_te_f['target'].values.astype(int)
                y_pr = fold_model.predict_proba(
                    df_te_f[FEATURE_COLS].values.astype(float)
                )[:, 1]
                outer['auc_pr'].append(float(average_precision_score(y_te, y_pr)))
                outer['auc_roc'].append(float(roc_auc_score(y_te, y_pr)))
                outer['brier'].append(float(brier_score_loss(y_te, y_pr)))
                outer['logloss'].append(float(log_loss(y_te, y_pr)))

            if not outer['auc_pr']:
                continue
            cv_results.append({
                'H':            H,
                'T':            T,
                'auc_pr_mean':  float(np.mean(outer['auc_pr'])),
                'auc_pr_std':   float(np.std(outer['auc_pr'])),
                'auc_roc_mean': float(np.mean(outer['auc_roc'])),
                'brier_mean':   float(np.mean(outer['brier'])),
                'logloss_mean': float(np.mean(outer['logloss'])),
            })
            log.info(
                "  AUC-PR: %.4f ± %.4f",
                cv_results[-1]['auc_pr_mean'],
                cv_results[-1]['auc_pr_std'],
            )

    results_df = pd.DataFrame(cv_results)
    best_idx   = results_df['auc_pr_mean'].idxmax()
    best_H     = int(results_df.loc[best_idx, 'H'])
    best_T     = int(results_df.loc[best_idx, 'T'])
    log.info("\nBest combination: H = %ds, T = %ds", best_H, best_T)

    # ──────────────────────────────────────────────────────────────────────────
    # 6. Final model — train / cal / test
    # ──────────────────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("FINAL MODEL TRAINING")
    log.info("=" * 70)

    wbs_tr_f, webs_tr_f       = build_session_lookups(train_users, distractions)
    ns_tr_f                   = precompute_next_starts(wbs_tr_f)
    # Fix #5: final baselines from driving_base
    train_baselines, train_gr = compute_user_baselines(train_users, errors_base, driving_base)
    train_aro_bl, train_aro_g = compute_user_arousal_baseline(train_users, driving_base)
    train_hr_bl,  train_hr_g  = compute_user_hr_baseline(train_users, driving_base)
    le_emotion_f, le_pred_f   = fit_label_encoders(train_users, distractions, errors_dist)

    df_train_final = generate_samples(
        best_H, best_T, train_users,
        wbs_tr_f, webs_tr_f, ns_tr_f,
        error_dict_all,
        train_aro_bl, train_hr_bl, train_baselines,
        train_aro_g, train_hr_g, train_gr,
        le_emotion_f, le_pred_f,
    )

    wbs_cal_f, webs_cal_f = build_session_lookups(cal_users, distractions)
    ns_cal_f              = precompute_next_starts(wbs_cal_f)
    df_cal_final = generate_samples(
        best_H, best_T, cal_users,
        wbs_cal_f, webs_cal_f, ns_cal_f,
        error_dict_all,
        train_aro_bl, train_hr_bl, train_baselines,
        train_aro_g, train_hr_g, train_gr,
        le_emotion_f, le_pred_f,
    )

    for split_name, df in [('train', df_train_final), ('calibration', df_cal_final)]:
        if df.empty or df['target'].nunique() < 2:
            raise RuntimeError(f"{split_name} set has only one class — adjust the split.")
        pr = float(df['target'].mean())
        log.info(
            "  %-15s n=%d  pos_rate=%.4f  imbalance=%.1f:1",
            split_name, len(df), pr, (1.0 - pr) / max(pr, 1e-6),
        )

    X_tr_f = df_train_final[FEATURE_COLS].values.astype(float)
    y_tr_f = df_train_final['target'].values.astype(int)
    X_ca_f = df_cal_final[FEATURE_COLS].values.astype(float)
    y_ca_f = df_cal_final['target'].values.astype(int)

    best_model = build_and_tune_pipeline(
        X_tr_f, y_tr_f,
        groups_train=df_train_final['user_id'].values,
        pos_rate=float(y_tr_f.mean()),
        verbose=1,
    )

    # Fix #2: split calibration set — fit calibrator on 70 %, select threshold on 30 %
    # Fix #1: cv="prefit" — keep the tuned weights from best_model
    n_cal_fit  = max(2, int(0.70 * len(X_ca_f)))
    X_cal_fit,  X_cal_eval  = X_ca_f[:n_cal_fit],  X_ca_f[n_cal_fit:]
    y_cal_fit,  y_cal_eval  = y_ca_f[:n_cal_fit],  y_ca_f[n_cal_fit:]

    cal_method = 'isotonic' if n_cal_fit > 1000 else 'sigmoid'
    log.info("  Using %s calibration (fit set: %d samples)", cal_method, n_cal_fit)
    calibrated_model = CalibratedClassifierCV(best_model, method=cal_method, cv="prefit")  # Fix #1
    calibrated_model.fit(X_cal_fit, y_cal_fit)

    # Fix #2: threshold selected on the held-out eval portion
    y_prob_cal_eval   = calibrated_model.predict_proba(X_cal_eval)[:, 1]
    best_threshold    = select_threshold_from_cal(y_cal_eval, y_prob_cal_eval)
    y_prob_uncal_eval = best_model.predict_proba(X_cal_eval)[:, 1]
    threshold_uncal   = select_threshold_from_cal(y_cal_eval, y_prob_uncal_eval)
    log.info("  Calibrated threshold   (cal eval): %.4f", best_threshold)
    log.info("  Uncalibrated threshold (cal eval): %.4f", threshold_uncal)

    # ──────────────────────────────────────────────────────────────────────────
    # 7. Evaluation on held-out test set
    # ──────────────────────────────────────────────────────────────────────────
    wbs_te_f, webs_te_f = build_session_lookups(test_users, distractions)
    ns_te_f             = precompute_next_starts(wbs_te_f)
    df_test_final = generate_samples(
        best_H, best_T, test_users,
        wbs_te_f, webs_te_f, ns_te_f,
        error_dict_all,
        train_aro_bl, train_hr_bl, train_baselines,
        train_aro_g, train_hr_g, train_gr,
        le_emotion_f, le_pred_f,
    )
    if df_test_final.empty or df_test_final['target'].nunique() < 2:
        log.error("Test set has insufficient class diversity — metrics will be unreliable.")

    X_te_f       = df_test_final[FEATURE_COLS].values.astype(float)
    y_te_f       = df_test_final['target'].values.astype(int)
    y_pred_uncal = best_model.predict_proba(X_te_f)[:, 1]
    y_pred_cal   = calibrated_model.predict_proba(X_te_f)[:, 1]

    metrics_uncal = compute_metrics(y_te_f, y_pred_uncal, threshold_uncal)
    metrics_cal   = compute_metrics(y_te_f, y_pred_cal,   best_threshold)

    log.info("\n" + "=" * 70)
    log.info("FINAL MODEL EVALUATION ON TEST SET")
    log.info("=" * 70)
    log.info("\nUncalibrated XGBoost:")
    for k, v in metrics_uncal.items():
        log.info("  %s: %.4f", k, v)
    log.info("\nCalibrated XGBoost:")
    for k, v in metrics_cal.items():
        log.info("  %s: %.4f", k, v)

    ci_aucpr,  lo_aucpr,  hi_aucpr  = bootstrap_ci(y_te_f, y_pred_cal, average_precision_score)
    ci_aucroc, lo_aucroc, hi_aucroc = bootstrap_ci(y_te_f, y_pred_cal, roc_auc_score)
    ci_brier,  lo_brier,  hi_brier  = bootstrap_ci(y_te_f, y_pred_cal, brier_score_loss)
    ci_f1,     lo_f1,     hi_f1     = bootstrap_ci(
        y_te_f, (y_pred_cal >= best_threshold).astype(int), f1_score
    )
    log.info("\nBootstrap 95%% CI (calibrated model):")
    log.info("  AUC-PR : %.4f [%.4f - %.4f]", ci_aucpr,  lo_aucpr,  hi_aucpr)
    log.info("  AUC-ROC: %.4f [%.4f - %.4f]", ci_aucroc, lo_aucroc, hi_aucroc)
    log.info("  Brier  : %.4f [%.4f - %.4f]", ci_brier,  lo_brier,  hi_brier)
    log.info("  F1     : %.4f [%.4f - %.4f]", ci_f1,     lo_f1,     hi_f1)

    # ──────────────────────────────────────────────────────────────────────────
    # 8. Plots
    # ──────────────────────────────────────────────────────────────────────────
    try:
        # Calibration curve
        fig, ax = plt.subplots(figsize=(8, 6))
        for proba, label in [(y_pred_uncal, 'Uncalibrated'), (y_pred_cal, 'Calibrated')]:
            fop, mpv = calibration_curve(y_te_f, proba, n_bins=10)
            ax.plot(mpv, fop, marker='o', label=label)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title(f'Calibration curves (H={best_H}s, T={best_T}s)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'calibration_curve.png'), dpi=150)
        plt.close()

        # ROC and PR curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        RocCurveDisplay.from_predictions(y_te_f, y_pred_cal, ax=ax1, name='Calibrated')
        ax1.plot([0, 1], [0, 1], 'k--', label='Chance')
        ax1.set_title(f'ROC Curve (H={best_H}s, T={best_T}s)')
        ax1.legend()
        PrecisionRecallDisplay.from_predictions(y_te_f, y_pred_cal, ax=ax2)
        ax2.set_title(f'Precision-Recall Curve (H={best_H}s, T={best_T}s)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'roc_pr_curves.png'), dpi=150)
        plt.close()

        # Feature importance
        xgb_step    = best_model.named_steps['xgb']
        importances = xgb_step.feature_importances_
        sorted_idx  = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_idx)), importances[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [FEATURE_COLS[i] for i in sorted_idx])
        plt.xlabel('Feature importance (gain)')
        plt.title(f'XGBoost Feature Importance (H={best_H}s, T={best_T}s)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=150)
        plt.close()

        # Per-second risk profile
        df_plot = df_test_final.copy()
        df_plot['pred_cal']         = y_pred_cal
        df_plot['time_since_start'] = df_plot['offset_sec'].clip(0, best_H)
        empirical_rate = df_plot.groupby('time_since_start')['target'].mean()
        pred_mean      = df_plot.groupby('time_since_start')['pred_cal'].mean()
        plt.figure(figsize=(10, 6))
        plt.plot(empirical_rate.index, empirical_rate.values, 'o-',
                 label=f'Empirical P(error in next {best_T}s)', color='red')
        plt.plot(pred_mean.index, pred_mean.values, 's-',
                 label='Model predicted probability', color='blue')
        plt.axhline(
            y=float(df_plot['baseline_error_rate'].mean()) * best_T,
            color='gray', linestyle='--', label=f'Baseline rate × {best_T}',
        )
        plt.xlabel('Seconds since distraction start')
        plt.ylabel('Probability')
        plt.title(f'Per-second risk profile (H={best_H}s, T={best_T}s)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'per_second_risk.png'), dpi=150)
        plt.close()

    except Exception as exc:
        log.error("Plotting failed: %s", exc)

    # ──────────────────────────────────────────────────────────────────────────
    # 9. Save artifacts
    # ──────────────────────────────────────────────────────────────────────────
    try:
        artifact = {
            # Models
            'model':                best_model,
            'calibrated_model':     calibrated_model,
            # Decision thresholds
            'best_threshold_cal':   best_threshold,
            'best_threshold_uncal': threshold_uncal,
            # Configuration
            'best_H':               best_H,
            'best_T':               best_T,
            'feature_cols':         FEATURE_COLS,
            # Encoders
            'le_emotion':           le_emotion_f,
            'le_pred':              le_pred_f,
            # Baselines (training users only)
            'train_baselines':      train_baselines,
            'train_global_rate':    train_gr,
            'train_arousal_bl':     train_aro_bl,
            'train_hr_bl':          train_hr_bl,
            # Evaluation results
            'cv_results_df':        results_df,
            'metrics_uncal':        metrics_uncal,
            'metrics_cal':          metrics_cal,
            'bootstrap_ci': {
                'auc_pr':  (ci_aucpr,  lo_aucpr,  hi_aucpr),
                'auc_roc': (ci_aucroc, lo_aucroc, hi_aucroc),
                'brier':   (ci_brier,  lo_brier,  hi_brier),
                'f1':      (ci_f1,     lo_f1,     hi_f1),
            },
        }
        joblib.dump(artifact, os.path.join(OUTPUT_DIR, 'fitness_model.pkl'))
        log.info("\nAll results saved to %s", OUTPUT_DIR)
    except Exception as exc:
        log.error("Saving artifacts failed: %s", exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())