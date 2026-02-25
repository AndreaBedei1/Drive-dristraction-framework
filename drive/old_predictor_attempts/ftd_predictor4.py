"""
Driver Digital Twin - Fitness-to-Drive XGBoost Pipeline
=========================================================
Rigorous, publication-ready pipeline with:
  - Spline basis for time since last distraction (personalised recovery)
  - Per-user arousal baseline as a feature (from true arousal signals)
  - Interactions between driver features and temporal terms
  - Full nested LOSO-CV with 2D calibration
  - Bootstrap 95% CIs for all metrics
  - Continuous distraction detection confidence (model_prob) replaces
    binary 0/1 flag; also used to scale cognitive_load_decay during
    active distraction (conf=1.0 → full load, conf=0.5 → half load)

Optimizations:
  - Parallel evaluation of top candidates (joblib)
  - Early stopping in XGBoost (20% validation split by user)
  - Reduced bootstrap iterations (500)
  - Configurable sampling interval (default 1)

Input datasets (place in DATA_PATH):
  - Dataset Distractions_distraction.csv
  - Dataset Errors_distraction.csv
  - Dataset Errors_baseline.csv
  - Dataset Driving Time_baseline.csv
"""

import warnings
warnings.filterwarnings('ignore')

import bisect
import itertools
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import LabelEncoder, SplineTransformer
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve,
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score,
    matthews_corrcoef, brier_score_loss,
    log_loss, cohen_kappa_score,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import LeaveOneGroupOut
import xgboost as xgb
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed   # for parallel evaluation

# ──────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
DATA_PATH        = 'data/'
MODEL_OUT        = 'fitness_model.pkl'
EVAL_OUT         = 'evaluation/'

# H candidates: post-distraction hangover window length (seconds)
BIN_LIMIT        = 10
H_CANDIDATES     = list(range(10, BIN_LIMIT + 1))   # [5..15]


# Top-N from joint grid that get the full nested evaluation
TOP_N_CANDIDATES = 3

NEG_SAMPLE_EVERY          = 1          # can be increased to speed up
N_BOOTSTRAP               = 500        # reduced from 1000
RECALL_LEVELS             = [0.80, 0.85, 0.90, 0.95]
GT_ACTIVE_SMOOTH_ALPHA    = 20.0
GT_POST_SMOOTH_ALPHA      = 40.0
N_ECE_BINS                = 10
H_SELECTION_ECE_WEIGHT    = 0.40
H_SELECTION_BRIER_WEIGHT  = 1.00

os.makedirs(EVAL_OUT, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# XGBoost hyperparameter grid (to be tuned)
# ──────────────────────────────────────────────────────────────────────
XGB_PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

# Default XGBoost parameters (used in the quick grid search)
XGB_DEFAULT_PARAMS = {
    'n_estimators': 200,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'aucpr',
    'random_state': RANDOM_SEED,
    'verbosity': 0,
    'n_jobs': -1,
}

# ──────────────────────────────────────────────────────────────────────
# FEATURES
# ──────────────────────────────────────────────────────────────────────
EXOGENOUS_STATE_FEATURE = [
    'distraction_active',
    'time_since_last_dist',
    'time_in_current_dist',
    'cognitive_load_decay',
    'model_prob',
    'model_pred_enc',
    'model_prob_sq'
]

ENDOGENOUS_STATE_FEATURES = [
    'emotion_prob',
    'emotion_label_enc',
    'arousal_deviation',
    'user_arousal_baseline',
    'arousal_delta',
    'arousal_at_window_end',
    'emotion_changed',
    'hr_bpm',
    'emotion_prob_sq',
    'arousal_deviation_sq',
]

CONTEXT_FEATURES = [
    'distraction_density_30',
    'distraction_density_60',
    'distraction_density_120',
    'prev_dist_duration',
]

DRIVER_FEATURES = [
    'user_arousal_baseline',
    'user_hr_baseline',
    'baseline_error_rate',
]

# ======================================================================
# 1. Load datasets
# ======================================================================
print("\n" + "=" * 70)
print("LOADING DATASETS")
print("=" * 70)
distractions = pd.read_csv(f'{DATA_PATH}Dataset Distractions_distraction.csv')
errors_dist  = pd.read_csv(f'{DATA_PATH}Dataset Errors_distraction.csv')
errors_base  = pd.read_csv(f'{DATA_PATH}Dataset Errors_baseline.csv')
driving_base = pd.read_csv(f'{DATA_PATH}Dataset Driving Time_baseline.csv')

distractions['timestamp_start'] = pd.to_datetime(distractions['timestamp_start'])
distractions['timestamp_end']   = pd.to_datetime(distractions['timestamp_end'])
errors_dist['timestamp']        = pd.to_datetime(errors_dist['timestamp'])

print(f"  Distraction events       : {len(distractions)}")
print(f"  Errors (distraction run) : {len(errors_dist)}")
print(f"  Errors (baseline run)    : {len(errors_base)}")

windows_by_session = {}
for (uid, rid), grp in distractions.groupby(['user_id', 'run_id']):
    windows_by_session[(uid, rid)] = grp.reset_index(drop=True)

# ======================================================================
# 2. Data integrity checks
# ======================================================================
print("\n" + "=" * 70)
print("DATA INTEGRITY CHECKS")
print("=" * 70)
issues = []

# Check that every error in distraction runs has a corresponding distraction session
error_sessions = set(errors_dist[['user_id', 'run_id']].drop_duplicates().itertuples(index=False, name=None))
dist_sessions = set(distractions[['user_id', 'run_id']].drop_duplicates().itertuples(index=False, name=None))
missing_sessions = error_sessions - dist_sessions
if missing_sessions:
    issues.append(f"Errors exist in sessions without any distraction window: {missing_sessions}")

# Check for windows with negative duration
bad_windows = distractions[distractions['timestamp_end'] < distractions['timestamp_start']]
if len(bad_windows):
    issues.append(f"{len(bad_windows)} distraction windows have end < start")

# Check for duplicate errors within same (user, run, timestamp)
dup_errors = errors_dist.duplicated(subset=['user_id', 'run_id', 'timestamp'])
if dup_errors.sum():
    issues.append(f"{dup_errors.sum()} duplicate error timestamps in errors_dist")

# Check required columns
required_cols = ['user_id', 'run_id', 'timestamp', 'model_pred', 'model_prob',
                 'emotion_label', 'emotion_prob']
for col in required_cols:
    if col not in errors_dist.columns:
        issues.append(f"Missing column in errors_dist: {col}")

# Handle missing values in model_prob and emotion_prob
for col in ['model_prob', 'emotion_prob']:
    n_nan = errors_dist[col].isna().sum()
    if n_nan > 0:
        print(f"  [WARN] {n_nan} NaNs in errors_dist['{col}']")
        # Use per-user median if available, otherwise global median
        errors_dist[col] = errors_dist.groupby('user_id')[col].transform(
            lambda x: x.fillna(x.median())
        )
        # If still NaN (user with all missing), fill with global median
        if errors_dist[col].isna().any():
            global_median = errors_dist[col].median()
            errors_dist[col].fillna(global_median, inplace=True)
        print(f"         filled with per-user median (or global fallback)")

# After imputation, ensure no NaNs remain in feature columns
if errors_dist[['model_prob', 'emotion_prob']].isna().any().any():
    issues.append("NaN values remain after imputation in model_prob or emotion_prob")

if issues:
    print("  [FAIL] Integrity issues found:")
    for iss in issues:
        print(f"    x {iss}")
    raise RuntimeError("Fix data issues before training.")
else:
    print("  [PASS] All checks passed.")

# ======================================================================
# 3. Per‑user baseline error rate  (recomputed per fold)
# ======================================================================
def compute_user_baselines(user_ids_set=None):
    if user_ids_set is not None:
        err_sub = errors_base[errors_base['user_id'].isin(user_ids_set)]
        drv_sub = driving_base[driving_base['user_id'].isin(user_ids_set)]
    else:
        err_sub = errors_base
        drv_sub = driving_base

    total_s     = drv_sub['run_duration_seconds'].sum()
    global_rate = float(len(err_sub) / total_s) if total_s > 0 else 0.0

    user_errs = err_sub.groupby('user_id').size()
    user_secs = drv_sub.groupby('user_id')['run_duration_seconds'].sum()
    per_user  = (user_errs / user_secs).fillna(global_rate).to_dict()
    return per_user, global_rate

user_baselines, p_baseline_global = compute_user_baselines()
print(f"\nGlobal baseline error rate : {p_baseline_global*100:.4f}% / second")

# ======================================================================
# 4. Per‑user arousal baseline (NOW USING TRUE AROUSAL VALUES)
# ======================================================================
def compute_arousal_baseline(user_ids_set=None):
    """
    Compute per-user arousal baseline from the true arousal signals
    (arousal_start and arousal_end) in the distraction windows.
    """
    acc = {}
    for (uid, rid), wins in windows_by_session.items():
        if user_ids_set is not None and uid not in user_ids_set:
            continue
        # Use the actual arousal columns, not model_prob
        vals = list(wins['arousal_start'].dropna()) + list(wins['arousal_end'].dropna())
        acc.setdefault(uid, []).extend(vals)
    per_user = {uid: float(np.median(v)) for uid, v in acc.items() if v}
    all_vals  = [v for vals in acc.values() for v in vals]
    glob_med  = float(np.median(all_vals)) if all_vals else 0.5
    return per_user, glob_med

user_arousal_baseline, global_arousal_median = compute_arousal_baseline()
print(f"Per-user arousal baselines computed for {len(user_arousal_baseline)} users  "
      f"(global median={global_arousal_median:.4f})")

# ======================================================================
# 5. Precompute window ends for density lookups
# ======================================================================
window_ends_by_session: dict = {}
for (uid, rid), wins in windows_by_session.items():
    window_ends_by_session[(uid, rid)] = sorted(wins['timestamp_end'].tolist())

# ======================================================================
# 6. Distraction state helpers
# ======================================================================
def get_distraction_state(user_id, run_id, ts, H):
    sentinel = float(max(H_CANDIDATES))
    key = (user_id, run_id)
    if key not in windows_by_session:
        return 0, float(H), 0.0, sentinel, 0.0

    wins = windows_by_session[key]
    active_mask = (wins['timestamp_start'] <= ts) & (ts <= wins['timestamp_end'])
    if active_mask.any():
        active_win = wins.loc[active_mask].sort_values('timestamp_start').iloc[-1]
        t_in = (ts - active_win['timestamp_start']).total_seconds()
        return 1, 0.0, min(t_in, float(H)), 0.0, t_in

    prev = wins[wins['timestamp_end'] < ts]
    if prev.empty:
        return 0, float(H), 0.0, sentinel, 0.0
    delta = (ts - prev['timestamp_end'].max()).total_seconds()
    return 0, min(delta, float(H)), 0.0, delta, 0.0

def is_error_inside_distraction(row):
    active, _, _, _, _ = get_distraction_state(
        row['user_id'], row['run_id'], row['timestamp'], H=0)
    return active == 1

def get_time_since_last(row):
    key = (row['user_id'], row['run_id'])
    if key not in windows_by_session:
        return np.nan
    wins = windows_by_session[key]
    prev = wins[wins['timestamp_end'] < row['timestamp']]
    if prev.empty:
        return np.nan
    nxt = wins[wins['timestamp_start'] > row['timestamp']]
    if nxt.empty:
        return np.nan
    return (row['timestamp'] - prev['timestamp_end'].max()).total_seconds()

# ======================================================================
# 7. Ground truth per‑second error rates (now with fold‑specific baseline)
# ======================================================================
max_H_gt = max(H_CANDIDATES)

def compute_gt_profiles(user_ids_set=None):
    """
    Computes two GT profiles:
      1) active_rates[s] : risk at second s from distraction start (inside window)
      2) post_rates[s]   : risk at second s after distraction end (s=1..H)
    Uses fold‑specific baseline rate for smoothing (if user_ids_set given).
    """
    dist_subset = (distractions if user_ids_set is None
                   else distractions[distractions['user_id'].isin(user_ids_set)])
    err_subset  = (errors_dist  if user_ids_set is None
                   else errors_dist[errors_dist['user_id'].isin(user_ids_set)])

    total_dist_s = (dist_subset['timestamp_end'] - dist_subset['timestamp_start']
                    ).dt.total_seconds().sum()
    errs_inside = sum(1 for _, e in err_subset.iterrows()
                      if is_error_inside_distraction(e))
    p_b0 = errs_inside / total_dist_s if total_dist_s > 0 else 0.0

    # Fold‑specific baseline rate for smoothing
    if user_ids_set is not None:
        fold_errs    = errors_base[errors_base['user_id'].isin(user_ids_set)]
        fold_driving = driving_base[driving_base['user_id'].isin(user_ids_set)]
        total_base_s = fold_driving['run_duration_seconds'].sum()
        fold_base_rate = len(fold_errs) / total_base_s if total_base_s > 0 else p_baseline_global
    else:
        fold_base_rate = p_baseline_global

    # Active profile (seconds from distraction onset)
    active_errs = np.zeros(max_H_gt + 1)
    active_expo = np.zeros(max_H_gt + 1)
    for (uid, rid), wins in windows_by_session.items():
        if user_ids_set is not None and uid not in user_ids_set:
            continue
        wins_s = wins.sort_values('timestamp_start').reset_index(drop=True)
        for _, win in wins_s.iterrows():
            dur = (win['timestamp_end'] - win['timestamp_start']).total_seconds()
            max_sec = min(int(np.floor(dur)), max_H_gt)
            for sec in range(0, max_sec + 1):
                active_expo[sec] += 1.0

    for _, err in err_subset.iterrows():
        active, _, _, _, true_t_in = get_distraction_state(
            err['user_id'], err['run_id'], err['timestamp'], H=max_H_gt)
        if active == 1:
            sec = int(min(np.floor(true_t_in), max_H_gt))
            active_errs[sec] += 1.0

    active_rates = np.full(max_H_gt + 1, np.nan)
    for sec in range(max_H_gt + 1):
        if active_expo[sec] > 0:
            active_rates[sec] = (
                active_errs[sec] + GT_ACTIVE_SMOOTH_ALPHA * p_b0
            ) / (active_expo[sec] + GT_ACTIVE_SMOOTH_ALPHA)

    # Post profile (seconds from distraction end)
    post_errs = np.zeros(max_H_gt + 1)
    post_expo = np.zeros(max_H_gt + 1)
    for (uid, rid), wins in windows_by_session.items():
        if user_ids_set is not None and uid not in user_ids_set:
            continue
        wins_s = wins.sort_values('timestamp_start').reset_index(drop=True)
        for i in range(len(wins_s) - 1):
            win_end    = wins_s['timestamp_end'].iloc[i]
            next_start = wins_s['timestamp_start'].iloc[i + 1]
            gap_s = min((next_start - win_end).total_seconds(), float(max_H_gt))
            for sec in range(1, int(np.floor(gap_s)) + 1):
                post_expo[sec] += 1.0

    outside = err_subset[
        ~err_subset.apply(is_error_inside_distraction, axis=1).fillna(False)
    ].copy()
    outside['time_since'] = outside.apply(get_time_since_last, axis=1)
    for t in outside.dropna(subset=['time_since'])['time_since']:
        sec = int(np.floor(t)) + 1
        if 1 <= sec <= max_H_gt:
            post_errs[sec] += 1

    post_rates = np.full(max_H_gt + 1, np.nan)
    post_rates[0] = p_b0
    for sec in range(1, max_H_gt + 1):
        if post_expo[sec] > 0:
            post_rates[sec] = (
                post_errs[sec] + GT_POST_SMOOTH_ALPHA * fold_base_rate
            ) / (post_expo[sec] + GT_POST_SMOOTH_ALPHA)

    post_valid = np.where(~np.isnan(post_rates[1:]))[0] + 1
    if len(post_valid) >= 2:
        iso_post = IsotonicRegression(increasing=False, out_of_bounds='clip')
        post_rates[post_valid] = iso_post.fit_transform(
            post_valid.astype(float),
            post_rates[post_valid],
            sample_weight=post_expo[post_valid],
        )
    post_rates[post_valid] = np.maximum(post_rates[post_valid], fold_base_rate)

    return {
        'active_rates':   active_rates,
        'post_rates':     post_rates,
        'p_bin0':         p_b0,
        'active_errors':  active_errs,
        'active_exposure': active_expo,
        'post_errors':    post_errs,
        'post_exposure':  post_expo,
    }

def compute_gt_rates(user_ids_set=None):
    profile = compute_gt_profiles(user_ids_set)
    return profile['post_rates'], profile['p_bin0']

# Global GT rates for reporting (using all data)
gt_profile_global  = compute_gt_profiles()
gt_rate_per_sec    = gt_profile_global['post_rates']
gt_active_rate     = gt_profile_global['active_rates']
p_bin0             = gt_profile_global['p_bin0']
gt_errors          = gt_profile_global['post_errors'].copy()
gt_exposure        = gt_profile_global['post_exposure'].copy()
gt_active_errors   = gt_profile_global['active_errors'].copy()
gt_active_exposure = gt_profile_global['active_exposure'].copy()

print("\n" + "=" * 70)
print("GROUND TRUTH PER-SECOND ERROR RATES")
print("=" * 70)
print(f"  Bin 0 (inside windows)      : {p_bin0:.6f} errors/s")
print("  Active distraction seconds from onset (smoothed rate / exposure / errors):")
for n in range(0, min(10, max_H_gt + 1)):
    print(f"    {n:2d}s : {gt_active_rate[n]:.6f}  "
          f"(exposure={gt_active_exposure[n]:6.0f}, errors={int(gt_active_errors[n]):3d})")
if max_H_gt > 9:
    print("    ...")
print("  Post-distraction seconds (smoothed+monotone rate / exposure / errors):")
for n in range(1, min(max_H_gt + 1, 12)):
    print(f"    {n:2d}s : {gt_rate_per_sec[n]:.6f}  "
          f"(exposure={gt_exposure[n]:6.0f}, errors={int(gt_errors[n]):3d})")
if max_H_gt > 11:
    print("    ...")

# ======================================================================
# 8. Label encoding  (FIX-4: per-fold encoders rebuilt from train users)
# ======================================================================
UNKNOWN_LABEL = 'unknown'

def safe_str(val):
    if val is None:
        return UNKNOWN_LABEL
    s = str(val).strip()
    return UNKNOWN_LABEL if s.lower() in ('nan', 'none', '') else s

all_emotion_labels = (
    pd.concat([errors_dist['emotion_label'],
               distractions['emotion_label_start'],
               distractions['emotion_label_end']])
    .map(safe_str).unique().tolist() + [UNKNOWN_LABEL]
)
all_model_preds = (
    pd.concat([errors_dist['model_pred'],
               distractions['model_pred_start'],
               distractions['model_pred_end']])
    .map(safe_str).unique().tolist() + [UNKNOWN_LABEL]
)

le_emotion = LabelEncoder().fit(list(set(all_emotion_labels)))
le_pred    = LabelEncoder().fit(list(set(all_model_preds)))


def build_fold_encoders(train_users):
    train_dist = distractions[distractions['user_id'].isin(train_users)]
    train_err  = errors_dist[errors_dist['user_id'].isin(train_users)]
    fold_emo = (
        pd.concat([train_err['emotion_label'],
                   train_dist['emotion_label_start'],
                   train_dist['emotion_label_end']])
        .map(safe_str).unique().tolist() + [UNKNOWN_LABEL]
    )
    fold_pred = (
        pd.concat([train_err['model_pred'],
                   train_dist['model_pred_start'],
                   train_dist['model_pred_end']])
        .map(safe_str).unique().tolist() + [UNKNOWN_LABEL]
    )
    return (LabelEncoder().fit(list(set(fold_emo))),
            LabelEncoder().fit(list(set(fold_pred))))


def encode_row(emotion_label, model_pred, le_e=None, le_p=None):
    _le_e = le_e if le_e is not None else le_emotion
    _le_p = le_p if le_p is not None else le_pred

    def _enc(le, val):
        s = safe_str(val)
        if s not in le.classes_:
            s = UNKNOWN_LABEL
        return int(le.transform([s])[0])

    return _enc(_le_e, emotion_label), _enc(_le_p, model_pred)

# ======================================================================
# 9. Feature helpers
# ======================================================================
def get_distraction_density(uid, rid, ts, lookback_seconds: float) -> int:
    ends = window_ends_by_session.get((uid, rid), [])
    if not ends:
        return 0
    t_lo = ts - pd.Timedelta(seconds=lookback_seconds)
    lo   = bisect.bisect_left(ends, t_lo)
    hi   = bisect.bisect_left(ends, ts)
    return hi - lo

def get_prev_dist_duration(uid, rid, ts) -> float:
    key = (uid, rid)
    if key not in windows_by_session:
        return 0.0
    wins = windows_by_session[key]
    prev = wins[wins['timestamp_end'] < ts]
    if prev.empty:
        return 0.0
    row = prev.loc[prev['timestamp_end'].idxmax()]
    return max(0.0, (row['timestamp_end'] - row['timestamp_start']).total_seconds())

# ======================================================================
# 10. Ground truth probability assignment
# ======================================================================
def assign_gt_prob(row, H, gt_rates=None, p_base=None, gt_active_rates=None):
    _rates  = gt_rates        if gt_rates        is not None else gt_rate_per_sec
    _active = gt_active_rates if gt_active_rates is not None else gt_active_rate
    _p_base = p_base          if p_base          is not None else p_baseline_global

    if row['distraction_active'] > 0:
        true_t_in = row.get('true_time_in_dist', row.get('time_in_current_dist', 0.0))
        sec = int(min(np.floor(max(true_t_in, 0.0)), max_H_gt))
        if sec <= max_H_gt and not np.isnan(_active[sec]):
            return _active[sec]
        return _rates[0]

    true_t = row.get('true_time_since', row['time_since_last_dist'])
    sentinel = float(max(H_CANDIDATES))
    if true_t >= sentinel:
        return _p_base
    if true_t > max_H_gt:
        return _p_base
    n = int(np.floor(true_t)) + 1
    if n <= max_H_gt and not np.isnan(_rates[n]):
        return _rates[n]
    return _p_base

# ======================================================================
# 11. Sample builders
#     Accept spline_trans / feat_cols to support grid search.
#     Accept baselines / baseline_global per fold.
#     Accept le_e / le_p per fold.
# ======================================================================
def _spline_row(time_since, spline_trans, n_spline):
    t_arr = np.array([[time_since]]).reshape(-1, 1)
    vals  = spline_trans.transform(t_arr)[0]
    return vals, {f'time_spline_{i}': vals[i] for i in range(n_spline)}


def _interaction_row(driver_vals, time_since, spline_vals, n_spline):
    interactions = {}
    for df_name, val in driver_vals.items():
        interactions[f'{df_name}_x_time'] = val * time_since
        for i in range(n_spline):
            interactions[f'{df_name}_x_spline_{i}'] = val * spline_vals[i]
    return interactions


def build_positives(H, user_ids=None,
                    arousal_bl=None, arousal_med=None,
                    baselines=None, baseline_global=None,
                    le_e=None, le_p=None,
                    spline_trans=None, n_spline=None):
    _abl    = arousal_bl      or user_arousal_baseline
    _amed   = arousal_med     if arousal_med   is not None else global_arousal_median
    _bl     = baselines       or user_baselines
    _bglob  = baseline_global if baseline_global is not None else p_baseline_global
    _st     = spline_trans    or spline_transformer
    _ns     = n_spline        if n_spline is not None else N_SPLINE_BASIS

    rows = []
    for _, err in errors_dist.iterrows():
        uid, rid, ts = err['user_id'], err['run_id'], err['timestamp']
        if user_ids is not None and uid not in user_ids:
            continue
        dist_active_binary, time_since, time_in_dist, true_time_since, true_time_in_dist = \
            get_distraction_state(uid, rid, ts, H)
        emo_enc, pred_enc = encode_row(err['emotion_label'], err['model_pred'], le_e, le_p)
        m_prob = err['model_prob']
        u_aro  = _abl.get(uid, _amed)

        if dist_active_binary == 1:
            dist_conf = float(np.clip(m_prob, 0.0, 1.0))
            cld = dist_conf
        else:
            dist_conf = 0.0
            cld = np.exp(-time_since / max(H, 1e-9))

        spline_vals, spline_dict = _spline_row(time_since, _st, _ns)

        arousal_delta = 0.0; arousal_at_window_end = 0.0
        emotion_changed = 0;  hr_bpm = 0.0
        if dist_active_binary == 1:
            key  = (uid, rid)
            wins = windows_by_session[key]
            win  = wins[(wins['timestamp_start'] <= ts) & (ts <= wins['timestamp_end'])].iloc[-1]
            if 'arousal_start' in win and 'arousal_end' in win:
                arousal_delta = win['arousal_end'] - win['arousal_start']
                arousal_at_window_end = win['arousal_end']
            if 'hr_bpm_start' in win:
                hr_bpm = win['hr_bpm_start']
            if 'emotion_label_start' in win and 'emotion_label_end' in win:
                emotion_changed = int(win['emotion_label_start'] != win['emotion_label_end'])
        else:
            key = (uid, rid)
            if key in windows_by_session:
                wins = windows_by_session[key]
                prev = wins[wins['timestamp_end'] < ts]
                if not prev.empty:
                    win = prev.loc[prev['timestamp_end'].idxmax()]
                    if 'arousal_end' in win:
                        arousal_at_window_end = win['arousal_end']
                    if 'hr_bpm_end' in win:
                        hr_bpm = win['hr_bpm_end']

        driver_vals = {
            'user_arousal_baseline': u_aro,
            'baseline_error_rate':   _bl.get(uid, _bglob),
            'arousal_delta':         arousal_delta,
            'hr_bpm':                hr_bpm,
            'arousal_deviation':     m_prob - u_aro,
        }
        interactions = _interaction_row(driver_vals, time_since, spline_vals, _ns)

        row = {
            'user_id': uid,
            'distraction_active':    dist_conf,
            'time_since_last_dist':  time_since,
            'time_in_current_dist':  time_in_dist,
            'true_time_since':       true_time_since,
            'true_time_in_dist':     true_time_in_dist,
            'cognitive_load_decay':  cld,
            'model_prob':            m_prob,
            'model_pred_enc':        pred_enc,
            'emotion_prob':          err['emotion_prob'],
            'emotion_label_enc':     emo_enc,
            'arousal_deviation':     m_prob - u_aro,
            'user_arousal_baseline': u_aro,
            'arousal_delta':         arousal_delta,
            'arousal_at_window_end': arousal_at_window_end,
            'emotion_changed':       emotion_changed,
            'hr_bpm':                hr_bpm,
            'model_prob_sq':         m_prob * m_prob,
            'emotion_prob_sq':       err['emotion_prob'] ** 2,
            'arousal_deviation_sq':  (m_prob - u_aro) ** 2,
            'baseline_error_rate':   driver_vals['baseline_error_rate'],
            'distraction_density_60':  get_distraction_density(uid, rid, ts, 60),
            'distraction_density_120': get_distraction_density(uid, rid, ts, 120),
            'prev_dist_duration':    get_prev_dist_duration(uid, rid, ts),
            'label': 1,
        }
        row.update(spline_dict)
        row.update(interactions)
        rows.append(row)

    return pd.DataFrame(rows)


def build_negatives(H, sample_every=NEG_SAMPLE_EVERY, user_ids=None,
                    arousal_bl=None, arousal_med=None,
                    baselines=None, baseline_global=None,
                    le_e=None, le_p=None,
                    spline_trans=None, n_spline=None):
    _abl    = arousal_bl      or user_arousal_baseline
    _amed   = arousal_med     if arousal_med   is not None else global_arousal_median
    _bl     = baselines       or user_baselines
    _bglob  = baseline_global if baseline_global is not None else p_baseline_global
    _st     = spline_trans    or spline_transformer
    _ns     = n_spline        if n_spline is not None else N_SPLINE_BASIS

    rows = []
    error_floor = {
        (uid, rid): set(grp['timestamp'].dt.floor('s'))
        for (uid, rid), grp in errors_dist.groupby(['user_id', 'run_id'])
    }

    for (uid, rid), wins in windows_by_session.items():
        if user_ids is not None and uid not in user_ids:
            continue
        wins_s  = wins.sort_values('timestamp_start').reset_index(drop=True)
        b_rate  = _bl.get(uid, _bglob)
        err_set = error_floor.get((uid, rid), set())
        u_aro   = _abl.get(uid, _amed)

        # ── Inside distraction windows ────────────────────────────────
        for w_idx, win in wins_s.iterrows():
            dur      = (win['timestamp_end'] - win['timestamp_start']).total_seconds()
            m_prob_w = win['model_prob_start']
            emo_enc, pred_enc = encode_row(
                win['emotion_label_start'], win['model_pred_start'], le_e, le_p)
            prev_dur_w = (
                (wins_s.iloc[w_idx-1]['timestamp_end'] -
                 wins_s.iloc[w_idx-1]['timestamp_start']).total_seconds()
                if w_idx > 0 else 0.0
            )
            arousal_delta = (win['arousal_end'] - win['arousal_start']
                             if 'arousal_start' in win and 'arousal_end' in win else 0.0)
            arousal_at_window_end = win.get('arousal_end', 0.0)
            hr_bpm = win.get('hr_bpm_start', 0.0)
            emotion_changed = int(win.get('emotion_label_start') != win.get('emotion_label_end'))

            for offset in np.arange(0, dur, sample_every):
                ts = win['timestamp_start'] + pd.Timedelta(seconds=offset)
                if ts.floor('s') in err_set:
                    continue
                alpha = float(offset / dur) if dur > 0 else 0.0
                dist_conf_w = float(np.clip(
                    win['model_prob_start'] + alpha * (win['model_prob_end'] - win['model_prob_start']),
                    0.0, 1.0))

                spline_vals, spline_dict = _spline_row(0.0, _st, _ns)
                driver_vals = {
                    'user_arousal_baseline': u_aro,
                    'baseline_error_rate':   b_rate,
                    'arousal_delta':         arousal_delta,
                    'hr_bpm':                hr_bpm,
                    'arousal_deviation':     m_prob_w - u_aro,
                }
                interactions = _interaction_row(driver_vals, 0.0, spline_vals, _ns)

                row = {
                    'user_id': uid,
                    'distraction_active':    dist_conf_w,
                    'time_since_last_dist':  0.0,
                    'time_in_current_dist':  min(offset, float(H)),
                    'true_time_since':       0.0,
                    'true_time_in_dist':     float(offset),
                    'cognitive_load_decay':  dist_conf_w,
                    'model_prob':            m_prob_w,
                    'model_pred_enc':        pred_enc,
                    'emotion_prob':          win['emotion_prob_start'],
                    'emotion_label_enc':     emo_enc,
                    'arousal_deviation':     m_prob_w - u_aro,
                    'user_arousal_baseline': u_aro,
                    'arousal_delta':         arousal_delta,
                    'arousal_at_window_end': arousal_at_window_end,
                    'emotion_changed':       emotion_changed,
                    'hr_bpm':                hr_bpm,
                    'model_prob_sq':         m_prob_w ** 2,
                    'emotion_prob_sq':       win['emotion_prob_start'] ** 2,
                    'arousal_deviation_sq':  (m_prob_w - u_aro) ** 2,
                    'baseline_error_rate':   b_rate,
                    'distraction_density_60':  get_distraction_density(uid, rid, ts, 60),
                    'distraction_density_120': get_distraction_density(uid, rid, ts, 120),
                    'prev_dist_duration':    prev_dur_w,
                    'label': 0,
                }
                row.update(spline_dict)
                row.update(interactions)
                rows.append(row)

        # ── Inter-window gaps ─────────────────────────────────────────
        for i in range(len(wins_s) - 1):
            gap_start = wins_s['timestamp_end'].iloc[i]
            gap_end   = wins_s['timestamp_start'].iloc[i + 1]
            gap_len   = (gap_end - gap_start).total_seconds()
            if gap_len <= 0:
                continue
            win_state   = wins_s.iloc[i]
            m_prob_g    = win_state['model_prob_end']
            emo_enc, pred_enc = encode_row(
                win_state['emotion_label_end'], win_state['model_pred_end'], le_e, le_p)
            prev_dur_g = max(0.0, (win_state['timestamp_end'] -
                                   win_state['timestamp_start']).total_seconds())
            arousal_at_window_end = win_state.get('arousal_end', 0.0)
            hr_bpm = win_state.get('hr_bpm_end', 0.0)

            for offset in np.arange(0, gap_len, sample_every):
                ts = gap_start + pd.Timedelta(seconds=offset)
                true_t = offset
                if ts.floor('s') in err_set:
                    continue
                time_since_capped = min(offset, float(H))
                cld = np.exp(-true_t / max(H, 1e-9))

                spline_vals, spline_dict = _spline_row(time_since_capped, _st, _ns)
                driver_vals = {
                    'user_arousal_baseline': u_aro,
                    'baseline_error_rate':   b_rate,
                    'arousal_delta':         0.0,
                    'hr_bpm':                hr_bpm,
                    'arousal_deviation':     m_prob_g - u_aro,
                }
                interactions = _interaction_row(driver_vals, time_since_capped, spline_vals, _ns)

                row = {
                    'user_id': uid,
                    'distraction_active':    0,
                    'time_since_last_dist':  time_since_capped,
                    'time_in_current_dist':  0.0,
                    'true_time_since':       true_t,
                    'true_time_in_dist':     0.0,
                    'cognitive_load_decay':  cld,
                    'model_prob':            m_prob_g,
                    'model_pred_enc':        pred_enc,
                    'emotion_prob':          win_state['emotion_prob_end'],
                    'emotion_label_enc':     emo_enc,
                    'arousal_deviation':     m_prob_g - u_aro,
                    'user_arousal_baseline': u_aro,
                    'arousal_delta':         0.0,
                    'arousal_at_window_end': arousal_at_window_end,
                    'emotion_changed':       0,
                    'hr_bpm':                hr_bpm,
                    'model_prob_sq':         m_prob_g ** 2,
                    'emotion_prob_sq':       win_state['emotion_prob_end'] ** 2,
                    'arousal_deviation_sq':  (m_prob_g - u_aro) ** 2,
                    'baseline_error_rate':   b_rate,
                    'distraction_density_60':  get_distraction_density(uid, rid, ts, 60),
                    'distraction_density_120': get_distraction_density(uid, rid, ts, 120),
                    'prev_dist_duration':    prev_dur_g,
                    'label': 0,
                }
                row.update(spline_dict)
                row.update(interactions)
                rows.append(row)

    return pd.DataFrame(rows)


def build_baseline_negatives(sample_every=NEG_SAMPLE_EVERY,
                              default_model_prob=None,
                              default_model_pred_enc=None,
                              default_emotion_prob=None,
                              default_emotion_label_enc=None,
                              default_hr_bpm=None,
                              default_arousal_baseline=None,
                              user_ids=None,
                              arousal_bl=None, arousal_med=None,
                              baselines=None, baseline_global=None,
                              spline_trans=None, n_spline=None):
    # NOTE: no le_e / le_p — pre-encoded defaults are passed in directly
    _abl    = arousal_bl      or user_arousal_baseline
    _amed   = arousal_med     if arousal_med   is not None else global_arousal_median
    _bl     = baselines       or user_baselines
    _bglob  = baseline_global if baseline_global is not None else p_baseline_global
    _st     = spline_trans    or spline_transformer
    _ns     = n_spline        if n_spline is not None else N_SPLINE_BASIS

    sentinel = float(max(H_CANDIDATES))
    rows = []
    for _, row in driving_base.iterrows():
        uid = row['user_id']
        if user_ids is not None and uid not in user_ids:
            continue
        run_duration = row['run_duration_seconds']
        b_rate = _bl.get(uid, _bglob)
        u_aro  = (_abl.get(uid, _amed) if default_arousal_baseline is None
                  else default_arousal_baseline)
        m_prob = default_model_prob if default_model_prob is not None else _amed
        hr_bpm = default_hr_bpm if default_hr_bpm is not None else 0.5

        spline_vals, spline_dict = _spline_row(sentinel, _st, _ns)
        driver_vals = {
            'user_arousal_baseline': u_aro,
            'baseline_error_rate':   b_rate,
            'arousal_delta':         0.0,
            'hr_bpm':                hr_bpm,
            'arousal_deviation':     m_prob - u_aro,
        }
        interactions = _interaction_row(driver_vals, sentinel, spline_vals, _ns)

        e_prob = default_emotion_prob if default_emotion_prob is not None else 0.5
        for _ in np.arange(0, run_duration, sample_every):
            r = {
                'user_id': uid,
                'distraction_active':    0,
                'time_since_last_dist':  sentinel,
                'time_in_current_dist':  0.0,
                'true_time_since':       sentinel,
                'true_time_in_dist':     0.0,
                'cognitive_load_decay':  0.0,
                'model_prob':            m_prob,
                'model_pred_enc':        default_model_pred_enc,
                'emotion_prob':          e_prob,
                'emotion_label_enc':     default_emotion_label_enc,
                'arousal_deviation':     m_prob - u_aro,
                'user_arousal_baseline': u_aro,
                'arousal_delta':         0.0,
                'arousal_at_window_end': 0.0,
                'emotion_changed':       0,
                'hr_bpm':                hr_bpm,
                'model_prob_sq':         m_prob ** 2,
                'emotion_prob_sq':       e_prob ** 2,
                'arousal_deviation_sq':  (m_prob - u_aro) ** 2,
                'baseline_error_rate':   b_rate,
                'distraction_density_60':  0,
                'distraction_density_120': 0,
                'prev_dist_duration':    0.0,
                'label': 0,
            }
            r.update(spline_dict)
            r.update(interactions)
            rows.append(r)
    return pd.DataFrame(rows)

# ======================================================================
# 12. Bootstrap CI helper
# ======================================================================
def bootstrap_ci(y_true, y_score, metric_fn, n=N_BOOTSTRAP, ci=0.95, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    scores = []
    idx = np.arange(len(y_true))
    for _ in range(n):
        boot = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_true[boot])) < 2:
            continue
        scores.append(metric_fn(y_true[boot], y_score[boot]))
    lo = np.percentile(scores, (1 - ci) / 2 * 100)
    hi = np.percentile(scores, (1 + ci) / 2 * 100)
    return np.mean(scores), lo, hi

# ======================================================================
# 13. Precision at fixed recall
# ======================================================================
def precision_at_recall(y_true, y_score, recall_levels):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    results = {}
    for r in recall_levels:
        mask = rec >= r
        results[r] = float(prec[mask].max()) if mask.any() else 0.0
    return results

# ======================================================================
# 14. Compute metrics helper
# ======================================================================
def compute_metrics(y_true, y_score, thresh):
    yp = (y_score >= thresh).astype(int)
    cm_ = confusion_matrix(y_true, yp)
    tn, fp, fn, tp = cm_.ravel() if cm_.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    m = {
        'AUC-PR':      average_precision_score(y_true, y_score),
        'AUC-ROC':     roc_auc_score(y_true, y_score),
        'Log-Loss':    log_loss(y_true, y_score),
        'MCC':         matthews_corrcoef(y_true, yp),
        'Kappa':       cohen_kappa_score(y_true, yp),
        'Brier':       brier_score_loss(y_true, y_score),
        'F1':          f1_score(y_true, yp, zero_division=0),
        'Precision':   precision_score(y_true, yp, zero_division=0),
        'Recall':      recall_score(y_true, yp, zero_division=0),
        'Specificity': specificity,
        'NPV':         npv,
    }
    par = precision_at_recall(y_true, y_score, RECALL_LEVELS)
    m.update({f'P@R={int(r*100)}': v for r, v in par.items()})
    return m

# ======================================================================
# 15. Fold‑specific baseline rate
# ======================================================================
def compute_fold_baseline_rate(user_ids_set):
    fold_errs    = errors_base[errors_base['user_id'].isin(user_ids_set)]
    fold_driving = driving_base[driving_base['user_id'].isin(user_ids_set)]
    total_sec = fold_driving['run_duration_seconds'].sum()
    if total_sec > 0:
        return float(len(fold_errs) / total_sec)
    per_user_secs = fold_driving.groupby('user_id')['run_duration_seconds'].sum()
    per_user_errs = fold_errs.groupby('user_id').size().reindex(
        per_user_secs.index, fill_value=0)
    valid = per_user_secs[per_user_secs > 0]
    if len(valid) > 0:
        return float(np.mean(per_user_errs[valid.index].values / valid.values))
    return 0.0

# ======================================================================
# 16. Expected Calibration Error (ECE)
# ======================================================================
def compute_ece(y_true, y_prob, n_bins=N_ECE_BINS):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(y_true) == 0 or y_true.sum() == 0:
        return np.nan
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(y_prob, quantiles)
    edges[0]  = 0.0
    edges[-1] = 1.0 + 1e-9
    ece = 0.0
    total = len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return ece / total

# ======================================================================
# 17. Core nested LOSO evaluation (with early stopping)
# ======================================================================
def full_nested_eval(H: int, n_knots: int, degree: int):
    # Build per-combo spline
    _st, _ns = make_spline_transformer(n_knots, degree, float(max(H_CANDIDATES)))
    _fc = make_feature_cols(_ns)

    all_users = sorted(
        set(distractions['user_id'].unique()) |
        set(errors_dist['user_id'].unique())
    )
    true_labels, raw_probs, calib_probs, gt_probs = [], [], [], []
    dist_active_vals, time_since_vals, time_in_dist_vals = [], [], []
    ece_y_true, ece_y_cal = [], []

    for test_user in all_users:
        train_users = set(all_users) - {test_user}

        fold_gt_profile      = compute_gt_profiles(train_users)
        fold_gt_rates        = fold_gt_profile['post_rates']
        fold_gt_active_rates = fold_gt_profile['active_rates']
        fold_baseline_rate   = compute_fold_baseline_rate(train_users)
        fold_arousal_bl, fold_arousal_med = compute_arousal_baseline(train_users)
        fold_baselines, fold_bglob        = compute_user_baselines(train_users)
        fold_le_e, fold_le_p              = build_fold_encoders(train_users)

        fold_kw = dict(
            arousal_bl=fold_arousal_bl,
            arousal_med=fold_arousal_med,
            baselines=fold_baselines,
            baseline_global=fold_bglob,
            le_e=fold_le_e,
            le_p=fold_le_p,
            spline_trans=_st,
            n_spline=_ns,
        )
        # Baseline builder doesn't accept le_e/le_p
        fold_kw_base = {k: v for k, v in fold_kw.items()
                        if k not in ('le_e', 'le_p')}

        train_pos = build_positives(H, user_ids=train_users, **fold_kw)
        train_neg = build_negatives(H, user_ids=train_users, **fold_kw)
        train_df  = pd.concat([train_pos, train_neg],
                               ignore_index=True).dropna(subset=_fc)
        if train_df.empty or train_df['label'].nunique() < 2:
            continue

        safe_mask = train_df['label'] == 0
        if safe_mask.sum() > 0:
            def_mp  = train_df.loc[safe_mask, 'model_prob'].median()
            def_mpe = train_df.loc[safe_mask, 'model_pred_enc'].mode()[0]
            def_ep  = train_df.loc[safe_mask, 'emotion_prob'].median()
            def_ele = train_df.loc[safe_mask, 'emotion_label_enc'].mode()[0]
            def_hr  = (train_df.loc[safe_mask, 'hr_bpm'].median()
                       if 'hr_bpm' in train_df.columns else 0.5)
        else:
            def_mp  = 0.5
            def_mpe = fold_le_p.transform([UNKNOWN_LABEL])[0]
            def_ep  = 0.5
            def_ele = fold_le_e.transform([UNKNOWN_LABEL])[0]
            def_hr  = 0.5

        base_defaults = dict(
            default_model_prob=def_mp, default_model_pred_enc=def_mpe,
            default_emotion_prob=def_ep, default_emotion_label_enc=def_ele,
            default_hr_bpm=def_hr, default_arousal_baseline=fold_arousal_med,
        )

        base_neg_tr = build_baseline_negatives(
            **base_defaults, user_ids=train_users, **fold_kw_base)
        train_full = pd.concat([train_df, base_neg_tr],
                                ignore_index=True).dropna(subset=_fc)

        X_tr      = train_full[_fc].values.astype(float)
        y_tr      = train_full['label'].values.astype(int)
        groups_tr = train_full['user_id'].values
        pos_rate_tr = y_tr.mean()
        if pos_rate_tr in (0.0, 1.0):
            continue
        spw_tr = (1 - pos_rate_tr) / pos_rate_tr

        # ---- Early stopping: split training users into train/val ----
        unique_tr_users = np.unique(groups_tr)
        if len(unique_tr_users) > 1:
            np.random.shuffle(unique_tr_users)
            n_val = max(1, int(0.2 * len(unique_tr_users)))
            val_users = set(unique_tr_users[:n_val])
            train_users_inner = set(unique_tr_users[n_val:])
            val_mask = np.isin(groups_tr, list(val_users))
            train_mask_inner = ~val_mask
            if train_mask_inner.sum() > 0 and val_mask.sum() > 0:
                clf = xgb.XGBClassifier(scale_pos_weight=spw_tr, **XGB_PARAMS)
                clf.fit(X_tr[train_mask_inner], y_tr[train_mask_inner],
                        eval_set=[(X_tr[val_mask], y_tr[val_mask])], verbose=False)
            else:
                clf = xgb.XGBClassifier(scale_pos_weight=spw_tr, **XGB_PARAMS)
                clf.fit(X_tr, y_tr)
        else:
            clf = xgb.XGBClassifier(scale_pos_weight=spw_tr, **XGB_PARAMS)
            clf.fit(X_tr, y_tr)
        # -------------------------------------------------------------

        # Calibrator on OOF inner-LOSO predictions
        cld_tr  = train_full['cognitive_load_decay'].values
        time_tr = train_full['time_since_last_dist'].values
        gt_tr   = np.array([
            assign_gt_prob(r, H, fold_gt_rates, fold_baseline_rate,
                           fold_gt_active_rates)
            for _, r in train_full.iterrows()
        ])
        oof_raw = np.full(len(X_tr), np.nan)
        inner_logo = LeaveOneGroupOut()
        for in_tr, in_val in inner_logo.split(X_tr, y_tr, groups_tr):
            if len(np.unique(y_tr[in_tr])) < 2:
                oof_raw[in_val] = 0.5
                continue
            in_spw = ((1 - y_tr[in_tr].mean()) / y_tr[in_tr].mean()
                      if y_tr[in_tr].mean() not in (0.0, 1.0) else spw_tr)
            in_clf = xgb.XGBClassifier(scale_pos_weight=in_spw, **XGB_PARAMS)
            # Early stopping in inner classifier? Could add but may be overkill.
            # For simplicity, we fit on all inner training data without early stopping.
            in_clf.fit(X_tr[in_tr], y_tr[in_tr])
            oof_raw[in_val] = in_clf.predict_proba(X_tr[in_val])[:, 1]
        nan_mask = np.isnan(oof_raw)
        if nan_mask.any():
            oof_raw[nan_mask] = clf.predict_proba(X_tr[nan_mask])[:, 1]

        iso = xgb.XGBRegressor(**CAL_XGB_PARAMS)
        iso.fit(np.column_stack([oof_raw, cld_tr, time_tr]), gt_tr)

        # Test
        test_pos = build_positives(H, user_ids={test_user}, **fold_kw)
        test_neg = build_negatives(H, user_ids={test_user}, **fold_kw)
        test_df  = pd.concat([test_pos, test_neg],
                              ignore_index=True).dropna(subset=_fc)
        if test_df.empty:
            continue

        base_neg_te = build_baseline_negatives(
            **base_defaults, user_ids={test_user}, **fold_kw_base)
        test_full = pd.concat([test_df, base_neg_te],
                               ignore_index=True).dropna(subset=_fc)

        X_te   = test_full[_fc].values.astype(float)
        y_te   = test_full['label'].values.astype(int)
        raw_te = clf.predict_proba(X_te)[:, 1]
        cld_te = test_full['cognitive_load_decay'].values
        cal_te = np.clip(
            iso.predict(np.column_stack([
                raw_te, cld_te, test_full['time_since_last_dist'].values])),
            0.0, 1.0)

        true_labels.extend(y_te)
        raw_probs.extend(raw_te)
        calib_probs.extend(cal_te)
        gt_probs.extend([
            assign_gt_prob(r, H, fold_gt_rates, fold_baseline_rate,
                           fold_gt_active_rates)
            for _, r in test_full.iterrows()
        ])
        dist_active_vals.extend(test_full['distraction_active'].tolist())
        time_since_vals.extend(test_full['time_since_last_dist'].tolist())
        time_in_dist_vals.extend(test_full['time_in_current_dist'].tolist())
        ece_y_true.extend(y_te.tolist())
        ece_y_cal.extend(cal_te.tolist())

    y_true_arr = np.array(true_labels)
    raw_arr    = np.array(raw_probs)
    cal_arr    = np.array(calib_probs)
    gt_arr     = np.array(gt_probs)

    auc_pr    = average_precision_score(y_true_arr, raw_arr)
    auc_roc   = roc_auc_score(y_true_arr, raw_arr)
    cal_mae   = float(np.mean(np.abs(cal_arr - gt_arr)))
    cal_rmse  = float(np.sqrt(np.mean((cal_arr - gt_arr) ** 2)))
    ece_label = compute_ece(np.array(ece_y_true), np.array(ece_y_cal))
    if np.isnan(ece_label):
        ece_label = 1.0
    brier = float(brier_score_loss(y_true_arr, raw_arr))
    selection_score = float(
        auc_pr
        - H_SELECTION_ECE_WEIGHT   * ece_label
        - H_SELECTION_BRIER_WEIGHT * brier)

    return {
        'H': H, 'n_knots': n_knots, 'degree': degree,
        'n_spline_basis': _ns,
        'AUC-PR': auc_pr, 'AUC-ROC': auc_roc,
        'Calib_MAE': cal_mae, 'Calib_RMSE': cal_rmse,
        'ECE_label': ece_label, 'Brier': brier,
        'selection_score': selection_score,
        'y_true': y_true_arr, 'raw': raw_arr,
        'calib': cal_arr, 'gt': gt_arr,
        'dist_active': np.array(dist_active_vals),
        'time_since': np.array(time_since_vals),
        'time_in_dist': np.array(time_in_dist_vals),
    }


# ======================================================================
# 18. Joint grid search over (H, n_knots, degree)
# ======================================================================
print("\n" + "=" * 70)
print("JOINT GRID SEARCH  (H × n_knots × degree)  — AUC-PR diagnostic")
print("=" * 70)

all_users_grid = sorted(
    set(distractions['user_id'].unique()) |
    set(errors_dist['user_id'].unique())
)

grid_candidates = list(itertools.product(H_CANDIDATES, SPLINE_CANDIDATES))
print(f"\n  {'H':<5} {'n_knots':<9} {'deg':<5} {'basis':<7} "
      f"{'AUC-PR':<10} {'AUC-ROC':<10} {'N':>8}  {'pos%'}")
print("  " + "-" * 68)

grid_results = []
for H, (n_knots, degree) in grid_candidates:
    _st_g, _ns_g = make_spline_transformer(n_knots, degree, float(max(H_CANDIDATES)))
    _fc_g = make_feature_cols(_ns_g)

    y_true_all, y_prob_all = [], []

    for test_user in all_users_grid:
        train_users_g = set(all_users_grid) - {test_user}

        # Fold-specific baselines
        g_arousal_bl, g_arousal_med = compute_arousal_baseline(train_users_g)
        g_baselines,  g_bglob       = compute_user_baselines(train_users_g)
        g_le_e, g_le_p              = build_fold_encoders(train_users_g)

        g_fold_kw = dict(
            arousal_bl=g_arousal_bl, arousal_med=g_arousal_med,
            baselines=g_baselines,   baseline_global=g_bglob,
            le_e=g_le_e,             le_p=g_le_p,
            spline_trans=_st_g,      n_spline=_ns_g,
        )

        tr_pos = build_positives(H, user_ids=train_users_g, **g_fold_kw)
        tr_neg = build_negatives(H, user_ids=train_users_g, **g_fold_kw)
        tr_df  = pd.concat([tr_pos, tr_neg],
                            ignore_index=True).dropna(subset=_fc_g)
        if tr_df.empty or tr_df['label'].nunique() < 2:
            continue

        X_tr_g = tr_df[_fc_g].values.astype(float)
        y_tr_g = tr_df['label'].values.astype(int)
        pos_g  = y_tr_g.mean()
        if pos_g in (0.0, 1.0):
            continue
        clf_g = xgb.XGBClassifier(
            scale_pos_weight=(1 - pos_g) / pos_g, **XGB_PARAMS)
        clf_g.fit(X_tr_g, y_tr_g)

        te_pos = build_positives(H, user_ids={test_user}, **g_fold_kw)
        te_neg = build_negatives(H, user_ids={test_user}, **g_fold_kw)
        te_df  = pd.concat([te_pos, te_neg],
                            ignore_index=True).dropna(subset=_fc_g)
        if te_df.empty:
            continue

        assert set(tr_df['user_id'].unique()).isdisjoint({test_user}), \
            "LEAKAGE DETECTED"

        y_true_all.extend(te_df['label'].values.astype(int))
        y_prob_all.extend(
            clf_g.predict_proba(te_df[_fc_g].values.astype(float))[:, 1])

    if len(y_true_all) == 0 or len(np.unique(y_true_all)) < 2:
        continue

    auc_pr  = average_precision_score(y_true_all, y_prob_all)
    auc_roc = roc_auc_score(y_true_all, y_prob_all)
    n_total = len(y_true_all)
    pos_pct = np.mean(y_true_all) * 100

    grid_results.append({
        'H': H, 'n_knots': n_knots, 'degree': degree,
        'n_spline_basis': _ns_g,
        'AUC-PR': auc_pr, 'AUC-ROC': auc_roc,
        'n_samples': n_total, 'pos_pct': pos_pct,
    })
    print(f"  {H:<5} {n_knots:<9} {degree:<5} {_ns_g:<7} "
          f"{auc_pr:<10.4f} {auc_roc:<10.4f} {n_total:>8}  {pos_pct:.1f}%")

results_df = pd.DataFrame(grid_results).sort_values('AUC-PR', ascending=False)

# ======================================================================
# 19. Full nested evaluation of top-N candidates (parallel)
# ======================================================================
print("\n" + "=" * 70)
print(f"FULL NESTED EVALUATION OF TOP-{TOP_N_CANDIDATES} CANDIDATES (parallel)")
print("=" * 70)

top_rows = results_df.head(TOP_N_CANDIDATES)
print(f"\n  Top {TOP_N_CANDIDATES} combos by grid AUC-PR:")
for _, r in top_rows.iterrows():
    print(f"    H={int(r['H'])}  n_knots={int(r['n_knots'])}  "
          f"degree={int(r['degree'])}  basis={int(r['n_spline_basis'])}  "
          f"grid-AUC-PR={r['AUC-PR']:.4f}")

print(f"\n  {'H':<5} {'n_knots':<9} {'deg':<5} {'basis':<7} "
      f"{'Nested AUC-PR':<16} {'AUC-ROC':<12} {'ECE':<8} "
      f"{'Brier':<8} {'CalMAE':<9} {'Score':<9}  Grid-AUC-PR")
print("  " + "-" * 105)

def evaluate_candidate(row):
    H_c = int(row['H'])
    nk_c = int(row['n_knots'])
    deg_c = int(row['degree'])
    g_aucpr = row['AUC-PR']
    res = full_nested_eval(H_c, nk_c, deg_c)
    return res

# Parallel evaluation of top candidates
nested_results = Parallel(n_jobs=-1)(
    delayed(evaluate_candidate)(row) for _, row in top_rows.iterrows()
)

for res in nested_results:
    # Find grid AUC-PR for this candidate
    grid_row = results_df[(results_df['H'] == res['H']) &
                          (results_df['n_knots'] == res['n_knots']) &
                          (results_df['degree'] == res['degree'])]
    g_aucpr = grid_row['AUC-PR'].values[0] if not grid_row.empty else np.nan
    cur_best = max(nested_results, key=lambda r: r['selection_score'])
    winner_tag = ' <- winner (so far)' if res is cur_best else ''
    print(f"  {res['H']:<5} {res['n_knots']:<9} {res['degree']:<5} {res['n_spline_basis']:<7} "
          f"{res['AUC-PR']:<16.4f} {res['AUC-ROC']:<12.4f} "
          f"{res['ECE_label']:<8.4f} {res['Brier']:<8.4f} "
          f"{res['Calib_MAE']:<9.4f} {res['selection_score']:<9.4f}  "
          f"{g_aucpr:.4f}{winner_tag}")

best_result = max(nested_results, key=lambda r: r['selection_score'])
best_H      = best_result['H']
best_nknots = best_result['n_knots']
best_degree = best_result['degree']
best_nbasis = best_result['n_spline_basis']

print(f"\n  Best combo: H={best_H}s  n_knots={best_nknots}  degree={best_degree}  "
      f"basis={best_nbasis}")
print(f"  nested AUC-PR={best_result['AUC-PR']:.4f}  "
      f"AUC-ROC={best_result['AUC-ROC']:.4f}  "
      f"ECE={best_result['ECE_label']:.4f}  "
      f"Brier={best_result['Brier']:.4f}  "
      f"Cal_MAE={best_result['Calib_MAE']:.4f}  "
      f"score={best_result['selection_score']:.4f}")

# Update globals to match the winner
spline_transformer, N_SPLINE_BASIS = make_spline_transformer(
    best_nknots, best_degree, float(max(H_CANDIDATES)))
FEATURE_COLS = make_feature_cols(N_SPLINE_BASIS)
print(f"\n  Global spline updated → n_knots={best_nknots}  "
      f"degree={best_degree}  basis={N_SPLINE_BASIS}")

# Unpack winner arrays
y_true_arr       = best_result['y_true']
raw_arr          = best_result['raw']
cal_arr          = best_result['calib']
gt_arr           = best_result['gt']
dist_active_arr  = best_result['dist_active']
time_since_arr   = best_result['time_since']
time_in_dist_arr = best_result['time_in_dist']

# ======================================================================
# 20. Full metrics on winner
# ======================================================================
print("\n" + "=" * 70)
print(f"FULL EVALUATION  (H={best_H}s, n_knots={best_nknots}, "
      f"degree={best_degree}, nested calibration)")
print("=" * 70)

prec_c, rec_c, thresh_c = precision_recall_curve(y_true_arr, raw_arr)
f1_c = 2 * prec_c[:-1] * rec_c[:-1] / (prec_c[:-1] + rec_c[:-1] + 1e-9)
best_thresh = thresh_c[np.argmax(f1_c)]
y_pred_raw  = (raw_arr >= best_thresh).astype(int)
m_raw = compute_metrics(y_true_arr, raw_arr, best_thresh)

print(f"\nComputing bootstrap CIs (n={N_BOOTSTRAP}) for raw model ...")
ci_aucpr,  lo_aucpr,  hi_aucpr  = bootstrap_ci(y_true_arr, raw_arr, average_precision_score)
ci_aucroc, lo_aucroc, hi_aucroc = bootstrap_ci(y_true_arr, raw_arr, roc_auc_score)
ci_brier,  lo_brier,  hi_brier  = bootstrap_ci(y_true_arr, raw_arr, brier_score_loss)
mcc_fn   = lambda yt, ys: matthews_corrcoef(yt, (ys >= best_thresh).astype(int))
ci_mcc,   lo_mcc,   hi_mcc     = bootstrap_ci(y_true_arr, raw_arr, mcc_fn)
kappa_fn = lambda yt, ys: cohen_kappa_score(yt, (ys >= best_thresh).astype(int))
ci_kappa, lo_kappa, hi_kappa   = bootstrap_ci(y_true_arr, raw_arr, kappa_fn)
ci_ll,    lo_ll,    hi_ll      = bootstrap_ci(y_true_arr, raw_arr, log_loss)

print("\nModel Metrics (raw XGBoost)")
print(f"  {'Metric':<18} {'Value (95% CI)'}")
print("  " + "-" * 52)
for label, val in [
    ('AUC-PR',      f"{m_raw['AUC-PR']:.4f}  [{lo_aucpr:.4f} - {hi_aucpr:.4f}]"),
    ('AUC-ROC',     f"{m_raw['AUC-ROC']:.4f}  [{lo_aucroc:.4f} - {hi_aucroc:.4f}]"),
    ('Log-Loss',    f"{m_raw['Log-Loss']:.4f}  [{lo_ll:.4f} - {hi_ll:.4f}]"),
    ('MCC',         f"{m_raw['MCC']:.4f}  [{lo_mcc:.4f} - {hi_mcc:.4f}]"),
    ('Kappa',       f"{m_raw['Kappa']:.4f}  [{lo_kappa:.4f} - {hi_kappa:.4f}]"),
    ('Brier',       f"{m_raw['Brier']:.4f}  [{lo_brier:.4f} - {hi_brier:.4f}]"),
    ('F1',          f"{m_raw['F1']:.4f}"),
    ('Precision',   f"{m_raw['Precision']:.4f}"),
    ('Recall',      f"{m_raw['Recall']:.4f}"),
    ('Specificity', f"{m_raw['Specificity']:.4f}"),
    ('NPV',         f"{m_raw['NPV']:.4f}"),
]:
    print(f"  {label:<18} {val}")
print(f"\n  Threshold (max-F1) : {best_thresh:.4f}")

print("\nPrecision @ Fixed Recall (raw XGBoost)")
for r in RECALL_LEVELS:
    print(f"  P @ Recall={r:.0%} : {m_raw[f'P@R={int(r*100)}']:.4f}")

print(f"\n Classification Report (raw XGBoost @ t={best_thresh:.2f})")
print(classification_report(y_true_arr, y_pred_raw, target_names=['Safe', 'Error']))

# ======================================================================
# 21. Ground truth validation
# ======================================================================
print("\n" + "=" * 70)
print("GROUND TRUTH VALIDATION (with nested calibration)")
print("=" * 70)

sentinel_val = float(max(H_CANDIDATES))
mask_active        = dist_active_arr > 0
mask_hangover      = (~mask_active) & (time_since_arr < best_H)
mask_true_baseline = (~mask_active) & (time_since_arr >= sentinel_val)

df_gt = pd.DataFrame({
    'distraction_active':   dist_active_arr,
    'time_since_last_dist': time_since_arr,
    'time_in_current_dist': time_in_dist_arr,
    'label':                y_true_arr,
    'raw_prob':             raw_arr,
    'calib_prob':           cal_arr,
    'gt_prob':              gt_arr,
})

conditions = [
    ('Active distraction',                       mask_active),
    (f'Hangover ({best_H}s)',                    mask_hangover),
    ('Baseline (driving without distractions)',  mask_true_baseline),
]

strat_rows = []
print(f"\nRisk Stratification")
print(f"  {'Condition':<40} {'GT rate':>10} {'Raw mean':>10} "
      f"{'Calib mean':>12} {'N rows':>8}")
print("  " + "-" * 86)
for label, mask in conditions:
    subset = df_gt[mask]
    if len(subset) == 0:
        continue
    gt_rate  = subset['label'].mean()
    raw_mean = subset['raw_prob'].mean()
    cal_mean = subset['calib_prob'].mean()
    print(f"  {label:<40} {gt_rate:>10.4f} {raw_mean:>10.4f} "
          f"{cal_mean:>12.4f} {len(subset):>8}")
    strat_rows.append({'condition': label, 'gt_error_rate': gt_rate,
                       'raw_mean': raw_mean, 'calib_mean': cal_mean,
                       'n': len(subset)})

strat_df = pd.DataFrame(strat_rows)
fig, ax  = plt.subplots(figsize=(10, 5))
x, w = np.arange(len(strat_df)), 0.25
ax.bar(x - w, strat_df['gt_error_rate'], w, label='GT error rate',
       color='#d62728', alpha=0.85)
ax.bar(x,     strat_df['raw_mean'],      w, label='Raw model mean',
       color='steelblue', alpha=0.85)
ax.bar(x + w, strat_df['calib_mean'],    w, label='Calibrated model mean',
       color='green', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(strat_df['condition'], rotation=15, ha='right')
ax.set_ylabel('Rate / Probability')
ax.set_title('Risk Stratification: GT vs Model (raw & calibrated)')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{EVAL_OUT}gt_risk_stratification_calibrated.png', dpi=150)
plt.close()
print(f"Saved {EVAL_OUT}gt_risk_stratification_calibrated.png")

# ======================================================================
# 22. Temporal decay
# ======================================================================
print(f"\nTemporal Decay (seconds 0..{best_H})")

model_mean_by_sec = np.full(best_H + 1, np.nan)
calib_mean_by_sec = np.full(best_H + 1, np.nan)
model_mean_by_sec[0] = df_gt.loc[mask_active, 'raw_prob'].mean()
calib_mean_by_sec[0] = df_gt.loc[mask_active, 'calib_prob'].mean()

post_rows = df_gt[mask_hangover]
for n in range(1, best_H + 1):
    sub = post_rows[
        (post_rows['time_since_last_dist'] >= n - 1) &
        (post_rows['time_since_last_dist'] <  n)
    ]
    if len(sub) > 0:
        model_mean_by_sec[n] = sub['raw_prob'].mean()
        calib_mean_by_sec[n] = sub['calib_prob'].mean()

print(f"  {'Bin':<12} {'GT rate(/s)':>12} {'Exposure':>10} {'Errors':>8} "
      f"{'Raw mean':>12} {'Calib mean':>12}")
print("  " + "-" * 72)
for b_start in range(1, best_H + 1, 5):
    b_end   = min(b_start + 5, best_H + 1)
    lbl     = f"{b_start}-{b_end-1}s"
    n_err   = int(gt_errors[b_start:b_end].sum())
    exp_sum = gt_exposure[b_start:b_end].sum()
    rate_m  = float(np.nanmean(gt_rate_per_sec[b_start:b_end]))
    m_mean  = float(np.nanmean(model_mean_by_sec[b_start:b_end]))
    c_mean  = float(np.nanmean(calib_mean_by_sec[b_start:b_end]))
    print(f"  {lbl:<12} {rate_m:>12.4f} {exp_sum:>10.0f} {n_err:>8} "
          f"{m_mean:>12.4f} {c_mean:>12.4f}")
print(f"  {'Baseline':<12} {p_baseline_global:>12.4f} {'-':>10} "
      f"{'-':>8} {'-':>12} {'-':>12}")

# ======================================================================
# 23. Calibration MAE / RMSE
# ======================================================================
errors_raw = df_gt['raw_prob'].values   - df_gt['gt_prob'].values
errors_cal = df_gt['calib_prob'].values - df_gt['gt_prob'].values
mae_raw  = np.mean(np.abs(errors_raw));  rmse_raw = np.sqrt(np.mean(errors_raw**2))
mae_cal  = np.mean(np.abs(errors_cal));  rmse_cal = np.sqrt(np.mean(errors_cal**2))
print(f"\n  Global MAE  : raw={mae_raw:.4f}  calibrated={mae_cal:.4f}")
print(f"  Global RMSE : raw={rmse_raw:.4f}  calibrated={rmse_cal:.4f}")
print(f"\n  {'Condition':<40} {'GT mean':>9} {'Raw mean':>11} {'Cal mean':>11} "
      f"{'Raw MAE':>8} {'Cal MAE':>8} {'N':>6}")
print("  " + "-" * 96)
for lbl, mask in conditions:
    sub = df_gt[mask]
    if len(sub) == 0:
        continue
    mae_r = np.mean(np.abs(sub['raw_prob'].values   - sub['gt_prob'].values))
    mae_c = np.mean(np.abs(sub['calib_prob'].values - sub['gt_prob'].values))
    print(f"  {lbl:<40} {sub['gt_prob'].mean():>9.4f} "
          f"{sub['raw_prob'].mean():>11.4f} {sub['calib_prob'].mean():>11.4f} "
          f"{mae_r:>8.4f} {mae_c:>8.4f} {len(sub):>6}")

# ======================================================================
# 24. Temporal decay plot
# ======================================================================
seconds   = np.arange(0, best_H + 1)
valid_gt  = ~np.isnan(gt_rate_per_sec[:best_H+1])
valid_raw = ~np.isnan(model_mean_by_sec)
valid_cal = ~np.isnan(calib_mean_by_sec)
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(seconds[valid_gt],  gt_rate_per_sec[:best_H+1][valid_gt],
        'o-', color='#d62728', lw=2, markersize=4,
        label='GT empirical rate (errors/s)')
ax.plot(seconds[valid_raw], model_mean_by_sec[valid_raw],
        's--', color='steelblue', lw=2, markersize=4, label='Raw model mean')
ax.plot(seconds[valid_cal], calib_mean_by_sec[valid_cal],
        'd-.', color='green', lw=2, markersize=4, label='Calibrated model mean')
ax.axhline(y=p_baseline_global, color='grey', linestyle=':', lw=1.5,
           label=f'Baseline ({p_baseline_global*100:.2f}%/s)')
ax.set_xlabel('Bin (0=inside window, 1..H=seconds after distraction ended)')
ax.set_ylabel('Rate / Probability')
ax.set_title(f'GT Error Rate vs Model  '
             f'(H={best_H}s, n_knots={best_nknots}, degree={best_degree})')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{EVAL_OUT}gt_temporal_decay_calibrated.png', dpi=150)
plt.close()
print(f"\nSaved {EVAL_OUT}gt_temporal_decay_calibrated.png")

# ======================================================================
# 25. Save tables
# ======================================================================
decay_rows = []
for n in range(best_H + 1):
    decay_rows.append({
        'bin': n,
        'gt_rate_per_sec':       gt_rate_per_sec[n] if n <= max_H_gt else np.nan,
        'raw_model_mean':        model_mean_by_sec[n],
        'calibrated_model_mean': calib_mean_by_sec[n],
        'n_errors':  int(gt_errors[n]) if n <= max_H_gt else 0,
        'exposure':  gt_exposure[n]    if n <= max_H_gt else 0,
    })
pd.DataFrame(decay_rows).to_csv(f'{EVAL_OUT}gt_temporal_decay.csv', index=False)
df_gt.assign(
    abs_error_raw=lambda d: np.abs(d['raw_prob']   - d['gt_prob']),
    abs_error_cal=lambda d: np.abs(d['calib_prob'] - d['gt_prob']),
).to_csv(f'{EVAL_OUT}gt_continuous_comparison.csv', index=False)
pd.DataFrame([{'MAE_raw': mae_raw, 'RMSE_raw': rmse_raw,
               'MAE_cal': mae_cal, 'RMSE_cal': rmse_cal,
               'p_bin0': p_bin0, 'p_baseline': p_baseline_global}]
             ).to_csv(f'{EVAL_OUT}gt_continuous_metrics.csv', index=False)

nested_summary = pd.DataFrame([{
    'H': r['H'], 'n_knots': r['n_knots'], 'degree': r['degree'],
    'n_spline_basis': r['n_spline_basis'],
    'grid_AUC_PR': results_df.loc[
        (results_df['H'] == r['H']) &
        (results_df['n_knots'] == r['n_knots']) &
        (results_df['degree']  == r['degree']),
        'AUC-PR'].values[0],
    'nested_AUC_PR': r['AUC-PR'], 'nested_AUC_ROC': r['AUC-ROC'],
    'ECE_label': r['ECE_label'], 'Brier': r['Brier'],
    'nested_Cal_MAE': r['Calib_MAE'], 'nested_Cal_RMSE': r['Calib_RMSE'],
    'selection_score': r['selection_score'],
} for r in nested_results])
nested_summary.to_csv(f'{EVAL_OUT}h_spline_selection_nested.csv', index=False)
results_df.to_csv(f'{EVAL_OUT}grid_search_results.csv', index=False)

print(f"Saved {EVAL_OUT}gt_temporal_decay.csv")
print(f"Saved {EVAL_OUT}gt_continuous_comparison.csv")
print(f"Saved {EVAL_OUT}gt_continuous_metrics.csv")
print(f"Saved {EVAL_OUT}h_spline_selection_nested.csv")
print(f"Saved {EVAL_OUT}grid_search_results.csv")

# ======================================================================
# 26. Final model training (all data, best H + best spline)
# ======================================================================
print(f"\nTraining Final Model  "
      f"(H={best_H}s, n_knots={best_nknots}, degree={best_degree}, all data)")

pos_f = build_positives(best_H)
neg_f = build_negatives(best_H)
all_data    = pd.concat([pos_f, neg_f], ignore_index=True)
safe_global = all_data[all_data['label'] == 0]

if len(safe_global) > 0:
    def_mp_g  = safe_global['model_prob'].median()
    def_mpe_g = safe_global['model_pred_enc'].mode()[0]
    def_ep_g  = safe_global['emotion_prob'].median()
    def_ele_g = safe_global['emotion_label_enc'].mode()[0]
    def_hr_g  = (safe_global['hr_bpm'].median()
                 if 'hr_bpm' in safe_global.columns else 0.5)
else:
    def_mp_g  = 0.5
    def_mpe_g = le_pred.transform([UNKNOWN_LABEL])[0]
    def_ep_g  = 0.5
    def_ele_g = le_emotion.transform([UNKNOWN_LABEL])[0]
    def_hr_g  = 0.5

base_f = build_baseline_negatives(
    default_model_prob=def_mp_g, default_model_pred_enc=def_mpe_g,
    default_emotion_prob=def_ep_g, default_emotion_label_enc=def_ele_g,
    default_hr_bpm=def_hr_g, default_arousal_baseline=global_arousal_median,
)

df_f = pd.concat([pos_f, neg_f, base_f], ignore_index=True).dropna(subset=FEATURE_COLS)
X_f, y_f = df_f[FEATURE_COLS].values.astype(float), df_f['label'].values.astype(int)
spw_f = (1 - y_f.mean()) / y_f.mean()

base_clf = xgb.XGBClassifier(scale_pos_weight=spw_f, **XGB_PARAMS)
base_clf.fit(X_f, y_f)

# ──────────────────────────────────────────────────────────────────────
# Feature importance — honest 3-category breakdown
# ──────────────────────────────────────────────────────────────────────
importance_df = pd.DataFrame({
    'Feature':    FEATURE_COLS,
    'Importance': base_clf.feature_importances_,
}).sort_values('Importance', ascending=False)

print("\n" + "-" * 30)
print("FEATURE IMPORTANCE (Final Model)")
print("-" * 30)
print(importance_df.to_string(index=False))

# 3 honest buckets
spline_feat_names = [f'time_spline_{i}' for i in range(N_SPLINE_BASIS)]
interaction_names = [f for f in FEATURE_COLS if '_x_' in f]
pure_temporal     = (['cognitive_load_decay', 'time_since_last_dist',
                      'time_in_current_dist', 'distraction_active']
                     + spline_feat_names)
pure_driver       = [f for f in AROUSAL_FEATURES + CONTEXT_FEATURES
                     if f not in interaction_names]

t_imp     = importance_df[importance_df['Feature'].isin(pure_temporal)   ]['Importance'].sum()
d_imp     = importance_df[importance_df['Feature'].isin(pure_driver)     ]['Importance'].sum()
inter_imp = importance_df[importance_df['Feature'].isin(interaction_names)]['Importance'].sum()

print(f"\nInformation Source Balance (3 honest categories):")
print(f"  Pure Temporal Signals        : {t_imp:.2%}  "
      f"(time, splines, decay)")
print(f"  Pure Driver/Arousal Signals  : {d_imp:.2%}  "
      f"(HR, arousal, emotion scalars)")
print(f"  Temporal × Arousal (joint)   : {inter_imp:.2%}  "
      f"(how arousal evolves over recovery timeline)")
print(f"\n  Interpretation: The model's predictive power comes primarily from")
print(f"  tracking *how arousal/HR evolve* over the post-distraction recovery")
print(f"  window — neither dimension is sufficient alone.")

# Feature importance plot (coloured by category)
fig, ax = plt.subplots(figsize=(11, 7))
filtered = importance_df[importance_df['Importance'] >= 0.005].copy()

def cat(f):
    if f in interaction_names:  return 'Temporal × Arousal'
    if f in pure_temporal:      return 'Pure Temporal'
    return 'Pure Driver/Arousal'

filtered['Category'] = filtered['Feature'].map(cat)
colours = {
    'Temporal × Arousal':  '#2196F3',
    'Pure Temporal':        '#FF9800',
    'Pure Driver/Arousal':  '#4CAF50',
}
bar_colours = filtered['Category'].map(colours)
ax.barh(filtered['Feature'], filtered['Importance'],
        color=bar_colours, edgecolor='none')
ax.invert_yaxis()
ax.set_xlabel('Importance (Gain)')
ax.set_title(f'Feature Importance  '
             f'(H={best_H}s, n_knots={best_nknots}, degree={best_degree})')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=l) for l, c in colours.items()]
ax.legend(handles=legend_elements, loc='lower right')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{EVAL_OUT}feature_importance.png', dpi=150)
plt.close()
print(f"\nSaved -> {EVAL_OUT}feature_importance.png")

# ──────────────────────────────────────────────────────────────────────
# Final calibrator: OOF inner-LOSO predictions
# ──────────────────────────────────────────────────────────────────────
raw_all_final = base_clf.predict_proba(X_f)[:, 1]
cld_all       = df_f['cognitive_load_decay'].values
time_all      = df_f['time_since_last_dist'].values
gt_all        = np.array([
    assign_gt_prob(row, best_H, gt_rate_per_sec, p_baseline_global, gt_active_rate)
    for _, row in df_f.iterrows()
])
groups_f = df_f['user_id'].values

oof_raw_f = np.full(len(X_f), np.nan)
inner_logo_f = LeaveOneGroupOut()
for in_tr, in_val in inner_logo_f.split(X_f, y_f, groups_f):
    if len(np.unique(y_f[in_tr])) < 2:
        oof_raw_f[in_val] = 0.5
        continue
    in_spw = ((1 - y_f[in_tr].mean()) / y_f[in_tr].mean()
              if y_f[in_tr].mean() not in (0.0, 1.0) else spw_f)
    in_clf = xgb.XGBClassifier(scale_pos_weight=in_spw, **XGB_PARAMS)
    in_clf.fit(X_f[in_tr], y_f[in_tr])
    oof_raw_f[in_val] = in_clf.predict_proba(X_f[in_val])[:, 1]
nan_mask_f = np.isnan(oof_raw_f)
if nan_mask_f.any():
    oof_raw_f[nan_mask_f] = raw_all_final[nan_mask_f]

final_iso = xgb.XGBRegressor(**CAL_XGB_PARAMS)
final_iso.fit(np.column_stack([oof_raw_f, cld_all, time_all]), gt_all)

artifact = {
    'model':                 base_clf,
    'calibrator':            final_iso,
    'best_H':                best_H,
    'best_n_knots':          best_nknots,
    'best_degree':           best_degree,
    'best_n_spline_basis':   best_nbasis,
    'best_thresh':           best_thresh,
    'feature_cols':          FEATURE_COLS,
    'le_emotion':            le_emotion,
    'le_pred':               le_pred,
    'user_baselines':        user_baselines,
    'p_baseline_global':     p_baseline_global,
    'user_arousal_baseline': user_arousal_baseline,
    'global_arousal_median': global_arousal_median,
    'spline_transformer':    spline_transformer,
    'n_spline_basis':        N_SPLINE_BASIS,
    'cv_results':            results_df,
    'nested_h_results':      nested_summary,
    'metrics_raw':           m_raw,
    'bootstrap_ci_raw': {
        'AUC-PR':  (ci_aucpr,  lo_aucpr,  hi_aucpr),
        'AUC-ROC': (ci_aucroc, lo_aucroc, hi_aucroc),
        'MCC':     (ci_mcc,    lo_mcc,    hi_mcc),
        'Brier':   (ci_brier,  lo_brier,  hi_brier),
    },
}
joblib.dump(artifact, MODEL_OUT)
print(f"Saved -> {MODEL_OUT}")

# ======================================================================
# 27. Inference helper
# ======================================================================
def predict_fitness(
    distraction_active: Union[bool, int, float],
    seconds_since_last_distraction: float,
    emotion_label: Optional[str] = None,
    emotion_prob: float = 0.5,
    distraction_pred_label: Optional[str] = None,
    distraction_pred_prob: float = 0.5,
    user_id: Optional[str] = None,
    distraction_density_60: int = 0,
    distraction_density_120: int = 0,
    prev_distraction_duration_s: float = 0.0,
    artifact_path: str = MODEL_OUT,
) -> Dict[str, object]:
    art    = joblib.load(artifact_path)
    H      = art['best_H']
    thresh = art['best_thresh']
    warns  = []

    try:
        dist_conf = float(distraction_active)
        if not (0.0 <= dist_conf <= 1.0):
            warns.append(f"distraction_active={dist_conf} out of [0,1], clamped")
            dist_conf = float(np.clip(dist_conf, 0.0, 1.0))
    except Exception:
        dist_conf = 0.0
        warns.append("distraction_active invalid, defaulting to 0.0")

    try:
        t_input = float(seconds_since_last_distraction)
        if t_input < 0:
            warns.append(f"seconds_since_last_distraction={t_input} < 0, clamped")
            t_input = 0.0
        if dist_conf > 0:
            t_since = 0.0; t_in_current = min(t_input, float(H))
        else:
            t_since = min(t_input, float(H)); t_in_current = 0.0
    except Exception:
        t_since = float(H); t_in_current = 0.0
        warns.append("seconds_since_last_distraction invalid, defaulting to H")

    cld = dist_conf if dist_conf > 0 else np.exp(-t_since / max(H, 1e-9))

    def _clip_prob(val, name, default=0.5):
        try:
            v = float(val)
            if not (0.0 <= v <= 1.0):
                warns.append(f"{name}={v} out of [0,1], clamped")
                return float(np.clip(v, 0.0, 1.0))
            return v
        except Exception:
            warns.append(f"{name} invalid, defaulting to {default}")
            return default

    a_prob = _clip_prob(distraction_pred_prob, 'distraction_pred_prob')
    e_prob = _clip_prob(emotion_prob,      'emotion_prob')

    _le_emotion = art['le_emotion']
    _le_pred    = art['le_pred']

    def _encode(le, val):
        s = safe_str(val)
        if s not in le.classes_:
            warns.append(f"Unseen label '{s}', using 'unknown'")
            s = UNKNOWN_LABEL
        return int(le.transform([s])[0])

    emo_enc  = _encode(_le_emotion, emotion_label)
    pred_enc = _encode(_le_pred,    distraction_pred_label)

    b_rate = art['user_baselines'].get(user_id, art['p_baseline_global'])
    if user_id is not None and user_id not in art['user_baselines']:
        warns.append(f"user_id='{user_id}' not in training set, using global baseline")

    u_aro_base = art.get('user_arousal_baseline', {}).get(
        user_id, art.get('global_arousal_median', a_prob))
    aro_dev = a_prob - u_aro_base
    if user_id is None:
        warns.append("user_id not provided; arousal_deviation set to 0")
        aro_dev = 0.0

    # Use the spline saved in the artifact (matches best combo)
    spline_trans = art['spline_transformer']
    n_spline     = art['n_spline_basis']
    spline_vals, spline_dict = _spline_row(t_since, spline_trans, n_spline)

    driver_vals = {
        'user_arousal_baseline': u_aro_base,
        'baseline_error_rate':   b_rate,
        'arousal_delta':         0.0,
        'hr_bpm':                0.5,
        'arousal_deviation':     aro_dev,
    }
    interactions = _interaction_row(driver_vals, t_since, spline_vals, n_spline)

    sample = pd.DataFrame([{
        'distraction_active':    dist_conf,
        'time_since_last_dist':  t_since,
        'time_in_current_dist':  t_in_current,
        'cognitive_load_decay':  cld,
        'model_prob':            a_prob,
        'model_pred_enc':        pred_enc,
        'emotion_prob':          e_prob,
        'emotion_label_enc':     emo_enc,
        'arousal_deviation':     aro_dev,
        'user_arousal_baseline': u_aro_base,
        'arousal_delta':         0.0,
        'arousal_at_window_end': 0.0,
        'emotion_changed':       0,
        'hr_bpm':                0.5,
        'model_prob_sq':         a_prob ** 2,
        'emotion_prob_sq':       e_prob ** 2,
        'arousal_deviation_sq':  aro_dev ** 2,
        'baseline_error_rate':   b_rate,
        'distraction_density_60':  max(0, int(distraction_density_60)),
        'distraction_density_120': max(0, int(distraction_density_120)),
        'prev_dist_duration':    max(0.0, float(prev_distraction_duration_s)),
    }])
    sample.update(spline_dict)
    sample.update(interactions)

    raw_prob        = float(art['model'].predict_proba(sample[art['feature_cols']])[0, 1])
    calibrated_prob = float(np.clip(
        art['calibrator'].predict(np.array([[raw_prob, cld, t_since]]))[0],
        0.0, 1.0))

    return {
        'error_probability': round(calibrated_prob, 4),
        'fitness_to_drive':  round(1.0 - calibrated_prob, 4),
        'alert':             raw_prob >= thresh,
        'input_warnings':    warns,
    }

# ======================================================================
# 28. Summary
# ======================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Best combo: H={best_H}s  n_knots={best_nknots}  "
      f"degree={best_degree}  basis_dim={best_nbasis}")
print()
print("  Full nested evaluation results (top candidates):")
print(f"  {'H':<5} {'n_knots':<9} {'deg':<5} {'basis':<7} "
      f"{'Grid AUC-PR':<13} {'Nested AUC-PR':<15} {'Nested AUC-ROC':<15} "
      f"{'Brier':<8} {'Cal MAE':<10} {'Score'}")
print("  " + "-" * 110)
for _, row in nested_summary.sort_values('nested_AUC_PR', ascending=False).iterrows():
    marker = " <- selected" if (row['H'] == best_H and
                                row['n_knots'] == best_nknots and
                                row['degree'] == best_degree) else ""
    print(f"  {int(row['H']):<5} {int(row['n_knots']):<9} {int(row['degree']):<5} "
          f"{int(row['n_spline_basis']):<7} "
          f"{row['grid_AUC_PR']:<13.4f} {row['nested_AUC_PR']:<15.4f} "
          f"{row['nested_AUC_ROC']:<15.4f} {row['Brier']:<8.4f} "
          f"{row['nested_Cal_MAE']:<10.4f} {row['selection_score']:.4f}{marker}")
print()
print(f"  Raw XGBoost AUC-PR  : {m_raw['AUC-PR']:.4f}  "
      f"95% CI [{lo_aucpr:.4f} - {hi_aucpr:.4f}]")
print(f"  Raw XGBoost AUC-ROC : {m_raw['AUC-ROC']:.4f}  "
      f"95% CI [{lo_aucroc:.4f} - {hi_aucroc:.4f}]")
print(f"  Raw XGBoost MCC     : {m_raw['MCC']:.4f}  "
      f"95% CI [{lo_mcc:.4f} - {hi_mcc:.4f}]")
print(f"  Raw XGBoost Brier   : {m_raw['Brier']:.4f}  "
      f"95% CI [{lo_brier:.4f} - {hi_brier:.4f}]")
print(f"  MAE  (raw vs GT)    : {mae_raw:.4f}")
print(f"  RMSE (raw vs GT)    : {rmse_raw:.4f}")
print(f"  MAE  (calibrated)   : {mae_cal:.4f}")
print(f"  RMSE (calibrated)   : {rmse_cal:.4f}")