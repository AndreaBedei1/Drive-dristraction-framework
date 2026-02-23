"""
Driver Digital Twin â€“ Fitness-to-Drive XGBoost Pipeline  (v4)
==============================================================
Rigorous, publication-ready pipeline with four new cognitive features:

  1. cognitive_load_decay     : exp(-t/H) â€” continuous recovery curve, not a cliff
  2. arousal_deviation        : model_prob minus per-user personal arousal baseline;
                                uses relative stress rather than raw absolute values
  3. distraction_density_30/60/120 : count of windows ending in last 30/60/120 s;
                                     captures "cognitive saturation" from stacking
  4. prev_dist_duration       : duration (s) of the most recent completed window;
                                a 10-s distraction leaves a longer hangover than 2 s

Also:
  - Sampling every second (NEG_SAMPLE_EVERY = 1)
  - Baseline negatives with realistic feature defaults (fold-derived medians)
  - Nested LOSO-CV with isotonic calibration mapped to GT error rates
  - Option-C H selection: fast grid â†’ top-N full nested eval
  - Full bootstrap 95% CIs for all metrics

Input datasets (place in DATA_PATH):
  - Dataset Distractions_distraction.csv
  - Dataset Errors_distraction.csv
  - Dataset Errors_baseline.csv
  - Dataset Driving Time_baseline.csv
"""

import warnings
warnings.filterwarnings('ignore')

import bisect
import os
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
from sklearn.preprocessing import LabelEncoder
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

# â”€â”€ Reproducibility â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# â”€â”€ Configuration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DATA_PATH        = 'data/'
MODEL_OUT        = 'fitness_model_calibrated_union.pkl'
EVAL_OUT         = 'evaluation/union/'
BIN_LIMIT        = 14
H_CANDIDATES     = list(range(1, BIN_LIMIT))
TOP_N_CANDIDATES = 10          # how many top grid-search H values to fully evaluate
NEG_SAMPLE_EVERY = 1
N_BOOTSTRAP      = 1000
RECALL_LEVELS    = [0.80, 0.85, 0.90, 0.95]
GT_ACTIVE_SMOOTH_ALPHA = 20.0
GT_POST_SMOOTH_ALPHA = 40.0
H_SELECTION_CAL_MAE_WEIGHT = 0.15

os.makedirs(EVAL_OUT, exist_ok=True)

XGB_PARAMS = dict(
    n_estimators     = 300,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    n_jobs           = -1,
    eval_metric      = 'aucpr',
    random_state     = RANDOM_SEED,
    verbosity        = 0,
)

FEATURE_COLS = [
    # Core temporal state
    'distraction_active',
    'time_since_last_dist',
    'time_in_current_dist',
    'cognitive_load_decay',       # exp(âˆ’time_since / H); 1.0 if active, ~0 if baseline
    # Arousal signals
    'model_prob',
    'model_pred_enc',
    'emotion_prob',
    'emotion_label_enc',
    'arousal_deviation',          # model_prob minus user's personal arousal median
    # Long-run risk context
    'baseline_error_rate',
    'distraction_density_30',     # # windows ending in last  30 s (cognitive saturation)
    'distraction_density_60',     # # windows ending in last  60 s
    'distraction_density_120',    # # windows ending in last 120 s
    'prev_dist_duration',         # duration (s) of most recent completed distraction
]

# â”€â”€ 1. Load â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("=" * 70)
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

# Build fast lookup for distraction windows
windows_by_session = {}
for (uid, rid), grp in distractions.groupby(['user_id', 'run_id']):
    windows_by_session[(uid, rid)] = grp.reset_index(drop=True)

# â”€â”€ 2. Data integrity checks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n" + "=" * 70)
print("DATA INTEGRITY CHECKS")
print("=" * 70)
issues = []

dist_users  = set(distractions['user_id'].unique())
err_users   = set(errors_dist['user_id'].unique())
only_in_err = err_users - dist_users
if only_in_err:
    issues.append(f"Users in errors but not in distractions: {only_in_err}")

bad_windows = distractions[distractions['timestamp_end'] < distractions['timestamp_start']]
if len(bad_windows):
    issues.append(f"{len(bad_windows)} distraction windows have end < start")

dup_errors = errors_dist.duplicated(subset=['user_id', 'run_id', 'timestamp'])
if dup_errors.sum():
    issues.append(f"{dup_errors.sum()} duplicate error timestamps in errors_dist")

for col in ['user_id', 'run_id', 'timestamp', 'model_pred', 'model_prob',
            'emotion_label', 'emotion_prob']:
    if col not in errors_dist.columns:
        issues.append(f"Missing column in errors_dist: {col}")

for col in ['model_prob', 'emotion_prob']:
    n_nan = errors_dist[col].isna().sum()
    if n_nan:
        print(f"  [WARN] {n_nan} NaNs in errors_dist['{col}'] -> filled with median")
        errors_dist[col] = errors_dist[col].fillna(errors_dist[col].median())

if issues:
    print("  [FAIL] Integrity issues found:")
    for iss in issues:
        print(f"    x {iss}")
    raise RuntimeError("Fix data issues before training.")
else:
    print("  [PASS] All checks passed.")

# â”€â”€ 3. Per-user baseline error rate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
total_base_s      = driving_base['run_duration_seconds'].sum()
p_baseline_global = len(errors_base) / total_base_s
user_base_errs    = errors_base.groupby('user_id').size()
user_base_secs    = driving_base.groupby('user_id')['run_duration_seconds'].sum()
user_baselines    = (user_base_errs / user_base_secs).fillna(p_baseline_global).to_dict()

print(f"\nGlobal baseline error rate : {p_baseline_global*100:.4f}% / second")

# â”€â”€ 3b. Per-user arousal baseline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Extracted into a reusable function so full_nested_eval can recompute it on
# training users only, avoiding test-user leakage into arousal_deviation.
def compute_arousal_baseline(user_ids_set=None):
    # Collect model_prob_start / model_prob_end from distraction window snapshots.
    # If user_ids_set is given, only those users are included.
    # Returns (per_user_dict, global_median).
    acc = {}
    for (uid, rid), wins in windows_by_session.items():
        if user_ids_set is not None and uid not in user_ids_set:
            continue
        vals = list(wins['model_prob_start'].dropna()) + list(wins['model_prob_end'].dropna())
        acc.setdefault(uid, []).extend(vals)
    per_user = {uid: float(np.median(v)) for uid, v in acc.items() if v}
    all_vals  = [v for vals in acc.values() for v in vals]
    glob_med  = float(np.median(all_vals)) if all_vals else 0.5
    return per_user, glob_med

# Global baselines â€“ used by the final model and grid search (all data available).
# full_nested_eval overrides these per fold with training users only.
user_arousal_baseline, global_arousal_median = compute_arousal_baseline()
print(f"Per-user arousal baselines computed for {len(user_arousal_baseline)} users  "
      f"(global median={global_arousal_median:.4f})")

# â”€â”€ 3c. Precompute sorted window-end timestamps for O(log n) density lookups â”€â”€
window_ends_by_session: dict = {}
for (uid, rid), wins in windows_by_session.items():
    window_ends_by_session[(uid, rid)] = sorted(wins['timestamp_end'].tolist())

# â”€â”€ 4. Helper functions for distraction state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def get_distraction_state(user_id, run_id, ts, H):
    """
    Returns:
      - distraction_active (0/1)
      - time_since_last_dist (capped at H for model features)
      - time_in_current_dist (capped at H for model features)
      - true_time_since_last_dist (uncapped, sentinel for baseline)
      - true_time_in_current_dist (uncapped)
    """
    sentinel = float(H_CANDIDATES[-1])
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
        row['user_id'], row['run_id'], row['timestamp'], H=0
    )
    return active == 1

def get_time_since_last(row):
    key  = (row['user_id'], row['run_id'])
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

# â”€â”€ 5. Ground truth perâ€‘second error rates â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
max_H_gt = BIN_LIMIT - 1

def compute_gt_profiles(user_ids_set=None):
    """
    Computes two GT profiles:
      1) active_rates[s] : risk at second s from distraction start (inside window)
      2) post_rates[s]   : risk at second s after distraction end (s=1..H)
    """
    dist_subset = (distractions if user_ids_set is None
                   else distractions[distractions['user_id'].isin(user_ids_set)])
    err_subset = (errors_dist if user_ids_set is None
                  else errors_dist[errors_dist['user_id'].isin(user_ids_set)])

    total_dist_s = (dist_subset['timestamp_end'] - dist_subset['timestamp_start']).dt.total_seconds().sum()
    errs_inside = sum(1 for _, e in err_subset.iterrows() if is_error_inside_distraction(e))
    p_b0 = errs_inside / total_dist_s if total_dist_s > 0 else 0.0

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
            err['user_id'], err['run_id'], err['timestamp'], H=max_H_gt
        )
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
            win_end = wins_s['timestamp_end'].iloc[i]
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
                post_errs[sec] + GT_POST_SMOOTH_ALPHA * p_baseline_global
            ) / (post_expo[sec] + GT_POST_SMOOTH_ALPHA)

    post_valid = np.where(~np.isnan(post_rates[1:]))[0] + 1
    if len(post_valid) >= 2:
        iso_post = IsotonicRegression(increasing=False, out_of_bounds='clip')
        post_rates[post_valid] = iso_post.fit_transform(
            post_valid.astype(float),
            post_rates[post_valid],
            sample_weight=post_expo[post_valid],
        )
    post_rates[post_valid] = np.maximum(post_rates[post_valid], p_baseline_global)

    return {
        'active_rates': active_rates,
        'post_rates': post_rates,
        'p_bin0': p_b0,
        'active_errors': active_errs,
        'active_exposure': active_expo,
        'post_errors': post_errs,
        'post_exposure': post_expo,
    }


def compute_gt_rates(user_ids_set=None):
    profile = compute_gt_profiles(user_ids_set)
    return profile['post_rates'], profile['p_bin0']


# Global rates for reporting/grid/final model.
gt_profile_global = compute_gt_profiles()
gt_rate_per_sec = gt_profile_global['post_rates']
gt_active_rate = gt_profile_global['active_rates']
p_bin0 = gt_profile_global['p_bin0']
gt_errors = gt_profile_global['post_errors'].copy()
gt_exposure = gt_profile_global['post_exposure'].copy()
gt_active_errors = gt_profile_global['active_errors'].copy()
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
for n in range(1, min(11, max_H_gt + 1)):
    print(f"    {n:2d}s : {gt_rate_per_sec[n]:.6f}  "
          f"(exposure={gt_exposure[n]:6.0f}, errors={int(gt_errors[n]):3d})")
if max_H_gt > 10:
    print("    ...")

# -- 6. Label encoding ---------------------------------------------------------
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

def encode_row(emotion_label, model_pred):
    return (le_emotion.transform([safe_str(emotion_label)])[0],
            le_pred.transform([safe_str(model_pred)])[0])

# â”€â”€ 4c. New feature helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def get_distraction_density(uid, rid, ts, lookback_seconds: float) -> int:
    # Count distraction windows whose end falls in [ts-lookback, ts).
    # Only completed windows counted; ongoing window never included.
    # Uses bisect for O(log n) speed on sorted end-time list.
    ends = window_ends_by_session.get((uid, rid), [])
    if not ends:
        return 0
    t_lo = ts - pd.Timedelta(seconds=lookback_seconds)
    lo   = bisect.bisect_left(ends, t_lo)
    hi   = bisect.bisect_left(ends, ts)   # excludes ts itself
    return hi - lo


def get_prev_dist_duration(uid, rid, ts) -> float:
    # Duration (s) of the most recently completed distraction window before ts.
    # Returns 0 if no prior window exists.
    key  = (uid, rid)
    if key not in windows_by_session:
        return 0.0
    wins = windows_by_session[key]
    prev = wins[wins['timestamp_end'] < ts]
    if prev.empty:
        return 0.0
    row  = prev.loc[prev['timestamp_end'].idxmax()]
    return max(0.0, (row['timestamp_end'] - row['timestamp_start']).total_seconds())


# â”€â”€ 7. Ground truth probability assignment â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def assign_gt_prob(row, H, gt_rates=None, p_base=None, gt_active_rates=None):
    # Assign empirical GT error rate to a sample row.
    # gt_rates: array returned by compute_gt_rates() â€” pass fold-specific array
    #           inside nested eval to avoid Leakage-2; defaults to global.
    # p_base:   baseline error rate for the relevant user set; defaults to global.
    _rates   = gt_rates if gt_rates is not None else gt_rate_per_sec
    _active  = gt_active_rates if gt_active_rates is not None else gt_active_rate
    _p_base  = p_base   if p_base   is not None else p_baseline_global
    if row['distraction_active'] == 1:
        true_t_in = row.get('true_time_in_dist', row.get('time_in_current_dist', 0.0))
        sec = int(min(np.floor(max(true_t_in, 0.0)), max_H_gt))
        if sec <= max_H_gt and not np.isnan(_active[sec]):
            return _active[sec]
        return _rates[0]
    true_t = row.get('true_time_since', row['time_since_last_dist'])
    if true_t >= float(H_CANDIDATES[-1]):
        return _p_base
    if true_t > max_H_gt:
        return _p_base
    n = int(np.floor(true_t)) + 1
    if n <= max_H_gt and not np.isnan(_rates[n]):
        return _rates[n]
    return _p_base

# â”€â”€ 8. Sample builders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def build_positives(H, user_ids=None, arousal_bl=None, arousal_med=None):
    # user_ids: if given, only build rows for those users (fold filtering).
    # arousal_bl / arousal_med: fold-specific baseline; fall back to globals.
    _abl = arousal_bl  if arousal_bl  is not None else user_arousal_baseline
    _amed= arousal_med if arousal_med is not None else global_arousal_median
    rows = []
    for _, err in errors_dist.iterrows():
        uid, rid, ts           = err['user_id'], err['run_id'], err['timestamp']
        if user_ids is not None and uid not in user_ids:
            continue
        dist_active, time_since, time_in_dist, true_time_since, true_time_in_dist = \
            get_distraction_state(uid, rid, ts, H)
        emo_enc, pred_enc      = encode_row(err['emotion_label'], err['model_pred'])
        m_prob                 = err['model_prob']
        u_aro                  = _abl.get(uid, _amed)
        # cognitive_load_decay: 1.0 while active; exp(-t/H) in recovery
        cld = 1.0 if dist_active == 1 else np.exp(-time_since / max(H, 1e-9))
        rows.append({
            'user_id':                uid,
            'distraction_active':     dist_active,
            'time_since_last_dist':   time_since,
            'time_in_current_dist':   time_in_dist,
            'true_time_since':        true_time_since,
            'true_time_in_dist':      true_time_in_dist,
            'cognitive_load_decay':   cld,
            'model_prob':             m_prob,
            'model_pred_enc':         pred_enc,
            'emotion_prob':           err['emotion_prob'],
            'emotion_label_enc':      emo_enc,
            'arousal_deviation':      m_prob - u_aro,
            'baseline_error_rate':    user_baselines.get(uid, p_baseline_global),
            'distraction_density_30':  get_distraction_density(uid, rid, ts, 30),
            'distraction_density_60':  get_distraction_density(uid, rid, ts, 60),
            'distraction_density_120': get_distraction_density(uid, rid, ts, 120),
            'prev_dist_duration':      get_prev_dist_duration(uid, rid, ts),
            'label': 1,
        })
    return pd.DataFrame(rows)

def build_negatives(H, sample_every=NEG_SAMPLE_EVERY,
                    user_ids=None, arousal_bl=None, arousal_med=None):
    # user_ids / arousal_bl / arousal_med: same fold-filtering contract as build_positives.
    _abl = arousal_bl  if arousal_bl  is not None else user_arousal_baseline
    _amed= arousal_med if arousal_med is not None else global_arousal_median
    rows = []
    error_floor = {
        (uid, rid): set(grp['timestamp'].dt.floor('s'))
        for (uid, rid), grp in errors_dist.groupby(['user_id', 'run_id'])
    }
    for (uid, rid), wins in windows_by_session.items():
        if user_ids is not None and uid not in user_ids:
            continue
        wins_s  = wins.sort_values('timestamp_start').reset_index(drop=True)
        b_rate  = user_baselines.get(uid, p_baseline_global)
        err_set = error_floor.get((uid, rid), set())
        u_aro   = _abl.get(uid, _amed)

        # â”€â”€ Inside distraction windows â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        for w_idx, win in wins_s.iterrows():
            dur       = (win['timestamp_end'] - win['timestamp_start']).total_seconds()
            m_prob_w  = win['model_prob_start']
            emo_enc, pred_enc = encode_row(win['emotion_label_start'], win['model_pred_start'])
            # prev_dist_duration: duration of the window *before* this one (if any)
            prev_dur_w = (
                (wins_s.iloc[w_idx - 1]['timestamp_end'] - wins_s.iloc[w_idx - 1]['timestamp_start']).total_seconds()
                if w_idx > 0 else 0.0
            )
            for offset in np.arange(0, dur, sample_every):
                ts = win['timestamp_start'] + pd.Timedelta(seconds=offset)
                if ts.floor('s') in err_set:
                    continue
                rows.append({
                    'user_id':                uid,
                    'distraction_active':     1,
                    'time_since_last_dist':   0.0,
                    'time_in_current_dist':   min(offset, float(H)),
                    'true_time_since':        0.0,
                    'true_time_in_dist':      float(offset),
                    'cognitive_load_decay':   1.0,
                    'model_prob':             m_prob_w,
                    'model_pred_enc':         pred_enc,
                    'emotion_prob':           win['emotion_prob_start'],
                    'emotion_label_enc':      emo_enc,
                    'arousal_deviation':      m_prob_w - u_aro,
                    'baseline_error_rate':    b_rate,
                    'distraction_density_30':  get_distraction_density(uid, rid, ts, 30),
                    'distraction_density_60':  get_distraction_density(uid, rid, ts, 60),
                    'distraction_density_120': get_distraction_density(uid, rid, ts, 120),
                    'prev_dist_duration':      prev_dur_w,
                    'label': 0,
                })

        # â”€â”€ Inter-window gaps â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        for i in range(len(wins_s) - 1):
            gap_start = wins_s['timestamp_end'].iloc[i]
            gap_end   = wins_s['timestamp_start'].iloc[i + 1]
            gap_len   = (gap_end - gap_start).total_seconds()
            if gap_len <= 0:
                continue
            win_state         = wins_s.iloc[i]
            m_prob_g          = win_state['model_prob_end']
            emo_enc, pred_enc = encode_row(win_state['emotion_label_end'], win_state['model_pred_end'])
            # prev_dist_duration is constant for this whole gap (duration of win_state)
            prev_dur_g = max(0.0, (win_state['timestamp_end'] - win_state['timestamp_start']).total_seconds())
            for offset in np.arange(0, gap_len, sample_every):
                ts     = gap_start + pd.Timedelta(seconds=offset)
                true_t = offset
                if ts.floor('s') in err_set:
                    continue
                cld = np.exp(-true_t / max(H, 1e-9))
                rows.append({
                    'user_id':                uid,
                    'distraction_active':     0,
                    'time_since_last_dist':   min(offset, float(H)),
                    'time_in_current_dist':   0.0,
                    'true_time_since':        true_t,
                    'true_time_in_dist':      0.0,
                    'cognitive_load_decay':   cld,
                    'model_prob':             m_prob_g,
                    'model_pred_enc':         pred_enc,
                    'emotion_prob':           win_state['emotion_prob_end'],
                    'emotion_label_enc':      emo_enc,
                    'arousal_deviation':      m_prob_g - u_aro,
                    'baseline_error_rate':    b_rate,
                    'distraction_density_30':  get_distraction_density(uid, rid, ts, 30),
                    'distraction_density_60':  get_distraction_density(uid, rid, ts, 60),
                    'distraction_density_120': get_distraction_density(uid, rid, ts, 120),
                    'prev_dist_duration':      prev_dur_g,
                    'label': 0,
                })
    return pd.DataFrame(rows)

def build_baseline_negatives(sample_every=NEG_SAMPLE_EVERY,
                              default_model_prob=None,
                              default_model_pred_enc=None,
                              default_emotion_prob=None,
                              default_emotion_label_enc=None,
                              user_ids=None,
                              arousal_bl=None, arousal_med=None):
    # Negatives from undistracted baseline sessions.
    # user_ids / arousal_bl / arousal_med: same fold-filtering contract as other builders.
    _abl = arousal_bl  if arousal_bl  is not None else user_arousal_baseline
    _amed= arousal_med if arousal_med is not None else global_arousal_median
    sentinel = float(H_CANDIDATES[-1])
    rows = []
    for _, row in driving_base.iterrows():
        uid          = row['user_id']
        if user_ids is not None and uid not in user_ids:
            continue
        run_duration = row['run_duration_seconds']
        b_rate       = user_baselines.get(uid, p_baseline_global)
        u_aro        = _abl.get(uid, _amed)
        m_prob       = default_model_prob if default_model_prob is not None else _amed
        for offset in np.arange(0, run_duration, sample_every):
            rows.append({
                'user_id':                uid,
                'distraction_active':     0,
                'time_since_last_dist':   sentinel,
                'time_in_current_dist':   0.0,
                'true_time_since':        sentinel,
                'true_time_in_dist':      0.0,
                'cognitive_load_decay':   0.0,      # no recent distraction
                'model_prob':             m_prob,
                'model_pred_enc':         default_model_pred_enc,
                'emotion_prob':           default_emotion_prob,
                'emotion_label_enc':      default_emotion_label_enc,
                'arousal_deviation':      m_prob - u_aro,
                'baseline_error_rate':    b_rate,
                'distraction_density_30':  0,        # no distractions in baseline session
                'distraction_density_60':  0,
                'distraction_density_120': 0,
                'prev_dist_duration':      0.0,      # no prior distraction
                'label': 0,
            })
    return pd.DataFrame(rows)

# â”€â”€ 9. Bootstrap CI helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def bootstrap_ci(y_true, y_score, metric_fn, n=N_BOOTSTRAP, ci=0.95, seed=RANDOM_SEED):
    rng    = np.random.RandomState(seed)
    scores = []
    idx    = np.arange(len(y_true))
    for _ in range(n):
        boot = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_true[boot])) < 2:
            continue
        scores.append(metric_fn(y_true[boot], y_score[boot]))
    lo = np.percentile(scores, (1 - ci) / 2 * 100)
    hi = np.percentile(scores, (1 + ci) / 2 * 100)
    return np.mean(scores), lo, hi

# â”€â”€ 10. Precision at fixed recall levels â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def precision_at_recall(y_true, y_score, recall_levels):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    results = {}
    for r in recall_levels:
        mask       = rec >= r
        results[r] = float(prec[mask].max()) if mask.any() else 0.0
    return results

# â”€â”€ 11. Compute metrics helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def compute_metrics(y_true, y_score, thresh):
    yp  = (y_score >= thresh).astype(int)
    cm_ = confusion_matrix(y_true, yp)
    tn, fp, fn, tp = cm_.ravel() if cm_.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0
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

# â”€â”€ 12. Core: full nested LOSO evaluation for a given H â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def compute_fold_baseline_rate(user_ids_set):
    """
    Compute the baseline error rate (errors/second) from baseline driving sessions
    restricted to users in user_ids_set.
    If no baseline data exists for these users, fall back to the average of their
    per-user baseline rates (from user_baselines), or finally the global baseline.
    """
    # Filter baseline data to training users only
    base_errors = errors_base[errors_base['user_id'].isin(user_ids_set)]
    base_driving = driving_base[driving_base['user_id'].isin(user_ids_set)]

    total_sec = base_driving['run_duration_seconds'].sum()
    if total_sec > 0:
        return len(base_errors) / total_sec

    # No baseline driving seconds for these users â€“ use average of their per-user baselines
    train_user_baselines = [user_baselines[uid] for uid in user_ids_set if uid in user_baselines]
    if train_user_baselines:
        return float(np.mean(train_user_baselines))

    # Fallback to global baseline (should not happen with full data)
    return p_baseline_global

def full_nested_eval(H):
    # Fully leak-free nested LOSO-CV with fold-first architecture.
    #
    # Both leakages from the previous design are closed here:
    #
    #   Leakage-1 (arousal baseline): compute_arousal_baseline() is called with
    #     train_users only.  The test user is entirely absent; they receive the
    #     training-fold population median as their personal baseline (correct
    #     treatment for an unseen participant at inference time too).
    #
    #   Leakage-2 (GT rates): compute_gt_profiles() is called with train_users only.
    #     The calibration targets (isotonic regression) reflect only the error
    #     distribution of training users; the test user's post-distraction error
    #     pattern cannot influence how the calibrator is fitted.
    #
    # Architecture: fold-first (outer loop over test users, build samples inside).
    #   All sample construction is scoped to the relevant user set per fold.

    # Enumerate all users that appear in distraction data (union with errors_dist
    # ensures we never lose a user who has errors but a borderline window overlap).
    all_users = sorted(
        set(distractions['user_id'].unique()) |
        set(errors_dist['user_id'].unique())
    )

    true_labels, raw_probs, calib_probs, gt_probs = [], [], [], []
    dist_active_vals, time_since_vals, time_in_dist_vals = [], [], []

    for test_user in all_users:
        train_users = set(all_users) - {test_user}

        # â”€â”€ Fold-specific GT rates and arousal baseline (train users only) â”€â”€â”€
        fold_gt_profile = compute_gt_profiles(train_users)
        fold_gt_rates = fold_gt_profile['post_rates']
        fold_gt_active_rates = fold_gt_profile['active_rates']
        fold_baseline_rate = compute_fold_baseline_rate(train_users)
        fold_arousal_bl, fold_arousal_med = compute_arousal_baseline(train_users)

        fold_kw = dict(arousal_bl=fold_arousal_bl, arousal_med=fold_arousal_med)

        # â”€â”€ Build training samples â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        train_pos = build_positives(H, user_ids=train_users, **fold_kw)
        train_neg = build_negatives(H, user_ids=train_users, **fold_kw)
        train_df  = pd.concat([train_pos, train_neg], ignore_index=True)                       .dropna(subset=FEATURE_COLS)

        if train_df.empty or train_df['label'].nunique() < 2:
            continue   # degenerate fold; skip

        # Defaults for baseline negatives derived from safe train seconds only
        safe_mask = train_df['label'] == 0
        if safe_mask.sum() > 0:
            def_mp  = train_df.loc[safe_mask, 'model_prob'].median()
            def_mpe = train_df.loc[safe_mask, 'model_pred_enc'].mode()[0]
            def_ep  = train_df.loc[safe_mask, 'emotion_prob'].median()
            def_ele = train_df.loc[safe_mask, 'emotion_label_enc'].mode()[0]
        else:
            def_mp  = 0.5
            def_mpe = le_pred.transform([UNKNOWN_LABEL])[0]
            def_ep  = 0.5
            def_ele = le_emotion.transform([UNKNOWN_LABEL])[0]

        base_defaults = dict(default_model_prob=def_mp, default_model_pred_enc=def_mpe,
                             default_emotion_prob=def_ep, default_emotion_label_enc=def_ele)

        base_neg_tr = build_baseline_negatives(**base_defaults,
                                               user_ids=train_users, **fold_kw)
        train_full = pd.concat([train_df, base_neg_tr], ignore_index=True)                        .dropna(subset=FEATURE_COLS)

        X_tr        = train_full[FEATURE_COLS].values.astype(float)
        y_tr        = train_full['label'].values.astype(int)
        pos_rate_tr = y_tr.mean()
        if pos_rate_tr in (0.0, 1.0):
            continue
        spw_tr = (1 - pos_rate_tr) / pos_rate_tr

        clf = xgb.XGBClassifier(scale_pos_weight=spw_tr, **XGB_PARAMS)
        clf.fit(X_tr, y_tr)

        # Fit isotonic calibrator on training fold raw scores â†’ fold GT rates
        raw_tr = clf.predict_proba(X_tr)[:, 1]
        gt_tr  = np.array([assign_gt_prob(r, H, fold_gt_rates, fold_baseline_rate, fold_gt_active_rates)
                           for _, r in train_full.iterrows()])
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(raw_tr, gt_tr)

        # â”€â”€ Build test samples (test user only, fold arousal/GT applied) â”€â”€â”€â”€â”€
        # The test user is absent from fold_arousal_bl; they receive fold_arousal_med
        # (training-population median) as their personal baseline â€” the correct
        # treatment for an unseen participant.
        test_pos = build_positives(H, user_ids={test_user}, **fold_kw)
        test_neg = build_negatives(H, user_ids={test_user}, **fold_kw)
        test_df  = pd.concat([test_pos, test_neg], ignore_index=True)                      .dropna(subset=FEATURE_COLS)

        if test_df.empty:
            continue

        base_neg_te = build_baseline_negatives(**base_defaults,
                                               user_ids={test_user}, **fold_kw)
        test_full = pd.concat([test_df, base_neg_te], ignore_index=True)                       .dropna(subset=FEATURE_COLS)

        X_te   = test_full[FEATURE_COLS].values.astype(float)
        y_te   = test_full['label'].values.astype(int)
        raw_te = clf.predict_proba(X_te)[:, 1]
        cal_te = iso.predict(raw_te)

        true_labels.extend(y_te)
        raw_probs.extend(raw_te)
        calib_probs.extend(cal_te)
        gt_probs.extend([assign_gt_prob(r, H, fold_gt_rates, fold_baseline_rate, fold_gt_active_rates)
                         for _, r in test_full.iterrows()])
        dist_active_vals.extend(test_full['distraction_active'].tolist())
        time_since_vals.extend(test_full['time_since_last_dist'].tolist())
        time_in_dist_vals.extend(test_full['time_in_current_dist'].tolist())

    y_true_arr = np.array(true_labels)
    raw_arr    = np.array(raw_probs)
    cal_arr    = np.array(calib_probs)
    gt_arr     = np.array(gt_probs)

    auc_pr  = average_precision_score(y_true_arr, raw_arr)
    auc_roc = roc_auc_score(y_true_arr, raw_arr)
    cal_mae = float(np.mean(np.abs(cal_arr - gt_arr)))
    cal_rmse = float(np.sqrt(np.mean((cal_arr - gt_arr) ** 2)))
    selection_score = float(auc_pr - H_SELECTION_CAL_MAE_WEIGHT * cal_mae)

    return {
        'H':               H,
        'AUC-PR':          auc_pr,
        'AUC-ROC':         auc_roc,
        'Calib_MAE':       cal_mae,
        'Calib_RMSE':      cal_rmse,
        'selection_score': selection_score,
        'y_true':          y_true_arr,
        'raw':             raw_arr,
        'calib':           cal_arr,
        'gt':              gt_arr,
        'dist_active':     np.array(dist_active_vals),
        'time_since':      np.array(time_since_vals),
        'time_in_dist':    np.array(time_in_dist_vals),
    }

# â”€â”€ 13. Fast H grid search (no calibration, no baseline negatives) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Note on residual leakage: the grid search uses global arousal baselines, so
# each test fold's arousal_deviation is computed with that user's own windows.
# This is intentional: fixing it would require per-fold sample rebuilding on
# every H candidate (~25Ã— cost) for a screening step whose output is only used
# to select a shortlist â€” not to report metrics.  The final metrics come from
# full_nested_eval, which is fully leak-free.
print("\n" + "=" * 70)
print("H GRID SEARCH (Leave-One-User-Out, fast)")
print("=" * 70)
print(f"{'H (s)':<8} {'AUC-PR':<10} {'AUC-ROC':<10} {'N':<8} {'Pos%'}")
print("-" * 50)

logo         = LeaveOneGroupOut()
grid_results = []

for H in H_CANDIDATES:
    pos = build_positives(H)
    neg = build_negatives(H)
    df  = pd.concat([pos, neg], ignore_index=True).dropna(subset=FEATURE_COLS)

    X      = df[FEATURE_COLS].values.astype(float)
    y      = df['label'].values.astype(int)
    groups = df['user_id'].values

    pos_rate         = y.mean()
    scale_pos_weight = (1 - pos_rate) / pos_rate

    y_true_all, y_prob_all = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx])), \
            "LEAKAGE DETECTED"
        clf = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, **XGB_PARAMS)
        clf.fit(X[train_idx], y[train_idx])
        y_true_all.extend(y[test_idx])
        y_prob_all.extend(clf.predict_proba(X[test_idx])[:, 1])

    auc_pr  = average_precision_score(y_true_all, y_prob_all)
    auc_roc = roc_auc_score(y_true_all, y_prob_all)
    grid_results.append({'H': H, 'AUC-PR': auc_pr, 'AUC-ROC': auc_roc,
                         'n_samples': len(df), 'pos_pct': pos_rate * 100})
    print(f"{H:<8} {auc_pr:<10.4f} {auc_roc:<10.4f} {len(df):<8} {pos_rate*100:.1f}%")

results_df = pd.DataFrame(grid_results)

# â”€â”€ 14. full nested evaluation of top-N candidates â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n" + "=" * 70)
print(f"FULL NESTED EVALUATION OF TOP-{TOP_N_CANDIDATES} CANDIDATES")
print("=" * 70)

top_candidates = results_df.nlargest(TOP_N_CANDIDATES, 'AUC-PR')['H'].tolist()

print(f"\n  Top {TOP_N_CANDIDATES} H values by grid AUC-PR : {top_candidates}")
print(f"  Running full nested evaluation for each ...\n")
print(f"  {'H (s)':<8} {'Nested AUC-PR':<16} {'Nested AUC-ROC':<16} {'Cal MAE':<10} {'Score':<10} {'Grid AUC-PR'}")
print("  " + "-" * 88)

nested_results = []
for H_cand in top_candidates:
    grid_aucpr = results_df.loc[results_df['H'] == H_cand, 'AUC-PR'].values[0]
    res = full_nested_eval(H_cand)
    nested_results.append(res)
    current_best = max(nested_results, key=lambda r: r['selection_score'])
    print(f"  {H_cand:<8} {res['AUC-PR']:<16.4f} {res['AUC-ROC']:<16.4f} "
          f"{res['Calib_MAE']:<10.4f} {res['selection_score']:<10.4f} "
          f"{grid_aucpr:.4f}  {'<- winner (so far)' if res is current_best else ''}")

# Pick the winner
best_result = max(nested_results, key=lambda r: r['selection_score'])
best_H      = best_result['H']

print(f"\n  Best H (nested) = {best_H}s  "
      f"(nested AUC-PR={best_result['AUC-PR']:.4f}, "
      f"AUC-ROC={best_result['AUC-ROC']:.4f}, "
      f"Cal_MAE={best_result['Calib_MAE']:.4f}, "
      f"score={best_result['selection_score']:.4f})")

# Unpack the winner's arrays for all downstream reporting
y_true_arr       = best_result['y_true']
raw_arr          = best_result['raw']
cal_arr          = best_result['calib']
gt_arr           = best_result['gt']
dist_active_arr  = best_result['dist_active']
time_since_arr   = best_result['time_since']
time_in_dist_arr = best_result['time_in_dist']

# â”€â”€ 15. Full metrics on winner â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n" + "=" * 70)
print(f"FULL EVALUATION  (H={best_H}s, nested calibration, realistic baseline)")
print("=" * 70)

prec_c, rec_c, thresh_c = precision_recall_curve(y_true_arr, raw_arr)
f1_c        = 2 * prec_c[:-1] * rec_c[:-1] / (prec_c[:-1] + rec_c[:-1] + 1e-9)
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

print("\nâ”€â”€ Model Metrics (raw XGBoost) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
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

print("\nâ”€â”€ Precision @ Fixed Recall (raw XGBoost) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
for r in RECALL_LEVELS:
    print(f"  P @ Recall={r:.0%} : {m_raw[f'P@R={int(r*100)}']:.4f}")

print(f"\nâ”€â”€ Classification Report (raw XGBoost @ t={best_thresh:.2f}) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
print(classification_report(y_true_arr, y_pred_raw, target_names=['Safe', 'Error']))

# â”€â”€ 16. Ground truth validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n" + "=" * 70)
print("GROUND TRUTH VALIDATION (with nested calibration)")
print("=" * 70)

sentinel          = float(H_CANDIDATES[-1])
mask_active       = dist_active_arr == 1
mask_hangover     = (~mask_active) & (time_since_arr < best_H)
mask_true_baseline = (~mask_active) & (time_since_arr >= sentinel)

df_gt = pd.DataFrame({
    'distraction_active':   dist_active_arr,
    'time_since_last_dist': time_since_arr,
    'time_in_current_dist': time_in_dist_arr,
    'label':                y_true_arr,
    'raw_prob':             raw_arr,
    'calib_prob':           cal_arr,
    'gt_prob':              gt_arr,
})

# â”€â”€ Risk stratification â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
conditions = [
    ('Active distraction',                  mask_active),
    (f'Hangover (0â€“{best_H}s)',             mask_hangover),
    ('Baseline (driving without distractions)', mask_true_baseline),
]

strat_rows = []
print(f"\nâ”€â”€ Risk Stratification â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
print(f"  {'Condition':<40} {'GT rate':>10} {'Raw mean':>10} {'Calib mean':>12} {'N rows':>8}")
print("  " + "-" * 86)
for label, mask in conditions:
    subset = df_gt[mask]
    if len(subset) == 0:
        continue
    gt_rate  = subset['label'].mean()
    raw_mean = subset['raw_prob'].mean()
    cal_mean = subset['calib_prob'].mean()
    print(f"  {label:<40} {gt_rate:>10.4f} {raw_mean:>10.4f} {cal_mean:>12.4f} {len(subset):>8}")
    strat_rows.append({'condition': label, 'gt_error_rate': gt_rate,
                       'raw_mean': raw_mean, 'calib_mean': cal_mean, 'n': len(subset)})

strat_df = pd.DataFrame(strat_rows)
fig, ax  = plt.subplots(figsize=(10, 5))
x, w     = np.arange(len(strat_df)), 0.25
ax.bar(x - w, strat_df['gt_error_rate'], w, label='GT error rate',
       color='#d62728', alpha=0.85)
ax.bar(x,     strat_df['raw_mean'],      w, label='Raw model mean',
       color='steelblue', alpha=0.85)
ax.bar(x + w, strat_df['calib_mean'],    w, label='Calibrated model mean',
       color='green', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(strat_df['condition'], rotation=15, ha='right')
ax.set_ylabel('Rate / Probability')
ax.set_title('Risk Stratification: Ground Truth vs Model (raw & calibrated)')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{EVAL_OUT}gt_risk_stratification_calibrated.png', dpi=150); plt.close()
print(f"  âœ“ Saved â†’ {EVAL_OUT}gt_risk_stratification_calibrated.png")

# â”€â”€ Temporal decay â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\nâ”€â”€ Temporal Decay (seconds 0..{best_H}) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")

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

bin_size_print = 5
print(f"  {'Bin':<12} {'GT rate(/s)':>12} {'Exposure':>10} {'Errors':>8} "
      f"{'Raw mean':>12} {'Calib mean':>12}")
print("  " + "-" * 72)
# print(f"  {'0 (active)':<12} {gt_rate_per_sec[0]:>12.4f} {total_dist_s:>10.0f} "
#       f"{errors_inside:>8} {model_mean_by_sec[0]:>12.4f} {calib_mean_by_sec[0]:>12.4f}")
for b_start in range(1, best_H + 1, bin_size_print):
    b_end   = min(b_start + bin_size_print, best_H + 1)
    lbl     = f"{b_start}â€“{b_end-1}s"
    n_err   = int(gt_errors[b_start:b_end].sum())
    exp_sum = gt_exposure[b_start:b_end].sum()
    rate_m  = float(np.nanmean(gt_rate_per_sec[b_start:b_end]))
    m_mean  = float(np.nanmean(model_mean_by_sec[b_start:b_end]))
    c_mean  = float(np.nanmean(calib_mean_by_sec[b_start:b_end]))
    print(f"  {lbl:<12} {rate_m:>12.4f} {exp_sum:>10.0f} {n_err:>8} "
          f"{m_mean:>12.4f} {c_mean:>12.4f}")
print(f"  {'Baseline':<12} {p_baseline_global:>12.4f} {'â€”':>10} {'â€”':>8} {'â€”':>12} {'â€”':>12}")

# â”€â”€ Calibration MAE / RMSE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
errors_raw = df_gt['raw_prob'].values   - df_gt['gt_prob'].values
errors_cal = df_gt['calib_prob'].values - df_gt['gt_prob'].values
mae_raw    = np.mean(np.abs(errors_raw));  rmse_raw = np.sqrt(np.mean(errors_raw**2))
mae_cal    = np.mean(np.abs(errors_cal));  rmse_cal = np.sqrt(np.mean(errors_cal**2))
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
    print(f"  {lbl:<40} {sub['gt_prob'].mean():>9.4f} {sub['raw_prob'].mean():>11.4f} "
          f"{sub['calib_prob'].mean():>11.4f} {mae_r:>8.4f} {mae_c:>8.4f} {len(sub):>6}")

# â”€â”€ Temporal decay plot â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
seconds   = np.arange(0, best_H + 1)
valid_gt  = ~np.isnan(gt_rate_per_sec[:best_H+1])
valid_raw = ~np.isnan(model_mean_by_sec)
valid_cal = ~np.isnan(calib_mean_by_sec)

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(seconds[valid_gt],  gt_rate_per_sec[:best_H+1][valid_gt],
        'o-',  color='#d62728', lw=2, markersize=4, label='GT empirical rate (errors/s)')
ax.plot(seconds[valid_raw], model_mean_by_sec[valid_raw],
        's--', color='steelblue', lw=2, markersize=4, label='Raw model mean')
ax.plot(seconds[valid_cal], calib_mean_by_sec[valid_cal],
        'd-.', color='green', lw=2, markersize=4, label='Calibrated model mean')
ax.axhline(y=p_baseline_global, color='grey', linestyle=':', lw=1.5,
           label=f'Baseline rate ({p_baseline_global*100:.2f}%/s)')
ax.axvline(x=0.5, color='black', linestyle=':', lw=1, alpha=0.4)
ax.text(0.1, ax.get_ylim()[1] * 0.95, 'inside window', fontsize=8, alpha=0.6)
ax.text(1.5, ax.get_ylim()[1] * 0.95, 'postâ€‘distraction recovery â†’', fontsize=8, alpha=0.6)
ax.set_xlabel('Bin (0 = inside window, 1..H = seconds after distraction ended)')
ax.set_ylabel('Rate / Probability')
ax.set_title(f'GT Error Rate vs Model Prediction (H={best_H}s, raw & calibrated)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{EVAL_OUT}gt_temporal_decay_calibrated.png', dpi=150); plt.close()
print(f"\n  âœ“ Saved â†’ {EVAL_OUT}gt_temporal_decay_calibrated.png")

# â”€â”€ Save tables â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
decay_rows = []
for n in range(best_H + 1):
    decay_rows.append({
        'bin':                  n,
        'gt_rate_per_sec':      gt_rate_per_sec[n] if n <= max_H_gt else np.nan,
        'raw_model_mean':       model_mean_by_sec[n],
        'calibrated_model_mean': calib_mean_by_sec[n],
        'n_errors':             int(gt_errors[n]) if n <= max_H_gt else 0,
        'exposure':             gt_exposure[n]   if n <= max_H_gt else 0,
    })
pd.DataFrame(decay_rows).to_csv(f'{EVAL_OUT}gt_temporal_decay.csv', index=False)

cont_df = df_gt.copy()
cont_df['abs_error_raw'] = np.abs(cont_df['raw_prob']   - cont_df['gt_prob'])
cont_df['abs_error_cal'] = np.abs(cont_df['calib_prob'] - cont_df['gt_prob'])
cont_df.to_csv(f'{EVAL_OUT}gt_continuous_comparison.csv', index=False)

pd.DataFrame([{'MAE_raw': mae_raw, 'RMSE_raw': rmse_raw,
               'MAE_cal': mae_cal, 'RMSE_cal': rmse_cal,
               'p_bin0': p_bin0, 'p_baseline': p_baseline_global}]
             ).to_csv(f'{EVAL_OUT}gt_continuous_metrics.csv', index=False)

# Also save the nested H comparison table
nested_summary = pd.DataFrame([{
    'H':             r['H'],
    'grid_AUC_PR':   results_df.loc[results_df['H'] == r['H'], 'AUC-PR'].values[0],
    'nested_AUC_PR': r['AUC-PR'],
    'nested_AUC_ROC': r['AUC-ROC'],
    'nested_Cal_MAE': r['Calib_MAE'],
    'nested_Cal_RMSE': r['Calib_RMSE'],
    'selection_score': r['selection_score'],
} for r in nested_results])
nested_summary.to_csv(f'{EVAL_OUT}h_selection_nested.csv', index=False)

print(f"  âœ“ Saved â†’ {EVAL_OUT}gt_temporal_decay.csv")
print(f"  âœ“ Saved â†’ {EVAL_OUT}gt_continuous_comparison.csv")
print(f"  âœ“ Saved â†’ {EVAL_OUT}gt_continuous_metrics.csv")
print(f"  âœ“ Saved â†’ {EVAL_OUT}h_selection_nested.csv")

# â”€â”€ 17. Final model (all data) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\nâ”€â”€ Training Final Model (H={best_H}s, all data) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")

pos_f = build_positives(best_H)
neg_f = build_negatives(best_H)
all_data = pd.concat([pos_f, neg_f], ignore_index=True)
safe_global = all_data[all_data['label'] == 0]

if len(safe_global) > 0:
    default_model_prob_global        = safe_global['model_prob'].median()
    default_model_pred_enc_global    = safe_global['model_pred_enc'].mode()[0]
    default_emotion_prob_global      = safe_global['emotion_prob'].median()
    default_emotion_label_enc_global = safe_global['emotion_label_enc'].mode()[0]
else:
    default_model_prob_global        = 0.5
    default_model_pred_enc_global    = le_pred.transform([UNKNOWN_LABEL])[0]
    default_emotion_prob_global      = 0.5
    default_emotion_label_enc_global = le_emotion.transform([UNKNOWN_LABEL])[0]

base_f = build_baseline_negatives(
    default_model_prob=default_model_prob_global,
    default_model_pred_enc=default_model_pred_enc_global,
    default_emotion_prob=default_emotion_prob_global,
    default_emotion_label_enc=default_emotion_label_enc_global,
)

df_f      = pd.concat([pos_f, neg_f, base_f], ignore_index=True).dropna(subset=FEATURE_COLS)
X_f, y_f  = df_f[FEATURE_COLS].values.astype(float), df_f['label'].values.astype(int)
spw_f     = (1 - y_f.mean()) / y_f.mean()

base_clf = xgb.XGBClassifier(scale_pos_weight=spw_f, **XGB_PARAMS)
base_clf.fit(X_f, y_f)

# Fit final calibrator on all data
raw_all = base_clf.predict_proba(X_f)[:, 1]
gt_all  = np.array([assign_gt_prob(row, best_H, gt_rate_per_sec, p_baseline_global, gt_active_rate)
                    for _, row in df_f.iterrows()])
final_iso = IsotonicRegression(out_of_bounds='clip')
final_iso.fit(raw_all, gt_all)

artifact = {
    'model':                 base_clf,
    'calibrator':            final_iso,
    'best_H':                best_H,
    'best_thresh':           best_thresh,
    'feature_cols':          FEATURE_COLS,
    'le_emotion':            le_emotion,
    'le_pred':               le_pred,
    'user_baselines':        user_baselines,
    'p_baseline_global':     p_baseline_global,
    'user_arousal_baseline': user_arousal_baseline,   # for arousal_deviation at inference
    'global_arousal_median': global_arousal_median,
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
print(f"  Saved -> {MODEL_OUT}")

# â”€â”€ 18. Inference helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def predict_fitness(
    distraction_active:              Union[bool, int],
    seconds_since_last_distraction:  float,
    emotion_label:                   Optional[str] = None,
    emotion_prob:                    float = 0.5,
    arousal_pred_label:              Optional[str] = None,
    arousal_pred_prob:               float = 0.5,
    user_id:                         Optional[str] = None,
    # New context features â€” pass what you know; defaults give conservative estimates
    distraction_density_30:          int   = 0,
    distraction_density_60:          int   = 0,
    distraction_density_120:         int   = 0,
    prev_distraction_duration_s:     float = 0.0,
    artifact_path:                   str   = MODEL_OUT,
) -> Dict[str, object]:
    # Returns calibrated probability of an error in the next second.
    # New parameters:
    #   distraction_density_*        : # windows that ended in the last 30/60/120 s
    #   prev_distraction_duration_s  : duration (s) of the most recent completed window
    art    = joblib.load(artifact_path)
    H      = art['best_H']
    thresh = art['best_thresh']
    warns  = []

    try:
        dist_active = int(bool(distraction_active))
    except Exception:
        dist_active = 0
        warns.append("distraction_active invalid, defaulting to 0")

    try:
        t_input = float(seconds_since_last_distraction)
        if t_input < 0:
            warns.append(f"seconds_since_last_distraction={t_input} < 0, clamped to 0")
            t_input = 0.0
        if dist_active == 1:
            t_since = 0.0
            t_in_current = min(t_input, float(H))
        else:
            t_since = min(t_input, float(H))
            t_in_current = 0.0
    except Exception:
        t_since = float(H)
        t_in_current = 0.0
        warns.append("seconds_since_last_distraction invalid, defaulting to H (recovered)")

    # cognitive_load_decay derived from t_since and H
    cld = 1.0 if dist_active == 1 else np.exp(-t_since / max(H, 1e-9))

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

    a_prob = _clip_prob(arousal_pred_prob, 'arousal_pred_prob')
    e_prob = _clip_prob(emotion_prob,      'emotion_prob')

    _le_emotion = art['le_emotion']
    _le_pred    = art['le_pred']

    def _encode(le, val):
        s = safe_str(val)
        if s in le.classes_:
            return int(le.transform([s])[0])
        warns.append(f"Unseen label '{s}', using 'unknown'")
        return int(le.transform(['unknown'])[0])

    emo_enc  = _encode(_le_emotion, emotion_label)
    pred_enc = _encode(_le_pred,    arousal_pred_label)

    b_rate = art['user_baselines'].get(user_id, art['p_baseline_global'])
    if user_id is not None and user_id not in art['user_baselines']:
        warns.append(f"user_id='{user_id}' not in training set, using global baseline")

    # arousal_deviation using stored per-user baseline (falls back to 0 if unknown user)
    u_aro_base = art.get('user_arousal_baseline', {}).get(
        user_id, art.get('global_arousal_median', a_prob))
    aro_dev = a_prob - u_aro_base
    if user_id is None:
        warns.append("user_id not provided; arousal_deviation set to 0 (unknown personal baseline)")
        aro_dev = 0.0

    sample = pd.DataFrame([{
        'distraction_active':     dist_active,
        'time_since_last_dist':   t_since,
        'time_in_current_dist':   t_in_current,
        'cognitive_load_decay':   cld,
        'model_prob':             a_prob,
        'model_pred_enc':         pred_enc,
        'emotion_prob':           e_prob,
        'emotion_label_enc':      emo_enc,
        'arousal_deviation':      aro_dev,
        'baseline_error_rate':    b_rate,
        'distraction_density_30':  max(0, int(distraction_density_30)),
        'distraction_density_60':  max(0, int(distraction_density_60)),
        'distraction_density_120': max(0, int(distraction_density_120)),
        'prev_dist_duration':      max(0.0, float(prev_distraction_duration_s)),
    }])

    raw_prob        = float(art['model'].predict_proba(sample[art['feature_cols']])[0, 1])
    calibrated_prob = float(art['calibrator'].predict([raw_prob])[0])

    return {
        'error_probability': round(calibrated_prob, 4),
        'fitness_to_drive':  round(1.0 - calibrated_prob, 4),
        'alert':             raw_prob >= thresh,
        'input_warnings':    warns,
    }

# â”€â”€ 19. Summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Best H (nested selection)  : {best_H}s")
print()
print("  H selection comparison (top candidates):")
print(f"  {'H':>4}  {'Grid AUC-PR':>12}  {'Nested AUC-PR':>14}  {'Nested AUC-ROC':>15}  {'Cal MAE':>9}  {'Score':>9}")
print("  " + "-" * 86)
for _, row in nested_summary.sort_values('nested_AUC_PR', ascending=False).iterrows():
    marker = " <- selected" if row['H'] == best_H else ""
    print(f"  {int(row['H']):>4}  {row['grid_AUC_PR']:>12.4f}  "
          f"{row['nested_AUC_PR']:>14.4f}  {row['nested_AUC_ROC']:>15.4f}  "
          f"{row['nested_Cal_MAE']:>9.4f}  {row['selection_score']:>9.4f}{marker}")
print()
print(f"  Raw XGBoost AUC-PR  : {m_raw['AUC-PR']:.4f}  95% CI [{lo_aucpr:.4f} - {hi_aucpr:.4f}]")
print(f"  Raw XGBoost AUC-ROC : {m_raw['AUC-ROC']:.4f}  95% CI [{lo_aucroc:.4f} - {hi_aucroc:.4f}]")
print(f"  Raw XGBoost MCC     : {m_raw['MCC']:.4f}  95% CI [{lo_mcc:.4f} - {hi_mcc:.4f}]")
print(f"  Raw XGBoost Brier   : {m_raw['Brier']:.4f}  95% CI [{lo_brier:.4f} - {hi_brier:.4f}]")
print(f"  MAE  (raw vs GT)    : {mae_raw:.4f}")
print(f"  RMSE (raw vs GT)    : {rmse_raw:.4f}")
print(f"  MAE  (calibrated)   : {mae_cal:.4f}")
print(f"  RMSE (calibrated)   : {rmse_cal:.4f}")

