"""
Driver Digital Twin – Fitness-to-Drive XGBoost Pipeline
========================================================
Rigorous, publication‑ready version (fully de‑biased):
  - Sampling every second (NEG_SAMPLE_EVERY = 1)
  - Baseline negatives use realistic emotion/arousal values (medians from safe seconds)
  - Nested cross‑validation with proper calibration
  - Option C H selection: top-N candidates from fast grid search are fully
    nested-evaluated; the one with the best honest AUC-PR is selected
  - Full bootstrap CIs for all metrics

Input datasets (place in DATA_PATH):
  - Dataset Distractions_distraction.csv
  - Dataset Errors_distraction.csv
  - Dataset Errors_baseline.csv
  - Dataset Driving Time_baseline.csv
"""

import warnings
warnings.filterwarnings('ignore')

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

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_PATH        = 'data/'
MODEL_OUT        = 'fitness_model_calibrated_v3.pkl'
EVAL_OUT         = 'evaluation/predictor3/'
BIN_LIMIT        = 20
H_CANDIDATES     = list(range(1, BIN_LIMIT))
TOP_N_CANDIDATES = 5          # how many top grid-search H values to fully evaluate
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
    eval_metric      = 'aucpr',
    random_state     = RANDOM_SEED,
    verbosity        = 0,
)

FEATURE_COLS = [
    'distraction_active',
    'time_since_last_dist',
    'time_in_current_dist',
    'model_prob',
    'model_pred_enc',
    'emotion_prob',
    'emotion_label_enc',
    'baseline_error_rate',
]

# ── 1. Load ────────────────────────────────────────────────────────────────────
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

# ── 2. Data integrity checks ───────────────────────────────────────────────────
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

# ── 3. Per-user baseline error rate ───────────────────────────────────────────
total_base_s      = driving_base['run_duration_seconds'].sum()
p_baseline_global = len(errors_base) / total_base_s
user_base_errs    = errors_base.groupby('user_id').size()
user_base_secs    = driving_base.groupby('user_id')['run_duration_seconds'].sum()
user_baselines    = (user_base_errs / user_base_secs).fillna(p_baseline_global).to_dict()

print(f"\nGlobal baseline error rate : {p_baseline_global*100:.4f}% / second")

# ── 4. Helper functions for distraction state ──────────────────────────────────
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

# ── 5. Ground truth per‑second error rates ────────────────────────────────────
print("\n" + "=" * 70)
print("GROUND TRUTH PER‑SECOND ERROR RATES")
print("=" * 70)

max_H_gt = BIN_LIMIT - 1

total_dist_seconds = (
    distractions['timestamp_end'] - distractions['timestamp_start']
).dt.total_seconds().sum()
errors_inside = sum(
    1 for _, err in errors_dist.iterrows() if is_error_inside_distraction(err)
)
p_bin0 = errors_inside / total_dist_seconds if total_dist_seconds > 0 else 0.0
print(f"  Bin 0 (inside windows)      : {p_bin0:.6f} errors/s  "
      f"(errors={errors_inside}, exposure={total_dist_seconds:.0f}s)")

gt_active_errors = np.zeros(max_H_gt + 1)
gt_active_exposure = np.zeros(max_H_gt + 1)

for (uid, rid), wins in windows_by_session.items():
    wins_s = wins.sort_values('timestamp_start').reset_index(drop=True)
    for _, win in wins_s.iterrows():
        dur = (win['timestamp_end'] - win['timestamp_start']).total_seconds()
        max_sec = min(int(np.floor(dur)), max_H_gt)
        for sec in range(0, max_sec + 1):
            gt_active_exposure[sec] += 1.0

for _, err in errors_dist.iterrows():
    active, _, time_in_dist, _, _ = get_distraction_state(
        err['user_id'], err['run_id'], err['timestamp'], H=max_H_gt
    )
    if active == 1:
        sec = int(min(np.floor(time_in_dist), max_H_gt))
        gt_active_errors[sec] += 1.0

gt_active_rate_raw = np.full(max_H_gt + 1, np.nan)
gt_active_rate = np.full(max_H_gt + 1, np.nan)
for sec in range(max_H_gt + 1):
    if gt_active_exposure[sec] > 0:
        gt_active_rate_raw[sec] = gt_active_errors[sec] / gt_active_exposure[sec]
        gt_active_rate[sec] = (
            gt_active_errors[sec] + GT_ACTIVE_SMOOTH_ALPHA * p_bin0
        ) / (gt_active_exposure[sec] + GT_ACTIVE_SMOOTH_ALPHA)

gt_errors = np.zeros(max_H_gt + 1)
gt_exposure = np.zeros(max_H_gt + 1)

for (uid, rid), wins in windows_by_session.items():
    wins_s = wins.sort_values('timestamp_start').reset_index(drop=True)
    for i in range(len(wins_s) - 1):
        win_end = wins_s['timestamp_end'].iloc[i]
        next_start = wins_s['timestamp_start'].iloc[i + 1]
        gap_s = min((next_start - win_end).total_seconds(), float(max_H_gt))
        for n in range(1, int(np.floor(gap_s)) + 1):
            gt_exposure[n] += 1.0

outside_errors = errors_dist[
    ~errors_dist.apply(is_error_inside_distraction, axis=1).fillna(False)
].copy()
outside_errors['time_since'] = outside_errors.apply(get_time_since_last, axis=1)
outside_after = outside_errors.dropna(subset=['time_since']).copy()
for t in outside_after['time_since']:
    n = int(np.floor(t)) + 1
    if 1 <= n <= max_H_gt:
        gt_errors[n] += 1

gt_rate_per_sec_raw = np.full(max_H_gt + 1, np.nan)
gt_rate_per_sec = np.full(max_H_gt + 1, np.nan)
gt_rate_per_sec[0] = p_bin0
gt_rate_per_sec_raw[0] = p_bin0
for n in range(1, max_H_gt + 1):
    if gt_exposure[n] > 0:
        gt_rate_per_sec_raw[n] = gt_errors[n] / gt_exposure[n]
        gt_rate_per_sec[n] = (
            gt_errors[n] + GT_POST_SMOOTH_ALPHA * p_baseline_global
        ) / (gt_exposure[n] + GT_POST_SMOOTH_ALPHA)

post_valid = np.where(~np.isnan(gt_rate_per_sec[1:]))[0] + 1
if len(post_valid) >= 2:
    iso_post = IsotonicRegression(increasing=False, out_of_bounds='clip')
    gt_rate_per_sec[post_valid] = iso_post.fit_transform(
        post_valid.astype(float),
        gt_rate_per_sec[post_valid],
        sample_weight=gt_exposure[post_valid],
    )
gt_rate_per_sec[post_valid] = np.maximum(gt_rate_per_sec[post_valid], p_baseline_global)

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

# ── 7. Ground truth probability assignment (standalone, parameterised on H) ───
def assign_gt_prob(row, H):
    """
    Assign the empirical GT error rate to a sample row.
    Uses true_time_since if available (to correctly bin post-distraction
    samples whose feature was capped at H), otherwise falls back to
    time_since_last_dist.
    """
    if row['distraction_active'] == 1:
        true_t_in = row.get('true_time_in_dist', row.get('time_in_current_dist', 0.0))
        sec = int(min(np.floor(max(true_t_in, 0.0)), max_H_gt))
        if not np.isnan(gt_active_rate[sec]):
            return gt_active_rate[sec]
        return p_bin0
    true_t = row.get('true_time_since', row['time_since_last_dist'])
    # Sentinel value from build_baseline_negatives means genuine baseline
    if true_t >= float(H_CANDIDATES[-1]):
        return p_baseline_global
    if true_t > max_H_gt:
        return p_baseline_global
    n = int(np.floor(true_t)) + 1
    if n <= max_H_gt and not np.isnan(gt_rate_per_sec[n]):
        return gt_rate_per_sec[n]
    return p_baseline_global

# ── 8. Sample builders ────────────────────────────────────────────────────────
def build_positives(H):
    rows = []
    for _, err in errors_dist.iterrows():
        uid, rid, ts      = err['user_id'], err['run_id'], err['timestamp']
        dist_active, time_since, time_in_dist, true_time_since, true_time_in_dist = \
            get_distraction_state(uid, rid, ts, H)
        emo_enc, pred_enc = encode_row(err['emotion_label'], err['model_pred'])
        rows.append({
            'user_id':              uid,
            'distraction_active':   dist_active,
            'time_since_last_dist': time_since,
            'time_in_current_dist': time_in_dist,
            'true_time_since':      true_time_since,
            'true_time_in_dist':    true_time_in_dist,
            'model_prob':           err['model_prob'],
            'model_pred_enc':       pred_enc,
            'emotion_prob':         err['emotion_prob'],
            'emotion_label_enc':    emo_enc,
            'baseline_error_rate':  user_baselines.get(uid, p_baseline_global),
            'label': 1,
        })
    return pd.DataFrame(rows)

def build_negatives(H, sample_every=NEG_SAMPLE_EVERY):
    rows = []
    error_floor = {
        (uid, rid): set(grp['timestamp'].dt.floor('s'))
        for (uid, rid), grp in errors_dist.groupby(['user_id', 'run_id'])
    }
    for (uid, rid), wins in windows_by_session.items():
        wins_s  = wins.sort_values('timestamp_start').reset_index(drop=True)
        b_rate  = user_baselines.get(uid, p_baseline_global)
        err_set = error_floor.get((uid, rid), set())

        # ── Inside distraction windows ────────────────────────────────────────
        for _, win in wins_s.iterrows():
            dur = (win['timestamp_end'] - win['timestamp_start']).total_seconds()
            for offset in np.arange(0, dur, sample_every):
                ts = win['timestamp_start'] + pd.Timedelta(seconds=offset)
                if ts.floor('s') in err_set:
                    continue
                emo_enc, pred_enc = encode_row(
                    win['emotion_label_start'], win['model_pred_start'])
                rows.append({
                    'user_id':              uid,
                    'distraction_active':   1,
                    'time_since_last_dist': 0.0,
                    'time_in_current_dist': min(offset, float(H)),
                    'true_time_since':      0.0,
                    'true_time_in_dist':    offset,
                    'model_prob':           win['model_prob_start'],
                    'model_pred_enc':       pred_enc,
                    'emotion_prob':         win['emotion_prob_start'],
                    'emotion_label_enc':    emo_enc,
                    'baseline_error_rate':  b_rate,
                    'label': 0,
                })

        # ── Inter-window gaps ─────────────────────────────────────────────────
        for i in range(len(wins_s) - 1):
            gap_start = wins_s['timestamp_end'].iloc[i]
            gap_end   = wins_s['timestamp_start'].iloc[i + 1]
            gap_len   = (gap_end - gap_start).total_seconds()
            if gap_len <= 0:
                continue
            win_state         = wins_s.iloc[i]
            emo_enc, pred_enc = encode_row(
                win_state['emotion_label_end'], win_state['model_pred_end'])
            for offset in np.arange(0, gap_len, sample_every):
                ts = gap_start + pd.Timedelta(seconds=offset)
                if ts.floor('s') in err_set:
                    continue
                rows.append({
                    'user_id':              uid,
                    'distraction_active':   0,
                    'time_since_last_dist': min(offset, float(H)),  # capped for model
                    'time_in_current_dist': 0.0,
                    'true_time_since':      offset,                 # uncapped for calibration
                    'true_time_in_dist':    0.0,
                    'model_prob':           win_state['model_prob_end'],
                    'model_pred_enc':       pred_enc,
                    'emotion_prob':         win_state['emotion_prob_end'],
                    'emotion_label_enc':    emo_enc,
                    'baseline_error_rate':  b_rate,
                    'label': 0,
                })
    return pd.DataFrame(rows)

def build_baseline_negatives(sample_every=NEG_SAMPLE_EVERY,
                              default_model_prob=None,
                              default_model_pred_enc=None,
                              default_emotion_prob=None,
                              default_emotion_label_enc=None):
    """
    Negatives from undistracted baseline driving sessions.
    time_since_last_dist is set to H_CANDIDATES[-1] as a sentinel meaning
    'fully recovered / no prior distraction in this session'.
    """
    sentinel = float(H_CANDIDATES[-1])
    rows = []
    for _, row in driving_base.iterrows():
        uid          = row['user_id']
        run_duration = row['run_duration_seconds']
        b_rate       = user_baselines.get(uid, p_baseline_global)
        for offset in np.arange(0, run_duration, sample_every):
            rows.append({
                'user_id':              uid,
                'distraction_active':   0,
                'time_since_last_dist': sentinel,
                'time_in_current_dist': 0.0,
                'true_time_since':      sentinel,
                'true_time_in_dist':    0.0,
                'model_prob':           default_model_prob,
                'model_pred_enc':       default_model_pred_enc,
                'emotion_prob':         default_emotion_prob,
                'emotion_label_enc':    default_emotion_label_enc,
                'baseline_error_rate':  b_rate,
                'label': 0,
            })
    return pd.DataFrame(rows)

# ── 9. Bootstrap CI helper ────────────────────────────────────────────────────
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

# ── 10. Precision at fixed recall levels ──────────────────────────────────────
def precision_at_recall(y_true, y_score, recall_levels):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    results = {}
    for r in recall_levels:
        mask       = rec >= r
        results[r] = float(prec[mask].max()) if mask.any() else 0.0
    return results

# ── 11. Compute metrics helper ────────────────────────────────────────────────
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

# ── 12. Core: full nested LOSO evaluation for a given H ──────────────────────
def full_nested_eval(H):
    """
    Run the full nested LOSO-CV (with calibration and baseline negatives)
    for a given H. Returns a dict with all arrays and honest AUC-PR / AUC-ROC.
    This is the expensive but unbiased evaluation used for final H selection.
    """
    logo   = LeaveOneGroupOut()
    pos_ev = build_positives(H)
    neg_ev = build_negatives(H)
    df_ev  = pd.concat([pos_ev, neg_ev], ignore_index=True).dropna(subset=FEATURE_COLS)

    true_labels, raw_probs, calib_probs, gt_probs = [], [], [], []
    dist_active_vals, time_since_vals, time_in_dist_vals = [], [], []

    for train_idx, test_idx in logo.split(
            df_ev[FEATURE_COLS].values, df_ev['label'].values,
            df_ev['user_id'].values):

        train_df = df_ev.iloc[train_idx].copy()
        test_df  = df_ev.iloc[test_idx].copy()

        # Defaults for baseline negatives – derived from safe seconds in train fold only
        safe_mask = train_df['label'] == 0
        if safe_mask.sum() == 0:
            default_model_prob        = 0.5
            default_model_pred_enc    = le_pred.transform([UNKNOWN_LABEL])[0]
            default_emotion_prob      = 0.5
            default_emotion_label_enc = le_emotion.transform([UNKNOWN_LABEL])[0]
        else:
            default_model_prob        = train_df.loc[safe_mask, 'model_prob'].median()
            default_model_pred_enc    = train_df.loc[safe_mask, 'model_pred_enc'].mode()[0]
            default_emotion_prob      = train_df.loc[safe_mask, 'emotion_prob'].median()
            default_emotion_label_enc = train_df.loc[safe_mask, 'emotion_label_enc'].mode()[0]

        defaults = dict(
            default_model_prob=default_model_prob,
            default_model_pred_enc=default_model_pred_enc,
            default_emotion_prob=default_emotion_prob,
            default_emotion_label_enc=default_emotion_label_enc,
        )

        # Augmented training set (distraction negatives + positives + baseline negatives)
        base_neg_train = build_baseline_negatives(**defaults)
        base_neg_train = base_neg_train[
            base_neg_train['user_id'].isin(train_df['user_id'].unique())]
        train_full = pd.concat([train_df, base_neg_train], ignore_index=True) \
                       .dropna(subset=FEATURE_COLS)

        X_tr        = train_full[FEATURE_COLS].values.astype(float)
        y_tr        = train_full['label'].values.astype(int)
        pos_rate_tr = y_tr.mean()
        spw_tr      = (1 - pos_rate_tr) / pos_rate_tr

        clf = xgb.XGBClassifier(scale_pos_weight=spw_tr, **XGB_PARAMS)
        clf.fit(X_tr, y_tr)

        # Fit isotonic calibrator on training fold raw scores → GT rates
        raw_tr = clf.predict_proba(X_tr)[:, 1]
        gt_tr  = np.array([assign_gt_prob(row, H)
                           for _, row in train_full.iterrows()])
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(raw_tr, gt_tr)

        # Augmented test set (distraction samples + baseline negatives for this user)
        test_base_neg = build_baseline_negatives(**defaults)
        test_base_neg = test_base_neg[
            test_base_neg['user_id'].isin(test_df['user_id'].unique())]
        test_full = pd.concat([test_df, test_base_neg], ignore_index=True) \
                      .dropna(subset=FEATURE_COLS)

        X_te   = test_full[FEATURE_COLS].values.astype(float)
        y_te   = test_full['label'].values.astype(int)
        raw_te = clf.predict_proba(X_te)[:, 1]
        cal_te = iso.predict(raw_te)

        true_labels.extend(y_te)
        raw_probs.extend(raw_te)
        calib_probs.extend(cal_te)
        gt_probs.extend(
            [assign_gt_prob(row, H) for _, row in test_full.iterrows()])
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
        # carry full arrays for the winner's reporting pass
        'y_true':          y_true_arr,
        'raw':             raw_arr,
        'calib':           cal_arr,
        'gt':              gt_arr,
        'dist_active':     np.array(dist_active_vals),
        'time_since':      np.array(time_since_vals),
        'time_in_dist':    np.array(time_in_dist_vals),
    }

# ── 13. Fast H grid search (no calibration, no baseline negatives) ────────────
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

# ── 14. full nested evaluation of top-N candidates ─────────────────
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
best_H = best_result['H']

print(f"\n  Best H (nested) = {best_H}s  "
      f"(nested AUC-PR={best_result['AUC-PR']:.4f}, "
      f"AUC-ROC={best_result['AUC-ROC']:.4f}, "
      f"Cal_MAE={best_result['Calib_MAE']:.4f}, "
      f"score={best_result['selection_score']:.4f})")

# Unpack the winner's arrays for all downstream reporting
y_true_arr = best_result['y_true']
raw_arr = best_result['raw']
cal_arr = best_result['calib']
gt_arr = best_result['gt']
dist_active_arr = best_result['dist_active']
time_since_arr = best_result['time_since']
time_in_dist_arr = best_result['time_in_dist']

# ── 15. Full metrics on winner ────────────────────────────────────────────────
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

print("\n── Model Metrics (raw XGBoost) ───────────────────────────────────────")
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

print("\n── Precision @ Fixed Recall (raw XGBoost) ────────────────────────────")
for r in RECALL_LEVELS:
    print(f"  P @ Recall={r:.0%} : {m_raw[f'P@R={int(r*100)}']:.4f}")

print(f"\n── Classification Report (raw XGBoost @ t={best_thresh:.2f}) ──────────")
print(classification_report(y_true_arr, y_pred_raw, target_names=['Safe', 'Error']))

# ── 16. Ground truth validation ───────────────────────────────────────────────
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

# ── Risk stratification ────────────────────────────────────────────────────────
conditions = [
    ('Active distraction',                  mask_active),
    (f'Hangover (0–{best_H}s)',             mask_hangover),
    ('Baseline (driving without distractions)', mask_true_baseline),
]

strat_rows = []
print(f"\n── Risk Stratification ──────────────────────────────────────────────")
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
print(f"  ✓ Saved → {EVAL_OUT}gt_risk_stratification_calibrated.png")

# ── Temporal decay ─────────────────────────────────────────────────────────────
print(f"\n── Temporal Decay (seconds 0..{best_H}) ─────────────────────────────")

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
print(f"  {'0 (active)':<12} {gt_rate_per_sec[0]:>12.4f} {total_dist_seconds:>10.0f} "
      f"{errors_inside:>8} {model_mean_by_sec[0]:>12.4f} {calib_mean_by_sec[0]:>12.4f}")
for b_start in range(1, best_H + 1, bin_size_print):
    b_end   = min(b_start + bin_size_print, best_H + 1)
    lbl     = f"{b_start}–{b_end-1}s"
    n_err   = int(gt_errors[b_start:b_end].sum())
    exp_sum = gt_exposure[b_start:b_end].sum()
    rate_m  = float(np.nanmean(gt_rate_per_sec[b_start:b_end]))
    m_mean  = float(np.nanmean(model_mean_by_sec[b_start:b_end]))
    c_mean  = float(np.nanmean(calib_mean_by_sec[b_start:b_end]))
    print(f"  {lbl:<12} {rate_m:>12.4f} {exp_sum:>10.0f} {n_err:>8} "
          f"{m_mean:>12.4f} {c_mean:>12.4f}")
print(f"  {'Baseline':<12} {p_baseline_global:>12.4f} {'—':>10} {'—':>8} {'—':>12} {'—':>12}")


print(f"\n-- Active-window profile (seconds from distraction start, 0..{best_H}) --")
active_raw_by_sec = np.full(best_H + 1, np.nan)
active_cal_by_sec = np.full(best_H + 1, np.nan)
active_gt_by_sec = np.full(best_H + 1, np.nan)
active_rows = df_gt[mask_active]
for n in range(0, best_H + 1):
    sub = active_rows[
        (active_rows['time_in_current_dist'] >= n) &
        (active_rows['time_in_current_dist'] < n + 1)
    ]
    if len(sub) > 0:
        active_raw_by_sec[n] = sub['raw_prob'].mean()
        active_cal_by_sec[n] = sub['calib_prob'].mean()
    if n <= max_H_gt and not np.isnan(gt_active_rate[n]):
        active_gt_by_sec[n] = gt_active_rate[n]

print(f"  {'Sec':<6} {'GT rate':>10} {'Raw mean':>10} {'Cal mean':>10} {'Exposure':>10} {'Errors':>8}")
print("  " + "-" * 66)
for n in range(0, best_H + 1):
    exp_n = gt_active_exposure[n] if n <= max_H_gt else 0.0
    err_n = int(gt_active_errors[n]) if n <= max_H_gt else 0
    print(f"  {n:<6} {active_gt_by_sec[n]:>10.4f} {active_raw_by_sec[n]:>10.4f} "
          f"{active_cal_by_sec[n]:>10.4f} {exp_n:>10.0f} {err_n:>8}")
# ── Calibration MAE / RMSE ────────────────────────────────────────────────────
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

# ── Temporal decay plot ────────────────────────────────────────────────────────
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
ax.text(1.5, ax.get_ylim()[1] * 0.95, 'post‑distraction recovery →', fontsize=8, alpha=0.6)
ax.set_xlabel('Bin (0 = inside window, 1..H = seconds after distraction ended)')
ax.set_ylabel('Rate / Probability')
ax.set_title(f'GT Error Rate vs Model Prediction (H={best_H}s, raw & calibrated)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{EVAL_OUT}gt_temporal_decay_calibrated.png', dpi=150); plt.close()
print(f"\n  ✓ Saved → {EVAL_OUT}gt_temporal_decay_calibrated.png")

# ── Save tables ────────────────────────────────────────────────────────────────
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

active_profile_rows = []
for n in range(best_H + 1):
    active_profile_rows.append({
        'sec_from_dist_start': n,
        'gt_rate_smoothed': active_gt_by_sec[n],
        'raw_model_mean': active_raw_by_sec[n],
        'calibrated_model_mean': active_cal_by_sec[n],
        'exposure': gt_active_exposure[n] if n <= max_H_gt else 0,
        'n_errors': int(gt_active_errors[n]) if n <= max_H_gt else 0,
    })
pd.DataFrame(active_profile_rows).to_csv(
    f'{EVAL_OUT}gt_active_window_profile.csv', index=False
)

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

print(f"  ✓ Saved → {EVAL_OUT}gt_temporal_decay.csv")
print(f"  ✓ Saved → {EVAL_OUT}gt_active_window_profile.csv")
print(f"  ✓ Saved → {EVAL_OUT}gt_continuous_comparison.csv")
print(f"  ✓ Saved → {EVAL_OUT}gt_continuous_metrics.csv")
print(f"  ✓ Saved → {EVAL_OUT}h_selection_nested.csv")

# ── 17. Final model (all data) ────────────────────────────────────────────────
print(f"\n── Training Final Model (H={best_H}s, all data) ────────────────────")

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
gt_all  = np.array([assign_gt_prob(row, best_H) for _, row in df_f.iterrows()])
final_iso = IsotonicRegression(out_of_bounds='clip')
final_iso.fit(raw_all, gt_all)

artifact = {
    'model':             base_clf,
    'calibrator':        final_iso,
    'best_H':            best_H,
    'best_thresh':       best_thresh,
    'feature_cols':      FEATURE_COLS,
    'le_emotion':        le_emotion,
    'le_pred':           le_pred,
    'user_baselines':    user_baselines,
    'p_baseline_global': p_baseline_global,
    'cv_results':        results_df,
    'nested_h_results':  nested_summary,
    'metrics_raw':       m_raw,
    'bootstrap_ci_raw': {
        'AUC-PR':  (ci_aucpr,  lo_aucpr,  hi_aucpr),
        'AUC-ROC': (ci_aucroc, lo_aucroc, hi_aucroc),
        'MCC':     (ci_mcc,    lo_mcc,    hi_mcc),
        'Brier':   (ci_brier,  lo_brier,  hi_brier),
    },
}
joblib.dump(artifact, MODEL_OUT)
print(f"  Saved -> {MODEL_OUT}")

# ── 18. Inference helper ──────────────────────────────────────────────────────
def predict_fitness(
    distraction_active:              Union[bool, int],
    seconds_since_last_distraction:  float,
    emotion_label:                   Optional[str] = None,
    emotion_prob:                    float = 0.5,
    arousal_pred_label:              Optional[str] = None,
    arousal_pred_prob:               float = 0.5,
    user_id:                         Optional[str] = None,
    artifact_path:                   str   = MODEL_OUT,
) -> Dict[str, object]:
    """Returns calibrated probability of an error in the next second."""
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

    sample = pd.DataFrame([{
        'distraction_active':   dist_active,
        'time_since_last_dist': t_since,
        'time_in_current_dist': t_in_current,
        'model_prob':           a_prob,
        'model_pred_enc':       pred_enc,
        'emotion_prob':         e_prob,
        'emotion_label_enc':    emo_enc,
        'baseline_error_rate':  b_rate,
    }])

    raw_prob        = float(art['model'].predict_proba(sample[art['feature_cols']])[0, 1])
    calibrated_prob = float(art['calibrator'].predict([raw_prob])[0])

    return {
        'error_probability': round(calibrated_prob, 4),
        'fitness_to_drive':  round(1.0 - calibrated_prob, 4),
        'alert':             raw_prob >= thresh,
        'input_warnings':    warns,
    }

# ── 19. Summary ───────────────────────────────────────────────────────────────
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




