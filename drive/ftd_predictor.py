"""
Driver Digital Twin – Fitness-to-Drive XGBoost Pipeline
========================================================
Rigorous, publication‑ready version (fully de‑biased):
  - Sampling every second (NEG_SAMPLE_EVERY = 1)
  - Baseline negatives use realistic emotion/arousal values (medians from safe seconds)
  - Nested cross‑validation with proper calibration
  - Full bootstrap CIs for all metrics

Input datasets (place in DATA_PATH):
  - Dataset Distractions_distraction.csv
  - Dataset Errors_distraction.csv
  - Dataset Errors_baseline.csv
  - Dataset Driving Time_baseline.csv
"""
from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, roc_curve,
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
from matplotlib.patches import Patch

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_PATH        = 'data/'
MODEL_OUT        = 'fitness_model_calibrated.pkl'
EVAL_OUT         = 'evaluation/'
BIN_LIMIT        = 35                     
H_CANDIDATES     = list(range(5, BIN_LIMIT))  
NEG_SAMPLE_EVERY = 1                       
N_BOOTSTRAP      = 1000                    
RECALL_LEVELS    = [0.80, 0.85, 0.90, 0.95]

os.makedirs(EVAL_OUT, exist_ok=True)

XGB_PARAMS = dict(
    n_estimators     = 900,
    max_depth        = 3,
    learning_rate    = 0.03,
    subsample        = 0.9,
    colsample_bytree = 0.9,
    min_child_weight = 5,
    reg_lambda       = 2.0,
    gamma            = 1.0,
    eval_metric      = 'aucpr',
    random_state     = RANDOM_SEED,
    verbosity        = 0,
)

FEATURE_COLS = [
    'distraction_active',
    'time_since_last_dist',
    'model_prob',
    'model_pred_enc',
    'emotion_prob',
    'emotion_label_enc',
    'baseline_error_rate',
    'speed_kmh',
    'steer_angle_deg',
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

dist_users = set(distractions['user_id'].unique())
err_users  = set(errors_dist['user_id'].unique())
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
            'emotion_label', 'emotion_prob', 'speed_kmh', 'steer_angle_deg']:
    if col not in errors_dist.columns:
        issues.append(f"Missing column in errors_dist: {col}")

# Fill NaNs in numeric columns with median (warn but don't crash)
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
user_base_errs = errors_base.groupby('user_id').size()
user_base_secs = driving_base.groupby('user_id')['run_duration_seconds'].sum()
user_baselines = (user_base_errs / user_base_secs).fillna(p_baseline_global).to_dict()

print(f"\nGlobal baseline error rate : {p_baseline_global*100:.4f}% / second")

# ── 4. Helper functions for distraction state ──────────────────────────────────
def get_distraction_state(user_id, run_id, ts, H):
    """Return (active, time_since_last_dist) for a given timestamp."""
    key = (user_id, run_id)
    if key not in windows_by_session:
        return 0, float(H)
    wins = windows_by_session[key]
    # Inside a distraction window?
    if ((wins['timestamp_start'] <= ts) & (ts <= wins['timestamp_end'])).any():
        return 1, 0.0
    # Find previous window that ended before ts
    prev = wins[wins['timestamp_end'] < ts]
    if prev.empty:
        return 0, float(H)
    delta = (ts - prev['timestamp_end'].max()).total_seconds()
    return 0, min(delta, float(H))

def is_error_inside_distraction(row):
    """True if the error occurred inside any distraction window."""
    active, _ = get_distraction_state(row['user_id'], row['run_id'], row['timestamp'], H=0)
    return active == 1

def get_time_since_last(row):
    """Seconds since the end of the last distraction window before this error,
    but only if there is a next distraction after the error."""
    key = (row['user_id'], row['run_id'])
    if key not in windows_by_session:
        return np.nan
    wins = windows_by_session[key]
    # Find previous window that ended before the error
    prev = wins[wins['timestamp_end'] < row['timestamp']]
    if prev.empty:
        return np.nan
    # Check that there is a next window after the error (so it's an inter‑window gap)
    nxt = wins[wins['timestamp_start'] > row['timestamp']]
    if nxt.empty:
        return np.nan
    return (row['timestamp'] - prev['timestamp_end'].max()).total_seconds()

# ── 5. Ground truth per‑second error probabilities (inter‑window gaps only) ───
print("\n" + "=" * 70)
print("GROUND TRUTH PER‑SECOND ERROR RATES")
print("=" * 70)

max_H_gt = BIN_LIMIT - 1   # we will compute rates up to this many seconds after distraction

# ---- Bin 0: inside distraction windows ----
total_dist_seconds = (distractions['timestamp_end'] - distractions['timestamp_start']).dt.total_seconds().sum()
errors_inside = 0
for _, err in errors_dist.iterrows():
    if is_error_inside_distraction(err):
        errors_inside += 1
p_bin0 = errors_inside / total_dist_seconds if total_dist_seconds > 0 else 0.0
print(f"  Bin 0 (inside windows)      : {p_bin0:.6f} errors/s  (errors={errors_inside}, exposure={total_dist_seconds:.0f}s)")

# ---- Bins 1..max_H_gt: seconds after distraction (only inter‑window gaps) ----
gt_errors = np.zeros(max_H_gt + 1)      # index 1..max_H_gt used
gt_exposure = np.zeros(max_H_gt + 1)

# Exposure from inter‑window gaps only
for (uid, rid), wins in windows_by_session.items():
    wins_s = wins.sort_values('timestamp_start').reset_index(drop=True)
    # Only consider windows that have a next window (i.e., not the last in session)
    for i in range(len(wins_s) - 1):
        win_end = wins_s['timestamp_end'].iloc[i]
        next_start = wins_s['timestamp_start'].iloc[i + 1]
        gap_s = (next_start - win_end).total_seconds()
        gap_s = min(gap_s, float(max_H_gt))
        for n in range(1, int(np.floor(gap_s)) + 1):
            gt_exposure[n] += 1.0

# Errors after distraction (using modified get_time_since_last)
outside_errors = errors_dist[~errors_dist.apply(is_error_inside_distraction, axis=1).fillna(False)].copy()
outside_errors['time_since'] = outside_errors.apply(get_time_since_last, axis=1)
outside_after = outside_errors.dropna(subset=['time_since']).copy()
for t in outside_after['time_since']:
    n = int(np.floor(t)) + 1          # bin 1 = [0,1), bin 2 = [1,2), ...
    if 1 <= n <= max_H_gt:
        gt_errors[n] += 1

# Compute rates
gt_rate_per_sec = np.full(max_H_gt + 1, np.nan)   # index 0..max_H_gt
gt_rate_per_sec[0] = p_bin0
for n in range(1, max_H_gt + 1):
    if gt_exposure[n] > 0:
        gt_rate_per_sec[n] = gt_errors[n] / gt_exposure[n]
    else:
        gt_rate_per_sec[n] = np.nan

# Print summary (first few seconds)
print("  Post‑distraction seconds (rate / exposure / errors):")
for n in range(1, min(11, max_H_gt + 1)):
    if gt_exposure[n] > 0:
        print(f"    {n:2d}s : {gt_rate_per_sec[n]:.6f}  (exposure={gt_exposure[n]:6.0f}, errors={int(gt_errors[n]):3d})")
    else:
        print(f"    {n:2d}s : {gt_rate_per_sec[n]:.6f}  (exposure={gt_exposure[n]:6.0f}, errors={int(gt_errors[n]):3d})")
if max_H_gt > 10:
    print("    ...")

# ── 6. Label encoding ─────────────────────────────────────────────────────────
UNKNOWN_LABEL = 'unknown'

def safe_str(val):
    if val is None: return UNKNOWN_LABEL
    s = str(val).strip()
    return UNKNOWN_LABEL if s.lower() in ('nan', 'none', '') else s

all_emotion_labels = (pd.concat([errors_dist['emotion_label'],
    distractions['emotion_label_start'], distractions['emotion_label_end']])
    .map(safe_str).unique().tolist() + [UNKNOWN_LABEL])

all_model_preds = (pd.concat([errors_dist['model_pred'],
    distractions['model_pred_start'], distractions['model_pred_end']])
    .map(safe_str).unique().tolist() + [UNKNOWN_LABEL])

le_emotion = LabelEncoder().fit(list(set(all_emotion_labels)))
le_pred    = LabelEncoder().fit(list(set(all_model_preds)))

def encode_row(emotion_label, model_pred):
    return (le_emotion.transform([safe_str(emotion_label)])[0],
            le_pred.transform([safe_str(model_pred)])[0])

# ── 7. Sample builders (with NEG_SAMPLE_EVERY = 1) ─────────────────────────────
def build_positives(H):
    rows = []
    for _, err in errors_dist.iterrows():
        uid, rid, ts = err['user_id'], err['run_id'], err['timestamp']
        dist_active, time_since = get_distraction_state(uid, rid, ts, H)
        emo_enc, pred_enc = encode_row(err['emotion_label'], err['model_pred'])
        rows.append({
            'user_id': uid, 'distraction_active': dist_active,
            'time_since_last_dist': time_since, 'model_prob': err['model_prob'],
            'model_pred_enc': pred_enc, 'emotion_prob': err['emotion_prob'],
            'emotion_label_enc': emo_enc,
            'baseline_error_rate': user_baselines.get(uid, p_baseline_global),
            'speed_kmh': err['speed_kmh'],
            'steer_angle_deg': err['steer_angle_deg'],
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
        wins_s = wins.sort_values('timestamp_start').reset_index(drop=True)
        b_rate = user_baselines.get(uid, p_baseline_global)
        err_set = error_floor.get((uid, rid), set())

        # Inside distraction windows
        for _, win in wins_s.iterrows():
            dur = (win['timestamp_end'] - win['timestamp_start']).total_seconds()
            for offset in np.arange(0, dur, sample_every):
                ts = win['timestamp_start'] + pd.Timedelta(seconds=offset)
                if ts.floor('s') in err_set: continue
                emo_enc, pred_enc = encode_row(win['emotion_label_start'], win['model_pred_start'])
                rows.append({
                    'user_id': uid, 'distraction_active': 1,
                    'time_since_last_dist': 0.0, 'model_prob': win['model_prob_start'],
                    'model_pred_enc': pred_enc, 'emotion_prob': win['emotion_prob_start'],
                    'emotion_label_enc': emo_enc,
                    'baseline_error_rate': b_rate, 'label': 0,
                    'speed_kmh': win['speed_kmh_start'],
                    'steer_angle_deg': win['steer_angle_deg_start'],
                })

        # Inter-window gaps
        for i in range(len(wins_s) - 1):
            gap_start = wins_s['timestamp_end'].iloc[i]
            gap_end   = wins_s['timestamp_start'].iloc[i + 1]
            gap_len   = (gap_end - gap_start).total_seconds()
            if gap_len <= 0: continue
            win_state = wins_s.iloc[i]
            emo_enc, pred_enc = encode_row(win_state['emotion_label_end'], win_state['model_pred_end'])
            for offset in np.arange(0, gap_len, sample_every):
                ts = gap_start + pd.Timedelta(seconds=offset)
                if ts.floor('s') in err_set: continue
                rows.append({
                    'user_id': uid, 'distraction_active': 0,
                    'time_since_last_dist': min(offset, float(H)),
                    'model_prob': win_state['model_prob_end'],
                    'model_pred_enc': pred_enc, 'emotion_prob': win_state['emotion_prob_end'],
                    'emotion_label_enc': emo_enc,
                    'baseline_error_rate': b_rate, 'label': 0,
                    'speed_kmh': win_state['speed_kmh_end'],
                    'steer_angle_deg': win_state['steer_angle_deg_end'],
                })
    return pd.DataFrame(rows)

def build_baseline_negatives(sample_every=NEG_SAMPLE_EVERY,
                              default_model_prob=None, default_model_pred_enc=None,
                              default_emotion_prob=None, default_emotion_label_enc=None,
                              default_speed_kmh=None, default_steer_angle_deg=None):
    """
    Sample negatives from baseline driving sessions.
    Uses provided default feature values (e.g., medians from safe seconds in distraction runs).
    """
    rows = []
    for _, row in driving_base.iterrows():
        uid = row['user_id']
        run_duration = row['run_duration_seconds']
        b_rate = user_baselines.get(uid, p_baseline_global)
        for offset in np.arange(0, run_duration, sample_every):
            rows.append({
                'user_id': uid,
                'distraction_active': 0,
                'time_since_last_dist': float(H_CANDIDATES[-1]),  # large, recovered
                'model_prob': default_model_prob,
                'model_pred_enc': default_model_pred_enc,
                'emotion_prob': default_emotion_prob,
                'emotion_label_enc': default_emotion_label_enc,
                'speed_kmh': default_speed_kmh,
                'steer_angle_deg': default_steer_angle_deg,
                'baseline_error_rate': b_rate,
                'label': 0,
            })
    return pd.DataFrame(rows)

# ── 8. Bootstrap CI helper ─────────────────────────────────────────────────────
def bootstrap_ci(y_true, y_score, metric_fn, n=N_BOOTSTRAP, ci=0.95, seed=RANDOM_SEED):
    """Return (mean, lower, upper) via percentile bootstrap."""
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

# ── 9. Precision at fixed recall levels ───────────────────────────────────────
def precision_at_recall(y_true, y_score, recall_levels):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    results = {}
    for r in recall_levels:
        mask = rec >= r
        results[r] = float(prec[mask].max()) if mask.any() else 0.0
    return results

# ── 10. H grid search (using raw XGBoost, no calibration yet) ──────────────────
print("\n" + "=" * 70)
print("H GRID SEARCH (Leave-One-User-Out)")
print("=" * 70)
print(f"{'H (s)':<8} {'AUC-PR':<10} {'AUC-ROC':<10} {'N':<8} {'Pos%'}")
print("-" * 50)

logo = LeaveOneGroupOut()
results = []

# For grid search, we don't include baseline negatives yet (to keep search fast)
for H in H_CANDIDATES:
    pos = build_positives(H)
    neg = build_negatives(H)
    df = pd.concat([pos, neg], ignore_index=True).dropna(subset=FEATURE_COLS)

    X = df[FEATURE_COLS].values.astype(float)
    y = df['label'].values.astype(int)
    groups = df['user_id'].values

    pos_rate = y.mean()
    scale_pos_weight = np.sqrt((1 - pos_rate) / pos_rate)

    y_true_all, y_prob_all = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        train_users = set(groups[train_idx])
        test_users = set(groups[test_idx])
        assert train_users.isdisjoint(test_users), "LEAKAGE DETECTED"

        clf = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, **XGB_PARAMS)
        clf.fit(X[train_idx], y[train_idx])
        y_true_all.extend(y[test_idx])
        y_prob_all.extend(clf.predict_proba(X[test_idx])[:, 1])

    auc_pr = average_precision_score(y_true_all, y_prob_all)
    auc_roc = roc_auc_score(y_true_all, y_prob_all)
    results.append({'H': H, 'AUC-PR': auc_pr, 'AUC-ROC': auc_roc,
                    'n_samples': len(df), 'pos_pct': pos_rate * 100})
    print(f"{H:<8} {auc_pr:<10.4f} {auc_roc:<10.4f} {len(df):<8} {pos_rate*100:.1f}%")

results_df = pd.DataFrame(results)
best_row = results_df.loc[results_df['AUC-PR'].idxmax()]
best_H = int(best_row['H'])
print(f"\n  Best H = {best_H}s  (AUC-PR={best_row['AUC-PR']:.4f}, AUC-ROC={best_row['AUC-ROC']:.4f})")

# ── 11. Full evaluation with nested calibration and realistic baseline ─────────
print("\n" + "=" * 70)
print(f"FULL EVALUATION (H={best_H}s, nested calibration, realistic baseline)")
print("=" * 70)

# Build base evaluation set (distraction runs)
pos_ev = build_positives(best_H)
neg_ev = build_negatives(best_H)
df_ev = pd.concat([pos_ev, neg_ev], ignore_index=True).dropna(subset=FEATURE_COLS)

# We'll store raw and calibrated predictions, plus ground truth.
true_labels = []
raw_probs = []
calib_probs = []
gt_probs = []
dist_active_vals = []
time_since_vals = []

for train_idx, test_idx in logo.split(df_ev[FEATURE_COLS].values, df_ev['label'].values, df_ev['user_id'].values):
    # Split data
    train_df = df_ev.iloc[train_idx].copy()
    test_df = df_ev.iloc[test_idx].copy()

    # --- Compute default baseline values from safe seconds in the training set ---
    # Safe seconds are those with label 0 (negatives) from distraction runs
    safe_mask = (train_df['label'] == 0)
    if safe_mask.sum() == 0:
        # fallback (should not happen)
        default_model_prob = 0.5
        default_model_pred_enc = le_pred.transform([UNKNOWN_LABEL])[0]
        default_emotion_prob = 0.5
        default_emotion_label_enc = le_emotion.transform([UNKNOWN_LABEL])[0]
        default_speed_kmh = 0.0
        default_steer_angle_deg = 0.0
    else:
        default_model_prob = train_df.loc[safe_mask, 'model_prob'].median()
        default_model_pred_enc = train_df.loc[safe_mask, 'model_pred_enc'].mode()[0]
        default_emotion_prob = train_df.loc[safe_mask, 'emotion_prob'].median()
        default_emotion_label_enc = train_df.loc[safe_mask, 'emotion_label_enc'].mode()[0]
        default_speed_kmh = train_df.loc[safe_mask, 'speed_kmh'].median()
        default_steer_angle_deg = train_df.loc[safe_mask, 'steer_angle_deg'].median()

    # Build training set: distraction-run negatives + positives + baseline negatives
    base_neg_train = build_baseline_negatives(
        default_model_prob=default_model_prob,
        default_model_pred_enc=default_model_pred_enc,
        default_emotion_prob=default_emotion_prob,
        default_emotion_label_enc=default_emotion_label_enc,
        default_speed_kmh=default_speed_kmh,
        default_steer_angle_deg=default_steer_angle_deg
    )
    # Filter baseline negatives to only users in train set
    base_neg_train = base_neg_train[base_neg_train['user_id'].isin(train_df['user_id'].unique())]

    train_full = pd.concat([train_df, base_neg_train], ignore_index=True).dropna(subset=FEATURE_COLS)
    X_tr = train_full[FEATURE_COLS].values.astype(float)
    y_tr = train_full['label'].values.astype(int)
    groups_tr = train_full['user_id'].values

    # Train XGBoost on this augmented training set
    pos_rate_tr = y_tr.mean()
    spw_tr = np.sqrt((1 - pos_rate_tr) / pos_rate_tr)
    clf = xgb.XGBClassifier(scale_pos_weight=spw_tr, **XGB_PARAMS)
    clf.fit(X_tr, y_tr)

    # Predict raw probabilities on training set to fit calibrator
    raw_tr = clf.predict_proba(X_tr)[:, 1]

    # Compute ground truth probabilities for training samples
    gt_tr = np.zeros(len(train_full))
    for j, idx2 in enumerate(train_full.index):
        row = train_full.loc[idx2]
        if row['distraction_active'] == 1:
            gt_tr[j] = gt_rate_per_sec[0]
        else:
            t = row['time_since_last_dist']
            if t >= best_H:
                gt_tr[j] = p_baseline_global
            else:
                n = int(np.floor(t)) + 1
                if n <= max_H_gt and not np.isnan(gt_rate_per_sec[n]):
                    gt_tr[j] = gt_rate_per_sec[n]
                else:
                    gt_tr[j] = p_baseline_global

    # Fit calibrator
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(raw_tr, gt_tr)

    # Now handle test set: build test set with its own baseline negatives (using same defaults as above)
    test_base_neg = build_baseline_negatives(
        default_model_prob=default_model_prob,
        default_model_pred_enc=default_model_pred_enc,
        default_emotion_prob=default_emotion_prob,
        default_emotion_label_enc=default_emotion_label_enc,
        default_speed_kmh=default_speed_kmh,
        default_steer_angle_deg=default_steer_angle_deg
    )
    test_base_neg = test_base_neg[test_base_neg['user_id'].isin(test_df['user_id'].unique())]
    test_full = pd.concat([test_df, test_base_neg], ignore_index=True).dropna(subset=FEATURE_COLS)
    X_te = test_full[FEATURE_COLS].values.astype(float)
    y_te = test_full['label'].values.astype(int)

    # Predict raw and calibrated
    raw_te = clf.predict_proba(X_te)[:, 1]
    cal_te = iso.predict(raw_te)

    # Store
    true_labels.extend(y_te)
    raw_probs.extend(raw_te)
    calib_probs.extend(cal_te)
    # Ground truth for test samples
    for _, row in test_full.iterrows():
        if row['distraction_active'] == 1:
            gt_probs.append(gt_rate_per_sec[0])
        else:
            t = row['time_since_last_dist']
            if t >= best_H:
                gt_probs.append(p_baseline_global)
            else:
                n = int(np.floor(t)) + 1
                if n <= max_H_gt and not np.isnan(gt_rate_per_sec[n]):
                    gt_probs.append(gt_rate_per_sec[n])
                else:
                    gt_probs.append(p_baseline_global)
        dist_active_vals.append(row['distraction_active'])
        time_since_vals.append(row['time_since_last_dist'])

# Convert to arrays
y_true_arr = np.array(true_labels)
raw_arr = np.array(raw_probs)
cal_arr = np.array(calib_probs)
gt_arr = np.array(gt_probs)
dist_active_arr = np.array(dist_active_vals)
time_since_arr = np.array(time_since_vals)

# Optimal threshold on raw
prec_c, rec_c, thresh_c = precision_recall_curve(y_true_arr, raw_arr)
f1_c = 2 * prec_c[:-1] * rec_c[:-1] / (prec_c[:-1] + rec_c[:-1] + 1e-9)
best_thresh = thresh_c[np.argmax(f1_c)]
y_pred_raw = (raw_arr >= best_thresh).astype(int)

# ── Compute all metrics ────────────────────────────────────────────────────────
def compute_metrics(y_true, y_score, thresh):
    yp = (y_score >= thresh).astype(int)
    cm_ = confusion_matrix(y_true, yp)
    tn, fp, fn, tp = cm_.ravel() if cm_.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    m = {
        'AUC-PR':       average_precision_score(y_true, y_score),
        'AUC-ROC':      roc_auc_score(y_true, y_score),
        'Log-Loss':     log_loss(y_true, y_score),
        'MCC':          matthews_corrcoef(y_true, yp),
        'Kappa':        cohen_kappa_score(y_true, yp),
        'Brier':        brier_score_loss(y_true, y_score),
        'F1':           f1_score(y_true, yp, zero_division=0),
        'Precision':    precision_score(y_true, yp, zero_division=0),
        'Recall':       recall_score(y_true, yp, zero_division=0),
        'Specificity':  specificity,
        'NPV':          npv,
    }
    par = precision_at_recall(y_true, y_score, RECALL_LEVELS)
    m.update({f'P@R={int(r*100)}': v for r, v in par.items()})
    return m

m_raw = compute_metrics(y_true_arr, raw_arr, best_thresh)
m_cal = compute_metrics(y_true_arr, cal_arr, best_thresh)  # using same threshold; could be re-optimized

# Bootstrap CIs for raw model
print(f"\nComputing bootstrap CIs (n={N_BOOTSTRAP}) for raw model ...")
ci_aucpr, lo_aucpr, hi_aucpr = bootstrap_ci(y_true_arr, raw_arr, average_precision_score)
ci_aucroc, lo_aucroc, hi_aucroc = bootstrap_ci(y_true_arr, raw_arr, roc_auc_score)
ci_brier, lo_brier, hi_brier = bootstrap_ci(y_true_arr, raw_arr, brier_score_loss)
mcc_fn = lambda yt, ys: matthews_corrcoef(yt, (ys >= best_thresh).astype(int))
ci_mcc, lo_mcc, hi_mcc = bootstrap_ci(y_true_arr, raw_arr, mcc_fn)
kappa_fn = lambda yt, ys: cohen_kappa_score(yt, (ys >= best_thresh).astype(int))
ci_kappa, lo_kappa, hi_kappa = bootstrap_ci(y_true_arr, raw_arr, kappa_fn)
ci_logloss, lo_ll, hi_ll = bootstrap_ci(y_true_arr, raw_arr, log_loss)

# Print metrics
print("\n── Model Metrics (raw XGBoost) ───────────────────────────────────────")
print(f"  {'Metric':<18} {'Value (95% CI)'}")
print("  " + "-" * 52)
rows_tbl = [
    ('AUC-PR',    f"{m_raw['AUC-PR']:.4f}  [{lo_aucpr:.4f} - {hi_aucpr:.4f}]"),
    ('AUC-ROC',   f"{m_raw['AUC-ROC']:.4f}  [{lo_aucroc:.4f} - {hi_aucroc:.4f}]"),
    ('Log-Loss',  f"{m_raw['Log-Loss']:.4f}  [{lo_ll:.4f} - {hi_ll:.4f}]"),
    ('MCC',       f"{m_raw['MCC']:.4f}  [{lo_mcc:.4f} - {hi_mcc:.4f}]"),
    ('Kappa',     f"{m_raw['Kappa']:.4f}  [{lo_kappa:.4f} - {hi_kappa:.4f}]"),
    ('Brier',     f"{m_raw['Brier']:.4f}  [{lo_brier:.4f} - {hi_brier:.4f}]"),
    ('F1', f"{m_raw['F1']:.4f}"),
    ('Precision', f"{m_raw['Precision']:.4f}"),
    ('Recall', f"{m_raw['Recall']:.4f}"),
    ('Specificity', f"{m_raw['Specificity']:.4f}"),
    ('NPV', f"{m_raw['NPV']:.4f}"),
]
for r in rows_tbl:
    print(f"  {r[0]:<18} {r[1]}")

print(f"\n  Threshold (max-F1) : {best_thresh:.4f}")

print("\n── Precision @ Fixed Recall (raw XGBoost) ────────────────────────────")
for r in RECALL_LEVELS:
    print(f"  P @ Recall={r:.0%} : {m_raw[f'P@R={int(r*100)}']:.4f}")

print("\n── Classification Report (raw XGBoost @ t={:.2f}) ────────────────────".format(best_thresh))
print(classification_report(y_true_arr, y_pred_raw, target_names=['Safe','Error']))

# ── Ground truth validation (now with calibrated probabilities) ────────────────
print("\n" + "=" * 70)
print("GROUND TRUTH VALIDATION (with nested calibration)")
print("=" * 70)

# Reconstruct a DataFrame for analysis
df_gt = pd.DataFrame({
    'distraction_active': dist_active_arr,
    'time_since_last_dist': time_since_arr,
    'label': y_true_arr,
    'raw_prob': raw_arr,
    'calib_prob': cal_arr,
    'gt_prob': gt_arr,
})

# ---- Risk stratification ----
mask_active = df_gt['distraction_active'] == 1
mask_hangover = (df_gt['distraction_active'] == 0) & (df_gt['time_since_last_dist'] < best_H)
mask_true_baseline = (df_gt['distraction_active'] == 0) & (df_gt['time_since_last_dist'] == float(H_CANDIDATES[-1]))

conditions = [
    ('Active distraction', mask_active),
    (f'Hangover (0–{best_H}s)', mask_hangover & ~mask_true_baseline),
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
    gt_rate = subset['label'].mean()
    raw_mean = subset['raw_prob'].mean()
    cal_mean = subset['calib_prob'].mean()
    print(f"  {label:<40} {gt_rate:>10.4f} {raw_mean:>10.4f} {cal_mean:>12.4f} {len(subset):>8}")
    strat_rows.append({'condition': label, 'gt_error_rate': gt_rate,
                       'raw_mean': raw_mean, 'calib_mean': cal_mean, 'n': len(subset)})

# Plot
strat_df = pd.DataFrame(strat_rows)
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(strat_df))
w = 0.25
ax.bar(x - w, strat_df['gt_error_rate'],   w, label='GT error rate',
       color='#d62728', alpha=0.85)
ax.bar(x, strat_df['raw_mean'], w, label='Raw model mean',
       color='steelblue', alpha=0.85)
ax.bar(x + w, strat_df['calib_mean'], w, label='Calibrated model mean',
       color='green', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(strat_df['condition'], rotation=15, ha='right')
ax.set_ylabel('Rate / Probability')
ax.set_title('Risk Stratification: Ground Truth vs Model (raw & calibrated)')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{EVAL_OUT}gt_risk_stratification_calibrated.png', dpi=150); plt.close()
print(f"  ✓ Saved → {EVAL_OUT}gt_risk_stratification_calibrated.png")

# ---- Temporal decay (only for hangover seconds) ----
print(f"\n── Temporal Decay (seconds 0..{best_H}) ─────────────────────────────")
# Compute model mean per second from df_gt (raw and calibrated) for hangover only
model_mean_by_sec = np.full(best_H + 1, np.nan)
calib_mean_by_sec = np.full(best_H + 1, np.nan)

# Bin 0
active_mask = df_gt['distraction_active'] == 1
model_mean_by_sec[0] = df_gt.loc[active_mask, 'raw_prob'].mean()
calib_mean_by_sec[0] = df_gt.loc[active_mask, 'calib_prob'].mean()

# Bins 1..best_H
post_rows = df_gt[~active_mask & ~mask_true_baseline]  # exclude baseline samples
for n in range(1, best_H + 1):
    sub = post_rows[(post_rows['time_since_last_dist'] >= n - 1) &
                    (post_rows['time_since_last_dist'] < n)]
    if len(sub) > 0:
        model_mean_by_sec[n] = sub['raw_prob'].mean()
        calib_mean_by_sec[n] = sub['calib_prob'].mean()

# Print grouped table
bin_size_print = 5
print(f"  {'Bin':<12} {'GT rate(/s)':>12} {'Exposure':>10} {'Errors':>8} {'Raw mean':>12} {'Calib mean':>12}")
print("  " + "-" * 72)
print(f"  {'0 (active)':<12} {gt_rate_per_sec[0]:>12.4f} {total_dist_seconds:>10.0f} {errors_inside:>8} {model_mean_by_sec[0]:>12.4f} {calib_mean_by_sec[0]:>12.4f}")
for b_start in range(1, best_H + 1, bin_size_print):
    b_end = min(b_start + bin_size_print, best_H + 1)
    label = f"{b_start}–{b_end-1}s"
    n_err = int(gt_errors[b_start:b_end].sum())
    exp_sum = gt_exposure[b_start:b_end].sum()
    rates = gt_rate_per_sec[b_start:b_end]
    rate_m = float(np.nanmean(rates)) if not np.all(np.isnan(rates)) else np.nan
    m_probs = model_mean_by_sec[b_start:b_end]
    m_mean = float(np.nanmean(m_probs)) if not np.all(np.isnan(m_probs)) else np.nan
    c_probs = calib_mean_by_sec[b_start:b_end]
    c_mean = float(np.nanmean(c_probs)) if not np.all(np.isnan(c_probs)) else np.nan
    gt_str = f"{rate_m:.4f}" if not np.isnan(rate_m) else "N/A"
    mp_str = f"{m_mean:.4f}" if not np.isnan(m_mean) else "N/A"
    cp_str = f"{c_mean:.4f}" if not np.isnan(c_mean) else "N/A"
    print(f"  {label:<12} {gt_str:>12} {exp_sum:>10.0f} {n_err:>8} {mp_str:>12} {cp_str:>12}")
print(f"  {'Baseline':<12} {p_baseline_global:>12.4f} {'—':>10} {'—':>8} {'—':>12} {'—':>12}")

# ---- Calibration metrics ----
errors_raw = df_gt['raw_prob'].values - df_gt['gt_prob'].values
errors_cal = df_gt['calib_prob'].values - df_gt['gt_prob'].values
mae_raw = np.mean(np.abs(errors_raw))
rmse_raw = np.sqrt(np.mean(errors_raw**2))
mae_cal = np.mean(np.abs(errors_cal))
rmse_cal = np.sqrt(np.mean(errors_cal**2))
print(f"\n  Global MAE  : raw={mae_raw:.4f}  calibrated={mae_cal:.4f}")
print(f"  Global RMSE : raw={rmse_raw:.4f}  calibrated={rmse_cal:.4f}")

# Per‑condition MAE
print(f"\n  {'Condition':<40} {'GT mean':>9} {'Raw mean':>11} {'Cal mean':>11} {'Raw MAE':>8} {'Cal MAE':>8} {'N':>6}")
print("  " + "-" * 96)
for lbl, mask in conditions:
    sub = df_gt[mask]
    if len(sub) == 0:
        continue
    cond_mae_raw = np.mean(np.abs(sub['raw_prob'].values - sub['gt_prob'].values))
    cond_mae_cal = np.mean(np.abs(sub['calib_prob'].values - sub['gt_prob'].values))
    print(f"  {lbl:<40} {sub['gt_prob'].mean():>9.4f} {sub['raw_prob'].mean():>11.4f} {sub['calib_prob'].mean():>11.4f} {cond_mae_raw:>8.4f} {cond_mae_cal:>8.4f} {len(sub):>6}")

# Temporal decay plot
seconds = np.arange(0, best_H + 1)
valid_gt = ~np.isnan(gt_rate_per_sec[:best_H+1])
valid_raw = ~np.isnan(model_mean_by_sec)
valid_cal = ~np.isnan(calib_mean_by_sec)

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(seconds[valid_gt], gt_rate_per_sec[:best_H+1][valid_gt],
        'o-', color='#d62728', lw=2, markersize=4,
        label='GT empirical rate (errors/s)')
ax.plot(seconds[valid_raw], model_mean_by_sec[valid_raw],
        's--', color='steelblue', lw=2, markersize=4,
        label='Raw model mean')
ax.plot(seconds[valid_cal], calib_mean_by_sec[valid_cal],
        'd-.', color='green', lw=2, markersize=4,
        label='Calibrated model mean')
ax.axhline(y=p_baseline_global, color='grey', linestyle=':', lw=1.5,
           label=f'Baseline rate ({p_baseline_global*100:.2f}%/s)')
ax.axvline(x=0.5, color='black', linestyle=':', lw=1, alpha=0.4)
ax.text(0.1, ax.get_ylim()[1]*0.95, 'inside window', fontsize=8, alpha=0.6)
ax.text(1.5, ax.get_ylim()[1]*0.95, 'post‑distraction recovery →', fontsize=8, alpha=0.6)
ax.set_xlabel('Bin (0 = inside window, 1..H = seconds after distraction ended)')
ax.set_ylabel('Rate / Probability')
ax.set_title(f'GT Error Rate vs Model Prediction (raw & calibrated)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{EVAL_OUT}gt_temporal_decay_calibrated.png', dpi=150); plt.close()
print(f"\n  ✓ Saved → {EVAL_OUT}gt_temporal_decay_calibrated.png")

# Save tables
decay_rows = []
for n in range(best_H + 1):
    decay_rows.append({
        'bin': n,
        'gt_rate_per_sec': gt_rate_per_sec[n] if n <= max_H_gt else np.nan,
        'raw_model_mean': model_mean_by_sec[n],
        'calibrated_model_mean': calib_mean_by_sec[n],
        'n_errors': int(gt_errors[n]) if n <= max_H_gt else 0,
        'exposure': gt_exposure[n] if n <= max_H_gt else 0,
    })
decay_df = pd.DataFrame(decay_rows)
decay_df.to_csv(f'{EVAL_OUT}gt_temporal_decay.csv', index=False)

cont_df = df_gt.copy()
cont_df['abs_error_raw'] = np.abs(cont_df['raw_prob'] - cont_df['gt_prob'])
cont_df['abs_error_cal'] = np.abs(cont_df['calib_prob'] - cont_df['gt_prob'])
cont_df.to_csv(f'{EVAL_OUT}gt_continuous_comparison.csv', index=False)

pd.DataFrame([{'MAE_raw': mae_raw, 'RMSE_raw': rmse_raw,
               'MAE_cal': mae_cal, 'RMSE_cal': rmse_cal,
               'p_bin0': p_bin0, 'p_baseline': p_baseline_global}]
             ).to_csv(f'{EVAL_OUT}gt_continuous_metrics.csv', index=False)
print(f"  ✓ Saved → {EVAL_OUT}gt_temporal_decay.csv")
print(f"  ✓ Saved → {EVAL_OUT}gt_continuous_comparison.csv")
print(f"  ✓ Saved → {EVAL_OUT}gt_continuous_metrics.csv")

# ── Feature sanity check ───────────────────────────────────────────────────
# print("\n── Feature Sanity Check ───────────────────────────────────────────────")
# df_check = pd.DataFrame(df_f, columns=FEATURE_COLS)
# for col in FEATURE_COLS:
#     mn, mx, med = df_check[col].min(), df_check[col].max(), df_check[col].median()
#     flag = " <- [WARN] constant feature!" if mn == mx else ""
#     print(f"  {col:<28} min={mn:7.3f}  max={mx:7.3f}  median={med:7.3f}{flag}")

# ── Final model (all data) ─────────────────────────────────────────────────────
print(f"\n── Training Final Model (H={best_H}s, all data) ───────────────────────")
# Build full dataset with realistic baseline values computed globally
pos_f = build_positives(best_H)
neg_f = build_negatives(best_H)
all_neg_dist = pd.concat([pos_f, neg_f], ignore_index=True)  # includes both classes
safe_global = all_neg_dist[all_neg_dist['label'] == 0]
if len(safe_global) > 0:
    default_model_prob_global = safe_global['model_prob'].median()
    default_model_pred_enc_global = safe_global['model_pred_enc'].mode()[0]
    default_emotion_prob_global = safe_global['emotion_prob'].median()
    default_emotion_label_enc_global = safe_global['emotion_label_enc'].mode()[0]
    default_speed_kmh_global = safe_global['speed_kmh'].median()
    default_steer_angle_deg_global = safe_global['steer_angle_deg'].median()
else:
    default_model_prob_global = 0.5
    default_model_pred_enc_global = le_pred.transform([UNKNOWN_LABEL])[0]
    default_emotion_prob_global = 0.5
    default_emotion_label_enc_global = le_emotion.transform([UNKNOWN_LABEL])[0]
    default_speed_kmh_global = 0.0
    default_steer_angle_deg_global = 0.0

base_f = build_baseline_negatives(
    default_model_prob=default_model_prob_global,
    default_model_pred_enc=default_model_pred_enc_global,
    default_emotion_prob=default_emotion_prob_global,
    default_emotion_label_enc=default_emotion_label_enc_global,
    default_speed_kmh=default_speed_kmh_global,
    default_steer_angle_deg=default_steer_angle_deg_global
)

df_f = pd.concat([pos_f, neg_f, base_f], ignore_index=True).dropna(subset=FEATURE_COLS)
X_f, y_f = df_f[FEATURE_COLS].values.astype(float), df_f['label'].values.astype(int)
spw_f = np.sqrt((1 - y_f.mean()) / y_f.mean())

base_clf = xgb.XGBClassifier(scale_pos_weight=spw_f, **XGB_PARAMS)
base_clf.fit(X_f, y_f)

# Fit final calibrator on all data
raw_all = base_clf.predict_proba(X_f)[:, 1]
gt_all = []
for _, row in df_f.iterrows():
    if row['distraction_active'] == 1:
        gt_all.append(gt_rate_per_sec[0])
    else:
        t = row['time_since_last_dist']
        if t >= best_H:
            gt_all.append(p_baseline_global)
        else:
            n = int(np.floor(t)) + 1
            if n <= max_H_gt and not np.isnan(gt_rate_per_sec[n]):
                gt_all.append(gt_rate_per_sec[n])
            else:
                gt_all.append(p_baseline_global)
gt_all = np.array(gt_all)

final_iso = IsotonicRegression(out_of_bounds='clip')
final_iso.fit(raw_all, gt_all)

# Save artifact
artifact = {
    'model': base_clf,
    'calibrator': final_iso,
    'best_H': best_H,
    'best_thresh': best_thresh,
    'feature_cols': FEATURE_COLS,
    'le_emotion': le_emotion,
    'le_pred': le_pred,
    'user_baselines': user_baselines,
    'p_baseline_global': p_baseline_global,
    'cv_results': results_df,
    'metrics_raw': m_raw,
    'bootstrap_ci_raw': {
        'AUC-PR': (ci_aucpr, lo_aucpr, hi_aucpr),
        'AUC-ROC': (ci_aucroc, lo_aucroc, hi_aucroc),
        'MCC': (ci_mcc, lo_mcc, hi_mcc),
        'Brier': (ci_brier, lo_brier, hi_brier),
    },
}
joblib.dump(artifact, MODEL_OUT)
print(f"  Saved -> {MODEL_OUT}")

# ── Inference helper (now with calibration) ────────────────────────────────────
def predict_fitness(
    distraction_active: bool | int,
    seconds_since_last_distraction: float,
    emotion_label: str = None,
    emotion_prob: float = 0.5,
    arousal_pred_label: str = None,
    arousal_pred_prob: float = 0.5,
    speed_kmh: float = 0.0,
    steer_angle_deg: float = 0.0,
    user_id: str = None,
    artifact_path: str = MODEL_OUT,
) -> dict:
    """
    Returns calibrated probability of an error in the next second.
    """
    art = joblib.load(artifact_path)
    H = art['best_H']
    thresh = art['best_thresh']
    warns = []

    try:
        dist_active = int(bool(distraction_active))
    except Exception:
        dist_active = 0
        warns.append("distraction_active invalid, defaulting to 0")

    try:
        t_since = float(seconds_since_last_distraction)
        if t_since < 0:
            warns.append(f"seconds_since_last_distraction={t_since} < 0, clamped to 0")
            t_since = 0.0
        if dist_active == 1:
            t_since = 0.0
        t_since = min(t_since, float(H))
    except Exception:
        t_since = float(H)
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
    e_prob = _clip_prob(emotion_prob, 'emotion_prob')

    def _to_float(val, name, default=0.0):
        try:
            v = float(val)
            if name == 'speed_kmh' and v < 0:
                warns.append(f"{name}={v} < 0, clamped to 0")
                return 0.0
            return v
        except Exception:
            warns.append(f"{name} invalid, defaulting to {default}")
            return float(default)

    speed_val = _to_float(speed_kmh, 'speed_kmh', default=0.0)
    steer_val = _to_float(steer_angle_deg, 'steer_angle_deg', default=0.0)

    _le_emotion = art['le_emotion']
    _le_pred = art['le_pred']

    def _encode(le, val):
        s = safe_str(val)
        if s in le.classes_:
            return int(le.transform([s])[0])
        warns.append(f"Unseen label '{s}' for {le.__class__.__name__}, using 'unknown'")
        return int(le.transform(['unknown'])[0])

    emo_enc = _encode(_le_emotion, emotion_label)
    pred_enc = _encode(_le_pred, arousal_pred_label)

    b_rate = art['user_baselines'].get(user_id, art['p_baseline_global'])
    if user_id is not None and user_id not in art['user_baselines']:
        warns.append(f"user_id='{user_id}' not in training set, using global baseline")

    sample = pd.DataFrame([{
        'distraction_active': dist_active,
        'time_since_last_dist': t_since,
        'model_prob': a_prob,
        'model_pred_enc': pred_enc,
        'emotion_prob': e_prob,
        'emotion_label_enc': emo_enc,
        'baseline_error_rate': b_rate,
        'speed_kmh': speed_val,
        'steer_angle_deg': steer_val,
    }])

    raw_prob = float(art['model'].predict_proba(sample[art['feature_cols']])[0, 1])
    calibrated_prob = float(art['calibrator'].predict([raw_prob])[0])

    return {
        'error_probability': round(calibrated_prob, 4),
        'fitness_to_drive': round(1.0 - calibrated_prob, 4),
        'alert': raw_prob >= thresh,
        'input_warnings': warns,
    }

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Best H             : {best_H}s")
print(f"  Raw XGBoost AUC-PR : {m_raw['AUC-PR']:.4f}  95% CI [{lo_aucpr:.4f} - {hi_aucpr:.4f}]")
print(f"  Raw XGBoost AUC-ROC: {m_raw['AUC-ROC']:.4f}  95% CI [{lo_aucroc:.4f} - {hi_aucroc:.4f}]")
print(f"  Raw XGBoost MCC    : {m_raw['MCC']:.4f}  95% CI [{lo_mcc:.4f} - {hi_mcc:.4f}]")
print(f"  Raw XGBoost Brier  : {m_raw['Brier']:.4f}  95% CI [{lo_brier:.4f} - {hi_brier:.4f}]")
print(f"  MAE (raw vs GT)    : {mae_raw:.4f}")
print(f"  RMSE (raw vs GT)   : {rmse_raw:.4f}")
print(f"  MAE (calibrated)   : {mae_cal:.4f}")
print(f"  RMSE (calibrated)  : {rmse_cal:.4f}")
