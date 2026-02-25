"""
Fitness-to-Drive - Per-second error probability after distraction
===================================================================
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Set, Tuple
from bisect import bisect_left
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss, log_loss,
    precision_recall_curve, f1_score, precision_score, recall_score,
    matthews_corrcoef, cohen_kappa_score,
    RocCurveDisplay, PrecisionRecallDisplay,
)
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
import joblib

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
DATA_PATH  = 'data/'
OUTPUT_DIR = 'evaluation/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
rng_global  = np.random.RandomState(RANDOM_SEED)

H_CANDIDATES  = [5]
RECALL_LEVELS = [0.80, 0.85, 0.90, 0.95]
N_BOOTSTRAP   = 500
BOOTSTRAP_CI  = 0.95
UNKNOWN_LABEL = 'unknown'

XGB_PARAM_GRID = {
    'xgb__n_estimators':     [100, 200, 300],
    'xgb__max_depth':        [3, 5, 7],
    'xgb__learning_rate':    [0.01, 0.05, 0.1],
    'xgb__subsample':        [0.8, 1.0],
    'xgb__colsample_bytree': [0.8, 1.0],
}

# ──────────────────────────────────────────────────────────────────────────────
# FEATURE_COLS — distraction_active removed
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'time_since_last_dist',
    'time_in_current_dist',
    'cognitive_load_decay',
    'model_prob',
    'model_pred_enc',
    'model_prob_sq',
    'emotion_prob',
    'emotion_label_enc',
    'arousal_deviation',
    'user_arousal_baseline',
    'hr_bpm',
    'emotion_prob_sq',
    'arousal_deviation_sq',
    'distraction_density_30',
    'distraction_density_60',
    'distraction_density_120',
    'prev_dist_duration',
    'user_hr_baseline',
    'baseline_error_rate',
]

# ──────────────────────────────────────────────────────────────────────────────
# 1. Load datasets
# ──────────────────────────────────────────────────────────────────────────────
log.info("Loading datasets …")
distractions = pd.read_csv(f'{DATA_PATH}Dataset Distractions_distraction.csv')
errors_dist  = pd.read_csv(f'{DATA_PATH}Dataset Errors_distraction.csv')
errors_base  = pd.read_csv(f'{DATA_PATH}Dataset Errors_baseline.csv')
driving_base = pd.read_csv(f'{DATA_PATH}Dataset Driving Time_baseline.csv')

distractions['timestamp_start'] = pd.to_datetime(distractions['timestamp_start'], format='ISO8601')
distractions['timestamp_end']   = pd.to_datetime(distractions['timestamp_end'], format='ISO8601')
errors_dist['timestamp']        = pd.to_datetime(errors_dist['timestamp'], format='ISO8601')

log.info(f"  Distraction events       : {len(distractions)}")
log.info(f"  Errors (distraction run) : {len(errors_dist)}")
log.info(f"  Errors (baseline run)    : {len(errors_base)}")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Data integrity checks (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
log.info("Running data integrity checks …")
issues = []

error_sessions   = set(errors_dist[['user_id', 'run_id']].drop_duplicates().itertuples(index=False, name=None))
dist_sessions    = set(distractions[['user_id', 'run_id']].drop_duplicates().itertuples(index=False, name=None))
missing_sessions = error_sessions - dist_sessions
if missing_sessions:
    issues.append(f"Errors in sessions without distractions: {missing_sessions}")

bad_windows = distractions[distractions['timestamp_end'] < distractions['timestamp_start']]
if len(bad_windows):
    issues.append(f"{len(bad_windows)} distraction windows have end < start")

for (uid, rid), grp in distractions.groupby(['user_id', 'run_id']):
    sorted_grp = grp.sort_values('timestamp_start').reset_index(drop=True)
    for i in range(1, len(sorted_grp)):
        if sorted_grp.iloc[i]['timestamp_start'] < sorted_grp.iloc[i-1]['timestamp_end']:
            issues.append(f"Overlap in distraction windows for {uid}, {rid}")

dup_errors = errors_dist.duplicated(subset=['user_id', 'run_id', 'timestamp'])
if dup_errors.sum():
    issues.append(f"{dup_errors.sum()} duplicate error timestamps in errors_dist")

required_cols = [
    'user_id', 'run_id', 'timestamp', 'model_pred', 'model_prob',
    'emotion_label', 'emotion_prob',
]
for col in required_cols:
    if col not in errors_dist.columns:
        issues.append(f"Missing column in errors_dist: '{col}'")

for col in ['model_prob', 'emotion_prob']:
    n_nan = errors_dist[col].isna().sum()
    if n_nan:
        log.warning(f"{n_nan} NaNs in errors_dist['{col}'] — imputing per-user median")
        errors_dist[col] = errors_dist.groupby('user_id')[col].transform(
            lambda x: x.fillna(x.median())
        )
        remaining = errors_dist[col].isna().sum()
        if remaining:
            global_med = errors_dist[col].median()
            errors_dist[col] = errors_dist[col].fillna(global_med)
            log.warning(f"  {remaining} values still NaN; filled with global median {global_med:.4f}")

if issues:
    for iss in issues:
        log.error(f"  [FAIL] {iss}")
    raise RuntimeError("Fix data issues before training.")
log.info("  [PASS] All integrity checks passed.")

# ──────────────────────────────────────────────────────────────────────────────
# Remaining functions unchanged except generate_seconds_for_H
# ──────────────────────────────────────────────────────────────────────────────

def build_session_lookups(users_set: Set) -> Tuple[Dict, Dict]:
    wbs: Dict  = {}
    webs: Dict = {}
    subset = distractions[distractions['user_id'].isin(users_set)]
    for (uid, rid), grp in subset.groupby(['user_id', 'run_id']):
        sorted_grp       = grp.sort_values('timestamp_start').reset_index(drop=True)
        wbs[(uid, rid)]  = sorted_grp
        webs[(uid, rid)] = sorted(sorted_grp['timestamp_end'].tolist())
    return wbs, webs

def compute_user_baselines(users_set: Set) -> Tuple[Dict, float]:
    err_sub = errors_base[errors_base['user_id'].isin(users_set)]
    drv_sub = driving_base[driving_base['user_id'].isin(users_set)]
    total_s     = drv_sub['run_duration_seconds'].sum()
    global_rate = len(err_sub) / total_s if total_s > 0 else 0.0
    user_errs   = err_sub.groupby('user_id').size()
    user_secs   = drv_sub.groupby('user_id')['run_duration_seconds'].sum()
    per_user    = (user_errs / user_secs).fillna(global_rate).to_dict()
    return per_user, global_rate

def compute_user_arousal_baseline(users_set: Set, wbs: Dict) -> Tuple[Dict, float]:
    raw: Dict = {}
    all_vals  = []
    for (uid, rid), wins in wbs.items():
        if uid not in users_set:
            continue
        vals = list(wins['arousal_start'].dropna()) + list(wins['arousal_end'].dropna())
        if vals:
            raw.setdefault(uid, []).extend(vals)
            all_vals.extend(vals)
    per_user   = {uid: float(np.median(v)) for uid, v in raw.items()}
    global_med = float(np.median(all_vals)) if all_vals else 0.5
    return per_user, global_med

def compute_user_hr_baseline(users_set: Set, wbs: Dict) -> Tuple[Dict, float]:
    raw: Dict = {}
    all_vals  = []
    for (uid, rid), wins in wbs.items():
        if uid not in users_set:
            continue
        vals = list(wins['hr_bpm_start'].dropna()) + list(wins['hr_bpm_end'].dropna())
        if vals:
            raw.setdefault(uid, []).extend(vals)
            all_vals.extend(vals)
    per_user   = {uid: float(np.median(v)) for uid, v in raw.items()}
    global_med = float(np.median(all_vals)) if all_vals else 70.0
    return per_user, global_med

def fit_label_encoders(train_users: Set, wbs_train: Dict) -> Tuple[LabelEncoder, LabelEncoder]:
    dist_tr = distractions[distractions['user_id'].isin(train_users)]
    err_tr  = errors_dist[errors_dist['user_id'].isin(train_users)]

    emo_vocab = sorted(set(
        pd.concat([dist_tr['emotion_label_start'], dist_tr['emotion_label_end'], err_tr['emotion_label']])
          .dropna().tolist() + [UNKNOWN_LABEL]
    ))
    pred_vocab = sorted(set(
        pd.concat([dist_tr['model_pred_start'], dist_tr['model_pred_end'], err_tr['model_pred']])
          .dropna().tolist() + [UNKNOWN_LABEL]
    ))

    le_emo  = LabelEncoder().fit(emo_vocab)
    le_pred = LabelEncoder().fit(pred_vocab)
    return le_emo, le_pred

def _safe_encode(label, le: LabelEncoder) -> int:
    s = UNKNOWN_LABEL if pd.isna(label) else str(label).strip()
    if s not in le.classes_:
        s = UNKNOWN_LABEL
    return int(le.transform([s])[0])

def _get_prev_dist_duration(uid, rid, current_ts, wbs: Dict, current_window_start=None) -> float:
    key = (uid, rid)
    if key not in wbs:
        return 0.0
    wins = wbs[key]
    prev = wins[wins['timestamp_end'] < current_ts]
    if current_window_start is not None:
        prev = prev[prev['timestamp_start'] != current_window_start]
    if prev.empty:
        return 0.0
    last = prev.loc[prev['timestamp_end'].idxmax()]
    return (last['timestamp_end'] - last['timestamp_start']).total_seconds()

def _get_distraction_density(uid, rid, ts, lookback_s: int, webs: Dict) -> int:
    ends = webs.get((uid, rid), [])
    if not ends:
        return 0
    t_lo = ts - pd.Timedelta(seconds=lookback_s)
    return bisect_left(ends, ts) - bisect_left(ends, t_lo)

# ──────────────────────────────────────────────────────────────────────────────
# 7. Updated sample generator - no distraction_active
# ──────────────────────────────────────────────────────────────────────────────
def generate_seconds_for_H(
    H: int,
    users_set: Set,
    wbs: Dict,
    webs: Dict,
    fold_arousal_bl: Dict,
    fold_hr_bl: Dict,
    fold_baseline_rate: Dict,
    fold_global_rate: float,
    le_emo: LabelEncoder,
    le_pred_enc: LabelEncoder,
) -> pd.DataFrame:
    """
    Real-time compatible version: no ground-truth distraction_active flag.
    Uses start-state features during distraction window, end-state after.
    """
    rows = []
    dist_sel = (
        distractions[distractions['user_id'].isin(users_set)]
        .sort_values(['user_id', 'run_id', 'timestamp_start'])
        .reset_index(drop=True)
    )

    err_sel: Dict = {}
    for (uid, rid), grp in (
        errors_dist[errors_dist['user_id'].isin(users_set)]
        .groupby(['user_id', 'run_id'])
    ):
        err_sel[(uid, rid)] = sorted(grp['timestamp'].tolist())

    for _, row in dist_sel.iterrows():
        uid       = row['user_id']
        rid       = row['run_id']
        start_ts  = row['timestamp_start']
        end_ts    = row['timestamp_end']
        win_dur_s = (end_ts - start_ts).total_seconds()

        next_rows  = dist_sel[
            (dist_sel['user_id'] == uid) &
            (dist_sel['run_id']  == rid) &
            (dist_sel['timestamp_start'] > start_ts)
        ]
        next_start = next_rows['timestamp_start'].min() if not next_rows.empty else None
        hangover_s = H
        if next_start:
            hangover_s = min(H, (next_start - end_ts).total_seconds())
        if hangover_s < 0:
            hangover_s = 0
        upper_ts   = end_ts + pd.Timedelta(seconds=hangover_s)
        total_span_s = (upper_ts - start_ts).total_seconds()

        sf = {  # start-of-window state
            'arousal':           row['arousal_start'],
            'hr_bpm':            row['hr_bpm_start'],
            'model_prob':        row['model_prob_start'],
            'model_pred_enc':    _safe_encode(row['model_pred_start'], le_pred_enc),
            'emotion_prob':      row['emotion_prob_start'],
            'emotion_label_enc': _safe_encode(row['emotion_label_start'], le_emo),
        }
        ef = {  # end-of-window state
            'arousal':           row['arousal_end'],
            'hr_bpm':            row['hr_bpm_end'],
            'model_prob':        row['model_prob_end'],
            'model_pred_enc':    _safe_encode(row['model_pred_end'], le_pred_enc),
            'emotion_prob':      row['emotion_prob_end'],
            'emotion_label_enc': _safe_encode(row['emotion_label_end'], le_emo),
        }

        u_aro_bl    = fold_arousal_bl.get(uid, 0.5)
        u_hr_bl     = fold_hr_bl.get(uid, 70.0)
        u_base_rate = fold_baseline_rate.get(uid, fold_global_rate)

        errs = err_sel.get((uid, rid), [])
        num_bins = int(np.ceil(total_span_s))
        for offset_sec in range(num_bins):
            bin_start = start_ts + pd.Timedelta(seconds=offset_sec)
            bin_end   = min(bin_start + pd.Timedelta(seconds=1), upper_ts)
            left   = bisect_left(errs, bin_start)
            right  = bisect_left(errs, bin_end)
            target = 1 if right > left else 0

            inside_window = offset_sec < win_dur_s

            if inside_window:
                feat            = sf
                time_in_current = offset_sec
                time_since_last = 0.0
                cld             = feat['model_prob']
            else:
                feat            = ef
                time_in_current = 0.0
                time_since_last = offset_sec - win_dur_s
                cld             = feat['model_prob'] * np.exp(-time_since_last / H)

            arousal_dev = feat['arousal'] - u_aro_bl

            rows.append({
                'user_id':               uid,
                'run_id':                rid,
                'distraction_start_ts':  start_ts,
                'offset_sec':            offset_sec,
                'target':                target,

                'time_since_last_dist':  time_since_last,
                'time_in_current_dist':  time_in_current,
                'cognitive_load_decay':  cld,
                'model_prob':            feat['model_prob'],
                'model_pred_enc':        feat['model_pred_enc'],
                'model_prob_sq':         feat['model_prob'] ** 2,

                'emotion_prob':          feat['emotion_prob'],
                'emotion_label_enc':     feat['emotion_label_enc'],
                'arousal_deviation':     arousal_dev,
                'user_arousal_baseline': u_aro_bl,
                'hr_bpm':                feat['hr_bpm'],
                'emotion_prob_sq':       feat['emotion_prob'] ** 2,
                'arousal_deviation_sq':  arousal_dev ** 2,

                'distraction_density_30':  _get_distraction_density(uid, rid, bin_start, 30,  webs),
                'distraction_density_60':  _get_distraction_density(uid, rid, bin_start, 60,  webs),
                'distraction_density_120': _get_distraction_density(uid, rid, bin_start, 120, webs),
                'prev_dist_duration':      _get_prev_dist_duration(uid, rid, bin_start, wbs),

                'user_hr_baseline':      u_hr_bl,
                'baseline_error_rate':   u_base_rate,
            })

    return pd.DataFrame(rows)

# ──────────────────────────────────────────────────────────────────────────────
# The rest of the pipeline remains identical
# ──────────────────────────────────────────────────────────────────────────────

def build_and_tune_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    pos_rate: float,
    n_inner_splits: int = 3,
    verbose: int = 0,
) -> Pipeline:
    scale_pos_weight = (1 - pos_rate) / pos_rate
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', xgb.XGBClassifier(
            objective         = 'binary:logistic',
            random_state      = RANDOM_SEED,
            eval_metric       = 'logloss',
            n_jobs            = 1,
            scale_pos_weight  = scale_pos_weight,
        )),
    ])
    grid = GridSearchCV(
        pipeline, XGB_PARAM_GRID,
        cv      = GroupKFold(n_splits=n_inner_splits),
        scoring = 'average_precision',
        n_jobs  = 1,
        verbose = verbose,
        refit   = True,
    )
    grid.fit(X_train, y_train, groups=groups_train)
    return grid.best_estimator_

def select_threshold_from_cal(y_cal: np.ndarray, y_prob_cal: np.ndarray) -> float:
    prec, rec, thresh = precision_recall_curve(y_cal, y_prob_cal)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    return float(thresh[np.argmax(f1)])

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> Dict:
    y_bin = (y_pred >= threshold).astype(int)
    prec, rec, _ = precision_recall_curve(y_true, y_pred)
    metrics = {
        'AUC-PR':          average_precision_score(y_true, y_pred),
        'AUC-ROC':         roc_auc_score(y_true, y_pred),
        'Brier':           brier_score_loss(y_true, y_pred),
        'Log-Loss':        log_loss(y_true, y_pred),
        'MCC':             matthews_corrcoef(y_true, y_bin),
        'Kappa':           cohen_kappa_score(y_true, y_bin),
        'F1':              f1_score(y_true, y_bin),
        'Precision':       precision_score(y_true, y_bin),
        'Recall':          recall_score(y_true, y_bin),
        'Threshold (cal)': threshold,
    }
    for r in RECALL_LEVELS:
        mask = rec >= r
        metrics[f'P@{int(r*100)}'] = float(prec[mask].max()) if mask.any() else 0.0
    return metrics

def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, metric_func, n: int = N_BOOTSTRAP) -> Tuple[float, float, float]:
    rng    = np.random.RandomState(RANDOM_SEED)
    scores = []
    idx    = np.arange(len(y_true))
    for _ in range(n):
        boot = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_true[boot])) < 2:
            continue
        scores.append(metric_func(y_true[boot], y_score[boot]))
    if len(scores) < n * 0.8:
        log.warning(f"Only {len(scores)} valid bootstrap samples")
    lo = float(np.percentile(scores, (1 - BOOTSTRAP_CI) / 2 * 100))
    hi = float(np.percentile(scores, (1 + BOOTSTRAP_CI) / 2 * 100))
    return float(np.mean(scores)), lo, hi

# ──────────────────────────────────────────────────────────────────────────────
# 11. Nested cross-validation  (outer: H selection; inner: XGB tuning)
# ──────────────────────────────────────────────────────────────────────────────
all_users   = sorted(
    set(distractions['user_id'].unique()) | set(errors_dist['user_id'].unique())
)
group_kfold = GroupKFold(n_splits=5)

log.info("\n" + "=" * 70)
log.info("NESTED CROSS-VALIDATION OVER H")
log.info("=" * 70)

cv_results = []

for H in H_CANDIDATES:
    log.info(f"\n--- Evaluating H = {H} seconds ---")
    outer_scores: Dict = {'auc_pr': [], 'auc_roc': [], 'brier': [], 'logloss': []}

    for fold_i, (train_idx, test_idx) in enumerate(
        group_kfold.split(np.zeros(len(all_users)), groups=all_users)
    ):
        train_users_f = set(all_users[i] for i in train_idx)
        test_users_f  = set(all_users[i] for i in test_idx)

        # ── Training-side resources ──────────────────────────────
        wbs_tr, webs_tr  = build_session_lookups(train_users_f)
        fold_bsl, fold_gr = compute_user_baselines(train_users_f)
        fold_aro, _       = compute_user_arousal_baseline(train_users_f, wbs_tr)
        fold_hr,  _       = compute_user_hr_baseline(train_users_f, wbs_tr)
        le_emo_f, le_pr_f = fit_label_encoders(train_users_f, wbs_tr)

        df_train_f = generate_seconds_for_H(
            H, train_users_f,
            wbs_tr, webs_tr,
            fold_aro, fold_hr, fold_bsl, fold_gr,
            le_emo_f, le_pr_f,
        )
        if df_train_f.empty or df_train_f['target'].nunique() < 2:
            log.warning(f"  Fold {fold_i}: skipped (train set insufficient class diversity)")
            continue

        pos_r = df_train_f['target'].mean()
        log.info(
            f"  Fold {fold_i}: n_train={len(df_train_f):,}  "
            f"pos_rate={pos_r:.4f}  imbalance={((1-pos_r)/pos_r):.1f}:1"
        )

        best_fold_model = build_and_tune_pipeline(
            df_train_f[FEATURE_COLS].values.astype(float),
            df_train_f['target'].values.astype(int),
            groups_train=df_train_f['user_id'].values,
            pos_rate=pos_r,
        )

        # ── Test-side resources: own baselines, training encoders ────────────
        wbs_te, webs_te  = build_session_lookups(test_users_f)
        tst_bsl, tst_gr  = compute_user_baselines(test_users_f)
        tst_aro, _       = compute_user_arousal_baseline(test_users_f, wbs_te)
        tst_hr,  _       = compute_user_hr_baseline(test_users_f, wbs_te)

        df_test_f = generate_seconds_for_H(
            H, test_users_f,
            wbs_te, webs_te,
            tst_aro, tst_hr, tst_bsl, tst_gr,
            le_emo_f, le_pr_f,     # encoders trained on training fold
        )
        if df_test_f.empty or df_test_f['target'].nunique() < 2:
            log.warning(f"  Fold {fold_i}: skipped (test set insufficient class diversity)")
            continue

        y_te = df_test_f['target'].values.astype(int)
        y_pr = best_fold_model.predict_proba(
            df_test_f[FEATURE_COLS].values.astype(float)
        )[:, 1]

        outer_scores['auc_pr'].append(average_precision_score(y_te, y_pr))
        outer_scores['auc_roc'].append(roc_auc_score(y_te, y_pr))
        outer_scores['brier'].append(brier_score_loss(y_te, y_pr))
        outer_scores['logloss'].append(log_loss(y_te, y_pr))

    if not outer_scores['auc_pr']:
        continue

    cv_results.append({
        'H':            H,
        'auc_pr_mean':  np.mean(outer_scores['auc_pr']),
        'auc_pr_std':   np.std(outer_scores['auc_pr']),
        'auc_roc_mean': np.mean(outer_scores['auc_roc']),
        'brier_mean':   np.mean(outer_scores['brier']),
        'logloss_mean': np.mean(outer_scores['logloss']),
    })
    log.info(
        f"  AUC-PR: {cv_results[-1]['auc_pr_mean']:.4f} "
        f"± {cv_results[-1]['auc_pr_std']:.4f}"
    )

results_df = pd.DataFrame(cv_results)
best_H     = int(results_df.loc[results_df['auc_pr_mean'].idxmax(), 'H'])
log.info(f"\nBest H = {best_H} seconds")

# ──────────────────────────────────────────────────────────────────────────────
# 12. Final model — user splits
# ──────────────────────────────────────────────────────────────────────────────
all_users_arr = np.array(all_users)
rng_global.shuffle(all_users_arr)

n_test      = int(0.20 * len(all_users_arr))
n_cal       = int(0.20 * (len(all_users_arr) - n_test))
test_users  = set(all_users_arr[:n_test])
cal_users   = set(all_users_arr[n_test: n_test + n_cal])
train_users = set(all_users_arr[n_test + n_cal:])

log.info(
    f"\nFinal split — "
    f"train: {len(train_users)} users | "
    f"cal: {len(cal_users)} users | "
    f"test: {len(test_users)} users"
)

# Training resources
wbs_tr_f, webs_tr_f    = build_session_lookups(train_users)
train_baselines, train_gr = compute_user_baselines(train_users)
train_aro_bl, _           = compute_user_arousal_baseline(train_users, wbs_tr_f)
train_hr_bl, _            = compute_user_hr_baseline(train_users, wbs_tr_f)
le_emotion_f, le_pred_f   = fit_label_encoders(train_users, wbs_tr_f)

df_train_final = generate_seconds_for_H(
    best_H, train_users,
    wbs_tr_f, webs_tr_f,
    train_aro_bl, train_hr_bl, train_baselines, train_gr,
    le_emotion_f, le_pred_f,
)

# Calibration resources 
wbs_cal_f, webs_cal_f = build_session_lookups(cal_users)       
cal_baselines, cal_gr = compute_user_baselines(cal_users)
cal_aro_bl, _         = compute_user_arousal_baseline(cal_users, wbs_cal_f)
cal_hr_bl, _          = compute_user_hr_baseline(cal_users, wbs_cal_f)

df_cal_final = generate_seconds_for_H(
    best_H, cal_users,
    wbs_cal_f, webs_cal_f,
    cal_aro_bl, cal_hr_bl, cal_baselines, cal_gr,
    le_emotion_f, le_pred_f,
)

for split_name, df in [('train', df_train_final), ('calibration', df_cal_final)]:
    if df['target'].nunique() < 2:
        raise RuntimeError(f"{split_name} set has only one class — adjust the split.")
    pr = df['target'].mean()
    log.info(
        f"  {split_name:13s}: n={len(df):,}  "
        f"pos_rate={pr:.4f}  imbalance={((1-pr)/pr):.1f}:1"
    )

X_tr_f = df_train_final[FEATURE_COLS].values.astype(float)
y_tr_f = df_train_final['target'].values.astype(int)
X_ca_f = df_cal_final[FEATURE_COLS].values.astype(float)
y_ca_f = df_cal_final['target'].values.astype(int)

best_model = build_and_tune_pipeline(
    X_tr_f, y_tr_f,
    groups_train=df_train_final['user_id'].values,
    pos_rate=y_tr_f.mean(),
    verbose=1,
)

# Calibrate on calibration set
calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv=None)
calibrated_model.fit(X_ca_f, y_ca_f)

y_prob_cal_f     = calibrated_model.predict_proba(X_ca_f)[:, 1]
best_threshold   = select_threshold_from_cal(y_ca_f, y_prob_cal_f)
y_prob_uncal_ca  = best_model.predict_proba(X_ca_f)[:, 1]
threshold_uncal  = select_threshold_from_cal(y_ca_f, y_prob_uncal_ca)
log.info(f"  Calibrated threshold (cal set)   : {best_threshold:.4f}")
log.info(f"  Uncalibrated threshold (cal set) : {threshold_uncal:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 13. Evaluation on held-out test set
# ──────────────────────────────────────────────────────────────────────────────
wbs_te_f, webs_te_f   = build_session_lookups(test_users)
test_baselines, tst_gr = compute_user_baselines(test_users)
test_aro_bl, _         = compute_user_arousal_baseline(test_users, wbs_te_f)
test_hr_bl, _          = compute_user_hr_baseline(test_users, wbs_te_f)

df_test_final = generate_seconds_for_H(
    best_H, test_users,
    wbs_te_f, webs_te_f,
    test_aro_bl, test_hr_bl, test_baselines, tst_gr,
    le_emotion_f, le_pred_f,        # encoders from training only
)

X_te_f       = df_test_final[FEATURE_COLS].values.astype(float)
y_te_f       = df_test_final['target'].values.astype(int)
y_pred_uncal = best_model.predict_proba(X_te_f)[:, 1]
y_pred_cal   = calibrated_model.predict_proba(X_te_f)[:, 1]

# Thresholds from calibration set
metrics_uncal = compute_metrics(y_te_f, y_pred_uncal, threshold_uncal)
metrics_cal   = compute_metrics(y_te_f, y_pred_cal,   best_threshold)

log.info("\n" + "=" * 70)
log.info("FINAL MODEL EVALUATION ON TEST SET")
log.info("=" * 70)
log.info("\nUncalibrated XGBoost:")
for k, v in metrics_uncal.items():
    log.info(f"  {k}: {v:.4f}")
log.info("\nCalibrated XGBoost:")
for k, v in metrics_cal.items():
    log.info(f"  {k}: {v:.4f}")

ci_aucpr,  lo_aucpr,  hi_aucpr  = bootstrap_ci(y_te_f, y_pred_cal, average_precision_score)
ci_aucroc, lo_aucroc, hi_aucroc = bootstrap_ci(y_te_f, y_pred_cal, roc_auc_score)
ci_brier,  lo_brier,  hi_brier  = bootstrap_ci(y_te_f, y_pred_cal, brier_score_loss)

log.info("\nBootstrap 95% CI (calibrated model):")
log.info(f"  AUC-PR : {ci_aucpr:.4f} [{lo_aucpr:.4f} - {hi_aucpr:.4f}]")
log.info(f"  AUC-ROC: {ci_aucroc:.4f} [{lo_aucroc:.4f} - {hi_aucroc:.4f}]")
log.info(f"  Brier  : {ci_brier:.4f} [{lo_brier:.4f} - {hi_brier:.4f}]")

# ──────────────────────────────────────────────────────────────────────────────
# 14. Plots
# ──────────────────────────────────────────────────────────────────────────────

try:
    # Calibration curve
    fig, ax = plt.subplots(figsize=(8, 6))
    for proba, label in [(y_pred_uncal, 'Uncalibrated'), (y_pred_cal, 'Calibrated')]:
        fop, mpv = calibration_curve(y_te_f, proba, n_bins=10)
        ax.plot(mpv, fop, marker='o', label=label)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title(f'Calibration curves (H={best_H}s)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'calibration_curve.png'), dpi=150)
    plt.close()

    # ROC and PR curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    RocCurveDisplay.from_predictions(y_te_f, y_pred_cal, ax=ax1, name='Calibrated')
    ax1.plot([0, 1], [0, 1], 'k--', label='Chance')
    ax1.set_title(f'ROC Curve (H={best_H}s)')
    ax1.legend()
    PrecisionRecallDisplay.from_predictions(y_te_f, y_pred_cal, ax=ax2)
    ax2.set_title(f'Precision-Recall Curve (H={best_H}s)')
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
    plt.title(f'XGBoost Feature Importance (H={best_H}s)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=150)
    plt.close()

    # Per-second risk profile
    df_test_final = df_test_final.copy()
    df_test_final['pred_cal']         = y_pred_cal
    df_test_final['time_since_start'] = df_test_final['offset_sec'].clip(0, best_H)
    empirical_rate = df_test_final.groupby('time_since_start')['target'].mean()
    pred_mean      = df_test_final.groupby('time_since_start')['pred_cal'].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(empirical_rate.index, empirical_rate.values, 'o-',
             label='Empirical error rate', color='red')
    plt.plot(pred_mean.index, pred_mean.values, 's-',
             label='Model predicted probability', color='blue')
    plt.axhline(y=df_test_final['baseline_error_rate'].mean(),
                color='gray', linestyle='--', label='Baseline rate')
    plt.xlabel('Seconds since distraction start')
    plt.ylabel('Error rate / Probability')
    plt.title(f'Per-second risk profile (H={best_H}s)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'per_second_risk.png'), dpi=150)
    plt.close()
except Exception as e:
    log.error(f"Plotting failed: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# 15. Save artifacts
# ──────────────────────────────────────────────────────────────────────────────
try:
    artifact = {
        # Models
        'model':               best_model,
        'calibrated_model':    calibrated_model,
        # Decision thresholds (chosen on calibration set)
        'best_threshold_cal':  best_threshold,
        'best_threshold_uncal':threshold_uncal,
        # Configuration
        'best_H':              best_H,
        'feature_cols':        FEATURE_COLS,
        # Encoders (trained on training users only)
        'le_emotion':          le_emotion_f,
        'le_pred':             le_pred_f,
        # Baselines (training users)
        'train_baselines':     train_baselines,
        'train_global_rate':   train_gr,
        'train_arousal_bl':    train_aro_bl,
        'train_hr_bl':         train_hr_bl,
        # Evaluation
        'cv_results_df':       results_df,
        'metrics_uncal':       metrics_uncal,
        'metrics_cal':         metrics_cal,
        'bootstrap_ci': {
            'auc_pr':  (ci_aucpr,  lo_aucpr,  hi_aucpr),
            'auc_roc': (ci_aucroc, lo_aucroc, hi_aucroc),
            'brier':   (ci_brier,  lo_brier,  hi_brier),
        },
    }
    joblib.dump(artifact, os.path.join(OUTPUT_DIR, 'fitness_model.pkl'))
    log.info(f"\nAll results saved to {OUTPUT_DIR}")
except Exception as e:
    log.error(f"Saving artifacts failed: {e}")


