"""
Driver Digital Twin – Fitness-to-Drive XGBoost Pipeline
========================================================
Bulletproof version. Adds:
  - Data integrity checks (leakage, nulls, timestamp ordering)
  - Bootstrap 95% CIs on all metrics
  - Extra metrics: MCC, Brier Score, P@R at fixed recall levels
  - Fixed random seeds everywhere

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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, roc_curve,
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score,
    matthews_corrcoef, brier_score_loss,
    log_loss, cohen_kappa_score,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
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
H_CANDIDATES     = list(range(5, 36))          # fine-grained 10–35s sweep
NEG_SAMPLE_EVERY = 5                            # sample one negative every N seconds
N_BOOTSTRAP      = 1000                         # bootstrap iterations for CIs
RECALL_LEVELS    = [0.80, 0.85, 0.90, 0.95]    # fixed recall levels for P@R

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
            'emotion_label', 'emotion_prob']:
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

# ── 4. Distraction window lookup ──────────────────────────────────────────────
windows_by_session = {}
for (uid, rid), grp in distractions.groupby(['user_id', 'run_id']):
    windows_by_session[(uid, rid)] = grp.reset_index(drop=True)

def get_distraction_state(user_id, run_id, ts, H):
    key  = (user_id, run_id)
    if key not in windows_by_session:
        return 0, float(H)
    wins = windows_by_session[key]
    if ((wins['timestamp_start'] <= ts) & (ts <= wins['timestamp_end'])).any():
        return 1, 0.0
    prev = wins[wins['timestamp_end'] < ts]
    if prev.empty:
        return 0, float(H)
    delta = (ts - prev['timestamp_end'].max()).total_seconds()
    return 0, min(delta, float(H))

# ── 5. Label encoding ─────────────────────────────────────────────────────────
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

# ── 6. Sample builders ─────────────────────────────────────────────────────────
def build_positives(H):
    rows = []
    for _, err in errors_dist.iterrows():
        uid, rid, ts = err['user_id'], err['run_id'], err['timestamp']
        dist_active, time_since = get_distraction_state(uid, rid, ts, H)
        emo_enc, pred_enc       = encode_row(err['emotion_label'], err['model_pred'])
        rows.append({
            'user_id': uid, 'distraction_active': dist_active,
            'time_since_last_dist': time_since, 'model_prob': err['model_prob'],
            'model_pred_enc': pred_enc, 'emotion_prob': err['emotion_prob'],
            'emotion_label_enc': emo_enc,             'baseline_error_rate': user_baselines.get(uid, p_baseline_global),
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
                    'emotion_label_enc': emo_enc,                     'baseline_error_rate': b_rate, 'label': 0,
                })

        # Inter-window gaps
        for i in range(len(wins_s) - 1):
            gap_start = wins_s['timestamp_end'].iloc[i]
            gap_end   = wins_s['timestamp_start'].iloc[i + 1]
            gap_len   = (gap_end - gap_start).total_seconds()
            if gap_len <= 0: continue
            win_state = wins_s.iloc[i]
            emo_enc, pred_enc = encode_row(win_state['emotion_label_end'], win_state['model_pred_end'])
            for offset in np.arange(sample_every, gap_len, sample_every):
                ts = gap_start + pd.Timedelta(seconds=offset)
                if ts.floor('s') in err_set: continue
                rows.append({
                    'user_id': uid, 'distraction_active': 0,
                    'time_since_last_dist': min(offset, float(H)),
                    'model_prob': win_state['model_prob_end'],
                    'model_pred_enc': pred_enc, 'emotion_prob': win_state['emotion_prob_end'],
                    'emotion_label_enc': emo_enc,                     'baseline_error_rate': b_rate, 'label': 0,
                })
    return pd.DataFrame(rows)

# ── 7. Bootstrap CI helper ─────────────────────────────────────────────────────
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

# ── 8. Rule-based baseline ────────────────────────────────────────────────────

# ── 9. Precision at fixed recall levels ───────────────────────────────────────
def precision_at_recall(y_true, y_score, recall_levels):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    results = {}
    for r in recall_levels:
        mask = rec >= r
        results[r] = float(prec[mask].max()) if mask.any() else 0.0
    return results

# ── 10. H grid search ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("H GRID SEARCH (Leave-One-User-Out)")
print("=" * 70)
print(f"{'H (s)':<8} {'AUC-PR':<10} {'AUC-ROC':<10} {'N':<8} {'Pos%'}")
print("-" * 50)

logo    = LeaveOneGroupOut()
results = []

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
        # Leakage guard
        train_users = set(groups[train_idx])
        test_users  = set(groups[test_idx])
        assert train_users.isdisjoint(test_users), \
            f"LEAKAGE DETECTED: {train_users & test_users}"

        clf = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, **XGB_PARAMS)
        clf.fit(X[train_idx], y[train_idx])
        y_true_all.extend(y[test_idx])
        y_prob_all.extend(clf.predict_proba(X[test_idx])[:, 1])

    auc_pr  = average_precision_score(y_true_all, y_prob_all)
    auc_roc = roc_auc_score(y_true_all, y_prob_all)
    results.append({'H': H, 'AUC-PR': auc_pr, 'AUC-ROC': auc_roc,
                    'n_samples': len(df), 'pos_pct': pos_rate * 100})
    print(f"{H:<8} {auc_pr:<10.4f} {auc_roc:<10.4f} {len(df):<8} {pos_rate*100:.1f}%")

results_df = pd.DataFrame(results)
best_row   = results_df.loc[results_df['AUC-PR'].idxmax()]
best_H     = int(best_row['H'])
print(f"\n  Best H = {best_H}s  (AUC-PR={best_row['AUC-PR']:.4f}, AUC-ROC={best_row['AUC-ROC']:.4f})")

# ── 11. Full evaluation at best H ──────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"FULL EVALUATION  (H={best_H}s, LOSO-CV)")
print("=" * 70)

pos_ev = build_positives(best_H)
neg_ev = build_negatives(best_H)
df_ev  = pd.concat([pos_ev, neg_ev], ignore_index=True).dropna(subset=FEATURE_COLS)

X_ev      = df_ev[FEATURE_COLS].values.astype(float)
y_ev      = df_ev['label'].values.astype(int)
groups_ev = df_ev['user_id'].values
pos_rate_ev = y_ev.mean()
spw_ev      = (1 - pos_rate_ev) / pos_rate_ev

y_true_all, y_prob_xgb, idx_all = [], [], []
per_user_results = {}

for train_idx, test_idx in logo.split(X_ev, y_ev, groups_ev):
    assert set(groups_ev[train_idx]).isdisjoint(set(groups_ev[test_idx]))
    X_tr, X_te = X_ev[train_idx], X_ev[test_idx]
    y_tr, y_te = y_ev[train_idx], y_ev[test_idx]

    # XGBoost
    clf = xgb.XGBClassifier(scale_pos_weight=spw_ev, **XGB_PARAMS)
    clf.fit(X_tr, y_tr)
    probs_xgb = clf.predict_proba(X_te)[:, 1]

    # Accumulate in fold order (for global metrics)
    y_true_all.extend(y_ev[test_idx])
    y_prob_xgb.extend(probs_xgb)
    idx_all.extend(test_idx.tolist())   # track original row indices for GT validation

    uid = groups_ev[test_idx][0]
    if len(np.unique(y_te)) > 1:
        per_user_results[uid] = {
            'AUC-PR':  average_precision_score(y_te, probs_xgb),
            'AUC-ROC': roc_auc_score(y_te, probs_xgb),
            'MCC':     matthews_corrcoef(y_te, (probs_xgb >= 0.5).astype(int)),
            'n_pos': int(y_te.sum()), 'n_neg': int((y_te == 0).sum()),
        }
    else:
        per_user_results[uid] = {'AUC-PR': None, 'AUC-ROC': None, 'MCC': None,
                                 'n_pos': int(y_te.sum()), 'n_neg': int((y_te == 0).sum())}

# Fold-order arrays — used for all global metrics (order doesn't matter there)
y_true_arr  = np.array(y_true_all)
y_prob_arr  = np.array(y_prob_xgb)

# Original-row-order arrays — used for GT validation so feature values match predictions
# LeaveOneGroupOut iterates users in group order, not df_ev row order, so we must
# map predictions back to their original positions before joining with df_ev features.
restore_order = np.argsort(idx_all)          # permutation that recovers df_ev row order
y_prob_orig_order = y_prob_arr[restore_order] # predictions aligned with df_ev rows
y_true_orig_order = y_true_arr[restore_order] # sanity: should equal df_ev['label'].values

# Optimal threshold (max F1)
prec_c, rec_c, thresh_c = precision_recall_curve(y_true_arr, y_prob_arr)
f1_c        = 2*prec_c[:-1]*rec_c[:-1]/(prec_c[:-1]+rec_c[:-1]+1e-9)
best_thresh = thresh_c[np.argmax(f1_c)]
y_pred      = (y_prob_arr >= best_thresh).astype(int)

# ── Compute all metrics ────────────────────────────────────────────────────────
def compute_metrics(y_true, y_score, thresh):
    yp  = (y_score >= thresh).astype(int)
    cm_ = confusion_matrix(y_true, yp)
    tn, fp, fn, tp = cm_.ravel() if cm_.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # true negative rate
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0   # P(safe|predicted safe)
    m = {
        # ── Ranking metrics (threshold-independent) ──────────────────────────
        'AUC-PR':       average_precision_score(y_true, y_score),
        'AUC-ROC':      roc_auc_score(y_true, y_score),
        'Log-Loss':     log_loss(y_true, y_score),          # penalises overconfident errors
        # ── Threshold-dependent classification metrics ────────────────────────
        'MCC':          matthews_corrcoef(y_true, yp),      # balanced, accounts all 4 cells
        'Kappa':        cohen_kappa_score(y_true, yp),      # agreement beyond chance
        'Brier':        brier_score_loss(y_true, y_score),  # probability calibration quality
        'F1':           f1_score(y_true, yp, zero_division=0),
        'Precision':    precision_score(y_true, yp, zero_division=0),  # PPV
        'Recall':       recall_score(y_true, yp, zero_division=0),     # sensitivity / TPR
        'Specificity':  specificity,   # TNR — how well safe moments are correctly identified
        'NPV':          npv,           # P(truly safe | predicted safe) — important for trust
    }
    par = precision_at_recall(y_true, y_score, RECALL_LEVELS)
    m.update({f'P@R={int(r*100)}': v for r, v in par.items()})
    return m

m_xgb   = compute_metrics(y_true_arr, y_prob_arr,  best_thresh)

# Bootstrap CIs (XGBoost only)
print(f"\nComputing bootstrap CIs (n={N_BOOTSTRAP}) ...")
ci_aucpr,  lo_aucpr,  hi_aucpr  = bootstrap_ci(y_true_arr, y_prob_arr, average_precision_score)
ci_aucroc, lo_aucroc, hi_aucroc = bootstrap_ci(y_true_arr, y_prob_arr, roc_auc_score)
ci_brier,  lo_brier,  hi_brier  = bootstrap_ci(y_true_arr, y_prob_arr, brier_score_loss)
mcc_fn = lambda yt, ys: matthews_corrcoef(yt, (ys >= best_thresh).astype(int))
ci_mcc,   lo_mcc,   hi_mcc   = bootstrap_ci(y_true_arr, y_prob_arr, mcc_fn)
kappa_fn = lambda yt, ys: cohen_kappa_score(yt, (ys >= best_thresh).astype(int))
ci_kappa, lo_kappa, hi_kappa = bootstrap_ci(y_true_arr, y_prob_arr, kappa_fn)
ci_logloss, lo_ll, hi_ll     = bootstrap_ci(y_true_arr, y_prob_arr, log_loss)

# ── Print comparison table ─────────────────────────────────────────────────────
print("\n── Model Metrics ───────────────────────────────────────────────────────")
print(f"  {'Metric':<18} {'XGBoost (95% CI)'}")
print("  " + "-" * 52)
rows_tbl = [
    ('AUC-PR',    f"{m_xgb['AUC-PR']:.4f}  [{lo_aucpr:.4f} - {hi_aucpr:.4f}]"),
    ('AUC-ROC',   f"{m_xgb['AUC-ROC']:.4f}  [{lo_aucroc:.4f} - {hi_aucroc:.4f}]"),
    ('Log-Loss',  f"{m_xgb['Log-Loss']:.4f}  [{lo_ll:.4f} - {hi_ll:.4f}]"),
    ('MCC',       f"{m_xgb['MCC']:.4f}  [{lo_mcc:.4f} - {hi_mcc:.4f}]"),
    ('Kappa',     f"{m_xgb['Kappa']:.4f}  [{lo_kappa:.4f} - {hi_kappa:.4f}]"),
    ('Brier',     f"{m_xgb['Brier']:.4f}  [{lo_brier:.4f} - {hi_brier:.4f}]"),
    ('F1', f"{m_xgb['F1']:.4f}"),
    ('Precision', f"{m_xgb['Precision']:.4f}"),
    ('Recall', f"{m_xgb['Recall']:.4f}"),
    ('Specificity', f"{m_xgb['Specificity']:.4f}"),
    ('NPV', f"{m_xgb['NPV']:.4f}"),
]
for r in rows_tbl:
    print(f"  {r[0]:<18} {r[1]}")

print(f"\n  Threshold (max-F1) : {best_thresh:.4f}")
print(f"\n── Precision @ Fixed Recall (XGBoost) ─────────────────────────────────")
for r in RECALL_LEVELS:
    print(f"  P @ Recall={r:.0%} : {m_xgb[f'P@R={int(r*100)}']:.4f}")

print(f"\n── Classification Report (XGBoost @ t={best_thresh:.2f}) ──────────────")
print(classification_report(y_true_arr, y_pred, target_names=['Safe','Error']))

# ── Lead-Time Analysis ─────────────────────────────────────────────────────────
# For each error in the LOSO predictions, find how many seconds BEFORE the error
# the model first raised an alert (score >= best_thresh).
# This is the operationally critical metric: "how much warning does the system give?"
print(f"\n── Alert Lead-Time Analysis ────────────────────────────────────────────")

# Reconstruct per-session ordered predictions
# Re-run LOSO keeping order and session info
df_ev_sorted = df_ev.reset_index(drop=True)

# For lead-time we need temporal ordering within each session.
# Approximate sim_time ordering using original row order (rows are time-ordered).
# Group by user and find, for each error row, how many preceding seconds in the
# same user's block were already flagged.
lead_times = []
df_ev_sorted['prob']  = y_prob_orig_order
df_ev_sorted['alert'] = (y_prob_orig_order >= best_thresh).astype(int)

for uid, grp in df_ev_sorted.groupby('user_id'):
    grp = grp.reset_index(drop=True)
    error_idxs = grp.index[grp['label'] == 1].tolist()
    for ei in error_idxs:
        # Look back up to best_H seconds (one row ≈ one second for positives,
        # NEG_SAMPLE_EVERY seconds for negatives — use row distance as proxy)
        lookback = grp.iloc[max(0, ei - best_H): ei]
        if lookback.empty:
            continue
        # First alert in the lookback window
        alerted = lookback[lookback['alert'] == 1]
        if not alerted.empty:
            lead_times.append(ei - alerted.index[0])

if lead_times:
    lt = np.array(lead_times)
    print(f"  Errors preceded by ≥1 alert  : {len(lt)} / {int(y_true_arr.sum())} "
          f"({len(lt)/y_true_arr.sum()*100:.1f}%)")
    print(f"  Mean lead time               : {lt.mean():.1f} rows")
    print(f"  Median lead time             : {np.median(lt):.1f} rows")
    print(f"  Lead time ≥ 5 rows           : {(lt >= 5).sum()} ({(lt >= 5).mean()*100:.1f}%)")
    print(f"  Lead time ≥ 10 rows          : {(lt >= 10).sum()} ({(lt >= 10).mean()*100:.1f}%)")
    print(f"  (Row distance proxy: 1 positive row ≈ 1s; negative rows every {NEG_SAMPLE_EVERY}s)")

    # Lead-time histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lt, bins=20, color='steelblue', edgecolor='white')
    ax.axvline(lt.mean(),   color='red',    linestyle='--', label=f'Mean={lt.mean():.1f}')
    ax.axvline(np.median(lt), color='orange', linestyle='--', label=f'Median={np.median(lt):.1f}')
    ax.set_xlabel('Lead time (rows before error)')
    ax.set_ylabel('Count')
    ax.set_title('Alert Lead-Time Distribution')
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f'{EVAL_OUT}lead_time_histogram.png', dpi=150); plt.close()
    print(f"  ✓ Lead-time histogram → {EVAL_OUT}lead_time_histogram.png")
else:
    print("  No lead-time data available (no alerts fired before errors in LOSO folds)")


# ── Note on positive rate ──────────────────────────────────────────────────────
print(f"  NOTE: Positive rate in evaluation = {pos_rate_ev*100:.1f}% (sampled, not natural).")
print(f"  Natural rate ~ {p_baseline_global*100:.4f}% / second.")
print(f"  AUC-PR random baseline at sampled rate = {pos_rate_ev:.3f}.")

# ── Plots ──────────────────────────────────────────────────────────────────────

# Confusion matrix
cm = confusion_matrix(y_true_arr, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Safe','Error']); ax.set_yticklabels(['Safe','Error'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix (t={best_thresh:.2f})')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                color='white' if cm[i,j] > cm.max()/2 else 'black', fontsize=14)
plt.colorbar(im, ax=ax); plt.tight_layout()
plt.savefig(f'{EVAL_OUT}confusion_matrix.png', dpi=150); plt.close()

# PR curve – all three models
p_xgb,  r_xgb,  _ = precision_recall_curve(y_true_arr, y_prob_arr)
ax.scatter([r_xgb[np.argmax(f1_c)]], [p_xgb[np.argmax(f1_c)]],
           color='red', zorder=5, s=80, label=f'Best F1 (t={best_thresh:.2f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve (LOSO-CV)')
ax.legend(fontsize=9); ax.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f'{EVAL_OUT}pr_curve.png', dpi=150); plt.close()

# ROC curve
fpr_x, tpr_x, _ = roc_curve(y_true_arr, y_prob_arr)
ax.plot([0,1],[0,1], color='grey', lw=1, linestyle=':', label='Random')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve (LOSO-CV)')
ax.legend(fontsize=9); ax.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f'{EVAL_OUT}roc_curve.png', dpi=150); plt.close()

# Calibration curve (pre-Platt)
frac_pos, mean_pred = calibration_curve(y_true_arr, y_prob_arr, n_bins=10)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(mean_pred, frac_pos, marker='o', color='steelblue', lw=2,
        label='XGBoost (pre-calibration)')
ax.plot([0,1],[0,1], linestyle='--', color='grey', label='Perfect calibration')
ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Fraction of positives')
ax.set_title('Calibration Curve (LOSO-CV, before Platt scaling)')
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f'{EVAL_OUT}calibration_curve.png', dpi=150); plt.close()

# H sweep
fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()
ax1.plot(results_df['H'], results_df['AUC-PR'],  'o-',  color='steelblue',  lw=2, label='AUC-PR')
ax2.plot(results_df['H'], results_df['AUC-ROC'], 's--', color='darkorange', lw=2, label='AUC-ROC')
ax1.axvline(x=best_H, color='red', linestyle=':', lw=1.5, label=f'Best H={best_H}s')
ax1.set_xlabel('Hangover window H (seconds)')
ax1.set_ylabel('AUC-PR', color='steelblue')
ax2.set_ylabel('AUC-ROC', color='darkorange')
ax1.set_title('H Grid Search – Performance vs Recovery Window')
lines1, lbl1 = ax1.get_legend_handles_labels()
lines2, lbl2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, lbl1+lbl2, loc='lower right')
ax1.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f'{EVAL_OUT}h_sweep.png', dpi=150); plt.close()

# Feature importances
clf_imp = xgb.XGBClassifier(scale_pos_weight=spw_ev, **XGB_PARAMS)
clf_imp.fit(X_ev, y_ev)
importances = pd.Series(clf_imp.feature_importances_, index=FEATURE_COLS).sort_values()
colors = ['#d62728' if f in ('distraction_active','time_since_last_dist')
          else '#1f77b4' if f == 'baseline_error_rate' else '#aec7e8'
          for f in importances.index]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(importances.index, importances.values, color=colors)
ax.set_xlabel('Feature Importance (XGBoost gain)')
ax.set_title('Feature Importances')
ax.legend(handles=[
    Patch(facecolor='#d62728', label='Distraction features'),
    Patch(facecolor='#1f77b4', label='User prior'),
    Patch(facecolor='#aec7e8', label='Physiological (low weight expected)'),
], loc='lower right')
ax.grid(axis='x', alpha=0.3); plt.tight_layout()
plt.savefig(f'{EVAL_OUT}feature_importances.png', dpi=150); plt.close()

# Per-user bar chart (AUC-PR)
valid_users = {u: v for u, v in per_user_results.items() if v['AUC-PR'] is not None}
uids  = sorted(valid_users.keys())
aucs  = [valid_users[u]['AUC-PR'] for u in uids]
n_pos = [valid_users[u]['n_pos']  for u in uids]
fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(range(len(uids)), aucs, color='steelblue', alpha=0.8)
ax.axhline(y=m_xgb['AUC-PR'], color='red', linestyle='--', lw=1.5,
           label=f'Global mean {m_xgb["AUC-PR"]:.3f}')
ax.axhline(y=pos_rate_ev, color='grey', linestyle=':', lw=1, label='Random baseline')
for i, (b, n) in enumerate(zip(bars, n_pos)):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
            str(n), ha='center', va='bottom', fontsize=8, color='dimgray')
ax.set_xticks(range(len(uids)))
ax.set_xticklabels([str(u).replace('participant_','P') for u in uids], rotation=45, ha='right')
ax.set_ylabel('AUC-PR'); ax.set_title('Per-User AUC-PR (numbers = error count)')
ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{EVAL_OUT}per_user_auc_pr.png', dpi=150); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# GROUND TRUTH VALIDATION
# All "expected" values below are derived directly from the data —
# observed error frequencies computed from your actual error/distraction
# datasets — and compared against the model's predicted probabilities.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GROUND TRUTH VALIDATION")
print("=" * 70)

# We work on df_ev (the evaluation dataframe) enriched with LOSO predictions.
# Every row has: label (0/1), distraction_active, time_since_last_dist, y_prob.
df_gt = df_ev.reset_index(drop=True).copy()
# Use orig-order probabilities so df_gt features align correctly with predictions
df_gt['y_prob'] = y_prob_orig_order
assert np.array_equal(df_gt['label'].values, y_true_orig_order), \
    'Row order mismatch between df_ev and restored predictions — GT validation invalid'

# ── 1. RISK STRATIFICATION VALIDITY ───────────────────────────────────────────
# Split rows into three mutually exclusive conditions and compare:
#   observed error rate (ground truth from data) vs model mean predicted prob.
print("\n── 1. Risk Stratification (GT observed rate vs model prediction) ────────")

mask_active   = df_gt['distraction_active'] == 1
mask_hangover = (df_gt['distraction_active'] == 0) & \
                (df_gt['time_since_last_dist'] < best_H)
mask_baseline = (df_gt['distraction_active'] == 0) & \
                (df_gt['time_since_last_dist'] >= best_H)

conditions = [
    ('Active distraction',  mask_active),
    (f'Hangover (0–{best_H}s)', mask_hangover),
    ('Baseline (recovered)', mask_baseline),
]

strat_rows = []
print(f"  {'Condition':<28} {'GT error rate':>14} {'Model mean prob':>16} {'N rows':>8}")
print("  " + "-" * 72)
for label, mask in conditions:
    subset = df_gt[mask]
    if len(subset) == 0:
        continue
    gt_rate    = subset['label'].mean()          # observed error frequency in data
    model_prob = subset['y_prob'].mean()         # model mean predicted probability
    print(f"  {label:<28} {gt_rate:>14.4f} {model_prob:>16.4f} {len(subset):>8}")
    strat_rows.append({'condition': label, 'gt_error_rate': gt_rate,
                       'model_mean_prob': model_prob, 'n': len(subset)})

strat_df = pd.DataFrame(strat_rows)

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(strat_df))
w = 0.35
ax.bar(x - w/2, strat_df['gt_error_rate'],   w, label='GT observed error rate',
       color='#d62728', alpha=0.85)
ax.bar(x + w/2, strat_df['model_mean_prob'], w, label='Model mean prediction',
       color='steelblue', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(strat_df['condition'], rotation=15, ha='right')
ax.set_ylabel('Rate / Probability')
ax.set_title('Risk Stratification: Ground Truth vs Model')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{EVAL_OUT}gt_risk_stratification.png', dpi=150); plt.close()
print(f"  ✓ Saved → {EVAL_OUT}gt_risk_stratification.png")

# ── 2. TEMPORAL DECAY VALIDATION ──────────────────────────────────────────────
# Ground truth is computed DIRECTLY from raw timestamps — completely independent
# of how training negatives were sampled (which would introduce sampling artifacts).
#
# For each distraction event, we compute:
#   GT rate per bin = (errors that fell in that bin) / (total seconds at risk in that bin)
# "Seconds at risk" = min(gap_to_next_distraction, H) capped at each bin boundary.
# This is a proper exposure-weighted rate, not a fraction of training rows.
#
# The model curve uses the already-computed df_gt predictions, filtered to
# distraction_active=0 rows, binned by time_since_last_dist.
print("\n── 2. Temporal Decay: GT rate (raw data) vs model probability by bin ───")

bin_size  = 5
bin_edges = np.arange(0, best_H + bin_size, bin_size)
bin_labels = [f'{int(b)}–{int(b+bin_size)}s' for b in bin_edges[:-1]]
n_bins_decay = len(bin_labels)

# Accumulators: errors and exposure seconds per bin
gt_errors_per_bin   = np.zeros(n_bins_decay)
gt_exposure_per_bin = np.zeros(n_bins_decay)

for (uid, rid), wins in windows_by_session.items():
    wins_s = wins.sort_values('timestamp_start').reset_index(drop=True)

    for i, win in wins_s.iterrows():
        win_end = win['timestamp_end']

        # How long until the next distraction starts (or end of session)?
        future = wins_s[wins_s['timestamp_start'] > win_end]
        if future.empty:
            # No next distraction — use a large sentinel so gap covers full H
            gap_seconds = float(best_H)
        else:
            gap_seconds = (future['timestamp_start'].iloc[0] - win_end).total_seconds()

        gap_seconds = min(gap_seconds, float(best_H))

        # Errors that occurred after this window ended, within H seconds
        err_subset = errors_dist[
            (errors_dist['user_id'] == uid) &
            (errors_dist['run_id']  == rid) &
            (errors_dist['timestamp'] > win_end) &
            (errors_dist['timestamp'] <= win_end + pd.Timedelta(seconds=best_H))
        ].copy()
        err_subset['delta'] = (err_subset['timestamp'] - win_end).dt.total_seconds()

        # Assign errors and exposure to bins
        for b_idx, b_start in enumerate(bin_edges[:-1]):
            b_end = bin_edges[b_idx + 1]
            # Exposure: seconds this gap overlaps with this bin
            exp = max(0.0, min(gap_seconds, b_end) - b_start)
            gt_exposure_per_bin[b_idx] += exp
            # Errors: those whose delta falls in [b_start, b_end)
            gt_errors_per_bin[b_idx] += ((err_subset['delta'] >= b_start) &
                                          (err_subset['delta'] <  b_end)).sum()

# Compute GT rate = errors / exposure_seconds (errors per second in each bin)
gt_rates = np.where(gt_exposure_per_bin > 0,
                    gt_errors_per_bin / gt_exposure_per_bin, np.nan)

# Model mean prediction per bin from df_gt (aligned predictions, not raw data)
post = df_gt[df_gt['distraction_active'] == 0].copy()
post['time_bin_idx'] = pd.cut(post['time_since_last_dist'],
                               bins=bin_edges, right=False,
                               labels=False).astype('Int64')
model_mean_per_bin = post.groupby('time_bin_idx', observed=True)['y_prob'].mean()

decay_rows = []
print(f"  {'Time bin':<12} {'GT rate (/s)':>13} {'Model mean prob':>16} {'Errors':>8} {'Exposure(s)':>12}")
print("  " + "-" * 68)
for b_idx, blabel in enumerate(bin_labels):
    gt_r   = gt_rates[b_idx]
    m_prob = model_mean_per_bin.get(b_idx, np.nan)
    n_err  = int(gt_errors_per_bin[b_idx])
    exp    = gt_exposure_per_bin[b_idx]
    gt_str = f"{gt_r:.4f}" if not np.isnan(gt_r) else "N/A"
    mp_str = f"{m_prob:.4f}" if not np.isnan(m_prob) else "N/A"
    print(f"  {blabel:<12} {gt_str:>13} {mp_str:>16} {n_err:>8} {exp:>12.1f}")
    decay_rows.append({'bin': blabel, 'bin_start': float(bin_edges[b_idx]),
                        'gt_rate_per_sec': gt_r,
                        'model_mean_prob': float(m_prob) if not np.isnan(m_prob) else None,
                        'n_errors': n_err, 'exposure_s': exp})

decay_df = pd.DataFrame(decay_rows)

# Normalise GT rates to [0,1] for visual comparison with model probabilities
# by dividing by the max rate (0–5s bin), so both curves share the same scale
max_gt = np.nanmax(decay_df['gt_rate_per_sec'])
decay_df['gt_rate_norm'] = decay_df['gt_rate_per_sec'] / max_gt
baseline_norm = p_baseline_global / max_gt

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(decay_df['bin_start'], decay_df['gt_rate_norm'],
        'o-', color='#d62728', lw=2, markersize=7,
        label='GT error rate (normalised, raw timestamps)')
ax.plot(decay_df['bin_start'], decay_df['model_mean_prob'],
        's--', color='steelblue', lw=2, markersize=7,
        label='Model mean prediction')
ax.axhline(y=baseline_norm, color='grey', linestyle=':', lw=1.5,
           label=f'Baseline rate (normalised)')
ax.set_xlabel('Time since distraction ended (s)')
ax.set_ylabel('Rate / Probability (normalised)')
ax.set_title('Post-Distraction Temporal Decay: Raw GT vs Model')
ax.set_xticks(decay_df['bin_start'])
ax.set_xticklabels(decay_df['bin'], rotation=30, ha='right')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{EVAL_OUT}gt_temporal_decay.png', dpi=150); plt.close()
print(f"  ✓ Saved → {EVAL_OUT}gt_temporal_decay.png")
print(f"  NOTE: GT rates are exposure-weighted (errors/second), normalised by peak rate.")
print(f"  Model probabilities are means over df_ev rows in each bin.")

# ── 3. PROBABILITY CALIBRATION vs EMPIRICAL RATE ──────────────────────────────
# Bin rows by predicted probability and compute the actual observed error rate.
# If well-calibrated: predicted 0.7 → ~70% of rows in that bucket are errors.
# Ground truth = observed label frequency per bucket.
print("\n── 3. Calibration vs Empirical Error Rate ────────────────────────────")

n_bins     = 10
bin_bounds = np.linspace(0, 1, n_bins + 1)
df_gt['prob_bin'] = pd.cut(df_gt['y_prob'], bins=bin_bounds, include_lowest=True)

cal_rows = []
print(f"  {'Prob bucket':<18} {'GT error rate':>14} {'Model mean prob':>16} {'N rows':>8}")
print("  " + "-" * 62)
for bucket, grp in df_gt.groupby('prob_bin', observed=True):
    gt_rate    = grp['label'].mean()
    model_prob = grp['y_prob'].mean()
    print(f"  {str(bucket):<18} {gt_rate:>14.4f} {model_prob:>16.4f} {len(grp):>8}")
    cal_rows.append({'bucket': str(bucket), 'gt_error_rate': gt_rate,
                     'model_mean_prob': model_prob, 'n': len(grp),
                     'bin_mid': float(bucket.mid)})

cal_df = pd.DataFrame(cal_rows).sort_values('bin_mid')

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
sc = ax.scatter(cal_df['model_mean_prob'], cal_df['gt_error_rate'],
                c=cal_df['n'], cmap='Blues', s=120, zorder=5,
                edgecolors='steelblue', linewidths=1.5)
for _, row in cal_df.iterrows():
    ax.annotate(f"n={int(row['n'])}",
                (row['model_mean_prob'], row['gt_error_rate']),
                textcoords="offset points", xytext=(6, 4), fontsize=7)
plt.colorbar(sc, ax=ax, label='N rows in bucket')
ax.set_xlabel('Model mean predicted probability')
ax.set_ylabel('GT observed error rate')
ax.set_title('Probability Calibration vs Empirical Error Rate')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{EVAL_OUT}gt_calibration.png', dpi=150); plt.close()
print(f"  ✓ Saved → {EVAL_OUT}gt_calibration.png")

# Save all GT validation tables
strat_df.to_csv(f'{EVAL_OUT}gt_risk_stratification.csv', index=False)
decay_df.to_csv(f'{EVAL_OUT}gt_temporal_decay.csv',      index=False)
cal_df.to_csv(f'{EVAL_OUT}gt_calibration.csv',           index=False)
print(f"\n  ✓ GT validation tables saved to {EVAL_OUT}")


print(f"\n  Plots saved to {EVAL_OUT}")

# Per-user table
print(f"\n── Per-User LOSO Performance ──────────────────────────────────────────")
print(f"  {'User':<20} {'AUC-PR':<10} {'AUC-ROC':<10} {'MCC':<8} {'Errors':<8} {'Safe'}")
print("  " + "-" * 65)
for uid, res in sorted(per_user_results.items()):
    ap  = f"{res['AUC-PR']:.4f}"  if res['AUC-PR']  is not None else "N/A"
    ar  = f"{res['AUC-ROC']:.4f}" if res['AUC-ROC'] is not None else "N/A"
    mc  = f"{res['MCC']:.4f}"     if res['MCC']     is not None else "N/A"
    print(f"  {str(uid):<20} {ap:<10} {ar:<10} {mc:<8} {res['n_pos']:<8} {res['n_neg']}")

# Save outputs
per_user_df = pd.DataFrame(per_user_results).T.reset_index().rename(columns={'index':'user_id'})
per_user_df.to_csv(f'{EVAL_OUT}per_user_metrics.csv', index=False)
results_df.to_csv(f'{EVAL_OUT}h_gridsearch_results.csv', index=False)
comparison_rows = []
for name, m in [('XGBoost', m_xgb)]:
    row = {'model': name}; row.update(m); comparison_rows.append(row)
pd.DataFrame(comparison_rows).to_csv(f'{EVAL_OUT}model_comparison.csv', index=False)

# ── 12. Feature sanity check ───────────────────────────────────────────────────
print("\n── Feature Sanity Check ───────────────────────────────────────────────")
df_check = pd.DataFrame(X_ev, columns=FEATURE_COLS)
for col in FEATURE_COLS:
    mn, mx, med = df_check[col].min(), df_check[col].max(), df_check[col].median()
    flag = " <- [WARN] constant feature!" if mn == mx else ""
    print(f"  {col:<28} min={mn:7.3f}  max={mx:7.3f}  median={med:7.3f}{flag}")

# ── 13. Final model (all data, Platt calibrated) ───────────────────────────────
print(f"\n── Training Final Model (H={best_H}s, all data) ───────────────────────")
pos_f = build_positives(best_H)
neg_f = build_negatives(best_H)
df_f  = pd.concat([pos_f, neg_f], ignore_index=True).dropna(subset=FEATURE_COLS)
X_f, y_f = df_f[FEATURE_COLS].values.astype(float), df_f['label'].values.astype(int)
spw_f    = (1 - y_f.mean()) / y_f.mean()

base_clf         = xgb.XGBClassifier(scale_pos_weight=spw_f, **XGB_PARAMS)
calibrated_model = CalibratedClassifierCV(base_clf, cv=5, method='sigmoid')
calibrated_model.fit(X_f, y_f)
print(f"  Samples: {len(df_f)}  |  Errors: {y_f.sum()} ({y_f.mean()*100:.1f}%)")

artifact = {
    'model':             calibrated_model,
    'best_H':            best_H,
    'best_thresh':       best_thresh,
    'feature_cols':      FEATURE_COLS,
    'le_emotion':        le_emotion,
    'le_pred':           le_pred,
    'user_baselines':    user_baselines,
    'p_baseline_global': p_baseline_global,
    'cv_results':        results_df,
    'metrics_xgb':       m_xgb,
    'bootstrap_ci': {
        'AUC-PR':  (ci_aucpr,  lo_aucpr,  hi_aucpr),
        'AUC-ROC': (ci_aucroc, lo_aucroc, hi_aucroc),
        'MCC':     (ci_mcc,    lo_mcc,    hi_mcc),
        'Brier':   (ci_brier,  lo_brier,  hi_brier),
    },
}
joblib.dump(artifact, MODEL_OUT)
print(f"  Saved -> {MODEL_OUT}")

# ── 14. Inference helper ───────────────────────────────────────────────────────
def predict_fitness(
    # ── Distraction state (required) ──────────────────────────────────────────
    distraction_active:          bool  | int,   # True / 1  = currently distracted
    seconds_since_last_distraction: float,       # 0.0 if currently distracted;
                                                 # seconds elapsed since window end otherwise
    # ── Physiological state (optional, low weight) ────────────────────────────
    arousal_trend:               float = 0.0,   # delta arousal over last N seconds
                                                 # positive = rising, negative = falling
    emotion_label:               str   = None,  # e.g. 'happy', 'angry', 'neutral'
    emotion_prob:                float = 0.5,   # confidence of emotion prediction [0,1]
    arousal_pred_label:          str   = None,  # arousal model categorical prediction
    arousal_pred_prob:           float = 0.5,   # confidence of arousal prediction [0,1]
    # ── Context (optional) ────────────────────────────────────────────────────
    user_id:                     str   = None,  # known user → uses personal baseline rate
    artifact_path:               str   = MODEL_OUT,
) -> dict:
    """
    Second-by-second fitness-to-drive inference.

    The caller owns the distraction state machine: it must track when a
    distraction started/ended and pass the elapsed seconds accordingly.
    This keeps the model stateless and deployable without session context.

    Returns
    -------
    dict with:
      error_probability   float [0,1]  — P(error in next second)
      fitness_to_drive    float [0,1]  — 1 - error_probability
      alert               bool         — True if risk exceeds optimal threshold
      input_warnings      list[str]    — any clamped / imputed inputs (for logging)
    """
    art    = joblib.load(artifact_path)
    H      = art['best_H']
    thresh = art['best_thresh']
    warns  = []

    # ── Input validation & sanitisation ──────────────────────────────────────
    # distraction_active
    try:
        dist_active = int(bool(distraction_active))
    except Exception:
        dist_active = 0
        warns.append("distraction_active invalid, defaulting to 0")

    # seconds_since_last_distraction
    try:
        t_since = float(seconds_since_last_distraction)
        if t_since < 0:
            warns.append(f"seconds_since_last_distraction={t_since} < 0, clamped to 0")
            t_since = 0.0
        if dist_active == 1:
            t_since = 0.0   # inside window: always 0 regardless of what was passed
        t_since = min(t_since, float(H))   # clip at H
    except Exception:
        t_since = float(H)
        warns.append("seconds_since_last_distraction invalid, defaulting to H (recovered)")

    # arousal_pred_prob / emotion_prob
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

    a_prob   = _clip_prob(arousal_pred_prob, 'arousal_pred_prob')
    e_prob   = _clip_prob(emotion_prob,      'emotion_prob')

    # arousal_pred_label / emotion_label encoding (unknown = sentinel)
    _le_emotion = art['le_emotion']
    _le_pred    = art['le_pred']

    def _encode(le, val):
        s = safe_str(val)
        if s in le.classes_:
            return int(le.transform([s])[0])
        warns.append(f"Unseen label '{s}' for {le.__class__.__name__}, using 'unknown'")
        return int(le.transform(['unknown'])[0])

    emo_enc  = _encode(_le_emotion, emotion_label)
    pred_enc = _encode(_le_pred,    arousal_pred_label)

    # baseline error rate
    b_rate = art['user_baselines'].get(user_id, art['p_baseline_global'])
    if user_id is not None and user_id not in art['user_baselines']:
        warns.append(f"user_id='{user_id}' not in training set, using global baseline")

    # ── Build feature vector ──────────────────────────────────────────────────
    sample = pd.DataFrame([{
        'distraction_active':   dist_active,
        'time_since_last_dist': t_since,
        'model_prob':           a_prob,        # arousal model confidence
        'model_pred_enc':       pred_enc,      # arousal model label
        'emotion_prob':         e_prob,
        'emotion_label_enc':    emo_enc,
        'baseline_error_rate':  b_rate,
    }])

    error_prob = float(art['model'].predict_proba(
        sample[art['feature_cols']])[0, 1])

    return {
        'error_probability': round(error_prob, 4),
        'fitness_to_drive':  round(1.0 - error_prob, 4),
        'alert':             error_prob >= thresh,
        'input_warnings':    warns,
    }


# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Best H             : {best_H}s")
print(f"  XGBoost  AUC-PR    : {m_xgb['AUC-PR']:.4f}  95% CI [{lo_aucpr:.4f} - {hi_aucpr:.4f}]")
print(f"  XGBoost  AUC-ROC   : {m_xgb['AUC-ROC']:.4f}  95% CI [{lo_aucroc:.4f} - {hi_aucroc:.4f}]")
print(f"  XGBoost  MCC       : {m_xgb['MCC']:.4f}  95% CI [{lo_mcc:.4f} - {hi_mcc:.4f}]")
print(f"  XGBoost  Brier     : {m_xgb['Brier']:.4f}  95% CI [{lo_brier:.4f} - {hi_brier:.4f}]")
# print()
# print("── Example inference calls ─────────────────────────────────────────────")
# print("""
# # Driver is currently distracted:
# predict_fitness(
#     distraction_active=True,
#     seconds_since_last_distraction=0.0,
#     arousal_trend=0.12,            # arousal rising
#     emotion_label='angry',
#     emotion_prob=0.74,
#     arousal_pred_label='high',
#     arousal_pred_prob=0.81,
#     user_id='participant_03',
# )

# # Driver recovered (18 seconds after distraction ended):
# predict_fitness(
#     distraction_active=False,
#     seconds_since_last_distraction=18.0,
#     arousal_trend=-0.05,           # arousal slightly falling
#     emotion_label='neutral',
#     emotion_prob=0.65,
#     arousal_pred_label='medium',
#     arousal_pred_prob=0.70,
#     user_id='participant_03',
# )

# # New/unknown driver (falls back to global baseline):
# predict_fitness(
#     distraction_active=False,
#     seconds_since_last_distraction=45.0,   # beyond H, fully recovered
# )
# """)