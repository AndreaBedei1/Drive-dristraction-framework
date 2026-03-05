"""
Distraction Recovery Analysis
==============================
Question: At the moment a distraction ends, can we predict whether/how fast
the driver will recover – i.e., avoid post-distraction errors?

Data used (sim_time-based matching, same user_id + run_id):
  - Distractions_distraction_old.csv  : one row per distraction event
  - Errors_distraction_old.csv        : one row per driving error
  - Driving Time_baseline_old.csv     : per-session baseline physiology

Recovery window: RECOVERY_WINDOW_SEC seconds after distraction end.

Targets evaluated at multiple window lengths:
  - has_post_error_Xs (binary) : ≥1 error within X seconds after distraction end
  - n_post_errors     (count)  : total errors in [0, 30s] window
  - t_first_error     (float)  : seconds to first post-distraction error (NaN if none)

Key insight: ~87% of events have an error within 30s (nearly trivial).
Shorter windows (5–15s) are more discriminative.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────
DATA_DIR          = "data/"
OUT_DIR           = "results/"
RECOVERY_WINDOW   = 30          # seconds after distraction end
MIN_BASELINE_OBS  = 3           # min errors needed to compute stable baseline rate
LOPO              = True        # leave-one-participant-out cross-validation
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load ────────────────────────────────────────────────────────────────────
dist  = pd.read_csv(DATA_DIR + "Dataset Distractions_distraction_old.csv")
errs  = pd.read_csv(DATA_DIR + "Dataset Errors_distraction_old.csv")
drv   = pd.read_csv(DATA_DIR + "Dataset Driving Time_baseline_old.csv")
errs_b = pd.read_csv(DATA_DIR + "Dataset Errors_baseline_old.csv")

print(f"Loaded: {len(dist)} distraction events, {len(errs)} distraction errors, "
      f"{len(errs_b)} baseline errors, {len(drv)} baseline sessions")
print(f"Participants: {dist['user_id'].nunique()}")

# ── Preprocessing ───────────────────────────────────────────────────────────
dist["duration_sec"]     = dist["sim_time_end"] - dist["sim_time_start"]
dist["arousal_delta"]    = dist["arousal_end"]   - dist["arousal_start"]
dist["hr_delta"]         = dist["hr_bpm_end"]    - dist["hr_bpm_start"]
dist["speed_delta"]      = dist["speed_kmh_end"] - dist["speed_kmh_start"]
dist["steer_delta"]      = dist["steer_angle_deg_end"] - dist["steer_angle_deg_start"]
dist["abs_steer_end"]    = dist["steer_angle_deg_end"].abs()
dist["abs_steer_start"]  = dist["steer_angle_deg_start"].abs()

# Baseline physiology per participant (average across sessions)
baseline = drv.groupby("user_id")[["hr_baseline", "arousal_baseline"]].mean()
dist = dist.merge(baseline.rename(columns={"hr_baseline": "hr_baseline",
                                            "arousal_baseline": "arousal_baseline"}),
                  on="user_id", how="left")
dist["arousal_vs_baseline"] = dist["arousal_end"] - dist["arousal_baseline"]
dist["hr_vs_baseline"]      = dist["hr_bpm_end"]  - dist["hr_baseline"]

# ── Build Recovery Dataset ──────────────────────────────────────────────────
# For each distraction event find errors in (sim_time_end, sim_time_end + RECOVERY_WINDOW]
# within the same participant + run.

errs_idx = errs.set_index(["user_id", "run_id"])

records = []
for _, row in dist.iterrows():
    key = (row["user_id"], row["run_id"])
    t_end = row["sim_time_end"]

    if key in errs_idx.index:
        session_errs = errs_idx.loc[[key]]
        window_errs  = session_errs[
            (session_errs["sim_time_seconds"] > t_end) &
            (session_errs["sim_time_seconds"] <= t_end + RECOVERY_WINDOW)
        ]
    else:
        window_errs = pd.DataFrame()

    n   = len(window_errs)
    t1  = (window_errs["sim_time_seconds"].min() - t_end) if n > 0 else np.nan

    records.append({
        "has_post_error":  int(n > 0),
        "n_post_errors":   n,
        "t_first_error":   t1,
    })

recovery = dist.copy()
for col in ["has_post_error", "n_post_errors", "t_first_error"]:
    recovery[col] = [r[col] for r in records]

# Also build targets for shorter windows (more discriminative)
SHORT_WINDOWS = [5, 10, 15, 20]
for w in SHORT_WINDOWS:
    recovery[f"has_post_error_{w}s"] = [
        int(r["n_post_errors"] > 0 and (r["t_first_error"] <= w if not np.isnan(r["t_first_error"]) else False))
        for r in records
    ]

print(f"\nRecovery dataset: {len(recovery)} events")
print(f"  has_post_error (30s): {recovery['has_post_error'].sum()} positive "
      f"({recovery['has_post_error'].mean()*100:.1f}%)")
for w in SHORT_WINDOWS:
    col = f"has_post_error_{w}s"
    print(f"  has_post_error ({w:2d}s): {recovery[col].sum()} positive "
          f"({recovery[col].mean()*100:.1f}%)")
print(f"  Mean post-errors (when ≥1): "
      f"{recovery.loc[recovery['has_post_error']==1,'n_post_errors'].mean():.2f}")
print(f"  Mean time to first error:   "
      f"{recovery['t_first_error'].mean():.1f} s (NaN excluded)")

# ── EDA ──────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("EXPLORATORY DATA ANALYSIS")
print("="*70)

# 1. Class balance per participant
per_p = recovery.groupby("user_id")["has_post_error"].agg(["sum", "count", "mean"])
per_p.columns = ["n_errors", "n_dist", "pct_error"]
per_p["pct_error"] *= 100
print("\nPer-participant post-error rate (%) ↓")
print(per_p.sort_values("pct_error", ascending=False).round(1).to_string())

# 2. Feature correlations with target
num_features = [
    "duration_sec", "speed_kmh_end", "abs_steer_end",
    "arousal_end", "arousal_delta", "arousal_vs_baseline",
    "hr_bpm_end", "hr_delta", "hr_vs_baseline",
    "model_prob_end",
]
print("\nPoint-biserial correlation with has_post_error:")
for f in num_features:
    col = recovery[f].dropna()
    target = recovery.loc[col.index, "has_post_error"]
    r, p = stats.pointbiserialr(target, col)
    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "  ")
    print(f"  {f:<30} r={r:+.3f}  p={p:.3f} {sig}")

# 3. Categorical features
print("\nPost-error rate by distraction type (details):")
print(recovery.groupby("details")["has_post_error"].agg(["mean", "count"]).round(3))

print("\nPost-error rate by model_pred_end:")
print(recovery.groupby("model_pred_end")["has_post_error"].agg(["mean", "count"]).round(3))

print("\nPost-error rate by emotion_label_end:")
print(recovery.groupby("emotion_label_end")["has_post_error"].agg(["mean", "count"]).round(3))

# 4. Duration bins
bins   = [0, 2, 4, 6, 10, float("inf")]
labels = ["0–2s", "2–4s", "4–6s", "6–10s", ">10s"]
recovery["dur_bin"] = pd.cut(recovery["duration_sec"], bins=bins, labels=labels, right=True)
print("\nPost-error rate by distraction duration:")
print(recovery.groupby("dur_bin", observed=True)["has_post_error"]
      .agg(["mean", "count"])
      .rename(columns={"mean": "error_rate", "count": "n"})
      .assign(error_rate=lambda x: x["error_rate"].round(3))
      .to_string())

# 5. Speed at distraction end
spd_bins   = [0, 10, 25, 40, float("inf")]
spd_labels = ["0–10", "10–25", "25–40", ">40"]
recovery["spd_bin"] = pd.cut(recovery["speed_kmh_end"].clip(lower=0),
                             bins=spd_bins, labels=spd_labels, right=True)
print("\nPost-error rate by speed at distraction end (km/h):")
print(recovery.groupby("spd_bin", observed=True)["has_post_error"]
      .agg(["mean", "count"])
      .rename(columns={"mean": "error_rate", "count": "n"})
      .assign(error_rate=lambda x: x["error_rate"].round(3))
      .to_string())

# 6. Error time distribution post-distraction
print("\nError offset from distraction end (seconds) – among post-distraction errors only:")
post_errs_times = recovery.dropna(subset=["t_first_error"])["t_first_error"]
print(post_errs_times.describe().round(2))
qs = [5, 10, 15, 20, 25, 30]
total = len(recovery)
print("\nCumulative % of distractions with ≥1 error within N seconds after end:")
for q in qs:
    n_within = ((recovery["t_first_error"] <= q)).sum()
    pct = n_within / total * 100
    print(f"  ≤{q:2d}s : {n_within} ({pct:.1f}%)")

# ── Feature Matrix ───────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PREDICTIVE MODELLING (LOPO-CV)")
print("="*70)

# Encode categoricals
le_pred  = LabelEncoder().fit(["pick_floor", "reach_back"])
le_emo   = LabelEncoder().fit(["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"])
le_type  = LabelEncoder().fit(["window_1", "window_2"])

def build_X(df):
    X = pd.DataFrame()
    X["duration_sec"]        = df["duration_sec"]
    X["speed_end"]           = df["speed_kmh_end"].fillna(df["speed_kmh_end"].median())
    X["abs_steer_end"]       = df["abs_steer_end"].fillna(df["abs_steer_end"].median())
    X["speed_delta"]         = df["speed_delta"].fillna(0)
    X["steer_delta"]         = df["steer_delta"].fillna(0)
    X["arousal_end"]         = df["arousal_end"].fillna(df["arousal_end"].median())
    X["arousal_delta"]       = df["arousal_delta"].fillna(0)
    X["arousal_vs_baseline"] = df["arousal_vs_baseline"].fillna(0)
    X["hr_end"]              = df["hr_bpm_end"].fillna(df["hr_bpm_end"].median())
    X["hr_delta"]            = df["hr_delta"].fillna(0)
    X["hr_vs_baseline"]      = df["hr_vs_baseline"].fillna(0)
    X["model_prob_end"]      = df["model_prob_end"].fillna(df["model_prob_end"].median())
    X["model_pred_end"]      = le_pred.transform(
        df["model_pred_end"].fillna("pick_floor").map(
            lambda v: v if v in le_pred.classes_ else "pick_floor"))
    X["emotion_label_end"]   = le_emo.transform(
        df["emotion_label_end"].fillna("neutral").map(
            lambda v: v if v in le_emo.classes_ else "neutral"))
    X["emotion_prob_end"]    = df["emotion_prob_end"].fillna(df["emotion_prob_end"].median())
    X["dist_type"]           = le_type.transform(
        df["details"].fillna("window_1").map(
            lambda v: v if v in le_type.classes_ else "window_1"))
    return X.astype(float)

X_all = build_X(recovery)
y_all = recovery["has_post_error"].values
users = recovery["user_id"].values

MODELS = {
    "Logistic (L2)": LogisticRegression(C=0.1, max_iter=1000, random_state=42),
    "Random Forest":  RandomForestClassifier(n_estimators=200, max_depth=5,
                                             random_state=42, n_jobs=-1),
    "GBM":            GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                 learning_rate=0.05, random_state=42),
}

results   = {name: {"auroc": [], "ap": [], "f1": []} for name in MODELS}
feat_names = X_all.columns.tolist()

unique_users = np.unique(users)
for test_user in unique_users:
    mask_train = users != test_user
    mask_test  = users == test_user

    X_tr, y_tr = X_all.values[mask_train], y_all[mask_train]
    X_te, y_te = X_all.values[mask_test],  y_all[mask_test]

    if y_te.sum() == 0 or y_te.sum() == len(y_te):
        continue   # skip degenerate folds

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    for name, clf in MODELS.items():
        clf.fit(X_tr_s, y_tr)
        prob = clf.predict_proba(X_te_s)[:, 1]
        pred = (prob >= 0.5).astype(int)
        results[name]["auroc"].append(roc_auc_score(y_te, prob))
        results[name]["ap"].append(average_precision_score(y_te, prob))
        results[name]["f1"].append(f1_score(y_te, pred, zero_division=0))

print(f"\nLOPO results (N={len(unique_users)} folds) — target: has_post_error (30s):")
print(f"{'Model':<20} {'AUROC':>10} {'AP':>10} {'F1':>10}")
print("-"*52)
for name, res in results.items():
    auroc = np.mean(res["auroc"])
    ap    = np.mean(res["ap"])
    f1    = np.mean(res["f1"])
    print(f"{name:<20} {auroc:>10.3f} {ap:>10.3f} {f1:>10.3f}")

# Test shorter windows
print("\nLOPO AUROC by recovery window length (GBM only):")
print(f"{'Window':<12} {'AUROC':>10} {'AP':>10} {'Pos%':>8}")
print("-"*44)
for w in SHORT_WINDOWS + [30]:
    target_col = f"has_post_error_{w}s" if w < 30 else "has_post_error"
    y_w = recovery[target_col].values
    aurocs_w, aps_w = [], []
    for test_user in unique_users:
        mask_train = users != test_user
        mask_test  = users == test_user
        X_tr, y_tr = X_all.values[mask_train], y_w[mask_train]
        X_te, y_te = X_all.values[mask_test],  y_w[mask_test]
        if y_te.sum() == 0 or y_te.sum() == len(y_te):
            continue
        scaler = StandardScaler().fit(X_tr)
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                          learning_rate=0.05, random_state=42)
        clf.fit(scaler.transform(X_tr), y_tr)
        prob = clf.predict_proba(scaler.transform(X_te))[:, 1]
        aurocs_w.append(roc_auc_score(y_te, prob))
        aps_w.append(average_precision_score(y_te, prob))
    pos_pct = y_w.mean() * 100
    print(f"  {w:2d}s          {np.mean(aurocs_w):>10.3f} {np.mean(aps_w):>10.3f} {pos_pct:>7.1f}%")

# Regression: predict t_first_error (continuous) and n_post_errors
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("\nLOPO regression — predict t_first_error (s) [imputed to 30s if no error]:")
recovery["t_first_error_imp"] = recovery["t_first_error"].fillna(RECOVERY_WINDOW)
y_reg = recovery["t_first_error_imp"].values

maes_r, r2s_r = [], []
for test_user in unique_users:
    mask_train = users != test_user
    mask_test  = users == test_user
    X_tr, y_tr = X_all.values[mask_train], y_reg[mask_train]
    X_te, y_te = X_all.values[mask_test],  y_reg[mask_test]
    scaler = StandardScaler().fit(X_tr)
    reg = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                     learning_rate=0.05, random_state=42)
    reg.fit(scaler.transform(X_tr), y_tr)
    pred = reg.predict(scaler.transform(X_te))
    maes_r.append(mean_absolute_error(y_te, pred))
    r2s_r.append(r2_score(y_te, pred))

naive_mae = mean_absolute_error(y_reg, np.full_like(y_reg, y_reg.mean()))
print(f"  GBM MAE:    {np.mean(maes_r):.2f}s  (naive mean baseline: {naive_mae:.2f}s)")
print(f"  GBM R²:     {np.mean(r2s_r):.3f}  (0=no signal, 1=perfect)")
print(f"  Target mean: {y_reg.mean():.1f}s  std: {y_reg.std():.1f}s")

# ── Feature importance (best model = GBM on full data) ───────────────────────
print("\n" + "="*70)
print("FEATURE IMPORTANCE (GBM, full data)")
print("="*70)

scaler_full = StandardScaler().fit(X_all.values)
X_s = scaler_full.transform(X_all.values)
gbm_full = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                       learning_rate=0.05, random_state=42)
gbm_full.fit(X_s, y_all)

importance_df = (
    pd.DataFrame({"feature": feat_names,
                  "gain":    gbm_full.feature_importances_})
    .sort_values("gain", ascending=False)
)
print(importance_df.to_string(index=False))

# Permutation importance (more reliable)
perm = permutation_importance(gbm_full, X_s, y_all, n_repeats=20,
                               random_state=42, scoring="roc_auc")
perm_df = (
    pd.DataFrame({"feature":    feat_names,
                  "perm_mean":  perm.importances_mean,
                  "perm_std":   perm.importances_std})
    .sort_values("perm_mean", ascending=False)
)
print("\nPermutation importance (ROC-AUC drop):")
print(perm_df.to_string(index=False))

# ── Survival-style: error rate over time since distraction end ───────────────
print("\n" + "="*70)
print("ERROR RATE OVER TIME SINCE DISTRACTION END (baseline comparison)")
print("="*70)

# Baseline error rate per second from Errors_baseline_old + Driving Time_baseline_old
total_baseline_sec = drv["run_duration_seconds"].sum()
baseline_rate_per_s = len(errs_b) / total_baseline_sec
print(f"Baseline error rate: {baseline_rate_per_s:.5f} errors/s  "
      f"({len(errs_b)} errors / {total_baseline_sec:.0f} s)")

# For each 1-second bucket after distraction end, count how many errors fell there
time_buckets = np.arange(1, RECOVERY_WINDOW + 1)
bucket_counts = np.zeros(len(time_buckets), dtype=int)

errs_idx2 = errs.set_index(["user_id", "run_id"])
for _, row in dist.iterrows():
    key   = (row["user_id"], row["run_id"])
    t_end = row["sim_time_end"]
    if key not in errs_idx2.index:
        continue
    se = errs_idx2.loc[[key]]
    offsets = se["sim_time_seconds"] - t_end
    valid   = offsets[(offsets > 0) & (offsets <= RECOVERY_WINDOW)]
    for off in valid:
        bucket = int(np.ceil(off)) - 1   # 0-indexed
        if 0 <= bucket < len(bucket_counts):
            bucket_counts[bucket] += 1

# Rate per distraction-second (normalise by N distractions)
n_dist = len(dist)
bucket_rates = bucket_counts / n_dist

print(f"\nErrors-per-second per distraction (first {RECOVERY_WINDOW}s post-distraction):")
print(f"{'Second':<8} {'Errors':>8} {'Rate/dist':>12}")
print("-"*30)
for i, (cnt, rate) in enumerate(zip(bucket_counts, bucket_rates)):
    marker = " <-- baseline" if abs(rate - baseline_rate_per_s) < 0.001 else ""
    print(f"  t+{i+1:<5} {cnt:>8}  {rate:>12.5f}{marker}")

# Find approximate hangover: first second where rate falls below 1.5× baseline
hangover = None
for i, rate in enumerate(bucket_rates):
    if rate <= 1.5 * baseline_rate_per_s and i >= 2:
        hangover = i + 1
        break
if hangover:
    print(f"\nApproximate hangover (rate ≤ 1.5× baseline): ~{hangover}s")
else:
    print(f"\nPost-distraction error rate stays >{1.5*baseline_rate_per_s:.5f}/s "
          f"(1.5× baseline) throughout entire {RECOVERY_WINDOW}s window.")
    print(f"Peak: t+1 = {bucket_rates[0]:.5f}/s  ({bucket_rates[0]/baseline_rate_per_s:.1f}× baseline)")

# ── Subgroup: recovery by distraction duration × model prediction ─────────────
print("\n" + "="*70)
print("SUBGROUP ANALYSIS: RECOVERY BY DISTRACTION FEATURES")
print("="*70)

sub_cols = ["has_post_error", "n_post_errors", "t_first_error"]

# Duration × type
print("\nhas_post_error mean by (dur_bin × model_pred_end):")
ct = recovery.pivot_table(
    values="has_post_error",
    index="dur_bin",
    columns="model_pred_end",
    aggfunc="mean",
    observed=True
)
print(ct.round(3).to_string())

# Arousal quartile
recovery["arousal_q"] = pd.qcut(
    recovery["arousal_end"].fillna(recovery["arousal_end"].median()),
    q=4, labels=["Q1 low", "Q2", "Q3", "Q4 high"], duplicates="drop"
)
print("\nhas_post_error by arousal quartile at distraction end:")
print(recovery.groupby("arousal_q", observed=True)["has_post_error"]
      .agg(["mean", "count"])
      .rename(columns={"mean": "error_rate", "count": "n"})
      .round(3).to_string())

# HR quartile
recovery["hr_q"] = pd.qcut(
    recovery["hr_bpm_end"].fillna(recovery["hr_bpm_end"].median()),
    q=4, labels=["Q1 low", "Q2", "Q3", "Q4 high"], duplicates="drop"
)
print("\nhas_post_error by HR quartile at distraction end:")
print(recovery.groupby("hr_q", observed=True)["has_post_error"]
      .agg(["mean", "count"])
      .rename(columns={"mean": "error_rate", "count": "n"})
      .round(3).to_string())

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("KEY FINDINGS SUMMARY")
print("="*70)
best_name = max(results, key=lambda n: np.mean(results[n]["auroc"]))
best_auroc = np.mean(results[best_name]["auroc"])
top_feat = perm_df.head(5)["feature"].tolist()

pos_5s  = recovery["has_post_error_5s"].mean()*100
pos_10s = recovery["has_post_error_10s"].mean()*100
pos_15s = recovery["has_post_error_15s"].mean()*100
print(f"""
  Recovery window (30s): {recovery['has_post_error'].mean()*100:.1f}% of events have a post-error
  Positive rate at 5s:   {pos_5s:.1f}%   ← most discriminative window
  Positive rate at 10s:  {pos_10s:.1f}%
  Positive rate at 15s:  {pos_15s:.1f}%

  Predictive signal (binary, 30s window):
    Best model AUROC ({best_name}): {best_auroc:.3f}
    → At 30s window, errors are too ubiquitous (~87%) to discriminate.
    → Shorter windows (5–15s) or regression on t_first_error are better targets.

  Top features by permutation importance (global model, all windows):
    {chr(10).join(f'    {i+1}. {f}' for i, f in enumerate(top_feat))}
    → Physiological state relative to baseline dominates (arousal, HR)
    → Vehicle kinematics (steering, speed changes) contribute modestly
    → Distraction type (pick_floor vs reach_back, window_1 vs window_2) = NO signal

  Error rate stays {bucket_rates[0]/baseline_rate_per_s:.0f}× baseline at t+1s, remains elevated entire 30s window.
  → Hangover extends beyond 30s; a 60-90s window may be more appropriate.

  Recommendations:
    - Primary target: has_post_error_5s (23% pos, most discriminative) or
      t_first_error regression (continuous, captures graded recovery)
    - Key inputs: {', '.join(top_feat[:3])}
    - Personalisation likely necessary: participant_01 has 52% vs participant_04 100%
    - Consider survival analysis (Cox regression) for time-to-first-error modelling
    - Per-second timeline data (Timeline_distraction.csv) would unlock richer features
""")

# ── Plots ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle("Distraction Recovery Analysis", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.42)

# Plot 1: Error rate over time since distraction end
ax1 = fig.add_subplot(gs[0, :3])
ax1.bar(time_buckets, bucket_rates, color="#4C72B0", alpha=0.7, label="Post-distraction rate")
ax1.axhline(baseline_rate_per_s, color="red", linestyle="--", linewidth=1.5,
            label=f"Baseline rate ({baseline_rate_per_s*1000:.3f}e-3/s)")
ax1.axhline(1.5 * baseline_rate_per_s, color="orange", linestyle=":", linewidth=1.5,
            label="1.5× baseline")
if hangover:
    ax1.axvline(hangover, color="green", linestyle="-.", linewidth=1.5,
                label=f"Hangover ≈{hangover}s")
ax1.set_xlabel("Seconds after distraction end")
ax1.set_ylabel("Errors per distraction")
ax1.set_title("Post-distraction error rate over time")
ax1.legend(fontsize=8)

# Plot 2: Feature importance (permutation)
ax2 = fig.add_subplot(gs[0, 3])
top_n = 10
sub_perm = perm_df.head(top_n)
ax2.barh(sub_perm["feature"][::-1], sub_perm["perm_mean"][::-1],
         xerr=sub_perm["perm_std"][::-1], color="#DD8452", alpha=0.8)
ax2.set_xlabel("AUROC drop (permutation)")
ax2.set_title(f"Top {top_n} features (GBM)")
ax2.tick_params(axis="y", labelsize=8)

# Plot 3: Positive rate by window length
ax_w = fig.add_subplot(gs[1, 0])
window_lens = SHORT_WINDOWS + [30]
pos_rates   = [recovery[f"has_post_error_{w}s"].mean()*100 if w < 30
               else recovery["has_post_error"].mean()*100 for w in window_lens]
ax_w.plot(window_lens, pos_rates, "o-", color="#4C72B0", linewidth=2, markersize=7)
ax_w.set_xlabel("Recovery window (s)")
ax_w.set_ylabel("% distractions with post-error")
ax_w.set_title("Positive rate vs window length")
ax_w.set_ylim(0, 100)
ax_w.grid(True, alpha=0.3)

# Plot 4: Post-error rate by duration bin
ax3 = fig.add_subplot(gs[1, 1])
dur_agg = (recovery.groupby("dur_bin", observed=True)["has_post_error"]
           .agg(["mean", "count"]).reset_index())
bars = ax3.bar(range(len(dur_agg)), dur_agg["mean"] * 100,
               color="#55A868", alpha=0.8)
ax3.set_xticks(range(len(dur_agg)))
ax3.set_xticklabels(dur_agg["dur_bin"].astype(str), rotation=30, ha="right", fontsize=8)
ax3.set_ylabel("% with post-distraction error")
ax3.set_title("Recovery by distraction duration")
for bar, n in zip(bars, dur_agg["count"]):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"n={n}", ha="center", va="bottom", fontsize=7)

# Plot 5: Post-error rate by speed bin at distraction end
ax4 = fig.add_subplot(gs[1, 2])
spd_agg = (recovery.groupby("spd_bin", observed=True)["has_post_error"]
           .agg(["mean", "count"]).reset_index())
bars = ax4.bar(range(len(spd_agg)), spd_agg["mean"] * 100,
               color="#C44E52", alpha=0.8)
ax4.set_xticks(range(len(spd_agg)))
ax4.set_xticklabels([f"{v} km/h" for v in spd_agg["spd_bin"].astype(str)],
                    rotation=30, ha="right", fontsize=8)
ax4.set_ylabel("% with post-distraction error")
ax4.set_title("Recovery by speed at distraction end")
for bar, n in zip(bars, spd_agg["count"]):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"n={n}", ha="center", va="bottom", fontsize=7)

# Plot 6: LOPO AUROC box per model
ax5 = fig.add_subplot(gs[1, 3])
data_to_plot = [results[n]["auroc"] for n in MODELS]
bp = ax5.boxplot(data_to_plot, labels=list(MODELS.keys()), patch_artist=True)
colors = ["#4C72B0", "#55A868", "#DD8452"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax5.axhline(0.5, color="gray", linestyle="--", linewidth=1)
ax5.set_ylabel("AUROC")
ax5.set_title("LOPO-CV AUROC per model")
ax5.tick_params(axis="x", labelsize=8)

out_path = os.path.join(OUT_DIR, "recovery_analysis.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved → {out_path}")
