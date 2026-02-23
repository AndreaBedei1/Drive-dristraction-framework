"""
Self‑contained Cox Time‑Varying Hazard Model
=============================================
Compares with the XGBoost pipeline using the same data.
"""

import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, average_precision_score

# ==============================================================
# 1) FILE PATHS  — set these to your real files
# ==============================================================
DISTRACTION_EVENTS_CSV = "data/Dataset Distractions_distraction.csv"
ERRORS_DISTRACTION_CSV = "data/Dataset Errors_distraction.csv"
ERRORS_BASELINE_CSV    = "data/Dataset Errors_baseline.csv"

# ==============================================================
# 2) LOAD RAW DATA
# ==============================================================
dist = pd.read_csv(DISTRACTION_EVENTS_CSV, parse_dates=["timestamp_start", "timestamp_end"])
err_d = pd.read_csv(ERRORS_DISTRACTION_CSV, parse_dates=["timestamp"])
err_b = pd.read_csv(ERRORS_BASELINE_CSV, parse_dates=["timestamp"])

errors = pd.concat([err_d, err_b], ignore_index=True)

# ==============================================================
# 3) BUILD PER‑SECOND TIMELINE FOR EACH USER/RUN
# ==============================================================
def build_timeline(group):
    """Create a second‑by‑second timeline for one user/run."""
    t0 = group["timestamp"].min().floor("s")
    t1 = group["timestamp"].max().ceil("s")
    timeline = pd.DataFrame({
        "timestamp": pd.date_range(t0, t1, freq="s")
    })
    timeline["user_id"] = group["user_id"].iloc[0]
    timeline["run_id"]  = group["run_id"].iloc[0]
    return timeline

# Build the base timeline from all error timestamps
base = (
    errors
    .groupby(["user_id", "run_id"], group_keys=False)
    .apply(build_timeline)
    .reset_index(drop=True)
)

# Mark seconds that contain an error
err_sec = errors.copy()
err_sec["timestamp"] = err_sec["timestamp"].dt.floor("s")
err_sec["label"] = 1

base = base.merge(
    err_sec[["user_id", "run_id", "timestamp", "label"]],
    on=["user_id", "run_id", "timestamp"],
    how="left"
)
base["label"] = base["label"].fillna(0).astype(int)

# ==============================================================
# 4) ADD DISTRACTION FEATURES
# ==============================================================
dist = dist.sort_values("timestamp_end")

# Pre‑compute lists of window end timestamps for each (user, run)
window_ends = (
    dist.groupby(["user_id", "run_id"])["timestamp_end"]
    .apply(list)
    .to_dict()
)

def get_density(uid, rid, ts, lookback_s):
    """Number of distraction windows ending in [ts-lookback_s, ts)."""
    ends = window_ends.get((uid, rid), [])
    if not ends:
        return 0
    lo = ts - pd.Timedelta(seconds=lookback_s)
    return sum((e >= lo) and (e < ts) for e in ends)

def compute_time_since_last(uid, rid, ts):
    """Seconds since the last distraction window ended (before ts)."""
    ends = window_ends.get((uid, rid), [])
    past = [e for e in ends if e < ts]
    if not past:
        return 1e6   # large sentinel (effectively infinite)
    return (ts - past[-1]).total_seconds()

base["time_since_last_dist"] = base.apply(
    lambda r: compute_time_since_last(r.user_id, r.run_id, r.timestamp), axis=1
)

base["dist_density_30"] = base.apply(
    lambda r: get_density(r.user_id, r.run_id, r.timestamp, 30), axis=1
)

# Cognitive decay with a fixed H – you may also cross‑validate H
H = 13.0
base["cognitive_load_decay"] = np.exp(-base["time_since_last_dist"] / H)

# ==============================================================
# 5) COX TIME‑VARYING FORMAT (per‑user cumulative seconds)
# ==============================================================
base = base.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

# Create start/stop as cumulative seconds within each user
base["start"] = base.groupby("user_id").cumcount().astype(float)
base["stop"]  = base["start"] + 1.0
base["event"] = base["label"]

FEATURES = [
    "time_since_last_dist",
    "dist_density_30",
    "cognitive_load_decay"
]

df = base[FEATURES + ["start", "stop", "event", "user_id"]].copy()
for c in FEATURES:
    df[c] = df[c].astype(float)

# ==============================================================
# 6) LEAVE‑ONE‑USER‑OUT CROSS‑VALIDATION
# ==============================================================
logo = LeaveOneGroupOut()

aucs = []
aprs = []

print("\n" + "="*70)
print("COX TIME-VARYING HAZARD MODEL (LOUO)")
print("="*70)

fold = 1
for tr, te in logo.split(df, groups=df["user_id"]):
    train = df.iloc[tr]
    test  = df.iloc[te]

    ctv = CoxTimeVaryingFitter(penalizer=0.01)  # small L2 penalty for stability
    ctv.fit(
        train,
        id_col="user_id",
        start_col="start",
        stop_col="stop",
        event_col="event",
        show_progress=False
    )

    risk = ctv.predict_partial_hazard(test).values.ravel()
    y_true = test["event"].values

    # In case of constant predictions (rare), skip fold
    if len(np.unique(risk)) < 2 or y_true.sum() == 0:
        print(f"Fold {fold}: degenerate, skipping")
        continue

    auc = roc_auc_score(y_true, risk)
    apr = average_precision_score(y_true, risk)

    print(f"\nFold {fold}")
    print(f"AUC-ROC : {auc:.4f}")
    print(f"AUC-PR  : {apr:.4f}")

    aucs.append(auc)
    aprs.append(apr)
    fold += 1

print("\n" + "-"*60)
print("SUMMARY")
print("-"*60)
if len(aucs) > 0:
    print(f"Mean AUC-ROC : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"Mean AUC-PR  : {np.mean(aprs):.4f} ± {np.std(aprs):.4f}")
else:
    print("No valid folds.")