import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OFFSET = 10

# 1. Load Data
distractions = pd.read_csv('data/Dataset Distractions_distraction.csv')
errors_dist = pd.read_csv('data/Dataset Errors_distraction.csv')
errors_base = pd.read_csv('data/Dataset Errors_baseline.csv')
driving_time = pd.read_csv('data/Dataset Driving Time_baseline.csv')
distractions['start'] = pd.to_datetime(distractions['timestamp_start'])
distractions['end']   = pd.to_datetime(distractions['timestamp_end'])
errors_dist['timestamp'] = pd.to_datetime(errors_dist['timestamp'])

# 2. Baseline P(error | 1s)
total_base_duration = driving_time['run_duration_seconds'].sum()
p_error_baseline = len(errors_base) / total_base_duration

print(f"BASELINE --> P(error | 1s): {100*p_error_baseline:.4f}%")

# 3. Distraction P(Error outside distraction events | distraction events)
intervals = list(zip(distractions['start'], distractions['end']))
inside_mask = errors_dist['timestamp'].apply(
lambda ts: any(start <= ts <= end for start, end in intervals))
outside_errors = errors_dist[~inside_mask].copy() 
print(f"DISTRACTION --> P(Error outside distraction events | distraction events): {100*len(outside_errors) / len(errors_dist)}%")

# For each outside error, find the most recent window end (same run)
def most_recent_window_end(error_ts, run_windows):
    # run_windows: DataFrame of windows for that run
    prev_windows = run_windows[run_windows['end'] < error_ts]
    if prev_windows.empty:
        return pd.NaT  # no previous window
    return prev_windows['end'].max()

# Group windows by run
windows_by_run = {run: grp for run, grp in distractions.groupby('run_id')}

# Compute time since last window for each outside error
time_diffs = []
for _, err in outside_errors.iterrows():
    run = err['run_id']
    if run not in windows_by_run:
        time_diffs.append(np.nan)
        continue
    last_end = most_recent_window_end(err['timestamp'], windows_by_run[run])
    if pd.isna(last_end):
        time_diffs.append(np.nan)   # error before any window in that run
    else:
        diff = (err['timestamp'] - last_end).total_seconds()
        time_diffs.append(diff)

outside_errors['time_since_last_window'] = time_diffs

# Keep only errors that occurred after at least one window
outside_after = outside_errors.dropna(subset=['time_since_last_window']).copy()

# Now bin the times (e.g., into 1‑second intervals)
bins = np.arange(0, OFFSET+1, 1)   # 0‑60 seconds in 1‑sec steps
labels = bins[1:]             # 1,2,...,60

# Assign each error to a bin (right‑closed intervals, e.g., (0,1] seconds)
outside_after['bin'] = pd.cut(
    outside_after['time_since_last_window'],
    bins=bins,
    labels=labels,
    right=True,
    include_lowest=False   # bin 0 is (0,1] – we exclude exactly 0
)

# Count errors per bin
bin_counts = outside_after['bin'].value_counts().sort_index()
bin_probs = bin_counts / len(outside_after)   # proportion of all outside‑after errors

# Print results for first few bins
for i in range(1, OFFSET + 1):
    prob = bin_probs.get(i, 0)
    print(f"Errors in second {i}: {prob * 100:.3f}")

print(f"Sum (first {OFFSET} seconds): {bin_probs.loc[1:OFFSET].sum():.3f}")