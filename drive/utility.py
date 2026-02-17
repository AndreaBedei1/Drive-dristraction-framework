import pandas as pd
import numpy as np
import math

bins_of_interest = [9, 13, 15, 24]

# 1. Load data
distractions = pd.read_csv('data/Dataset Distractions_distraction.csv')
errors_dist = pd.read_csv('data/Dataset Errors_distraction.csv')

# 2. Convert timestamps to datetime
distractions['start'] = pd.to_datetime(distractions['timestamp_start'])
distractions['end']   = pd.to_datetime(distractions['timestamp_end'])
errors_dist['timestamp'] = pd.to_datetime(errors_dist['timestamp'])

# 3. Build a list of intervals per run for quick lookup
#    We'll keep them as DataFrames grouped by run
windows_by_run = {run: grp[['start','end']] for run, grp in distractions.groupby('run_id')}

# 4. Function to check if an error is inside any distraction window of its run
def is_inside_window(error_ts, run_windows):
    for _, row in run_windows.iterrows():
        if row['start'] <= error_ts <= row['end']:
            return True
    return False

# 5. Identify errors outside all distraction windows
outside_mask = []
for _, err in errors_dist.iterrows():
    run = err['run_id']
    if run not in windows_by_run:
        outside_mask.append(True)   # no distraction windows in this run? (shouldn't happen)
    else:
        outside_mask.append(not is_inside_window(err['timestamp'], windows_by_run[run]))
outside_errors = errors_dist[outside_mask].copy()

# 6. For each outside error, find the most recent window end (same run) and compute time difference
def most_recent_window_end(error_ts, run_windows):
    # windows that ended before the error
    prev = run_windows[run_windows['end'] < error_ts]
    if prev.empty:
        return None
    return prev['end'].max()

time_diffs = []
for _, err in outside_errors.iterrows():
    run = err['run_id']
    run_windows = windows_by_run[run]
    last_end = most_recent_window_end(err['timestamp'], run_windows)
    if last_end is None:
        time_diffs.append(None)   # error before any distraction in that run
    else:
        diff = (err['timestamp'] - last_end).total_seconds()
        time_diffs.append(diff)

outside_errors['time_since_last_window'] = time_diffs

# 7. Keep only errors that occurred after at least one distraction window
outside_after = outside_errors.dropna(subset=['time_since_last_window']).copy()

# 8. Assign a bin (second) using ceiling (so t in (0,1] -> bin 1, (1,2] -> bin 2, etc.)
outside_after['bin'] = outside_after['time_since_last_window'].apply(lambda t: math.ceil(t) if t > 0 else 0)

# 9. Select bins of interest
selected = outside_after[outside_after['bin'].isin(bins_of_interest)]

# 10. Display the selected errors (all columns)
print("Errors occurring exactly in the following seconds after a distraction window ended:")
for bin_val in bins_of_interest:
    rows = selected[selected['bin'] == bin_val]
    print(f"\n--- Second {bin_val} ---")
    if rows.empty:
        print("None")
    else:
        # Print relevant columns; you can adjust the list as needed
        print(rows[['timestamp', 'run_id', 'error_type', 'model_pred', 'model_prob', 
                    'emotion_label', 'emotion_prob', 'speed_kmh', 'details']].to_string(index=False))