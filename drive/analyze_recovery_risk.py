import pandas as pd
import numpy as np

# --- Configuration ---
DATA_PATH = 'data/'
ROLLING_WINDOW = 5

# 1. Load Data
try:
    distractions = pd.read_csv(f'{DATA_PATH}Dataset Distractions_distraction.csv')
    errors_dist  = pd.read_csv(f'{DATA_PATH}Dataset Errors_distraction.csv')
    errors_base  = pd.read_csv(f'{DATA_PATH}Dataset Errors_baseline.csv')
    driving_base = pd.read_csv(f'{DATA_PATH}Dataset Driving Time_baseline.csv')
except FileNotFoundError as e:
    print(f"Error: Could not find files. {e}")
    exit()

# Preprocessing
distractions['start']    = pd.to_datetime(distractions['timestamp_start'])
distractions['end']      = pd.to_datetime(distractions['timestamp_end'])
errors_dist['timestamp'] = pd.to_datetime(errors_dist['timestamp'])

df_sorted = distractions.sort_values(by=['user_id', 'run_id', 'timestamp_start'])

def get_session_gaps(group):
    gaps_sec = (pd.to_datetime(group['timestamp_start']) - pd.to_datetime(group['timestamp_end'].shift(1))).dt.total_seconds()
    # dropna removes the first row of each session which has no predecessor
    return gaps_sec.dropna()

all_gaps = df_sorted.groupby(['user_id', 'run_id'], group_keys=False).apply(get_session_gaps)

print(f"Minimum distance between distraction windows: {all_gaps.min():.4f}s")
print(f"Maximum distance between distraction windows: {all_gaps.max():.4f}s")
print(f"Average distance between distraction windows: {all_gaps.mean():.4f}s")


# 2. Baseline P(error | 1s)
total_base_duration = driving_base['run_duration_seconds'].sum()
p_error_baseline    = len(errors_base) / total_base_duration
print(f"BASELINE --> P(error | 1s): {100 * p_error_baseline:.5f}%")

# 3. Identify Errors Outside Distractions
windows_by_session = {
    (user, run): grp 
    for (user, run), grp in distractions.groupby(['user_id', 'run_id'])
}

def is_error_inside_distraction(row):
    session_key = (row['user_id'], row['run_id'])
    if session_key not in windows_by_session:
        return False
    run_windows = windows_by_session[session_key]
    return ((run_windows['start'] <= row['timestamp']) &
            (row['timestamp']     <= run_windows['end'])).any()

errors_dist['is_inside'] = errors_dist.apply(is_error_inside_distraction, axis=1)
outside_errors = errors_dist[~errors_dist['is_inside']].copy()

pct_outside = (len(outside_errors) / len(errors_dist)) * 100
print(f"DISTRACTION RUNS --> % of errors occurring OUTSIDE distraction windows: {pct_outside:.2f}%")

# 4. Compute gaps BETWEEN consecutive distraction windows
df_sorted = distractions.sort_values(by=['user_id', 'run_id', 'start'])

def get_inter_window_gaps(group):
    """Returns (gap_duration, gap_start, gap_end) for each inter-window gap."""
    records = []
    for i in range(1, len(group)):
        gap_start = group['end'].iloc[i - 1]
        gap_end   = group['start'].iloc[i]
        gap_len   = (gap_end - gap_start).total_seconds()
        if gap_len > 0:
            records.append({'gap_start': gap_start, 'gap_end': gap_end, 'gap_len': gap_len})
    return pd.DataFrame(records)

inter_window_gaps = df_sorted.groupby(['user_id', 'run_id'], group_keys=False).apply(get_inter_window_gaps)
inter_window_gaps = inter_window_gaps.reset_index(drop=True)

all_gap_lengths = inter_window_gaps['gap_len']
OFFSET = int(all_gap_lengths.max())

exposure_per_bin = np.zeros(OFFSET + 1)
for gap_len in all_gap_lengths:
    max_bin = int(min(gap_len, OFFSET))
    exposure_per_bin[1:max_bin + 1] += 1

def get_time_since_last(row):
    session_key = (row['user_id'], row['run_id'])
    if session_key not in windows_by_session:
        return np.nan, np.nan
    
    run_windows = windows_by_session[session_key]
    prev_windows = run_windows[run_windows['end'] < row['timestamp']]
    
    if prev_windows.empty:
        return np.nan, np.nan
        
    last_end = prev_windows['end'].max()
    
    next_windows = run_windows[run_windows['start'] > row['timestamp']]
    if next_windows.empty:
        return np.nan, np.nan  # After last window — excluded (no matching exposure)
        
    time_since = (row['timestamp'] - last_end).total_seconds()
    return time_since, last_end

outside_errors[['time_since_last_window', 'last_end']] = outside_errors.apply(
    lambda row: pd.Series(get_time_since_last(row)), axis=1
)
outside_after = outside_errors.dropna(subset=['time_since_last_window']).copy()

print(f"Errors in inter-window gaps (used for rate): {len(outside_after)}")
print(f"Errors after last window (excluded): {len(outside_errors) - len(outside_after)}")

# 5. Binning
bins   = np.arange(0, OFFSET + 1, 1)
labels = bins[1:]

outside_after['bin'] = pd.cut(
    outside_after['time_since_last_window'],
    bins=bins,
    labels=labels,
    right=True 
)

bin_counts = outside_after['bin'].value_counts().sort_index()

# 6. Rate Calculation
rates = {}
for i in range(1, OFFSET + 1):
    exposure = exposure_per_bin[i]
    if exposure > 0:
        rates[i] = bin_counts.get(i, 0) / exposure
    else:
        rates[i] = np.nan

rates_series = pd.Series(rates).dropna()

smoothed       = rates_series.rolling(window=ROLLING_WINDOW, center=True, min_periods=1).mean()
above_baseline = smoothed[smoothed > p_error_baseline]
hangover_limit = int(above_baseline.index.max()) if not above_baseline.empty else 0

# 7. Print Results
print(f"\n{'Sec':<5} | {'Error Rate (%)':<15} | {'Smoothed Rate (%)':<20} | {'vs Baseline':<12}")
print("-" * 70)

for i in rates_series.index:    
    diff = "ABOVE" if smoothed[i] > p_error_baseline else "BELOW"
    print(f"{i:<5} | {100 * rates_series[i]:<15.4f} | {100 * smoothed[i]:<20.4f} | {diff}")

# 9. Summary
print("-" * 70)
print(f"Estimated Hangover Time (smoothed): {hangover_limit} seconds")
print(f"Total Expected Errors per recovery gap: {rates_series.sum():.4f}")