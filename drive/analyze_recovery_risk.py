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
distractions['start']    = pd.to_datetime(distractions['timestamp_start'], format='ISO8601')
distractions['end']      = pd.to_datetime(distractions['timestamp_end'], format='ISO8601')
errors_dist['timestamp'] = pd.to_datetime(errors_dist['timestamp'], format='ISO8601')

# Compute distraction duration (seconds)
distractions['duration_sec'] = (distractions['end'] - distractions['start']).dt.total_seconds()

# ------------------------------------------------------------------
# Distraction statistics per user
# ------------------------------------------------------------------
distraction_stats = distractions.groupby('user_id').agg(
    num_distractions=('duration_sec', 'size'),
    avg_duration_sec=('duration_sec', 'mean')
).reset_index()

# Global distraction statistics
total_distractions = len(distractions)
global_avg_duration = distractions['duration_sec'].mean()

# 2. Baseline Calculation (Global and Per-User)
total_base_duration = driving_base['run_duration_seconds'].sum()
p_error_baseline_global = len(errors_base) / total_base_duration

user_base_errs = errors_base.groupby('user_id').size()
user_base_time = driving_base.groupby('user_id')['run_duration_seconds'].sum()
user_baselines = (user_base_errs / user_base_time).fillna(0).to_dict()

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

# 4. Compute Inter-window Gaps and Errors
df_sorted = distractions.sort_values(by=['user_id', 'run_id', 'start'])

def get_inter_window_gaps(group):
    user_id, run_id = group.name
    records = []
    for i in range(1, len(group)):
        gap_start, gap_end = group['end'].iloc[i - 1], group['start'].iloc[i]
        gap_len = (gap_end - gap_start).total_seconds()
        if gap_len > 0:
            records.append({'user_id': user_id, 'run_id': run_id, 'gap_len': gap_len})
    return pd.DataFrame(records)

inter_window_gaps = df_sorted.groupby(['user_id', 'run_id'], group_keys=False).apply(get_inter_window_gaps)
inter_window_gaps = inter_window_gaps.reset_index(drop=True)

def get_time_since_last(row):
    session_key = (row['user_id'], row['run_id'])
    if session_key not in windows_by_session: return np.nan
    run_windows = windows_by_session[session_key]
    prev_windows = run_windows[run_windows['end'] < row['timestamp']]
    if prev_windows.empty: return np.nan
    next_windows = run_windows[run_windows['start'] > row['timestamp']]
    if next_windows.empty: return np.nan 
    return (row['timestamp'] - prev_windows['end'].max()).total_seconds()

outside_errors['time_since'] = outside_errors.apply(get_time_since_last, axis=1)
outside_after = outside_errors.dropna(subset=['time_since']).copy()

# 5. Core Analysis Function
def calculate_hangover(gap_lengths, error_times, baseline_rate):
    if len(gap_lengths) == 0: return 0, pd.Series(dtype=float)
    
    max_offset = int(gap_lengths.max())
    exposure = np.zeros(max_offset + 1)
    for gl in gap_lengths:
        exposure[1:int(min(gl, max_offset)) + 1] += 1
    
    bins = np.arange(0, max_offset + 1, 1)
    counts = pd.cut(error_times, bins=bins, labels=bins[1:]).value_counts().sort_index()
    
    rates = pd.Series({i: (counts.get(i, 0) / exposure[i] if exposure[i] > 0 else np.nan) 
                       for i in range(1, max_offset + 1)}).dropna()
        
    if rates.empty: return 0, rates
    
    smoothed = rates.rolling(window=ROLLING_WINDOW, center=True, min_periods=1).mean()
    above = smoothed[smoothed > baseline_rate]
    limit = int(above.index.max()) if not above.empty else 0
    return limit, smoothed, rates

# 6. Global Analysis
global_limit, global_smoothed, rates = calculate_hangover(
    inter_window_gaps['gap_len'], outside_after['time_since'], p_error_baseline_global
)

# 7. Per-Participant Analysis
participant_results = []
all_users = distractions['user_id'].unique()

for user in all_users:
    user_gaps = inter_window_gaps[inter_window_gaps['user_id'] == user]['gap_len']
    user_errs = outside_after[outside_after['user_id'] == user]['time_since']
    user_baseline = user_baselines.get(user, p_error_baseline_global)
    
    limit, _, _ = calculate_hangover(user_gaps, user_errs, user_baseline)

    # Retrieve distraction stats for this user
    dist_stats = distraction_stats[distraction_stats['user_id'] == user]
    num_dist = dist_stats['num_distractions'].values[0] if not dist_stats.empty else 0
    avg_dur = dist_stats['avg_duration_sec'].values[0] if not dist_stats.empty else 0.0

    participant_results.append({
        'User': user,
        'Baseline (%)': user_baseline * 100,
        'Hangover (s)': limit,
        'Errors Found': len(user_errs),
        'Distractions': num_dist,
        'Avg Distraction (s)': avg_dur
    })

# 8. Results Reporting
print("=" * 100)
print("GLOBAL ANALYSIS")
print(f"Baseline Error Rate: {p_error_baseline_global*100:.4f}%")
print(f"Global Hangover: {global_limit} seconds")
print(f"Total Distractions: {total_distractions}")
print(f"Total Errors: {len(errors_dist)}")
print(f"Global Avg Distraction Duration: {global_avg_duration:.2f} s")
print("=" * 100)

# Print header
header = f"{'User ID':<18} | {'Baseline %':<10} | {'Hangover (s)':<12} | {'Errors':<6} | {'Distractions':<12} | {'Avg Dist (s)':<12}"
print(header)
print("-" * len(header))

for res in sorted(participant_results, key=lambda x: x['User']):
    print(f"{res['User']:<18} | {res['Baseline (%)']:<10.4f} | {res['Hangover (s)']:<12} | {res['Errors Found']:<6} | {res['Distractions']:<12} | {res['Avg Distraction (s)']:<12.2f}")

print("-" * len(header))

N_seconds = int(inter_window_gaps['gap_len'].max())

# Create the global probability table
global_prob_df = pd.DataFrame({
    'second_after_distraction': range(1, N_seconds + 1),
    'probability': [rates.get(i, 0) for i in range(1, N_seconds + 1)]
})

print("\nGLOBAL PROBABILITY PER SECOND (Post-Distraction)")
print(global_prob_df.head(16)['probability'])

# After the global analysis, add:

# Compute raw error counts per second (global)
if len(outside_after) > 0:
    max_offset_global = int(inter_window_gaps['gap_len'].max())
    bins = np.arange(0, max_offset_global + 1, 1)
    # Cut time_since into 1‑second bins, label with the right edge (i.e., second 1 means (0,1])
    global_counts = pd.cut(outside_after['time_since'], bins=bins, labels=bins[1:], right=True).value_counts().sort_index()
    raw_counts_df = pd.DataFrame({
        'second_after_distraction': global_counts.index.astype(int),
        'error_count': global_counts.values
    })
    print("\nRAW ERROR COUNTS PER SECOND (Post-Distraction)")
    print(raw_counts_df.head(16).to_string(index=False))
else:
    print("\nNo outside errors to compute raw counts.")