from utils import *

# --- Configuration ---
DATA_PATH = 'data/'


# ----------------------------------------------------------------------
# Helper functions (unchanged)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Main script (client code) – now uses the new function
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Load Data
    try:
        distractions = pd.read_csv(f'{DATA_PATH}Dataset Distractions_distraction.csv')
        errors_dist = pd.read_csv(f'{DATA_PATH}Dataset Errors_distraction.csv')
        errors_base = pd.read_csv(f'{DATA_PATH}Dataset Errors_baseline.csv')
        driving_base = pd.read_csv(f'{DATA_PATH}Dataset Driving Time_baseline.csv')
    except FileNotFoundError as e:
        print(f"Error: Could not find files. {e}")
        exit()

    # --- Preprocessing (shared with per‑participant analysis) ---
    distractions['timestamp_start'] = pd.to_datetime(distractions['timestamp_start'], format='ISO8601')
    distractions['timestamp_end'] = pd.to_datetime(distractions['timestamp_end'], format='ISO8601')
    errors_dist['timestamp'] = pd.to_datetime(errors_dist['timestamp'], format='ISO8601')
    distractions['duration_sec'] = (distractions['timestamp_end'] - distractions['timestamp_start']).dt.total_seconds()

    # Distraction statistics per user (used later)
    distraction_stats = distractions.groupby('user_id').agg(
        num_distractions=('duration_sec', 'size'),
        avg_duration_sec=('duration_sec', 'mean')
    ).reset_index()

    total_distractions = len(distractions)
    global_avg_duration = distractions['duration_sec'].mean()

    # Baseline calculation (global and per‑user)
    total_base_duration = driving_base['run_duration_seconds'].sum()
    p_error_baseline_global = len(errors_base) / total_base_duration

    user_base_errs = errors_base.groupby('user_id').size()
    user_base_time = driving_base.groupby('user_id')['run_duration_seconds'].sum()
    user_baselines = (user_base_errs / user_base_time).fillna(0).to_dict()

    # --- Global analysis using the new function ---
    global_results = compute_global_probabilities(
        distractions, errors_dist, errors_base, driving_base, rolling_window=ROLLING_WINDOW
    )
    rates = global_results['rates_series']
    global_limit = global_results['global_hangover']
    p_error_baseline_global = global_results['global_baseline']  # already computed, but keep for clarity

    # --- Per‑participant analysis (unchanged, but now uses precomputed data) ---
    windows_by_session = {
        (user, run): grp
        for (user, run), grp in distractions.groupby(['user_id', 'run_id'])
    }

    # Identify outside errors (we could reuse from global function, but we recompute for clarity)
    errors_dist['is_inside'] = errors_dist.apply(
        lambda row: is_error_inside_distraction(row, windows_by_session), axis=1
    )
    outside_errors = errors_dist[~errors_dist['is_inside']].copy()

    # Compute inter‑window gaps
    df_sorted = distractions.sort_values(by=['user_id', 'run_id', 'timestamp_start'])
    inter_window_gaps = df_sorted.groupby(['user_id', 'run_id'], group_keys=False).apply(
        get_inter_window_gaps
    ).reset_index(drop=True)

    # Compute time_since for outside errors
    outside_errors['time_since'] = outside_errors.apply(
        lambda row: get_time_since_last(row, windows_by_session), axis=1
    )
    outside_after = outside_errors.dropna(subset=['time_since']).copy()

    participant_results = []
    all_users = distractions['user_id'].unique()

    for user in all_users:
        user_gaps = inter_window_gaps[inter_window_gaps['user_id'] == user]['gap_len']
        user_errs = outside_after[outside_after['user_id'] == user]['time_since']
        user_baseline = user_baselines.get(user, p_error_baseline_global)

        limit, _, _ = calculate_hangover(user_gaps, user_errs, user_baseline)

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

    # --- Results Reporting (using global_results) ---
    print("=" * 100)
    print("GLOBAL ANALYSIS")
    print(f"Baseline Error Rate: {p_error_baseline_global*100:.4f}%")
    print(f"Global Hangover: {global_limit} seconds")
    print(f"Total Distractions: {total_distractions}")
    print(f"Total Errors: {len(errors_dist)}")
    print(f"Global Avg Distraction Duration: {global_avg_duration:.2f} s")
    print("=" * 100)

    header = f"{'User ID':<18} | {'Baseline %':<10} | {'Hangover (s)':<12} | {'Errors':<6} | {'Distractions':<12} | {'Avg Dist (s)':<12}"
    print(header)
    print("-" * len(header))

    for res in sorted(participant_results, key=lambda x: x['User']):
        print(f"{res['User']:<18} | {res['Baseline (%)']:<10.4f} | {res['Hangover (s)']:<12} | {res['Errors Found']:<6} | {res['Distractions']:<12} | {res['Avg Distraction (s)']:<12.2f}")

    print("-" * len(header))

    # Print global probability table from the packaged results
    print("\nGLOBAL PROBABILITY PER SECOND (Post-Distraction)")
    print(global_results['probabilities'].head(16)['probability'])

    # Print raw error counts
    if not global_results['raw_counts'].empty:
        print("\nRAW ERROR COUNTS PER SECOND (Post-Distraction)")
        print(global_results['raw_counts'].head(16).to_string(index=False))
    else:
        print("\nNo outside errors to compute raw counts.")