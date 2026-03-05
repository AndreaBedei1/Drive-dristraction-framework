from utils import *

# --- Configuration ---
DATA_PATH = ''


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

    # --- Extension: Errors per distraction window duration ---
    # First, identify outside errors
    windows_by_session = {
        (user, run): grp
        for (user, run), grp in distractions.groupby(['user_id', 'run_id'])
    }

    errors_dist['is_inside'] = errors_dist.apply(
        lambda row: is_error_inside_distraction(row, windows_by_session), axis=1
    )
    outside_errors = errors_dist[~errors_dist['is_inside']].copy()

    # New function to get time_since, preceding_duration, preceding_dist_id
    def get_time_since_preceding(row, windows_by_session):
        user, run, ts = row['user_id'], row['run_id'], row['timestamp']
        key = (user, run)
        if key not in windows_by_session:
            return np.nan, np.nan, np.nan
        session_windows = windows_by_session[key]
        prev_ends = session_windows[session_windows['timestamp_end'] < ts]['timestamp_end']
        if prev_ends.empty:
            return np.nan, np.nan, np.nan
        last_end = prev_ends.max()
        time_since = (ts - last_end).total_seconds()
        preceding = session_windows[session_windows['timestamp_end'] == last_end]
        if preceding.empty:
            return time_since, np.nan, np.nan
        dur = preceding['duration_sec'].iloc[0]
        dist_id = preceding.index[0]  # using DataFrame index as ID
        return time_since, dur, dist_id

    # Apply to outside_errors
    outside_errors[['time_since', 'preceding_duration', 'preceding_dist_id']] = outside_errors.apply(
        lambda row: pd.Series(get_time_since_preceding(row, windows_by_session)), axis=1
    )

    # Filter to within global hangover limit
    outside_within_limit = outside_errors[(outside_errors['time_since'] > 0) & (outside_errors['time_since'] < global_limit)].copy()

    # Count errors per preceding distraction
    error_counts = outside_within_limit.groupby('preceding_dist_id').size().reset_index(name='error_count')

    # Merge with distractions (using index as key)
    dist_with_errors = distractions.merge(error_counts, left_index=True, right_on='preceding_dist_id', how='left')
    dist_with_errors['error_count'] = dist_with_errors['error_count'].fillna(0)

    # Bin durations
    bins = [0, 1, 2, 3, 5, 10, 20, float('inf')]
    labels = ['0-1s', '1-2s', '2-3s', '3-5s', '5-10s', '10-20s', '>20s']
    dist_with_errors['duration_bin'] = pd.cut(dist_with_errors['duration_sec'], bins=bins, labels=labels, right=True)

    # Aggregate
    agg = dist_with_errors.groupby('duration_bin', observed=True).agg(
        num_distractions=('duration_sec', 'count'),
        total_errors=('error_count', 'sum'),
        avg_errors_per_dist=('error_count', 'mean')
    ).reset_index()

    # --- Per‑participant analysis (unchanged, but now uses precomputed data) ---
    df_sorted = distractions.sort_values(by=['user_id', 'run_id', 'timestamp_start'])
    inter_window_gaps = df_sorted.groupby(['user_id', 'run_id'], group_keys=False).apply(
        get_inter_window_gaps
    ).reset_index(drop=True)

    # Compute time_since for outside errors (already done above, but kept for per-participant)
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

    # New: Print errors per duration bin
    print("\nERRORS PER DISTRACTION WINDOW DURATION BIN")
    print(agg.to_string(index=False))

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

# Add inside/outside classification per distraction
def count_errors_for_distraction(dist_row, errors_dist):
    start = dist_row['timestamp_start']
    end = dist_row['timestamp_end']
    user = dist_row['user_id']
    run = dist_row['run_id']
    
    relevant_errors = errors_dist[
        (errors_dist['user_id'] == user) &
        (errors_dist['run_id'] == run) &
        (errors_dist['timestamp'] >= start) &
        (errors_dist['timestamp'] <= end)
    ]
    return len(relevant_errors)

# Apply to all distractions
dist_with_errors['inside_errors'] = dist_with_errors.apply(
    lambda row: count_errors_for_distraction(row, errors_dist), axis=1
)

# Now aggregate both inside and outside
agg_both = dist_with_errors.groupby('duration_bin', observed=True).agg(
    num_distractions=('duration_sec', 'count'),
    total_inside_errors=('inside_errors', 'sum'),
    avg_inside=('inside_errors', 'mean'),
    total_outside_errors=('error_count', 'sum'),
    avg_outside=('error_count', 'mean')
).reset_index()

print("\nERRORS INSIDE vs OUTSIDE DISTRACTION BY DURATION BIN")
print(agg_both.to_string(index=False))

# ────────────────────────────────────────────────────────────────
# RECOVERY PHASE ANALYSIS – URBAN SPEED (0–50 km/h) + STEERING
# ────────────────────────────────────────────────────────────────

print("\n" + "="*100)
print("RECOVERY PHASE ANALYSIS – URBAN ENVIRONMENT (speed 0–50 km/h)")
print("="*100)

# 1. Prepare kinematics summaries per distraction (post-distraction errors)
kinematics_post = outside_errors.groupby('preceding_dist_id').agg(
    mean_speed_post=('speed_kmh', 'mean'),
    max_speed_post=('speed_kmh', 'max'),               # sometimes max tells more in urban stop-go
    steer_var_post=('steer_angle_deg', 'var'),         # steering instability during recovery
    steer_range_post=('steer_angle_deg', lambda x: x.max() - x.min()),  # total steering sweep
    n_errors_post=('timestamp', 'count')
).reset_index()

# Merge to dist_with_errors
dist_recovery = dist_with_errors.merge(
    kinematics_post,
    on='preceding_dist_id',
    how='left'
)

# Fill missing (no post-errors) with neutral / median values
dist_recovery['mean_speed_post'] = dist_recovery['mean_speed_post'].fillna(dist_recovery['mean_speed_post'].median())
dist_recovery['steer_var_post']  = dist_recovery['steer_var_post'].fillna(0)
dist_recovery['steer_range_post'] = dist_recovery['steer_range_post'].fillna(0)

# 2. Urban-appropriate speed bins
speed_bins_urban = [0, 20, 35, 50, float('inf')]
speed_labels_urban = ['0–20 km/h', '20–35 km/h', '35–50 km/h', '>50 km/h (rare)']
dist_recovery['speed_bin'] = pd.cut(
    dist_recovery['mean_speed_post'],
    bins=speed_bins_urban,
    labels=speed_labels_urban,
    include_lowest=True,
    right=False
)

# Steering variance bins (adjust thresholds after seeing .describe() if needed)
steer_var_bins = [0, 4, 12, 30, float('inf')]
steer_var_labels = ['Very stable (<4°)', 'Stable (4–12°)', 'Moderate (12–30°)', 'Erratic (>30°)']
dist_recovery['steer_var_bin'] = pd.cut(
    dist_recovery['steer_var_post'],
    bins=steer_var_bins,
    labels=steer_var_labels,
    include_lowest=True
)

# 3. Summary statistics per group
def recovery_stats(g):
    n = len(g)
    n_with_err = (g['error_count'] > 0).sum()
    pct_with_err = n_with_err / n * 100 if n > 0 else 0
    avg_err = g['error_count'].mean()
    avg_time_first = outside_errors[outside_errors['preceding_dist_id'].isin(g['preceding_dist_id'])]['time_since'].mean()
    avg_steer_range = g['steer_range_post'].mean()
    return pd.Series({
        'n_dist': n,
        '% with ≥1 post-error': round(pct_with_err, 1),
        'avg post-errors': round(avg_err, 3),
        'avg time to 1st post-error (s)': round(avg_time_first, 1) if not pd.isna(avg_time_first) else '—',
        'avg steering range post (deg)': round(avg_steer_range, 1)
    })

print("\nPOST-DISTRACTION RECOVERY BY MEAN SPEED BIN (urban)")
speed_rec = dist_recovery.groupby('speed_bin', observed=True).apply(recovery_stats).reset_index()
print(speed_rec.to_string(index=False))

print("\nPOST-DISTRACTION RECOVERY BY STEERING VARIANCE BIN")
steer_rec = dist_recovery.groupby('steer_var_bin', observed=True).apply(recovery_stats).reset_index()
print(steer_rec.to_string(index=False))

# 4. Simple heuristic hangover per bin (last observed error time)
def approx_hangover(group):
    ts = outside_errors[outside_errors['preceding_dist_id'].isin(group['preceding_dist_id'])]['time_since']
    if len(ts) == 0:
        return 0.0
    return max(ts.max(), ts.quantile(0.90), 2.0)  # at least 2 s if any error

print("\nApproximate hangover duration per speed bin")
for lbl, grp in dist_recovery.groupby('speed_bin', observed=True):
    h = approx_hangover(grp)
    n = len(grp)
    print(f"{lbl:<14} → ~{h:.1f} s   (n={n})")

print("\nApproximate hangover duration per steering variance bin")
for lbl, grp in dist_recovery.groupby('steer_var_bin', observed=True):
    h = approx_hangover(grp)
    n = len(grp)
    print(f"{lbl:<22} → ~{h:.1f} s   (n={n})")

# 5. Quick correlation check
print("\nCorrelations with post-distraction error count:")
corr_table = dist_recovery[['error_count', 'mean_speed_post', 'steer_var_post', 'steer_range_post']].corr()['error_count'].round(3)
print(corr_table)


print("\nCross-tab: % with post-error by duration_bin and steer_var_bin")
cross = pd.crosstab(dist_recovery['duration_bin'], dist_recovery['steer_var_bin'], 
                    values=dist_recovery['error_count'].gt(0).astype(int), 
                    aggfunc='mean') * 100
print(cross.round(1))

import statsmodels.formula.api as smf

dist_recovery['has_post_error'] = (dist_recovery['error_count'] > 0).astype(int)
dist_recovery['steer_var_log'] = np.log1p(dist_recovery['steer_var_post'])  # handle skew

model = smf.logit(
    'has_post_error ~ duration_sec + steer_var_log + duration_sec * steer_var_log',
    data=dist_recovery
).fit_regularized(method='l1', alpha=0.1, disp=0)   # small L1 penalty helps
print(model.summary())

print("\n" + "="*80)
print("ANALYSIS: AROUSAL CHANGE vs ERROR OFFSET / POST-ERROR TIMING")
print("="*80)

# 1. Compute arousal delta per distraction
if 'arousal_start' in distractions.columns and 'arousal_end' in distractions.columns:
    dist_with_errors['arousal_delta'] = dist_with_errors['arousal_end'] - dist_with_errors['arousal_start']
else:
    print("Warning: No arousal_start / arousal_end columns found → skipping")
    # Alternative: if you only have emotion probs, approximate delta as proxy
    if 'emotion_prob_start' in distractions.columns and 'emotion_prob_end' in distractions.columns:
        dist_with_errors['arousal_delta_proxy'] = dist_with_errors['emotion_prob_end'] - dist_with_errors['emotion_prob_start']
        print("Using emotion_prob delta as arousal proxy")
    else:
        print("No arousal or emotion probability columns → cannot proceed")
        # exit or skip

# 2. Link to post-distraction error timing
# Get first post-error time_since for each distraction
first_post_error = outside_errors.groupby('preceding_dist_id')['time_since'].min().reset_index(name='first_error_offset')

dist_arousal_error = dist_with_errors.merge(
    first_post_error,
    left_on='preceding_dist_id',
    right_on='preceding_dist_id',
    how='left'
)

# Flag if there was any post-error
dist_arousal_error['has_post_error'] = dist_arousal_error['error_count'] > 0

# 3. Basic summaries
print("\nArousal delta statistics overall:")
print(dist_arousal_error['arousal_delta'].describe())

print("\nArousal delta | has post-error")
print(dist_arousal_error.groupby('has_post_error')['arousal_delta'].agg(['count', 'mean', 'median', 'std']))

print("\nArousal delta | first error offset (among those with post-error)")
post_only = dist_arousal_error[dist_arousal_error['has_post_error']]
print(post_only.groupby(pd.qcut(post_only['arousal_delta'], q=4, duplicates='drop'))['first_error_offset'].agg(['count', 'mean', 'median']))

# 4. Correlation
print("\nCorrelations with arousal_delta:")
print(dist_arousal_error[['arousal_delta', 'error_count', 'first_error_offset']].corr()['arousal_delta'].round(3))
print("\nArousal delta by duration_bin and has_post_error")
print(dist_arousal_error.groupby(['duration_bin', 'has_post_error'])['arousal_delta'].mean().unstack())