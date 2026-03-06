import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from utils import *
import statsmodels.formula.api as smf

DATA_PATH = ''


def floor_to_second(series):
    """Truncate datetime series to second granularity (drop ms)."""
    return pd.to_datetime(series).dt.floor('s')


def compute_arousal_endpoints(distractions, timeline):
    """
    For each distraction event, compute mean arousal in the 2s window
    around timestamp_start and timestamp_end from the timeline.
    """
    timeline = timeline.copy()
    timeline['timestamp'] = floor_to_second(timeline['timestamp'])

    rows = []
    for _, d in distractions.iterrows():
        pid, rid, ts, te = d['user_id'], d['run_id'], d['timestamp_start'], d['timestamp_end']
        tl = timeline[(timeline['user_id'] == pid) & (timeline['run_id'] == rid)]

        def wmean(t0, a, b):
            sub = tl[(tl['timestamp'] >= t0 + pd.Timedelta(f'{a}s')) &
                     (tl['timestamp'] <  t0 + pd.Timedelta(f'{b}s'))]
            return sub['arousal'].mean() if len(sub) else np.nan

        rows.append({
            'arousal_start': wmean(ts, -1, 1),
            'arousal_end':   wmean(te, -1, 1),
        })

    return pd.DataFrame(rows, index=distractions.index)


if __name__ == "__main__":

    # ── 1. Load data ──────────────────────────────────────────────────────
    try:
        distractions = pd.read_csv(f'{DATA_PATH}Dataset Distractions_distraction.csv')
        errors_dist  = pd.read_csv(f'{DATA_PATH}Dataset Errors_distraction.csv')
        errors_base  = pd.read_csv(f'{DATA_PATH}Dataset Errors_baseline.csv')
        driving_base = pd.read_csv(f'{DATA_PATH}Dataset Driving Time_baseline.csv')
        timeline_d   = pd.read_csv(f'{DATA_PATH}Dataset Timeline_distraction.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit()

    # ── 2. Preprocessing — truncate to second granularity ─────────────────
    distractions['timestamp_start'] = floor_to_second(distractions['timestamp_start'])
    distractions['timestamp_end']   = floor_to_second(distractions['timestamp_end'])
    errors_dist['timestamp']        = floor_to_second(errors_dist['timestamp'])
    errors_base['timestamp']        = floor_to_second(errors_base['timestamp'])

    distractions['duration_sec'] = (
        distractions['timestamp_end'] - distractions['timestamp_start']
    ).dt.total_seconds()

    # ── 3. Arousal endpoints from timeline ────────────────────────────────
    arousal_cols = compute_arousal_endpoints(distractions, timeline_d)
    distractions = pd.concat([distractions, arousal_cols], axis=1)

    # ── 4. Distraction statistics per user ────────────────────────────────
    distraction_stats = distractions.groupby('user_id').agg(
        num_distractions=('duration_sec', 'size'),
        avg_duration_sec=('duration_sec', 'mean')
    ).reset_index()

    total_distractions  = len(distractions)
    global_avg_duration = distractions['duration_sec'].mean()

    # ── 5. Baseline calculation ───────────────────────────────────────────
    total_base_duration       = driving_base['run_duration_seconds'].sum()
    p_error_baseline_global   = len(errors_base) / total_base_duration

    user_base_errs  = errors_base.groupby('user_id').size()
    user_base_time  = driving_base.groupby('user_id')['run_duration_seconds'].sum()
    user_baselines  = (user_base_errs / user_base_time).fillna(0).to_dict()

    # ── 6. Global analysis ────────────────────────────────────────────────
    global_results = compute_global_probabilities(
        distractions, errors_dist, errors_base, driving_base,
        rolling_window=ROLLING_WINDOW
    )
    rates        = global_results['rates_series']
    global_limit = global_results['global_hangover']

    # ── 7. Identify outside errors ────────────────────────────────────────
    windows_by_session = {
        (user, run): grp
        for (user, run), grp in distractions.groupby(['user_id', 'run_id'])
    }

    errors_dist['is_inside'] = errors_dist.apply(
        lambda row: is_error_inside_distraction(row, windows_by_session), axis=1
    )
    outside_errors = errors_dist[~errors_dist['is_inside']].copy()

    # ── 8. Time since preceding distraction + preceding duration/id ───────
    def get_time_since_preceding(row, windows_by_session):
        user, run, ts = row['user_id'], row['run_id'], row['timestamp']
        key = (user, run)
        if key not in windows_by_session:
            return np.nan, np.nan, np.nan
        session_windows = windows_by_session[key]
        prev_ends = session_windows[session_windows['timestamp_end'] < ts]['timestamp_end']
        if prev_ends.empty:
            return np.nan, np.nan, np.nan
        last_end   = prev_ends.max()
        time_since = (ts - last_end).total_seconds()
        preceding  = session_windows[session_windows['timestamp_end'] == last_end]
        if preceding.empty:
            return time_since, np.nan, np.nan
        return time_since, preceding['duration_sec'].iloc[0], preceding.index[0]

    outside_errors[['time_since', 'preceding_duration', 'preceding_dist_id']] = \
        outside_errors.apply(
            lambda row: pd.Series(get_time_since_preceding(row, windows_by_session)),
            axis=1
        )

    outside_within_limit = outside_errors[
        (outside_errors['time_since'] > 0) &
        (outside_errors['time_since'] < global_limit)
    ].copy()

    # ── 9. Errors per preceding distraction ──────────────────────────────
    error_counts = outside_within_limit.groupby('preceding_dist_id').size().reset_index(
        name='error_count'
    )
    dist_with_errors = distractions.merge(
        error_counts, left_index=True, right_on='preceding_dist_id', how='left'
    )
    dist_with_errors['error_count'] = dist_with_errors['error_count'].fillna(0)

    # ── 10. Bin by distraction duration ──────────────────────────────────
    dur_bins   = [0, 1, 2, 3, 5, 10, 20, float('inf')]
    dur_labels = ['0-1s', '1-2s', '2-3s', '3-5s', '5-10s', '10-20s', '>20s']
    dist_with_errors['duration_bin'] = pd.cut(
        dist_with_errors['duration_sec'], bins=dur_bins, labels=dur_labels, right=True
    )

    agg = dist_with_errors.groupby('duration_bin', observed=True).agg(
        num_distractions=('duration_sec', 'count'),
        total_errors=('error_count', 'sum'),
        avg_errors_per_dist=('error_count', 'mean')
    ).reset_index()

    # ── 11. Per-participant hangover ──────────────────────────────────────
    df_sorted = distractions.sort_values(['user_id', 'run_id', 'timestamp_start'])
    inter_window_gaps = df_sorted.groupby(
        ['user_id', 'run_id'], group_keys=False
    ).apply(get_inter_window_gaps).reset_index(drop=True)

    outside_errors['time_since'] = outside_errors.apply(
        lambda row: get_time_since_last(row, windows_by_session), axis=1
    )
    outside_after = outside_errors.dropna(subset=['time_since']).copy()

    participant_results = []
    for user in distractions['user_id'].unique():
        user_gaps     = inter_window_gaps[inter_window_gaps['user_id'] == user]['gap_len']
        user_errs     = outside_after[outside_after['user_id'] == user]['time_since']
        user_baseline = user_baselines.get(user, p_error_baseline_global)
        limit, _, _   = calculate_hangover(user_gaps, user_errs, user_baseline)
        dist_stats    = distraction_stats[distraction_stats['user_id'] == user]
        num_dist      = dist_stats['num_distractions'].values[0] if not dist_stats.empty else 0
        avg_dur       = dist_stats['avg_duration_sec'].values[0] if not dist_stats.empty else 0.0
        participant_results.append({
            'User':               user,
            'Baseline (%)':       user_baseline * 100,
            'Hangover (s)':       limit,
            'Errors Found':       len(user_errs),
            'Distractions':       num_dist,
            'Avg Distraction (s)': avg_dur
        })

    # ── 12. Print global results ──────────────────────────────────────────
    print("=" * 100)
    print("GLOBAL ANALYSIS")
    print(f"Baseline Error Rate: {p_error_baseline_global*100:.4f}%")
    print(f"Global Hangover: {global_limit} seconds")
    print(f"Total Distractions: {total_distractions}")
    print(f"Total Errors: {len(errors_dist)}")
    print(f"Global Avg Distraction Duration: {global_avg_duration:.2f} s")
    print("=" * 100)

    print("\nERRORS PER DISTRACTION WINDOW DURATION BIN")
    print(agg.to_string(index=False))

    header = (f"{'User ID':<20} | {'Baseline %':<10} | {'Hangover (s)':<12} | "
              f"{'Errors':<6} | {'Distractions':<12} | {'Avg Dist (s)':<12}")
    print("\n" + header)
    print("-" * len(header))
    for res in sorted(participant_results, key=lambda x: x['User']):
        print(f"{res['User']:<20} | {res['Baseline (%)']:<10.4f} | {res['Hangover (s)']:<12} | "
              f"{res['Errors Found']:<6} | {res['Distractions']:<12} | {res['Avg Distraction (s)']:<12.2f}")
    print("-" * len(header))

    print("\nGLOBAL PROBABILITY PER SECOND (Post-Distraction)")
    print(global_results['probabilities'].head(16)['probability'])

    if not global_results['raw_counts'].empty:
        print("\nRAW ERROR COUNTS PER SECOND (Post-Distraction)")
        print(global_results['raw_counts'].head(16).to_string(index=False))
    else:
        print("\nNo outside errors to compute raw counts.")

    