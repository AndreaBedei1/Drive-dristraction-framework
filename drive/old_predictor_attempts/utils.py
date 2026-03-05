import pandas as pd
import numpy as np

ROLLING_WINDOW = 5

def is_error_inside_distraction(row, windows_by_session):
    """Check if an error occurred inside any distraction window of the same session."""
    session_key = (row['user_id'], row['run_id'])
    if session_key not in windows_by_session:
        return False
    run_windows = windows_by_session[session_key]
    return ((run_windows['timestamp_start'] <= row['timestamp']) &
            (row['timestamp'] <= run_windows['timestamp_end'])).any()


def get_inter_window_gaps(group):
    """For one (user, run) group, compute gaps between consecutive distractions."""
    user_id, run_id = group.name
    records = []
    for i in range(1, len(group)):
        gap_start, gap_end = group['timestamp_end'].iloc[i - 1], group['timestamp_start'].iloc[i]
        gap_len = (gap_end - gap_start).total_seconds()
        if gap_len > 0:
            records.append({'user_id': user_id, 'run_id': run_id, 'gap_len': gap_len})
    return pd.DataFrame(records)


def get_time_since_last(row, windows_by_session):
    """For an error outside a distraction, compute seconds since last distraction ended."""
    session_key = (row['user_id'], row['run_id'])
    if session_key not in windows_by_session:
        return np.nan
    run_windows = windows_by_session[session_key]
    prev_windows = run_windows[run_windows['timestamp_end'] < row['timestamp']]
    if prev_windows.empty:
        return np.nan
    # Keep original redundant check for exact behaviour
    next_windows = run_windows[run_windows['timestamp_start'] > row['timestamp']]
    if next_windows.empty:
        return np.nan
    return (row['timestamp'] - prev_windows['timestamp_end'].max()).total_seconds()


def calculate_hangover(gap_lengths, error_times, baseline_rate, rolling_window=ROLLING_WINDOW):
    """
    Compute hangover limit, smoothed probabilities, and raw per‑second probabilities.
    Returns (limit, smoothed_series, raw_series).
    """
    if len(gap_lengths) == 0:
        return 0, pd.Series(dtype=float), pd.Series(dtype=float)

    max_offset = int(gap_lengths.max())
    exposure = np.zeros(max_offset + 1)
    for gl in gap_lengths:
        exposure[1:int(min(gl, max_offset)) + 1] += 1

    bins = np.arange(0, max_offset + 1, 1)
    counts = pd.cut(error_times, bins=bins, labels=bins[1:]).value_counts().sort_index()

    rates = pd.Series({
        i: (counts.get(i, 0) / exposure[i] if exposure[i] > 0 else np.nan)
        for i in range(1, max_offset + 1)
    }).dropna()

    if rates.empty:
        return 0, rates, rates

    smoothed = rates.rolling(window=rolling_window, center=True, min_periods=1).mean()
    above = smoothed[smoothed > baseline_rate]
    limit = int(above.index.max()) if not above.empty else 0
    return limit, smoothed, rates


# ----------------------------------------------------------------------
# New function that packages the global probability computation
# ----------------------------------------------------------------------
def compute_global_probabilities(distractions, errors_dist, errors_base, driving_base, rolling_window=ROLLING_WINDOW):
    """
    Compute global per‑second error probabilities and raw error counts after a distraction.

    Parameters
    ----------
    distractions : pd.DataFrame
        DataFrame with columns: user_id, run_id, timestamp_start, timestamp_end.
    errors_dist : pd.DataFrame
        DataFrame with columns: user_id, run_id, timestamp.
    errors_base : pd.DataFrame
        Baseline errors (same structure as errors_dist).
    driving_base : pd.DataFrame
        Baseline driving times with column 'run_duration_seconds'.
    rolling_window : int, optional
        Window size for smoothing (default ROLLING_WINDOW).

    Returns
    -------
    dict with keys:
        'probabilities' : pd.DataFrame with columns ['second_after_distraction', 'probability']
        'raw_counts'    : pd.DataFrame with columns ['second_after_distraction', 'error_count']
        'rates_series'  : pd.Series (index = seconds, values = raw probability)
        'global_baseline' : float
        'global_hangover' : int
    """
    # Make copies to avoid modifying the original DataFrames
    distractions = distractions.copy()
    errors_dist = errors_dist.copy()

    # Convert timestamps
    distractions['timestamp_start'] = pd.to_datetime(distractions['timestamp_start'], format='ISO8601')
    distractions['timestamp_end'] = pd.to_datetime(distractions['timestamp_end'], format='ISO8601')
    errors_dist['timestamp'] = pd.to_datetime(errors_dist['timestamp'], format='ISO8601')
    distractions['duration_sec'] = (distractions['timestamp_end'] - distractions['timestamp_start']).dt.total_seconds()

    # Build lookup dictionary for distraction windows per (user, run)
    windows_by_session = {
        (user, run): grp
        for (user, run), grp in distractions.groupby(['user_id', 'run_id'])
    }

    # Identify errors that occurred outside any distraction
    errors_dist['is_inside'] = errors_dist.apply(
        lambda row: is_error_inside_distraction(row, windows_by_session), axis=1
    )
    outside_errors = errors_dist[~errors_dist['is_inside']].copy()

    # Compute gaps between consecutive distraction windows
    df_sorted = distractions.sort_values(by=['user_id', 'run_id', 'timestamp_start'])
    inter_window_gaps = df_sorted.groupby(['user_id', 'run_id'], group_keys=False).apply(
        get_inter_window_gaps
    ).reset_index(drop=True)

    # Compute time since last distraction for each outside error
    outside_errors['time_since'] = outside_errors.apply(
        lambda row: get_time_since_last(row, windows_by_session), axis=1
    )
    outside_after = outside_errors.dropna(subset=['time_since']).copy()

    # Global baseline error rate
    total_base_duration = driving_base['run_duration_seconds'].sum()
    p_error_baseline_global = len(errors_base) / total_base_duration

    # Run the core analysis
    global_limit, _, rates = calculate_hangover(
        inter_window_gaps['gap_len'],
        outside_after['time_since'],
        p_error_baseline_global,
        rolling_window
    )

    # Build probability DataFrame
    max_gap = int(inter_window_gaps['gap_len'].max()) if not inter_window_gaps.empty else 0
    prob_df = pd.DataFrame({
        'second_after_distraction': range(1, max_gap + 1),
        'probability': [rates.get(i, 0) for i in range(1, max_gap + 1)]
    })

    # Build raw counts DataFrame
    if len(outside_after) > 0 and max_gap > 0:
        bins = np.arange(0, max_gap + 1, 1)
        global_counts = pd.cut(outside_after['time_since'], bins=bins, labels=bins[1:], right=True).value_counts().sort_index()
        raw_counts_df = pd.DataFrame({
            'second_after_distraction': global_counts.index.astype(int),
            'error_count': global_counts.values
        })
    else:
        raw_counts_df = pd.DataFrame(columns=['second_after_distraction', 'error_count'])

    return {
        'probabilities': prob_df,
        'raw_counts': raw_counts_df,
        'rates_series': rates,
        'global_baseline': p_error_baseline_global,
        'global_hangover': global_limit
    }

