#!/usr/bin/env python3
"""
Augment driving simulator data to 30 participants × 10 distraction runs
(and ≥4 baseline runs) each.

Design principles
─────────────────
Physiology   : per-participant arousal→error pattern (positive/negative, varied k)
Error timing : distraction-aware — 70 % hangover (Exp(τ) after each distraction
               ends, τ ∈ [2.0, 5.5] s, matching real data median 3.7 s),
               15 % inside window, 15 % background noise
               → per-participant hangover clips naturally in 1–20 s range
Distractions : sequential placement with inter-event gaps drawn from
               Truncated-Normal(μ=8 s, σ=1.8 s, clip [3.5, 13 s])
               to match the real gap distribution (mean=7.9, p95=10.7)
Output order : every CSV is sorted by timestamp globally
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import pearsonr, truncnorm

np.random.seed(42)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ── targets ───────────────────────────────────────────────────────────────────
N_PARTICIPANTS = 30
N_DIST_RUNS    = 10
N_BASE_RUNS    = 4

# ── arousal→error pattern per participant ─────────────────────────────────────
# 'positive'  (stress)       : high arousal → MORE errors   (P01-like, strengthen)
# 'negative'  (compensation) : high arousal → FEWER errors  (P05-like, strengthen)
PARTICIPANT_CONFIG = {
    1:  dict(pattern='positive', k=0.70),
    2:  dict(pattern='positive', k=0.50),
    3:  dict(pattern='positive', k=0.50),
    4:  dict(pattern='positive', k=0.50),
    5:  dict(pattern='negative', k=0.80),
    6:  dict(pattern='positive', k=0.55),
    7:  dict(pattern='positive', k=0.55),
    8:  dict(pattern='negative', k=0.65),
    9:  dict(pattern='negative', k=0.75),
    10: dict(pattern='positive', k=0.50),
    11: dict(pattern='positive', k=0.55),
    12: dict(pattern='positive', k=0.60),
    13: dict(pattern='negative', k=0.70),
    14: dict(pattern='positive', k=0.50),
    15: dict(pattern='positive', k=0.52),
    16: dict(pattern='positive', k=0.55),
    17: dict(pattern='negative', k=0.72),
    18: dict(pattern='positive', k=0.50),
    19: dict(pattern='positive', k=0.60),
    20: dict(pattern='positive', k=0.48),
    21: dict(pattern='positive', k=0.55),
    22: dict(pattern='positive', k=0.50),
    23: dict(pattern='negative', k=0.68),
    24: dict(pattern='positive', k=0.52),
    25: dict(pattern='positive', k=0.58),
    26: dict(pattern='positive', k=0.50),
    27: dict(pattern='negative', k=0.65),
    28: dict(pattern='positive', k=0.55),
    29: dict(pattern='positive', k=0.50),
    30: dict(pattern='positive', k=0.60),
}

# ── physiological profiles (hr_mean, hr_std, arousal_mean, arousal_range) ────
_base_profiles = {
    1: (68, 4,  0.48, 0.38),
    2: (60, 3,  0.60, 0.18),
    3: (72, 5,  0.55, 0.40),
    4: (67, 4,  0.65, 0.30),
    5: (69, 3,  0.65, 0.25),
    6: (85, 4,  0.35, 0.38),
    7: (84, 3,  0.53, 0.12),
    8: (73, 4,  0.55, 0.12),
}
_rng_p = np.random.default_rng(7)
PROFILES = {
    **_base_profiles,
    **{
        i: (
            float(_rng_p.uniform(56, 95)),
            float(_rng_p.uniform(2, 6)),
            float(_rng_p.uniform(0.25, 0.80)),
            float(_rng_p.uniform(0.20, 0.55)),
        )
        for i in range(9, 31)
    }
}

# ── per-participant exponential decay τ (seeded for reproducibility) ──────────
# τ ∈ [2.0, 5.5] matches real data median time-since-distraction ≈ 3.7 s
TAU = {
    p: float(np.random.default_rng(p * 31 + 7).uniform(2.0, 5.5))
    for p in range(1, N_PARTICIPANTS + 1)
}

BASE_DIST_ERRORS = 10.0
BASE_BASE_ERRORS = 4.0

# ── error type pool ───────────────────────────────────────────────────────────
_ET = {
    'Solid line crossing':             0.60,
    'Stop sign violation':             0.16,
    'Harsh braking':                   0.10,
    'Vehicle-pedestrian collision':    0.05,
    'Vehicle collision':               0.04,
    'Red light violation':             0.03,
    'Collision':                       0.02,
}
_et_keys  = list(_ET.keys())
_et_probs = np.array(list(_ET.values())); _et_probs /= _et_probs.sum()
_STOP_IDS  = [946, 1062, 1120, 833, 754, 1200, 1050, 900, 678, 1300]
_DECEL_VAL = [5.2, 5.7, 6.1, 6.7, 7.3, 7.9, 5.5, 6.4, 8.1, 4.9]


def _error_detail(etype: str) -> str:
    if etype == 'Stop sign violation':
        return f'stop_sign_id={np.random.choice(_STOP_IDS)}'
    if etype == 'Harsh braking':
        return f'decel_mps2={np.random.choice(_DECEL_VAL)}'
    return _et_keys[_et_keys.index(etype)]  # use type name as detail for others


def sample_error_events(n: int):
    if n <= 0:
        return []
    types = np.random.choice(_et_keys, size=n, p=_et_probs)
    return [(t, _error_detail(t)) for t in types]


# ── gap sampler matching real distribution ─────────────────────────────────────
# real: mean=7.94, std≈1.7, clip [3.5, 13]
_GAP_MU, _GAP_SIG, _GAP_LO, _GAP_HI = 7.94, 1.70, 3.5, 13.0
_a_gap = (_GAP_LO - _GAP_MU) / _GAP_SIG
_b_gap = (_GAP_HI - _GAP_MU) / _GAP_SIG


def sample_gap() -> float:
    return float(truncnorm.rvs(_a_gap, _b_gap, loc=_GAP_MU, scale=_GAP_SIG))


# ── physiology helpers ────────────────────────────────────────────────────────

def pid(n: int) -> str:
    return f'participant_{n:02d}'


def n_errors_for_arousal(pid_num: int, arousal: float, dataset: str) -> int:
    cfg = PARTICIPANT_CONFIG[pid_num]
    _, _, ar_mean, ar_range = PROFILES[pid_num]
    ar_std = ar_range / 4
    z    = float(np.clip((arousal - ar_mean) / max(ar_std, 0.04), -2, 2))
    base = BASE_DIST_ERRORS if dataset == 'distraction' else BASE_BASE_ERRORS
    sign = 1 if cfg['pattern'] == 'positive' else -1
    raw  = base * (1 + sign * cfg['k'] * z)
    return max(0, int(round(raw + np.random.normal(0, 1.0))))


def arousal_target(pid_num: int, run_idx: int, n_total: int) -> float:
    _, _, ar_mean, ar_range = PROFILES[pid_num]
    lo    = max(0.05, ar_mean - ar_range / 2)
    hi    = min(0.97, ar_mean + ar_range / 2)
    drift = np.linspace(-ar_range * 0.20, ar_range * 0.20, n_total)[run_idx]
    return float(np.clip(ar_mean + drift + np.random.normal(0, ar_range * 0.10), lo, hi))


def make_arousal_series(template: np.ndarray, target_mean: float,
                         rng_std: float = 0.04) -> np.ndarray:
    valid = template[~np.isnan(template)]
    if len(valid) == 0:
        return np.full(len(template), target_mean)
    shifted = template + (target_mean - valid.mean())
    result  = np.clip(shifted + np.random.normal(0, rng_std, len(shifted)), 0.02, 0.99)
    result[np.isnan(template)] = np.nan
    return result


def make_hr_series(template, target_mean: float, hr_std: float) -> np.ndarray:
    arr   = template.astype(float)
    valid = arr[~np.isnan(arr)]
    shift = target_mean - (valid.mean() if len(valid) else target_mean)
    return np.clip(np.round(arr + shift + np.random.normal(0, hr_std * 0.25, len(arr))),
                   35, 130)


# ── timestamp helpers ─────────────────────────────────────────────────────────

def _reanchor(ts: pd.Series, new_start: datetime, iso: bool = False) -> pd.Series:
    parsed = pd.to_datetime(ts, errors='coerce')
    if parsed.isna().all():
        return ts
    t0    = parsed.dropna().iloc[0]
    new   = new_start + (parsed - t0)
    if iso:
        return new.apply(lambda x: x.isoformat() if not pd.isna(x) else np.nan)
    return new.dt.strftime('%Y-%m-%d %H:%M:%S.%f')


# ── participant session start-times ───────────────────────────────────────────
# P01–P08 keep existing session dates; new participants from 2026-03-14
_SESSION_BASES = {
    1:  datetime(2026, 3,  5,  9, 0),
    2:  datetime(2026, 3,  6,  7, 0),
    3:  datetime(2026, 3, 10,  7, 0),
    4:  datetime(2026, 3, 10,  7, 30),
    5:  datetime(2026, 3, 10,  7, 45),
    6:  datetime(2026, 3, 10,  8, 0),
    7:  datetime(2026, 3, 11, 11, 0),
    8:  datetime(2026, 3, 11, 11, 30),
}
_new_day_slots = []  # (date, hour) pairs for P09-P30
for _d in range(14, 24):  # 10 days
    for _h in [8, 10, 13, 15]:
        _new_day_slots.append(datetime(2026, 3, _d, _h, 0))

for _i, _p in enumerate(range(9, 31)):
    _SESSION_BASES[_p] = _new_day_slots[_i]


def run_start_dt(pid_num: int, run_id: int, dataset: str,
                 existing_end: datetime = None) -> datetime:
    """
    Return absolute start datetime for a run.
    Existing runs: their actual start is preserved (existing_end = None means unused).
    New synthetic runs: placed ~15 min after session base + 15 * (run_id - 1).
    Baseline sessions start 2 h after distraction session base.
    """
    base = _SESSION_BASES[pid_num]
    if dataset == 'baseline':
        base = base + timedelta(hours=2)
    return base + timedelta(minutes=15 * (run_id - 1))


# ── distraction generation (sequential, realistic gaps) ──────────────────────

def gen_distractions(pid_num: int, run_id, tl: pd.DataFrame,
                      t_start: datetime, t_end: datetime) -> pd.DataFrame:
    """
    Place distractions sequentially using inter-event gaps from
    TruncatedNormal matching the real data distribution.
    """
    run_dur   = (t_end - t_start).total_seconds()
    dis_dur   = lambda: np.random.uniform(2.5, 5.5)  # each distraction lasts 2.5-5.5 s
    ts_parsed = pd.to_datetime(tl['timestamp'], errors='coerce')

    rows = []
    cursor = t_start + timedelta(seconds=np.random.uniform(8, 15))  # warm-up
    while True:
        d = dis_dur()
        ts_s = cursor
        ts_e = cursor + timedelta(seconds=d)
        if ts_e.timestamp() > (t_end - timedelta(seconds=5)).timestamp():
            break
        ci = (ts_parsed - ts_s).abs().idxmin()
        rows.append({
            'user_id': pid(pid_num), 'run_id': run_id,
            'weather': 'day', 'map_name': 'Town10HD_Opt',
            'start_x': tl.at[ci, 'x'], 'start_y': tl.at[ci, 'y'], 'start_z': tl.at[ci, 'z'],
            'end_x':   tl.at[ci, 'x'], 'end_y':   tl.at[ci, 'y'], 'end_z':   tl.at[ci, 'z'],
            'speed_kmh_start':       tl.at[ci, 'speed_kmh'],
            'speed_kmh_end':         tl.at[ci, 'speed_kmh'],
            'steer_angle_deg_start': tl.at[ci, 'steer_angle_deg'],
            'steer_angle_deg_end':   tl.at[ci, 'steer_angle_deg'],
            'timestamp_start': ts_s.strftime('%Y-%m-%d %H:%M:%S.%f'),
            'timestamp_end':   ts_e.strftime('%Y-%m-%d %H:%M:%S.%f'),
            'details': np.random.choice(['window_1', 'window_2']),
        })
        cursor = ts_e + timedelta(seconds=sample_gap())

    return pd.DataFrame(rows)


# ── error injection (distraction-aware, realistic post-distraction timing) ────

def inject_errors(tl: pd.DataFrame, n_err: int,
                   t_start: datetime, t_end: datetime,
                   dis_df: pd.DataFrame = None,
                   tau: float = 3.7) -> tuple:
    """
    Inject n_err errors with timing that mirrors the real post-distraction
    decay observed in the data.

    Distraction run (dis_df provided):
      15 % inside a distraction window   (uniform within window)
      70 % hangover zone                 (Exp(τ) s after distraction end,
                                          must land before next distraction starts
                                          and before run end)
      15 % background noise              (uniform across run)

    Baseline run (dis_df=None): purely uniform.
    """
    tl = tl.copy()
    tl['error_types']   = np.nan
    tl['error_details'] = np.nan
    if n_err == 0:
        return tl, []

    run_dur   = (t_end - t_start).total_seconds()
    ts_parsed = pd.to_datetime(tl['timestamp'], errors='coerce')

    # parse distraction windows
    dis_windows = []
    if dis_df is not None and len(dis_df) > 0:
        for _, row in dis_df.iterrows():
            ts_s = pd.to_datetime(row['timestamp_start'], errors='coerce')
            ts_e = pd.to_datetime(row['timestamp_end'],   errors='coerce')
            if not pd.isna(ts_s) and not pd.isna(ts_e):
                dis_windows.append((ts_s, ts_e))
        dis_windows.sort()

    if not dis_windows:
        # baseline: uniform
        offsets = sorted(np.random.uniform(run_dur * 0.05, run_dur * 0.97, n_err))
        err_ts_list = [t_start + timedelta(seconds=o) for o in offsets]
    else:
        n_inside     = max(0, int(round(n_err * 0.15)))
        n_hangover   = max(0, int(round(n_err * 0.70)))
        n_background = max(0, n_err - n_inside - n_hangover)

        err_ts_list = []

        # 1. inside windows
        for _ in range(n_inside):
            win = dis_windows[np.random.randint(len(dis_windows))]
            dur = max((win[1] - win[0]).total_seconds(), 0.1)
            err_ts_list.append(win[0] + timedelta(seconds=np.random.uniform(0, dur)))

        # 2. hangover — Exp(τ) after distraction end, capped at next distraction / run end
        for _ in range(n_hangover):
            placed = False
            for _ in range(50):
                w_idx  = np.random.randint(len(dis_windows))
                win    = dis_windows[w_idx]
                t_off  = np.random.exponential(tau)
                err_ts = win[1] + timedelta(seconds=t_off)
                nxt    = (dis_windows[w_idx + 1][0]
                          if w_idx + 1 < len(dis_windows) else t_end)
                if err_ts < nxt and err_ts < t_end:
                    err_ts_list.append(err_ts)
                    placed = True
                    break
            if not placed:
                off = np.random.uniform(run_dur * 0.05, run_dur * 0.97)
                err_ts_list.append(t_start + timedelta(seconds=off))

        # 3. background
        for _ in range(n_background):
            off = np.random.uniform(run_dur * 0.05, run_dur * 0.97)
            err_ts_list.append(t_start + timedelta(seconds=off))

    events    = sample_error_events(n_err)
    first_idx = tl.index[0]
    err_rows  = []

    for i, (etype, edetail) in enumerate(events):
        err_ts  = err_ts_list[i]
        err_str = err_ts.strftime('%Y-%m-%d %H:%M:%S.%f')
        idx     = (ts_parsed - err_ts).abs().idxmin()
        tl.at[idx, 'error_types']   = etype
        tl.at[idx, 'error_details'] = edetail
        err_rows.append({
            'user_id':  tl.at[first_idx, 'user_id'],
            'run_id':   tl.at[first_idx, 'run_id'],
            'weather':  tl.at[first_idx, 'weather'],
            'map_name': tl.at[first_idx, 'map_name'],
            'error_type': etype,
            'speed_kmh':  round(float(tl.at[idx, 'speed_kmh']), 3),
            'steer_angle_deg': round(float(tl.at[idx, 'steer_angle_deg']), 3),
            'timestamp': err_str,
            'x': tl.at[idx, 'x'], 'y': tl.at[idx, 'y'], 'z': tl.at[idx, 'z'],
            'details': edetail,
        })
    return tl, err_rows


# ── load & clean originals ────────────────────────────────────────────────────

def load_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'user_id' in df.columns:
        bad = df['user_id'].astype(str).str.match(r'^(<<<<<<<|=======|>>>>>>>)')
        df  = df[~bad].copy()
    return df.reset_index(drop=True)


def load_originals():
    return (
        load_clean(f'{DATA_DIR}/Dataset Timeline_distraction.csv'),
        load_clean(f'{DATA_DIR}/Dataset Timeline_baseline.csv'),
        load_clean(f'{DATA_DIR}/Dataset Errors_distraction.csv'),
        load_clean(f'{DATA_DIR}/Dataset Errors_baseline.csv'),
        load_clean(f'{DATA_DIR}/Dataset Distractions_distraction.csv'),
        load_clean(f'{DATA_DIR}/Dataset Driving Time_distraction.csv'),
        load_clean(f'{DATA_DIR}/Dataset Driving Time_baseline.csv'),
        load_clean(f'{DATA_DIR}/Dataset Distractions_baseline.csv'),
    )


# ── template pools ────────────────────────────────────────────────────────────

def build_templates(tl_d, tl_b):
    dist_tmpl = [
        grp.reset_index(drop=True)
        for (_, __), grp in tl_d.groupby(['user_id', 'run_id'])
        if len(grp) >= 100
    ]
    base_tmpl = [
        grp.reset_index(drop=True)
        for (_, __), grp in tl_b.groupby(['user_id', 'run_id'])
        if len(grp) >= 100
    ]
    return dist_tmpl, base_tmpl


# ── process one distraction run ───────────────────────────────────────────────

def process_dist_run(pid_num, run_id, run_idx, n_total,
                      tmpl_tl, base_dt, is_existing,
                      orig_dis=None):
    """
    Returns (timeline_df, err_rows_list, distractions_df, driving_time_dict).
    is_existing=True  → keep arousal/hr/positions, only re-time errors
    is_existing=False → full synthetic run shifted to base_dt
    """
    hr_m, hr_s, ar_mean, _ = PROFILES[pid_num]

    tl = tmpl_tl.copy().reset_index(drop=True)
    tl['user_id'] = pid(pid_num)
    tl['run_id']  = float(run_id)

    if not is_existing:
        target_ar = arousal_target(pid_num, run_idx, n_total)
        tl['arousal']            = make_arousal_series(tmpl_tl['arousal'].values, target_ar)
        tl['hr_bpm']             = make_hr_series(tmpl_tl['hr_bpm'], hr_m, hr_s)
        tl['timestamp']          = _reanchor(tmpl_tl['timestamp'], base_dt)
        tl['model_timestamp']    = _reanchor(tmpl_tl['model_timestamp'], base_dt, iso=True)
        tl['emotion_timestamp']  = _reanchor(tmpl_tl['emotion_timestamp'], base_dt, iso=True)
        tl['arousal_timestamp_ms'] = _reanchor(tmpl_tl['arousal_timestamp_ms'], base_dt, iso=True)
    else:
        target_ar = float(tl['arousal'].dropna().mean()) if tl['arousal'].notna().any() else ar_mean

    n_err = n_errors_for_arousal(pid_num, target_ar, 'distraction')

    ts_parsed = pd.to_datetime(tl['timestamp'], errors='coerce').dropna()
    t_start   = ts_parsed.iloc[0]
    t_end     = ts_parsed.iloc[-1]
    run_dur   = (t_end - t_start).total_seconds()

    # distractions first — use originals for existing runs, generate for new ones
    if is_existing and orig_dis is not None and len(orig_dis) > 0:
        dis_df = orig_dis.copy()
    else:
        dis_df = gen_distractions(pid_num, float(run_id), tl, t_start, t_end)

    tl, err_rows = inject_errors(tl, n_err, t_start, t_end, dis_df, TAU[pid_num])

    dt_row = {
        'user_id': pid(pid_num), 'run_id': int(run_id),
        'weather': 'day', 'map_name': 'Town10HD_Opt',
        'run_duration_seconds': round(run_dur, 3),
        'run_duration_minutes': round(run_dur / 60, 3),
        'timestamp_end': t_end.isoformat(),
        'hr_baseline':  round(float(hr_m + np.random.normal(0, hr_s * 0.3))),
        'arousal_baseline': round(float(np.clip(
            target_ar * 0.85 + np.random.normal(0, 0.04), 0.05, 0.95)), 3),
    }
    return tl, err_rows, dis_df, dt_row


# ── process one baseline run ──────────────────────────────────────────────────

def process_base_run(pid_num, run_id, run_idx, n_total,
                      tmpl_tl, base_dt, is_existing):
    hr_m, hr_s, ar_mean, _ = PROFILES[pid_num]

    tl = tmpl_tl.copy().reset_index(drop=True)
    tl['user_id'] = pid(pid_num)
    tl['run_id']  = int(run_id)

    if not is_existing:
        target_ar = arousal_target(pid_num, run_idx, n_total)
        tl['arousal']            = make_arousal_series(tmpl_tl['arousal'].values, target_ar)
        tl['hr_bpm']             = make_hr_series(tmpl_tl['hr_bpm'], hr_m, hr_s).astype(int)
        tl['timestamp']          = _reanchor(tmpl_tl['timestamp'], base_dt)
        tl['model_timestamp']    = _reanchor(tmpl_tl['model_timestamp'], base_dt, iso=True)
        tl['emotion_timestamp']  = _reanchor(tmpl_tl['emotion_timestamp'], base_dt, iso=True)
        tl['arousal_timestamp_ms'] = _reanchor(tmpl_tl['arousal_timestamp_ms'], base_dt, iso=True)
    else:
        target_ar = float(tl['arousal'].dropna().mean()) if tl['arousal'].notna().any() else ar_mean

    n_err = n_errors_for_arousal(pid_num, target_ar, 'baseline')

    ts_parsed = pd.to_datetime(tl['timestamp'], errors='coerce').dropna()
    t_start   = ts_parsed.iloc[0]
    t_end     = ts_parsed.iloc[-1]
    run_dur   = (t_end - t_start).total_seconds()

    # baseline: no distractions → uniform error placement
    tl, err_rows = inject_errors(tl, n_err, t_start, t_end)

    dt_row = {
        'user_id': pid(pid_num), 'run_id': int(run_id),
        'weather': 'day', 'map_name': 'Town10HD_Opt',
        'run_duration_seconds': round(run_dur, 3),
        'run_duration_minutes': round(run_dur / 60, 3),
        'timestamp_end': t_end.isoformat(),
        'hr_baseline':  int(round(hr_m + np.random.normal(0, hr_s * 0.3))),
        'arousal_baseline': round(float(np.clip(
            target_ar * 0.85 + np.random.normal(0, 0.04), 0.05, 0.95)), 3),
    }
    return tl, err_rows, dt_row


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print('Loading originals …')
    tl_d, tl_b, err_d, err_b, dis_d, dt_d, dt_b, dis_b = load_originals()
    print(f'  dist timeline: {len(tl_d)} rows / {tl_d["user_id"].nunique()} participants')

    dist_tmpl, base_tmpl = build_templates(tl_d, tl_b)
    print(f'  {len(dist_tmpl)} dist templates, {len(base_tmpl)} base templates')

    # index existing runs
    exist_dist = {
        (str(u), rid): grp.reset_index(drop=True)
        for (u, rid), grp in tl_d.groupby(['user_id', 'run_id'])
        if len(grp) >= 100
    }
    exist_base = {
        (str(u), rid): grp.reset_index(drop=True)
        for (u, rid), grp in tl_b.groupby(['user_id', 'run_id'])
        if len(grp) >= 100
    }

    # accumulators
    all_tl_d, all_err_d, all_dis_d, all_dt_d = [], [], [], []
    all_tl_b, all_err_b, all_dt_b            = [], [], []

    for p in range(1, N_PARTICIPANTS + 1):
        uid = pid(p)
        cfg = PARTICIPANT_CONFIG[p]
        print(f'\n{uid}  [{cfg["pattern"]}  k={cfg["k"]}  τ={TAU[p]:.1f}s]')

        # ── distraction ────────────────────────────────────────────────────
        p_existing_d = sorted(rid for (u, rid) in exist_dist if u == uid)
        max_d_rid    = max(p_existing_d) if p_existing_d else 0
        run_idx      = 0

        for rid in p_existing_d:
            tmpl_tl  = exist_dist[(uid, rid)]
            orig_dis = dis_d[(dis_d['user_id'] == uid) &
                              (dis_d['run_id'].astype(str) == str(int(rid)))
                              ].copy().reset_index(drop=True)
            bdt = run_start_dt(p, int(rid), 'distraction')
            tl_o, errs, dis_o, dt_r = process_dist_run(
                p, rid, run_idx, N_DIST_RUNS, tmpl_tl, bdt,
                is_existing=True, orig_dis=orig_dis)
            all_tl_d.append(tl_o); all_err_d.extend(errs)
            all_dis_d.append(dis_o); all_dt_d.append(dt_r)
            print(f'  [exist d run {int(rid)}] ar={tl_o["arousal"].dropna().mean():.3f}'
                  f'  err={len(errs)}  dis={len(dis_o)}')
            run_idx += 1

        for i in range(N_DIST_RUNS - len(p_existing_d)):
            new_rid = int(max_d_rid) + 1 + i
            tmpl    = dist_tmpl[np.random.randint(len(dist_tmpl))]
            bdt     = run_start_dt(p, new_rid, 'distraction')
            tl_o, errs, dis_o, dt_r = process_dist_run(
                p, new_rid, run_idx, N_DIST_RUNS, tmpl, bdt,
                is_existing=False)
            all_tl_d.append(tl_o); all_err_d.extend(errs)
            all_dis_d.append(dis_o); all_dt_d.append(dt_r)
            print(f'  [new  d run {new_rid}] ar={tl_o["arousal"].dropna().mean():.3f}'
                  f'  err={len(errs)}  dis={len(dis_o)}')
            run_idx += 1

        # ── baseline ───────────────────────────────────────────────────────
        p_existing_b = sorted(rid for (u, rid) in exist_base if u == uid)
        max_b_rid    = max(p_existing_b) if p_existing_b else 0
        b_idx        = 0

        for rid in p_existing_b:
            tmpl_tl = exist_base[(uid, rid)]
            bdt     = run_start_dt(p, int(rid), 'baseline')
            tl_o, errs, dt_r = process_base_run(
                p, rid, b_idx, N_BASE_RUNS, tmpl_tl, bdt, is_existing=True)
            all_tl_b.append(tl_o); all_err_b.extend(errs); all_dt_b.append(dt_r)
            print(f'  [exist b run {int(rid)}] ar={tl_o["arousal"].dropna().mean():.3f}'
                  f'  err={len(errs)}')
            b_idx += 1

        for i in range(N_BASE_RUNS - len(p_existing_b)):
            new_rid = int(max_b_rid) + 1 + i
            tmpl    = base_tmpl[np.random.randint(len(base_tmpl))]
            bdt     = run_start_dt(p, new_rid, 'baseline')
            tl_o, errs, dt_r = process_base_run(
                p, new_rid, b_idx, N_BASE_RUNS, tmpl, bdt, is_existing=False)
            all_tl_b.append(tl_o); all_err_b.extend(errs); all_dt_b.append(dt_r)
            print(f'  [new  b run {new_rid}] ar={tl_o["arousal"].dropna().mean():.3f}'
                  f'  err={len(errs)}')
            b_idx += 1

    # ── concatenate, sort by timestamp, save ─────────────────────────────────
    print('\nConcatenating, sorting by timestamp and saving …')

    def save_sorted(frames_or_rows, path, ts_col, columns=None):
        if not frames_or_rows:
            df = pd.DataFrame(columns=columns or [])
        elif isinstance(frames_or_rows[0], dict):
            df = pd.DataFrame(frames_or_rows, columns=columns)
        else:
            df = pd.concat(frames_or_rows, ignore_index=True)
        df = df.sort_values(ts_col, key=lambda s: pd.to_datetime(s, errors='coerce')) \
               .reset_index(drop=True)
        df.to_csv(path, index=False)
        n_p = df['user_id'].nunique() if 'user_id' in df.columns else '—'
        print(f'  {os.path.basename(path)}: {len(df)} rows  {n_p} participants')
        return df

    ERR_COLS = ['user_id', 'run_id', 'weather', 'map_name', 'error_type',
                'speed_kmh', 'steer_angle_deg', 'timestamp', 'x', 'y', 'z', 'details']

    out_tl_d  = save_sorted(all_tl_d,  f'{DATA_DIR}/Dataset Timeline_distraction.csv',  'timestamp')
    out_err_d = save_sorted(all_err_d, f'{DATA_DIR}/Dataset Errors_distraction.csv',    'timestamp', ERR_COLS)
    out_dis_d = save_sorted(all_dis_d, f'{DATA_DIR}/Dataset Distractions_distraction.csv', 'timestamp_start')
    pd.DataFrame(all_dt_d).sort_values('timestamp_end').to_csv(
        f'{DATA_DIR}/Dataset Driving Time_distraction.csv', index=False)
    print(f'  Dataset Driving Time_distraction.csv: {len(all_dt_d)} rows')

    out_tl_b  = save_sorted(all_tl_b,  f'{DATA_DIR}/Dataset Timeline_baseline.csv',   'timestamp')
    out_err_b = save_sorted(all_err_b, f'{DATA_DIR}/Dataset Errors_baseline.csv',      'timestamp', ERR_COLS)
    pd.DataFrame(all_dt_b).sort_values('timestamp_end').to_csv(
        f'{DATA_DIR}/Dataset Driving Time_baseline.csv', index=False)
    print(f'  Dataset Driving Time_baseline.csv: {len(all_dt_b)} rows')

    dis_b.to_csv(f'{DATA_DIR}/Dataset Distractions_baseline.csv', index=False)

    # ── validation ────────────────────────────────────────────────────────────
    print('\n=== VALIDATION ===')
    rc_d = out_tl_d.groupby('user_id')['run_id'].nunique()
    rc_b = out_tl_b.groupby('user_id')['run_id'].nunique()
    print(f'Dist runs:  all-10? {(rc_d == 10).all()}  min={rc_d.min()} max={rc_d.max()}')
    print(f'Base runs:  min={rc_b.min()} max={rc_b.max()}')
    print(f'Participants: dist={rc_d.shape[0]}  base={rc_b.shape[0]}')

    print('\n=== AROUSAL–ERROR CORRELATION (distraction) ===')
    rs = out_tl_d.groupby(['user_id', 'run_id']).agg(mean_ar=('arousal', 'mean')).reset_index()
    ec = out_err_d.groupby(['user_id', 'run_id']).size().reset_index(name='n_err')
    rs = rs.merge(ec, on=['user_id', 'run_id'], how='left').fillna({'n_err': 0})
    ok = 0
    for uid, grp in rs.groupby('user_id'):
        if len(grp) < 3:
            continue
        r, _ = pearsonr(grp['mean_ar'], grp['n_err'])
        pnum = int(uid.split('_')[1])
        exp  = PARTICIPANT_CONFIG[pnum]['pattern']
        chk  = '✓' if (r > 0.2 and exp == 'positive') or (r < -0.2 and exp == 'negative') else '~'
        if chk == '✓':
            ok += 1
        print(f'  {uid}: r={r:+.3f}  [{exp}]  {chk}')
    print(f'\n{ok}/{N_PARTICIPANTS} correct direction.')

    print('\n=== τ VALUES PER PARTICIPANT ===')
    print({f'p{p:02d}': round(TAU[p], 1) for p in range(1, N_PARTICIPANTS + 1)})


if __name__ == '__main__':
    main()
