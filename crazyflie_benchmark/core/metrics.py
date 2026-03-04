import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def calculate_step_metrics(df: pd.DataFrame, step_start_time: float, step_duration: float, channel: str) -> Dict[str, float]:
    """
    Calculate step response metrics.
    """
    var_map = {
        'roll': 'stabilizer.roll',
        'pitch': 'stabilizer.pitch',
        'thrust': 'stabilizer.thrust',
        'yaw': 'stabilizer.yaw'
    }

    var_name = var_map.get(channel)
    if not var_name:
        # Try to find a matching column if exact match fails
        # e.g. if channel is 'roll', look for any column ending in .roll
        for col in df.columns:
            if col.endswith(f".{channel}"):
                var_name = col
                break

    if not var_name or var_name not in df.columns:
        return {'error': f"Variable for {channel} not found"}

    # Slice data
    mask = (df['timestamp'] >= step_start_time) & (df['timestamp'] <= step_start_time + step_duration)
    segment = df[mask].copy()

    # Drop NaNs for the variable of interest
    segment = segment.dropna(subset=[var_name])

    if segment.empty:
        return {'error': "No data in time range"}

    times = segment['timestamp'].values - step_start_time
    values = segment[var_name].values

    # Basic stats
    initial_value = values[0]
    # Steady state from last 20% of data
    steady_state_idx = int(len(values) * 0.8)
    final_value = np.mean(values[steady_state_idx:])

    step_size = final_value - initial_value

    metrics = {
        'initial_value': initial_value,
        'final_value': final_value,
        'step_size': step_size
    }

    if abs(step_size) < 0.1: # Threshold for significant step
        return metrics

    # Rise time (10% to 90%)
    target_10 = initial_value + 0.1 * step_size
    target_90 = initial_value + 0.9 * step_size

    t10 = None
    t90 = None

    # Simple search
    for t, v in zip(times, values):
        if t10 is None:
            if (step_size > 0 and v >= target_10) or (step_size < 0 and v <= target_10):
                t10 = t
        if t90 is None:
            if (step_size > 0 and v >= target_90) or (step_size < 0 and v <= target_90):
                t90 = t

    if t10 is not None and t90 is not None:
        metrics['rise_time'] = t90 - t10

    # Overshoot
    if step_size > 0:
        peak_value = np.max(values)
        overshoot = max(0, peak_value - final_value)
    else:
        peak_value = np.min(values)
        overshoot = max(0, final_value - peak_value)

    metrics['overshoot_pct'] = (overshoot / abs(step_size)) * 100.0

    return metrics
