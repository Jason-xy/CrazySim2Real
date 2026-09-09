"""Windowed SISO closed-loop estimates and explicitly qualified metrics."""

from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import signal

from .protocol import AnalysisConfig, CHANNELS, RESULT_SCHEMA_VERSION, RunDataset, ValidationError, load_run, signal_series


METRIC_NAMES = (
    "dc_gain", "steady_state_error", "rise_time_s", "peak_time_s",
    "overshoot_pct", "settling_time_s",
)
LIMITATIONS = [
    "Empirical SISO closed-loop estimate, not isolated plant dynamics.",
    "Correlated commands, feedback noise and operating-point changes can bias the estimate.",
    "Coherence is a quality diagnostic, not proof of an unbiased or uncoupled model.",
    "Step reconstruction is band-limited; no claims are made outside the reliable band.",
    "Finite-window settling is not a proof of global stability.",
]


def _pieces(t, values, max_gap):
    split = np.flatnonzero(np.diff(t) > max_gap) + 1
    return [(t[idx], values[idx]) for idx in np.split(np.arange(len(t)), split) if len(idx) >= 2]


def _uniform(t, values, start, end, native_fs, fs, hold):
    native_t = start + np.arange(int(np.floor((end - start) * native_fs)) + 1) / native_fs
    if hold:
        # Roundoff at an exactly coincident sample must not add a one-sample delay.
        uniform = values[np.maximum(0, np.searchsorted(t, native_t + 1e-8 / native_fs, side="right") - 1)]
    else:
        uniform = np.interp(native_t, t, values)
    if native_fs > fs * (1 + 1e-6):
        ratio = Fraction(float(fs / native_fs)).limit_denominator(10000)
        uniform = signal.resample_poly(uniform, ratio.numerator, ratio.denominator, padtype="line")
        actual_fs = native_fs * ratio.numerator / ratio.denominator
        native_t = start + np.arange(len(uniform)) / actual_fs
    return native_t, uniform


def _analysis_intervals(dataset: RunDataset, config: AnalysisConfig):
    start, end = float(dataset.samples.t_s.iloc[0]), float(dataset.samples.t_s.iloc[-1])
    intervals = config.analysis_intervals_s
    if intervals is not None and any(lo < start or hi > end for lo, hi in intervals):
        raise ValidationError(f"{dataset.csv_path}: analysis intervals must lie within the recorded time range")
    return intervals if intervals is not None else ((start, end),)


def prepare_channel(dataset: RunDataset, channel: str, config: AnalysisConfig):
    """Return continuous aligned blocks; never interpolate over a dropout."""
    r_col, y_col = CHANNELS[channel]
    rt, r = signal_series(dataset, r_col)
    yt, y = signal_series(dataset, y_col)
    if min(len(rt), len(yt)) < 3:
        return [], {}, "too_few_samples"
    r_dt, y_dt = float(np.median(np.diff(rt))), float(np.median(np.diff(yt)))
    r_fs, y_fs = 1 / r_dt, 1 / y_dt
    fs = min(r_fs, y_fs, config.sample_rate_hz or np.inf)
    details = {
        "reference_rate_hz": r_fs, "measurement_rate_hz": y_fs, "sample_rate_hz": fs,
        "anti_alias_downsampled": r_fs > fs * 1.000001 or y_fs > fs * 1.000001,
        "reference_long_gaps": int(np.sum(np.diff(rt) > config.max_gap_periods * r_dt)),
        "measurement_long_gaps": int(np.sum(np.diff(yt) > config.max_gap_periods * y_dt)),
    }
    r_pieces = _pieces(rt, r, config.max_gap_periods * r_dt)
    y_pieces = _pieces(yt, y, config.max_gap_periods * y_dt)
    intervals = _analysis_intervals(dataset, config)
    blocks = []
    for a, rv in r_pieces:
        for b, yv in y_pieces:
            overlap_start, overlap_end = max(a[0], b[0]), min(a[-1], b[-1])
            if overlap_end <= overlap_start:
                continue
            # Unwrap each uninterrupted piece separately, never across missing data.
            if channel.startswith("angle."):
                rv_local = np.rad2deg(np.unwrap(np.deg2rad(rv)))
                yv_local = np.rad2deg(np.unwrap(np.deg2rad(yv)))
            else:
                rv_local, yv_local = rv, yv
            for lo, hi in intervals:
                start, end = max(lo, overlap_start), min(hi, overlap_end)
                if end - start < 3 / fs:
                    continue
                ar, vr = _uniform(a, rv_local, start, end, r_fs, fs, True)
                ay, vy = _uniform(b, yv_local, start, end, y_fs, fs, False)
                end = min(end, ar[-1], ay[-1])
                grid = start + np.arange(int(np.floor((end - start) * fs)) + 1) / fs
                rr, yy = np.interp(grid, ar, vr), np.interp(grid, ay, vy)
                if details["anti_alias_downsampled"]:
                    # Discard FIR boundary transients without changing timestamps.
                    trim = min(int(np.ceil(0.1 * fs)), len(grid) // 4)
                    if trim:
                        grid, rr, yy = grid[trim:-trim], rr[trim:-trim], yy[trim:-trim]
                if len(grid) >= 3:
                    blocks.append((grid, rr, yy))
    details["continuous_blocks"] = len(blocks)
    details["selected_samples"] = sum(len(b[0]) for b in blocks)
    return blocks, details, None if blocks else "no_continuous_overlap"


def lowpass_mask(reliable):
    """Only a contiguous band extending to DC can reconstruct a lowpass step."""
    reliable = np.asarray(reliable, dtype=bool)
    stop = np.flatnonzero(~reliable)
    end = int(stop[0]) if len(stop) else len(reliable)
    mask = np.zeros(len(reliable), dtype=bool)
    if end >= 3:
        mask[:end] = True
    return mask


def reconstruct_step(transfer, mask, fs, nfft, response_s):
    """Zero-phase band limiting with the negative-lag integral retained.

    Simply discarding the wrapped negative-lag impulse loses DC gain after
    low-pass filtering. Integrate the centered impulse first, then select t>=0.
    This does not shift a response or remove its physical delay.
    """
    mask = lowpass_mask(mask)
    if not mask.any():
        return None
    end = int(mask.sum())
    weights = mask.astype(float)
    if end < len(mask):
        taper_len = max(2, int(np.ceil(end * 0.2)))
        weights[end - taper_len:end] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, taper_len)))
    impulse = np.fft.fftshift(np.fft.irfft(np.asarray(transfer) * weights, n=nfft))
    center = nfft // 2
    count = min(int(round(response_s * fs)) + 1, nfft - center)
    step = np.cumsum(impulse)[center:center + count]
    energy = float(np.sum(impulse ** 2))
    return {
        "time_s": np.arange(count, dtype=float) / fs,
        "values": step,
        "band_hz": [0.0, (end - 1) * fs / nfft],
        "negative_lag_energy_fraction": float(np.sum(impulse[:center] ** 2) / energy) if energy else 0.0,
    }


def step_metrics(time_s, values, config: AnalysisConfig):
    metrics = {key: None for key in METRIC_NAMES}
    reasons = {}
    t, y = np.asarray(time_s), np.asarray(values)
    if len(t) < 10:
        return metrics, {key: "response_too_short" for key in metrics}
    tail_start = max(0, int(len(y) * 0.8))
    tail = y[tail_start:]
    gain = float(np.mean(tail))
    if abs(gain) < 1e-8:
        return metrics, {key: "near_zero_final_response" for key in metrics}
    tolerance = config.settling_band * abs(gain)
    drift = abs(float(np.polyfit(t[tail_start:] - t[tail_start], tail, 1)[0])) * (
        t[-1] - t[tail_start]
    )
    if drift > tolerance or float(np.std(tail)) > tolerance:
        return metrics, {key: "not_settled_in_response_window" for key in metrics}
    metrics["dc_gain"] = gain
    metrics["steady_state_error"] = 1.0 - gain
    normalized = y / gain

    def crossing(level):
        if normalized[0] >= level:
            return None
        indices = np.flatnonzero((normalized[:-1] < level) & (normalized[1:] >= level))
        if not len(indices):
            return None
        i = indices[0]
        return float(t[i] + (t[i + 1] - t[i]) * (level - normalized[i]) / (
            normalized[i + 1] - normalized[i]
        ))

    t10, t90 = crossing(0.1), crossing(0.9)
    if t10 is not None and t90 is not None:
        metrics["rise_time_s"] = t90 - t10
    else:
        reasons["rise_time_s"] = "crossing_missing_or_below_time_bandwidth_resolution"
    peak = int(np.argmax(normalized))
    metrics["overshoot_pct"] = max(0.0, float(normalized[peak] - 1)) * 100
    if normalized[peak] > 1 + 1e-6 and 0 < peak < len(t) - 1:
        metrics["peak_time_s"] = float(t[peak])
    else:
        reasons["peak_time_s"] = "no_resolved_peak"
    outside = np.flatnonzero(np.abs(normalized - 1) > config.settling_band)
    last = int(outside[-1]) if len(outside) else -1
    if last < tail_start - 1:
        metrics["settling_time_s"] = float(t[last + 1])
    else:
        reasons["settling_time_s"] = "insufficient_sustained_settling"
    return metrics, reasons


def _analyze_channel(dataset, channel, config):
    result = {
        "status": "unavailable", "reasons": [],
        "metrics": {key: None for key in (*METRIC_NAMES, "tracking_bias", "tracking_rmse")},
        "metric_reasons": {}, "unit": "deg" if channel.startswith("angle.") else "deg/s",
        "warnings": [],
    }
    if channel not in dataset.channels:
        result["reasons"] = ["channel_not_recorded"]
        result["metric_reasons"] = {key: "channel_not_recorded" for key in result["metrics"]}
        return result
    blocks, preparation, error = prepare_channel(dataset, channel, config)
    result["preprocessing"] = preparation
    if error:
        result["status"], result["reasons"] = "insufficient_data", [error]
        result["metric_reasons"] = {key: error for key in result["metrics"]}
        return result
    fs = preparation["sample_rate_hz"]
    nfft = int(round(config.window_s * fs))
    result["spectral_settings"] = {"nfft": nfft, "window_s": nfft / fs, "sample_rate_hz": fs}
    if nfft < 8:
        result["status"], result["reasons"] = "insufficient_data", ["too_low_sample_rate"]
        result["metric_reasons"] = {key: "too_low_sample_rate" for key in result["metrics"]}
        return result
    hop = max(1, int(round(nfft * (1 - config.overlap))))
    can_window = nfft <= max(len(block[0]) for block in blocks)
    window = signal.windows.hann(nfft, sym=False) if can_window else np.empty(0)
    bins = nfft // 2 + 1 if can_window else 0
    pxx, pyy = np.zeros(bins), np.zeros(bins)
    pxy = np.zeros(bins, complex)
    accepted = rejected = 0
    errors, preview = [], []
    for t, r, y in blocks:
        error_values = y - r
        if channel.startswith("angle."):
            error_values = (error_values + 180) % 360 - 180
        errors.append(error_values)
        stride = max(1, int(np.ceil(len(t) / 2000)))
        preview.append({
            "time_s": t[::stride].tolist(), "reference": r[::stride].tolist(),
            "measurement": y[::stride].tolist(),
        })
        centered_r, centered_y = r - np.mean(r), y - np.mean(y)
        for start in range(0, len(r) - nfft + 1, hop):
            rr, yy = centered_r[start:start + nfft], centered_y[start:start + nfft]
            if np.var(rr) < config.min_input_variance:
                rejected += 1
                continue
            # Remove one operating-point offset per continuous block, not each
            # window's random mean (which biases the near-DC transfer estimate).
            R = np.fft.rfft(rr * window)
            Y = np.fft.rfft(yy * window)
            pxx += np.abs(R) ** 2
            pyy += np.abs(Y) ** 2
            pxy += np.conj(R) * Y
            accepted += 1
    error_values = np.concatenate(errors)
    result["metrics"].update(
        tracking_bias=float(np.mean(error_values)),
        tracking_rmse=float(np.sqrt(np.mean(error_values ** 2))),
    )
    result["time_series"] = preview
    result["spectral_settings"].update(hop_samples=hop, accepted_windows=accepted, rejected_windows=rejected)
    if accepted < config.min_windows:
        reason = "insufficient_excited_windows"
        result["status"], result["reasons"] = "insufficient_data", [reason]
        result["metric_reasons"] = {key: reason for key in METRIC_NAMES}
        return result
    scale = 1 / (accepted * fs * np.sum(window ** 2))
    one_sided = np.full(len(pxx), 2.0)
    one_sided[0] = 1.0
    if nfft % 2 == 0:
        one_sided[-1] = 1.0
    pxx, pyy, pxy = pxx * scale * one_sided, pyy * scale * one_sided, pxy * scale * one_sided
    if (
        not np.isfinite(pxx).all() or not np.isfinite(pyy).all() or not np.isfinite(pxy).all()
        or np.max(pxx) <= np.finfo(float).tiny
    ):
        reason = "zero_or_nonfinite_windowed_spectral_power"
        result["status"], result["reasons"] = "insufficient_data", [reason]
        result["metric_reasons"].update({key: reason for key in METRIC_NAMES})
        return result
    regularizer = config.regularization * float(np.max(pxx))
    transfer = pxy / (pxx + regularizer)
    coherence = np.clip(np.abs(pxy) ** 2 / np.maximum(pxx * pyy, np.finfo(float).tiny), 0, 1)
    reliable = (coherence >= config.coherence_min) & (pxx >= config.excitation_floor * np.max(pxx))
    freq = np.fft.rfftfreq(nfft, 1 / fs)
    result["spectrum"] = {
        "frequency_hz": freq.tolist(), "reference_psd": pxx.tolist(), "measurement_psd": pyy.tolist(),
        "transfer_real": transfer.real.tolist(), "transfer_imag": transfer.imag.tolist(),
        "coherence": coherence.tolist(), "reliable": reliable.tolist(),
        "regularization_absolute": regularizer,
    }
    result["status"] = "ok" if reliable.any() else "insufficient_data"
    if not reliable.any():
        result["reasons"].append("no_reliable_frequency_bins")
    step = reconstruct_step(transfer, reliable, fs, nfft, config.response_s)
    if step is None:
        result["metric_reasons"] = {key: "insufficient_contiguous_low_frequency_support" for key in METRIC_NAMES}
        result["warnings"].append("no_lowpass_step: only trusted FRF bins may be interpreted")
    else:
        result["step_response"] = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in step.items()
        }
        metrics, reasons = step_metrics(step["time_s"], step["values"], config)
        if step["negative_lag_energy_fraction"] > 0.1:
            result["warnings"].append("large_negative_lag_energy: timing is bandwidth_or_bias_limited")
            for key in ("rise_time_s", "peak_time_s", "settling_time_s"):
                metrics[key], reasons[key] = None, "large_negative_lag_energy"
        result["metrics"].update(metrics)
        result["metric_reasons"].update(reasons)
    return result


def _input_correlations(dataset, config):
    correlations = []
    for prefix in ("angle.", "rate."):
        channels = [key for key in dataset.channels if key.startswith(prefix)]
        for i, a in enumerate(channels):
            for b in channels[i + 1:]:
                at, av = signal_series(dataset, CHANNELS[a][0])
                bt, bv = signal_series(dataset, CHANNELS[b][0])
                if min(len(at), len(bt)) < 3:
                    continue
                mask = (at >= bt[0]) & (at <= bt[-1])
                intervals = config.analysis_intervals_s
                if intervals:
                    mask &= np.logical_or.reduce([(at >= lo) & (at <= hi) for lo, hi in intervals])
                x = at[mask]
                tolerance = 1e-8 * np.median(np.diff(bt))
                j = np.clip(np.searchsorted(bt, x + tolerance, side="right") - 1, 0, len(bt) - 1)
                following = np.minimum(j + 1, len(bt) - 1)
                valid = bt[following] - bt[j] <= config.max_gap_periods * np.median(np.diff(bt))
                aa, bb = av[mask][valid], bv[j[valid]]
                if prefix == "angle.":
                    aa, bb = np.rad2deg(np.unwrap(np.deg2rad(aa))), np.rad2deg(np.unwrap(np.deg2rad(bb)))
                if len(aa) >= 3 and min(np.std(aa), np.std(bb)) > 1e-10:
                    corr = float(np.corrcoef(aa, bb)[0, 1])
                    correlations.append({
                        "channels": [a, b], "correlation": corr, "samples": len(aa),
                        "high_correlation": abs(corr) >= 0.8,
                    })
    return correlations


def analyze_run(run: str | Path | RunDataset, config: AnalysisConfig | None = None) -> dict:
    """Analyze a run. Unidentifiable quantities are null with explicit reasons."""
    dataset = run if isinstance(run, RunDataset) else load_run(run)
    config = config or AnalysisConfig()
    intervals = _analysis_intervals(dataset, config)
    return {
        "schema_version": RESULT_SCHEMA_VERSION, "kind": "analysis", "method": "regularized_welch_siso_v1",
        "run_id": dataset.run_id, "input_path": str(dataset.path), "csv_path": str(dataset.csv_path),
        "checksum": dataset.checksum, "config": asdict(config),
        "analysis_intervals_s": [list(interval) for interval in intervals],
        "validation_diagnostics": dataset.diagnostics,
        "limitations": LIMITATIONS,
        "channels": {key: _analyze_channel(dataset, key, config) for key in CHANNELS},
        "input_correlations": _input_correlations(dataset, config),
    }
