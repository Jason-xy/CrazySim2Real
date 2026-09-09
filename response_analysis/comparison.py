"""Equal-run group comparison on one explicitly shared frequency grid."""

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .analysis import METRIC_NAMES, analyze_run, lowpass_mask, reconstruct_step, step_metrics
from .protocol import AnalysisConfig, CHANNELS, RESULT_SCHEMA_VERSION, ValidationError, load_run, read_json


GAP_NAMES = ("step_rmse", "frf_complex_nrmse", "magnitude_rmse_db", "phase_rmse_deg")


def summarize(values, config):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    result = {
        "n": len(values), "mean": None, "std": None, "ci95": None,
        "std_reason": "fewer_than_two_valid_runs", "ci95_reason": "fewer_than_three_independent_valid_runs",
    }
    if not len(values):
        return result
    result["mean"] = float(values.mean())
    if len(values) >= 2:
        result["std"] = float(values.std(ddof=1))
        result["std_reason"] = None
    if len(values) >= 3:
        rng = np.random.default_rng(config.seed)
        means = values[rng.integers(0, len(values), (config.bootstrap_samples, len(values)))].mean(axis=1)
        result["ci95"] = np.quantile(means, [0.025, 0.975]).tolist()
        result["ci95_reason"] = None
    return result


def gap_values(reference_h, candidate_h, mask, reference_step=None, candidate_step=None):
    reference_h, candidate_h = np.asarray(reference_h)[mask], np.asarray(candidate_h)[mask]
    result = {name: None for name in GAP_NAMES}
    if not len(reference_h):
        return result
    denominator = float(np.sum(np.abs(reference_h) ** 2))
    if denominator > 1e-20:
        result["frf_complex_nrmse"] = float(np.sqrt(
            np.sum(np.abs(candidate_h - reference_h) ** 2) / denominator
        ))
    if np.all(np.abs(reference_h) > 1e-12) and np.all(np.abs(candidate_h) > 1e-12):
        result["magnitude_rmse_db"] = float(np.sqrt(np.mean(
            (20 * np.log10(np.abs(candidate_h) / np.abs(reference_h))) ** 2
        )))
        # Circular phase difference avoids artificial 360-degree jumps.
        result["phase_rmse_deg"] = float(np.sqrt(np.mean(
            np.rad2deg(np.angle(candidate_h * np.conj(reference_h))) ** 2
        )))
    if reference_step is not None and candidate_step is not None:
        result["step_rmse"] = float(np.sqrt(np.mean(
            (np.asarray(candidate_step) - np.asarray(reference_step)) ** 2
        )))
    return result


def _gap_intervals(ref_h, cand_h, mask, ref_steps, cand_steps, config):
    result = {name: None for name in GAP_NAMES}
    if min(len(ref_h), len(cand_h)) < 3:
        return result
    rng = np.random.default_rng(config.seed)
    samples = {name: [] for name in GAP_NAMES}
    for _ in range(config.bootstrap_samples):
        ri = rng.integers(0, len(ref_h), len(ref_h))
        ci = rng.integers(0, len(cand_h), len(cand_h))
        values = gap_values(
            ref_h[ri].mean(axis=0), cand_h[ci].mean(axis=0), mask,
            ref_steps[ri].mean(axis=0) if ref_steps is not None else None,
            cand_steps[ci].mean(axis=0) if cand_steps is not None else None,
        )
        for name, value in values.items():
            if value is not None:
                samples[name].append(value)
    for name, values in samples.items():
        if len(values) == config.bootstrap_samples:
            result[name] = np.quantile(values, [0.025, 0.975]).tolist()
    return result


def _curve_summary(curves, config):
    summary = {"mean": curves.mean(axis=0).tolist(), "std": None, "ci95": None}
    if len(curves) >= 2:
        summary["std"] = curves.std(axis=0, ddof=1).tolist()
    if len(curves) >= 3:
        rng = np.random.default_rng(config.seed)
        # Loop to avoid allocating bootstrap x runs x frequency large arrays.
        means = np.asarray([
            curves[rng.integers(0, len(curves), len(curves))].mean(axis=0)
            for _ in range(config.bootstrap_samples)
        ])
        summary["ci95"] = np.quantile(means, [0.025, 0.975], axis=0).tolist()
    return summary


def _case(channel, groups, reference, baseline, config, cache):
    case = {
        "channel": channel, "status": "insufficient_data",
        "reasons": [], "warnings": [], "groups": {}, "excluded_runs": [],
    }
    available = [
        d for datasets in groups.values() for d in datasets if channel in d.channels
    ]
    if not available:
        case["reasons"] = ["channel_not_recorded"]
        return case
    rates = [
        d.diagnostics[column]["effective_rate_hz"] for d in available for column in CHANNELS[channel]
        if d.diagnostics[column]["effective_rate_hz"] is not None
    ]
    if not rates:
        case["reasons"] = ["too_few_samples"]
        return case
    common_fs = min(rates + ([config.sample_rate_hz] if config.sample_rate_hz else []))
    common_config = replace(config, sample_rate_hz=common_fs)
    case["common_config"] = asdict(common_config)
    entries = {}
    for name, datasets in groups.items():
        entries[name] = []
        for dataset in datasets:
            key = (str(dataset.path), common_fs)
            if key not in cache:
                cache[key] = analyze_run(dataset, common_config)
            result = cache[key]["channels"][channel]
            if result["status"] == "ok" and "spectrum" in result:
                entries[name].append((dataset, result))
            else:
                case["excluded_runs"].append({
                    "group": name, "run_id": dataset.run_id,
                    "reasons": result["reasons"],
                })
        if not entries[name]:
            case["groups"][name] = {"status": "unavailable", "n": 0, "reason": "no_usable_matched_runs"}
    if not entries[reference] or not entries[baseline]:
        case["reasons"] = ["reference_or_baseline_has_no_usable_matched_runs"]
        return case
    usable = {name: runs for name, runs in entries.items() if runs}
    spectra = [r["spectrum"] for runs in usable.values() for _, r in runs]
    frequency = np.asarray(spectra[0]["frequency_hz"])
    if any(
        len(s["frequency_hz"]) != len(frequency)
        or not np.allclose(s["frequency_hz"], frequency, rtol=1e-7, atol=1e-9)
        for s in spectra
    ):
        case["reasons"] = ["could_not_establish_common_frequency_grid"]
        return case
    mask = np.logical_and.reduce([np.asarray(s["reliable"]) for s in spectra])
    case["common_frequency_hz"], case["common_reliable"] = frequency.tolist(), mask.tolist()
    if np.sum(mask) < 3:
        case["reasons"] = ["no_common_trusted_band"]
        return case
    step_mask = lowpass_mask(mask)
    case["step_band_hz"] = [0.0, float(frequency[np.flatnonzero(step_mask)[-1]])] if step_mask.any() else None
    if not step_mask.any():
        case["warnings"].append("step_gap_unavailable: no_shared_contiguous_low_frequency_support")
    matrices, step_matrices = {}, {}
    for name, runs in usable.items():
        transfers = np.asarray([
            np.asarray(r["spectrum"]["transfer_real"]) + 1j * np.asarray(r["spectrum"]["transfer_imag"])
            for _, r in runs
        ])
        matrices[name] = transfers
        step_curves, metrics_by_run = [], []
        time_s = None
        for transfer, (dataset, result) in zip(transfers, runs):
            step = reconstruct_step(
                transfer, step_mask, common_fs, result["spectral_settings"]["nfft"], config.response_s
            )
            metric_reasons = {key: "no_common_lowpass_band" for key in METRIC_NAMES}
            metrics = {key: None for key in METRIC_NAMES}
            if step is not None:
                time_s = step["time_s"]
                step_curves.append(step["values"])
                metrics, metric_reasons = step_metrics(time_s, step["values"], common_config)
                if step["negative_lag_energy_fraction"] > 0.1:
                    for key in ("rise_time_s", "peak_time_s", "settling_time_s"):
                        metrics[key], metric_reasons[key] = None, "large_negative_lag_energy"
            metrics.update({key: result["metrics"].get(key) for key in ("tracking_bias", "tracking_rmse")})
            metrics_by_run.append({
                "run_id": dataset.run_id, "metrics": metrics, "metric_reasons": metric_reasons,
                "preprocessing": result["preprocessing"], "spectral_settings": result["spectral_settings"],
                "coherence": result["spectrum"]["coherence"], "reliable": result["spectrum"]["reliable"],
                "warnings": result["warnings"],
                "negative_lag_energy_fraction": step["negative_lag_energy_fraction"] if step is not None else None,
            })
        step_matrices[name] = np.asarray(step_curves) if step_curves else None
        mean = transfers.mean(axis=0)
        group = {
            "status": "ok", "n": len(runs), "runs": metrics_by_run,
            "metric_statistics": {
                key: summarize([run["metrics"][key] for run in metrics_by_run], common_config)
                for key in (*METRIC_NAMES, "tracking_bias", "tracking_rmse")
            },
            "transfer_mean_real": mean.real.tolist(), "transfer_mean_imag": mean.imag.tolist(),
            "transfer_complex_std": np.sqrt(np.sum(np.abs(transfers - mean) ** 2, axis=0) / (
                len(transfers) - 1
            )).tolist() if len(transfers) >= 2 else None,
            "step_response": None,
        }
        if step_curves:
            group["step_response"] = {"time_s": time_s.tolist(), **_curve_summary(step_matrices[name], config)}
        case["groups"][name] = group
    reference_h = matrices[reference].mean(axis=0)
    reference_step = step_matrices[reference].mean(axis=0) if step_matrices[reference] is not None else None
    for name in usable:
        if name == reference:
            continue
        group = case["groups"][name]
        candidate_step = step_matrices[name].mean(axis=0) if step_matrices[name] is not None else None
        group["gaps"] = gap_values(reference_h, matrices[name].mean(axis=0), mask, reference_step, candidate_step)
        group["gap_reasons"] = {
            metric: "no_common_lowpass_band" if metric == "step_rmse" else "zero_or_undefined_frequency_reference"
            for metric, value in group["gaps"].items() if value is None
        }
        group["gap_ci95"] = _gap_intervals(
            matrices[reference], matrices[name], mask, step_matrices[reference], step_matrices[name], config
        )
        group["gap_ci95_reasons"] = {
            key: "gap_unavailable" if group["gaps"][key] is None else "requires_three_independent_runs_in_both_groups"
            for key, value in group["gap_ci95"].items() if value is None
        }
        group["metric_deltas"] = {}
        for key in METRIC_NAMES:
            r_value = case["groups"][reference]["metric_statistics"][key]["mean"]
            c_value = group["metric_statistics"][key]["mean"]
            group["metric_deltas"][key] = c_value - r_value if r_value is not None and c_value is not None else None
    baseline_gaps = case["groups"][baseline]["gaps"]
    for name in usable:
        if name == reference:
            continue
        group = case["groups"][name]
        group["improvement_pct"], group["improvement_reasons"] = {}, {}
        for key, before in baseline_gaps.items():
            after = group["gaps"][key]
            if before is None or after is None:
                group["improvement_pct"][key] = None
                group["improvement_reasons"][key] = "gap_unavailable"
            elif abs(before) <= 1e-12:
                group["improvement_pct"][key] = None
                group["improvement_reasons"][key] = "zero_baseline_gap"
            else:
                group["improvement_pct"][key] = 100 * (before - after) / before
    case["status"] = "ok"
    return case


def compare_groups(manifest: str | Path | dict, config: AnalysisConfig | None = None) -> dict:
    """Compare named groups; relative run paths resolve against the manifest."""
    config = config or AnalysisConfig()
    if isinstance(manifest, dict):
        spec, base = manifest, Path.cwd()
    else:
        manifest_path = Path(manifest).resolve()
        spec, base = read_json(manifest_path), manifest_path.parent
    mapping = spec.get("groups")
    if not isinstance(mapping, dict) or len(mapping) < 2:
        raise ValidationError("At least two named groups are required")
    reference, baseline = spec.get("reference_group"), spec.get("baseline_group")
    if not isinstance(reference, str) or not isinstance(baseline, str) or (
        reference not in mapping or baseline not in mapping or reference == baseline
    ):
        raise ValidationError("Distinct reference_group and baseline_group must exist")
    groups = {}
    for name, paths in mapping.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(paths, list) or not paths:
            raise ValidationError("Each named group requires a nonempty list of CSV files or run directories")
        if any(not isinstance(path, str) or not path for path in paths):
            raise ValidationError("Run paths must be nonempty strings")
        groups[name] = [load_run(base / path) for path in paths]
        for identity in (
            [d.run_id for d in groups[name]],
            [d.checksum for d in groups[name]],
        ):
            if len(set(identity)) != len(identity):
                raise ValidationError(f"{name}: duplicate runs cannot count as independent repeats")
    warnings = [
        "Tracking errors describe each input trajectory and are not cross-flight dynamics gaps.",
        "All usable runs are equally weighted; overlapping windows are not independent repetitions.",
        "Gap confidence intervals resample both groups and require at least three runs in each.",
        "Group intervals describe recorded runs, not all sources of systematic estimation bias.",
    ]
    all_datasets = [d for datasets in groups.values() for d in datasets]
    if len({d.checksum for d in all_datasets}) != len(all_datasets):
        warnings.append("shared_data_between_groups: groups are not independent")
    cache = {}
    cases = [
        _case(channel, groups, reference, baseline, config, cache) for channel in CHANNELS
    ]
    # Suppress independent-group intervals when a comparison reuses recordings.
    for case in cases:
        for name, datasets in groups.items():
            if name == reference or name not in case["groups"]:
                continue
            a = {d.checksum for d in groups[reference]}
            b = {d.checksum for d in datasets}
            if a & b and "gap_ci95" in case["groups"][name]:
                case["groups"][name]["gap_ci95"] = {key: None for key in GAP_NAMES}
                case["groups"][name]["gap_ci95_reasons"] = {
                    key: "shared_recordings_between_groups" for key in GAP_NAMES
                }
    return {
        "schema_version": RESULT_SCHEMA_VERSION, "kind": "comparison", "method": "regularized_welch_siso_v1",
        "config": asdict(config), "reference_group": reference, "baseline_group": baseline,
        "warnings": warnings, "cases": cases,
        "inputs": {
            name: [
                {
                    "path": str(d.path), "csv_path": str(d.csv_path), "run_id": d.run_id,
                    "checksum": d.checksum, "validation_diagnostics": d.diagnostics,
                }
                for d in datasets
            ] for name, datasets in groups.items()
        },
    }
