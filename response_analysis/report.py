"""Portable, headless reports. Writes only into a new or empty output directory."""

import csv
import html
import json
from pathlib import Path
import platform

import numpy as np
import scipy
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def metric_unit(name, channel):
    if name.endswith("_s"):
        return "s"
    if name.endswith("_pct"):
        return "%"
    if name.endswith("_db"):
        return "dB"
    if name.endswith("_deg"):
        return "deg"
    if name.startswith("tracking_"):
        return "deg" if channel.startswith("angle.") else "deg/s"
    return "1"


def summary_rows(result):
    if result.get("kind") == "analysis":
        for channel, entry in result["channels"].items():
            for name, value in entry["metrics"].items():
                yield {
                    "channel": channel,
                    "group": result["run_id"], "metric": name, "value": value,
                    "unit": metric_unit(name, channel), "n": 1 if value is not None else 0,
                    "status": entry["status"],
                    "reason": entry["metric_reasons"].get(name, ""),
                }
    elif result.get("kind") == "comparison":
        for case in result["cases"]:
            base = {"channel": case["channel"], "status": case["status"]}
            if case["status"] != "ok":
                yield {**base, "reason": "; ".join(case["reasons"])}
                continue
            for name, group in case["groups"].items():
                if group["status"] != "ok":
                    yield {**base, "group": name, "status": group["status"], "reason": group["reason"]}
                    continue
                for metric, stats in group["metric_statistics"].items():
                    ci = stats["ci95"] or (None, None)
                    reasons = sorted({
                        run["metric_reasons"].get(metric, "") for run in group["runs"]
                        if run["metrics"][metric] is None
                    } - {""})
                    yield {
                        **base, "group": name, "metric": metric, "value": stats["mean"],
                        "std": stats["std"], "n": stats["n"], "ci95_low": ci[0], "ci95_high": ci[1],
                        "unit": metric_unit(metric, case["channel"]), "reason": "; ".join(reasons),
                    }
                for metric, value in group.get("gaps", {}).items():
                    ci = group["gap_ci95"][metric] or (None, None)
                    yield {
                        **base, "group": name, "metric": f"gap.{metric}", "value": value,
                        "unit": metric_unit(metric, case["channel"]), "n": group["n"],
                        "ci95_low": ci[0], "ci95_high": ci[1],
                        "reason": group["gap_reasons"].get(metric, ""),
                    }
                for metric, value in group.get("improvement_pct", {}).items():
                    yield {
                        **base, "group": name, "metric": f"improvement.{metric}", "value": value,
                        "unit": "%", "reason": group["improvement_reasons"].get(metric, ""),
                    }
                for metric, value in group.get("metric_deltas", {}).items():
                    yield {
                        **base, "group": name, "metric": f"delta.{metric}", "value": value,
                        "unit": metric_unit(metric, case["channel"]),
                        "reason": "metric_unavailable" if value is None else "",
                    }
    else:
        yield {"status": "valid" if result["valid"] else "invalid", "reason": "; ".join(result.get("errors", []))}


def _save_figure(fig, path):
    FigureCanvasAgg(fig)
    for axis in fig.axes:
        axis.grid(alpha=0.2)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    fig.clear()


def _phase_curve(transfer, valid):
    phase = np.full(len(transfer), np.nan)
    indices = np.flatnonzero(valid)
    for block in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
        if len(block):
            phase[block] = np.rad2deg(np.unwrap(np.angle(transfer[block])))
    return phase


def _analysis_plot(channel, entry, path):
    fig = Figure(figsize=(11, 10))
    axes = fig.subplots(3, 2)
    fig.suptitle(f"{channel}: empirical closed-loop response", fontsize=14)
    for i, block in enumerate(entry.get("time_series", [])):
        axes[0, 0].plot(block["time_s"], block["reference"], label="Reference" if i == 0 else None, alpha=0.8)
        axes[0, 0].plot(block["time_s"], block["measurement"], label="Feedback" if i == 0 else None, alpha=0.8)
    axes[0, 0].set(xlabel="Recorded time (s)", ylabel=entry["unit"])
    step = entry.get("step_response")
    if step:
        axes[0, 1].plot(step["time_s"], step["values"], label="Band-limited estimate")
        axes[0, 1].axhline(1, linestyle=":", color="gray", label="Unit target")
        axes[0, 1].set_title(f"Reconstruction band: {step['band_hz'][1]:.2f} Hz")
    else:
        axes[0, 1].text(0.05, 0.5, "Step unavailable; see metric reasons", transform=axes[0, 1].transAxes)
    axes[0, 1].set(xlabel="Time after reference step (s)", ylabel="Input-normalized response (1)")
    spectrum = entry.get("spectrum")
    if spectrum:
        f = np.asarray(spectrum["frequency_hz"])
        h = np.asarray(spectrum["transfer_real"]) + 1j * np.asarray(spectrum["transfer_imag"])
        good = np.asarray(spectrum["reliable"])
        magnitude = 20 * np.log10(np.maximum(np.abs(h), 1e-15))
        axes[1, 0].plot(f, magnitude, color="gray", alpha=0.35, linestyle=":", label="All estimated bins")
        axes[1, 0].plot(f, np.where(good, magnitude, np.nan), label="Trusted bins")
        axes[1, 0].set(xlabel="Frequency (Hz)", ylabel="Magnitude (dB)")
        axes[1, 1].plot(f, _phase_curve(h, good), label="Trusted bands, separately unwrapped")
        axes[1, 1].set(xlabel="Frequency (Hz)", ylabel="Unwrapped phase (deg)")
        axes[2, 0].plot(f, spectrum["coherence"], label="Magnitude-squared coherence")
        axes[2, 0].fill_between(f, 0, 1, where=good, alpha=0.12, label="Trusted bins")
        axes[2, 0].set(xlabel="Frequency (Hz)", ylabel="Coherence (1)", ylim=(0, 1.05))
        for field, label in (("reference_psd", "Reference"), ("measurement_psd", "Feedback")):
            axes[2, 1].semilogy(f, np.maximum(spectrum[field], 1e-30), label=label)
        axes[2, 1].set(xlabel="Frequency (Hz)", ylabel=f"PSD ({entry['unit']})^2/Hz")
    _save_figure(fig, path)


def _comparison_plot(case, path):
    fig = Figure(figsize=(11, 8))
    axes = fig.subplots(2, 2)
    fig.suptitle(f"{case['channel']}: matched-band group comparison")
    f, good = np.asarray(case["common_frequency_hz"]), np.asarray(case["common_reliable"])
    for name, group in case["groups"].items():
        if group["status"] != "ok":
            continue
        h = np.asarray(group["transfer_mean_real"]) + 1j * np.asarray(group["transfer_mean_imag"])
        axes[0, 1].plot(f, np.where(good, 20 * np.log10(np.maximum(np.abs(h), 1e-15)), np.nan), label=name)
        axes[1, 0].plot(f, _phase_curve(h, good), label=name)
        step = group["step_response"]
        if step:
            line, = axes[0, 0].plot(step["time_s"], step["mean"], label=f"{name} (n={group['n']})")
            if step["ci95"]:
                axes[0, 0].fill_between(step["time_s"], *step["ci95"], color=line.get_color(), alpha=0.15)
            if step["std"]:
                axes[1, 1].plot(step["time_s"], step["std"], label=name)
    axes[0, 0].set(xlabel="Time after reference step (s)", ylabel="Mean unit-step response (1)")
    axes[0, 1].set(xlabel="Frequency (Hz)", ylabel="Magnitude (dB)")
    axes[1, 0].set(xlabel="Frequency (Hz)", ylabel="Unwrapped phase (deg)")
    axes[1, 1].set(xlabel="Time after reference step (s)", ylabel="Between-run step std (1)")
    _save_figure(fig, path)


def write_report(result: dict, output: str | Path, protected_paths=()) -> Path:
    """Export reports without overwriting source CSVs or existing output."""
    output = Path(output).resolve()
    inputs = list(protected_paths)
    if "input_path" in result:
        inputs.append(result["input_path"])
        inputs.append(result["csv_path"])
    elif result.get("kind") == "comparison":
        for group in result["inputs"].values():
            for run in group:
                inputs.append(run["path"])
                inputs.append(run["csv_path"])
    for source in inputs:
        source = Path(source).resolve()
        if output == source or source in output.parents:
            raise ValueError("Output must not overlap a source file or be inside a source run directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be new or empty; existing files are never overwritten")
    # Serialize before creating files so invalid nonfinite results cannot silently escape.
    exported = {
        **result,
        "software": {
            "response_analysis": "0.2.0", "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
        },
    }
    serialized = json.dumps(exported, indent=2, ensure_ascii=True, allow_nan=False)
    rows = list(summary_rows(result))
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(serialized + "\n", encoding="utf-8")
    fields = ("channel", "group", "metric", "value", "unit", "n", "std",
              "ci95_low", "ci95_high", "status", "reason")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    images = []
    if result.get("kind") == "analysis":
        for i, (channel, entry) in enumerate(result["channels"].items()):
            if "time_series" in entry:
                name = f"channel-{i}.png"
                _analysis_plot(channel, entry, output / name)
                images.append((channel, name))
    elif result.get("kind") == "comparison":
        for i, case in enumerate(result["cases"]):
            if case["status"] == "ok":
                name = f"comparison-{i}.png"
                _comparison_plot(case, output / name)
                images.append((case["channel"], name))
    escape = lambda value: html.escape(str(value))
    table = "<tr>" + "".join(f"<th>{escape(field)}</th>" for field in fields) + "</tr>"
    for row in rows:
        table += "<tr>" + "".join(
            f"<td>{escape('' if row.get(field) is None else row.get(field, ''))}</td>" for field in fields
        ) + "</tr>"
    diagnostics = {"warnings": result.get("warnings", []), "limitations": result.get("limitations", [])}
    if result.get("kind") == "analysis":
        diagnostics.update({
            "input_correlations": result["input_correlations"],
            "channels": {
                key: {field: entry.get(field) for field in
                      ("status", "reasons", "warnings", "metric_reasons", "preprocessing", "spectral_settings")}
                for key, entry in result["channels"].items()
            },
        })
    elif result.get("kind") == "comparison":
        diagnostics["cases"] = [
            {key: case[key] for key in ("channel", "status", "reasons", "warnings", "excluded_runs")}
            for case in result["cases"]
        ]
    else:
        diagnostics = result
    figures = "".join(
        f'<section><h2>{escape(title)}</h2><img src="{name}" alt="{escape(title)} response plots"></section>'
        for title, name in images
    )
    settings = {
        key: result[key] for key in
        ("run_id", "input_path", "csv_path", "checksum", "inputs", "config", "analysis_intervals_s")
        if key in result
    }
    entries = list(result.get("channels", {}).values()) if result.get("kind") == "analysis" else result.get("cases", [])
    availability = (
        f"{sum(entry['status'] == 'ok' for entry in entries)} / {len(entries)} "
        "channels or cases have a usable frequency estimate. Individual metrics may still be unresolved."
    ) if entries else "Protocol validation result; see diagnostics below."
    notices = list(result.get("warnings", []))
    for entry in entries:
        notices.extend(entry.get("warnings", []))
        if entry["status"] != "ok":
            notices.extend(entry.get("reasons", []))
    notices = list(dict.fromkeys(notices))
    notice_html = "<ul>" + "".join(f"<li>{escape(value)}</li>" for value in notices) + "</ul>" if notices else ""
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Closed-loop response analysis</title>
<style>
:root {{ color-scheme: light; --ink: #24332d; --paper: #f5f3e9; --accent: #916326; }}
body {{ margin: 0; color: var(--ink); background: var(--paper); font-family: Georgia, serif; }}
main {{ max-width: 1180px; margin: auto; padding: 28px 18px; }}
h1 {{ font-size: clamp(1.8rem, 5vw, 3rem); border-bottom: 3px solid var(--accent); padding-bottom: 20px; }}
a {{ color: #235d64; }} section {{ margin: 28px 0; }} img {{ width: 100%; height: auto; }}
.table {{ overflow-x: auto; }} table {{ border-collapse: collapse; font: 12px monospace; width: 100%; }}
td, th {{ border-bottom: 1px solid #c5c9be; padding: 8px; text-align: left; }}
th {{ background: #e0e5d9; }} pre {{ white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px; }}
summary {{ cursor: pointer; padding: 12px 0; font-weight: bold; }}
</style></head><body><main><h1>Closed-loop response analysis</h1>
<p>Offline empirical estimates. Missing values are not zero; consult the validity reasons.
Timing and gain have not been aligned to minimize differences.</p>
<p>{availability}</p>{notice_html}
<p><a href="results.json">Results and analysis settings (JSON)</a> /
<a href="summary.csv">Metric summary (CSV)</a></p>
<details><summary>Detailed validity and limitations</summary><pre>{escape(json.dumps(diagnostics, indent=2))}</pre></details>
<details><summary>Inputs and analysis settings</summary><pre>{escape(json.dumps(settings, indent=2))}</pre></details>
<p>Group response shading, when present, is a pointwise 95% run-bootstrap interval, not a simultaneous confidence band.</p>
{figures}
<section><h2>Metrics</h2><div class="table"><table>{table}</table></div></section></main></body></html>
"""
    (output / "report.html").write_text(page, encoding="utf-8")
    return output / "report.html"
