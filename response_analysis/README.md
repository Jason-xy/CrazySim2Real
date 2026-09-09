# Closed-Loop Response Analysis

An independent, offline analyzer for angle and body-rate control loops. It
estimates responses from general flight excitation, compares named groups of
runs, and evaluates whether a revised simulator is closer to a real reference.

**This is not a flight controller, an automatic PID tuner, or a physical-parameter
identification solver.** It does not import the benchmark, connect to a drone,
start Isaac Lab, or modify any recording code. It reads only the standard CSV
columns in [FORMAT.md](FORMAT.md); existing benchmark CSVs are not silently
converted.

## Quick Start

Run commands from the repository root with Python 3.10 or newer. The numerical
dependencies are NumPy, SciPy and pandas; reports also use Matplotlib. These are
already listed in the benchmark's existing requirements. No new dependency is
introduced.

**Only a CSV is required.** A single recorded channel needs time, reference and
feedback columns:

```csv
t_s,angle_ref_roll_deg,angle_meas_roll_deg
0.000,0.0,
0.005,,0.1
0.020,2.0,
0.025,,0.3
```

This snippet illustrates the format, not enough excitation or duration for
identification. The checked-in CSV-only example can be validated directly:

```bash
python -m response_analysis validate response_analysis/fixtures/format_example/samples.csv
```

For a sufficiently long recording named `flight.csv`:

```bash
python -m response_analysis analyze flight.csv --out /tmp/response-flight-report
```

Passing a directory containing `samples.csv` is also supported.

For a reproducible complete demo, generate six channels, three groups, and three
independent synthetic runs per group. Each run contains only `samples.csv`:

```bash
python -m response_analysis.examples --out /tmp/response-demo
python -m response_analysis validate /tmp/response-demo/real-0
python -m response_analysis analyze /tmp/response-demo/real-0 \
  --out /tmp/response-single-report
python -m response_analysis compare /tmp/response-demo/comparison.json \
  --out /tmp/response-comparison-report
```

Use new or empty output directories. The example's `real` label is a comparison
role: **all example signals are synthetic, not hardware recordings**. The
original simulator has different gain, lag and delay; the revised simulator is
closer but not identical. Different runs use different commands.

Each report directory contains:

| File | Contents |
| --- | --- |
| `results.json` | Numerical results, validity reasons, analysis settings and input CSV hashes |
| `summary.csv` | Metrics, units, group counts, variability, gaps and improvements |
| `*.png` | Time traces, response estimates, frequency responses and quality diagnostics |
| `report.html` | Offline report with relative local resources and no network requests |

Inputs are read-only. Directory-mode output must be outside the input run
directory. With a direct CSV input, an adjacent report directory is allowed.
CSV files cannot be overwritten, and report directories must be new or empty.

The small checked-in `fixtures/format_example` illustrates sparse asynchronous
rows. It passes `validate`, but is intentionally too short for response
identification. Use the generated demo for a complete analysis.

## Recording Contract

See [FORMAT.md](FORMAT.md) for column names, units, coordinate conventions and
the comparison manifest.

Record each loop's **actual internal reference and feedback**, not merely the
host's requested command. Angle references must not stand in for body-rate
references; differentiated Euler angles must not stand in for gyro measurements.
Choose comparable feedback signals when comparing recordings.

References and feedback can have different rates and timestamps, but all
timestamps within a run must share one aligned vehicle or simulator clock.
These requirements are fixed conventions, not recording attributes inspected
by the analyzer. It does not verify acquisition wiring, clock alignment, PID
settings, filter configuration or operating conditions. No delay is optimized
away. Available channels are detected directly from populated CSV columns.
Experiment design and selecting comparable recordings remain the caller's
responsibility.

## Select a Time Range

By default the whole CSV is analyzed. Select one or more recorded-time
intervals explicitly:

```bash
python -m response_analysis analyze flight.csv \
  --interval 5 25 --interval 30 50 --out /tmp/response-selected-report
```

The same option works for `compare` and applies to each input's own `t_s`
coordinates. Intervals must be ordered, nonoverlapping and inside every
recording being analyzed. They do not rebase timestamps or align responses.
For different per-file ranges, prepare appropriately cropped CSVs or analyze
files separately with different settings.

## Python API

```python
from response_analysis import (
    AnalysisConfig,
    load_run,
    validate_run,
    analyze_run,
    compare_groups,
)
from response_analysis.report import write_report

validation = validate_run("/tmp/response-demo/real-0/samples.csv")
dataset = load_run("/tmp/response-demo/real-0/samples.csv")
config = AnalysisConfig(
    window_s=4.0,
    response_s=2.0,
    analysis_intervals_s=((5.0, 25.0),),
)
analysis = analyze_run(dataset, config)
comparison = compare_groups("/tmp/response-demo/comparison.json", config)
write_report(comparison, "/tmp/response-long-window-report")
```

`load_run` raises `ValidationError` for invalid input. `validate_run` returns
`valid`, `errors`, available channels and per-signal diagnostics. Analysis and comparison
return JSON-compatible dictionaries; unavailable numerical quantities are
`None`, with reasons. `RunDataset` contains the input path, resolved CSV path,
raw sparse DataFrame, available-channel names, CSV checksum and sample
diagnostics. Its `run_id` is a display label derived from the CSV filename and
content hash. Treat a loaded dataset as immutable.

Results use `schema_version: 2`. File identity uses `input_path` (or comparison
input `path`), `csv_path`, `run_id` and a single `checksum`. CSV summaries list
the group, channel, metric value, unit and statistical results.

`compare_groups` also accepts a manifest dictionary. Dictionary paths resolve
relative to the current directory; file-manifest paths resolve relative to the
manifest file. Numerical APIs do not write files.

## Method and Interpretation

1. Check samples and select the requested intervals. Split long signal gaps and intersect
   continuous input/output pieces; do not bridge dropouts.
2. Reconstruct a common uniform grid no faster than the lower effective signal
   rate. Use held references, interpolated feedback and polyphase FIR
   anti-aliasing when downsampling. Discard filter-edge transients without moving
   timestamps.
3. Remove one mean working-point offset per continuous block for identification.
   Use periodic Hann windows and averaged one-sided spectra. Tracking error is
   calculated separately before mean removal.
4. Estimate `H = Sry / (Srr + lambda)`, with
   `Sry = mean(conj(R) * Y)` and
   `lambda = regularization * max(Srr)`. This is a regularized H1-style
   closed-loop estimate, not a fitted physical plant.
5. Qualify bins by magnitude-squared coherence and reference power. Reconstruct
   a step only when at least three contiguous trusted bins extend from DC.
   Isolated reliable frequency bands can still support frequency comparisons.

Step reconstruction uses an inverse real FFT and impulse integration. Where
the trusted lowpass band ends below Nyquist, its final 20% receives a cosine
taper. The centered impulse's negative-lag integral is retained before selecting
nonnegative times: discarding it would change the estimated DC gain. This is
zero-phase band limiting, **not** response alignment. Finite-band reconstruction
can produce pre-response and ringing; the negative-lag energy fraction is
reported, and values above 0.1 disable timing metrics.

There is no fixed 25 Hz cutoff, expected-final-value acceptance filter, or
output-gain normalization. Low coherence, poor excitation or unresolved
low-frequency behavior yields unavailable metrics rather than a fabricated
response. The response horizon and frequency resolution limit what can be
inferred, especially for slow dynamics. Increase both window and horizon when
needed; more output samples alone do not add information.

### Metrics

| Metric | Definition |
| --- | --- |
| `tracking_bias`, `tracking_rmse` | Mean and RMS of feedback minus reference on aligned, uncentered selected samples; angular errors use the shortest signed angular difference |
| `dc_gain` | Mean of the reconstructed response's final 20%, conditional on tail stability |
| `steady_state_error` | `1 - dc_gain` for the reconstructed unit-reference response |
| `rise_time_s` | First resolved 10% to 90% crossings of the estimated final response, with linear crossing interpolation |
| `peak_time_s` | Time of a resolved interior overshoot peak; absent for a monotonic response |
| `overshoot_pct` | `100 * max(0, peak(response / final) - 1)` |
| `settling_time_s` | First sample after the last exit from the final-value tolerance band, with the entire final 20% remaining inside |

Tail stability requires both tail standard deviation and fitted tail drift to
be no greater than `settling_band * abs(final_value)`. An unresolved tail returns
null for all step metrics. This finite-window check is **not proof of global
stability**. A 10% crossing already passed at the first sample is unresolved,
not an invented zero rise time. Very small estimated final gain also disables
step metrics.

### Group Gaps

The comparison manifest alone selects runs and groups. There is one case per
standard channel, with no condition matching or configuration compatibility
checks. Use separate manifests for separate experiments or operating conditions.

All usable runs in a case share a sampling rate, frequency grid, trustworthy-bin intersection,
reconstruction band, response horizon and thresholds. Cases without usable
reference/baseline runs or at least three common trusted bins are unavailable.
An unavailable optional group is explicitly reported, not filled with other data.

Each run contributes equally, regardless of duration. Within-run overlapping
windows are never treated as independent flights. Reported group metric deltas
are differences between the means of per-run metrics; gap curves compare group
mean transfers/responses.

| Gap | Definition on the shared domain |
| --- | --- |
| `step_rmse` | RMS difference between group mean, input-normalized step curves |
| `frf_complex_nrmse` | `sqrt(sum(abs(H_sim-H_ref)^2) / sum(abs(H_ref)^2))` |
| `magnitude_rmse_db` | RMS of `20*log10(abs(H_sim)/abs(H_ref))` |
| `phase_rmse_deg` | RMS circular phase difference `angle(H_sim*conj(H_ref))`, in degrees |

Phase plots unwrap separately within each trusted band for readability; the
phase gap remains circular. No raw-output cross-flight RMSE, time shifts, time
warps, optimized gain factors or weighted overall score are computed.

Improvement is `100 * (baseline_gap - candidate_gap) / baseline_gap`. The
reference, participating-run set and frequency/response domain are fixed across
all variants in that case. A zero or unavailable baseline yields null, not an
infinite percentage. Negative improvement means a larger gap.

Group standard deviation uses `ddof=1` and requires two valid runs. A 95%
percentile bootstrap interval requires three independent valid runs. Gap
intervals independently resample both reference and candidate groups, requiring
three runs in each. Duplicate run IDs or sample-file hashes within a group are
rejected. Shared recordings across groups produce a warning and suppress their
independent-group gap intervals. Improvement percentages do not have a
bootstrap interval. Curve intervals are pointwise, not simultaneous
confidence bands.

High input correlation, controller/filter changes, sensor filtering,
nonstationarity, noise in a feedback-derived reference and unmeasured
disturbances can all affect the result. Coherence is not an unbiasedness test.
Keep those factors controlled when interpreting a gap reduction as an
improvement in physical-system identification.

## Configuration and Exit Status

Pass a JSON object of `AnalysisConfig` overrides with `--config`:

```json
{
  "window_s": 4.0,
  "response_s": 2.0,
  "sample_rate_hz": 100.0,
  "coherence_min": 0.8,
  "analysis_intervals_s": [[5.0, 25.0]]
}
```

| Setting | Default |
| --- | --- |
| `window_s`, `response_s` | 2.0 s, 1.0 s; response must not exceed half the window |
| `overlap`, `min_windows` | 0.5, 4 |
| `sample_rate_hz` | null: lower median-interval-derived signal rate; an override is a ceiling, not permission to upsample |
| `coherence_min`, `excitation_floor` | 0.6, 0.0001 of maximum input PSD |
| `regularization` | 0.000001 of maximum input PSD |
| `max_gap_periods` | 3.0 median signal periods |
| `min_input_variance` | 0.0000000001 in squared channel units |
| `settling_band` | 0.02 of absolute final response |
| `bootstrap_samples`, `seed` | 1000, 0 |
| `analysis_intervals_s` | null: whole recording; otherwise an ordered list of `[start, end]` seconds |

Explicit `--interval` flags replace `analysis_intervals_s` from `--config`.
Without those flags the config selection is used; without either, the whole
recording is used. The actual analyzed intervals are included in single-run
results.

CLI exit status: `0` = validation passed / at least one usable channel or case;
`2` = invalid input, settings or output location; `3` = report written but no
usable frequency estimate/comparison. A successful exit does not imply every
metric or axis is identifiable. `validate` writes to stdout unless `--out` is
specified; `analyze` and `compare` require `--out`.

## Offline Verification

```bash
python -m unittest discover -s response_analysis/tests -t . -v
git diff --check
```

Tests use synthetic discrete first/second-order systems and temporary files.
They cover gain, delay, damping, all six channels, alias suppression, sparse
sampling, gaps, wraparound, invalid protocols, group statistics and report
generation, CSV-only loading, interval selection and ignoring companion files.
Numerical regressions also verify that ignored files cannot change estimates.
They do not run the repository's flight experiment plans.

The architecture is deliberately small: `protocol.py` owns the contract;
`analysis.py` owns preprocessing and estimation; `comparison.py` owns matched
group statistics; `report.py` and `__main__.py` are output/CLI layers.
