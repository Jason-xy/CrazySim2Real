# CSV Input Format

## Minimal Recording

Only a CSV is read. For one channel, supply recorded time, reference and feedback:

```csv
t_s,angle_ref_roll_deg,angle_meas_roll_deg
0.000,0.0,
0.005,,0.1
0.020,2.0,
0.025,,0.3
```

```bash
python -m response_analysis validate flight.csv
python -m response_analysis analyze flight.csv --out report-directory
```

A directory containing `samples.csv` is also accepted.
The short checked-in fixture illustrates the format, not enough duration or
excitation for response estimation.

## Columns

| Channel | Reference column | Feedback column |
| --- | --- | --- |
| `angle.roll` | `angle_ref_roll_deg` | `angle_meas_roll_deg` |
| `angle.pitch` | `angle_ref_pitch_deg` | `angle_meas_pitch_deg` |
| `angle.yaw` | `angle_ref_yaw_deg` | `angle_meas_yaw_deg` |
| `rate.x` | `rate_ref_x_dps` | `rate_meas_x_dps` |
| `rate.y` | `rate_ref_y_dps` | `rate_meas_y_dps` |
| `rate.z` | `rate_ref_z_dps` | `rate_meas_z_dps` |

Every row has `t_s`, finite nonnegative elapsed seconds. Each nonempty value was
sampled at that row's time. References and measurements may arrive on different
rows and at different sampling rates. Leave missing values empty, not zero.

Channels are detected from the CSV. A populated reference or feedback column
requires a populated counterpart; unused axes can be omitted or entirely empty.
There is no separate availability declaration.

The remaining input checks protect numerical computation:

- Required columns and numeric, finite samples.
- Nondecreasing timestamps; split clock resets into separate CSVs.
- Equal repeated values at the same signal/time can be deduplicated; conflicting
  values are rejected.
- Unsupported columns and textual null/NaN markers are rejected.

Insufficient duration, weak excitation, gaps and unreliable frequency bins are
handled by the analysis quality checks. They produce unavailable results with
reasons, not invented values.

## Fixed Recording Conventions

Record signals using the following conventions:

- Time is in seconds, angles in degrees and angular rates in degrees/second.
- Body axes are right-handed FLU: +X forward, +Y left, +Z up. Rates x/y/z are
  body-frame p/q/r, not Euler-angle derivatives.
- Attitude describes the body-to-world rotation
  `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)` in a right-handed, Z-up world frame.
- Record each loop's actual reference and feedback, not an unrelated host
  request. Use the same clock for the paired signals.

Wrapped and unwrapped angles are accepted. The analyzer does not infer signs,
convert radians, inspect filters/PID settings, or verify acquisition wiring and
clock synchronization. Choose comparable recordings and perform any necessary
conversion before recording this format.

## Analysis Intervals

Select ranges as analysis arguments, not in files next to a recording:

```bash
python -m response_analysis analyze flight.csv \
  --interval 5 25 --interval 30 50 --out selected-report
```

Alternatively, use an analysis config with `--config settings.json`:

```json
{
  "window_s": 4.0,
  "response_s": 2.0,
  "analysis_intervals_s": [[5.0, 25.0], [30.0, 50.0]]
}
```

`--interval` overrides the config selection. With neither, the whole recording
is analyzed. Ranges are in recorded `t_s` coordinates, must be ordered,
nonoverlapping and within the file, and never rebase time. Windows cannot cross
separate ranges or long dropouts.

For group comparison, the selection applies to every input's own timebase.
Crop inputs beforehand or analyze separately if different files need different
selections.

## Comparison Groups

The manifest only selects the files and their roles:

```json
{
  "reference_group": "real",
  "baseline_group": "sim_before",
  "groups": {
    "real": ["real-001.csv", "real-002.csv", "real-003.csv"],
    "sim_before": ["before-001.csv", "before-002.csv", "before-003.csv"],
    "sim_after": ["after-001.csv", "after-002.csv", "after-003.csv"]
  }
}
```

```bash
python -m response_analysis compare comparison.json --out comparison-report
```

Paths are relative to the manifest file unless absolute; run directories are
also accepted. Each group needs at least one input. Reference and baseline
groups must exist and differ.

All selected runs are compared per channel. There is no automatic grouping by
condition, PID compatibility check, filter-stage check or source classification.
Organize separate experiments in separate manifests. File names and group
names are just labels.

Each usable run has equal weight. Duplicating a CSV does not create an
independent experiment: repeated content within a group is rejected. Shared
data across groups disables independent-group gap confidence intervals.
The common trustworthy frequency domain is fixed across variants in one report;
adding files can change that domain, so compare improvements within one report.

## Results

The result structure is version `2`; the numerical method remains
`regularized_welch_siso_v1`. It contains the metrics, valid ranges, quality
diagnostics and actual analysis settings. File identity is just `input_path`
(or comparison input `path`), resolved `csv_path`, CSV SHA-256 `checksum`, and
a filename/hash-derived `run_id`.

Comparisons have one `cases` entry per channel. CSV metric summaries list the
group, channel, metric value, unit and statistical results.

Results use separate real/imaginary arrays for frequency responses.
Unavailable numerical metrics are JSON null or empty CSV cells, with reasons;
they are not zero. Reports use only local resources and are written into an
explicit new or empty directory without altering inputs.
