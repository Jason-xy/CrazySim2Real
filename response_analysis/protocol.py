"""CSV loading and the numerical parameters needed for offline analysis."""

from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd


RESULT_SCHEMA_VERSION = 2
CHANNELS = {
    **{
        f"angle.{axis}": (f"angle_ref_{axis}_deg", f"angle_meas_{axis}_deg")
        for axis in ("roll", "pitch", "yaw")
    },
    **{
        f"rate.{axis}": (f"rate_ref_{axis}_dps", f"rate_meas_{axis}_dps")
        for axis in ("x", "y", "z")
    },
}


class ValidationError(ValueError):
    """The selected data or analysis arguments cannot be used."""


@dataclass(frozen=True)
class AnalysisConfig:
    window_s: float = 2.0
    overlap: float = 0.5
    min_windows: int = 4
    response_s: float = 1.0
    sample_rate_hz: float | None = None
    coherence_min: float = 0.6
    excitation_floor: float = 1e-4
    regularization: float = 1e-6
    max_gap_periods: float = 3.0
    min_input_variance: float = 1e-10
    settling_band: float = 0.02
    bootstrap_samples: int = 1000
    seed: int = 0
    analysis_intervals_s: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key == "analysis_intervals_s":
                continue
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValueError(f"{key} must be a finite number")
        for key in ("min_windows", "bootstrap_samples", "seed"):
            if not isinstance(getattr(self, key), int):
                raise ValueError(f"{key} must be an integer")
        if self.window_s <= 0 or not 0 < self.response_s <= self.window_s / 2:
            raise ValueError("Require 0 < response_s <= window_s / 2")
        if not 0 <= self.overlap < 1 or self.min_windows < 4:
            raise ValueError("Require 0 <= overlap < 1 and min_windows >= 4")
        if not 0 < self.coherence_min <= 1 or not 0 < self.excitation_floor < 1:
            raise ValueError("Invalid coherence or excitation threshold")
        if self.regularization <= 0 or self.max_gap_periods <= 1:
            raise ValueError("Regularization must be positive; gap threshold must exceed 1")
        if self.min_input_variance <= 0 or not 0 < self.settling_band < 1:
            raise ValueError("Invalid input variance or settling band")
        if self.sample_rate_hz is not None and self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.bootstrap_samples < 100 or self.seed < 0:
            raise ValueError("Require bootstrap_samples >= 100 and seed >= 0")
        if self.analysis_intervals_s is not None:
            intervals = self.analysis_intervals_s
            if not isinstance(intervals, (tuple, list)) or not intervals:
                raise ValueError("analysis_intervals_s must be a nonempty list of [start, end]")
            normalized, previous_end = [], -np.inf
            for interval in intervals:
                if not isinstance(interval, (tuple, list)) or len(interval) != 2 or any(
                    isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x)
                    for x in interval
                ):
                    raise ValueError("Invalid analysis interval")
                start, end = map(float, interval)
                if start < 0 or start >= end or start < previous_end:
                    raise ValueError("Analysis intervals must be nonnegative, ordered and nonoverlapping")
                normalized.append((start, end))
                previous_end = end
            object.__setattr__(self, "analysis_intervals_s", tuple(normalized))


@dataclass
class RunDataset:
    path: Path
    csv_path: Path
    samples: pd.DataFrame
    channels: tuple[str, ...]
    checksum: str
    diagnostics: dict

    @property
    def run_id(self) -> str:
        return f"{self.csv_path.stem}-{self.checksum[:12]}"


def read_json(path: Path) -> dict:
    """Read analysis settings or a comparison manifest."""
    def finite_float(text):
        value = float(text)
        if not np.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value

    def unique_object(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(text):
        raise ValueError(f"JSON numbers must be finite: {text}")

    try:
        value = json.loads(
            path.read_bytes(), parse_float=finite_float,
            parse_constant=reject_constant, object_pairs_hook=unique_object,
        )
    except (OSError, ValueError, UnicodeError) as exc:
        raise ValidationError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{path}: expected a JSON object")
    return value


def signal_series(dataset: RunDataset, column: str) -> tuple[np.ndarray, np.ndarray]:
    frame = dataset.samples[["t_s", column]].dropna(subset=[column])
    frame = frame.drop_duplicates("t_s")
    return frame["t_s"].to_numpy(float), frame[column].to_numpy(float)


def load_run(path: str | Path) -> RunDataset:
    """Read only the CSV. Other files beside it are never opened or changed."""
    path = Path(path).absolute()
    csv_path = path / "samples.csv" if path.is_dir() else path
    if csv_path.suffix.lower() != ".csv":
        raise ValidationError(f"Expected a CSV file or run directory: {path}")
    try:
        contents = csv_path.read_bytes()
        frame = pd.read_csv(io.BytesIO(contents), keep_default_na=False, na_values=[""])
    except (OSError, ValueError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValidationError(f"Cannot read CSV {csv_path}: {exc}") from exc
    errors = []
    known = {"t_s"} | {column for pair in CHANNELS.values() for column in pair}
    extra = set(frame) - known
    if extra:
        errors.append(f"Unexpected columns: {sorted(extra)}")
    if "t_s" not in frame or frame.empty:
        raise ValidationError("CSV requires t_s and at least one row")
    for column in frame:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (ValueError, TypeError):
            errors.append(f"{column}: nonnumeric sample")
    if errors:
        raise ValidationError("; ".join(errors))
    times = frame["t_s"].to_numpy(float)
    if not np.isfinite(times).all() or (times < 0).any():
        raise ValidationError("t_s must be finite, nonnegative elapsed seconds")
    if (np.diff(times) < 0).any():
        raise ValidationError("Time moved backwards; split clock resets into separate runs")
    available, diagnostics = [], {}
    for channel, pair in CHANNELS.items():
        if not any(column in frame and frame[column].notna().any() for column in pair):
            continue
        available.append(channel)
        for column in pair:
            if column not in frame or frame[column].notna().sum() == 0:
                errors.append(f"{channel}: missing paired signal {column}")
                continue
            series = frame.loc[frame[column].notna(), ["t_s", column]]
            if not np.isfinite(series[column]).all():
                errors.append(f"{column}: nonfinite values")
            if (series.groupby("t_s")[column].nunique() > 1).any():
                errors.append(f"{column}: conflicting values at the same time")
            unique = series.drop_duplicates("t_s")
            dt = np.diff(unique["t_s"].to_numpy(float))
            median_dt = float(np.median(dt)) if len(dt) else None
            diagnostics[column] = {
                "samples": len(unique), "duplicate_samples_removed": len(series) - len(unique),
                "empty_rows": int(frame[column].isna().sum()),
                "effective_rate_hz": 1 / median_dt if median_dt and median_dt > 0 else None,
                "max_gap_s": float(np.max(dt)) if len(dt) else None,
                "jitter_std_s": float(np.std(dt)) if len(dt) else None,
            }
    if errors:
        raise ValidationError("; ".join(errors))
    return RunDataset(
        path, csv_path.resolve(), frame, tuple(available), hashlib.sha256(contents).hexdigest(), diagnostics
    )


def validate_run(run: str | Path | RunDataset) -> dict:
    """Check only CSV structure, samples and timestamps."""
    try:
        data = load_run(run.path if isinstance(run, RunDataset) else run)
    except (ValidationError, OSError) as exc:
        return {"schema_version": RESULT_SCHEMA_VERSION, "valid": False, "errors": [str(exc)]}
    return {
        "schema_version": RESULT_SCHEMA_VERSION, "valid": True, "errors": [],
        "run_id": data.run_id, "input_path": str(data.path), "csv_path": str(data.csv_path),
        "checksum": data.checksum, "channels": list(data.channels), "signals": data.diagnostics,
    }
