import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from response_analysis import AnalysisConfig, ValidationError, load_run, validate_run
from response_analysis.protocol import read_json, signal_series
from .helpers import RunTestCase


class ProtocolTests(RunTestCase):
    def test_round_trip_hashes_and_minimal_dataset(self):
        path = self.make()
        original = (path / "samples.csv").read_bytes()
        data = load_run(path)
        checked = validate_run(data)
        self.assertTrue(checked["valid"])
        self.assertEqual(data.checksum, hashlib.sha256(original).hexdigest())
        self.assertEqual(checked["checksum"], data.checksum)
        self.assertEqual(data.run_id, f"samples-{data.checksum[:12]}")
        self.assertEqual(data.channels, ("angle.roll",))
        self.assertEqual(set(vars(data)), {"path", "csv_path", "samples", "channels", "checksum", "diagnostics"})
        self.assertEqual(original, (path / "samples.csv").read_bytes())
        self.assertEqual({p.name for p in path.iterdir()}, {"samples.csv"})

    def test_directory_and_file_inputs_are_equivalent(self):
        path = self.make()
        directory, direct = load_run(path), load_run(path / "samples.csv")
        self.assertEqual(directory.checksum, direct.checksum)
        self.assertEqual(directory.channels, direct.channels)
        self.assertEqual(directory.csv_path, direct.csv_path)
        self.assertEqual(directory.run_id, direct.run_id)

    def test_adjacent_files_are_never_opened(self):
        for name in ("samples", "flight"):
            path = self.bare(name)
            (path.parent / "metadata.json").write_text("{invalid")
            path.with_suffix(".metadata.json").write_text('{"timebase":{"aligned":false}}')
            original_read = Path.read_bytes
            opened = []

            def record_read(filename):
                opened.append(filename)
                return original_read(filename)

            with patch.object(Path, "read_bytes", record_read):
                data = load_run(path)
            self.assertEqual(opened, [path])
            self.assertEqual(data.channels, ("angle.roll",))
            self.assertNotIn("unknown", json.dumps(validate_run(path)))

    def test_directory_with_unreadable_companion_name_is_valid(self):
        path = self.make()
        (path / "metadata.json").mkdir()
        self.assertTrue(validate_run(path)["valid"])
        self.assertTrue((path / "metadata.json").is_dir())

    def test_sparse_asynchronous_rows(self):
        path = self.make()
        self.samples(path, lambda f: pd.concat([
            f[["t_s", "angle_ref_roll_deg"]], f[["t_s", "angle_meas_roll_deg"]],
        ]).sort_values("t_s", kind="stable"))
        data = load_run(path)
        self.assertEqual(len(signal_series(data, "angle_ref_roll_deg")[0]), 4000)
        self.assertEqual(data.diagnostics["angle_meas_roll_deg"]["empty_rows"], 4000)

    def test_duplicate_samples_removed_but_conflicts_rejected(self):
        path = self.make()
        self.samples(path, lambda f: pd.concat([f, f.iloc[[0]]]).sort_values("t_s", kind="stable"))
        self.assertEqual(load_run(path).diagnostics["angle_ref_roll_deg"]["duplicate_samples_removed"], 1)
        self.samples(path, lambda f: f.assign(angle_ref_roll_deg=f.angle_ref_roll_deg + (f.index == 0)))
        self.assertIn("conflicting", validate_run(path)["errors"][0])

    def test_time_reset_not_sorted_away(self):
        path = self.make()
        self.samples(path, lambda f: f.assign(t_s=np.where(f.index >= 2000, f.t_s - 20, f.t_s)))
        self.assertIn("backwards", validate_run(path)["errors"][0])

    def test_missing_measurement_not_inferred(self):
        path = self.make()
        self.samples(path, lambda f: f.drop(columns="angle_meas_roll_deg"))
        self.assertIn("missing paired", validate_run(path)["errors"][0])

    def test_invalid_samples_and_timestamps(self):
        mutations = (
            lambda f: f.assign(angle_ref_roll_deg="invalid"),
            lambda f: f.assign(angle_ref_roll_deg="NaN"),
            lambda f: f.assign(angle_ref_roll_deg=float("inf")),
            lambda f: f.assign(t_s=-1),
            lambda f: f.assign(t_s=float("inf")),
            lambda f: f.drop(columns="t_s"),
            lambda f: f.assign(misspelled_reference=1),
        )
        for i, mutation in enumerate(mutations):
            with self.subTest(i=i):
                path = self.make(str(i), duration_s=1)
                self.samples(path, mutation)
                self.assertFalse(validate_run(path)["valid"])

    def test_channel_detection_and_empty_unused_columns(self):
        path = self.make(channels=["angle.roll", "rate.z"])
        self.samples(path, lambda f: f.assign(angle_ref_pitch_deg=np.nan, angle_meas_pitch_deg=np.nan))
        self.assertEqual(load_run(path).channels, ("angle.roll", "rate.z"))

    def test_missing_empty_and_wrong_file_inputs(self):
        self.assertFalse(validate_run(self.root / "missing.csv")["valid"])
        self.assertFalse(validate_run(self.root / "missing")["valid"])
        empty = self.root / "empty.csv"
        empty.write_text("")
        self.assertFalse(validate_run(empty)["valid"])
        empty.write_text("t_s,angle_ref_roll_deg,angle_meas_roll_deg\n")
        self.assertFalse(validate_run(empty)["valid"])

    def test_csv_symlink_and_companion_are_independent(self):
        directory = self.make()
        alias = self.root / "alias.csv"
        alias.symlink_to(directory / "samples.csv")
        alias.with_suffix(".metadata.json").write_text("{malformed")
        data = load_run(alias)
        self.assertEqual(data.csv_path, directory / "samples.csv")
        self.assertTrue(validate_run(data)["valid"])

    def test_checked_in_format_example(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "format_example"
        self.assertTrue(validate_run(fixture)["valid"])
        self.assertEqual({path.name for path in fixture.iterdir()}, {"samples.csv"})

    def test_analysis_json_rejects_invalid_values_and_duplicate_keys(self):
        path = self.root / "config.json"
        for text in ('{"window_s": NaN}', '{"window_s": 1e999}', '{"seed":0,"seed":1}', "[1]", "{"):
            with self.subTest(text=text):
                path.write_text(text)
                with self.assertRaises(ValidationError):
                    read_json(path)

    def test_config_validation(self):
        for override in (
            {"response_s": 3}, {"window_s": 0}, {"min_windows": 1},
            {"min_windows": 4.5}, {"sample_rate_hz": -1}, {"coherence_min": 0},
            {"regularization": 0}, {"max_gap_periods": 1}, {"overlap": 1},
            {"seed": -1}, {"window_s": float("inf")}, {"seed": True},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                AnalysisConfig(**override)

    def test_interval_validation_and_immutable_normalization(self):
        for intervals in (
            [], [1, 2], [[1]], [[2, 1]], [[-1, 2]], [[0, float("inf")]],
            [[0, 5], [4, 8]], [[5, 8], [0, 1]], [[False, 2]], "0,2",
        ):
            with self.subTest(intervals=intervals), self.assertRaises(ValueError):
                AnalysisConfig(analysis_intervals_s=intervals)
        supplied = [[0, 2], [4, 8]]
        config = AnalysisConfig(analysis_intervals_s=supplied)
        supplied[0][1] = 99
        self.assertEqual(config.analysis_intervals_s, ((0.0, 2.0), (4.0, 8.0)))
