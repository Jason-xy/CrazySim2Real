import json

import numpy as np

from response_analysis import AnalysisConfig, ValidationError, compare_groups
from response_analysis.comparison import gap_values, summarize
from .helpers import RunTestCase


class ComparisonTests(RunTestCase):
    config = AnalysisConfig(bootstrap_samples=100)

    def test_improved_model_and_independent_run_intervals(self):
        groups = {"real": [], "before": [], "after": []}
        for index in range(3):
            groups["real"].append(self.make(f"real-{index}", seed=index))
            groups["before"].append(self.make(f"before-{index}", seed=10 + index, gain=0.7, tau_s=0.09, delay_s=0.04))
            groups["after"].append(self.make(f"after-{index}", seed=20 + index, gain=0.98, tau_s=0.043))
        result = compare_groups(self.manifest(**groups), self.config)
        case = result["cases"][0]
        self.assertEqual(case["status"], "ok")
        after = case["groups"]["after"]
        for metric, improvement in after["improvement_pct"].items():
            self.assertGreater(improvement, 60, metric)
            self.assertIsNotNone(after["gap_ci95"][metric])
        self.assertEqual(after["metric_statistics"]["dc_gain"]["n"], 3)
        self.assertIsNotNone(after["step_response"]["ci95"])
        json.dumps(result, allow_nan=False)

    def test_only_manifest_groups_select_runs(self):
        reference = self.make("reference", channels=["angle.roll", "rate.x"])
        before = self.make("before", channels=["angle.roll", "rate.x"], seed=2, gain=0.7)
        spec = self.manifest(real=[reference], before=[before])
        expected = compare_groups(spec, self.config)
        for path, condition, stage in ((reference, "hover", "gyro_raw"), (before, "fast", "gyro_filtered")):
            (path / "metadata.json").write_text(json.dumps({
                "condition_id": condition, "channels": {"rate.x": {"feedback_stage": stage}},
                "analysis_intervals_s": [[0, 0.1]], "timebase": {"aligned": False},
            }))
        actual = compare_groups(spec, self.config)
        self.assertEqual(actual, expected)
        for case in actual["cases"]:
            self.assertNotIn("condition_id", case)
            self.assertNotIn("condition_status", case)
        for group in actual["inputs"].values():
            for run in group:
                self.assertNotIn("metadata", run)
                self.assertNotIn("metadata_sources", run)
        self.assertNotIn("unknown", json.dumps(actual))

    def test_identical_groups_zero_gap_and_undefined_improvement(self):
        reference = self.make("reference")
        result = compare_groups(self.manifest(real=[reference], before=[reference], after=[reference]), self.config)
        case = result["cases"][0]
        for name in ("before", "after"):
            group = case["groups"][name]
            for metric in group["gaps"]:
                self.assertAlmostEqual(group["gaps"][metric], 0)
                self.assertIsNone(group["improvement_pct"][metric])
                self.assertEqual(group["improvement_reasons"][metric], "zero_baseline_gap")
                self.assertIsNone(group["gap_ci95"][metric])
            self.assertIsNone(group["metric_statistics"]["dc_gain"]["ci95"])

    def test_durations_do_not_weight_group_means(self):
        reference = self.make("reference")
        short = self.make("short", duration_s=10, gain=0.6, seed=10)
        long = self.make("long", duration_s=80, gain=1.4, seed=11)
        case = compare_groups(self.manifest(real=[reference], before=[short, long]), self.config)["cases"][0]
        group = case["groups"]["before"]
        values = [run["metrics"]["dc_gain"] for run in group["runs"]]
        self.assertAlmostEqual(group["metric_statistics"]["dc_gain"]["mean"], np.mean(values))
        self.assertAlmostEqual(group["metric_statistics"]["dc_gain"]["mean"], 1, delta=0.07)
        self.assertIsNone(group["metric_statistics"]["dc_gain"]["ci95"])
        self.assertIsNone(group["step_response"]["ci95"])
        self.assertEqual(group["n"], 2)

    def test_three_candidate_runs_do_not_replace_reference_repetitions(self):
        reference = self.make("reference")
        before = [self.make(f"before-{i}", seed=i + 1) for i in range(3)]
        group = compare_groups(self.manifest(real=[reference], before=before), self.config)["cases"][0]["groups"]["before"]
        self.assertIsNotNone(group["metric_statistics"]["dc_gain"]["ci95"])
        self.assertTrue(all(value is None for value in group["gap_ci95"].values()))

    def test_duplicate_repeats_rejected(self):
        path = self.make()
        with self.assertRaisesRegex(ValidationError, "duplicate"):
            compare_groups(self.manifest(real=[path], before=[path, path]), self.config)

    def test_common_rate_not_highest_recording_rate(self):
        reference = self.make("reference", fs=100)
        before = self.make("before", seed=2, fs=50)
        case = compare_groups(self.manifest(real=[reference], before=[before]), self.config)["cases"][0]
        self.assertEqual(case["status"], "ok")
        self.assertAlmostEqual(case["common_config"]["sample_rate_hz"], 50)
        self.assertAlmostEqual(case["common_frequency_hz"][-1], 25)

    def test_nonoverlapping_excitation_has_no_common_gap(self):
        groups = {}
        for name, frequency in (("real", 2), ("before", 15)):
            path = self.make(name, duration_s=20)
            self.samples(path, lambda frame: frame.assign(
                angle_ref_roll_deg=np.sin(2 * np.pi * frequency * frame.t_s),
                angle_meas_roll_deg=0.8 * np.sin(2 * np.pi * frequency * frame.t_s - 0.3),
            ))
            groups[name] = [path]
        case = compare_groups(self.manifest(**groups), self.config)["cases"][0]
        self.assertEqual(case["status"], "insufficient_data")
        self.assertIn("no_common_trusted_band", case["reasons"])

    def test_manifest_paths_resolve_relative_to_file(self):
        reference = self.make("reference")
        self.bare("before", seed=2)
        filename = self.root / "comparison.json"
        filename.write_text(json.dumps({
            "reference_group": "real", "baseline_group": "before",
            "groups": {"real": ["reference"], "before": ["before.csv"]},
        }))
        result = compare_groups(filename, self.config)
        self.assertEqual(result["cases"][0]["status"], "ok")
        self.assertEqual(result["inputs"]["real"][0]["csv_path"], str(reference / "samples.csv"))

    def test_comparison_uses_analysis_intervals_for_every_run(self):
        reference = self.make("reference")
        before = self.make("before", seed=2)
        config = AnalysisConfig(bootstrap_samples=100, analysis_intervals_s=[[1, 10], [20, 30]])
        result = compare_groups(self.manifest(real=[reference], before=[before]), config)
        case = result["cases"][0]
        self.assertEqual(case["status"], "ok")
        self.assertEqual(case["common_config"]["analysis_intervals_s"], ((1.0, 10.0), (20.0, 30.0)))
        for group in case["groups"].values():
            self.assertEqual(group["runs"][0]["preprocessing"]["continuous_blocks"], 2)

    def test_empty_baseline_channel_is_explained(self):
        reference = self.make("reference", channels=["rate.x"])
        before = self.make("before", channels=["angle.roll"])
        result = compare_groups(self.manifest(real=[reference], before=[before]), self.config)
        case = next(case for case in result["cases"] if case["channel"] == "rate.x")
        self.assertEqual(case["status"], "insufficient_data")
        self.assertTrue(case["excluded_runs"])

    def test_invalid_comparison_arguments(self):
        path = self.make()
        for spec in (
            {}, {"groups": {}},
            {"reference_group": "same", "baseline_group": "same", "groups": {"same": [str(path)], "other": [str(path)]}},
            {"reference_group": "real", "baseline_group": "before", "groups": {"real": [str(path)], "before": []}},
        ):
            with self.subTest(spec=spec), self.assertRaises(ValidationError):
                compare_groups(spec, self.config)

    def test_metric_conventions_and_zero_reference(self):
        reference = np.array([1 + 0j, 1j, -1 + 0j])
        candidate = 0.5 * reference
        gaps = gap_values(reference, candidate, np.ones(3, bool), np.ones(5), np.ones(5) * 0.5)
        self.assertAlmostEqual(gaps["frf_complex_nrmse"], 0.5)
        self.assertAlmostEqual(gaps["magnitude_rmse_db"], 6.020599913279624)
        self.assertAlmostEqual(gaps["phase_rmse_deg"], 0)
        self.assertAlmostEqual(gaps["step_rmse"], 0.5)
        gaps = gap_values(np.zeros(3), candidate, np.ones(3, bool))
        self.assertIsNone(gaps["frf_complex_nrmse"])
        self.assertIsNone(gaps["magnitude_rmse_db"])
        angles = np.deg2rad([179, -179, 180])
        gaps = gap_values(np.exp(1j * angles), np.exp(-1j * angles), np.ones(3, bool))
        self.assertLess(gaps["phase_rmse_deg"], 2)

    def test_summary_reproducibility_and_insufficient_count(self):
        self.assertIsNone(summarize([1], self.config)["std"])
        self.assertIsNone(summarize([1, 2], self.config)["ci95"])
        first = summarize([1, 2, 3, None], self.config)
        self.assertEqual(first, summarize([1, 2, 3], self.config))
        self.assertEqual(first["n"], 3)
        self.assertEqual(first["mean"], 2)
