import json

import numpy as np
from response_analysis import AnalysisConfig, ValidationError, analyze_run, load_run
from response_analysis.analysis import prepare_channel, reconstruct_step, step_metrics
from response_analysis.protocol import CHANNELS
from .helpers import RunTestCase


class AnalysisTests(RunTestCase):
    def test_all_six_loops_first_order_gain_and_frf(self):
        path = self.make(channels=list(CHANNELS), duration_s=80, gain=0.65, delay_s=0.04, noise_std=0)
        result = analyze_run(path)
        for channel, entry in result["channels"].items():
            with self.subTest(channel=channel):
                self.assertEqual(entry["status"], "ok")
                self.assertAlmostEqual(entry["metrics"]["dc_gain"], 0.65, delta=0.04)
                self.assertAlmostEqual(entry["metrics"]["rise_time_s"], 0.04 * np.log(9), delta=0.02)
                spectrum = entry["spectrum"]
                f = np.asarray(spectrum["frequency_hz"])
                h = np.asarray(spectrum["transfer_real"]) + 1j * np.asarray(spectrum["transfer_imag"])
                alpha = np.exp(-1 / 4)
                z = np.exp(-2j * np.pi * f / 100)
                expected = 0.65 * (1 - alpha) / (1 - alpha * z) * z ** 4
                band = (f >= 1) & (f <= 12)
                self.assertLess(np.mean(np.abs(h[band] / expected[band] - 1)), 0.035)
        json.dumps(result, allow_nan=False)

    def test_companion_files_do_not_change_numeric_results(self):
        path = self.make(channels=list(CHANNELS), noise_std=0)
        full = analyze_run(path)
        (path / "metadata.json").write_text(json.dumps({
            "analysis_intervals_s": [[0, 1]], "timebase": {"aligned": False},
            "channels": {key: {"available": False} for key in CHANNELS},
        }))
        minimal = analyze_run(path / "samples.csv")
        for channel in CHANNELS:
            for key in ("metrics", "metric_reasons", "spectrum", "step_response", "preprocessing", "spectral_settings"):
                with self.subTest(channel=channel, key=key):
                    self.assertEqual(full["channels"][channel].get(key), minimal["channels"][channel].get(key))
        self.assertNotIn("metadata", minimal)
        self.assertNotIn("metadata_sources", minimal)
        self.assertNotIn("provenance", minimal["channels"]["angle.roll"])
        self.assertNotIn("unknown", json.dumps(minimal))

    def test_floating_timestamp_does_not_add_reference_sample_delay(self):
        path = self.make(duration_s=20, noise_std=0)
        data = load_run(path)
        blocks, _, error = prepare_channel(data, "angle.roll", AnalysisConfig())
        self.assertIsNone(error)
        t, r, _ = blocks[0]
        indices = np.round(t * 100).astype(int)
        np.testing.assert_allclose(r, data.samples.angle_ref_roll_deg.to_numpy()[indices], atol=1e-9)

    def test_delay_is_not_aligned_away(self):
        a = analyze_run(self.make("a", delay_s=0.02, duration_s=80, noise_std=0))["channels"]["angle.roll"]
        b = analyze_run(self.make("b", delay_s=0.06, duration_s=80, noise_std=0))["channels"]["angle.roll"]
        f = np.asarray(a["spectrum"]["frequency_hz"])
        h = lambda c: np.asarray(c["spectrum"]["transfer_real"]) + 1j * np.asarray(c["spectrum"]["transfer_imag"])
        band = (f >= 1) & (f <= 8)
        phase_change = np.unwrap(np.angle(h(b) / h(a)))
        np.testing.assert_allclose(phase_change[band], -2 * np.pi * f[band] * 0.04, atol=0.06)

    def test_second_order_overshoot_and_peak(self):
        entry = analyze_run(self.make(
            duration_s=100, second_order=True, damping=0.4,
            natural_frequency_hz=4, delay_s=0.02, noise_std=0
        ))["channels"]["angle.roll"]
        expected = 100 * np.exp(-np.pi * 0.4 / np.sqrt(1 - 0.4 ** 2))
        self.assertAlmostEqual(entry["metrics"]["overshoot_pct"], expected, delta=4)
        self.assertAlmostEqual(
            entry["metrics"]["peak_time_s"], 0.02 + 1 / (8 * np.sqrt(1 - 0.4 ** 2)), delta=0.03
        )

    def test_impulse_integration_preserves_gain_and_delay(self):
        fs, n = 128, 256
        impulse = np.zeros(n)
        impulse[8] = 0.7
        h = np.fft.rfft(impulse)
        for cutoff in (129, 40):
            mask = np.arange(len(h)) < cutoff
            step = reconstruct_step(h, mask, fs, n, 1)
            self.assertAlmostEqual(step["values"][-1], 0.7, delta=0.002)
            crossing = np.argmax(step["values"] >= 0.35)
            self.assertAlmostEqual(step["time_s"][crossing], 8 / fs, delta=1 / fs)
        step = reconstruct_step(np.full(129, 0.65), np.arange(129) < 30, fs, n, 1)
        self.assertAlmostEqual(step["values"][-1], 0.65, delta=0.002)

    def test_direct_analytic_metric_definitions(self):
        t = np.arange(0, 1, 0.001)
        y = 0.8 * (1 - np.exp(-np.maximum(t - 0.03, 0) / 0.08))
        metrics, reasons = step_metrics(t, y, AnalysisConfig())
        self.assertAlmostEqual(metrics["dc_gain"], 0.8, delta=0.001)
        self.assertAlmostEqual(metrics["steady_state_error"], 0.2, delta=0.001)
        self.assertAlmostEqual(metrics["rise_time_s"], 0.08 * np.log(9), delta=0.002)
        self.assertAlmostEqual(metrics["settling_time_s"], 0.03 + 0.08 * np.log(50), delta=0.003)
        self.assertIsNone(metrics["peak_time_s"])
        negative, _ = step_metrics(t, -y, AnalysisConfig())
        self.assertAlmostEqual(negative["dc_gain"], -0.8, delta=0.001)
        self.assertAlmostEqual(negative["rise_time_s"], metrics["rise_time_s"])

    def test_unsettled_zero_and_short_responses(self):
        t = np.linspace(0, 1, 101)
        for response in (t, np.sin(20 * t), np.zeros(len(t))):
            metrics, reasons = step_metrics(t, response, AnalysisConfig())
            self.assertTrue(all(value is None for value in metrics.values()))
            self.assertTrue(all(key in reasons for key in metrics))
        metrics, reasons = step_metrics(t[:5], t[:5], AnalysisConfig())
        self.assertTrue(all(value is None for value in metrics.values()))

    def test_asynchronous_downsample_filters_aliasing(self):
        path = self.make(duration_s=10, fs=200)
        def edit(frame):
            frame["angle_ref_roll_deg"] = np.where(frame.index % 4 == 0, np.sin(2 * np.pi * frame.t_s), np.nan)
            frame["angle_meas_roll_deg"] = np.sin(2 * np.pi * 80 * frame.t_s)
        self.samples(path, edit)
        blocks, details, _ = prepare_channel(load_run(path), "angle.roll", AnalysisConfig())
        self.assertAlmostEqual(details["sample_rate_hz"], 50)
        self.assertTrue(details["anti_alias_downsampled"])
        self.assertLess(np.sqrt(np.mean(blocks[0][2] ** 2)), 0.015)

    def test_long_gap_and_analysis_intervals_never_bridged(self):
        path = self.make(duration_s=30)
        def edit(frame):
            frame.loc[(frame.t_s > 5) & (frame.t_s < 15), "angle_meas_roll_deg"] = np.nan
        self.samples(path, edit)
        config = AnalysisConfig(analysis_intervals_s=[[1, 20], [22, 28]])
        blocks, details, _ = prepare_channel(load_run(path), "angle.roll", config)
        self.assertEqual(details["measurement_long_gaps"], 1)
        self.assertEqual(len(blocks), 3)
        for t, _, _ in blocks:
            self.assertFalse(t[0] < 5 and t[-1] > 15)
            self.assertFalse(t[0] < 20 and t[-1] > 22)
        result = analyze_run(path, config)["channels"]["angle.roll"]
        self.assertEqual(result["status"], "ok")

    def test_intervals_select_tracking_samples_without_rebasing_time(self):
        path = self.make(duration_s=30)
        self.samples(path, lambda f: f.assign(
            angle_meas_roll_deg=f.angle_ref_roll_deg + np.where(f.t_s < 15, 1.0, 4.0)
        ))
        for interval, expected_bias in (([1, 10], 1.0), ([18, 28], 4.0)):
            result = analyze_run(path, AnalysisConfig(analysis_intervals_s=[interval]))
            self.assertEqual(result["analysis_intervals_s"], [interval])
            entry = result["channels"]["angle.roll"]
            self.assertAlmostEqual(entry["metrics"]["tracking_bias"], expected_bias, places=6)
            self.assertGreaterEqual(entry["time_series"][0]["time_s"][0], interval[0])
            self.assertLessEqual(entry["time_series"][-1]["time_s"][-1], interval[1])

    def test_out_of_range_intervals_are_not_silently_clipped(self):
        path = self.make(duration_s=10)
        with self.assertRaisesRegex(ValidationError, "recorded time range"):
            analyze_run(path, AnalysisConfig(analysis_intervals_s=[[5, 20]]))

    def test_angle_wrap_does_not_create_error_spikes(self):
        path = self.make(channels=["angle.yaw"], noise_std=0)
        original = analyze_run(path)["channels"]["angle.yaw"]
        self.samples(path, lambda f: f.assign(
            angle_ref_yaw_deg=(f.angle_ref_yaw_deg + 359) % 360 - 180,
            angle_meas_yaw_deg=(f.angle_meas_yaw_deg + 359) % 360 - 180,
        ))
        wrapped = analyze_run(path)["channels"]["angle.yaw"]
        self.assertAlmostEqual(original["metrics"]["tracking_rmse"], wrapped["metrics"]["tracking_rmse"], places=8)
        np.testing.assert_allclose(original["spectrum"]["transfer_real"], wrapped["spectrum"]["transfer_real"], atol=1e-8)

    def test_unrecorded_rates_not_derived_from_angles(self):
        result = analyze_run(self.make())
        for channel in ("rate.x", "rate.y", "rate.z", "angle.pitch", "angle.yaw"):
            self.assertEqual(result["channels"][channel]["status"], "unavailable")

    def test_constant_short_and_uncorrelated_data(self):
        constant = self.make("constant")
        self.samples(constant, lambda f: f.assign(angle_ref_roll_deg=1))
        self.assertEqual(analyze_run(constant)["channels"]["angle.roll"]["status"], "insufficient_data")
        short = self.make("short", duration_s=3)
        self.assertEqual(analyze_run(short)["channels"]["angle.roll"]["status"], "insufficient_data")
        noise = self.make("noise", duration_s=100)
        self.samples(noise, lambda f: f.assign(angle_meas_roll_deg=np.random.default_rng(22).normal(size=len(f))))
        entry = analyze_run(noise)["channels"]["angle.roll"]
        self.assertEqual(entry["status"], "insufficient_data")
        self.assertIsNone(entry["metrics"]["dc_gain"])
        json.dumps(entry, allow_nan=False)

    def test_long_requested_window_does_not_allocate_unbounded_fft(self):
        path = self.make(duration_s=1)
        result = analyze_run(path, AnalysisConfig(window_s=1e9))["channels"]["angle.roll"]
        self.assertEqual(result["status"], "insufficient_data")
        self.assertIsNotNone(result["metrics"]["tracking_rmse"])

    def test_hann_null_excitation_does_not_produce_nan_transfer(self):
        path = self.make(duration_s=8, fs=128)
        def edit(frame):
            values = np.zeros(len(frame))
            values[::256] = [1, -1, 1, -1]
            frame["angle_ref_roll_deg"] = values
            frame["angle_meas_roll_deg"] = values
        self.samples(path, edit)
        result = analyze_run(path, AnalysisConfig(overlap=0))["channels"]["angle.roll"]
        self.assertEqual(result["status"], "insufficient_data")
        self.assertIn("zero_or_nonfinite_windowed_spectral_power", result["reasons"])
        json.dumps(result, allow_nan=False)

    def test_tracking_bias_is_not_removed_by_identification_centering(self):
        path = self.make(noise_std=0)
        original = analyze_run(path)["channels"]["angle.roll"]
        self.samples(path, lambda f: f.assign(angle_meas_roll_deg=f.angle_meas_roll_deg + 3))
        biased = analyze_run(path)["channels"]["angle.roll"]
        self.assertAlmostEqual(biased["metrics"]["tracking_bias"] - original["metrics"]["tracking_bias"], 3)
        self.assertAlmostEqual(biased["metrics"]["dc_gain"], original["metrics"]["dc_gain"], places=9)

    def test_unsettled_estimated_integrator_is_not_given_a_final_gain(self):
        path = self.make(duration_s=80)
        self.samples(path, lambda f: f.assign(angle_meas_roll_deg=np.cumsum(f.angle_ref_roll_deg) / 100))
        result = analyze_run(path)["channels"]["angle.roll"]
        self.assertIsNone(result["metrics"]["dc_gain"])
        self.assertTrue(result["metric_reasons"]["dc_gain"])

    def test_input_correlation_diagnostic(self):
        path = self.make(channels=["angle.roll", "angle.pitch"])
        self.samples(path, lambda f: f.assign(angle_ref_pitch_deg=f.angle_ref_roll_deg))
        correlations = analyze_run(path)["input_correlations"]
        self.assertEqual(len(correlations), 1)
        self.assertTrue(correlations[0]["high_correlation"])
        self.assertAlmostEqual(correlations[0]["correlation"], 1)
