import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
import signal
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

from crazyflie_sim.auto_client import AcquisitionError, AutoClient, AutoConfig, assess_quality, excitation, main
from crazyflie_sim.api.server import SimulatorAPIHandler
from crazyflie_sim.flight_recorder import COLUMNS, FlightRecorder
from .test_recording_integration import load_simulation_manager


class Response:
    def __init__(self, value=None, content=b"", status=200):
        self.value, self.content, self.status_code = value, content, status

    def json(self):
        return self.value

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.now += duration
        time.sleep(0)  # Let the real CSV writer consume the simulated samples.


class FakeSimulator:
    """HTTP-shaped test double, not a validation of real aircraft dynamics."""

    def __init__(self, directory, frozen_feedback=False):
        self.recorder = FlightRecorder(directory)
        self.dt = 0.005
        self.t = 0.0
        self.closed = False
        self.frozen_feedback = frozen_feedback
        self.posts = []
        self.service = "crazyflie_sim"
        self.stalled = False
        self.mode = "attitude"
        self.command = {"roll": 0, "pitch": 0, "yaw_rate": 0, "thrust": 0}
        self.reset_state()

    def reset_state(self):
        self.z, self.vz = 0.1, 0.0
        self.angles, self.rates = np.zeros(3), np.zeros(3)

    def tick(self):
        if self.stalled:
            return
        if self.mode == "position":
            angle_ref = np.array([0, 0, self.command["yaw"]], float)
            rate_ref = np.array([15, 15, 5]) * (angle_ref - self.angles)
            acceleration = 4 * (self.command["z"] - self.z) - 3 * self.vz
        else:
            angle_ref = np.array([30 * self.command["roll"], 30 * self.command["pitch"], 0.0])
            rate_ref = 15 * (angle_ref - self.angles)
            rate_ref[2] = 120 * self.command["yaw_rate"]
            tilt = math.cos(math.radians(self.angles[0])) * math.cos(math.radians(self.angles[1]))
            acceleration = 4 * 0.4 * self.command["thrust"] / 0.05 * tilt - 9.81
        values = tuple(value for pair in zip(angle_ref, self.angles) for value in pair)
        values += tuple(value for pair in zip(rate_ref, self.rates) for value in pair)
        self.recorder.record(self.t, self.mode, values)
        if self.recorder._queue.qsize() > 1024:
            time.sleep(0.001)
        if not self.frozen_feedback:
            alpha = math.exp(-self.dt / 0.04)
            self.rates = alpha * self.rates + (1 - alpha) * rate_ref
            self.angles += self.dt * self.rates
        self.vz += self.dt * acceleration
        self.z += self.dt * self.vz
        if self.z < 0.03:
            self.z, self.vz = 0.03, 0.0
        self.t += self.dt

    def state(self):
        return {
            "timestamp": self.t,
            "position": {"x": 0.0, "y": 0.0, "z": self.z},
            "velocity": {"x": 0.0, "y": 0.0, "z": self.vz},
            "orientation": dict(zip(("roll", "pitch", "yaw"), self.angles.tolist())),
            "angular_velocity": dict(zip(("x", "y", "z"), self.rates.tolist())),
        }

    def get(self, url, **kwargs):
        path = url.split(":8000")[-1]
        if path == "/recording/status":
            return Response({"service": self.service, "dt_s": self.dt, "recording": self.recorder.status()})
        if path == "/controller/params":
            return Response({"mass": 0.05, "thrust_max": 0.4})
        if path == "/controller/debug":
            pwm = 30000 if self.mode == "position" else self.command["thrust"] * 65535
            return Response({"thrust_pwm": [pwm], "motor_pwm": [[pwm] * 4]})
        if path == "/state":
            self.tick()
            return Response(self.state())
        if path == "/recording/file":
            name = kwargs["params"]["name"]
            if name not in self.recorder.status()["completed"]:
                return Response(status=404)
            return Response(content=(self.recorder.directory / name).read_bytes())
        raise AssertionError(path)

    def post(self, url, json, **kwargs):
        path = url.split(":8000")[-1]
        self.posts.append((path, copy.deepcopy(json)))
        if path == "/reset":
            self.reset_state()
            self.recorder.split()
            self.mode = "attitude"
            self.command = {"roll": 0, "pitch": 0, "yaw_rate": 0, "thrust": 0}
        elif path in ("/control/position", "/control/attitude"):
            self.mode = path.rsplit("/", 1)[-1]
            self.command = dict(json)
        else:
            raise AssertionError(path)
        return Response({"status": "ok"})

    def close(self):
        self.closed = True


class AutoClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="auto-client-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def client(self, **kwargs):
        simulator = FakeSimulator(self.root / "server", **kwargs)
        self.addCleanup(simulator.recorder.close)
        clock = FakeClock()
        client = AutoClient(
            config=AutoConfig(duration_s=20), log_dir=self.root / "client",
            session=simulator, clock=clock, sleep=clock.sleep,
        )
        return client, simulator, clock

    def test_excitation_is_bounded_balanced_and_reproducible(self):
        values = excitation(20, 100, 42)
        np.testing.assert_array_equal(values, excitation(20, 100, 42))
        self.assertFalse(np.array_equal(values, excitation(20, 100, 43)))
        self.assertEqual(set(values), {-1, 1})
        self.assertLess(abs(values.mean()), 0.001)
        power = abs(np.fft.rfft(values)) ** 2
        self.assertGreater(np.count_nonzero(power[1:] > power.max() * 1e-4), 0.9 * len(power[1:]))

    def test_complete_mock_flight_records_all_six_target_channels(self):
        client, simulator, _ = self.client()
        passed = client.run()
        self.assertTrue(simulator.closed)
        self.assertFalse(client.motion_started)
        self.assertEqual(simulator.posts[-1], ("/control/attitude", {"roll": 0, "pitch": 0, "yaw_rate": 0, "thrust": 0}))
        result = json.loads((client.output / "quality.json").read_text())
        self.assertIsNone(result["acquisition_error"])
        self.assertEqual(len(result["results"]), 4)
        channels = {key for item in result["results"] for key in item["channels"]}
        self.assertEqual(channels, {"angle.roll", "angle.pitch", "angle.yaw", "rate.x", "rate.y", "rate.z"})
        self.assertEqual(passed, result["passed"])
        self.assertTrue(passed, json.dumps(result["results"], indent=2))
        for item in result["results"]:
            frame = pd.read_csv(client.output / item["file"])
            self.assertEqual(tuple(frame.columns), COLUMNS)
            self.assertGreater(len(frame), 3900)
            self.assertTrue((client.output / "reports" / Path(item["file"]).stem / "report.html").is_file())

    def test_frozen_measurements_never_report_quality_success(self):
        client, simulator, _ = self.client(frozen_feedback=True)
        self.assertFalse(client.run())
        result = json.loads((client.output / "quality.json").read_text())
        self.assertFalse(result["passed"])
        self.assertTrue(all(not item["passed"] for item in result["results"]))
        self.assertEqual(simulator.posts[-1][1]["thrust"], 0)

    def test_preflight_rejects_unidentified_server_without_commands(self):
        client, simulator, _ = self.client()
        simulator.service = "not-a-simulator"
        with self.assertRaisesRegex(AcquisitionError, "expected Crazyflie"):
            client.run()
        self.assertEqual(simulator.posts, [])

    def test_preflight_requires_enabled_recorder_and_grounded_vehicle(self):
        client, simulator, _ = self.client()
        simulator.recorder.close()
        with self.assertRaisesRegex(AcquisitionError, "--record"):
            client.run()
        self.assertEqual(simulator.posts, [])
        other = FakeSimulator(self.root / "second-server")
        self.addCleanup(other.recorder.close)
        other.z = 1.0
        clock = FakeClock()
        client = AutoClient(session=other, clock=clock, sleep=clock.sleep, log_dir=self.root)
        with self.assertRaisesRegex(AcquisitionError, "grounded"):
            client.run()
        self.assertEqual(other.posts, [])

    def test_stalled_clock_fails_before_takeoff(self):
        client, simulator, _ = self.client()
        simulator.stalled = True
        with self.assertRaisesRegex(AcquisitionError, "stalled"):
            client.run()
        self.assertEqual(simulator.posts, [])

    def test_commands_faster_than_control_loop_are_rejected_before_motion(self):
        client, simulator, _ = self.client()
        client.config = AutoConfig(command_hz=300)
        with self.assertRaisesRegex(AcquisitionError, "exceeds"):
            client.run()
        self.assertEqual(simulator.posts, [])

    def test_infeasible_hover_is_rejected_before_motion(self):
        client, simulator, _ = self.client()
        original = simulator.get

        def get(url, **kwargs):
            if url.endswith("/controller/params"):
                return Response({"mass": 0.1, "thrust_max": 0.05})
            return original(url, **kwargs)

        with patch.object(simulator, "get", get):
            with self.assertRaisesRegex(AcquisitionError, "thrust margin"):
                client.run()
        self.assertEqual(simulator.posts, [])

    def test_failure_after_takeoff_sends_stop_and_preserves_failure_report(self):
        client, simulator, _ = self.client()
        with patch.object(client, "_settle", side_effect=AcquisitionError("hover failed")):
            with self.assertRaisesRegex(AcquisitionError, "hover failed"):
                client.run()
        self.assertEqual(simulator.posts[-1][1]["thrust"], 0)
        result = json.loads((client.output / "quality.json").read_text())
        self.assertFalse(result["passed"])
        self.assertIn("hover failed", result["acquisition_error"])

    def test_landing_does_not_claim_success_without_motor_stop_feedback(self):
        client, simulator, _ = self.client()
        client.verified = client.motion_started = True
        client.origin, client.mass, client.motor_thrust_max = (0, 0, 0), 0.05, 0.4
        state = simulator.state()
        state["position"]["z"] = 0.03
        with patch.object(client, "_state", return_value=state), patch.object(client, "_timed"):
            original = client._json

            def request(path, payload=None):
                if path == "/controller/debug":
                    return {"thrust_pwm": [100], "motor_pwm": [[100] * 4]}
                return original(path, payload)

            with patch.object(client, "_json", request):
                with self.assertRaisesRegex(AcquisitionError, "motors did not stop"):
                    client._land()
        self.assertTrue(client.motion_started)

    def test_safety_bounds_and_unexpected_clock_reset(self):
        client, simulator, _ = self.client()
        client.origin = (0, 0, 0)
        client.airborne = True
        state = simulator.state()
        state["position"]["z"] = 1
        client._check_bounds(state)
        for section, field, value in (
            ("position", "z", 2), ("position", "z", 0.1), ("position", "x", 2),
            ("orientation", "roll", 16), ("velocity", "x", 3), ("angular_velocity", "x", 121),
        ):
            changed = copy.deepcopy(state)
            changed[section][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(AcquisitionError):
                client._check_bounds(changed)
        client._state()
        simulator.t = -1
        with self.assertRaisesRegex(AcquisitionError, "backwards"):
            client._state()

    def test_client_commands_are_bounded_and_use_degree_normalization(self):
        client, simulator, _ = self.client()
        client.origin, client.mass, client.motor_thrust_max = (0, 0, 0), 0.05, 0.4
        state = simulator.state()
        state["position"]["z"] = 1
        client._attitude(state, "roll", 1)
        self.assertAlmostEqual(simulator.posts[-1][1]["roll"], 2 / 30)
        self.assertAlmostEqual(simulator.posts[-1][1]["thrust"], 0.05 * 9.81 / 1.6)
        client._attitude(state, "yaw_rate", -1)
        self.assertAlmostEqual(simulator.posts[-1][1]["yaw_rate"], -10 / 120)

    def test_config_rejects_short_weak_or_unsafe_protocols(self):
        for kwargs in (
            {"duration_s": 5}, {"command_hz": 5}, {"repeats": 0}, {"height_m": 3},
            {"max_frequency_hz": 50}, {"angle_amplitude_deg": 10}, {"seed": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AutoConfig(**kwargs)

    def test_invalid_telemetry_and_recorder_session_change_fail(self):
        client, simulator, _ = self.client()
        status = client._recording()
        client.recording_session = status["recording"]["session"]
        client.verified = client.motion_started = True
        simulator.recorder._session = "changed"
        with self.assertRaisesRegex(AcquisitionError, "session changed"):
            client._recording()
        self.assertFalse(client._stop())
        self.assertEqual(simulator.posts, [])
        with patch.object(simulator, "get", return_value=Response({"timestamp": 0})):
            with self.assertRaisesRegex(AcquisitionError, "Invalid simulator telemetry"):
                client._state()

    def test_cli_reports_quality_failure_as_nonzero(self):
        for passed, expected in ((True, 0), (False, 3)):
            fake = type("Client", (), {"output": self.root, "run": lambda self: passed})()
            with patch("crazyflie_sim.auto_client.AutoClient", return_value=fake):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.assertEqual(main(["--log-dir", str(self.root)]), expected)

    def test_cli_restores_signal_handler_after_termination(self):
        original = signal.getsignal(signal.SIGTERM)
        fake = type("Client", (), {
            "run": lambda self: signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None),
        })()
        with patch("crazyflie_sim.auto_client.AutoClient", return_value=fake):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--log-dir", str(self.root)]), 2)
        self.assertIs(signal.getsignal(signal.SIGTERM), original)


class QualityAndAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="auto-quality-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def data(self):
        from response_analysis.examples import make_run
        return make_run(self.root / "source", fs=200, duration_s=24, noise_std=0) / "samples.csv"

    def quality(self, path, command_step=0.01):
        return assess_quality(path, ("rate.x",), 200, 15, 24, np.arange(0, 24, command_step), 100)[0]

    def test_good_data_passes_and_slow_commands_fail(self):
        path = self.data()
        self.assertTrue(self.quality(path)["passed"])
        bad = self.quality(path, command_step=0.05)
        self.assertFalse(bad["passed"])
        self.assertIn("command_cadence_insufficient", bad["reasons"])

    def test_quality_does_not_require_unit_gain(self):
        from response_analysis.examples import make_run
        path = make_run(self.root / "nonunit", fs=200, duration_s=24, gain=0.6, noise_std=0) / "samples.csv"
        result = self.quality(path)
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["channels"]["rate.x"]["metrics"]["dc_gain"], 0.6, delta=0.05)

    def test_single_tone_does_not_pass_broadband_quality(self):
        path = self.data()
        frame = pd.read_csv(path)
        frame["rate_ref_x_dps"] = 10 * np.sin(2 * np.pi * frame.t_s)
        frame["rate_meas_x_dps"] = 8 * np.sin(2 * np.pi * frame.t_s - 0.2)
        frame.to_csv(path, index=False)
        result = self.quality(path)
        self.assertFalse(result["passed"])
        self.assertIn("insufficient_trusted_frequency_coverage", result["channels"]["rate.x"]["reasons"])

    def test_gaps_and_weak_input_fail(self):
        path = self.data()
        frame = pd.read_csv(path)
        frame.loc[~frame.t_s.between(8, 10)].to_csv(path, index=False)
        bad = self.quality(path)
        self.assertFalse(bad["passed"])
        self.assertIn("recording_has_long_gaps", bad["channels"]["rate.x"]["reasons"])
        frame["rate_ref_x_dps"] = 0
        frame.to_csv(path, index=False)
        self.assertIn("insufficient_reference_amplitude", self.quality(path)["channels"]["rate.x"]["reasons"])

    def test_cross_axis_excitation_and_recorded_safety_violations_fail(self):
        path = self.data()
        frame = pd.read_csv(path)
        frame["rate_ref_y_dps"] = frame["rate_ref_x_dps"]
        frame.to_csv(path, index=False)
        result = self.quality(path)
        self.assertFalse(result["passed"])
        self.assertIn("correlated_input_axes:rate.y", result["channels"]["rate.x"]["reasons"])
        frame.loc[10, "angle_meas_roll_deg"] = 20
        frame.to_csv(path, index=False)
        self.assertIn("recorded_safety_bound_exceeded:angle_meas_roll_deg", self.quality(path)["reasons"])

    def test_recording_status_and_download_are_read_only_and_session_scoped(self):
        Manager = load_simulation_manager()
        manager = Manager.__new__(Manager)
        manager.dt = 0.005
        manager.recorder = FlightRecorder(self.root / "logs")
        self.addCleanup(manager.recorder.close)
        manager.recorder.record(0, "position", range(12))
        manager.recorder.close()
        status = manager.get_recording_status()
        self.assertEqual(status["service"], "crazyflie_sim")
        name = status["recording"]["completed"][0]
        self.assertTrue(manager.read_recording(name).startswith(b"t_s,"))
        status["recording"]["completed"].append("invented.csv")
        with self.assertRaises(FileNotFoundError):
            manager.read_recording("invented.csv")
        secret = manager.recorder.directory / "other.csv"
        secret.write_text("not a recorder output")
        for name in ("../other.csv", "other.csv", "unfinished.csv.partial"):
            with self.subTest(name=name), self.assertRaises(FileNotFoundError):
                manager.read_recording(name)
        owned_name = status["recording"]["completed"][0]
        owned_path = manager.recorder.directory / owned_name
        owned_path.unlink()
        owned_path.symlink_to(secret)
        with self.assertRaisesRegex(ValueError, "Invalid recording path"):
            manager.read_recording(owned_name)

    def test_http_routes_return_csv_and_reject_bad_names(self):
        handler = SimulatorAPIHandler.__new__(SimulatorAPIHandler)
        handler.simulation_manager = type("Manager", (), {
            "get_recording_status": lambda self: {"service": "crazyflie_sim"},
            "read_recording": lambda self, name: b"t_s\n0\n" if name == "owned.csv" else (_ for _ in ()).throw(FileNotFoundError()),
        })()
        handler.wfile = io.BytesIO()
        with patch.object(handler, "_send_json") as json_response:
            handler.path = "/recording/status"
            handler.do_GET()
            json_response.assert_called_with({"service": "crazyflie_sim"})
            handler.path = "/recording/file"
            handler.do_GET()
            self.assertEqual(json_response.call_args.args[1], 400)
            handler.path = "/recording/file?name=../secret"
            handler.do_GET()
            self.assertEqual(json_response.call_args.args[1], 404)
        with patch.object(handler, "send_response") as status, patch.object(handler, "send_header"), patch.object(handler, "end_headers"):
            handler.path = "/recording/file?name=owned.csv"
            handler.do_GET()
            status.assert_called_with(200)
            self.assertEqual(handler.wfile.getvalue(), b"t_s\n0\n")
