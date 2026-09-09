#!/usr/bin/env python3
"""Simulator-only excitation, recording retrieval and measured-data acceptance."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path
import re
import signal
import sys
import time

import numpy as np
import pandas as pd
import requests

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crazyflie_sim.flight_recorder import DEFAULT_LOG_DIR
from response_analysis import AnalysisConfig, analyze_run, load_run
from response_analysis.report import write_report


logger = logging.getLogger("auto_client")
PHASES = (
    ("roll", "attitude", ("angle.roll", "rate.x")),
    ("pitch", "attitude", ("angle.pitch", "rate.y")),
    ("yaw_angle", "position", ("angle.yaw",)),
    ("yaw_rate", "attitude", ("rate.z",)),
)
ANALYSIS = AnalysisConfig(window_s=4.0, response_s=2.0, min_windows=8, coherence_min=0.8)


class AcquisitionError(RuntimeError):
    pass


@dataclass(frozen=True)
class AutoConfig:
    duration_s: float = 32.0
    repeats: int = 1
    command_hz: float = 100.0
    height_m: float = 1.0
    angle_amplitude_deg: float = 2.0
    yaw_angle_amplitude_deg: float = 5.0
    yaw_rate_amplitude_dps: float = 10.0
    max_frequency_hz: float = 15.0
    seed: int = 0

    def __post_init__(self):
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("Auto-client settings must be finite numbers")
        if not isinstance(self.repeats, int) or self.repeats < 1:
            raise ValueError("repeats must be a positive integer")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.duration_s < 20:
            raise ValueError("At least 20 seconds of excitation are required per phase")
        if not 10 <= self.command_hz <= 500:
            raise ValueError("command_hz must be between 10 and 500")
        if not 0.5 <= self.max_frequency_hz < self.command_hz / 2:
            raise ValueError("The analysis band must start at 0.5 Hz and end below command Nyquist")
        if not 0.6 <= self.height_m <= 1.5:
            raise ValueError("height_m must be between 0.6 and 1.5")
        if not 0.5 <= self.angle_amplitude_deg <= 3:
            raise ValueError("angle amplitude must be between 0.5 and 3 degrees")
        if not 1 <= self.yaw_angle_amplitude_deg <= 8 or not 2 <= self.yaw_rate_amplitude_dps <= 20:
            raise ValueError("Yaw angle/rate amplitudes must be within 1-8 deg / 2-20 deg/s")


def excitation(duration_s, hz, seed):
    """Bounded, repeatable broadband binary commands; no output-dependent tuning."""
    rng = np.random.default_rng(seed)
    count = int(math.ceil(duration_s * hz)) + 1
    values = np.tile((-1.0, 1.0), (count + 1) // 2)[:count]
    rng.shuffle(values)
    return values


def wrap_degrees(value):
    return (value + 180.0) % 360.0 - 180.0


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("expected a finite numeric value")
    return float(value)


def assess_quality(path, targets, expected_hz, max_frequency, expected_duration, command_times, command_hz):
    """Gate on actual recorded data, not merely on a successful HTTP flight."""
    data = load_run(path)
    analysis = analyze_run(data, ANALYSIS)
    reasons = []
    duration = float(data.samples.t_s.iloc[-1] - data.samples.t_s.iloc[0])
    if duration < expected_duration - 3 / expected_hz:
        reasons.append("recording_does_not_cover_excitation_interval")
    gaps = np.diff(command_times)
    achieved_fraction = len(command_times) / max(1, expected_duration * command_hz)
    if achieved_fraction < 0.9 or (len(gaps) and np.max(gaps) > 3 / command_hz + 1e-8):
        reasons.append("command_cadence_insufficient")
    for column, bound in (
        ("angle_meas_roll_deg", 15), ("angle_meas_pitch_deg", 15),
        ("rate_meas_x_dps", 120), ("rate_meas_y_dps", 120), ("rate_meas_z_dps", 120),
    ):
        if column in data.samples and (data.samples[column].abs() > bound).any():
            reasons.append(f"recorded_safety_bound_exceeded:{column}")
    channels = {}
    for channel in targets:
        entry = analysis["channels"][channel]
        failures = []
        prep = entry.get("preprocessing", {})
        if entry["status"] != "ok":
            failures.extend(entry["reasons"])
        if prep.get("sample_rate_hz", 0) < 0.95 * expected_hz:
            failures.append("recording_rate_below_simulator_rate")
        if prep.get("reference_long_gaps", 0) or prep.get("measurement_long_gaps", 0):
            failures.append("recording_has_long_gaps")
        prefix, axis = channel.split(".")
        column = f"{prefix}_ref_{axis}_{'deg' if prefix == 'angle' else 'dps'}"
        values = data.samples[column].dropna().to_numpy() if column in data.samples else np.array([])
        reference_std = float(np.std(values)) if len(values) else 0.0
        if reference_std < (0.1 if prefix == "angle" else 1.0):
            failures.append("insufficient_reference_amplitude")
        for correlation in analysis["input_correlations"]:
            if channel not in correlation["channels"] or not correlation["high_correlation"]:
                continue
            other = next(key for key in correlation["channels"] if key != channel)
            other_prefix, other_axis = other.split(".")
            other_col = f"{other_prefix}_ref_{other_axis}_{'deg' if other_prefix == 'angle' else 'dps'}"
            if data.samples[other_col].std() >= 0.1 * reference_std:
                failures.append(f"correlated_input_axes:{other}")
        spectrum = entry.get("spectrum")
        coverage = 0.0
        if spectrum:
            frequencies = np.asarray(spectrum["frequency_hz"])
            band = (frequencies >= 0.5) & (frequencies <= max_frequency)
            if band.any():
                coverage = float(np.mean(np.asarray(spectrum["reliable"])[band]))
        if coverage < 0.8:
            failures.append("insufficient_trusted_frequency_coverage")
        for metric in ("dc_gain", "rise_time_s", "settling_time_s"):
            if entry["metrics"].get(metric) is None:
                failures.append(f"{metric}:{entry['metric_reasons'].get(metric, 'unavailable')}")
        channels[channel] = {
            "passed": not failures, "reasons": failures, "trusted_band_fraction": coverage,
            "reference_std": reference_std, "metrics": entry["metrics"],
        }
    quality = {
        "passed": not reasons and all(value["passed"] for value in channels.values()),
        "reasons": reasons, "channels": channels, "duration_s": duration,
        "command_fraction": achieved_fraction,
        "requested_band_hz": [0.5, max_frequency],
    }
    return quality, analysis


class AutoClient:
    """One sequential client, using only the simulator HTTP control API."""

    def __init__(self, host="localhost", port=8000, config=None, log_dir=DEFAULT_LOG_DIR,
                 session=None, clock=time.monotonic, sleep=time.sleep):
        self.base_url = f"http://{host}:{port}"
        self.config = config or AutoConfig()
        self.log_dir = Path(log_dir).resolve()
        self.http = session or requests.Session()
        self.clock, self.sleep = clock, sleep
        self.verified = False
        self.motion_started = False
        self.airborne = False
        self.last_sim_time = None
        self.last_progress_wall = clock()
        self.recording_session = None
        self.origin = None
        self.expected_hz = None
        self.output = None
        self.results = []

    def _json(self, path, payload=None):
        response = (
            self.http.get(self.base_url + path, timeout=1.0) if payload is None
            else self.http.post(self.base_url + path, json=payload, timeout=1.0)
        )
        response.raise_for_status()
        value = response.json()
        if not isinstance(value, dict):
            raise AcquisitionError(f"{path}: expected a JSON object")
        return value

    def _recording(self):
        status = self._json("/recording/status")
        if status.get("service") != "crazyflie_sim":
            self.verified = False
            raise AcquisitionError("Endpoint is not the expected Crazyflie simulator")
        recording = status.get("recording")
        if self.recording_session is not None and (
            not isinstance(recording, dict) or recording.get("session") != self.recording_session
        ):
            self.verified = False  # Do not stop a replacement simulator we never took control of.
            raise AcquisitionError("Recorder session changed during the experiment")
        if isinstance(recording, dict) and recording.get("error"):
            raise AcquisitionError(f"Simulator recording failed: {recording['error']}")
        if not isinstance(recording, dict) or recording.get("enabled") is not True:
            raise AcquisitionError("Simulator recording is unavailable; start it with --record")
        session = recording.get("session")
        if not isinstance(session, str) or not session:
            raise AcquisitionError("Missing recorder session")
        if not isinstance(recording.get("completed"), list) or any(
            not isinstance(name, str) for name in recording["completed"]
        ):
            raise AcquisitionError("Invalid recorder file list")
        return status

    def _state(self):
        state = self._json("/state")
        try:
            for field, axes in (
                ("position", ("x", "y", "z")), ("velocity", ("x", "y", "z")),
                ("orientation", ("roll", "pitch", "yaw")), ("angular_velocity", ("x", "y", "z")),
            ):
                state[field] = {axis: finite_number(state[field][axis]) for axis in axes}
            timestamp = finite_number(state["timestamp"])
            state["timestamp"] = timestamp
        except (KeyError, TypeError, ValueError) as exc:
            raise AcquisitionError(f"Invalid simulator telemetry: {exc}") from exc
        now = self.clock()
        if self.last_sim_time is not None and timestamp < self.last_sim_time:
            raise AcquisitionError("Simulation clock moved backwards unexpectedly")
        if self.last_sim_time is None or timestamp > self.last_sim_time:
            self.last_progress_wall = now
        elif now - self.last_progress_wall > 5.0:
            raise AcquisitionError("Simulation clock stalled for five wall-clock seconds")
        self.last_sim_time = timestamp
        return state

    def _check_bounds(self, state):
        p, v, a = state["position"], state["velocity"], state["orientation"]
        if not -0.05 <= p["z"] <= self.config.height_m + 0.7:
            raise AcquisitionError("Height safety bound exceeded")
        if self.airborne and p["z"] < 0.25:
            raise AcquisitionError("Unexpected loss of flight height")
        if max(abs(a["roll"]), abs(a["pitch"])) > 15:
            raise AcquisitionError("Tilt safety bound exceeded")
        if math.hypot(p["x"] - self.origin[0], p["y"] - self.origin[1]) > 1.5:
            raise AcquisitionError("Horizontal safety radius exceeded")
        if math.sqrt(sum(v[axis] ** 2 for axis in ("x", "y", "z"))) > 2.0:
            raise AcquisitionError("Velocity safety bound exceeded")
        if max(abs(v) for v in state["angular_velocity"].values()) > 120:
            raise AcquisitionError("Body-rate safety bound exceeded")

    @staticmethod
    def _require_grounded(state):
        if (
            not -0.05 <= state["position"]["z"] <= 0.3
            or max(abs(v) for v in state["velocity"].values()) > 0.2
            or max(abs(v) for v in state["angular_velocity"].values()) > 10
        ):
            raise AcquisitionError("Start with a stationary grounded simulator; stop other control clients")

    def _position(self, z=None, yaw=None):
        self._json("/control/position", {
            "x": self.origin[0], "y": self.origin[1],
            "z": self.config.height_m if z is None else z,
            "yaw": self.origin[2] if yaw is None else yaw,
        })

    def _attitude(self, state, phase=None, value=0.0, height=None):
        p, v, a = state["position"], state["velocity"], state["orientation"]
        yaw = math.radians(a["yaw"])
        ax = 0.8 * (self.origin[0] - p["x"]) - 1.4 * v["x"]
        ay = 0.8 * (self.origin[1] - p["y"]) - 1.4 * v["y"]
        body_x = math.cos(yaw) * ax + math.sin(yaw) * ay
        body_y = -math.sin(yaw) * ax + math.cos(yaw) * ay
        roll = float(np.clip(-math.degrees(body_y / 9.81), -2, 2))
        pitch = float(np.clip(math.degrees(body_x / 9.81), -2, 2))
        yaw_rate = float(np.clip(1.5 * wrap_degrees(self.origin[2] - a["yaw"]), -10, 10))
        if phase == "roll":
            roll += value * self.config.angle_amplitude_deg
        elif phase == "pitch":
            pitch += value * self.config.angle_amplitude_deg
        elif phase == "yaw_rate":
            yaw_rate += value * self.config.yaw_rate_amplitude_dps
        height = self.config.height_m if height is None else height
        acceleration = 4.0 * (height - p["z"]) - 3.0 * v["z"]
        tilt = max(0.7, math.cos(math.radians(a["roll"])) * math.cos(math.radians(a["pitch"])))
        thrust = self.mass * (9.81 + acceleration) / (4 * self.motor_thrust_max * tilt)
        self._json("/control/attitude", {
            "roll": roll / 30.0, "pitch": pitch / 30.0, "yaw_rate": yaw_rate / 120.0,
            "thrust": float(np.clip(thrust, 0.08, 0.65)),
        })

    def _timed(self, duration, command, start=None, mode=None, filename=None):
        state = self._state()
        start = state["timestamp"] if start is None else start
        deadline = self.clock() + 300.0
        next_index = 0
        next_status = start
        times = []
        while state["timestamp"] - start < duration:
            if self.clock() > deadline:
                raise AcquisitionError("Phase exceeded five wall-clock minutes")
            self._check_bounds(state)
            elapsed = state["timestamp"] - start
            index = int(math.floor(max(0.0, elapsed) * self.config.command_hz + 1e-8))
            if index >= next_index:
                command(elapsed, state)
                times.append(float(state["timestamp"]))
                next_index = index + 1  # Never burst-send missed commands.
            if state["timestamp"] >= next_status:
                recording = self._recording()["recording"]
                if filename is not None and (recording["current_file"] != filename or recording["mode"] != mode):
                    raise AcquisitionError("Recording segment/mode changed unexpectedly; stop other control clients")
                next_status = state["timestamp"] + 1.0
            self.sleep(0.002)
            state = self._state()
        return times

    def _wait_mode(self, mode):
        deadline = self.clock() + 5
        while self.clock() < deadline:
            self._check_bounds(self._state())
            status = self._recording()["recording"]
            if status.get("mode") == mode and status.get("current_file"):
                if not isinstance(status["current_file"], str):
                    raise AcquisitionError("Invalid recording filename")
                finite_number(status["segment_start"])
                return status
            self.sleep(0.005)
        raise AcquisitionError(f"Simulator did not enter {mode} mode")

    def _settle(self):
        state = self._state()
        start = state["timestamp"]
        stable_since = None
        deadline = self.clock() + 180
        while state["timestamp"] - start < 20 and self.clock() < deadline:
            self._check_bounds(state)
            self._position()
            p, v, a = state["position"], state["velocity"], state["orientation"]
            stable = (
                abs(p["z"] - self.config.height_m) < 0.08
                and math.hypot(p["x"] - self.origin[0], p["y"] - self.origin[1]) < 0.15
                and max(abs(v[axis]) for axis in ("x", "y", "z")) < 0.15
                and max(abs(a["roll"]), abs(a["pitch"])) < 3
                and abs(wrap_degrees(a["yaw"] - self.origin[2])) < 3
            )
            if stable:
                stable_since = state["timestamp"] if stable_since is None else stable_since
                if state["timestamp"] - stable_since >= 1:
                    return
            else:
                stable_since = None
            self.sleep(0.02)
            self._recording()
            state = self._state()
        raise AcquisitionError("Vehicle did not reach a stable hover")

    def _excite(self, trial, phase_index):
        name, mode, targets = PHASES[phase_index]
        warmup, tail = 2.0, 1.0
        total = warmup + self.config.duration_s + tail
        signal = excitation(total, self.config.command_hz, self.config.seed + trial * 100 + phase_index)
        state = self._state()
        if mode == "position":
            self._position()
        else:
            self._attitude(state)
        recording = self._wait_mode(mode)
        start = self._state()["timestamp"]
        logger.info("Trial %d: %s excitation for %.1f simulation seconds", trial + 1, name, self.config.duration_s)

        def command(elapsed, current):
            index = min(len(signal) - 1, int(max(0, elapsed) * self.config.command_hz + 1e-8))
            ramp_in = min(1.0, max(0.0, elapsed - (warmup - 1)))
            ramp_out = min(1.0, max(0.0, (total - elapsed) / tail))
            value = float(signal[index] * min(ramp_in, ramp_out))
            if mode == "position":
                self._position(yaw=self.origin[2] + value * self.config.yaw_angle_amplitude_deg)
            else:
                self._attitude(current, name, value)

        times = self._timed(total, command, start, mode, recording["current_file"])
        begin, end = start + warmup, start + warmup + self.config.duration_s
        return {
            "trial": trial + 1, "phase": name, "targets": targets,
            "file": recording["current_file"],
            "interval": [begin - recording["segment_start"], end - recording["segment_start"]],
            "command_times": [t for t in times if begin <= t <= end],
        }

    def _land(self):
        self.airborne = False
        z = self._state()["position"]["z"]
        duration = max(2.0, z / 0.2)
        # The position controller's minimum thrust can be close to hover thrust.
        # Use the same bounded altitude feedback as the excitation phases to descend.
        self._timed(duration, lambda elapsed, state: self._attitude(state, height=max(0.03, z - 0.2 * elapsed)))
        self._timed(3.0, lambda elapsed, state: self._attitude(state, height=0.03))
        state = self._state()
        if state["position"]["z"] > 0.15 or abs(state["velocity"]["z"]) > 0.2:
            raise AcquisitionError("Landing was not confirmed")
        if not self._stop():
            raise AcquisitionError("Simulator did not accept the stop command")
        deadline = self.clock() + 3
        while self.clock() < deadline:
            self._state()
            debug = self._json("/controller/debug")
            try:
                thrust = [finite_number(v) for v in debug["thrust_pwm"]]
                motors = [finite_number(v) for v in debug["motor_pwm"][0]]
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                raise AcquisitionError("Cannot verify stopped simulator motors") from exc
            if len(thrust) == 1 and len(motors) == 4 and all(v == 0 for v in thrust + motors):
                self.motion_started = False
                return
            self.sleep(0.01)
        raise AcquisitionError("Simulator motors did not stop")

    def _stop(self):
        if self.verified and self.motion_started:
            try:
                self._json("/control/attitude", {"roll": 0, "pitch": 0, "yaw_rate": 0, "thrust": 0})
                return True
            except Exception:
                logger.exception("Could not send simulator stop command")
        return False

    def _retrieve(self, episode):
        name = episode["file"]
        if not isinstance(name, str) or not re.fullmatch(r"flight_[A-Za-z0-9_.-]+\.csv", name):
            raise AcquisitionError("Invalid recording filename")
        deadline = self.clock() + 5
        while name not in self._recording()["recording"]["completed"]:
            if self.clock() >= deadline:
                raise AcquisitionError(f"Recording was not finalized: {name}")
            self.sleep(0.02)
        raw = self.output / "raw" / name
        if not raw.exists():
            response = self.http.get(self.base_url + "/recording/file", params={"name": name}, timeout=10)
            response.raise_for_status()
            partial = raw.with_suffix(".csv.partial")
            with partial.open("xb") as stream:
                stream.write(response.content)
            os.link(partial, raw)
            partial.unlink()
        data = load_run(raw)
        lo, hi = episode["interval"]
        selected = data.samples.loc[(data.samples.t_s >= lo) & (data.samples.t_s <= hi)]
        if selected.empty:
            raise AcquisitionError(f"No recorded samples cover {episode['phase']}")
        output = self.output / f"trial_{episode['trial']:03d}_{episode['phase']}.csv"
        partial = output.with_suffix(".csv.partial")
        with partial.open("x", newline="") as stream, np.errstate(invalid="ignore"):
            selected.to_csv(stream, index=False)  # Crop only; preserve values and recorded timestamps.
        os.link(partial, output)
        partial.unlink()
        quality, analysis = assess_quality(
            output, episode["targets"], self.expected_hz, self.config.max_frequency_hz,
            self.config.duration_s, episode["command_times"], self.config.command_hz,
        )
        write_report(analysis, self.output / "reports" / output.stem)
        self.results.append({
            "file": output.name, "trial": episode["trial"], "phase": episode["phase"], **quality,
        })
        logger.info("%s: quality %s", output.name, "PASS" if quality["passed"] else "FAIL")

    def run(self):
        failure = None
        try:
            status = self._recording()
            self.verified = True
            self.recording_session = status["recording"]["session"]
            dt = finite_number(status["dt_s"])
            if dt <= 0:
                raise AcquisitionError("Invalid simulator time step")
            self.expected_hz = 1 / dt
            if not math.isfinite(self.expected_hz) or self.config.command_hz > self.expected_hz * 1.001:
                raise AcquisitionError("Command rate exceeds the simulator control rate")
            if self.expected_hz < 4 * self.config.max_frequency_hz:
                raise AcquisitionError("Simulator sampling is too slow for the requested frequency band")
            params = self._json("/controller/params")
            self.mass = finite_number(params["mass"])
            self.motor_thrust_max = finite_number(params["thrust_max"])
            if not (0.01 <= self.mass <= 0.1 and 0.05 <= self.motor_thrust_max <= 1.0):
                raise AcquisitionError("Unexpected simulator mass or thrust scale")
            hover_fraction = self.mass * 9.81 / (4 * self.motor_thrust_max)
            if not 0.12 <= hover_fraction <= 0.5:
                raise AcquisitionError("Insufficient thrust margin for the bounded automatic flight profile")
            state = self._state()
            self._require_grounded(state)
            initial_time = state["timestamp"]
            while state["timestamp"] == initial_time:
                self.sleep(0.002)
                state = self._state()
            self.output = self.log_dir / f"auto_{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}"
            self.output.mkdir(parents=True, exist_ok=False)
            (self.output / "raw").mkdir()

            for trial in range(self.config.repeats):
                self._require_grounded(self._state())
                self._json("/reset", {})
                self.last_sim_time = None
                self.last_progress_wall = self.clock()
                state = self._state()
                self.origin = (state["position"]["x"], state["position"]["y"], state["orientation"]["yaw"])
                self.motion_started = True
                z0 = state["position"]["z"]
                self._timed(4.0, lambda elapsed, current: self._position(
                    z=z0 + (self.config.height_m - z0) * min(1.0, elapsed / 4.0)
                ))
                self._settle()
                self.airborne = True
                episodes = []
                for index in range(len(PHASES)):
                    episodes.append(self._excite(trial, index))
                    self._position()
                    self._settle()
                self._land()
                # File I/O and analysis happen on the ground, not between flight commands.
                for episode in episodes:
                    self._retrieve(episode)
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._stop()
            self.http.close()
            if self.output is not None:
                result = {
                    "passed": failure is None and bool(self.results) and all(r["passed"] for r in self.results),
                    "acquisition_error": failure, "config": asdict(self.config),
                    "analysis_config": asdict(ANALYSIS), "results": self.results,
                }
                with (self.output / "quality.json").open("x") as stream:
                    json.dump(result, stream, indent=2, allow_nan=False)
                    stream.write("\n")
        return bool(self.results) and all(result["passed"] for result in self.results)


def _terminate(signum, frame):
    raise KeyboardInterrupt("termination requested")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Simulator-only automatic recording with post-flight quality gates")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--duration", type=float, default=32, help="Excitation seconds per phase (simulation time)")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--command-hz", type=float, default=100)
    parser.add_argument("--height", type=float, default=1)
    parser.add_argument("--angle-amplitude", type=float, default=2, help="Roll/Pitch excitation amplitude (deg)")
    parser.add_argument("--yaw-amplitude", type=float, default=5, help="Yaw angle excitation amplitude (deg)")
    parser.add_argument("--yaw-rate-amplitude", type=float, default=10, help="Yaw rate excitation amplitude (deg/s)")
    parser.add_argument("--max-frequency", type=float, default=15, help="Upper frequency required by the quality gate")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        config = AutoConfig(
            duration_s=args.duration, repeats=args.repeats, command_hz=args.command_hz,
            height_m=args.height, max_frequency_hz=args.max_frequency, seed=args.seed,
            angle_amplitude_deg=args.angle_amplitude, yaw_angle_amplitude_deg=args.yaw_amplitude,
            yaw_rate_amplitude_dps=args.yaw_rate_amplitude,
        )
        args.log_dir.mkdir(parents=True, exist_ok=True)
        handlers = [
            logging.FileHandler(args.log_dir / "auto_client.log", mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ]
        for handler in handlers:
            handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        try:
            signal.signal(signal.SIGTERM, _terminate)
            client = AutoClient(args.host, args.port, config, args.log_dir)
            passed = client.run()
            print(f"Results: {client.output}\nQuality: {'PASS' if passed else 'FAIL'}")
            return 0 if passed else 3
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
            for handler in handlers:
                logger.removeHandler(handler)
                handler.close()
    except KeyboardInterrupt:
        print("Automatic flight interrupted", file=sys.stderr)
        return 2
    except (AcquisitionError, requests.RequestException, OSError, ValueError, KeyError) as exc:
        print(f"Automatic acquisition failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
