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
from crazyflie_sim.flight_profiles import (
    Reference, Trajectory, QuinticTransition, TRAJECTORIES, tracking_command,
    predicted_clearance, validate_reference,
)
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


class CollisionError(AcquisitionError):
    pass


@dataclass(frozen=True)
class AutoConfig:
    mode: str = "both"
    duration_s: float = 32.0
    repeats: int = 1
    command_hz: float = 100.0
    height_m: float = 1.0
    angle_amplitude_deg: float = 2.0
    yaw_angle_amplitude_deg: float = 5.0
    yaw_rate_amplitude_dps: float = 10.0
    max_frequency_hz: float = 15.0
    seed: int = 0
    phase_timeout_s: float = 900.0
    trajectory_duration_s: float = 48.0
    agile_height_m: float = 4.0

    def __post_init__(self):
        if self.mode not in ("identification", "agile", "both"):
            raise ValueError("mode must be identification, agile or both")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for key, value in asdict(self).items() if key != "mode"):
            raise ValueError("Auto-client settings must be finite numbers")
        if not isinstance(self.repeats, int) or self.repeats < 1:
            raise ValueError("repeats must be a positive integer")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.phase_timeout_s <= 0:
            raise ValueError("phase_timeout_s must be positive")
        if self.trajectory_duration_s < 24 or self.agile_height_m < 1.5:
            raise ValueError("Trajectories require at least 24 seconds and agile height >= 1.5 m")
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


AGILE_COLUMNS = (
    "angle_meas_roll_deg", "angle_meas_pitch_deg",
    "rate_meas_x_dps", "rate_meas_y_dps", "rate_meas_z_dps",
)


def agile_coverage(frames):
    """Evaluate measured coverage over the complete agile suite, not each path."""
    if not frames:
        return {"passed": False, "reasons": ["no_agile_samples"], "channels": {}}
    samples = pd.concat(frames, ignore_index=True)
    channels, reasons = {}, []
    thresholds = {"angle_meas_roll_deg": 15.0, "angle_meas_pitch_deg": 15.0, "rate_meas_z_dps": 60.0}
    for column in AGILE_COLUMNS:
        values = samples[column].dropna().to_numpy(float)
        if not len(values) or not np.isfinite(values).all():
            reasons.append(f"missing_or_invalid_measurement:{column}")
            continue
        p95 = float(np.percentile(np.abs(values), 95))
        channels[column] = {"p95_abs": p95, "peak_abs": float(np.max(np.abs(values))),
                            "rms": float(np.sqrt(np.mean(values**2))),
                            "required_p95": thresholds.get(column)}
        if column in thresholds and p95 < thresholds[column]:
            reasons.append(f"insufficient_coverage:{column}")
    return {"passed": not reasons, "reasons": reasons, "channels": channels}


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
        self.target_height = self.config.height_m
        self.episodes = []
        self.active_episode = None
        self.events = []
        self.cleanup_errors = []
        self.preflight = []
        self.agile_frames = []
        self.last_state = None
        self.acceleration = np.zeros(3)
        self.last_reference = None
        self.contact_sequence = 0
        self.landing = False
        self.taking_off = False
        self.recovering = False
        self.ground_confirmed = True
        self.crashed = False
        self.cleaning_up = False
        self.completed_trials = 0

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
        if self.cleaning_up and isinstance(recording, dict):
            # A failed recorder must not prevent landing in the verified simulator.
            return status
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
            contact = state["ground_contact"]
            if contact.get("supported") is not True or not isinstance(contact.get("active"), bool):
                raise ValueError("contact telemetry is unavailable")
            if isinstance(contact.get("sequence"), bool) or not isinstance(contact.get("sequence"), int) or contact["sequence"] < 0:
                raise ValueError("invalid contact sequence")
            if contact["sequence"] and not isinstance(contact.get("last_event"), dict):
                raise ValueError("missing latched contact event")
        except (KeyError, TypeError, ValueError) as exc:
            raise AcquisitionError(f"Invalid simulator telemetry: {exc}") from exc
        now = self.clock()
        if self.last_sim_time is not None and timestamp < self.last_sim_time:
            self.verified = False
            raise AcquisitionError("Simulation clock moved backwards unexpectedly")
        if self.last_sim_time is None or timestamp > self.last_sim_time:
            self.last_progress_wall = now
        elif now - self.last_progress_wall > 5.0:
            raise AcquisitionError("Simulation clock stalled for five wall-clock seconds")
        self.last_sim_time = timestamp
        if self.last_state is not None and timestamp > self.last_state["timestamp"]:
            delta = timestamp - self.last_state["timestamp"]
            measured = np.array([(state["velocity"][a] - self.last_state["velocity"][a]) / delta for a in "xyz"])
            alpha = 1 - math.exp(-delta / 0.1)
            self.acceleration += alpha * (measured - self.acceleration)
        self.last_state = state
        return state

    def _check_bounds(self, state):
        contact = state["ground_contact"]
        if contact["sequence"] < self.contact_sequence:
            self.verified = False
            raise AcquisitionError("Contact sequence moved backwards; simulator may have changed")
        if contact["sequence"] > self.contact_sequence:
            self.contact_sequence = contact["sequence"]
            if self.airborne:
                event = contact["last_event"]
                before = event.get("before", {})
                try:
                    soft_landing = (
                        self.landing
                        and finite_number(event["position"]["z"]) <= 0.15
                        and abs(finite_number(before["velocity"]["z"])) <= 0.35
                        and max(abs(finite_number(before["orientation"][a])) for a in ("roll", "pitch")) < 10
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise AcquisitionError("Invalid latched contact state") from exc
                if not soft_landing:
                    self.crashed = True
                    self.events.append({"type": "collision", "contact": event})
                    if self.active_episode is not None:
                        self.active_episode["collision_time"] = finite_number(event["timestamp"])
                    # Reset first; retrieval and analysis must never delay the reset.
                    try:
                        self._json("/reset", {})
                        self.motion_started = self.airborne = False
                        self.ground_confirmed = True
                        self.events.append({"type": "collision_reset", "timestamp": state["timestamp"]})
                    except Exception as exc:
                        self.cleanup_errors.append(f"collision reset failed: {exc}")
                    raise CollisionError("Unexpected ground contact; acquisition terminated")
                self.ground_confirmed = True
        if self.motion_started and state["position"]["z"] > 0.3 and not self.landing:
            self.airborne = True
            self.ground_confirmed = False

    @staticmethod
    def _require_grounded(state):
        if (
            not -0.05 <= state["position"]["z"] <= 0.3
            or max(abs(v) for v in state["velocity"].values()) > 0.2
        ):
            raise AcquisitionError("Start with a stationary grounded simulator; stop other control clients")

    def _position(self, z=None, yaw=None):
        self._json("/control/position", {
            "x": self.origin[0], "y": self.origin[1],
            "z": self.target_height if z is None else z,
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
        height = self.target_height if height is None else height
        acceleration = 4.0 * (height - p["z"]) - 3.0 * v["z"]
        tilt = max(1e-6, math.cos(math.radians(a["roll"])) * math.cos(math.radians(a["pitch"])))
        thrust = self.mass * (9.81 + acceleration) / (4 * self.motor_thrust_max * tilt)
        self._json("/control/attitude", {
            "roll": float(np.clip(roll / 30.0, -1, 1)), "pitch": float(np.clip(pitch / 30.0, -1, 1)),
            "yaw_rate": float(np.clip(yaw_rate / 120.0, -1, 1)), "thrust": float(np.clip(thrust, 0, 1)),
        })

    def _follow(self, reference, state):
        self._json("/control/attitude", tracking_command(reference, state, self.mass, self.motor_thrust_max))
        self.last_reference = reference

    def _state_reference(self, state):
        return Reference(
            np.array([state["position"][a] for a in "xyz"]),
            np.array([state["velocity"][a] for a in "xyz"]), self.acceleration.copy(),
            state["orientation"]["yaw"], state["angular_velocity"]["z"],
        )

    def _transition(self, end, start=None, minimum_duration=4.0):
        state = self._state()
        self._check_bounds(state)
        start = self._state_reference(state) if start is None else start
        duration = max(minimum_duration, 2 * float(np.linalg.norm(end.position - start.position)) / 1.5)
        path = QuinticTransition(start, end, duration)
        validate_reference(path, self.mass, self.motor_thrust_max, minimum_height=0.03 if not self.airborne else 0.5)
        self._timed(duration, lambda elapsed, current: self._follow(path.sample(elapsed), current),
                    reference=path.sample)
        self.last_reference = end
        return path

    def _avoid_ground(self, resume):
        start = self._state()["timestamp"]
        event = {"type": "ground_recovery", "start": start, "end": None}
        self.events.append(event)
        self.recovering = True
        if self.active_episode is not None:
            self.active_episode["intervals"][-1][1] = start
            self.active_episode["nominal_open"] = False
        try:
            state = self._state()
            target = Reference.hover((*[state["position"][a] for a in "xy"], self.target_height), self.origin[2])
            stable_since = None
            deadline = self.clock() + self.config.phase_timeout_s
            while self.clock() < deadline:
                self._check_bounds(state)
                if state["velocity"]["z"] < 0 or state["position"]["z"] < self.target_height - 0.2:
                    a = 4 * (self.target_height - state["position"]["z"]) - 3 * state["velocity"]["z"]
                    tilt = math.cos(math.radians(state["orientation"]["roll"])) * math.cos(math.radians(state["orientation"]["pitch"]))
                    thrust = self.mass * max(9.81, 9.81 + a) / (4 * self.motor_thrust_max * max(1e-6, tilt))
                    self._json("/control/attitude", {"roll": 0, "pitch": 0, "yaw_rate": 0, "thrust": float(np.clip(thrust, 0, 1))})
                    stable_since = None
                else:
                    self._follow(target, state)
                    stable = (abs(state["position"]["z"] - self.target_height) < 0.15
                              and max(abs(v) for v in state["velocity"].values()) < 0.2
                              and max(abs(state["orientation"][a]) for a in ("roll", "pitch")) < 5)
                    stable_since = (state["timestamp"] if stable_since is None else stable_since) if stable else None
                    if stable_since is not None and state["timestamp"] - stable_since >= 1:
                        break
                self._recording()
                self.sleep(0.002)
                state = self._state()
            else:
                raise AcquisitionError("Ground recovery timed out")
            self._transition(resume)
        finally:
            self.recovering = False
            event["end"] = self.last_sim_time
            if self.active_episode is not None:
                self.active_episode["corrections"].append(dict(event))
        if self.active_episode is not None:
            self.active_episode["intervals"].append([self.last_sim_time, self.last_sim_time])
            self.active_episode["nominal_open"] = True

    def _timed(self, duration, command, start=None, mode=None, filename=None, reference=None):
        state = self._state()
        start = state["timestamp"] if start is None else start
        deadline = self.clock() + self.config.phase_timeout_s
        next_index = 0
        next_status = start
        times = []
        paused = 0.0
        while state["timestamp"] - start - paused < duration:
            if self.clock() > deadline:
                raise AcquisitionError(
                    f"Phase exceeded {self.config.phase_timeout_s:g} wall-clock seconds "
                    f"({state['timestamp'] - start:.3f}/{duration:g} simulation seconds completed)"
                )
            self._check_bounds(state)
            elapsed = state["timestamp"] - start - paused
            if self.airborne and not self.landing and not self.recovering and (
                not self.taking_off or state["velocity"]["z"] < -0.1
            ) and predicted_clearance(
                state, self.mass, self.motor_thrust_max
            ) < 0.5:
                before = state["timestamp"]
                resume = reference(elapsed) if reference is not None else Reference.hover(
                    (*self.origin[:2], self.target_height), self.origin[2])
                self._avoid_ground(resume)
                if self.taking_off:
                    # Recovery already reached the takeoff height; do not command
                    # the lower, interrupted ramp target again.
                    return times
                if reference is None and self.active_episode is not None:
                    raise AcquisitionError("Identification interrupted by ground recovery")
                state = self._state()
                paused += state["timestamp"] - before
                continue
            index = int(math.floor(max(0.0, state["timestamp"] - start) * self.config.command_hz + 1e-8))
            if index >= next_index:
                command(elapsed, state)
                times.append(float(state["timestamp"]))
                if self.active_episode is not None and not self.recovering:
                    self.active_episode["command_times"].append(float(state["timestamp"]))
                next_index = index + 1  # Never burst-send missed commands.
            if state["timestamp"] >= next_status:
                recording = self._recording()["recording"]
                if filename is not None and (recording["current_file"] != filename or recording["mode"] != mode):
                    raise AcquisitionError("Recording segment/mode changed unexpectedly; stop other control clients")
                next_status = state["timestamp"] + 1.0
            self.sleep(0.002)
            state = self._state()
            if self.active_episode is not None and not self.recovering and self.active_episode["nominal_open"]:
                self.active_episode["intervals"][-1][1] = state["timestamp"]
        self._check_bounds(state)
        return times

    def _wait_mode(self, mode, previous_file=None):
        deadline = self.clock() + 5
        while self.clock() < deadline:
            self._check_bounds(self._state())
            status = self._recording()["recording"]
            if status.get("mode") == mode and status.get("current_file") and status["current_file"] != previous_file:
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
        deadline = self.clock() + self.config.phase_timeout_s
        while state["timestamp"] - start < 20 and self.clock() < deadline:
            self._check_bounds(state)
            if self.airborne and not self.landing and predicted_clearance(state, self.mass, self.motor_thrust_max) < 0.5:
                before = state["timestamp"]
                self._avoid_ground(Reference.hover((*self.origin[:2], self.target_height), self.origin[2]))
                state = self._state()
                start += state["timestamp"] - before
                stable_since = None
                continue
            self._position()
            p, v, a = state["position"], state["velocity"], state["orientation"]
            stable = (
                abs(p["z"] - self.target_height) < 0.08
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
        if self.clock() >= deadline and state["timestamp"] - start < 20:
            raise AcquisitionError(
                f"Hover check exceeded {self.config.phase_timeout_s:g} wall-clock seconds "
                f"({state['timestamp'] - start:.3f}/20 simulation seconds completed)"
            )
        raise AcquisitionError("Vehicle did not reach a stable hover")

    def _recover(self):
        target = Reference.hover((*self.origin[:2], self.target_height), self.origin[2])
        self._transition(target, start=self.last_reference)
        self._settle()
        self.last_reference = target

    def _begin_episode(self, trial, name, profile, mode, expected_duration, targets=(), metadata=None):
        previous = self._json("/recording/split", {})["previous_file"]
        recording = self._wait_mode(mode, previous)
        start = self._state()["timestamp"]
        episode = {
            "trial": trial + 1, "phase": name, "profile": profile, "mode": mode, "targets": targets,
            "file": recording["current_file"], "segment_start": recording["segment_start"],
            "start": start, "intervals": [[start, start]], "expected_duration": expected_duration,
            "command_times": [], "corrections": [], "completed": False, "metadata": metadata or {},
            "nominal_open": True,
        }
        self.episodes.append(episode)
        self.active_episode = episode
        return episode

    def _end_episode(self, episode, completed):
        if self.last_sim_time is not None and episode["nominal_open"]:
            end = min(self.last_sim_time, episode.get("collision_time", self.last_sim_time))
            episode["intervals"][-1][1] = max(episode["intervals"][-1][0], end)
        episode["completed"] = completed
        episode["nominal_open"] = False
        self.active_episode = None
        if self.verified:
            try:
                self._json("/recording/split", {})
            except Exception as exc:
                episode["seal_error"] = str(exc)
                if completed:
                    raise

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
        self._wait_mode(mode)
        episode = self._begin_episode(trial, name, "identification", mode, self.config.duration_s, targets)
        start = episode["start"]
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

        complete = False
        try:
            self._timed(total, command, start, mode, episode["file"])
            complete = True
        finally:
            self._end_episode(episode, complete)
            if complete:
                episode["intervals"] = [[start + warmup, start + warmup + self.config.duration_s]]
            self.last_reference = None
        return episode

    def _trajectory(self, trial, path):
        self._transition(path.sample(0), start=self.last_reference)
        self._wait_mode("attitude")
        episode = self._begin_episode(trial, path.name, "agile", "attitude", path.duration, metadata=path.metadata())
        logger.info("Trial %d: %s trajectory for %.1f simulation seconds", trial + 1, path.name, path.duration)
        complete = False
        try:
            self._timed(path.duration, lambda elapsed, state: self._follow(path.sample(elapsed), state),
                        episode["start"], "attitude", episode["file"], reference=path.sample)
            complete = True
            self.last_reference = path.sample(path.duration)
        finally:
            self._end_episode(episode, complete)
        return episode

    def _land(self):
        self.landing = True
        z = self._state()["position"]["z"]
        duration = max(2.0, z / 0.2)
        # The position controller's minimum thrust can be close to hover thrust.
        # Use the same bounded altitude feedback as the excitation phases to descend.
        self._timed(duration, lambda elapsed, state: self._attitude(state, height=max(0.0, z - 0.2 * elapsed)))
        self._timed(3.0, lambda elapsed, state: self._attitude(state, height=0.0))
        state = self._state()
        if (state["position"]["z"] > 0.15 or abs(state["velocity"]["z"]) > 0.2
                or not state["ground_contact"]["active"]):
            raise AcquisitionError("Landing was not confirmed")
        self.ground_confirmed = True
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
                self.airborne = self.landing = False
                return
            self.sleep(0.01)
        raise AcquisitionError("Simulator motors did not stop")

    def _stop(self):
        if self.verified and self.motion_started and (self.ground_confirmed or self.crashed):
            try:
                self._json("/control/attitude", {"roll": 0, "pitch": 0, "yaw_rate": 0, "thrust": 0})
                return True
            except Exception:
                logger.exception("Could not send simulator stop command")
        return False

    def _download(self, name):
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
        return load_run(raw)

    def _retrieve(self, episode):
        data = self._download(episode["file"])
        intervals = [(lo - episode["segment_start"], hi - episode["segment_start"])
                     for lo, hi in episode["intervals"] if hi > lo]
        mask = np.zeros(len(data.samples), dtype=bool)
        chunks, command_times = [], []
        reasons = [] if episode["completed"] else ["incomplete_episode"]
        for (lo, hi), (absolute_lo, absolute_hi) in zip(
            intervals, [pair for pair in episode["intervals"] if pair[1] > pair[0]]
        ):
            current = (data.samples.t_s >= lo) & (data.samples.t_s <= hi)
            mask |= current
            chunk = data.samples.loc[current]
            chunks.append(chunk)
            times = [t for t in episode["command_times"] if absolute_lo <= t <= absolute_hi]
            command_times.extend(times)
            if len(times) / max(1, (hi - lo) * self.config.command_hz) < 0.9 or (
                len(times) > 1 and max(np.diff(times)) > 3 / self.config.command_hz + 1e-8
            ):
                reasons.append("command_cadence_insufficient")
            if len(chunk) < 2 or chunk.t_s.iloc[-1] - chunk.t_s.iloc[0] < hi - lo - 3 / self.expected_hz:
                reasons.append("recording_does_not_cover_interval")
            elif np.max(np.diff(chunk.t_s)) > 3 / self.expected_hz + 1e-8:
                reasons.append("recording_has_long_gaps")
            if len(chunk) < 0.95 * (hi - lo) * self.expected_hz:
                reasons.append("recording_rate_below_simulator_rate")
        selected = data.samples.loc[mask]
        if selected.empty:
            raise AcquisitionError(f"No recorded samples cover {episode['phase']}")
        prefix = "agile_" if episode["profile"] == "agile" else ""
        output = self.output / f"trial_{episode['trial']:03d}_{prefix}{episode['phase']}.csv"
        partial = output.with_suffix(".csv.partial")
        with partial.open("x", newline="") as stream, np.errstate(invalid="ignore"):
            selected.to_csv(stream, index=False)  # Crop only; preserve values and recorded timestamps.
        os.link(partial, output)
        partial.unlink()
        duration = sum(hi - lo for lo, hi in intervals)
        if duration < episode["expected_duration"] - 3 / self.expected_hz:
            reasons.append("insufficient_nominal_duration")
        quality = {"passed": not reasons, "reasons": sorted(set(reasons)), "duration_s": duration}
        if episode["profile"] == "identification" and episode["completed"]:
            identification, analysis = assess_quality(
                output, episode["targets"], self.expected_hz, self.config.max_frequency_hz,
                self.config.duration_s, command_times, self.config.command_hz,
            )
            write_report(analysis, self.output / "reports" / output.stem)
            quality = {**identification, "passed": not reasons and identification["passed"],
                       "reasons": sorted(set(reasons + identification["reasons"]))}
        elif episode["profile"] == "agile":
            for column in AGILE_COLUMNS:
                ref = column.replace("_meas_", "_ref_")
                if selected[[column, ref]].isna().any().any():
                    quality["reasons"].append(f"missing_paired_channel:{column}")
                    quality["passed"] = False
            self.agile_frames.append(selected)
            quality["channels"] = agile_coverage([selected])["channels"]
        self.results.append({
            "file": output.name, "trial": episode["trial"], "phase": episode["phase"],
            "profile": episode["profile"], "completed": episode["completed"], **quality,
        })
        episode["retrieved"] = True
        episode["output_file"] = output.name
        logger.info("%s: quality %s", output.name, "PASS" if quality["passed"] else "FAIL")

    def _paths(self, trial):
        return [Trajectory(name, self.config.trajectory_duration_s, self.config.agile_height_m,
                           self.config.seed + trial * 100 + index, self.origin or (0, 0, 0))
                for index, name in enumerate(TRAJECTORIES)]

    def _preflight(self):
        ground = Reference.hover((0, 0, 0.1), 0)
        hover = Reference.hover((0, 0, self.config.height_m), 0)
        takeoff = QuinticTransition(ground, hover, 4)
        self.preflight.append({"phase": "takeoff", **validate_reference(
            takeoff, self.mass, self.motor_thrust_max, minimum_height=0.03)})
        if self.config.mode != "identification":
            high = Reference.hover((0, 0, self.config.agile_height_m), 0)
            climb = QuinticTransition(hover, high, max(4, 2 * abs(self.config.agile_height_m - self.config.height_m) / 1.5))
            self.preflight.append({"phase": "climb", **validate_reference(climb, self.mass, self.motor_thrust_max)})
            for trial in range(self.config.repeats):
                for path in self._paths(trial):
                    self.preflight.append({"trial": trial + 1, "phase": path.name, **validate_reference(
                        path, self.mass, self.motor_thrust_max)})
                    endpoint = path.sample(path.duration)
                    duration = max(4, 2 * float(np.linalg.norm(high.position - endpoint.position)) / 1.5)
                    transition = QuinticTransition(endpoint, high, duration)
                    self.preflight.append({"trial": trial + 1, "phase": path.name + "_return",
                                           **validate_reference(transition, self.mass, self.motor_thrust_max)})

    def _safe_finish(self):
        if not self.verified or not self.motion_started:
            return
        if self.crashed:
            self._stop()
            return
        try:
            self._check_bounds(self._state())
            self._land()
        except (Exception, KeyboardInterrupt) as exc:
            self.cleanup_errors.append(f"controlled landing failed: {exc}")
            # Never intentionally cut power to an unconfirmed airborne vehicle.
            if self.ground_confirmed:
                self._stop()
            elif self.verified and self.last_state is not None and not self.crashed:
                try:
                    self._json("/control/position", {
                        **{a: self.last_state["position"][a] for a in "xy"},
                        "z": max(self.target_height, self.last_state["position"]["z"]),
                        "yaw": self.last_state["orientation"]["yaw"],
                    })
                except Exception as hold_error:
                    self.cleanup_errors.append(f"hover handoff failed: {hold_error}")

    def run(self):
        failure = error = traceback = None
        try:
            status = self._recording()
            if not {"contact_events", "recording_split"}.issubset(status.get("capabilities", [])):
                raise AcquisitionError("Simulator must support contact events and recording split")
            self.verified = True
            self.recording_session = status["recording"]["session"]
            dt = finite_number(status["dt_s"])
            if dt <= 0:
                raise AcquisitionError("Invalid simulator time step")
            self.expected_hz = 1 / dt
            if not math.isfinite(self.expected_hz) or self.config.command_hz > self.expected_hz * 1.001:
                raise AcquisitionError("Command rate exceeds the simulator control rate")
            if self.config.mode != "agile" and self.expected_hz < 4 * self.config.max_frequency_hz:
                raise AcquisitionError("Simulator sampling is too slow for the requested frequency band")
            params = self._json("/controller/params")
            self.mass = finite_number(params.get("total_mass", params["mass"]))
            self.motor_thrust_max = finite_number(params["thrust_max"])
            if not (0.01 <= self.mass <= 0.1 and 0.05 <= self.motor_thrust_max <= 1.0):
                raise AcquisitionError("Unexpected simulator mass or thrust scale")
            try:
                self._preflight()
            except ValueError as exc:
                raise AcquisitionError(f"Preflight failed: {exc}") from exc
            state = self._state()
            self._require_grounded(state)
            initial_time = state["timestamp"]
            while state["timestamp"] == initial_time:
                self.sleep(0.002)
                state = self._state()
            self.output = self.log_dir / f"auto_{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}"
            self.output.mkdir(parents=True, exist_ok=False)
            (self.output / "raw").mkdir()
            self.controller_params = params

            for trial in range(self.config.repeats):
                self._require_grounded(self._state())
                self._json("/reset", {})
                self.last_sim_time = None
                self.last_progress_wall = self.clock()
                state = self._state()
                self.contact_sequence = state["ground_contact"]["sequence"]
                self.origin = (state["position"]["x"], state["position"]["y"], state["orientation"]["yaw"])
                self.target_height = self.config.height_m
                self.landing = self.airborne = False
                self.ground_confirmed = False
                self.acceleration = np.zeros(3)
                self.last_reference = None
                self.motion_started = True
                takeoff = QuinticTransition(
                    Reference.hover((*self.origin[:2], state["position"]["z"]), self.origin[2]),
                    Reference.hover((*self.origin[:2], self.target_height), self.origin[2]), 4.0,
                )
                self.taking_off = True
                try:
                    self._timed(4.0, lambda elapsed, current: self._position(z=takeoff.sample(elapsed).position[2]))
                finally:
                    self.taking_off = False
                self._settle()
                self.airborne = True
                if self.config.mode != "agile":
                    for index in range(len(PHASES)):
                        self._excite(trial, index)
                        self._recover()
                if self.config.mode != "identification":
                    self.target_height = self.config.agile_height_m
                    self._recover()
                    for path in self._paths(trial):
                        self._trajectory(trial, path)
                        self._recover()
                self._land()
                self._json("/recording/split", {})
                self.completed_trials += 1
                for episode in self.episodes:
                    if episode.get("retrieved"):
                        continue
                    try:
                        self._retrieve(episode)
                    except Exception as exc:
                        episode["retrieval_error"] = str(exc)
                        raise
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"
            error, traceback = exc, exc.__traceback__
        finally:
            if error is not None:
                self.cleaning_up = True
                try:
                    self._safe_finish()
                finally:
                    self.cleaning_up = False
            if self.verified and self.output is not None and (self.ground_confirmed or self.crashed):
                if error is not None:
                    try:
                        self._json("/recording/split", {})
                    except Exception as exc:
                        self.cleanup_errors.append(f"final recording seal failed: {exc}")
                for episode in self.episodes:
                    if not episode.get("retrieved") and not episode.get("retrieval_error"):
                        try:
                            self._retrieve(episode)
                        except Exception as exc:
                            episode["retrieval_error"] = str(exc)
                            self.cleanup_errors.append(f"retrieval failed: {exc}")
            self.http.close()
            if self.output is not None:
                coverage = agile_coverage(self.agile_frames) if self.config.mode != "identification" else None
                complete = failure is None and self.completed_trials == self.config.repeats and not self.cleanup_errors
                result = {
                    "passed": complete and bool(self.results) and all(r["passed"] for r in self.results)
                              and (coverage is None or coverage["passed"]),
                    "acquisition_completed": complete,
                    "identification_passed": (
                        sum(r["profile"] == "identification" for r in self.results) == 4 * self.config.repeats
                        and all(r["passed"] for r in self.results if r["profile"] == "identification")
                    ) if self.config.mode != "agile" else None,
                    "agile_coverage": coverage, "cleanup_errors": self.cleanup_errors,
                    "acquisition_error": failure, "config": asdict(self.config),
                    "analysis_config": asdict(ANALYSIS), "results": self.results,
                }
                self.passed = result["passed"]
                with (self.output / "quality.json").open("x") as stream:
                    json.dump(result, stream, indent=2, allow_nan=False)
                    stream.write("\n")
                with (self.output / "manifest.json").open("x") as stream:
                    json.dump({"config": asdict(self.config), "controller_params": self.controller_params,
                               "preflight": self.preflight, "episodes": self.episodes, "events": self.events},
                              stream, indent=2, allow_nan=False)
                    stream.write("\n")
        if error is not None:
            raise error.with_traceback(traceback)
        return self.passed


def _terminate(signum, frame):
    raise KeyboardInterrupt("termination requested")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Simulator-only automatic recording with post-flight quality gates")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--mode", choices=("identification", "agile", "both"), default="both")
    parser.add_argument("--duration", type=float, default=32, help="Identification seconds per phase (simulation time)")
    parser.add_argument("--trajectory-duration", type=float, default=48, help="Seconds per agile trajectory (simulation time)")
    parser.add_argument("--agile-height", type=float, default=4, help="Agile reference height (m)")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--command-hz", type=float, default=100)
    parser.add_argument("--height", type=float, default=1, help="Identification/takeoff height (m)")
    parser.add_argument("--angle-amplitude", type=float, default=2, help="Identification Roll/Pitch amplitude (deg)")
    parser.add_argument("--yaw-amplitude", type=float, default=5, help="Identification Yaw angle amplitude (deg)")
    parser.add_argument("--yaw-rate-amplitude", type=float, default=10, help="Identification Yaw rate amplitude (deg/s)")
    parser.add_argument("--max-frequency", type=float, default=15, help="Upper frequency required by the quality gate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--phase-timeout", type=float, default=900,
                        help="Wall-clock seconds allowed per phase/hover check; clock-stall checks remain active")
    args = parser.parse_args(argv)
    try:
        config = AutoConfig(
            mode=args.mode, trajectory_duration_s=args.trajectory_duration, agile_height_m=args.agile_height,
            duration_s=args.duration, repeats=args.repeats, command_hz=args.command_hz,
            height_m=args.height, max_frequency_hz=args.max_frequency, seed=args.seed,
            angle_amplitude_deg=args.angle_amplitude, yaw_angle_amplitude_deg=args.yaw_amplitude,
            yaw_rate_amplitude_dps=args.yaw_rate_amplitude,
            phase_timeout_s=args.phase_timeout,
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
