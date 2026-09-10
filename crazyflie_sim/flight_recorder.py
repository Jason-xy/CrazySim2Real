"""Passive, standard-library-only recording of controller samples."""

import csv
from datetime import datetime
import logging
import math
import os
from pathlib import Path
import queue
import threading
import time


DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "logs"
COLUMNS = (
    "t_s",
    "angle_ref_roll_deg", "angle_meas_roll_deg",
    "angle_ref_pitch_deg", "angle_meas_pitch_deg",
    "angle_ref_yaw_deg", "angle_meas_yaw_deg",
    "rate_ref_x_dps", "rate_meas_x_dps",
    "rate_ref_y_dps", "rate_meas_y_dps",
    "rate_ref_z_dps", "rate_meas_z_dps",
)
logger = logging.getLogger(__name__)


class FlightRecorder:
    """One producer submits rows; one worker owns all CSV file operations."""

    QUEUE_CAPACITY = 2048
    FLUSH_INTERVAL_S = 0.5
    _ANGLE_COLUMNS = {"attitude": 4, "attitude_rate": 0, "velocity": 6, "position": 6}

    def __init__(self, log_dir=DEFAULT_LOG_DIR):
        self.directory = Path(log_dir).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._session = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}"
        self._queue = queue.Queue(maxsize=self.QUEUE_CAPACITY)
        self._stop = threading.Event()
        self._failed = threading.Event()
        self._input_lock = threading.Lock()
        self._closed = False
        self._success = True
        self.error = None
        self._part = 0
        self._mode = None
        self._last_time = None
        self._t0 = None
        self._split_pending = False
        self._sealed = False
        self._completed = []
        self._thread = threading.Thread(target=self._write_loop, name="flight-recorder", daemon=True)
        self._thread.start()

    @property
    def enabled(self):
        return not self._stop.is_set()

    def status(self):
        """Read-only progress for clients waiting for completed recordings."""
        with self._input_lock:
            return {
                "enabled": self.enabled, "error": self.error, "session": self._session,
                "mode": self._mode, "segment_start": self._t0, "last_time": self._last_time,
                "current_file": self._filename(self._part, self._mode) if self._part and not self._sealed else None,
                "completed": list(self._completed),
            }

    def _filename(self, part, mode):
        return f"flight_{self._session}_{part:03d}_{mode}.csv"

    def _wake_writer(self):
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass  # A full queue already keeps the consumer awake.

    def _fail(self, message):
        """Also used by the simulator adapter for snapshot failures; no disk I/O."""
        if self.error is None:
            self.error = str(message)
        self._failed.set()
        self._stop.set()
        self._wake_writer()

    def record(self, sim_time, mode, values):
        """Submit COLUMNS[1:] values and a lowercase mode name; never wait for I/O."""
        with self._input_lock:
            if not self.enabled:
                return False
            try:
                sim_time = float(sim_time)
                angle_columns = self._ANGLE_COLUMNS[mode]
                if not math.isfinite(sim_time):
                    raise ValueError("nonfinite simulation time")
                new_part = (
                    self._split_pending or self._last_time is None
                    or mode != self._mode or sim_time < self._last_time
                )
                if not new_part and sim_time == self._last_time:
                    return False
                values = tuple(values)
                if len(values) != 12:
                    raise ValueError("expected twelve controller values")
                snapshot = tuple(
                    float(value) if i < angle_columns or i >= 6 else None
                    for i, value in enumerate(values)
                )
                if any(value is not None and not math.isfinite(value) for value in snapshot):
                    raise ValueError("nonfinite controller sample")
                if new_part:
                    self._part += 1
                    self._t0 = sim_time
                    self._mode = mode
                    self._split_pending = False
                    self._sealed = False
                row = (sim_time - self._t0, *snapshot)
                self._queue.put_nowait((self._part, mode, row))
                self._last_time = sim_time
                return True
            except queue.Full:
                self._fail("recording queue full; new samples are no longer accepted")
            except Exception as exc:
                self._fail(f"cannot capture controller sample: {exc}")
            return False

    def split(self, finalize=False):
        """Optionally seal now, without waiting for a sample, mode change or disk I/O."""
        with self._input_lock:
            self._split_pending = True
            if finalize and not self._sealed:
                try:
                    self._queue.put_nowait((None, None, None))
                    self._sealed = True
                except queue.Full:
                    self._fail("recording queue full; cannot seal segment")

    def close(self, success=True):
        """Drain accepted rows and join the worker. Repeated calls are harmless."""
        with self._input_lock:
            if not self._closed:
                self._closed = True
                self._success = bool(success)
                self._stop.set()
                self._wake_writer()
        self._thread.join()

    def _finish_file(self, stream, partial):
        try:
            stream.flush()
        finally:
            stream.close()
        if self._success and not self._failed.is_set():
            # Hard-link publication is atomic and cannot replace an existing CSV.
            os.link(partial, partial.with_suffix(""))
            partial.unlink()
            with self._input_lock:
                self._completed.append(partial.with_suffix("").name)

    def _write_loop(self):
        stream = partial = writer = None
        current_part = None
        next_flush = time.monotonic() + self.FLUSH_INTERVAL_S
        try:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    item = self._queue.get(timeout=max(0.0, next_flush - time.monotonic()))
                except queue.Empty:
                    item = None
                if item is not None:
                    part, mode, row = item
                    if row is None:
                        if stream is not None:
                            previous, stream = stream, None
                            self._finish_file(previous, partial)
                        writer = current_part = None
                        continue
                    if part != current_part:
                        if stream is not None:
                            previous, stream = stream, None
                            self._finish_file(previous, partial)
                        partial = self.directory / (self._filename(part, mode) + ".partial")
                        stream = partial.open("x", newline="", encoding="utf-8")
                        writer = csv.writer(stream)
                        writer.writerow(COLUMNS)
                        current_part = part
                    writer.writerow(row)
                if time.monotonic() >= next_flush:
                    if stream is not None:
                        stream.flush()
                    next_flush = time.monotonic() + self.FLUSH_INTERVAL_S
        except Exception as exc:
            self._fail(f"CSV write failed: {exc}")
        finally:
            if stream is not None:
                try:
                    self._finish_file(stream, partial)
                except Exception as exc:
                    self._fail(f"CSV finalization failed: {exc}")
            if self.error is not None:
                logger.error("Flight recording stopped in %s: %s", self.directory, self.error)
