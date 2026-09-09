import csv
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from crazyflie_sim.flight_recorder import COLUMNS, DEFAULT_LOG_DIR, FlightRecorder


VALUES = tuple(range(1, 13))


def wait_until(predicate, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.001)
    return bool(predicate())


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="flight-recorder-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def recorder(self, name="logs"):
        recorder = FlightRecorder(self.root / name)
        self.addCleanup(recorder.close)
        return recorder

    def test_fixed_header_and_mode_masks(self):
        from response_analysis import validate_run
        from response_analysis.protocol import CHANNELS
        self.assertEqual(COLUMNS, ("t_s", *(column for pair in CHANNELS.values() for column in pair)))
        recorder = self.recorder()
        expected = {"attitude": 4, "attitude_rate": 0, "position": 6, "velocity": 6}
        for i, mode in enumerate(expected):
            self.assertTrue(recorder.record(100 + i, mode, VALUES))
        recorder.close()
        files = sorted(recorder.directory.glob("*.csv"))
        self.assertEqual(len(files), 4)
        self.assertFalse(list(recorder.directory.glob("*.partial")))
        for filename, (mode, count) in zip(files, expected.items()):
            self.assertIn(mode, filename.name)
            row = rows(filename)[0]
            self.assertEqual(float(row["t_s"]), 0)
            for i, column in enumerate(COLUMNS[1:]):
                self.assertEqual(row[column], str(float(VALUES[i])) if i < count or i >= 6 else "")
            self.assertTrue(validate_run(filename)["valid"])

    def test_held_commands_are_recorded_each_control_step(self):
        recorder = self.recorder()
        for i in range(200):
            values = [float(i // 4), float(i)] * 6
            self.assertTrue(recorder.record(10 + i * 0.005, "position", values))
        recorder.close()
        data = rows(next(recorder.directory.glob("*.csv")))
        self.assertEqual(len(data), 200)
        for i, row in enumerate(data):
            self.assertAlmostEqual(float(row["t_s"]), i * 0.005)
            self.assertEqual(float(row["rate_ref_x_dps"]), i // 4)
            self.assertEqual(float(row["rate_meas_x_dps"]), i)

    def test_duplicate_time_reset_and_clock_reversal(self):
        recorder = self.recorder()
        recorder.record(20, "position", VALUES)
        self.assertFalse(recorder.record(20, "position", object()))
        self.assertTrue(recorder.enabled)
        recorder.record(20.005, "position", VALUES)
        recorder.split()
        recorder.record(20.005, "position", VALUES)
        recorder.record(0, "position", VALUES)
        recorder.close()
        data = [rows(path) for path in sorted(recorder.directory.glob("*.csv"))]
        self.assertEqual([len(part) for part in data], [2, 1, 1])
        self.assertTrue(all(float(part[0]["t_s"]) == 0 for part in data))

    def test_snapshot_is_independent_and_close_is_idempotent(self):
        recorder = self.recorder()
        values = list(VALUES)
        recorder.record(0, "position", values)
        values[:] = [999] * 12
        recorder.close()
        recorder.close(success=False)
        self.assertFalse(recorder.record(1, "position", VALUES))
        self.assertFalse(recorder._thread.is_alive())
        self.assertEqual(float(rows(next(recorder.directory.glob("*.csv")))[0]["rate_ref_x_dps"]), VALUES[6])

    def test_exceptional_close_preserves_partial(self):
        recorder = self.recorder()
        recorder.record(0, "position", VALUES)
        recorder.close(success=False)
        recorder.close(success=True)
        self.assertFalse(list(recorder.directory.glob("*.csv")))
        self.assertEqual(len(rows(next(recorder.directory.glob("*.partial")))), 1)

    def test_completed_segments_survive_later_abort(self):
        recorder = self.recorder()
        recorder.record(0, "position", VALUES)
        recorder.record(0.005, "attitude", VALUES)
        self.assertTrue(wait_until(lambda: len(list(recorder.directory.glob("*.csv"))) == 1))
        recorder.close(success=False)
        self.assertEqual(len(list(recorder.directory.glob("*.csv"))), 1)
        self.assertEqual(len(list(recorder.directory.glob("*.partial"))), 1)

    def test_background_flush_without_additional_samples(self):
        recorder = self.recorder()
        flushed = threading.Event()
        original_open = Path.open

        class ObservedStream:
            def __init__(self, stream):
                self.stream = stream

            def __getattr__(self, name):
                return getattr(self.stream, name)

            def flush(self):
                self.stream.flush()
                flushed.set()

        def open_stream(path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            return ObservedStream(stream) if args and args[0] == "x" else stream

        with patch.object(Path, "open", open_stream):
            recorder.record(0, "position", VALUES)
            self.assertTrue(flushed.wait(1.5))
            recorder.close()

    def test_queue_full_never_waits_for_blocked_writer(self):
        recorder = self.recorder()
        entered, release, returned = threading.Event(), threading.Event(), threading.Event()
        original_writer = csv.writer

        class BlockedWriter:
            def __init__(self, stream):
                self.writer = original_writer(stream)

            def writerow(self, row):
                if row == COLUMNS:
                    entered.set()
                    release.wait(5)
                return self.writer.writerow(row)

        with self.assertLogs("crazyflie_sim.flight_recorder", level="ERROR") as output:
            with patch("crazyflie_sim.flight_recorder.csv.writer", BlockedWriter):
                recorder.record(0, "position", VALUES)
                self.assertTrue(entered.wait(2))
                for i in range(recorder.QUEUE_CAPACITY):
                    self.assertTrue(recorder.record(i + 1, "position", VALUES))

                def overflow():
                    recorder.record(recorder.QUEUE_CAPACITY + 1, "position", VALUES)
                    returned.set()

                producer = threading.Thread(target=overflow)
                producer.start()
                try:
                    self.assertTrue(returned.wait(1), "producer waited for the blocked disk writer")
                    self.assertFalse(recorder.enabled)
                finally:
                    release.set()
                    producer.join(2)
                    recorder.close()
        self.assertEqual(len(output.output), 1)
        self.assertIn("queue full", recorder.error)
        self.assertFalse(list(recorder.directory.glob("*.csv")))
        self.assertEqual(len(rows(next(recorder.directory.glob("*.partial")))), 2049)

    def test_write_failure_preserves_partial_and_stops_accepting(self):
        recorder = self.recorder()
        original_writer = csv.writer

        class BrokenWriter:
            def __init__(self, stream):
                self.writer = original_writer(stream)

            def writerow(self, row):
                if row != COLUMNS:
                    raise OSError("disk full")
                return self.writer.writerow(row)

        with self.assertLogs("crazyflie_sim.flight_recorder", level="ERROR"):
            with patch("crazyflie_sim.flight_recorder.csv.writer", BrokenWriter):
                recorder.record(0, "position", VALUES)
                recorder.close()
        self.assertIn("disk full", recorder.error)
        self.assertFalse(recorder.enabled)
        self.assertFalse(recorder.record(1, "position", VALUES))
        self.assertFalse(list(recorder.directory.glob("*.csv")))
        self.assertEqual(len(list(recorder.directory.glob("*.partial"))), 1)

    def test_existing_completed_file_is_never_overwritten(self):
        recorder = self.recorder()
        destination = recorder.directory / f"flight_{recorder._session}_001_position.csv"
        destination.write_text("keep this")
        with self.assertLogs("crazyflie_sim.flight_recorder", level="ERROR"):
            recorder.record(0, "position", VALUES)
            recorder.close()
        self.assertEqual(destination.read_text(), "keep this")
        self.assertTrue(list(recorder.directory.glob("*.partial")))

    def test_existing_partial_is_never_overwritten(self):
        recorder = self.recorder()
        destination = recorder.directory / f"flight_{recorder._session}_001_position.csv.partial"
        destination.write_text("keep partial")
        with self.assertLogs("crazyflie_sim.flight_recorder", level="ERROR"):
            recorder.record(0, "position", VALUES)
            recorder.close()
        self.assertEqual(destination.read_text(), "keep partial")
        self.assertFalse(list(recorder.directory.glob("*.csv")))

    def test_invalid_capture_stops_recorder_but_inactive_angles_are_ignored(self):
        recorder = self.recorder()
        recorder.record(0, "attitude_rate", [float("nan")] * 6 + list(VALUES[6:]))
        recorder.close()
        self.assertIsNone(recorder.error)
        for i, (timestamp, mode, values) in enumerate((
            (float("nan"), "position", VALUES), (0, "../escape", VALUES),
            (0, "position", [1]), (0, "position", [float("inf")] * 12),
        )):
            with self.subTest(i=i):
                recorder = self.recorder(f"invalid-{i}")
                with self.assertLogs("crazyflie_sim.flight_recorder", level="ERROR"):
                    self.assertFalse(recorder.record(timestamp, mode, values))
                    recorder.close()
                self.assertFalse(recorder.enabled)
                self.assertFalse(list(recorder.directory.glob("*.csv")))

    def test_import_is_standard_library_only_and_creates_no_logs(self):
        repo = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(repo.parent), PYTHONDONTWRITEBYTECODE="1")
        process = subprocess.run(
            [sys.executable, "-c",
             "import sys,threading; import crazyflie_sim.flight_recorder, crazyflie_sim.run; "
             "assert not any(n.startswith(('torch','isaaclab','numpy','response_analysis')) for n in sys.modules); "
             "assert not any(t.name == 'flight-recorder' for t in threading.enumerate())"],
            cwd=self.root, env=env, capture_output=True, text=True,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(DEFAULT_LOG_DIR, repo / "logs")

    def test_recorded_data_is_analyzable_and_comparable(self):
        import pandas as pd
        from response_analysis import AnalysisConfig, analyze_run, compare_groups, validate_run
        from response_analysis.examples import make_run
        from response_analysis.report import write_report

        groups = {}
        for i, (name, gain) in enumerate((("real", 1.0), ("before", 0.7), ("after", 0.98))):
            source = make_run(self.root / f"source-{name}", duration_s=12, gain=gain, seed=i)
            frame = pd.read_csv(source / "samples.csv")
            recorder = self.recorder(name)
            for row in frame[list(COLUMNS)].itertuples(index=False, name=None):
                self.assertTrue(recorder.record(row[0], "attitude_rate", row[1:]))
            recorder.close()
            filename = next(recorder.directory.glob("*.csv"))
            self.assertTrue(validate_run(filename)["valid"])
            self.assertEqual(analyze_run(filename)["channels"]["rate.x"]["status"], "ok")
            groups[name] = [str(filename)]
        result = compare_groups({
            "reference_group": "real", "baseline_group": "before", "groups": groups,
        }, AnalysisConfig(bootstrap_samples=100))
        rate_case = next(case for case in result["cases"] if case["channel"] == "rate.x")
        self.assertGreater(rate_case["groups"]["after"]["improvement_pct"]["step_rmse"], 50)
        output = self.root / "report"
        write_report(result, output)
        self.assertTrue((output / "report.html").is_file())
        self.assertEqual(json.loads((output / "results.json").read_text())["kind"], "comparison")
