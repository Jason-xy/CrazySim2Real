import copy
from contextlib import redirect_stderr
from enum import IntEnum
import importlib
import importlib.util
import io
import logging
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import MagicMock, patch

from crazyflie_sim.flight_recorder import DEFAULT_LOG_DIR, FlightRecorder
from crazyflie_sim import run
from .test_flight_recorder import rows


class FakeTensor:
    copies = 0

    def __init__(self, value):
        self.value = value

    def __getitem__(self, key):
        return FakeTensor(self.value[key])

    def item(self):
        return self.value

    def detach(self):
        return self

    def cpu(self):
        FakeTensor.copies += 1
        return self

    def tolist(self):
        return copy.deepcopy(self.value)


ControlMode = IntEnum("ControlMode", {"ATTITUDE": 0, "ATTITUDE_RATE": 1, "VELOCITY": 2, "POSITION": 3})


def stub_module(name, **values):
    module = types.ModuleType(name)
    module.__dict__.update(values)
    return module


def load_simulation_manager():
    """Use real manager methods but never import or start a simulator SDK."""
    names = (
        "torch", "isaaclab", "isaaclab.sim", "isaaclab.assets", "isaaclab.markers",
        "isaaclab.utils", "isaaclab.utils.assets", "isaaclab.utils.math",
        "isaacsim", "isaacsim.core", "isaacsim.core.utils", "isaacsim.core.utils.prims",
        "isaacsim.core.utils.stage", "isaaclab_assets",
    )
    modules = {name: MagicMock(name=name) for name in names}
    modules["crazyflie_sim.controllers.cf_controller"] = stub_module(
        "crazyflie_sim.controllers.cf_controller",
        CrazyflieController=MagicMock(), ControlMode=ControlMode, config=MagicMock(),
    )
    path = Path(__file__).resolve().parents[1] / "sim" / "simulation_manager.py"
    spec = importlib.util.spec_from_file_location("_recorder_test_manager", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {**modules, spec.name: module}):
        spec.loader.exec_module(module)
    return module.SimulationManager


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Manager = load_simulation_manager()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="recording-integration-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        FakeTensor.copies = 0

    def manager(self, recorder=None, mode=ControlMode.ATTITUDE):
        manager = self.Manager.__new__(self.Manager)
        manager.recorder = recorder
        manager._step_lock = threading.Lock()
        manager._recording_reset = False
        manager.state = types.SimpleNamespace(timestamp=10.0)
        manager.simulation_app = types.SimpleNamespace(is_running=lambda: True)
        manager.controller = types.SimpleNamespace(
            control_mode=FakeTensor([int(mode)]),
            last_debug={
                "attitude_desired": FakeTensor([[1, 2, 3]]),
                "attitude": FakeTensor([[4, 5, 6]]),
                "rate_desired": FakeTensor([[7, 8, 9]]),
                "rate_actual": FakeTensor([[10, 11, 12]]),
            },
        )
        manager.events = []
        manager._update_state = lambda: manager.events.append("state")
        manager._process_commands = lambda: manager.events.append("commands")

        def compute():
            manager.events.append("control")
            return 17, 23

        manager._compute_control = compute
        manager._apply_control = lambda force, torque: manager.events.append(("apply", force, torque))
        manager._update_frame_marker = lambda: manager.events.append("marker")
        manager.sim = types.SimpleNamespace(step=lambda: manager.events.append("physics"), current_time=999)
        return manager

    def test_capture_occurs_before_physics_with_same_control_outputs(self):
        recorder = FlightRecorder(self.root)
        self.addCleanup(recorder.close)
        manager = self.manager(recorder)
        original_record = recorder.record

        def record(*args):
            manager.events.append("capture")
            return original_record(*args)

        with patch.object(recorder, "record", record):
            self.assertTrue(manager.step())
        recorder.close()
        self.assertEqual(manager.events, [
            "state", "commands", "control", "capture", ("apply", 17, 23), "marker", "physics",
        ])
        row = rows(next(self.root.glob("*.csv")))[0]
        self.assertEqual(float(row["t_s"]), 0)
        self.assertEqual(float(row["angle_ref_roll_deg"]), 1)
        self.assertEqual(float(row["angle_meas_roll_deg"]), 4)
        self.assertEqual(row["angle_ref_yaw_deg"], "")
        self.assertEqual(float(row["rate_ref_z_dps"]), 9)
        self.assertEqual(float(row["rate_meas_z_dps"]), 12)
        without = self.manager()
        without.step()
        self.assertEqual([e for e in manager.events if isinstance(e, tuple)], [e for e in without.events if isinstance(e, tuple)])

    def test_disabled_recorder_never_reads_or_copies_debug_tensors(self):
        manager = self.manager()
        manager.controller = object()
        manager.step()
        self.assertEqual(FakeTensor.copies, 0)
        self.assertIn(("apply", 17, 23), manager.events)

    def test_rate_mode_does_not_copy_unused_angle_tensors(self):
        recorder = FlightRecorder(self.root)
        self.addCleanup(recorder.close)
        manager = self.manager(recorder, ControlMode.ATTITUDE_RATE)
        del manager.controller.last_debug["attitude"]
        del manager.controller.last_debug["attitude_desired"]
        manager.step()
        recorder.close()
        self.assertEqual(FakeTensor.copies, 2)
        row = rows(next(self.root.glob("*.csv")))[0]
        self.assertEqual(row["angle_ref_roll_deg"], "")
        self.assertEqual(float(row["rate_ref_x_dps"]), 7)

    def test_simulation_timestamp_is_used_and_snapshots_do_not_change(self):
        recorder = FlightRecorder(self.root)
        self.addCleanup(recorder.close)
        manager = self.manager(recorder)
        manager.step()
        manager.state.timestamp = 10.005
        manager.controller.last_debug["rate_actual"].value[0][0] = 20
        manager.step()
        manager.controller.last_debug["rate_actual"].value[0][0] = 999
        recorder.close()
        data = rows(next(self.root.glob("*.csv")))
        self.assertAlmostEqual(float(data[1]["t_s"]), 0.005)
        self.assertEqual([float(row["rate_meas_x_dps"]) for row in data], [10, 20])

    def test_reset_and_step_are_serialized_and_split_on_next_sample(self):
        recorder = FlightRecorder(self.root)
        self.addCleanup(recorder.close)
        manager = self.manager(recorder)
        entered, release, reset_done = threading.Event(), threading.Event(), threading.Event()

        def update_state():
            entered.set()
            release.wait(3)

        manager._update_state = update_state
        manager._reset = lambda: manager.events.append("reset")
        step_thread = threading.Thread(target=manager.step)

        def reset():
            manager.reset()
            reset_done.set()

        reset_thread = threading.Thread(target=reset)
        step_thread.start()
        try:
            self.assertTrue(entered.wait(1))
            reset_thread.start()
            self.assertFalse(reset_done.wait(0.05))
        finally:
            release.set()
            step_thread.join(3)
            if reset_thread.ident is not None:
                reset_thread.join(3)
        self.assertTrue(reset_done.is_set())
        self.assertGreater(manager.events.index("reset"), manager.events.index("physics"))
        self.assertTrue(manager._recording_reset)
        manager.step()  # Same clock, but reset must start a new segment.
        recorder.close()
        self.assertEqual(len(list(self.root.glob("*.csv"))), 2)
        self.assertFalse(manager._recording_reset)

    def test_snapshot_error_does_not_stop_control(self):
        recorder = FlightRecorder(self.root)
        self.addCleanup(recorder.close)
        manager = self.manager(recorder)
        manager.controller.last_debug = {}
        with self.assertLogs("crazyflie_sim.flight_recorder", level="ERROR"):
            self.assertTrue(manager.step())
            recorder.close()
        self.assertFalse(recorder.enabled)
        self.assertIn(("apply", 17, 23), manager.events)
        self.assertIn("physics", manager.events)

    def test_malformed_axis_counts_cannot_silently_shift_csv_columns(self):
        recorder = FlightRecorder(self.root)
        self.addCleanup(recorder.close)
        manager = self.manager(recorder)
        manager.controller.last_debug["attitude_desired"] = FakeTensor([[1, 2, 3, 4]])
        manager.controller.last_debug["attitude"] = FakeTensor([[1, 2, 3, 4]])
        manager.controller.last_debug["rate_desired"] = FakeTensor([[1, 2]])
        manager.controller.last_debug["rate_actual"] = FakeTensor([[1, 2]])
        with self.assertLogs("crazyflie_sim.flight_recorder", level="ERROR"):
            manager.step()
            recorder.close()
        self.assertIn("three axes", recorder.error)
        self.assertFalse(list(self.root.glob("*.csv")))
        self.assertIn(("apply", 17, 23), manager.events)


class EntrypointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="recording-entrypoint-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old_level = logging.getLogger().level
        self.addCleanup(logging.getLogger().setLevel, self.old_level)

    def config_module(self):
        return stub_module(
            "crazyflie_sim.config", SIM_DT=0.005, SERVER={"host": "localhost", "port": 8000},
            PHYSICS={"mass": 0.05, "arm_length": 0.05, "inertia": (1, 1, 1)},
            LOGGING={"level": "INFO"},
        )

    def test_simulator_arguments_and_append_logging(self):
        launcher = types.SimpleNamespace(add_app_launcher_args=lambda parser: None)
        with patch.dict(sys.modules, {
            "isaaclab.app": stub_module("isaaclab.app", AppLauncher=launcher),
            "crazyflie_sim.config": self.config_module(),
        }):
            defaults = run.parse_args([])
            selected = run.parse_args(["--record", "--log-dir", str(self.root)])
        self.assertFalse(defaults.record)
        self.assertEqual(defaults.log_dir, DEFAULT_LOG_DIR)
        self.assertTrue(selected.record)
        self.assertEqual(selected.log_dir, self.root)
        path = self.root / "simulator.log"
        path.write_text("previous session\n")
        for text in ("first new session", "second new session"):
            with redirect_stderr(io.StringIO()):
                handlers = run.setup_logging(self.root, {"level": "INFO"})
            try:
                logging.info(text)
            finally:
                for handler in handlers:
                    logging.getLogger().removeHandler(handler)
                    handler.close()
        contents = path.read_text()
        self.assertIn("previous session", contents)
        self.assertIn("first new session", contents)
        self.assertIn("second new session", contents)

    def xbox_module(self):
        with patch.dict(sys.modules, {"pygame": stub_module("pygame")}):
            return importlib.import_module("crazyflie_sim.xbox_client")

    def test_xbox_migration_preserves_conflicts_and_uses_selected_directory(self):
        xbox = self.xbox_module()
        script_dir, cwd, destination = (self.root / name for name in ("script", "cwd", "logs"))
        for directory in (script_dir, cwd, destination):
            directory.mkdir()
        (script_dir / "xbox_client.log").write_text("script history\n")
        (cwd / "xbox_client.log").write_text("cwd history\n")
        (destination / "xbox_client.log").write_text("existing destination\n")
        with patch.object(xbox, "__file__", str(script_dir / "xbox_client.py")), patch.object(Path, "cwd", return_value=cwd):
            handler = xbox.setup_logging(destination)
            try:
                xbox.logger.info("new session")
            finally:
                xbox.logger.removeHandler(handler)
                handler.close()
        self.assertFalse((script_dir / "xbox_client.log").exists())
        self.assertFalse((cwd / "xbox_client.log").exists())
        histories = [path.read_text() for path in destination.glob("xbox_client_history_*.log")]
        self.assertCountEqual(histories, ["script history\n", "cwd history\n"])
        current = (destination / "xbox_client.log").read_text()
        self.assertIn("existing destination", current)
        self.assertIn("new session", current)

    def test_xbox_migration_keeps_source_when_copy_fails(self):
        xbox = self.xbox_module()
        legacy = self.root / "xbox_client.log"
        legacy.write_text("keep original")
        with patch.object(xbox, "__file__", str(self.root / "xbox_client.py")), patch.object(Path, "cwd", return_value=self.root):
            with patch.object(xbox.shutil, "copyfileobj", side_effect=OSError("write failure")):
                with self.assertRaises(OSError):
                    xbox.setup_logging(self.root / "logs")
        self.assertEqual(legacy.read_text(), "keep original")

    def test_xbox_migrates_to_empty_destination_without_creating_history(self):
        xbox = self.xbox_module()
        legacy = self.root / "xbox_client.log"
        legacy.write_text("old session\n")
        destination = self.root / "logs"
        with patch.object(xbox, "__file__", str(self.root / "xbox_client.py")), patch.object(Path, "cwd", return_value=self.root):
            handler = xbox.setup_logging(destination)
            xbox.logger.removeHandler(handler)
            handler.close()
        self.assertFalse(legacy.exists())
        self.assertEqual((destination / "xbox_client.log").read_text(), "old session\n")
        self.assertEqual(list(destination.glob("*history*")), [])

    def test_xbox_main_accepts_log_dir_and_closes_handler_on_early_exit(self):
        xbox = self.xbox_module()
        handler = MagicMock()
        gamepad = MagicMock()
        gamepad.init_joystick.return_value = False
        network = MagicMock()
        with patch.object(sys, "argv", ["xbox_client.py", "--log-dir", str(self.root)]):
            with patch.object(xbox, "setup_logging", return_value=handler) as setup:
                with patch.object(xbox, "XboxHandler", return_value=gamepad), patch.object(xbox, "NetworkThread", return_value=network):
                    with patch("builtins.print"), self.assertRaises(SystemExit) as raised:
                        xbox.main()
        self.assertEqual(raised.exception.code, 1)
        setup.assert_called_once_with(self.root)
        handler.close.assert_called_once()
        gamepad.stop.assert_called_once()
        network.start.assert_not_called()
        network.stop.assert_called_once()

    def test_xbox_import_has_no_file_side_effects_and_default_path_is_fixed(self):
        xbox = self.xbox_module()
        with patch.object(Path, "cwd", return_value=self.root), patch.object(logging, "FileHandler") as handler:
            with patch.dict(sys.modules, {
                "pygame": stub_module("pygame"), "crazyflie_sim.xbox_client": xbox,
            }):
                importlib.reload(xbox)
            handler.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(xbox.DEFAULT_LOG_DIR, DEFAULT_LOG_DIR)

    def main_case(self, record, failure=None):
        events = []
        app = types.SimpleNamespace(close=lambda: events.append("app.close"))
        args = types.SimpleNamespace(record=record, log_dir=self.root / "logs", host="localhost", port=8000, dt=0.005)

        class Recorder:
            def __init__(self, directory):
                events.append("recorder.init")

            def close(self, success=True):
                events.append(("recorder.close", success))

        class Manager:
            def __init__(self, **kwargs):
                events.append(("manager.recorder", kwargs["recorder"] is not None))
                if failure == "init":
                    raise RuntimeError("init failure")

            def step(self):
                if failure == "step":
                    raise RuntimeError("step failure")
                return False

        class Server:
            def __init__(self, *args):
                pass

            def start(self):
                events.append("api.start")

            def stop(self):
                events.append("api.stop")
                if failure == "stop":
                    raise RuntimeError("stop failure")

        def launch(args):
            events.append("launcher")
            return types.SimpleNamespace(app=app)

        modules = {
            "isaaclab.app": stub_module("isaaclab.app", AppLauncher=launch),
            "crazyflie_sim.config": self.config_module(),
            "crazyflie_sim.sim.simulation_manager": stub_module("crazyflie_sim.sim.simulation_manager", SimulationManager=Manager),
            "crazyflie_sim.api.server": stub_module("crazyflie_sim.api.server", SimulatorAPIServer=Server),
        }
        with patch.dict(sys.modules, modules), patch.object(run, "parse_args", return_value=args):
            with patch.object(run, "FlightRecorder", Recorder), patch.object(
                run.signal, "signal", side_effect=lambda sig, handler: events.append(("signal", sig))
            ), redirect_stderr(io.StringIO()):
                if failure:
                    with self.assertRaisesRegex(RuntimeError, "failure"):
                        run.main()
                else:
                    run.main()
        return events

    def test_recording_disabled_creates_no_recorder(self):
        events = self.main_case(False)
        self.assertNotIn("recorder.init", events)
        self.assertIn(("manager.recorder", False), events)
        self.assertEqual(events[-2:], ["api.stop", "app.close"])
        for sig in (run.signal.SIGINT, run.signal.SIGTERM):
            self.assertGreater(events.index(("signal", sig)), events.index("launcher"))

    def test_cleanup_order_and_failure_marks_incomplete(self):
        events = self.main_case(True)
        self.assertEqual(events[-3:], ["api.stop", ("recorder.close", True), "app.close"])
        for failure in ("step", "stop"):
            with self.subTest(failure=failure):
                events = self.main_case(True, failure)
                self.assertEqual(events[-3:], ["api.stop", ("recorder.close", False), "app.close"])
        events = self.main_case(True, "init")
        self.assertEqual(events[-2:], [("recorder.close", False), "app.close"])
