#!/usr/bin/env python3
"""
Crazyflie Simulator main entry point.

Runs IsaacLab simulation with CF2.1 BL firmware-compatible controller.
Provides HTTP API for control and state access.
"""
import os
import sys
import argparse
from contextlib import ExitStack
import logging
from pathlib import Path
import signal

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crazyflie_sim.flight_recorder import DEFAULT_LOG_DIR, FlightRecorder


def parse_args(argv=None):
    from isaaclab.app import AppLauncher
    from crazyflie_sim.config import SIM_DT, SERVER

    parser = argparse.ArgumentParser(description="Crazyflie Simulator")
    parser.add_argument("--host", default=SERVER["host"], help="API host")
    parser.add_argument("--port", type=int, default=SERVER["port"], help="API port")
    parser.add_argument("--dt", type=float, default=SIM_DT, help="Time step (s)")
    parser.add_argument("--record", action="store_true", help="Record controller samples as CSV")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="Text and flight log directory")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args(argv)

running = True


def setup_logging(log_dir, settings):
    log_dir = Path(log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = settings.get("format", "[%(asctime)s] [%(levelname)s] %(message)s")
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_dir / "simulator.log", mode="a", encoding="utf-8"),
    ]
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.get("level", "INFO")))
    for handler in handlers:
        handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(handler)
    return handlers


def signal_handler(sig, frame):
    global running
    logging.info("Shutting down...")
    running = False


def main():
    from isaaclab.app import AppLauncher
    from crazyflie_sim.config import PHYSICS, LOGGING

    global running
    running = True
    args = parse_args()
    simulation_app = api_server = recorder = None
    handlers = []
    success = False
    try:
        handlers = setup_logging(args.log_dir, LOGGING)
        logging.info("Starting Crazyflie Simulator")
        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app
        # The launcher installs its own handlers; our graceful drain must win.
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Isaac/Omniverse imports must remain after application startup.
        from crazyflie_sim.sim.simulation_manager import SimulationManager
        from crazyflie_sim.api.server import SimulatorAPIServer

        if args.record:
            recorder = FlightRecorder(args.log_dir)
        sim_manager = SimulationManager(
            simulation_app=simulation_app, dt=args.dt, mass=PHYSICS["mass"],
            arm_length=PHYSICS["arm_length"], inertia=tuple(PHYSICS["inertia"]),
            recorder=recorder,
        )
        api_server = SimulatorAPIServer(args.host, args.port, sim_manager)
        api_server.start()
        logging.info(f"API: http://{args.host}:{args.port}")
        while running and sim_manager.step():
            pass
        success = True
    finally:
        # LIFO callbacks ensure all resources close even if another close fails.
        with ExitStack() as cleanup:
            for handler in handlers:
                cleanup.callback(handler.close)
                cleanup.callback(logging.getLogger().removeHandler, handler)
            cleanup.callback(logging.info, "Simulator stopped")
            if simulation_app is not None:
                cleanup.callback(simulation_app.close)
            if recorder is not None:
                def close_recording(exc_type, exc, traceback):
                    recorder.close(success=success and exc_type is None)
                cleanup.push(close_recording)
            if api_server is not None:
                cleanup.callback(api_server.stop)


if __name__ == "__main__":
    main()
