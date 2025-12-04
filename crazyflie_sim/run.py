#!/usr/bin/env python3
"""
Crazyflie Simulator main entry point.

Runs IsaacLab simulation with CF2.1 BL firmware-compatible controller.
Provides HTTP API for control and state access.
"""
import os
import sys
import argparse
import logging
import signal

from isaaclab.app import AppLauncher

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crazyflie_sim.config import SIM_DT, PHYSICS, SERVER, LOGGING


def parse_args():
    parser = argparse.ArgumentParser(description="Crazyflie Simulator")
    parser.add_argument("--host", default=SERVER["host"], help="API host")
    parser.add_argument("--port", type=int, default=SERVER["port"], help="API port")
    parser.add_argument("--dt", type=float, default=SIM_DT, help="Time step (s)")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args = parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from crazyflie_sim.sim.simulation_manager import SimulationManager
from crazyflie_sim.api.server import SimulatorAPIServer

running = True


def setup_logging():
    fmt = LOGGING.get("format", "[%(asctime)s] [%(levelname)s] %(message)s")
    level = getattr(logging, LOGGING.get("level", "INFO"))

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)


def signal_handler(sig, frame):
    global running
    logging.info("Shutting down...")
    running = False


def main():
    setup_logging()
    logging.info("Starting Crazyflie Simulator")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    sim_manager = SimulationManager(
        simulation_app=simulation_app,
        dt=args.dt,
        mass=PHYSICS["mass"],
        arm_length=PHYSICS["arm_length"],
        inertia=tuple(PHYSICS["inertia"]),
    )

    api_server = SimulatorAPIServer(args.host, args.port, sim_manager)
    api_server.start()

    logging.info(f"API: http://{args.host}:{args.port}")

    while running and sim_manager.step():
        pass

    api_server.stop()
    logging.info("Simulator stopped")


if __name__ == "__main__":
    main()
