#!/usr/bin/env python3
"""
Xbox Controller Client for Crazyflie Simulator.

Mode 2 (American Hand) Stick Mapping:
    Left Stick:  Throttle/Altitude (Y) + Yaw (X)
    Right Stick: Forward/Back (Y) + Left/Right (X)

Control Modes:
    ATTITUDE:  Direct roll/pitch/yaw_rate/thrust control
    VELOCITY:  Body-frame velocity control
    POSITION:  World-frame position control

Buttons:
    RB: Switch mode
    Y:  Takeoff (1m)
    A:  Land
    B:  Reset
"""
import os
import sys
import time
import math
import enum
import logging
import argparse
import threading
from dataclasses import dataclass, field, replace
from typing import Dict, Optional
import requests
import pygame

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    filename="xbox_client.log",
    filemode="w",
)
logger = logging.getLogger("xbox_client")

# Constants
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000
CONTROL_RATE_HZ = 50
DT = 1.0 / CONTROL_RATE_HZ
DEADZONE = 0.1

# Xbox Controller Axes (Linux/xpad)
AXIS_LEFT_X = 0   # Yaw
AXIS_LEFT_Y = 1   # Throttle/Alt (inverted)
AXIS_RIGHT_X = 2  # Roll/Strafe
AXIS_RIGHT_Y = 3  # Pitch/Forward (inverted)

# Xbox Controller Buttons
BTN_A = 0   # Land
BTN_B = 1   # Reset
BTN_Y = 4   # Takeoff
BTN_RB = 7  # Mode switch


class ControlMode(enum.Enum):
    ATTITUDE = 0
    VELOCITY = 1
    POSITION = 2

    def __str__(self):
        return self.name.capitalize()


@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @staticmethod
    def from_dict(d: Dict[str, float]) -> "Vector3":
        return Vector3(d.get("x", 0.0), d.get("y", 0.0), d.get("z", 0.0))


@dataclass
class Orientation:
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    @staticmethod
    def from_dict(d: Dict[str, float]) -> "Orientation":
        return Orientation(d.get("roll", 0.0), d.get("pitch", 0.0), d.get("yaw", 0.0))


@dataclass
class DroneState:
    position: Vector3 = field(default_factory=Vector3)
    velocity: Vector3 = field(default_factory=Vector3)
    orientation: Orientation = field(default_factory=Orientation)


@dataclass
class StickInput:
    left_x: float = 0.0   # Yaw
    left_y: float = 0.0   # Throttle
    right_x: float = 0.0  # Roll/Strafe
    right_y: float = 0.0  # Pitch/Forward


def apply_deadzone(val: float, dz: float = DEADZONE) -> float:
    if abs(val) < dz:
        return 0.0
    sign = 1.0 if val > 0 else -1.0
    return sign * (abs(val) - dz) / (1.0 - dz)


class Controller:
    def __init__(self):
        self.mode = ControlMode.ATTITUDE
        self.lock = threading.Lock()
        self.message = "Ready"
        self.running = True

        # Setpoints
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw_rate = 0.0
        self.thrust = 0.0

        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.yaw = 0.0

        # Scaling
        self.pos_rate = 0.05   # m per tick
        self.vel_scale = 1.0   # m/s
        self.yaw_deg_rate = 2.0  # deg per tick

    def update(self, sticks: StickInput, state: Optional[DroneState]):
        with self.lock:
            if self.mode == ControlMode.ATTITUDE:
                self._update_attitude(sticks)
            elif self.mode == ControlMode.VELOCITY:
                self._update_velocity(sticks, state)
            elif self.mode == ControlMode.POSITION:
                self._update_position(sticks, state)

    def _update_attitude(self, s: StickInput):
        # Right stick: roll (X) and pitch (Y)
        # Left stick: yaw rate (X, inverted for CW) and throttle (Y, inverted)
        self.roll = s.right_x
        self.pitch = -s.right_y
        self.yaw_rate = -s.left_x    # Right = positive yaw (clockwise from above)
        self.thrust = (-s.left_y + 1.0) / 2.0  # Up = more thrust

        self.roll = max(-1.0, min(1.0, self.roll))
        self.pitch = max(-1.0, min(1.0, self.pitch))
        self.yaw_rate = max(-1.0, min(1.0, self.yaw_rate))
        self.thrust = max(0.0, min(1.0, self.thrust))

    def _update_velocity(self, s: StickInput, state: Optional[DroneState]):
        # Body frame velocity
        # Right stick: forward/back (Y) and left/right strafe (X)
        # Left stick: yaw rate (X) and altitude (Y)
        self.vx = -s.right_y * self.vel_scale   # Up = forward
        self.vy = -s.right_x * self.vel_scale   # Right = strafe right
        self.vz = -s.left_y * self.vel_scale    # Up = climb
        self.yaw_rate = -s.left_x * 120.0       # Right = clockwise (deg/s)

    def _update_position(self, s: StickInput, state: Optional[DroneState]):
        # Get yaw for body-to-world transform
        yaw_rad = math.radians(self.yaw)
        if state:
            yaw_rad = math.radians(state.orientation.yaw)

        # Body frame -> world frame
        # Right stick: forward/back (Y) and left/right (X)
        body_dx = -s.right_y * self.pos_rate  # Up = forward
        body_dy = -s.right_x * self.pos_rate  # Right = strafe right
        self.x += body_dx * math.cos(yaw_rad) - body_dy * math.sin(yaw_rad)
        self.y += body_dx * math.sin(yaw_rad) + body_dy * math.cos(yaw_rad)

        # Left stick: altitude (Y) and yaw (X)
        self.z += -s.left_y * self.pos_rate   # Up = climb
        self.z = max(0.0, self.z)

        self.yaw -= s.left_x * self.yaw_deg_rate  # Right = clockwise
        self.yaw = ((self.yaw + 180) % 360) - 180

    def get_command(self):
        with self.lock:
            return {
                "mode": self.mode,
                "attitude": {"roll": self.roll, "pitch": self.pitch, "yaw_rate": self.yaw_rate, "thrust": self.thrust},
                "velocity": {"vx": self.vx, "vy": self.vy, "vz": self.vz, "yaw_rate": self.yaw_rate},
                "position": {"x": self.x, "y": self.y, "z": self.z, "yaw": self.yaw},
            }

    def switch_mode(self):
        with self.lock:
            modes = list(ControlMode)
            idx = (modes.index(self.mode) + 1) % len(modes)
            self.mode = modes[idx]
            self.message = f"Mode: {self.mode}"
            self._reset_setpoints()
            logger.info(f"Mode: {self.mode}")

    def _reset_setpoints(self):
        self.roll = self.pitch = self.yaw_rate = self.thrust = 0.0
        self.vx = self.vy = self.vz = 0.0
        self.x = self.y = 0.0
        self.yaw = 0.0
        if self.mode == ControlMode.POSITION:
            self.z = 1.0
        else:
            self.z = 0.0

    def takeoff(self):
        with self.lock:
            self.mode = ControlMode.POSITION
            self.x = self.y = 0.0
            self.z = 1.0
            self.yaw = 0.0
            self.message = "Takeoff"
            logger.info("Takeoff")

    def land(self):
        with self.lock:
            self.mode = ControlMode.POSITION
            self.z = 0.0
            self.message = "Landing"
            logger.info("Landing")


class NetworkThread(threading.Thread):
    def __init__(self, host: str, port: int, controller: Controller):
        super().__init__(daemon=True)
        self.base_url = f"http://{host}:{port}"
        self.controller = controller
        self.state: Optional[DroneState] = None
        self.state_lock = threading.Lock()
        self.running = True
        self._session = requests.Session()

    def get_state(self) -> Optional[DroneState]:
        with self.state_lock:
            return self.state

    def stop(self):
        self.running = False

    def reset_sim(self):
        try:
            self._session.post(f"{self.base_url}/reset", timeout=0.5)
        except:
            pass

    def run(self):
        while self.running:
            t0 = time.time()

            # Get state
            try:
                r = self._session.get(f"{self.base_url}/state", timeout=0.3)
                if r.ok:
                    d = r.json()
                    with self.state_lock:
                        self.state = DroneState(
                            position=Vector3.from_dict(d.get("position", {})),
                            velocity=Vector3.from_dict(d.get("velocity", {})),
                            orientation=Orientation.from_dict(d.get("orientation", {})),
                        )
            except:
                pass

            # Send command
            cmd = self.controller.get_command()
            try:
                if cmd["mode"] == ControlMode.POSITION:
                    self._session.post(f"{self.base_url}/control/position", json=cmd["position"], timeout=0.1)
                elif cmd["mode"] == ControlMode.VELOCITY:
                    self._session.post(f"{self.base_url}/control/velocity", json=cmd["velocity"], timeout=0.1)
                else:
                    self._session.post(f"{self.base_url}/control/attitude", json=cmd["attitude"], timeout=0.1)
            except:
                pass

            time.sleep(max(0, 0.02 - (time.time() - t0)))


class XboxHandler:
    def __init__(self, controller: Controller, network: NetworkThread):
        self.controller = controller
        self.network = network
        self.joystick = None
        self.sticks = StickInput()
        self._prev_buttons = {}

        pygame.init()
        pygame.joystick.init()

    def init_joystick(self) -> bool:
        if pygame.joystick.get_count() == 0:
            return False
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        logger.info(f"Controller: {self.joystick.get_name()}")
        return True

    def update(self):
        pygame.event.pump()
        if not self.joystick:
            return

        n = self.joystick.get_numaxes()
        self.sticks = StickInput(
            apply_deadzone(self.joystick.get_axis(AXIS_LEFT_X)) if n > AXIS_LEFT_X else 0.0,
            apply_deadzone(self.joystick.get_axis(AXIS_LEFT_Y)) if n > AXIS_LEFT_Y else 0.0,
            apply_deadzone(self.joystick.get_axis(AXIS_RIGHT_X)) if n > AXIS_RIGHT_X else 0.0,
            apply_deadzone(self.joystick.get_axis(AXIS_RIGHT_Y)) if n > AXIS_RIGHT_Y else 0.0,
        )

        nb = self.joystick.get_numbuttons()
        for btn in [BTN_A, BTN_B, BTN_Y, BTN_RB]:
            if btn >= nb:
                continue
            pressed = self.joystick.get_button(btn)
            was = self._prev_buttons.get(btn, False)
            if pressed and not was:
                self._on_button(btn)
            self._prev_buttons[btn] = pressed

    def _on_button(self, btn: int):
        if btn == BTN_RB:
            self.controller.switch_mode()
        elif btn == BTN_Y:
            self.controller.takeoff()
        elif btn == BTN_A:
            self.controller.land()
        elif btn == BTN_B:
            self.network.reset_sim()
            with self.controller.lock:
                self.controller.mode = ControlMode.ATTITUDE
                self.controller._reset_setpoints()
                self.controller.message = "Reset"

    def stop(self):
        pygame.quit()


class Display:
    def __init__(self):
        self.use_colors = os.environ.get("TERM", "") not in ("", "dumb")

    def _c(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.use_colors else text

    def render(self, state: Optional[DroneState], controller: Controller, sticks: StickInput):
        print("\033[2J\033[H" if self.use_colors else "\n" * 30, end="")
        cmd = controller.get_command()

        print(self._c("Crazyflie Xbox Controller", "1;36"))
        print(self._c("=" * 40, "36"))

        print(f"\n{self._c('Mode:', '1;33')} {controller.mode}")
        with controller.lock:
            print(f"{self._c('Msg:', '33')}  {controller.message}")

        print(f"\n{self._c('State:', '1;32')}")
        if state:
            p, o = state.position, state.orientation
            print(f"  Pos: x={p.x:+.2f} y={p.y:+.2f} z={p.z:+.2f}")
            print(f"  Att: r={o.roll:+.1f}° p={o.pitch:+.1f}° y={o.yaw:+.1f}°")
        else:
            print(self._c("  [No Connection]", "31"))

        print(f"\n{self._c('Command:', '37')}")
        if cmd["mode"] == ControlMode.ATTITUDE:
            a = cmd["attitude"]
            print(f"  roll={a['roll']:+.2f} pitch={a['pitch']:+.2f} yaw_rate={a['yaw_rate']:+.2f} thrust={a['thrust']:.2f}")
        elif cmd["mode"] == ControlMode.VELOCITY:
            v = cmd["velocity"]
            print(f"  vx={v['vx']:+.2f} vy={v['vy']:+.2f} vz={v['vz']:+.2f}")
        else:
            p = cmd["position"]
            print(f"  x={p['x']:+.2f} y={p['y']:+.2f} z={p['z']:+.2f} yaw={p['yaw']:+.1f}°")

        print(f"\n{self._c('Sticks:', '90')}")
        print(f"  L: ({sticks.left_x:+.2f}, {sticks.left_y:+.2f})  R: ({sticks.right_x:+.2f}, {sticks.right_y:+.2f})")


def main():
    parser = argparse.ArgumentParser(description="Xbox Controller for Crazyflie Simulator")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    controller = Controller()
    network = NetworkThread(args.host, args.port, controller)
    xbox = XboxHandler(controller, network)
    display = Display()

    if not xbox.init_joystick():
        print("No Xbox controller found!")
        sys.exit(1)

    network.start()

    try:
        while controller.running:
            t0 = time.time()
            xbox.update()
            controller.update(xbox.sticks, network.get_state())
            display.render(network.get_state(), controller, xbox.sticks)
            time.sleep(max(0, DT - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        xbox.stop()
        network.stop()
        print("\nExiting...")


if __name__ == "__main__":
    main()
