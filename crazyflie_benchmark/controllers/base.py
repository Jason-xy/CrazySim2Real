from abc import ABC, abstractmethod
import time
import logging
from typing import Optional
from ..core.config import FlightConfig
from ..core.connection import DroneConnectionBase
from ..core.utils import clamp

logger = logging.getLogger(__name__)

class FlightController(ABC):
    """
    Abstract base class for flight controllers.
    """
    def __init__(self, connection: DroneConnectionBase, config: FlightConfig):
        self.connection = connection
        self.config = config
        self.is_armed = False
        self._setpoint_initialized = False
        self._current_height = 0.0
        self.CONTROL_RATE_HZ = 100
        self.CONTROL_PERIOD = 1.0 / self.CONTROL_RATE_HZ

    @property
    def is_connected(self) -> bool:
        return self.connection.is_connected

    def initialize_setpoint(self) -> bool:
        if not self.is_connected:
            return False
        # Send zero setpoint to unlock
        # We use _send_setpoint_impl directly to avoid safety checks for initialization if needed,
        # but send_setpoint is safer.
        self._send_setpoint_impl(0, 0, 0, 0)
        self._setpoint_initialized = True
        time.sleep(0.1)
        return True

    @abstractmethod
    def arm(self) -> bool:
        pass

    @abstractmethod
    def disarm(self) -> bool:
        pass

    @abstractmethod
    def _send_setpoint_impl(self, roll: float, pitch: float, yaw_rate: float, thrust: int) -> bool:
        """Implementation specific setpoint sending."""
        pass

    def send_setpoint(self, roll: float, pitch: float, yaw_rate: float, thrust: int) -> bool:
        """
        Send setpoint with safety clamping.
        roll, pitch: degrees
        yaw_rate: degrees/s
        thrust: 0-65535
        """
        if not self.is_connected:
            return False

        if not self._setpoint_initialized:
            self.initialize_setpoint()

        # Safety clamping
        safe_roll = clamp(roll, -self.config.safety_max_roll_pitch_deg, self.config.safety_max_roll_pitch_deg)
        safe_pitch = clamp(pitch, -self.config.safety_max_roll_pitch_deg, self.config.safety_max_roll_pitch_deg)

        # Thrust safety
        min_thrust = getattr(self.config, 'safety_min_thrust_flight', 0)
        safe_thrust = clamp(thrust, min_thrust, self.config.safety_max_thrust)

        # If thrust is 0 (e.g. stop), allow it even if min_thrust is set
        if thrust == 0:
            safe_thrust = 0

        return self._send_setpoint_impl(safe_roll, safe_pitch, yaw_rate, int(safe_thrust))

    def send_hover_setpoint(self, vx: float, vy: float, yaw_rate: float, z: float) -> bool:
        if not self.is_connected:
            return False

        if not self._setpoint_initialized:
            self.initialize_setpoint()

        safe_z = clamp(z, 0.0, self.config.safety_max_height_m)
        if self._send_hover_impl(vx, vy, yaw_rate, safe_z):
            self._current_height = safe_z
            return True
        return False

    @abstractmethod
    def _send_hover_impl(self, vx: float, vy: float, yaw_rate: float, z: float) -> bool:
        """Transport-level hover command for the backend."""

    def stop(self) -> bool:
        self.send_setpoint(0, 0, 0, 0)
        return self.disarm()

    def take_off(self, target_height: float = 0.5, velocity: float = 0.3) -> bool:
        logger.info(f"TAKEOFF: Starting takeoff to {target_height}m")

        if not self.is_armed:
            if not self.arm():
                return False

        # Ramp up
        max_time = target_height / velocity
        start_time = time.monotonic()
        next_control_time = time.monotonic()

        while time.monotonic() - start_time < max_time + 1.0:
            elapsed = time.monotonic() - start_time
            progress = min(elapsed / max_time, 1.0)
            current_target = target_height * progress

            if not self.send_hover_setpoint(0, 0, 0, current_target):
                return False

            next_control_time += self.CONTROL_PERIOD
            sleep_time = next_control_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                if -sleep_time > 0.1:
                    next_control_time = time.monotonic() + self.CONTROL_PERIOD

        self._current_height = target_height
        return True

    def land(self, velocity: float = 0.2) -> bool:
        logger.info(f"LAND: Starting landing from {self._current_height}m")

        current_height = max(self._current_height, 0.1)
        height_step = velocity * self.CONTROL_PERIOD

        next_control_time = time.monotonic()

        while current_height > 0.05:
            current_height = max(current_height - height_step, 0.05)
            if not self.send_hover_setpoint(0, 0, 0, current_height):
                break

            next_control_time += self.CONTROL_PERIOD
            sleep_time = next_control_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                if -sleep_time > 0.1:
                    next_control_time = time.monotonic() + self.CONTROL_PERIOD

        # Final touchdown - reduce thrust
        current_thrust = int(self.config.hover_thrust * 0.75)
        for _ in range(30):
            current_thrust = max(0, current_thrust - 500)
            self.send_setpoint(0, 0, 0, current_thrust)
            time.sleep(self.CONTROL_PERIOD)

        self.send_setpoint(0, 0, 0, 0)
        return self.disarm()

    def maintain_hover(self, duration_s: float, target_height: float = 0.5) -> bool:
        logger.info(f"HOVER: Maintaining {target_height}m for {duration_s}s")
        start_time = time.monotonic()
        next_control_time = time.monotonic()

        while time.monotonic() - start_time < duration_s:
            if not self.send_hover_setpoint(0, 0, 0, target_height):
                return False

            next_control_time += self.CONTROL_PERIOD
            sleep_time = next_control_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                if -sleep_time > 0.1:
                    next_control_time = time.monotonic() + self.CONTROL_PERIOD
        return True
