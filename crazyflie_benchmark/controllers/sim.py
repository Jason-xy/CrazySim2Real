import logging
import requests

from .base import FlightController
from ..core.utils import clamp

logger = logging.getLogger(__name__)

class SimFlightController(FlightController):
    MAX_ROLL_PITCH_DEG = 30.0   # ±30deg -> ±1.0 (matches firmware)
    MAX_YAW_RATE_DEG = 200.0    # ±200deg/s -> ±1.0

    def arm(self) -> bool:
        self.is_armed = True
        return True

    def disarm(self) -> bool:
        self.is_armed = False
        return self._send_stop()

    def _send_setpoint_impl(self, roll: float, pitch: float, yaw_rate: float, thrust: int) -> bool:
        norm_roll = clamp(roll / self.MAX_ROLL_PITCH_DEG, -1.0, 1.0)
        norm_pitch = clamp(pitch / self.MAX_ROLL_PITCH_DEG, -1.0, 1.0)
        norm_yaw_rate = clamp(yaw_rate / self.MAX_YAW_RATE_DEG, -1.0, 1.0)
        norm_thrust = clamp(thrust / 65535.0, 0.0, 1.0)
        return self._post_attitude(norm_roll, norm_pitch, norm_yaw_rate, norm_thrust)

    def _send_hover_impl(self, vx: float, vy: float, yaw_rate: float, z: float) -> bool:
        """
        Simulator has no native hover primitive; convert velocity command into a
        position target on the client side to keep the connection layer pure I/O.
        """
        state = self.connection.get_state()
        position = state.get("position", {"x": 0.0, "y": 0.0, "z": 0.0})
        current_yaw = state.get("stabilizer", {}).get("yaw", 0.0)

        dt = self.CONTROL_PERIOD
        target_x = position["x"] + vx * dt
        target_y = position["y"] + vy * dt
        target_yaw = current_yaw + yaw_rate * dt

        return self._post_position(target_x, target_y, z, target_yaw)

    def _post_attitude(self, roll: float, pitch: float, yaw_rate: float, thrust: float) -> bool:
        if not self.connection.is_connected:
            logger.warning("Cannot send setpoint: simulator not connected.")
            return False

        try:
            response = requests.post(
                f"{self.connection.base_url}/control/attitude",
                json={
                    "roll": roll,
                    "pitch": pitch,
                    "yaw_rate": yaw_rate,
                    "thrust": thrust,
                },
                timeout=1.0,
            )
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error(f"Error sending attitude command to simulator: {e}")
            return False

    def _post_position(self, x: float, y: float, z: float, yaw: float) -> bool:
        if not self.connection.is_connected:
            logger.warning("Cannot send position setpoint: simulator not connected.")
            return False

        try:
            response = requests.post(
                f"{self.connection.base_url}/control/position",
                json={"x": x, "y": y, "z": z, "yaw": yaw},
                timeout=1.0,
            )
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error(f"Error sending position command to simulator: {e}")
            return False

    def _send_stop(self) -> bool:
        return self._post_attitude(0.0, 0.0, 0.0, 0.0)
