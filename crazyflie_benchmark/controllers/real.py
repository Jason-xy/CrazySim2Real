import logging
import time
from .base import FlightController
from cflib.crtp.crtpstack import CRTPPacket, CRTPPort

logger = logging.getLogger(__name__)

class RealFlightController(FlightController):
    def arm(self) -> bool:
        if not self.is_connected:
            return False

        logger.info("ARM: Sending platform ARM command...")
        pk = CRTPPacket()
        pk.port = CRTPPort.PLATFORM
        pk.channel = 0
        pk.data = bytes([1, 1]) # ARM command

        try:
            self.connection.scf.cf.send_packet(pk)
            self.is_armed = True
            time.sleep(0.3)
            return True
        except Exception as e:
            logger.error(f"ARM failed: {e}")
            return False

    def disarm(self) -> bool:
        if not self.is_connected:
            return False

        logger.info("DISARM: Sending platform DISARM command...")
        try:
            self.connection.scf.cf.platform.set_arming_request(state=False)
            self.is_armed = False
            time.sleep(0.2)
            return True
        except Exception as e:
            logger.error(f"DISARM failed: {e}")
            return False

    def _send_setpoint_impl(self, roll: float, pitch: float, yaw_rate: float, thrust: int) -> bool:
        if not self.is_connected or not getattr(self.connection, "scf", None):
            return False

        try:
            # Pitch is inverted in Crazyflie firmware
            self.connection.scf.cf.commander.send_setpoint(roll, -pitch, yaw_rate, thrust)
            return True
        except Exception as e:
            logger.error(f"Failed to send setpoint: {e}")
            return False

    def _send_hover_impl(self, vx: float, vy: float, yaw_rate: float, z: float) -> bool:
        if not self.is_connected or not getattr(self.connection, "scf", None):
            return False

        try:
            self.connection.scf.cf.commander.send_hover_setpoint(vx, vy, yaw_rate, z)
            return True
        except Exception as e:
            logger.error(f"Failed to send hover setpoint: {e}")
            return False
