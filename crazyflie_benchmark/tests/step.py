"""
Step test strategy for the Crazyflie Sweeper package.

Implements a step test strategy that applies a step input on the specified channel
(roll, pitch, yaw rate, or thrust) and measures the response.
"""
import logging
import time
from typing import Dict, Optional, Tuple

from ..config import FlightConfig
from ..controller import FlightController
from ..safety import SafetyMonitor
from ..logging_manager import LoggingManager
from .base import TestStrategy

logger = logging.getLogger(__name__)


class StepTest(TestStrategy):
    """
    Step test strategy implementation.

    Applies a step input on the specified channel (roll, pitch, yaw rate, or thrust) and
    measures the response.
    """

    def __init__(self,
                controller: FlightController,
                config: FlightConfig,
                safety_monitor: SafetyMonitor,
                logging_manager: LoggingManager,
                channel: str,
                amplitude: float,
                duration: Optional[float] = None):
        """
        Initialize the step test strategy.

        Args:
            controller: Flight controller for sending commands
            config: Configuration parameters
            safety_monitor: Safety monitor for checking safety limits
            logging_manager: Logging manager for recording test data
            channel: The channel to test ('roll', 'pitch', or 'thrust')
            amplitude: Amplitude of the step input (degrees for roll/pitch, thrust units for thrust)
            duration: Duration of the step input (if None, uses config.step_test_duration)
        """
        super().__init__(controller, config, safety_monitor, logging_manager)

        self.channel = channel
        self.amplitude = amplitude
        self.duration = duration if duration is not None else config.step_test_duration

        # Set test name and description
        self.test_name = f"step_{channel}_{amplitude}"
        self.test_description = f"Step Test: {channel.capitalize()} with amplitude {amplitude}"

    def _execute_test(self) -> bool:
        """
        Execute the step test.

        Returns:
            True if test completed successfully, False otherwise
        """
        # Default values
        roll, pitch, thrust_offset = 0, 0, 0
        yaw_rate_val = self.config.default_yaw_rate

        # Configure step input for the specified channel
        if self.channel == "roll":
            roll = self.amplitude
        elif self.channel == "pitch":
            pitch = self.amplitude
        elif self.channel == "yaw":
            yaw_rate_val = self.amplitude
        elif self.channel == "thrust":
            thrust_offset = self.amplitude
        else:
            logger.error(f"Invalid channel: {self.channel}")
            return False

        # Calculate target thrust
        target_thrust = self.config.hover_thrust + thrust_offset

        logger.info(
            f"Applying step input: Roll: {roll}°, Pitch: {pitch}°, "
            f"YawRate: {yaw_rate_val}°/s, Thrust: {target_thrust} for {self.duration}s"
        )

        # Send the step input commands continuously for the duration
        start_time = time.time()
        test_successful = True

        while time.time() - start_time < self.duration:
            # Check safety limits before sending command
            if not self.check_safety():
                logger.warning(
                    f"Step test on {self.channel} interrupted due to safety concerns!"
                )
                test_successful = False
                break

            # Send the step command and log it via the logging_manager
            self.controller.send_setpoint(
                roll, pitch, yaw_rate_val, target_thrust, logging_manager=self.logging_manager
            )
            logger.info(
                f"Sent setpoint: Roll: {roll}°, Pitch: {pitch}°, "
                f"YawRate: {yaw_rate_val}°/s, Thrust: {target_thrust}"
            )
            time.sleep(0.02)  # 50Hz control rate

        return test_successful