"""
Sine sweep test strategy for the Crazyflie Sweeper package.

Implements a sine sweep test strategy that applies a sinusoidal input with increasing
frequency on the specified channel (roll, pitch, yaw rate, or thrust) and measures the response.
"""
import logging
import math
import time
from typing import Dict, Optional, Tuple

from ..core.config import FlightConfig
from ..controllers.base import FlightController
from ..core.safety import SafetyMonitor
from ..core.logger import FlightLogger
from .base import TestStrategy

logger = logging.getLogger(__name__)


class SineSweepTest(TestStrategy):
    """
    Sine sweep test strategy implementation.

    Applies a sine wave input with increasing frequency on the specified channel
    (roll, pitch, yaw rate, or thrust) to measure frequency response.
    """

    def __init__(self,
                controller: FlightController,
                config: FlightConfig,
                safety_monitor: SafetyMonitor,
                logger_instance: FlightLogger,
                channel: str,
                amplitude: float,
                duration: float,
                start_freq: float,
                end_freq: float):
        """
        Initialize the sine sweep test strategy.

        Args:
            controller: Flight controller for sending commands
            config: Configuration parameters
            safety_monitor: Safety monitor for checking safety limits
            logger_instance: Logging manager for recording test data
            channel: The channel to test ('roll', 'pitch', or 'thrust')
            amplitude: Amplitude of the sine wave (degrees for roll/pitch, thrust units for thrust)
            duration: Duration of the sweep (if None, uses config.sine_sweep_duration)
            start_freq: Starting frequency in Hz (if None, uses config.sine_start_freq)
            end_freq: Ending frequency in Hz (if None, uses config.sine_end_freq)
        """
        super().__init__(controller, config, safety_monitor, logger_instance)

        self.channel = channel
        self.amplitude = amplitude
        self.duration = duration
        self.start_freq = start_freq
        self.end_freq = end_freq

        # Set test name and description
        self.test_name = f"sine_sweep_{channel}_{amplitude}"
        self.test_description = (f"Sine Sweep Test: {channel.capitalize()} with amplitude {amplitude}, "
                               f"frequency {self.start_freq}Hz to {self.end_freq}Hz")

    def _execute_test(self) -> bool:
        """
        Execute the sine sweep test.

        Returns:
            True if test completed successfully, False otherwise
        """
        # Calculate sweep rate (logarithmic sweep)
        sweep_rate = math.log(self.end_freq / self.start_freq) / self.duration

        logger.info(f"Starting sine sweep from {self.start_freq}Hz to {self.end_freq}Hz "
                  f"over {self.duration}s with amplitude {self.amplitude}")

        # Send the sine sweep commands continuously for the duration
        start_time = time.time()
        next_tick = start_time
        test_successful = True

        while time.time() - start_time < self.duration:
            # Check safety limits before sending command
            if not self.check_safety():
                logger.warning(f"Sine sweep test on {self.channel} interrupted due to safety concerns!")
                test_successful = False
                break

            # Calculate elapsed time
            elapsed = time.time() - start_time

            # Calculate current frequency using logarithmic sweep
            current_freq = self.start_freq * math.exp(sweep_rate * elapsed)

            # Calculate sine wave value at this time
            sine_value = self.amplitude * math.sin(2 * math.pi * current_freq * elapsed)

            # Default values
            roll, pitch, thrust_offset = 0, 0, 0
            yaw_rate_val = self.config.default_yaw_rate

            # Apply sine wave to the specified channel
            if self.channel == "roll":
                roll = sine_value
            elif self.channel == "pitch":
                pitch = sine_value
            elif self.channel == "yaw":
                # Apply sine to yaw rate (deg/s)
                yaw_rate_val = float(sine_value)
            elif self.channel == "thrust":
                thrust_offset = sine_value

            # Calculate target thrust
            target_thrust = self.config.hover_thrust + thrust_offset

            # Send the command and log it
            self.logger.log_command(roll, pitch, yaw_rate_val, target_thrust)
            self.controller.send_setpoint(
                roll, pitch, yaw_rate_val, target_thrust
            )

            # Log progress periodically
            if elapsed % 1.0 < 0.02:  # Every second
                logger.debug(f"Sweep progress: {elapsed:.1f}s / {self.duration:.1f}s, "
                           f"freq: {current_freq:.2f}Hz")

            # Maintain controller control rate
            next_tick += self.controller.CONTROL_PERIOD
            sleep_time = next_tick - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.time()

        return test_successful
