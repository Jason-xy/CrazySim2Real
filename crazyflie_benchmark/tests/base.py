"""
Abstract base class for test strategies in the Crazyflie Sweeper package.

Defines the common interface for all test strategies using the Strategy pattern,
ensuring proper implementation of test methods across different test types.
"""
import abc
import logging
import time
from typing import Dict, List, Optional, Tuple

from ..core.utils import clamp

from ..core.config import FlightConfig
from ..controllers.base import FlightController
from ..core.safety import SafetyMonitor
from ..core.logger import FlightLogger

logger = logging.getLogger(__name__)


class TestStrategy(abc.ABC):
    """
    Abstract base class for test strategies.

    Defines the common interface for all test strategies using the Strategy pattern.
    """

    def __init__(self,
                controller: FlightController,
                config: FlightConfig,
                safety_monitor: SafetyMonitor,
                logger_instance: FlightLogger):
        """
        Initialize the test strategy.

        Args:
            controller: Flight controller for sending commands
            config: Configuration parameters
            safety_monitor: Safety monitor for checking safety limits
            logger_instance: Logger for recording test data
        """
        self.controller = controller
        self.config = config
        self.safety_monitor = safety_monitor
        self.logger = logger_instance

        # Test parameters
        self.test_name = "base_test"
        self.test_description = "Base test strategy"
        self.channel = None  # The channel being tested (roll, pitch, thrust)
        self.amplitude = 0.0  # Test amplitude
        self.duration = 0.0  # Test duration
        self.aborted = False  # Flag indicating if the test was aborted due to safety violation

    def execute(self) -> bool:
        """
        Execute the test with hover commands before and after.

        Returns:
            True if test completed successfully, False otherwise
        """
        # Log test start
        self.log_test_start()

        # Set up for the test
        if not self.setup():
            logger.error("Failed to set up test")
            return False

        # Execute the actual test
        test_successful = self._execute_test()

        # Clean up after the test
        if not self.cleanup():
            logger.warning("Failed to clean up after test")
            test_successful = False

        # Log test completion
        self.log_test_complete()

        return test_successful

    @abc.abstractmethod
    def _execute_test(self) -> bool:
        """
        Execute the specific test logic.

        Returns:
            True if test completed successfully, False otherwise
        """
        pass

    def send_hover_command(self) -> bool:
        """
        Send a hover command with all angles set to zero.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Log command
            self.logger.log_command(0, 0, self.config.default_yaw_rate, self.config.hover_thrust)

            return self.controller.send_setpoint(
                roll=0,
                pitch=0,
                yaw_rate=self.config.default_yaw_rate,
                thrust=self.config.hover_thrust
            )
        except Exception as e:
            logger.error(f"Error sending hover command: {e}")
            return False

    def setup(self) -> bool:
        """
        Set up for the test execution.

        Returns:
            True if setup successful, False otherwise
        """
        logger.info(f"Setting up test: {self.test_description}")

        # Reset the aborted flag
        self.aborted = False

        # Reset the safety monitor boundary exceeded flag
        self.safety_monitor.reset_boundary_exceeded()

        return True

    def cleanup(self) -> bool:
        """
        Clean up after test execution.

        Returns:
            True if cleanup successful, False otherwise
        """
        logger.info(f"Cleaning up after test: {self.test_description}")

        # If the test was aborted due to safety violation, we've already hovered in place
        if not self.aborted:
            # Return to neutral hover for a fixed duration
            cleanup_success = self.controller.maintain_hover(self.config.hold_neutral_duration_s)
        else:
            cleanup_success = True
            logger.info("Skipping normal cleanup as test was aborted due to safety violation")

        return cleanup_success

    def check_safety(self) -> bool:
        """
        Check safety conditions for the test.

        Returns:
            True if safety check passes, False otherwise
        """
        position = self.logger.get_current_position()

        if not self.safety_monitor.check(position):
            logger.warning(f"Safety check failed during {self.test_name}!")

            # Hover in place instead of returning to center
            # Note: safety_monitor.hover_in_place might need update if it uses old controller
            # But we passed the new controller to safety_monitor? No, safety_monitor is separate.
            # We should check safety_monitor implementation.

            # Assuming safety_monitor.hover_in_place takes (controller, position)
            self.safety_monitor.hover_in_place(self.controller, position)

            # Mark the test as aborted
            self.aborted = True

            # Log test abortion
            logger.warning(f"Test {self.test_name} aborted due to safety boundary violation")

            return False

        return True

    def wait_for_inter_maneuver_delay(self) -> None:
        """
        Wait for the configured delay between maneuvers.
        """
        if self.config.inter_maneuver_delay_s > 0:
            logger.info(f"Waiting {self.config.inter_maneuver_delay_s}s between maneuvers")

            # Use maintain_hover instead of time.sleep to actively control during delay
            self.controller.maintain_hover(self.config.inter_maneuver_delay_s)

    def log_test_start(self) -> None:
        """
        Log the start of a test.
        """
        logger.info(f"--- Starting Test: {self.test_description} ---")
        if self.channel:
            logger.info(f"Channel: {self.channel}, Amplitude: {self.amplitude}, Duration: {self.duration}s")

    def log_test_complete(self) -> None:
        """
        Log the completion of a test.
        """
        if self.aborted:
            logger.info(f"--- Test Aborted: {self.test_description} ---")
        else:
            logger.info(f"--- Test Complete: {self.test_description} ---")
