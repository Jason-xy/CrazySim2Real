"""
Abstract base class for test strategies in the Crazyflie Sweeper package.

Defines the common interface for all test strategies using the Strategy pattern,
ensuring proper implementation of test methods across different test types.
"""
import abc
import logging
import time
from typing import Dict, List, Optional, Tuple

from ..config import FlightConfig
from ..controller import FlightController
from ..safety import SafetyMonitor
from ..logging_manager import LoggingManager

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
                logging_manager: LoggingManager):
        """
        Initialize the test strategy.
        
        Args:
            controller: Flight controller for sending commands
            config: Configuration parameters
            safety_monitor: Safety monitor for checking safety limits
            logging_manager: Logging manager for recording test data
        """
        self.controller = controller
        self.config = config
        self.safety_monitor = safety_monitor
        self.logging_manager = logging_manager
        
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
        
        Each test follows this pattern:
        1. Start with a hover command (all angles at 0)
        2. Execute the actual test via _execute_test method
        3. End with a hover command (all angles at 0)
        
        Returns:
            True if test completed successfully, False otherwise
        """
        # Log test start
        self.log_test_start()
        
        # Set up for the test
        if not self.setup():
            logger.error("Failed to set up test")
            return False
        
        # Send initial hover command
        logger.info(f"Sending initial hover command before starting {self.test_name}")
        if not self.send_hover_command():
            logger.warning("Failed to send initial hover command")
        
        # Execute the actual test
        test_successful = self._execute_test()
        
        # Send final hover command
        logger.info(f"Sending final hover command after completing {self.test_name}")
        if not self.send_hover_command():
            logger.warning("Failed to send final hover command")
            # Don't fail the test just because of hover command failure
        
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
        
        This method should be implemented by each test strategy subclass.
        
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
            return self.controller.send_setpoint(
                roll=0,
                pitch=0,
                yaw_rate=self.config.default_yaw_rate,
                thrust=self.config.hover_thrust,
                logging_manager=self.logging_manager
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
        
        # Set the current maneuver for logging
        self.logging_manager.set_current_maneuver(self.test_name)
        
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
        
        # Set the current maneuver to a return-to-neutral phase
        self.logging_manager.set_current_maneuver(f"{self.test_name}_return_to_neutral")
        
        return cleanup_success
    
    def check_safety(self) -> bool:
        """
        Check safety conditions for the test.
        
        Returns:
            True if safety check passes, False otherwise
        """
        position = self.logging_manager.get_current_position()
        
        if not self.safety_monitor.check(position):
            logger.warning(f"Safety check failed during {self.test_name}!")
            
            # Hover in place instead of returning to center
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