"""
Impulse test strategy for the Crazyflie Sweeper package.

Implements an impulse test strategy that applies a short impulse on the specified channel
(roll, pitch, or thrust) and measures the response.
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


class ImpulseTest(TestStrategy):
    """
    Impulse test strategy implementation.
    
    Applies a brief impulse input on the specified channel (roll, pitch, or thrust) 
    and measures the response.
    """
    
    def __init__(self, 
                controller: FlightController, 
                config: FlightConfig,
                safety_monitor: SafetyMonitor,
                logging_manager: LoggingManager,
                channel: str,
                amplitude: float,
                duration: Optional[float] = None,
                recovery_duration: Optional[float] = None):
        """
        Initialize the impulse test strategy.
        
        Args:
            controller: Flight controller for sending commands
            config: Configuration parameters
            safety_monitor: Safety monitor for checking safety limits
            logging_manager: Logging manager for recording test data
            channel: The channel to test ('roll', 'pitch', or 'thrust')
            amplitude: Amplitude of the impulse input (degrees for roll/pitch, thrust units for thrust)
            duration: Duration of the impulse input (if None, uses config.impulse_duration)
            recovery_duration: Duration to measure recovery after impulse 
                             (if None, uses config.hold_neutral_duration_s)
        """
        super().__init__(controller, config, safety_monitor, logging_manager)
        
        self.channel = channel
        self.amplitude = amplitude
        self.duration = duration if duration is not None else config.impulse_duration
        self.recovery_duration = recovery_duration if recovery_duration is not None else config.hold_neutral_duration_s
        
        # Set test name and description
        self.test_name = f"impulse_{channel}_{amplitude}"
        self.test_description = f"Impulse Test: {channel.capitalize()} with amplitude {amplitude}"
        
    def _execute_test(self) -> bool:
        """
        Execute the impulse test.
        
        Returns:
            True if test completed successfully, False otherwise
        """
        # Default values
        roll, pitch, thrust_offset = 0, 0, 0
        
        # Configure impulse input for the specified channel
        if self.channel == "roll":
            roll = self.amplitude
        elif self.channel == "pitch":
            pitch = self.amplitude
        elif self.channel == "thrust":
            thrust_offset = self.amplitude
        else:
            logger.error(f"Invalid channel: {self.channel}")
            return False
        
        # Calculate target thrust
        target_thrust = self.config.hover_thrust + thrust_offset
        
        logger.info(f"Applying impulse: Roll: {roll}°, Pitch: {pitch}°, "
                  f"Thrust: {target_thrust} for {self.duration}s")
        
        # Send the impulse commands continuously for the impulse duration
        start_time = time.time()
        test_successful = True
        
        # Phase 1: Apply impulse
        while time.time() - start_time < self.duration:
            # Check safety limits before sending command
            if not self.check_safety():
                logger.warning(f"Impulse test on {self.channel} interrupted due to safety concerns!")
                test_successful = False
                break
            
            # Send the impulse command and log it
            self.controller.send_setpoint(roll, pitch, self.config.default_yaw_rate, target_thrust, 
                                         logging_manager=self.logging_manager)
            time.sleep(0.02)  # 50Hz control rate
        
        # Phase 2: Return to neutral and measure recovery if phase 1 was successful
        if test_successful:
            logger.info(f"Impulse complete. Measuring recovery for {self.recovery_duration}s")
            
            # Update the logging maneuver to indicate recovery phase
            self.logging_manager.set_current_maneuver(f"{self.test_name}_recovery")
            
            recovery_start_time = time.time()
            
            while time.time() - recovery_start_time < self.recovery_duration:
                # Check safety limits before sending command
                if not self.check_safety():
                    logger.warning(f"Recovery phase interrupted due to safety concerns!")
                    test_successful = False
                    break
                
                # Send neutral commands during recovery and log them
                self.controller.send_setpoint(0, 0, self.config.default_yaw_rate, self.config.hover_thrust, 
                                             logging_manager=self.logging_manager)
                time.sleep(0.02)  # 50Hz control rate
        
        return test_successful 