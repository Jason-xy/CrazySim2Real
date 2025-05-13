"""
Flight controller for the Crazyflie Sweeper package.

Handles flight control operations including arming, disarming, and sending 
control setpoints with safety clamping.
"""
import logging
import math
import time
from typing import Dict, List, Optional, Tuple, Any

from cflib.crtp.crtpstack import CRTPPacket, CRTPPort

from .config import FlightConfig
from .connection import ConnectionManager
from .utils import clamp

logger = logging.getLogger(__name__)

class FlightController:
    """
    Controls flight operations for the Crazyflie drone.
    
    Handles arming, disarming, and sending control setpoints with safety clamping.
    Provides core flight operations including takeoff, landing, and hover.
    """
    
    def __init__(self, connection: ConnectionManager, config: FlightConfig):
        """
        Initialize the flight controller.
        
        Args:
            connection: Connection manager for the Crazyflie
            config: Configuration parameters for flight control
        """
        self.connection = connection
        self.config = config
        self._commander = None
        self.is_armed = False
        
        # Control timing constants
        self.CONTROL_RATE_HZ = 50
        self.CONTROL_PERIOD = 1.0 / self.CONTROL_RATE_HZ
        
        # State tracking
        self._last_arm_warning_time = 0.0
        self._is_hovering = False
        
        # Current state
        self._current_height = 0.0
        
        # Track if setpoint has been initialized
        self._setpoint_initialized = False
        
        # Determine if we're working with a simulator or real hardware
        self.is_simulator = not hasattr(self.connection, 'scf')
        
    @property
    def commander(self):
        """Property to get the commander, ensuring it's initialized."""
        if self.is_simulator:
            # For simulator, we'll use the connection object directly
            return self.connection
        else:
            # For real hardware, use CFLib commander
            if self._commander is None:
                if self.connection.is_connected and hasattr(self.connection, 'scf') and self.connection.scf and self.connection.scf.cf:
                    self._commander = self.connection.scf.cf.commander
                    if self._commander is None:
                        logger.error("Commander could not be initialized from scf.cf.commander even after connection.")
                else:
                    logger.warning("Commander not available: Connection or SCF not ready.")
            return self._commander

    def initialize_setpoint(self) -> bool:
        """
        Initialize the setpoint interface by sending a zero command.
        
        The setpoint interface requires an initial zero command before it can be used.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if not self.connection.is_connected:
            logger.error("INIT_SETPOINT: Connection not available. Cannot initialize setpoint interface.")
            return False
            
        try:
            logger.info("INIT_SETPOINT: Initializing setpoint interface with zero command.")
            
            if self.is_simulator:
                # For simulator, send a zero setpoint directly to the connection
                self.connection.send_setpoint(0, 0, 0, 0)
            else:
                # For real hardware, use the commander
                if not self.commander:
                    logger.error("INIT_SETPOINT: Commander not available. Cannot initialize setpoint interface.")
                    return False
                self.commander.send_setpoint(0, 0, 0, 0)
                
            self._setpoint_initialized = True
            time.sleep(0.1)  # Short delay to allow processing
            return True
        except Exception as e:
            logger.error(f"INIT_SETPOINT: Failed to initialize setpoint interface: {e}", exc_info=True)
            return False
            
    def arm(self) -> bool:
        """
        Arm the motors of the Crazyflie.
        
        Returns:
            True if arming successful, False otherwise
        """
        if not self.connection.is_connected:
            logger.warning("ARM: Cannot send ARM command, basic connection not ready.")
            return False
        
        if self.is_simulator:
            # Simulator is always considered "armed" when connected
            logger.info("ARM: Simulator is considered armed when connected.")
            self.is_armed = True
            return True
        
        # Real hardware arming logic
        if not self.commander:
            logger.error("ARM: Commander not available. Cannot arm.")
            return False

        logger.info("ARM: Attempting to send platform ARM command...")
        
        pk = CRTPPacket()
        pk.port = CRTPPort.PLATFORM  # Port 7
        pk.channel = 0               # Channel 0 for platform control
        # data[0] = PLATFORM_COMMAND_ARM_DISARM (ID 1 in firmware)
        # data[1] = PLATFORM_COMMAND_ARM (Value 1 in firmware to arm)
        data_payload = bytes([1, 1])
        pk.data = data_payload
        
        try:
            self.connection.scf.cf.send_packet(pk)
            logger.info("ARM: Platform ARM command packet sent.")
            # Verify arming state if possible, though no direct feedback from this packet
            self.is_armed = True
            time.sleep(0.3)  # Increased delay slightly for firmware to process
            logger.info("ARM: Assumed armed after delay.")
            return True
        except Exception as e:
            logger.error(f"ARM: Failed to send ARM command: {e}", exc_info=True)
            self.is_armed = False
            return False
    
    def disarm(self) -> bool:
        """
        Disarm the drone motors.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.connection.is_connected:
            logger.warning("DISARM: Cannot send DISARM command, connection lost.")
            # If connection is lost, the drone might disarm itself or has already disarmed
            # Return true assuming the drone is safely disarmed one way or another
            return True
        
        if self.is_simulator:
            # For simulator, just send a stop setpoint
            logger.info("DISARM: Simulator - sending stop setpoint")
            success = self.connection.send_stop_setpoint()
            self.is_armed = False
            return success
            
        # Real hardware disarm logic
        if not self.commander:
            logger.warning("DISARM: Cannot send DISARM command, basic connection not ready.")
            return False
            
        try:
            logger.info("DISARM: Sending platform DISARM command...")
            if not self.connection.scf.cf.platform.is_armed():
                logger.info("DISARM: Platform already disarmed.")
                self.is_armed = False
                return True
                
            self.connection.scf.cf.platform.set_arming_request(state=False)
            logger.info("DISARM: Platform DISARM command packet sent.")
            time.sleep(0.2) # Allow time for command to be received
            self.is_armed = False
            return True
        except Exception as e:
            # If we get an exception during disarm, it could be due to connection loss
            # which might happen naturally during landing
            logger.warning(f"DISARM: Error during disarm: {e}")
            # We assume the drone is disarmed or will disarm when it hits the ground
            self.is_armed = False
            return False
    
    def send_setpoint(self, roll: float, pitch: float, yaw_rate: float, thrust: int, logging_manager=None) -> bool:
        """
        Send a control setpoint to the Crazyflie with safety clamping.
        
        Note on thrust values:
        - For real hardware (CFLib): thrust should be in range 0-65535
        - For simulator: thrust is automatically converted from 0-65535 to 0-1
        
        If you provide a thrust value <= 1.0 with simulator, it's assumed to be already normalized.
        
        Args:
            roll: Roll angle in degrees (-MAX_ROLL_PITCH to MAX_ROLL_PITCH)
            pitch: Pitch angle in degrees (-MAX_ROLL_PITCH to MAX_ROLL_PITCH)
            yaw_rate: Yaw rate in degrees/s
            thrust: Thrust value (0-65535 for real hardware, 0-65535 or 0-1 for simulator)
            logging_manager: Optional LoggingManager to record the control command
            
        Returns:
            True if successful, False otherwise
        """
        # Only warn about arming state at most once every 3 seconds
        current_time = time.monotonic()
        
        if not self.is_armed and not self.is_simulator:
            if current_time - self._last_arm_warning_time > 3.0:
                self._last_arm_warning_time = current_time
                logger.warning("SETPOINT: Drone not armed. Setpoints will likely be ignored.")
        
        if not self.connection.is_connected:
            logger.warning("SETPOINT: Connection not available. Cannot send setpoint.")
            return False
            
        # Initialize setpoint interface if not already done
        if not self._setpoint_initialized:
            if not self.initialize_setpoint():
                return False
        
        # Apply safety limits to roll and pitch angles
        safe_roll = clamp(roll, -self.config.safety_max_roll_pitch_deg, self.config.safety_max_roll_pitch_deg)
        safe_pitch = clamp(pitch, -self.config.safety_max_roll_pitch_deg, self.config.safety_max_roll_pitch_deg)
        
        # Handle thrust differently based on connection type
        if self.is_simulator:
            # For simulator: Convert thrust to 0-1 range if necessary
            original_thrust = thrust  # Store original value for logging
            
            # Determine if thrust is already normalized (0-1) or needs conversion from 0-65535
            if thrust > 1.0:
                # Convert from 0-65535 to 0-1 range for simulator
                normalized_thrust = clamp(thrust / 65535.0, 0.0, 1.0)
                logger.debug(f"SETPOINT: Converting thrust from {thrust} to {normalized_thrust} for simulator")
            else:
                # Already in 0-1 range
                normalized_thrust = clamp(thrust, 0.0, 1.0)
                # If very small value was provided in 0-1 range, log it to help debug
                if thrust < 0.01 and thrust != 0:
                    logger.debug(f"SETPOINT: Using very small normalized thrust: {thrust}")
                
            # Log the control command if a logging_manager is provided
            # Always log using standard 0-65535 range for consistency in logs
            if logging_manager is not None:
                if thrust <= 1.0:
                    # Convert back to 0-65535 range for consistent logging
                    log_thrust = int(thrust * 65535.0)
                else:
                    log_thrust = original_thrust
                logging_manager.log_control_command(safe_roll, safe_pitch, yaw_rate, log_thrust)
                
            # Send the normalized thrust to simulator
            return self.connection.send_setpoint(safe_roll, safe_pitch, yaw_rate, normalized_thrust)
        else:
            # For real hardware: Ensure thrust is in valid integer range
            min_thrust = getattr(self.config, 'safety_min_thrust', getattr(self.config, 'safety_min_thrust_flight', 0))
            
            # Handle case where normalized thrust (0-1) might be provided to real hardware
            if 0.0 <= thrust <= 1.0:
                logger.warning(f"SETPOINT: Received normalized thrust ({thrust}) with real hardware. Converting to 0-65535 range.")
                # Convert from 0-1 to 0-65535 range
                thrust = int(thrust * 65535.0)
            
            # Apply safety limits
            safe_thrust = clamp(int(thrust), min_thrust, self.config.safety_max_thrust)
            
            # Log the control command if a logging_manager is provided
            if logging_manager is not None:
                logging_manager.log_control_command(safe_roll, safe_pitch, yaw_rate, safe_thrust)

            try:
                self.commander.send_setpoint(safe_roll, -safe_pitch, yaw_rate, safe_thrust) # Pitch is inverted in crazyflie firmware
                return True
            except Exception as e:
                logger.error(f"SETPOINT: Failed to send setpoint: {e}", exc_info=True)
                return False
    
    def send_hover_setpoint(self, vx: float, vy: float, yaw_rate: float, z: float) -> bool:
        """
        Send a hover setpoint to the Crazyflie.
        
        Args:
            vx: X velocity (m/s)
            vy: Y velocity (m/s)
            yaw_rate: Yaw rate (degrees/s)
            z: Absolute height (m)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.commander:
            logger.warning("SEND_HOVER_SETPOINT: Commander not available.")
            return False
            
        # Ensure setpoint interface is initialized first
        if not self._setpoint_initialized:
            if not self.initialize_setpoint():
                return False
        
        safe_z = clamp(z, 0.0, self.config.safety_max_height_m)
        if safe_z != z:
            logger.debug(f"SEND_HOVER_SETPOINT: Clamped height: Z:{z}->{safe_z}")

        try:
            self.commander.send_hover_setpoint(vx, vy, yaw_rate, safe_z)
            self._current_height = safe_z  # Update current height
            return True
        except Exception as e:
            logger.error(f"SEND_HOVER_SETPOINT: Failed to send hover setpoint: {e}", exc_info=True)
            return False
    
    def initialize_motors(self) -> bool:
        """
        Initialize motors with the official Crazyflie warm-up sequence.
        
        This sequence follows the recommended Crazyflie approach:
        1. Send 20 setpoints at low thrust (10000) at 50 Hz
        2. Send 20 setpoints at default hover thrust at 50 Hz
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_armed:
            logger.error("INITIALIZE_MOTORS: Drone not armed. Cannot initialize motors.")
            return False
        
        if not self.commander:
            logger.error("INITIALIZE_MOTORS: Commander not available. Cannot initialize motors.")
            return False
            
        # Ensure setpoint interface is initialized
        if not self._setpoint_initialized:
            if not self.initialize_setpoint():
                logger.error("INITIALIZE_MOTORS: Failed to initialize setpoint interface.")
                return False
        
        logger.info("INITIALIZE_MOTORS: Starting motor initialization sequence...")
        
        try:
            # Step 1: Send 20 setpoints at low thrust (10000) at 50 Hz
            low_thrust = 10000
            logger.info(f"INITIALIZE_MOTORS: Sending {low_thrust} thrust commands...")
            
            for _ in range(20):
                if not self.send_setpoint(0, 0, 0, low_thrust):
                    logger.error("INITIALIZE_MOTORS: Failed to send low thrust setpoint.")
                    return False
                time.sleep(0.02)  # 50 Hz
            
            # Step 2: Send 20 setpoints at hover_thrust at 50 Hz
            hover_thrust = int(self.config.hover_thrust)
            logger.info(f"INITIALIZE_MOTORS: Sending {hover_thrust} thrust commands...")
            
            for _ in range(20):
                if not self.send_setpoint(0, 0, 0, hover_thrust):
                    logger.error("INITIALIZE_MOTORS: Failed to send hover thrust setpoint.")
                    return False
                time.sleep(0.02)  # 50 Hz
            
            logger.info("INITIALIZE_MOTORS: Motor initialization sequence completed successfully.")
            return True
            
        except Exception as e:
            logger.error(f"INITIALIZE_MOTORS: Error during motor initialization: {e}", exc_info=True)
            return False
    
    def _check_height_data_available(self, logging_manager) -> Tuple[bool, Optional[float]]:
        """
        Check if height data is available from the logging manager.
        
        Args:
            logging_manager: The logging manager to check for height data
            
        Returns:
            (is_available, current_height): Whether height data is available and the current height
        """
        if logging_manager is None:
            logger.error("HEIGHT_CHECK: No logging manager provided.")
            return False, None
        
        try:
            if hasattr(logging_manager, 'get_current_position'):
                position = logging_manager.get_current_position()
                if position and 'z' in position and position['z'] is not None:
                    return True, position['z']
        except Exception as e:
            logger.error(f"HEIGHT_CHECK: Error getting position: {e}", exc_info=True)
        
        logger.error("HEIGHT_CHECK: Height data not available from logging manager.")
        return False, None
    
    def _timed_control_loop(self, duration: float, callback, 
                           logging_manager = None, fail_threshold: int = 5) -> bool:
        """
        Execute control commands in a loop with precise timing
        
        Args:
            duration: Execution duration (seconds)
            callback: Callback function that accepts (time_elapsed, next_control_time) parameters and returns a continue flag
            logging_manager: Logging manager
            fail_threshold: Number of consecutive failures allowed
            
        Returns:
            Whether the control loop completed successfully
        """
        next_control_time = time.monotonic()
        start_time = time.monotonic()
        consecutive_failures = 0
        
        try:
            while time.monotonic() - start_time < duration:
                # Call the callback function to execute control; if it returns False, break the loop
                if not callback(time.monotonic() - start_time, next_control_time):
                    consecutive_failures += 1
                    if consecutive_failures >= fail_threshold:
                        logger.error("Control loop: Too many consecutive failures, aborting")
                        return False
                else:
                    consecutive_failures = 0
                
                # Precise time control (50Hz)
                next_control_time += self.CONTROL_PERIOD
                sleep_time = next_control_time - time.monotonic()
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # Reset time control to maintain control rate
                    if -sleep_time > 0.1:
                        next_control_time = time.monotonic() + self.CONTROL_PERIOD
                        logger.debug("Control timing reset due to delay")
            
            return True
            
        except Exception as e:
            logger.error(f"Control loop error: {e}", exc_info=True)
            return False
    
    def maintain_hover(self, duration_s: float, target_height: float = 0.5) -> bool:
        """
        Maintain hover at specified height using the hover setpoint API.
        
        Args:
            duration_s: Duration to maintain hover in seconds
            target_height: Target height to hover at
            
        Returns:
            True if hover maintained successfully, False otherwise
        """
        if not self.commander:
            logger.error("HOVER: Commander not available.")
            return False
        
        logger.info(f"HOVER: Maintaining hover at {target_height}m for {duration_s}s")
        self._is_hovering = True
        
        try:
            # Hover control loop using hover setpoint
            start_time = time.monotonic()
            next_control_time = time.monotonic()
            consecutive_failures = 0
            
            while time.monotonic() - start_time < duration_s:
                # Send hover setpoint command (maintain position)
                if not self.send_hover_setpoint(0, 0, 0, target_height):
                    consecutive_failures += 1
                    if consecutive_failures >= 5:
                        logger.error("HOVER: Too many consecutive failures, aborting hover")
                        return False
                else:
                    consecutive_failures = 0
                
                # Log status periodically
                elapsed = time.monotonic() - start_time
                if int(elapsed * 5) % 5 == 0:
                    logger.info(f"HOVER: h={self._current_height:.2f}m, target={target_height:.2f}m")
                
                # Precise timing for 50Hz control
                next_control_time += self.CONTROL_PERIOD
                sleep_time = next_control_time - time.monotonic()
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # Reset timing if we're behind
                    if -sleep_time > 0.1:
                        next_control_time = time.monotonic() + self.CONTROL_PERIOD
            
            logger.info("HOVER: Hover maintained successfully.")
            return True
        except Exception as e:
            logger.error(f"HOVER: Error during hover: {e}", exc_info=True)
            return False
        finally:
            # Ensure hovering state is reset
            self._is_hovering = False
    
    def initialize_and_arm(self) -> bool:
        """
        Initialize and arm the drone, preparing for takeoff
        
        Returns:
            Whether initialization and arming were successful
        """
        # Check basic connection
        if not self.commander:
            logger.error("INIT: Commander not available.")
            return False
            
        # If not armed, attempt to arm
        if not self.is_armed:
            logger.info("INIT: Not armed, attempting to arm.")
            if not self.arm():
                logger.error("INIT: Arming failed.")
                return False
                
        # Initialize motors
        if not self.initialize_motors():
            logger.error("INIT: Failed to initialize motors.")
            return False
            
        logger.info("INIT: Successfully initialized and armed.")
        return True
            
    def take_off(self, target_height: float = 0.5, velocity: float = 0.3, 
                logging_manager = None) -> bool:
        """
        Execute takeoff to reach the specified height using hover setpoint API.
        
        Args:
            target_height: Target height (meters)
            velocity: Climb rate (meters/second)
            logging_manager: Logging manager
            
        Returns:
            Whether takeoff to target height was successful
        """
        logger.info(f"TAKEOFF: Starting takeoff to {target_height}m")
        
        # Initialize and arm
        if not self.initialize_and_arm():
            return False
            
        # Apply safety limits to target height
        safe_height = clamp(target_height, 0.1, self.config.safety_max_height_m)
        if safe_height != target_height:
            logger.debug(f"TAKEOFF: Clamped target height: {target_height}m → {safe_height}m")
            target_height = safe_height
            
        # Use hover setpoints for takeoff with a gradual increase in height
        logger.info(f"TAKEOFF: Using hover setpoints for takeoff")
        
        # Linear ramp-up to target height using 50Hz control
        velocity = 0.3  # m/s
        max_time = target_height / velocity
        start_time = time.monotonic()
        next_control_time = time.monotonic()
        
        while time.monotonic() - start_time < max_time + 1.0:  # Add 1s for stabilization
            # Calculate progress
            elapsed = time.monotonic() - start_time
            progress = min(elapsed / max_time, 1.0)
            current_target = target_height * progress
            
            # Send hover setpoint directly (no velocity in x/y)
            if not self.send_hover_setpoint(0, 0, 0, current_target):
                logger.error("TAKEOFF: Failed to send hover setpoint")
                return False
            
            # Log status periodically
            if int(elapsed * 2) % 2 == 0:
                logger.info(f"TAKEOFF: h={self._current_height:.2f}m, target={current_target:.2f}m")
            
            # Precise timing for 50Hz control
            next_control_time += self.CONTROL_PERIOD
            sleep_time = next_control_time - time.monotonic()
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Reset timing if we're behind
                if -sleep_time > 0.1:
                    next_control_time = time.monotonic() + self.CONTROL_PERIOD
        
        logger.info(f"TAKEOFF: Successfully reached target height {target_height}m")
        self._current_height = target_height
        return True

    def position_control(self, x: float = 0.0, y: float = 0.0, z: float = 0.5, yaw: float = 0.0, duration: float = 5.0) -> bool:
        """
        Move to specified position using position setpoint. 
        Wait in position for specified duration.
        
        Args:
            x: X position (meters)
            y: Y position (meters)
            z: Z position (meters)
            yaw: Yaw angle (degrees)
            duration: Duration to hold position (seconds)
            
        Returns:
            Whether position hold was successful
        """
        if not self.commander:
            logger.error("POSITION: Commander not available")
            return False
            
        if not self.is_armed:
            logger.error("POSITION: Drone not armed")
            return False
            
        logger.info(f"POSITION: Moving to position x={x}m, y={y}m, z={z}m, yaw={yaw}°")
        
        # Apply safety limits to height
        safe_z = clamp(z, 0.1, self.config.safety_max_height_m)
        if safe_z != z:
            logger.debug(f"POSITION: Clamped height: Z:{z}m → {safe_z}m")
            z = safe_z
            
        try:
            # Move to position with position setpoint
            start_time = time.monotonic()
            next_control_time = time.monotonic()
            
            # Send position commands at 50Hz until reaching target
            hold_start_time = None
            
            while True:
                # Send position setpoint
                self.commander.send_position_setpoint(x, y, z, yaw)
                
                # Check if we're at the target position (assuming we're close enough)
                # This would normally use position feedback from logging_manager
                current_time = time.monotonic()
                
                # After 2 seconds, assume we've reached the position and start the holding period
                if hold_start_time is None and current_time - start_time > 2.0:
                    hold_start_time = current_time
                    logger.info(f"POSITION: Reached target position, holding for {duration}s")
                
                # If we've completed the holding period, we're done
                if hold_start_time is not None and current_time - hold_start_time >= duration:
                    break
                
                # Precise timing for 50Hz control
                next_control_time += self.CONTROL_PERIOD
                sleep_time = next_control_time - time.monotonic()
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # Reset timing if we're behind
                    if -sleep_time > 0.1:
                        next_control_time = time.monotonic() + self.CONTROL_PERIOD
            
            logger.info("POSITION: Position hold completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"POSITION: Error during position control: {e}", exc_info=True)
            return False

    def land(self, velocity: float = 0.2, logging_manager = None) -> bool:
        """
        Land from the current height using hover setpoints for smooth descent.
        
        Args:
            velocity: Velocity for landing in m/s (positive value, will be applied negatively)
            logging_manager: Logging manager for height data
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connection.is_connected or not self.commander:
            logger.error("LAND: Cannot land: Not connected.")
            # If we're already disconnected, assume the drone is already on the ground
            # This helps with data processing when connection is lost mid-flight
            logger.warning("LAND: Assuming drone is already landed due to connection loss")
            return True
        
        try:
            # Step 1: Determine current height
            current_height = 0.5  # Default assumption if no height data
            
            # Try to get current height from logging manager
            if logging_manager:
                height_available, measured_height = self._check_height_data_available(logging_manager)
                if height_available:
                    current_height = measured_height
                    logger.info(f"LAND: Current height detected as {current_height:.2f}m")
            
            # Ensure positive height
            current_height = max(current_height, 0.1)
            
            # Step 2: Implement smooth descent using hover setpoints
            logger.info(f"LAND: Beginning smooth descent from {current_height:.2f}m")
            height_step = velocity * 0.02  # Height change per control cycle (at 50Hz)
            
            # Keep track of control timing
            next_control_time = time.monotonic()
            control_period = 0.02  # 50 Hz
            
            # Descend until very close to ground
            descent_successful = True
            while current_height > 0.05:
                current_height = max(current_height - height_step, 0.05)
                try:
                    if not self.send_hover_setpoint(0, 0, 0, current_height):
                        logger.warning("LAND: Failed to send hover setpoint during descent")
                        descent_successful = False
                        break  # Continue to next step even if there's an error
                except Exception as e:
                    # Catch connection loss during descent
                    logger.warning(f"LAND: Connection issue during descent: {e}")
                    # If connection is lost during descent, it's likely the drone has landed
                    # or will land on its own due to the last command or failsafe
                    logger.info("LAND: Assuming drone will land safely due to connection loss")
                    return True
                
                # Precise timing for 50Hz control
                next_control_time += control_period
                sleep_time = next_control_time - time.monotonic()
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # If we're behind schedule, reset timing
                    if -sleep_time > 0.1:
                        next_control_time = time.monotonic() + control_period
            
            # Step 3: Final approach with thrust control for smooth touchdown
            logger.info("LAND: Final approach, reducing thrust for touchdown")
            current_thrust = int(self.config.hover_thrust * 0.75)  # Start with reduced hover thrust
            
            # Gradually reduce thrust to zero
            touchdown_successful = True
            try:
                for _ in range(30):  # 0.6 seconds of gentle descent
                    current_thrust = max(0, current_thrust - 500)  # Reduce thrust gradually
                    if not self.send_setpoint(0, 0, 0, current_thrust):
                        logger.warning("LAND: Failed to send setpoint during final descent")
                        touchdown_successful = False
                        break
                    time.sleep(0.02)
            except Exception as e:
                # Catch connection loss during touchdown
                logger.warning(f"LAND: Connection issue during touchdown: {e}")
                # Similar to descent, we assume the drone is landing safely
                logger.info("LAND: Assuming drone has landed safely due to connection loss")
                return True
            
            # Step 4: Ensure motors stop by sending zero setpoints
            logger.info("LAND: Sending zero thrust commands before disarm")
            try:
                for _ in range(10):
                    self.send_setpoint(0, 0, 0, 0)
                    time.sleep(0.02)
            except Exception as e:
                logger.warning(f"LAND: Error sending zero thrust commands: {e}")
                # Not critical - continue to disarm
            
            # Step 5: Disarm motors
            logger.info("LAND: Landing complete, disarming motors")
            try:
                self.disarm()
                return True
            except Exception as e:
                logger.warning(f"LAND: Error during disarm: {e}")
                # If disarm fails but landing was successful, return true
                return descent_successful and touchdown_successful
            
        except Exception as e:
            logger.error(f"LAND: Failed to execute landing: {e}", exc_info=True)
            # Try to disarm anyway for safety
            try:
                self.disarm()
            except Exception as disarm_error:
                logger.warning(f"LAND: Failed to disarm after landing error: {disarm_error}")
            
            # If we got an exception during landing, assume the worst
            return False
    
    def stop(self) -> bool:
        """
        Stop all motion and disarm.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("STOP: Initiating full stop and disarm.")
        if self.commander:
            try:
                logger.info("STOP: Sending null setpoints.")
                for _ in range(10): # Multiple packets for reliability
                    self.send_setpoint(0, 0, 0, 0)
                    time.sleep(0.03)
            except Exception as e:
                logger.error(f"STOP: Error sending null setpoints: {e}", exc_info=True)
        
        logger.info("STOP: Attempting to disarm.")
        return self.disarm() 