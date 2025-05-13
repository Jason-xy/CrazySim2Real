"""
Safety monitor for the Crazyflie Sweeper package.

Monitors drone position and motion to ensure safety limits are not exceeded.
Provides functions to check limits and trigger emergency return-to-base procedures.
"""
import logging
import math
import time
from typing import Dict, Optional, Tuple

from .config import FlightConfig
from .utils import horizontal_distance

logger = logging.getLogger(__name__)


class SafetyMonitor:
    """
    Monitors drone position and ensures safety limits are not exceeded.
    
    Checks for limits on horizontal distance, height, and provides
    emergency procedures if limits are exceeded.
    """
    
    def __init__(self, config: FlightConfig):
        """
        Initialize the safety monitor.
        
        Args:
            config: Configuration parameters including safety limits
        """
        self.config = config
        self.takeoff_position: Dict[str, float] = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.position_recorded = False
        self.safety_enabled = True
        self.boundary_exceeded = False
        
    def record_takeoff_position(self, position: Dict[str, float]) -> None:
        """
        Record the takeoff position as reference for safety checks.
        
        Args:
            position: Current position with x, y, z coordinates
        """
        self.takeoff_position = position.copy()
        self.position_recorded = True
        logger.info(f"Takeoff position recorded: "
                   f"({self.takeoff_position['x']:.2f}, "
                   f"{self.takeoff_position['y']:.2f}, "
                   f"{self.takeoff_position['z']:.2f})")
        
    def enable_safety_checks(self, enabled: bool = True) -> None:
        """
        Enable or disable safety checks.
        
        Args:
            enabled: True to enable safety checks, False to disable
        """
        self.safety_enabled = enabled
        logger.info(f"Safety checks {'enabled' if enabled else 'disabled'}")
        
    def check(self, position: Dict[str, float]) -> bool:
        """
        Check if the current position is within safety limits.
        
        Args:
            position: Current position with x, y, z coordinates
            
        Returns:
            True if position is safe, False if safety limits are exceeded
        """
        if not self.safety_enabled:
            return True
            
        if not self.position_recorded:
            # Can't check distance if we don't have a reference point
            return True
            
        # Check horizontal distance from takeoff point
        h_dist = horizontal_distance(position, self.takeoff_position)
        
        # Check vertical distance (height)
        height = position['z']
        relative_height = height - self.takeoff_position['z']
        
        # Determine if limits are exceeded
        is_outside_safety_radius = h_dist > self.config.safety_radius_m
        is_above_max_height = height > self.config.safety_max_height_m
        
        # Early warning for approaching limits (80% of max)
        if not is_outside_safety_radius and h_dist > self.config.safety_radius_m * 0.8:
            logger.warning(f"Approaching safety radius limit: {h_dist:.2f}m / {self.config.safety_radius_m}m")
            
        if not is_above_max_height and height > self.config.safety_max_height_m * 0.85:
            logger.warning(f"Approaching maximum height: {height:.2f}m / {self.config.safety_max_height_m}m")
        
        # If any limit is exceeded, log the violation and return False
        if is_outside_safety_radius or is_above_max_height:
            if is_outside_safety_radius:
                logger.warning(f"SAFETY ALERT: Drone outside safety radius! "
                             f"Distance: {h_dist:.2f}m > {self.config.safety_radius_m}m")
            
            if is_above_max_height:
                logger.warning(f"SAFETY ALERT: Drone above maximum height! "
                             f"Height: {height:.2f}m > {self.config.safety_max_height_m}m")
                
            logger.warning(f"Current position: ({position['x']:.2f}, {position['y']:.2f}, {position['z']:.2f})")
            logger.warning(f"Takeoff position: ({self.takeoff_position['x']:.2f}, "
                         f"{self.takeoff_position['y']:.2f}, {self.takeoff_position['z']:.2f})")
            logger.warning(f"Relative height: {relative_height:.2f}m")
            
            self.boundary_exceeded = True
            return False
            
        return True
    
    def hover_in_place(self, controller, position: Dict[str, float], duration: float = 3.0) -> bool:
        """
        Trigger an emergency hover in place when safety boundaries are exceeded.
        This stops the current test and keeps the drone stable at the current position.
        
        Args:
            controller: FlightController instance to send commands
            position: Current position with x, y, z coordinates
            duration: Duration to maintain hover in seconds
            
        Returns:
            True if hover procedure completed, False if it failed
        """
        logger.warning("SAFETY ALERT: Boundary exceeded - Initiating hover in place")
        
        try:
            # Clamp height if we're above max height
            safe_height = min(position['z'], self.config.safety_max_height_m * 0.95)
            
            # Log the emergency hover action
            logger.warning(f"Emergency hover at position: ({position['x']:.2f}, {position['y']:.2f}, {safe_height:.2f})")
            
            # Use hover setpoint for stable hover (maintaining XY position but with safe height)
            start_time = time.time()
            
            while time.time() - start_time < duration:
                controller.send_hover_setpoint(0, 0, 0, safe_height)
                
                # Log status periodically
                if (time.time() - start_time) % 1.0 < 0.02:  # Every 1 second
                    logger.info(f"Emergency hover: maintaining position for {(duration - (time.time() - start_time)):.1f}s more")
                
                time.sleep(0.02)  # 50Hz control rate
            
            logger.warning("Emergency hover completed")
            return True
            
        except Exception as e:
            logger.error(f"Error during emergency hover: {e}")
            # Try to stabilize the drone in case of error
            controller.send_hover_setpoint(0, 0, 0, position['z'])
            return False
    
    def calculate_return_vector(self, position: Dict[str, float]) -> Tuple[float, float, float]:
        """
        Calculate the return vector to bring the drone back to safety.
        
        Args:
            position: Current position with x, y, z coordinates
            
        Returns:
            Tuple of (roll, pitch, thrust_adjustment) to guide drone back to safety
            where roll and pitch are in degrees and thrust_adjustment is in raw units
        """
        if not self.position_recorded:
            return 0.0, 0.0, 0.0
            
        # Calculate direction vector from current position to takeoff point
        dx = self.takeoff_position['x'] - position['x']
        dy = self.takeoff_position['y'] - position['y']
        dz = self.takeoff_position['z'] - position['z']
        
        # Calculate horizontal direction angle (in radians)
        angle = math.atan2(dy, dx)
        
        # Calculate horizontal distance
        h_dist = math.sqrt(dx*dx + dy*dy)
        
        # Calculate return thrust adjustment based on height difference
        thrust_adjustment = 0
        
        # More aggressive height correction if above safety height
        if position['z'] > self.config.safety_max_height_m:
            # Strong descent if over max height
            thrust_adjustment = -2000
            logger.warning(f"Emergency height reduction: "
                         f"{position['z']:.2f}m > {self.config.safety_max_height_m}m")
        elif abs(dz) > 0.1:  # If height difference is significant
            thrust_adjustment = int(800 * (1 if dz > 0 else -1))  # Positive to go up, negative to go down
        
        # Set attitude angles to move drone toward takeoff point
        # Note: Roll controls Y-axis movement, Pitch controls X-axis movement
        # Calculate return_roll and return_pitch in degrees (not radians)
        return_roll = -math.sin(angle) * 5.0  # Limit tilt angle to 5 degrees
        return_pitch = -math.cos(angle) * 5.0  # Limit tilt angle to 5 degrees
        
        return return_roll, return_pitch, thrust_adjustment
        
    def trigger_return(self, controller, position: Dict[str, float]) -> bool:
        """
        Trigger an emergency return to the takeoff position.
        
        Args:
            controller: FlightController instance to send commands
            position: Current position with x, y, z coordinates
            
        Returns:
            True if return procedure completed, False if it failed
        """
        if not self.position_recorded:
            logger.error("Cannot trigger return: Takeoff position not recorded.")
            # Try to just level out and maintain altitude
            controller.send_setpoint(0, 0, 0, controller.config.hover_thrust)
            return False
            
        logger.warning("EXECUTING SAFETY RETURN TO CENTER")
        
        try:
            # Try to return for 5 seconds, then resume normal control
            start_time = time.time()
            
            while time.time() - start_time < 5.0:
                # Calculate current horizontal distance
                curr_dx = self.takeoff_position['x'] - position['x']
                curr_dy = self.takeoff_position['y'] - position['y']
                curr_dist = math.sqrt(curr_dx*curr_dx + curr_dy*curr_dy)
                
                # If we're back within safety radius, stop the emergency return
                if curr_dist < self.config.safety_radius_m * 0.8:
                    logger.info(f"Successfully returned within safe zone: "
                              f"{curr_dist:.2f}m < {self.config.safety_radius_m * 0.8:.2f}m")
                    break
                    
                # Calculate return vector
                return_roll, return_pitch, thrust_adjustment = self.calculate_return_vector(position)
                
                # Adjust thrust based on vertical difference - dynamic adjustment
                if position['z'] > self.config.safety_max_height_m:
                    # Continue aggressive descent if still over max height
                    thrust_adjustment = -2000
                elif time.time() - start_time > 2.5:  # After 2.5 seconds, prioritize height correction
                    if position['z'] > self.takeoff_position['z'] + 0.3:
                        thrust_adjustment = -1000  # Go down more aggressively
                    elif position['z'] < self.takeoff_position['z'] - 0.2:
                        thrust_adjustment = 500   # Go up a bit if too low
                
                adjusted_thrust = controller.config.hover_thrust + thrust_adjustment
                
                controller.send_setpoint(return_roll, return_pitch, 
                                      controller.config.default_yaw_rate, 
                                      adjusted_thrust)
                
                # Log progress
                if (time.time() - start_time) % 0.5 < 0.02:  # Every 0.5 seconds
                    logger.info(f"Safety return: dist={curr_dist:.2f}m, "
                              f"height={position['z']:.2f}m, "
                              f"roll/pitch={return_roll:.1f}/{return_pitch:.1f}, "
                              f"thrust={adjusted_thrust}")
                
                time.sleep(0.02)  # 50Hz control rate
                
            # Resume normal hover
            logger.warning("Safety return completed, resuming hover")
            controller.send_setpoint(0, 0, controller.config.default_yaw_rate, controller.config.hover_thrust)
            time.sleep(1.0)  # Hold hover for a moment
            
            return True
            
        except Exception as e:
            logger.error(f"Error during safety return: {e}")
            # Try to stabilize the drone
            controller.send_setpoint(0, 0, 0, controller.config.hover_thrust)
            return False
    
    def is_boundary_exceeded(self) -> bool:
        """
        Check if a safety boundary has been exceeded.
        
        Returns:
            True if a boundary was exceeded, False otherwise
        """
        return self.boundary_exceeded
    
    def reset_boundary_exceeded(self) -> None:
        """Reset the boundary exceeded flag."""
        self.boundary_exceeded = False 