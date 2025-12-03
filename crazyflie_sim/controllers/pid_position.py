"""
PID controller for position control of the Crazyflie.
"""
import logging
import time
from typing import Dict, Any, Tuple
import math

from crazyflie_sim.controllers.base import BaseController

logger = logging.getLogger(__name__)

class PIDPositionController(BaseController):
    """
    PID controller for position control.
    Takes a position setpoint and computes a thrust vector.
    """

    def __init__(self, kp: Dict[str, float], ki: Dict[str, float], kd: Dict[str, float],
                 max_thrust: float = 1.0, min_thrust: float = 0.0):
        """
        Initialize the PID position controller with gains.

        Args:
            kp: Proportional gains for x, y, z
            ki: Integral gains for x, y, z
            kd: Derivative gains for x, y, z
            max_thrust: Maximum thrust output (normalized 0-1)
            min_thrust: Minimum thrust output (normalized 0-1)
        """
        super().__init__()
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_thrust = max_thrust
        self.min_thrust = min_thrust

        # Integrator and previous error states
        self.integral = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.prev_error = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.prev_time = time.time()

        # Anti-windup limits
        self.integral_limit = 1.0

        logger.info(f"Initialized PID position controller with gains: P={kp}, I={ki}, D={kd}")

    def reset(self):
        """Reset the PID controller state."""
        self.integral = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.prev_error = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.prev_time = time.time()
        logger.debug("Reset PID position controller")

    def update_params(self, kp: Dict[str, float] = None, ki: Dict[str, float] = None,
                      kd: Dict[str, float] = None):
        """
        Update the PID gains.

        Args:
            kp: New proportional gains
            ki: New integral gains
            kd: New derivative gains
        """
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd

        logger.info(f"Updated PID gains: P={self.kp}, I={self.ki}, D={self.kd}")

    def compute(self, state: Dict[str, Any], cmd: Dict[str, float], dt: float) -> Tuple[float, float, float]:
        """
        Compute thrust vector using PID based on position error.

        Args:
            state: Current drone state including position
            cmd: Target position {x, y, z}
            dt: Time step

        Returns:
            Tuple of (thrust_x, thrust_y, thrust_z)
        """
        # Extract current position
        pos = state.get("position", {"x": 0.0, "y": 0.0, "z": 0.0})
        vel = state.get("velocity", {"x": 0.0, "y": 0.0, "z": 0.0})

        # Compute the error
        error = {
            "x": cmd.get("x", 0.0) - pos["x"],
            "y": cmd.get("y", 0.0) - pos["y"],
            "z": cmd.get("z", 0.0) - pos["z"]
        }

        # For logging and debugging
        logger.debug(f"Position error: x={error['x']:.3f}, y={error['y']:.3f}, z={error['z']:.3f}")

        # Update integral term with anti-windup
        for axis in ["x", "y", "z"]:
            self.integral[axis] += error[axis] * dt
            # Apply anti-windup
            self.integral[axis] = max(-self.integral_limit, min(self.integral_limit, self.integral[axis]))

        # Compute derivative (use velocity directly for smoother control)
        derivative = {
            "x": -vel["x"],  # Negative because velocity reduces the error
            "y": -vel["y"],
            "z": -vel["z"]
        }

        # Compute control output for each axis
        thrust = {}
        for axis in ["x", "y", "z"]:
            p_term = self.kp[axis] * error[axis]
            i_term = self.ki[axis] * self.integral[axis]
            d_term = self.kd[axis] * derivative[axis]
            thrust[axis] = p_term + i_term + d_term

        # Store current error for next iteration
        self.prev_error = error.copy()

        # Add gravity compensation to Z thrust
        # Assuming normalized thrust where 0.6 is hovering (counteracting gravity)
        thrust["z"] += 0.6

        # Clamp thrust for Z to valid range [min_thrust, max_thrust]
        thrust["z"] = max(self.min_thrust, min(self.max_thrust, thrust["z"]))

        # Calculate thrust vector magnitude
        thrust_magnitude = math.sqrt(thrust["x"]**2 + thrust["y"]**2 + thrust["z"]**2)

        # Normalize if exceeding max_thrust
        if thrust_magnitude > self.max_thrust:
            scale = self.max_thrust / thrust_magnitude
            thrust["x"] *= scale
            thrust["y"] *= scale
            thrust["z"] *= scale

        logger.debug(f"PID thrust: x={thrust['x']:.3f}, y={thrust['y']:.3f}, z={thrust['z']:.3f}")

        return thrust["x"], thrust["y"], thrust["z"]