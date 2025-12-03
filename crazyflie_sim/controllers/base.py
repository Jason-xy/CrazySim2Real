"""
Abstract base controller for Crazyflie.
All controllers should inherit from this class.
"""
import abc
from typing import Dict, Tuple, Any, Union
import logging

logger = logging.getLogger(__name__)

class BaseController(abc.ABC):
    """
    Abstract base class for all controllers.

    Controllers take a state and a command, and compute force and torque
    values to be applied to the drone.
    """

    def __init__(self):
        """Initialize the controller."""
        logger.debug("Initializing base controller")

    @abc.abstractmethod
    def compute(self, state: Dict[str, Any], cmd: Dict[str, float], dt: float) -> Union[Tuple[Any, Any], float]:
        """
        Compute control outputs based on current state and command.

        Args:
            state: Dictionary containing drone state information
                (position, velocity, orientation, angular_velocity)
            cmd: Dictionary containing command values
            dt: Time step in seconds

        Returns:
            Either a tuple of (force, torque) or a single thrust value,
            depending on the controller implementation
        """
        pass

    def reset(self):
        """
        Reset the controller state (e.g., clear integrators, reset history).
        Should be overridden by derived classes if they maintain state.
        """
        logger.debug("Resetting controller")