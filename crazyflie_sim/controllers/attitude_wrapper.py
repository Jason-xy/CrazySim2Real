"""
Wrapper around the Quadrotor attitude controller from e2e_drone.utils.controller.
"""
import logging
import torch
from typing import Dict, Any, Tuple, List

from crazyflie_sim.controllers.controller import Quadrotor, QuadrotorDynamics
from crazyflie_sim.controllers.base import BaseController

logger = logging.getLogger(__name__)

class AttitudeControllerWrapper(BaseController):
    """
    Wrapper around the Quadrotor attitude controller to provide a consistent interface.
    """

    def __init__(self,
                 mass: float,
                 inertia: List[float],
                 arm_length: float,
                 krp_ang: List[float] = None,
                 kdrp_ang: List[float] = None,
                 kinv_ang_vel_tau: List[float] = None):
        """
        Initialize the attitude controller wrapper.

        Args:
            mass: Drone mass (kg)
            inertia: Moments of inertia [Ixx, Iyy, Izz] (kg.m^2)
            arm_length: Length of drone arm (m)
            krp_ang: Roll/pitch proportional gains [roll, pitch]
            kdrp_ang: Roll/pitch derivative gains [roll, pitch]
            kinv_ang_vel_tau: Angular velocity inverse time constants [roll, pitch, yaw]
        """
        super().__init__()

        # Check if CUDA is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"AttitudeControllerWrapper using device: {self.device}")

        # Convert to torch tensors
        mass_tensor = torch.tensor([mass], device=self.device)
        arm_l_tensor = torch.tensor([arm_length], device=self.device)
        inertia_tensor = torch.tensor([inertia], device=self.device)

        # Create dynamics model
        self.dynamics = QuadrotorDynamics(
            num_envs=1,
            mass=mass_tensor,
            inertia=inertia_tensor,
            arm_l=arm_l_tensor,
            device=self.device
        )

        # Create controller
        self.controller = Quadrotor(
            dynamics=self.dynamics,
            device=self.device,
            Krp_ang=krp_ang,
            Kdrp_ang=kdrp_ang,
            Kinv_ang_vel_tau=kinv_ang_vel_tau
        )

        logger.info("Initialized attitude controller wrapper")

    def reset(self):
        """Reset the controller."""
        self.controller.reset_error_history(batch_size=1)
        logger.debug("Reset attitude controller")

    def update_params(self, krp_ang: List[float] = None, kdrp_ang: List[float] = None,
                      kinv_ang_vel_tau: List[float] = None):
        """
        Update controller parameters.

        Args:
            krp_ang: Roll/pitch proportional gains
            kdrp_ang: Roll/pitch derivative gains
            kinv_ang_vel_tau: Angular velocity inverse time constants
        """
        if krp_ang is not None:
            self.controller.Krp_ang_ = torch.tensor(krp_ang, device=self.device).expand(1, 2)

        if kdrp_ang is not None:
            self.controller.Kdrp_ang_ = torch.tensor(kdrp_ang, device=self.device).expand(1, 2)

        if kinv_ang_vel_tau is not None:
            self.controller.Kinv_ang_vel_tau_ = torch.tensor(kinv_ang_vel_tau, device=self.device).expand(1, 3)

        logger.info(f"Updated attitude controller params: Krp_ang={krp_ang}, Kdrp_ang={kdrp_ang}, Kinv_ang_vel_tau={kinv_ang_vel_tau}")

    def compute(self, state: Dict[str, Any], cmd: Dict[str, float], dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attitude control outputs.

        Args:
            state: Current drone state (position, velocity, orientation, angular_velocity)
            cmd: Target attitude {roll, pitch, yaw_rate, thrust}
            dt: Time step

        Returns:
            Tuple of (force, torque) in body frame
        """
        # Convert state to the format expected by the Quadrotor controller
        state_tensor = torch.zeros(1, 19, device=self.device)

        # Extract state components
        position = state.get("position", {"x": 0.0, "y": 0.0, "z": 0.0})
        velocity = state.get("velocity", {"x": 0.0, "y": 0.0, "z": 0.0})
        orientation = state.get("orientation", {"roll": 0.0, "pitch": 0.0, "yaw": 0.0})
        angular_velocity = state.get("angular_velocity", {"x": 0.0, "y": 0.0, "z": 0.0})

        # Fill state tensor
        # position: x, y, z
        state_tensor[0, 0] = position["x"]
        state_tensor[0, 1] = position["y"]
        state_tensor[0, 2] = position["z"]

        # Convert orientation from Euler angles to quaternion if needed
        if "roll" in orientation and "pitch" in orientation and "yaw" in orientation:
            # Convert from degrees to radians for quaternion calculation
            # The orientation data from the SimulationManager is always in degrees
            roll_rad = torch.tensor(orientation["roll"] * 3.14159 / 180.0, device=self.device)
            pitch_rad = torch.tensor(orientation["pitch"] * 3.14159 / 180.0, device=self.device)
            yaw_rad = torch.tensor(orientation["yaw"] * 3.14159 / 180.0, device=self.device)

            # Calculate quaternion components
            cy = torch.cos(yaw_rad * 0.5)
            sy = torch.sin(yaw_rad * 0.5)
            cp = torch.cos(pitch_rad * 0.5)
            sp = torch.sin(pitch_rad * 0.5)
            cr = torch.cos(roll_rad * 0.5)
            sr = torch.sin(roll_rad * 0.5)

            # Fill quaternion (w, x, y, z)
            w = cr * cp * cy + sr * sp * sy
            x = sr * cp * cy - cr * sp * sy
            y = cr * sp * cy + sr * cp * sy
            z = cr * cp * sy - sr * sp * cy

            state_tensor[0, 3] = w
            state_tensor[0, 4] = x
            state_tensor[0, 5] = y
            state_tensor[0, 6] = z
        elif "w" in orientation and "x" in orientation and "y" in orientation and "z" in orientation:
            # Use quaternion directly if provided
            state_tensor[0, 3] = orientation["w"]
            state_tensor[0, 4] = orientation["x"]
            state_tensor[0, 5] = orientation["y"]
            state_tensor[0, 6] = orientation["z"]
        else:
            # Default to identity quaternion if orientation format is unknown
            state_tensor[0, 3] = 1.0  # w
            state_tensor[0, 4] = 0.0  # x
            state_tensor[0, 5] = 0.0  # y
            state_tensor[0, 6] = 0.0  # z
            logger.warning("Unknown orientation format. Using identity quaternion.")

        # velocity: vx, vy, vz
        state_tensor[0, 7] = velocity["x"]
        state_tensor[0, 8] = velocity["y"]
        state_tensor[0, 9] = velocity["z"]

        # angular velocity: wx, wy, wz
        # Convert from degrees/s to radians/s for internal processing
        # The angular velocity data from get_state() is in degrees/s
        DEG_TO_RAD = 3.14159 / 180.0
        state_tensor[0, 10] = angular_velocity["x"] * DEG_TO_RAD
        state_tensor[0, 11] = angular_velocity["y"] * DEG_TO_RAD
        state_tensor[0, 12] = angular_velocity["z"] * DEG_TO_RAD

        # Command tensor: [roll, pitch, yaw_rate, thrust]
        cmd_tensor = torch.zeros(1, 4, device=self.device)
        cmd_tensor[0, 0] = cmd.get("roll", 0.0)
        cmd_tensor[0, 1] = cmd.get("pitch", 0.0)
        cmd_tensor[0, 2] = cmd.get("yaw_rate", 0.0)
        cmd_tensor[0, 3] = cmd.get("thrust", 0.5)  # Default to mid-thrust

        # Compute control (force, torque)
        force, torque = self.controller.compute_control(state_tensor, cmd_tensor, dt)

        logger.debug(f"Attitude cmd: roll={cmd.get('roll', 0.0):.3f}, "
                    f"pitch={cmd.get('pitch', 0.0):.3f}, "
                    f"yaw_rate={cmd.get('yaw_rate', 0.0):.3f}, "
                    f"thrust={cmd.get('thrust', 0.5):.3f}")

        return force, torque