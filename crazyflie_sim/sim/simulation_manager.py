"""
SimulationManager class responsible for running the simulation and handling controllers.
"""
import logging
import threading
import queue
import time
import json
import math
from typing import Dict, Any, Tuple, Union
from dataclasses import dataclass, asdict

import torch
import numpy as np
from isaaclab.app import AppLauncher
from isaaclab.sim import SimulationContext
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim.schemas import MassPropertiesCfg, modify_mass_properties
from isaacsim.core.utils.prims import set_prim_property
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaacsim.core.utils.stage import add_reference_to_stage
from isaaclab.utils.math import matrix_from_quat

# Use absolute imports for controllers
from crazyflie_sim.controllers.base import BaseController
from crazyflie_sim.controllers.pid_position import PIDPositionController
from crazyflie_sim.controllers.attitude_wrapper import AttitudeControllerWrapper
from isaaclab_assets import CRAZYFLIE_CFG

# Define the drone state dataclass
@dataclass
class DroneState:
    """Thread-safe dataclass for storing drone state."""
    position: Dict[str, float] = None
    velocity: Dict[str, float] = None
    orientation: Dict[str, float] = None
    angular_velocity: Dict[str, float] = None
    timestamp: float = 0.0

    def __post_init__(self):
        """Initialize default values."""
        if self.position is None:
            self.position = {"x": 0.0, "y": 0.0, "z": 0.0}
        if self.velocity is None:
            self.velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
        if self.orientation is None:
            # Using Euler angles (roll, pitch, yaw) instead of quaternion
            self.orientation = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        if self.angular_velocity is None:
            self.angular_velocity = {"x": 0.0, "y": 0.0, "z": 0.0}

logger = logging.getLogger(__name__)

class SimulationManager:
    """
    Manages the simulation loop and controllers for the Crazyflie simulator.
    """

    def __init__(self, simulation_app, dt: float,
                 position_controller_params: Dict[str, Any],
                 attitude_controller_params: Dict[str, Any],
                 physics_params: Dict[str, Any]):
        """
        Initialize the simulation manager.

        Args:
            simulation_app: Application instance from AppLauncher
            dt: Simulation time step in seconds
            position_controller_params: Parameters for the position controller
            attitude_controller_params: Parameters for the attitude controller
            physics_params: Parameters for the physics simulation
        """
        self.simulation_app = simulation_app
        self.dt = dt
        self.physics_params = physics_params

        # Initialize state
        self.state = DroneState()
        self.state_lock = threading.Lock()

        # Command queues
        self.position_cmd_queue = queue.Queue()
        self.attitude_cmd_queue = queue.Queue()

        # Initialize current command
        self.current_position_cmd = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.current_attitude_cmd = {"roll": 0.0, "pitch": 0.0, "yaw_rate": 0.0, "thrust": 0.5}

        # Setup simulation and controllers
        self._setup_simulation()
        self._setup_controllers(position_controller_params, attitude_controller_params)

        logger.info("SimulationManager initialized with dt={dt:.3f}s")

    def _setup_simulation(self):
        """Set up the simulation environment and robot."""
        try:
            # Initialize simulation
            self.sim = SimulationContext(sim_utils.SimulationCfg(dt=self.dt))

            # Create ground plane
            ground_path = f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd"
            prim_path = "/World/Environment/Ground"
            add_reference_to_stage(ground_path, prim_path)

            # Setup robot
            robot_cfg = CRAZYFLIE_CFG.replace(prim_path="/World/Robot")
            self.robot = Articulation(robot_cfg)

            # Apply physics parameters
            mass = self.physics_params.get("mass", 0.041)
            modify_mass_properties(
                "/World/Robot/body",
                MassPropertiesCfg(mass=mass)
            )

            # Set inertia matrix
            inertia = self.physics_params.get("inertia", [1.3615e-5, 1.3615e-5, 3.257e-5])
            diag_inertia = torch.tensor(inertia, device="cpu")
            set_prim_property("/World/Robot/body", "physics:diagonalInertia",
                            (diag_inertia[0], diag_inertia[1], diag_inertia[2]))

            # Reset simulation
            self.sim.reset()

            # Initialize robot state
            joint_pos, joint_vel = self.robot.data.default_joint_pos, self.robot.data.default_joint_vel
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel)

            # Set initial position above ground
            default_root_state = self.robot.data.default_root_state.clone()
            default_root_state[:, 2] = 0.1  # Set height to 10cm
            self.robot.write_root_pose_to_sim(default_root_state[:, :7])
            self.robot.write_root_velocity_to_sim(default_root_state[:, 7:])

            # Set visibility
            set_prim_property("/World/Robot", "visibility", "visible")

            logger.info("Simulation setup completed")
        except Exception as e:
            logger.error(f"Error in simulation setup: {e}")
            raise

    def _setup_controllers(self, position_params: Dict[str, Any], attitude_params: Dict[str, Any]):
        """Set up the controllers."""
        try:
            # Initialize position controller
            self.position_controller = PIDPositionController(
                kp=position_params.get("Kp", {"x": 1.0, "y": 1.0, "z": 1.0}),
                ki=position_params.get("Ki", {"x": 0.0, "y": 0.0, "z": 0.1}),
                kd=position_params.get("Kd", {"x": 0.1, "y": 0.1, "z": 0.1}),
                max_thrust=position_params.get("max_thrust", 0.6),
                min_thrust=position_params.get("min_thrust", 0.0)
            )

            # Initialize attitude controller
            self.attitude_controller = AttitudeControllerWrapper(
                mass=self.physics_params.get("mass", 0.041),
                inertia=self.physics_params.get("inertia", [1.3615e-5, 1.3615e-5, 3.257e-5]),
                arm_length=self.physics_params.get("arm_length", 0.046),
                krp_ang=attitude_params.get("Krp_ang", [10.0, 10.0]),
                kdrp_ang=attitude_params.get("Kdrp_ang", [0.1, 0.1]),
                kinv_ang_vel_tau=attitude_params.get("Kinv_ang_vel_tau", [35.0, 35.0, 15.0])
            )

            # Current controller mode
            self.control_mode = "attitude"  # "position" or "attitude"

            logger.info("Controllers initialized")
        except Exception as e:
            logger.error(f"Error in controller setup: {e}")
            raise

    def step(self):
        """Perform a single simulation step.

        Returns:
            bool: True if simulation should continue, False if it should stop
        """
        try:
            # Check if simulation app is still running
            if not self.simulation_app.is_running():
                logger.warning("Simulation app is no longer running")
                return False

            # Check if robot and sim are valid
            if not hasattr(self, 'robot') or self.robot is None or not hasattr(self, 'sim') or self.sim is None:
                logger.error("Robot or simulation object is invalid")
                # Attempt to reinitialize
                try:
                    self._setup_simulation()
                    self._setup_controllers(
                        {"Kp": self.position_controller.kp, "Ki": self.position_controller.ki, "Kd": self.position_controller.kd},
                        {}  # Use default attitude controller parameters
                    )
                    logger.info("Successfully reinitialized simulation and controllers")
                except Exception as reinit_e:
                    logger.error(f"Failed to reinitialize simulation: {reinit_e}")
                    return False

            # Update state - wrapped in try/except to continue even if this fails
            try:
                self._update_state()
            except Exception as e:
                logger.error(f"Error in _update_state: {e}")

            # Process commands - wrapped in try/except to continue even if this fails
            try:
                self._process_commands()
            except Exception as e:
                logger.error(f"Error in _process_commands: {e}")

            # Compute control
            try:
                force, torque = self._compute_control()
                # print(f"cmd: {self.current_attitude_cmd}")
                # print(f"force: {force}, torque: {torque}")
            except Exception as e:
                logger.error(f"Error in _compute_control: {e}")
                # Default to zero force/torque
                force = torch.zeros(1, 3, device=self.attitude_controller.device)
                torque = torch.zeros(1, 3, device=self.attitude_controller.device)

            # Apply control
            try:
                self._apply_control(force, torque)
            except Exception as e:
                logger.error(f"Error in _apply_control: {e}")

            # Step simulation
            try:
                self.sim.step()
            except Exception as e:
                logger.error(f"Error in sim.step(): {e}")
                # Try to reset if stepping fails
                try:
                    self.sim.reset()
                except Exception as reset_e:
                    logger.error(f"Error resetting simulation after step failure: {reset_e}")

            return True

        except Exception as e:
            logger.error(f"Unhandled error in simulation step: {e}")
            return True  # Continue running despite errors

    def _update_state(self):
        """
        Update the state dataclass with current robot state.

        All angles (roll, pitch, yaw) in the orientation field are converted to degrees
        for consistency with the real Crazyflie's log data.
        All angular velocities are stored in radians/s internally, but will be converted
        to degrees/s in the get_state method for the API.
        """
        def normalize_angle(angle: float):
            return (angle + math.pi) % (2 * math.pi) - math.pi

        try:
            # Check if robot is valid
            if not hasattr(self, 'robot') or self.robot is None:
                logger.error("Robot object is None or invalid, cannot update state")
                return

            # Get current state from simulation
            self.robot.update(self.dt)
            cur_state = self.robot.data.root_state_w

            # Extract position, orientation, velocity and angular velocity
            position = {
                "x": cur_state[0, 0].item(),
                "y": cur_state[0, 1].item(),
                "z": cur_state[0, 2].item()
            }

            # Get quaternion values directly
            quat_w = cur_state[0, 3].item()
            quat_x = cur_state[0, 4].item()
            quat_y = cur_state[0, 5].item()
            quat_z = cur_state[0, 6].item()

            # Manual quaternion to Euler angles conversion
            # Roll (x-axis rotation)
            sinr_cosp = 2.0 * (quat_w * quat_x + quat_y * quat_z)
            cosr_cosp = 1.0 - 2.0 * (quat_x * quat_x + quat_y * quat_y)
            roll = math.atan2(sinr_cosp, cosr_cosp)

            # Pitch (y-axis rotation)
            sinp = 2.0 * (quat_w * quat_y - quat_z * quat_x)
            if abs(sinp) >= 1:
                pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
            else:
                pitch = math.asin(sinp)

            # Yaw (z-axis rotation)
            siny_cosp = 2.0 * (quat_w * quat_z + quat_x * quat_y)
            cosy_cosp = 1.0 - 2.0 * (quat_y * quat_y + quat_z * quat_z)
            yaw = math.atan2(siny_cosp, cosy_cosp)

            # Store Euler angles in degrees for consistency with real Crazyflie
            orientation = {
                "roll": math.degrees(normalize_angle(roll)),
                "pitch": math.degrees(normalize_angle(pitch)),
                "yaw": math.degrees(normalize_angle(yaw))
            }

            velocity = {
                "x": cur_state[0, 7].item(),
                "y": cur_state[0, 8].item(),
                "z": cur_state[0, 9].item()
            }

            angular_velocity = {
                "x": cur_state[0, 10].item(),
                "y": cur_state[0, 11].item(),
                "z": cur_state[0, 12].item()
            }

            # Convert angular velocity from rad/s to deg/s for API consistency with real Crazyflie
            RAD_TO_DEG = 180.0 / math.pi
            angular_velocity = {
                "x": angular_velocity["x"] * RAD_TO_DEG,
                "y": angular_velocity["y"] * RAD_TO_DEG,
                "z": angular_velocity["z"] * RAD_TO_DEG
            }

            # Update state with lock
            with self.state_lock:
                self.state.position = position
                self.state.velocity = velocity
                self.state.orientation = orientation
                self.state.angular_velocity = angular_velocity
                self.state.timestamp = time.time()

            logger.debug(f"State updated: pos=({position['x']:.2f}, {position['y']:.2f}, {position['z']:.2f}), "
                        f"orientation=({orientation['roll']:.2f}, {orientation['pitch']:.2f}, {orientation['yaw']:.2f})")
        except Exception as e:
            logger.error(f"Error updating state: {e}")

    def _process_commands(self):
        """Process any pending commands in the queues."""
        # Process position commands
        try:
            while not self.position_cmd_queue.empty():
                cmd = self.position_cmd_queue.get_nowait()
                self.current_position_cmd = cmd
                self.control_mode = "position"

                # Log the full command including yaw if present
                log_msg = f"Position command updated: x={cmd.get('x', 0.0):.2f}, y={cmd.get('y', 0.0):.2f}, z={cmd.get('z', 0.0):.2f}"
                if 'yaw' in cmd:
                    log_msg += f", yaw={cmd['yaw']:.2f}°"
                logger.debug(log_msg)
        except queue.Empty:
            pass

        # Process attitude commands
        try:
            while not self.attitude_cmd_queue.empty():
                cmd = self.attitude_cmd_queue.get_nowait()
                self.current_attitude_cmd = cmd
                self.control_mode = "attitude"
                logger.debug(f"Attitude command updated: {cmd}")
        except queue.Empty:
            pass

    def _compute_control(self):
        """Compute control outputs based on current mode and commands."""
        # Get current state dictionary
        with self.state_lock:
            state_dict = asdict(self.state)

        try:
            if self.control_mode == "position":
                # Position control mode
                thrust_vector = self.position_controller.compute(
                    state_dict, self.current_position_cmd, self.dt
                )

                # Convert thrust vector to attitude command
                # Calculate desired roll and pitch from thrust vector
                thrust_x, thrust_y, thrust_z = thrust_vector

                # Compute roll and pitch angles (simplified)
                # Limit roll and pitch to avoid instability
                max_angle = 0.5  # ~28.6 degrees
                roll = -thrust_y / max(thrust_z, 0.1)  # Roll is negative of y thrust
                pitch = thrust_x / max(thrust_z, 0.1)  # Pitch is x thrust

                # Clamp roll and pitch to valid range
                roll = max(-max_angle, min(max_angle, roll))
                pitch = max(-max_angle, min(max_angle, pitch))

                # Handle yaw control if specified in the position command
                yaw_rate = 0.0
                if 'yaw' in self.current_position_cmd:
                    # Get current yaw angle
                    current_yaw = state_dict.get('orientation', {}).get('yaw', 0.0)
                    # Get target yaw angle
                    target_yaw = self.current_position_cmd['yaw']

                    # Calculate yaw error (considering the circular nature of angles)
                    yaw_error = target_yaw - current_yaw
                    # Normalize to -180 to 180 degrees
                    if yaw_error > 180:
                        yaw_error -= 360
                    elif yaw_error < -180:
                        yaw_error += 360

                    # Simple proportional control for yaw
                    yaw_p_gain = 2.0
                    # Convert yaw error to yaw rate (normalized to [-1, 1])
                    MAX_YAW_RATE = 120.0  # degrees/s
                    yaw_rate = yaw_p_gain * yaw_error / MAX_YAW_RATE
                    yaw_rate = max(-1.0, min(1.0, yaw_rate))  # Clamp to [-1, 1]

                    logger.debug(f"Yaw control: current={current_yaw:.1f}°, target={target_yaw:.1f}°, error={yaw_error:.1f}°, rate={yaw_rate:.2f}")

                # Create attitude command with normalized values [-1, 1]
                attitude_cmd = {
                    "roll": roll / max_angle,
                    "pitch": pitch / max_angle,
                    "yaw_rate": yaw_rate,
                    "thrust": thrust_z
                }

                # Calculate force and torque using attitude controller
                force, torque = self.attitude_controller.compute(state_dict, attitude_cmd, self.dt)
            else:
                # Direct attitude control mode
                force, torque = self.attitude_controller.compute(
                    state_dict, self.current_attitude_cmd, self.dt
                )

            return force, torque

        except Exception as e:
            logger.error(f"Error computing control: {e}")
            # Return zero force and torque in case of error
            return (
                torch.zeros(1, 3, device=self.attitude_controller.device),
                torch.zeros(1, 3, device=self.attitude_controller.device)
            )

    def _apply_control(self, force, torque):
        """Apply force and torque to the drone."""
        try:
            # Check if robot is valid
            if not hasattr(self, 'robot') or self.robot is None:
                logger.error("Robot object is None or invalid, cannot apply control")
                return

            # Prepare force and torque tensors
            forces = torch.zeros((1, 1, 3), device=self.attitude_controller.device)
            torques = torch.zeros((1, 1, 3), device=self.attitude_controller.device)

            # Set force and torque values
            forces[0, 0, :] = force[0, :]
            torques[0, 0, :] = torque[0, :]

            # Apply to the robot
            body_id = self.robot.find_bodies("body")[0]
            self.robot.set_external_force_and_torque(forces, torques, body_ids=body_id)
            self.robot.write_data_to_sim()

            logger.debug(f"Applied force={force[0].cpu().numpy()} torque={torque[0].cpu().numpy()}")
        except Exception as e:
            logger.error(f"Error applying control: {e}")

    def enqueue_position_cmd(self, x: float, y: float, z: float, yaw: float = None):
        """
        Enqueue a position command.

        Args:
            x: X position (m)
            y: Y position (m)
            z: Z position (m)
            yaw: Yaw angle in degrees (optional)
        """
        cmd = {"x": x, "y": y, "z": z}
        if yaw is not None:
            cmd["yaw"] = yaw
        self.position_cmd_queue.put(cmd)
        log_msg = f"Position command enqueued: x={x:.2f}, y={y:.2f}, z={z:.2f}"
        if yaw is not None:
            log_msg += f", yaw={yaw:.2f}°"
        logger.info(log_msg)

    def enqueue_attitude_cmd(self, roll: float, pitch: float, yaw_rate: float, thrust: float):
        """
        Enqueue an attitude command.

        Args:
            roll: Roll angle normalized [-1, 1]
            pitch: Pitch angle normalized [-1, 1]
            yaw_rate: Yaw rate normalized [-1, 1]
            thrust: Thrust normalized [0, 1]
        """
        cmd = {
            "roll": max(-1.0, min(1.0, roll)),
            "pitch": max(-1.0, min(1.0, pitch)),
            "yaw_rate": max(-1.0, min(1.0, yaw_rate)),
            "thrust": max(0.0, min(1.0, thrust))
        }
        self.attitude_cmd_queue.put(cmd)
        logger.info(f"Attitude command enqueued: roll={roll:.2f}, pitch={pitch:.2f}, "
                  f"yaw_rate={yaw_rate:.2f}, thrust={thrust:.2f}")

    def update_pid_params(self, kp: Dict[str, float] = None, ki: Dict[str, float] = None,
                         kd: Dict[str, float] = None):
        """
        Update PID controller parameters.

        Args:
            kp: Proportional gains
            ki: Integral gains
            kd: Derivative gains
        """
        self.position_controller.update_params(kp, ki, kd)
        logger.info(f"PID parameters updated: Kp={kp}, Ki={ki}, Kd={kd}")

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current drone state.

        Returns:
            Dictionary containing drone state information with:
            - position: {x, y, z} in meters
            - velocity: {x, y, z} in m/s
            - orientation: {roll, pitch, yaw} in degrees
            - angular_velocity: {x, y, z} in degrees/s (converted from rad/s)
            - timestamp: time in seconds
        """
        with self.state_lock:
            return asdict(self.state)

    def reset(self):
        """Reset the simulation and controllers."""
        try:
            # Ensure robot object is valid before resetting
            if hasattr(self, 'robot') and self.robot is not None:
                # Reset robot position
                default_root_state = self.robot.data.default_root_state.clone()
                default_root_state[:, 2] = 0.1  # Set height to 10cm
                self.robot.write_root_pose_to_sim(default_root_state[:, :7])
                self.robot.write_root_velocity_to_sim(torch.zeros_like(default_root_state[:, 7:]))

                # Force an update to ensure robot state is properly initialized
                self.robot.update(self.dt)
            else:
                # Robot object is no longer valid, attempt to re-initialize simulation
                logger.warning("Robot object invalid, attempting to re-initialize simulation")
                self._setup_simulation()

            # Reset controllers
            self.position_controller.reset()
            self.attitude_controller.reset()

            # Clear command queues
            while not self.position_cmd_queue.empty():
                self.position_cmd_queue.get_nowait()
            while not self.attitude_cmd_queue.empty():
                self.attitude_cmd_queue.get_nowait()

            # Reset current commands
            self.current_position_cmd = {"x": 0.0, "y": 0.0, "z": 0.0}
            self.current_attitude_cmd = {"roll": 0.0, "pitch": 0.0, "yaw_rate": 0.0, "thrust": 0.5}

            # Reset state with Euler angles
            with self.state_lock:
                self.state = DroneState(
                    position={"x": 0.0, "y": 0.0, "z": 0.1},
                    velocity={"x": 0.0, "y": 0.0, "z": 0.0},
                    orientation={"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                    angular_velocity={"x": 0.0, "y": 0.0, "z": 0.0},
                    timestamp=time.time()
                )

            logger.info("Simulation reset completed")
        except Exception as e:
            logger.error(f"Error resetting simulation: {e}")
            raise

    def get_controller_params(self) -> Dict[str, Any]:
        """
        Get the current controller parameters.

        Returns:
            Dictionary with controller parameters
        """
        return {
            "position_controller": {
                "Kp": self.position_controller.kp,
                "Ki": self.position_controller.ki,
                "Kd": self.position_controller.kd,
                "max_thrust": self.position_controller.max_thrust,
                "min_thrust": self.position_controller.min_thrust
            }
        }