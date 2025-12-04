"""
SimulationManager: Orchestrates IsaacLab simulation with CF2.1 BL controller.

Provides:
- Physics simulation via IsaacLab
- CF firmware-compatible controller
- Thread-safe state access
- Command queuing (position, velocity, attitude)
"""
import logging
import threading
import queue
import time
import math
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import IntEnum

import torch
import numpy as np
from isaaclab.sim import SimulationContext
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaacsim.core.utils.prims import set_prim_property
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaacsim.core.utils.stage import add_reference_to_stage
from isaaclab_assets import CRAZYFLIE_CFG

from crazyflie_sim.controllers.cf_controller import (
    CrazyflieController,
    ControlMode,
    config as cf_config,
)

logger = logging.getLogger(__name__)


class CommandType(IntEnum):
    """Command types for the simulation."""
    POSITION = 0
    VELOCITY = 1
    ATTITUDE = 2


@dataclass
class DroneState:
    """Thread-safe dataclass for drone state."""
    position: Dict[str, float] = None
    velocity: Dict[str, float] = None
    orientation: Dict[str, float] = None  # roll, pitch, yaw in degrees
    angular_velocity: Dict[str, float] = None  # deg/s
    timestamp: float = 0.0

    def __post_init__(self):
        if self.position is None:
            self.position = {"x": 0.0, "y": 0.0, "z": 0.0}
        if self.velocity is None:
            self.velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
        if self.orientation is None:
            self.orientation = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        if self.angular_velocity is None:
            self.angular_velocity = {"x": 0.0, "y": 0.0, "z": 0.0}


class SimulationManager:
    """
    Manages IsaacLab simulation with CF2.1 BL firmware-compatible controller.

    The controller matches the real Crazyflie 2.1 Brushless firmware:
    - Cascaded PID (position -> velocity -> attitude -> rate)
    - Motor mixing with attitude priority
    - Same default gains as firmware
    """

    def __init__(
        self,
        simulation_app,
        dt: float = 0.01,
        mass: float = cf_config.CF_MASS,
        arm_length: float = cf_config.ARM_LENGTH,
        inertia: tuple = (cf_config.INERTIA_XX, cf_config.INERTIA_YY, cf_config.INERTIA_ZZ),
    ):
        """
        Initialize simulation manager.

        Args:
            simulation_app: IsaacLab application instance
            dt: Simulation time step (seconds)
            mass: Drone mass (kg) - default CF2.1 BL
            arm_length: Motor arm length (m) - default CF2.1 BL
            inertia: Inertia tensor diagonal (kg*m^2)
        """
        self.simulation_app = simulation_app
        self.dt = dt
        self.mass = mass
        self.arm_length = arm_length
        self.inertia = inertia

        # State and threading
        self.state = DroneState()
        self.state_lock = threading.Lock()
        self.cmd_queue = queue.Queue()


        # Current setpoints
        self.current_cmd_type = CommandType.ATTITUDE
        self.position_setpoint = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
        self.velocity_setpoint = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0}
        self.attitude_setpoint = {"roll": 0.0, "pitch": 0.0, "yaw_rate": 0.0, "thrust": 0.0}

        # Device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Setup simulation and controller
        self._setup_simulation()
        self._setup_controller()

        logger.info(f"SimulationManager initialized: dt={dt}s, mass={mass}kg, device={self.device}")

    def _setup_simulation(self):
        """Initialize IsaacLab simulation environment."""
        self.sim = SimulationContext(sim_utils.SimulationCfg(dt=self.dt))

        # Ground plane
        ground_path = f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd"
        add_reference_to_stage(ground_path, "/World/Environment/Ground")

        # Robot
        robot_cfg = CRAZYFLIE_CFG.replace(prim_path="/World/Robot")
        self.robot = Articulation(robot_cfg)

        # Body frame coordinate axes visualization
        # Scale appropriate for Crazyflie (arm length ~5cm)
        frame_scale = 0.08  # 8cm axes for visibility
        self.frame_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/World/Visuals/BodyFrame",
                markers={
                    "frame": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                        scale=(frame_scale, frame_scale, frame_scale),
                    )
                }
            )
        )

        # Apply physics parameters before reset
        set_prim_property("/World/Robot/body", "physics:mass", self.mass)
        set_prim_property("/World/Robot/body", "physics:diagonalInertia", self.inertia)
        logger.info(f"Physics parameters set: mass={self.mass}kg, inertia={self.inertia}")

        # Reset and initialize
        self.sim.reset()

        joint_pos, joint_vel = self.robot.data.default_joint_pos, self.robot.data.default_joint_vel
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)

        # Start at 10cm above ground
        default_root = self.robot.data.default_root_state.clone()
        default_root[:, 2] = 0.1
        self.robot.write_root_pose_to_sim(default_root[:, :7])
        self.robot.write_root_velocity_to_sim(default_root[:, 7:])

        set_prim_property("/World/Robot", "visibility", "visible")
        logger.info("Simulation environment initialized")

    def _setup_controller(self):
        """Initialize CF firmware-compatible controller."""
        self.controller = CrazyflieController(
            num_envs=1,
            device=self.device,
            attitude_dt=self.dt,  # Use sim dt for both loops in simulation
            position_dt=self.dt,
        )
        logger.info("Controller initialized with CF2.1 BL parameters")

    def step(self) -> bool:
        """
        Execute one simulation step.

        Returns:
            True if simulation should continue, False to stop
        """
        if not self.simulation_app.is_running():
            return False

        self._update_state()
        self._process_commands()
        force, torque = self._compute_control()
        self._apply_control(force, torque)
        self._update_frame_marker()
        self.sim.step()


        return True

    def _update_state(self):
        """Read state from simulation."""
        self.robot.update(self.dt)
        root_state = self.robot.data.root_state_w

        # Position
        pos = {
            "x": root_state[0, 0].item(),
            "y": root_state[0, 1].item(),
            "z": root_state[0, 2].item(),
        }

        # Quaternion (w, x, y, z)
        qw, qx, qy, qz = root_state[0, 3:7].cpu().numpy()

        # Quaternion to Euler (ZYX convention)
        roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        sinp = 2.0 * (qw * qy - qz * qx)
        # Firmware adjusts pitch sign because of the legacy CF2 body frame
        pitch = -math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else -math.asin(sinp)
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        # Wrap to [-180, 180]
        def wrap(a):
            return ((a + math.pi) % (2 * math.pi)) - math.pi

        orient = {
            "roll": math.degrees(wrap(roll)),
            "pitch": math.degrees(wrap(pitch)),
            "yaw": math.degrees(wrap(yaw)),
        }

        # Velocity (world frame)
        vel = {
            "x": root_state[0, 7].item(),
            "y": root_state[0, 8].item(),
            "z": root_state[0, 9].item(),
        }

        # Angular velocity - convert from world frame to body frame
        # The Crazyflie firmware expects body-frame angular velocity (gyro output)
        # ω_body = R^T * ω_world, where R is the rotation matrix from body to world
        omega_world = root_state[0, 10:13].cpu().numpy()

        # Build rotation matrix from quaternion (body to world)
        # R = [[1-2(qy²+qz²), 2(qx*qy-qz*qw), 2(qx*qz+qy*qw)],
        #      [2(qx*qy+qz*qw), 1-2(qx²+qz²), 2(qy*qz-qx*qw)],
        #      [2(qx*qz-qy*qw), 2(qy*qz+qx*qw), 1-2(qx²+qy²)]]
        r00 = 1 - 2 * (qy * qy + qz * qz)
        r01 = 2 * (qx * qy - qz * qw)
        r02 = 2 * (qx * qz + qy * qw)
        r10 = 2 * (qx * qy + qz * qw)
        r11 = 1 - 2 * (qx * qx + qz * qz)
        r12 = 2 * (qy * qz - qx * qw)
        r20 = 2 * (qx * qz - qy * qw)
        r21 = 2 * (qy * qz + qx * qw)
        r22 = 1 - 2 * (qx * qx + qy * qy)

        # R^T * omega_world (transpose of rotation matrix applied to world angular velocity)
        omega_body_x = r00 * omega_world[0] + r10 * omega_world[1] + r20 * omega_world[2]
        omega_body_y = r01 * omega_world[0] + r11 * omega_world[1] + r21 * omega_world[2]
        omega_body_z = r02 * omega_world[0] + r12 * omega_world[1] + r22 * omega_world[2]

        # Convert to deg/s (body frame: +X forward, +Y left, +Z up)
        # The attitude controller applies the firmware's -gyro.y convention itself,
        # so we keep the raw body rates here.
        ang_vel = {
            "x": math.degrees(omega_body_x),
            "y": math.degrees(omega_body_y),
            "z": math.degrees(omega_body_z),
        }

        with self.state_lock:
            self.state.position = pos
            self.state.velocity = vel
            self.state.orientation = orient
            self.state.angular_velocity = ang_vel
            self.state.timestamp = self.sim.current_time

    def _process_commands(self):
        """Process queued commands."""
        while not self.cmd_queue.empty():
            try:
                cmd_type, cmd_data = self.cmd_queue.get_nowait()
                self.current_cmd_type = cmd_type

                if cmd_type == CommandType.POSITION:
                    self.position_setpoint = cmd_data
                elif cmd_type == CommandType.VELOCITY:
                    self.velocity_setpoint = cmd_data
                elif cmd_type == CommandType.ATTITUDE:
                    self.attitude_setpoint = cmd_data
            except queue.Empty:
                break

    def _compute_control(self) -> tuple:
        """Compute control from current setpoints."""
        with self.state_lock:
            state_dict = self._state_to_tensors()

        if self.current_cmd_type == CommandType.POSITION:
            self.controller.set_position_setpoint(
                x=torch.tensor([self.position_setpoint["x"]], device=self.device),
                y=torch.tensor([self.position_setpoint["y"]], device=self.device),
                z=torch.tensor([self.position_setpoint["z"]], device=self.device),
                yaw=torch.tensor([self.position_setpoint.get("yaw", 0.0)], device=self.device),
            )
        elif self.current_cmd_type == CommandType.VELOCITY:
            self.controller.set_velocity_setpoint(
                vx=torch.tensor([self.velocity_setpoint["vx"]], device=self.device),
                vy=torch.tensor([self.velocity_setpoint["vy"]], device=self.device),
                vz=torch.tensor([self.velocity_setpoint["vz"]], device=self.device),
                yaw_rate=torch.tensor([self.velocity_setpoint.get("yaw_rate", 0.0)], device=self.device),
            )
        else:  # ATTITUDE
            self.controller.set_attitude_setpoint(
                roll=torch.tensor([self.attitude_setpoint["roll"]], device=self.device),
                pitch=torch.tensor([self.attitude_setpoint["pitch"]], device=self.device),
                yaw_rate=torch.tensor([self.attitude_setpoint["yaw_rate"]], device=self.device),
                thrust=torch.tensor([self.attitude_setpoint["thrust"]], device=self.device),
            )

        return self.controller.compute(state_dict)

    def _state_to_tensors(self) -> Dict[str, torch.Tensor]:
        """Convert state dict to tensor format for controller."""
        return {
            "position": torch.tensor([[
                self.state.position["x"],
                self.state.position["y"],
                self.state.position["z"],
            ]], device=self.device, dtype=torch.float32),
            "velocity": torch.tensor([[
                self.state.velocity["x"],
                self.state.velocity["y"],
                self.state.velocity["z"],
            ]], device=self.device, dtype=torch.float32),
            "attitude": torch.tensor([[
                self.state.orientation["roll"],
                self.state.orientation["pitch"],
                self.state.orientation["yaw"],
            ]], device=self.device, dtype=torch.float32),
            "angular_velocity": torch.tensor([[
                self.state.angular_velocity["x"],
                self.state.angular_velocity["y"],
                self.state.angular_velocity["z"],
            ]], device=self.device, dtype=torch.float32),
        }

    def _apply_control(self, force: torch.Tensor, torque: torch.Tensor):
        """Apply force and torque to simulation."""
        forces = torch.zeros((1, 1, 3), device=self.device)
        torques = torch.zeros((1, 1, 3), device=self.device)
        forces[0, 0, :] = force[0, :]
        torques[0, 0, :] = torque[0, :]

        body_id = self.robot.find_bodies("body")[0]
        self.robot.set_external_force_and_torque(forces, torques, body_ids=body_id)

        # Update propeller velocities based on motor thrust
        # Motor thrust is stored in controller's power_distribution
        self._update_propeller_velocities()

        self.robot.write_data_to_sim()

    def _update_propeller_velocities(self):
        """Update propeller joint velocities based on motor thrust."""
        # Get motor thrust from controller (PWM scale 0-65535)
        motor_thrust = self.controller.power_distribution.motor_thrust[0]  # [4]

        # Convert thrust to angular velocity
        # At max thrust (65535), propeller spins at ~1000 rad/s (~9500 RPM)
        # This is an approximation for visualization
        max_omega = 1000.0  # rad/s at max thrust
        omega = motor_thrust / 65535.0 * max_omega

        # Motor spin directions: M1 CW, M2 CCW, M3 CW, M4 CCW
        # In the model, positive joint velocity = one direction
        # CW motors (M1, M3): positive rotation
        # CCW motors (M2, M4): negative rotation
        joint_vel = torch.zeros((1, 4), device=self.device)
        joint_vel[0, 0] = omega[0]   # m1_joint (CW)
        joint_vel[0, 1] = -omega[1]  # m2_joint (CCW)
        joint_vel[0, 2] = omega[2]   # m3_joint (CW)
        joint_vel[0, 3] = -omega[3]  # m4_joint (CCW)

        # Keep current joint positions (propellers just spin)
        joint_pos = self.robot.data.joint_pos.clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)

    def _update_frame_marker(self):
        """Update body frame visualization marker."""
        root_state = self.robot.data.root_state_w
        # Position
        pos = root_state[0, :3].cpu().numpy().reshape(1, 3)
        # Orientation (w, x, y, z)
        quat = root_state[0, 3:7].cpu().numpy().reshape(1, 4)
        self.frame_marker.visualize(translations=pos, orientations=quat)

    # --- Public API ---

    def enqueue_position_cmd(self, x: float, y: float, z: float, yaw: float = 0.0):
        """Queue position command."""
        self.cmd_queue.put((CommandType.POSITION, {"x": x, "y": y, "z": z, "yaw": yaw}))
        logger.debug(f"Position cmd: x={x:.2f}, y={y:.2f}, z={z:.2f}, yaw={yaw:.2f}")

    def enqueue_velocity_cmd(self, vx: float, vy: float, vz: float, yaw_rate: float = 0.0):
        """Queue velocity command (body frame)."""
        self.cmd_queue.put((CommandType.VELOCITY, {"vx": vx, "vy": vy, "vz": vz, "yaw_rate": yaw_rate}))
        logger.debug(f"Velocity cmd: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}")

    def enqueue_attitude_cmd(self, roll: float, pitch: float, yaw_rate: float, thrust: float):
        """
        Queue attitude command.

        Args:
            roll: Normalized roll [-1, 1]
            pitch: Normalized pitch [-1, 1]
            yaw_rate: Normalized yaw rate [-1, 1]
            thrust: Normalized thrust [0, 1]
        """
        cmd = {
            "roll": max(-1.0, min(1.0, roll)),
            "pitch": max(-1.0, min(1.0, pitch)),
            "yaw_rate": max(-1.0, min(1.0, yaw_rate)),
            "thrust": max(0.0, min(1.0, thrust)),
        }
        self.cmd_queue.put((CommandType.ATTITUDE, cmd))

    def get_state(self) -> Dict[str, Any]:
        """Get current drone state (thread-safe)."""
        with self.state_lock:
            return asdict(self.state)

    def reset(self):
        """Reset simulation and controller."""
        # Clear command queue first to stop any ongoing commands
        while not self.cmd_queue.empty():
            try:
                self.cmd_queue.get_nowait()
            except queue.Empty:
                break

        # Reset setpoints
        self.current_cmd_type = CommandType.ATTITUDE
        self.position_setpoint = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
        self.velocity_setpoint = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0}
        self.attitude_setpoint = {"roll": 0.0, "pitch": 0.0, "yaw_rate": 0.0, "thrust": 0.0}

        # Clear external forces
        body_id = self.robot.find_bodies("body")[0]
        zero_force = torch.zeros((1, 1, 3), device=self.device)
        self.robot.set_external_force_and_torque(zero_force, zero_force, body_ids=body_id)
        self.robot.write_data_to_sim()

        # Reset robot to initial pose
        # Note: Do NOT call self.sim.reset() - it invalidates PhysX references
        root_state = torch.zeros((1, 13), device=self.device)
        root_state[:, 0] = 0.0   # x
        root_state[:, 1] = 0.0   # y
        root_state[:, 2] = 0.1   # z (10cm above ground)
        root_state[:, 3] = 1.0   # qw (identity quaternion)
        root_state[:, 4] = 0.0   # qx
        root_state[:, 5] = 0.0   # qy
        root_state[:, 6] = 0.0   # qz
        # velocity already zero

        self.robot.write_root_pose_to_sim(root_state[:, :7])
        self.robot.write_root_velocity_to_sim(root_state[:, 7:])

        # Reset joint states
        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)

        self.robot.write_data_to_sim()

        # Update robot data
        self.robot.update(self.dt)


        # Update internal state
        self._update_state()

        # Reset controller with current state
        with self.state_lock:
            state_tensors = self._state_to_tensors()
        self.controller.reset(state_tensors)

        logger.info("Simulation reset")

    def get_controller_params(self) -> Dict[str, Any]:
        """Get controller parameters for debugging."""
        return {
            "mass": self.mass,
            "arm_length": self.arm_length,
            "inertia": self.inertia,
            "thrust_max": cf_config.THRUST_MAX,
            "pid_roll_rate": {
                "kp": cf_config.PID_ROLL_RATE_KP,
                "ki": cf_config.PID_ROLL_RATE_KI,
                "kd": cf_config.PID_ROLL_RATE_KD,
                "i_limit": cf_config.PID_ROLL_RATE_INTEGRATION_LIMIT,
            },
            "pid_roll": {
                "kp": cf_config.PID_ROLL_KP,
                "ki": cf_config.PID_ROLL_KI,
                "kd": cf_config.PID_ROLL_KD,
                "i_limit": cf_config.PID_ROLL_INTEGRATION_LIMIT,
            },
            "pid_pitch_rate": {
                "kp": cf_config.PID_PITCH_RATE_KP,
                "ki": cf_config.PID_PITCH_RATE_KI,
                "kd": cf_config.PID_PITCH_RATE_KD,
                "i_limit": cf_config.PID_PITCH_RATE_INTEGRATION_LIMIT,
            },
            "pid_pitch": {
                "kp": cf_config.PID_PITCH_KP,
                "ki": cf_config.PID_PITCH_KI,
                "kd": cf_config.PID_PITCH_KD,
                "i_limit": cf_config.PID_PITCH_INTEGRATION_LIMIT,
            },
            "pid_yaw_rate": {
                "kp": cf_config.PID_YAW_RATE_KP,
                "ki": cf_config.PID_YAW_RATE_KI,
                "kd": cf_config.PID_YAW_RATE_KD,
                "i_limit": cf_config.PID_YAW_RATE_INTEGRATION_LIMIT,
            },
            "pid_yaw": {
                "kp": cf_config.PID_YAW_KP,
                "ki": cf_config.PID_YAW_KI,
                "kd": cf_config.PID_YAW_KD,
                "i_limit": cf_config.PID_YAW_INTEGRATION_LIMIT,
            },
        }

    def get_controller_debug(self) -> Dict[str, Any]:
        """Get latest controller debug telemetry."""
        dbg = getattr(self.controller, "last_debug", None)
        if not dbg:
            return {}

        def to_list(tensor):
            if tensor is None:
                return None
            return tensor.detach().cpu().numpy().tolist()

        return {
            "attitude_desired": to_list(dbg.get("attitude_desired")),
            "attitude": to_list(dbg.get("attitude")),
            "gyro": to_list(dbg.get("gyro")),
            "rate_desired": to_list(dbg.get("rate_desired")),
            "rate_actual": to_list(dbg.get("rate_actual")),
            "roll_cmd": to_list(dbg.get("roll_cmd")),
            "pitch_cmd": to_list(dbg.get("pitch_cmd")),
            "yaw_cmd": to_list(dbg.get("yaw_cmd")),
            "thrust_pwm": to_list(dbg.get("thrust_pwm")),
            "motor_pwm": to_list(dbg.get("motor_pwm")),
            "force": to_list(dbg.get("force")),
            "torque": to_list(dbg.get("torque")),
            "position_setpoint": to_list(dbg.get("position_setpoint")),
            "position": to_list(dbg.get("position")),
            "velocity_setpoint": to_list(dbg.get("velocity_setpoint")),
            "velocity": to_list(dbg.get("velocity")),
            "timestamp": self.sim.current_time,
        }

    def update_controller_params(self, params: Dict[str, Any]):
        """Update controller gains at runtime."""
        try:
            self.controller.set_gains(params)
        except Exception as e:
            logger.error(f"Failed to update controller params: {e}")
            raise
