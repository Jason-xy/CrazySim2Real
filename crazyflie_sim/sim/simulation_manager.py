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
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import IntEnum

import torch
import numpy as np
from isaaclab.sim import SimulationContext
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.utils.prims import add_usd_reference, change_prim_property, set_prim_visibility
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab_physx.physics import PhysxCfg
from pxr import Gf, Sdf
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import euler_xyz_from_quat, matrix_from_quat
from isaaclab_assets import CRAZYFLIE_CFG
from crazyflie_sim.contact_monitor import ContactMonitor

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
    ground_contact: Dict[str, Any] = None

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
        recorder=None,
        render_interval: int = 4,
    ):
        """
        Initialize simulation manager.

        Args:
            simulation_app: IsaacLab application instance
            dt: Simulation time step (seconds)
            mass: Drone mass (kg) - default CF2.1 BL
            arm_length: Motor arm length (m) - default CF2.1 BL
            inertia: Inertia tensor diagonal (kg*m^2)
            recorder: Optional passive flight recorder
            render_interval: Physics steps per GUI frame; control/recording remain per step
        """
        self.simulation_app = simulation_app
        self.dt = dt
        self.mass = mass
        self.arm_length = arm_length
        self.inertia = inertia
        self.recorder = recorder
        if isinstance(render_interval, bool) or not isinstance(render_interval, int) or render_interval < 1:
            raise ValueError("render_interval must be a positive integer")
        self.render_interval = render_interval
        self._step_lock = threading.Lock()
        self._recording_reset = False
        self.contact_monitor = ContactMonitor()
        self._suppress_contact_until = 0

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
        self.sim = SimulationContext(sim_utils.SimulationCfg(
            dt=self.dt, device=str(self.device), physics=PhysxCfg(), render_interval=self.render_interval,
        ))

        # Ground plane
        ground_path = f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd"
        add_usd_reference(prim_path="/World/Environment/Ground", usd_path=ground_path)

        # Robot
        robot_cfg = CRAZYFLIE_CFG.replace(prim_path="/World/Robot")
        robot_cfg.spawn.activate_contact_sensors = True
        self.robot = Articulation(robot_cfg)
        self.contact_sensor = ContactSensor(ContactSensorCfg(
            prim_path="/World/Robot/.*", update_period=0.0, history_length=0,
        ))
        set_prim_visibility(get_current_stage().GetPrimAtPath("/World/Robot"), True)
        self.sim.set_camera_view((1.5, 1.5, 1.2), (0.0, 0.0, 0.5))

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
        change_prim_property(
            "/World/Robot/body.physics:mass", self.mass,
            type_to_create_if_not_exist=Sdf.ValueTypeNames.Float,
        )
        change_prim_property(
            "/World/Robot/body.physics:diagonalInertia", Gf.Vec3f(*self.inertia),
            type_to_create_if_not_exist=Sdf.ValueTypeNames.Float3,
        )
        logger.info(f"Physics parameters set: mass={self.mass}kg, inertia={self.inertia}")

        # Reset and initialize
        self.sim.reset()

        self.body_ids = self.robot.find_bodies("body")[0]
        self.robot.write_joint_state_to_sim_index(
            position=self.robot.data.default_joint_pos.torch.clone(),
            velocity=self.robot.data.default_joint_vel.torch.clone(),
        )

        # Start at 10cm above ground
        pose = self.robot.data.default_root_pose.torch.clone()
        pose[:, 2] = 0.1
        pose[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        self.robot.write_root_link_pose_to_sim_index(root_pose=pose)
        self.robot.write_root_com_velocity_to_sim_index(
            root_velocity=self.robot.data.default_root_vel.torch.clone(),
        )
        actual_mass = self.robot.data.body_mass.torch[0, self.body_ids[0]].item()
        self.total_mass = self.robot.data.body_mass.torch[0].sum().item()
        actual_inertia = self.robot.data.body_inertia.torch[0, self.body_ids[0]].reshape(3, 3)
        diagonal = actual_inertia.diagonal().cpu().numpy()
        if not np.isclose(actual_mass, self.mass, rtol=1e-4) or not np.allclose(
            diagonal, self.inertia, rtol=1e-4, atol=1e-10,
        ):
            raise RuntimeError(f"Physics parameter mismatch: mass={actual_mass}, inertia={diagonal}")
        logger.info("Verified PhysX body mass=%s kg, inertia=%s kg*m^2", actual_mass, diagonal.tolist())

        logger.info("Simulation environment initialized")

    def _setup_controller(self):
        """Initialize CF firmware-compatible controller."""
        self.controller = CrazyflieController(
            num_envs=1,
            device=self.device,
            attitude_dt=self.dt,  # Use sim dt for both loops in simulation
            position_dt=self.dt,
            enable_debug=True,
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

        with self._step_lock:
            self._update_state()
            self._process_commands()
            force, torque = self._compute_control()
            self._record_control_sample()
            self._apply_control(force, torque)
            render = self.sim.is_rendering and (
                (self.sim.get_physics_step_count() + 1) % self.render_interval == 0
            )
            if render:
                self._update_frame_marker()
            self.sim.step(render=render)

        return True

    def _record_control_sample(self):
        if self.recorder is None or not self.recorder.enabled:
            return
        try:
            if self._recording_reset:
                self.recorder.split()
                self._recording_reset = False
            debug = self.controller.last_debug
            mode = ControlMode(int(self.controller.control_mode[0].item())).name.lower()
            values = []
            for reference, measurement in (
                ("attitude_desired", "attitude"), ("rate_desired", "rate_actual"),
            ):
                if reference == "attitude_desired" and mode == "attitude_rate":
                    values.extend([None] * 6)
                    continue
                ref = debug[reference][0].detach().cpu().tolist()
                meas = debug[measurement][0].detach().cpu().tolist()
                if len(ref) != 3 or len(meas) != 3:
                    raise ValueError(f"{reference}/{measurement} must contain three axes")
                values.extend(value for pair in zip(ref, meas) for value in pair)
            self.recorder.record(self.state.timestamp, mode, values)
        except Exception as exc:
            # Snapshot failures must not stop flight control or wait for the writer.
            self.recorder._fail(f"controller snapshot failed: {exc}")

    def _update_state(self):
        """Read state from simulation."""
        with self.state_lock:
            previous_state = {
                "position": dict(self.state.position), "velocity": dict(self.state.velocity),
                "orientation": dict(self.state.orientation),
                "angular_velocity": dict(self.state.angular_velocity),
                "timestamp": self.state.timestamp,
            }
        self.robot.update(self.dt)
        self.contact_sensor.update(self.dt, force_recompute=True)
        pose = self.robot.data.root_link_pose_w.torch
        velocity = self.robot.data.root_com_vel_w.torch

        # Position
        pos = {
            "x": pose[0, 0].item(),
            "y": pose[0, 1].item(),
            "z": pose[0, 2].item(),
        }

        # XYZW body-to-world quaternion, matching Isaac Lab 3.
        quat = pose[:, 3:7]
        roll, pitch, yaw = euler_xyz_from_quat(quat)
        angles = torch.stack((roll, pitch, yaw), dim=-1).squeeze(0)
        # Wrap to [-pi, pi]
        angles = torch.remainder(angles + torch.pi, 2 * torch.pi) - torch.pi
        orient = {
            "roll": torch.rad2deg(angles[0]).item(),
            "pitch": torch.rad2deg(angles[1]).item(),
            "yaw": torch.rad2deg(angles[2]).item(),
        }

        # Velocity (world frame)
        vel = {
            "x": velocity[0, 0].item(),
            "y": velocity[0, 1].item(),
            "z": velocity[0, 2].item(),
        }

        # Angular velocity - convert from world frame to body frame
        # The Crazyflie firmware expects body-frame angular velocity (gyro output)
        # ω_body = R^T * ω_world, where R is the rotation matrix from body to world
        omega_world = velocity[0, 3:6]

        # Rotation matrix from quaternion (body to world) from isaaclab util
        rotation_matrix = matrix_from_quat(quat)[0]
        omega_body = rotation_matrix.t().matmul(omega_world)

        # Convert to deg/s (body frame: +X forward, +Y left, +Z up)
        # The attitude controller applies the firmware's -gyro.y convention itself,
        # so we keep the raw body rates here.
        ang_vel = {
            "x": torch.rad2deg(omega_body[0]).item(),
            "y": torch.rad2deg(omega_body[1]).item(),
            "z": torch.rad2deg(omega_body[2]).item(),
        }

        with self.state_lock:
            self.state.position = pos
            self.state.velocity = vel
            self.state.orientation = orient
            self.state.angular_velocity = ang_vel
            self.state.timestamp = self.sim.get_physics_step_count() * self.dt
            forces = self.contact_sensor.data.net_forces_w.torch
            active = bool((torch.linalg.vector_norm(forces, dim=-1) > 0.001).any().item())
            if self.sim.get_physics_step_count() >= self._suppress_contact_until:
                self.contact_monitor.update(active, self.state.timestamp, previous_state, {"position": pos})
            self.state.ground_contact = self.contact_monitor.status()

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

        self.robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=forces, torques=torques, body_ids=self.body_ids, is_global=False,
        )

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
        joint_pos = self.robot.data.joint_pos.torch.clone()
        self.robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel)

    def _update_frame_marker(self):
        """Update body frame visualization marker."""
        pose = self.robot.data.root_link_pose_w.torch
        self.frame_marker.visualize(translations=pose[:, :3], orientations=pose[:, 3:7])

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

    def get_recording_status(self):
        return {
            "service": "crazyflie_sim", "dt_s": self.dt,
            "capabilities": ["contact_events", "recording_split"],
            "recording": self.recorder.status() if self.recorder is not None else None,
        }

    def split_recording(self):
        """Seal the current segment without changing control or blocking on disk."""
        with self._step_lock:
            if self.recorder is None or not self.recorder.enabled:
                raise ValueError("Recording is unavailable")
            previous = self.recorder.status()["current_file"]
            self.recorder.split(finalize=True)
            return {"previous_file": previous, "recording": self.recorder.status()}

    def read_recording(self, name):
        """Expose only completed CSVs published by this recorder session."""
        if self.recorder is None or name not in self.recorder.status()["completed"]:
            raise FileNotFoundError("Completed recording not found")
        path = self.recorder.directory / name
        if path.is_symlink() or path.resolve().parent != self.recorder.directory:
            raise ValueError("Invalid recording path")
        return path.read_bytes()

    def reset(self):
        """Reset simulation and controller."""
        with self._step_lock:
            self._recording_reset = True
            self._reset()

    def _reset(self):
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
        self.robot.permanent_wrench_composer.reset()
        self.robot.instantaneous_wrench_composer.reset()
        zero_force = torch.zeros((1, 1, 3), device=self.device)
        self.robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=zero_force, torques=zero_force, body_ids=self.body_ids, is_global=False,
        )
        self.robot.write_data_to_sim()

        # Reset robot to initial pose
        pose = torch.tensor([[0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]], device=self.device)
        self.robot.write_root_link_pose_to_sim_index(root_pose=pose)
        self.robot.write_root_com_velocity_to_sim_index(
            root_velocity=torch.zeros((1, 6), device=self.device),
        )

        # Reset joint states
        joint_pos = self.robot.data.default_joint_pos.torch.clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel)

        self.robot.write_data_to_sim()

        # Update robot data
        self.robot.update(self.dt)
        self.contact_sensor.reset()
        self.contact_monitor.reset()
        # PhysX contact buffers still describe the pre-reset step until physics advances.
        self._suppress_contact_until = self.sim.get_physics_step_count() + 1

        # Update internal state
        self._update_state()

        # Reset controller with current state
        with self.state_lock:
            state_tensors = self._state_to_tensors()
        self.controller.reset(state_tensors)

        logger.info("Simulation reset")

    def get_controller_params(self) -> Dict[str, Any]:
        """Snapshot active gains, including changes made through the HTTP API."""
        with self._step_lock:
            attitude = self.controller.attitude_controller
            position = self.controller.position_controller
            result = {
                "mass": self.mass,
                "total_mass": self.total_mass,
                "arm_length": self.arm_length,
                "inertia": self.inertia,
                "thrust_max": self.controller.power_distribution.thrust_max,
            }
            for name, pid in (
                ("roll_rate", attitude.pid_roll_rate),
                ("roll", attitude.pid_roll),
                ("pitch_rate", attitude.pid_pitch_rate),
                ("pitch", attitude.pid_pitch),
                ("yaw_rate", attitude.pid_yaw_rate),
                ("yaw", attitude.pid_yaw),
                ("pos_x", position.pid_x),
                ("pos_y", position.pid_y),
                ("pos_z", position.pid_z),
                ("vel_x", position.pid_vx),
                ("vel_y", position.pid_vy),
                ("vel_z", position.pid_vz),
            ):
                result[name] = {key: getattr(pid, key) for key in ("kp", "ki", "kd")}
                # Preserve the existing API shape for outer-loop parameters.
                if name in ("roll_rate", "roll", "pitch_rate", "pitch", "yaw_rate", "yaw"):
                    result[name]["i_limit"] = pid.i_limit
            return result

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
            "timestamp": self.state.timestamp,
        }

    def update_controller_params(self, params: Dict[str, Any]):
        """Update controller gains at runtime."""
        try:
            with self._step_lock:
                self.controller.set_gains(params)
        except Exception as e:
            logger.error(f"Failed to update controller params: {e}")
            raise
