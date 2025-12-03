import torch
import math
import time
import threading
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from collections import deque

from isaaclab.utils.math import euler_xyz_from_quat

def quat_derivative(q, omega):
    """
    Compute the derivative of quaternion q given body rates omega.

    Args:
        q: Quaternion tensor of shape (batch_size, 4) in [w, x, y, z] format
        omega: Angular velocity tensor of shape (batch_size, 3) in body frame

    Returns:
        dq: Quaternion derivative tensor of shape (batch_size, 4)
    """
    # Extract quaternion components
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Extract angular velocity components
    wx, wy, wz = omega[:, 0], omega[:, 1], omega[:, 2]

    # Compute quaternion derivative
    dqw = -0.5 * ( qx*wx + qy*wy + qz*wz)
    dqx =  0.5 * ( qw*wx +  wy*qz -  wz*qy)
    dqy =  0.5 * ( qw*wy +  wz*qx -  wx*qz)
    dqz =  0.5 * ( qw*wz +  wx*qy -  wy*qx)


    return torch.stack([dqw, dqx, dqy, dqz], dim=1)

class QuadrotorDynamics:
    """
    Simplified batched quadrotor dynamics model for Crazyflie 2.0.
    All parameters are tensors of shape (num_envs,).
    """
    def __init__(self,
                 num_envs: int = 1,
                 mass: torch.Tensor = None,
                 inertia: torch.Tensor = None,
                 arm_l: torch.Tensor = None,
                 device: torch.device = torch.device("cpu")):
        self.device = device
        self.num_envs = num_envs

        # Initialize mass with proper batch dimension
        if mass is None:
            self.mass_ = torch.full((num_envs,), 0.027, device=self.device)
        else:
            self.mass_ = mass.to(self.device)
            if self.mass_.dim() == 0:  # scalar
                self.mass_ = self.mass_.expand(num_envs)

        # Initialize arm length with proper batch dimension
        if arm_l is None:
            self.arm_l_ = torch.full((num_envs,), 0.046, device=self.device)
        else:
            self.arm_l_ = arm_l.to(self.device)
            if self.arm_l_.dim() == 0:  # scalar
                self.arm_l_ = self.arm_l_.expand(num_envs)

        # Initialize inertia with proper batch dimension
        default_inertia = torch.tensor([9.19e-6, 9.19e-6, 22.8e-6], device=self.device)
        if inertia is None:
            self.inertia_ = default_inertia.expand(num_envs, 3)
        else:
            if inertia.dim() == 1:  # Single inertia vector [Ixx, Iyy, Izz]
                self.inertia_ = inertia.to(self.device).expand(num_envs, 3)
            elif inertia.dim() == 2:  # Already batched [num_envs, 3]
                self.inertia_ = inertia.to(self.device)
            else:
                raise ValueError(f"Inertia must be tensor of shape (3,) or (num_envs, 3), got {inertia.shape}")

        # Motor limits - now directly initialized with batch dimension
        kv_rpm_per_v = 10000.0
        kt = 60.0 / (2 * math.pi * kv_rpm_per_v)
        self.motor_tau_inv_ = torch.full((num_envs,), 1.0 / kt, device=self.device)

        rpm_max, rpm_min = 24000.0, 1200.0
        self.motor_omega_max_ = torch.full((num_envs,), rpm_max * math.pi / 30, device=self.device)
        self.motor_omega_min_ = torch.full((num_envs,), rpm_min * math.pi / 30, device=self.device)

        # Thrust map coefficients - directly with batch dimension
        self.thrust_map_ = torch.tensor([9.96063125e-08, -2.55003087e-05, 5.84422691e-03],
                                      device=self.device).expand(num_envs, 3)

        # Drag-torque ratio
        self.kappa_ = torch.full((num_envs,), 0.005964552, device=self.device)

        # Thrust limits
        self.thrust_min_ = torch.zeros(num_envs, device=self.device)
        a, b, c = self.thrust_map_[:, 0], self.thrust_map_[:, 1], self.thrust_map_[:, 2]
        w = self.motor_omega_max_
        self.thrust_max_ = a * (w ** 2) + b * w + c
        self.thrust_max_ = 1.358 * self.thrust_max_  # 50% headroom for custom motor

        # Angular velocity limits
        self.omega_max_ = torch.full((num_envs, 3), 6.0, device=self.device)

        self.updateInertiaMatrix()

    def updateInertiaMatrix(self):
        # t_BM: mapping matrix for X-configuration, shape (num_envs, 3, 4)
        base_mat = torch.tensor([[1, -1, -1, 1],
                                 [-1, -1, 1, 1],
                                 [0, 0, 0, 0]], dtype=torch.float, device=self.device)
        factor = self.arm_l_ * math.sqrt(0.5)
        self.t_BM_ = factor.view(-1, 1, 1) * base_mat

        # Construct the diagonal inertia matrix J_ and its inverse efficiently
        self.J_ = torch.diag_embed(self.inertia_)
        # Efficient inverse for diagonal matrix - just reciprocal the diagonal elements
        self.J_inv_ = torch.diag_embed(1.0 / self.inertia_)

    def get_max_torque(self):
        max_prop_force = self.thrust_max_ / 4.0
        factor = self.arm_l_ * math.sqrt(0.5)
        max_torque_x = factor * 2 * max_prop_force
        max_torque_y = factor * 2 * max_prop_force
        max_torque_z = self.kappa_ * 2 * max_prop_force
        return torch.stack([max_torque_x, max_torque_y, max_torque_z], dim=1)

    def clampThrust(self, thrusts: torch.Tensor):
        t_min = self.thrust_min_.unsqueeze(1)
        t_max = self.thrust_max_.unsqueeze(1)
        return torch.max(torch.min(thrusts, t_max), t_min)

    def clampMotorOmega(self, omega: torch.Tensor):
        w_min = self.motor_omega_min_.unsqueeze(1)
        w_max = self.motor_omega_max_.unsqueeze(1)
        return torch.max(torch.min(omega, w_max), w_min)

    def clampBodyrates(self, omega: torch.Tensor):
        return torch.max(torch.min(omega, self.omega_max_), -self.omega_max_)

    def motorOmegaToThrust(self, omega: torch.Tensor):
        a = self.thrust_map_[:, 0].unsqueeze(1)
        b = self.thrust_map_[:, 1].unsqueeze(1)
        c = self.thrust_map_[:, 2].unsqueeze(1)
        return a * (omega ** 2) + b * omega + c

    def motorThrustToOmega(self, thrusts: torch.Tensor):
        a = self.thrust_map_[:, 0].unsqueeze(1)
        b = self.thrust_map_[:, 1].unsqueeze(1)
        c = self.thrust_map_[:, 2].unsqueeze(1)
        inside = b ** 2 - 4.0 * a * (c - thrusts)
        return (-b + torch.sqrt(torch.clamp(inside, min=0.0))) / (2.0 * a)

    def getAllocationMatrix(self):
        num_envs = self.mass_.shape[0]
        ones_row = torch.ones(num_envs, 4, device=self.device)
        alloc = torch.stack([
            ones_row,
            self.t_BM_[:, 0, :],
            self.t_BM_[:, 1, :],
            self.kappa_.unsqueeze(1) * torch.tensor([1, -1, 1, -1], device=self.device).expand(num_envs, 4)
        ], dim=1)
        return alloc

    def dState(self, state: torch.Tensor):
        """
        Computes the time-derivative of the state.
        State structure:
          - state[:, 0:3]: position
          - state[:, 3:7]: quaternion
          - state[:, 7:10]: velocity
          - state[:, 10:13]: angular velocity
          - state[:, 13:16]: body torque
          - state[:, 16:19]: linear acceleration
        """
        dstate = torch.zeros_like(state)
        # Position derivative = velocity
        dstate[:, 0:3] = state[:, 7:10]
        # Quaternion derivative from angular velocity
        dstate[:, 3:7] = quat_derivative(state[:, 3:7], state[:, 10:13])
        # Velocity derivative = linear acceleration
        dstate[:, 7:10] = state[:, 16:19]

        # Angular velocity derivative (angular acceleration)
        omega = state[:, 10:13]
        body_torque = state[:, 13:16]
        J_omega = torch.einsum('nij,nj->ni', self.J_, omega)
        torque_term = body_torque - torch.cross(omega, J_omega, dim=1)
        dstate[:, 10:13] = torch.einsum('nij,nj->ni', self.J_inv_, torque_term)

        return dstate

class Quadrotor:
    """
    Simplified batched quadrotor controller.
    Inputs:
      - state: tensor with keys: 'pos' (n,3), 'quat' (n,4),
               'vel' (n,3), 'ang_vel' (n,3)
      - cmd: tensor with keys: 'roll' (n,), 'pitch' (n,),
             'yaw_rate' (n,), 'thrust' (n,)
    Outputs: body-frame force and torque (n,3) each.
    """
    def __init__(self,
             dynamics: QuadrotorDynamics = None,
             num_envs: int = 1,
             device=torch.device("cpu"),
             Krp_ang=None,
             Kdrp_ang=None,
             Kinv_ang_vel_tau=None,
             debug_viz=False,
             viz_port=8000):
        self.dynamics = dynamics if dynamics is not None else QuadrotorDynamics(num_envs=num_envs, device=device)
        self.device = device
        # Get batch size consistently
        self.num_envs = self.dynamics.num_envs

        # Initialize controller gains directly with batch dimensions
        # Proportional gains for roll and pitch control
        if Krp_ang is None:
            self.Krp_ang_ = torch.full((self.num_envs, 2), 10.0, device=self.device)
        else:
            if isinstance(Krp_ang, list) or isinstance(Krp_ang, tuple):
                self.Krp_ang_ = torch.tensor(Krp_ang, device=self.device).expand(self.num_envs, 2)
            elif isinstance(Krp_ang, torch.Tensor):
                if Krp_ang.dim() == 1:
                    self.Krp_ang_ = Krp_ang.to(self.device).expand(self.num_envs, Krp_ang.size(0))
                else:
                    self.Krp_ang_ = Krp_ang.to(self.device)
            else:
                raise ValueError(f"Krp_ang must be a list, tuple or tensor, got {type(Krp_ang)}")

        # Derivative gains for roll and pitch control
        if Kdrp_ang is None:
            self.Kdrp_ang_ = torch.full((self.num_envs, 2), 0.2, device=self.device)
        else:
            if isinstance(Kdrp_ang, list) or isinstance(Kdrp_ang, tuple):
                self.Kdrp_ang_ = torch.tensor(Kdrp_ang, device=self.device).expand(self.num_envs, 2)
            elif isinstance(Kdrp_ang, torch.Tensor):
                if Kdrp_ang.dim() == 1:
                    self.Kdrp_ang_ = Kdrp_ang.to(self.device).expand(self.num_envs, Kdrp_ang.size(0))
                else:
                    self.Kdrp_ang_ = Kdrp_ang.to(self.device)
            else:
                raise ValueError(f"Kdrp_ang must be a list, tuple or tensor, got {type(Kdrp_ang)}")

        # Inverse time constants for angular velocity control
        if Kinv_ang_vel_tau is None:
            self.Kinv_ang_vel_tau_ = torch.tensor([25.0, 25.0, 15.0], device=self.device).expand(self.num_envs, 3)
        else:
            if isinstance(Kinv_ang_vel_tau, list) or isinstance(Kinv_ang_vel_tau, tuple):
                self.Kinv_ang_vel_tau_ = torch.tensor(Kinv_ang_vel_tau, device=self.device).expand(self.num_envs, 3)
            elif isinstance(Kinv_ang_vel_tau, torch.Tensor):
                if Kinv_ang_vel_tau.dim() == 1:
                    self.Kinv_ang_vel_tau_ = Kinv_ang_vel_tau.to(self.device).expand(self.num_envs, Kinv_ang_vel_tau.size(0))
                else:
                    self.Kinv_ang_vel_tau_ = Kinv_ang_vel_tau.to(self.device)
            else:
                raise ValueError(f"Kinv_ang_vel_tau must be a list, tuple or tensor, got {type(Kinv_ang_vel_tau)}")

        self.gravity = torch.tensor([0.0, 0.0, 9.81], device=self.device)
        self.alloc_matrix_ = self.dynamics.getAllocationMatrix()
        self.alloc_matrix_pinv_ = torch.linalg.pinv(self.alloc_matrix_)
        print(f"Quadrotor controller initialized with batch size: {self.num_envs}")

        # Debug visualization settings
        self.debug_viz = debug_viz
        if self.debug_viz:
            self.viz_data = {
                'timestamp': time.time(),
                'batch_size': self.num_envs,  # Renamed from num_envs for clarity in API
                'state': {},
                'cmd': {},
                'control': {},
                'params': {
                    'Krp_ang': self.Krp_ang_.tolist(),
                    'Kdrp_ang': self.Kdrp_ang_.tolist(),
                    'Kinv_ang_vel_tau': self.Kinv_ang_vel_tau_.tolist(),
                }
            }
            # Store historical data for the last 10 seconds
            self.viz_history = deque(maxlen=1000)  # Assuming 100Hz, 10 seconds
            # Start the server in a separate thread
            self._start_viz_server(viz_port)

    def _start_viz_server(self, port):
        """Start a simple HTTP server to serve visualization data"""
        controller = self  # Reference to self for the handler

        class ControllerHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/viz_data':
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(controller.viz_data).encode())
                elif self.path == '/viz_history':
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(list(controller.viz_history)).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path == '/update_params':
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length).decode('utf-8')
                    params = json.loads(post_data)

                    # Update controller parameters with proper batch dimension
                    if 'Krp_ang' in params:
                        values = torch.tensor(params['Krp_ang'], device=controller.device)
                        # Ensure proper shape and expand to match num_envs
                        if values.dim() == 1:
                            controller.Krp_ang_ = values.expand(controller.num_envs, values.size(0))
                        else:
                            controller.Krp_ang_ = values

                    if 'Kdrp_ang' in params:
                        values = torch.tensor(params['Kdrp_ang'], device=controller.device)
                        # Ensure proper shape and expand to match num_envs
                        if values.dim() == 1:
                            controller.Kdrp_ang_ = values.expand(controller.num_envs, values.size(0))
                        else:
                            controller.Kdrp_ang_ = values

                    if 'Kinv_ang_vel_tau' in params:
                        values = torch.tensor(params['Kinv_ang_vel_tau'], device=controller.device)
                        # Ensure proper shape and expand to match num_envs
                        if values.dim() == 1:
                            controller.Kinv_ang_vel_tau_ = values.expand(controller.num_envs, values.size(0))
                        else:
                            controller.Kinv_ang_vel_tau_ = values

                    # Update the viz_data with new parameters
                    controller.viz_data['params'] = {
                        'Krp_ang': controller.Krp_ang_.tolist(),
                        'Kdrp_ang': controller.Kdrp_ang_.tolist(),
                        'Kinv_ang_vel_tau': controller.Kinv_ang_vel_tau_.tolist(),
                    }

                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'status': 'success'}).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                # Suppress server logs
                return

        # Start HTTP server
        self.server = HTTPServer(('localhost', port), ControllerHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        print(f"Controller visualization server started on port {port}")

    def reset_error_history(self, batch_size=None):
        """Reset method kept for backward compatibility - no longer needed"""
        pass

    def compute_control(self, state: torch.Tensor, cmd: torch.Tensor, dt: float = 0.005):
        """
        Compute desired body-frame force and torque with simplified control.
        Args:
            state: [batch,19] tensor: pos(0:3), quat(3:7), vel(7:10), ang_vel(10:13), torque(13:16), lin_acc(16:19)
            cmd: [batch,4] tensor: [roll_norm, pitch_norm, yaw_rate_norm, thrust_norm]
            dt: timestep for derivative calculations
        Returns:
            force: [batch,3] desired body forces
            torque: [batch,3] desired body torques
        """
        batch = state.shape[0]

        # Unpack current orientation and angular velocity
        roll_cur, pitch_cur, _ = euler_xyz_from_quat(state[:, 3:7])
        omega_cur = state[:, 10:13]

        # Map normalized commands to physical setpoints
        roll_des     = cmd[:,0] * (math.pi/6)
        pitch_des    = cmd[:,1] * (math.pi/6)
        yaw_rate_des = cmd[:,2] * (math.pi/1.5)
        F_des        = cmd[:,3] * self.dynamics.thrust_max_

        # --- NEW: Gain scheduling based on thrust command ---
        # Scale gains based on thrust command (higher at high thrust, lower at low thrust)
        thrust_norm = cmd[:,3]  # Already normalized in [0.0, 1.0]
        # gain_scale = 0.5 + 0.5 * thrust_norm  # Map [0→0.5, 1→1.0]
        # Shut off gain scheduling for now
        gain_scale = torch.ones_like(thrust_norm)
        Krp_scaled = self.Krp_ang_ * gain_scale.unsqueeze(1)
        Kdrp_scaled = self.Kdrp_ang_ * gain_scale.unsqueeze(1)

        # --- 1. Roll/Pitch PD → desired angular rates ---
        # Compute angle errors in [-π, π]
        roll_err  = ((roll_des - roll_cur + math.pi) % (2*math.pi)) - math.pi
        pitch_err = ((pitch_des - pitch_cur + math.pi) % (2*math.pi)) - math.pi

        # Firmware-style PD: P·error - D·measurement (no derivative kick)
        # Using angular velocity measurement directly instead of error differentiation
        roll_rate_des = Krp_scaled[:, 0] * roll_err - Kdrp_scaled[:, 0] * omega_cur[:, 0]
        pitch_rate_des = Krp_scaled[:, 1] * pitch_err - Kdrp_scaled[:, 1] * omega_cur[:, 1]

        # --- 2. Yaw P control ---
        # Stack desired angular velocities
        w_des = torch.stack([roll_rate_des, pitch_rate_des, yaw_rate_des], dim=1)
        omega_err = w_des - omega_cur

        # --- 3. Map rate errors to torques ---
        # Using broadcasting for the Kinv_ang_vel_tau_ operation
        tau_rp_yaw = self.Kinv_ang_vel_tau_ * omega_err

        # Inertia and gyroscopic compensation
        Jtau = torch.einsum('nij,nj->ni', self.dynamics.J_, tau_rp_yaw)
        Jw   = torch.einsum('nij,nj->ni', self.dynamics.J_, omega_cur)
        gyro = torch.cross(omega_cur, Jw, dim=1)
        tau_des = Jtau + gyro

        # --- 4. Baseline-plus-differential thrust allocation ---
        # Step 1: Compute baseline thrust (equal for all motors)
        T0 = F_des / 4.0  # Shape: [batch]

        # --- NEW: Dynamic calculation of differential thrust bounds ---
        # Calculate per-motor limits rather than using a single delta_T_max value
        T_motor_max = self.dynamics.thrust_max_ / 4.0
        T_motor_min = self.dynamics.thrust_min_
        delta_T_max_pos = (T_motor_max - T0).clamp(min=0.0)  # Maximum positive differential
        delta_T_max_neg = (T0 - T_motor_min).clamp(min=0.0)  # Maximum negative differential

        # Step 2: Compute differential thrust for torque control (zero net thrust)
        # Create wrench with zero collective thrust component
        torque_only_wrench = torch.cat([
            torch.zeros_like(F_des).unsqueeze(1),  # Zero collective thrust
            tau_des  # Keep torque components
        ], dim=1)

        # Compute differential thrust components
        delta_T = torch.bmm(self.alloc_matrix_pinv_, torque_only_wrench.unsqueeze(-1)).squeeze(-1)

        # Create masks for positive and negative values
        pos_mask = delta_T > 0
        neg_mask = delta_T < 0

        # Apply appropriate limits based on direction
        scale_factors_pos = torch.where(pos_mask, delta_T / delta_T_max_pos.unsqueeze(1),
                                      torch.zeros_like(delta_T))
        scale_factors_neg = torch.where(neg_mask, -delta_T / delta_T_max_neg.unsqueeze(1),
                                      torch.zeros_like(delta_T))

        # Find maximum scaling factor per batch
        max_scale_per_batch, _ = torch.max(torch.maximum(scale_factors_pos, scale_factors_neg), dim=1, keepdim=True)

        # Apply scaling only when limits are exceeded
        delta_T_scaling = torch.where(
            max_scale_per_batch > 1.0,
            1.0 / max_scale_per_batch,
            torch.ones_like(max_scale_per_batch)
        )
        delta_T = delta_T * delta_T_scaling

        # Step 3: Combine baseline and differential thrust
        motor_thrusts = T0.unsqueeze(1) + delta_T  # Shape: [batch, 4]

        # Step 4: Enforce non-negative thrust
        motor_thrusts = torch.clamp(motor_thrusts, min=0.0)

        # Special case: When throttle command is zero, force all motors to be exactly zero
        # This ensures no lift with zero throttle command
        zero_throttle = (cmd[:, 3] == 0.0).unsqueeze(1)
        motor_thrusts = torch.where(zero_throttle, torch.zeros_like(motor_thrusts), motor_thrusts)

        # Step 5: Limit per-motor maximum thrust using uniform scaling
        thrust_max_expanded = self.dynamics.thrust_max_.unsqueeze(1).expand_as(motor_thrusts)
        scale_factors = motor_thrusts / thrust_max_expanded
        max_scale_per_batch, _ = torch.max(scale_factors, dim=1, keepdim=True)

        # Apply uniform scaling where needed (only when max_scale > 1.0)
        scaling = torch.where(
            max_scale_per_batch > 1.0,
            1.0 / max_scale_per_batch,
            torch.ones_like(max_scale_per_batch)
        )

        # Scale all motors uniformly
        motor_thrusts = motor_thrusts * scaling

        # Step 6: Reconstruct resulting force & torque
        # Map motor thrusts back to wrench
        alloc = torch.bmm(self.alloc_matrix_, motor_thrusts.unsqueeze(-1)).squeeze(-1)

        # Extract force and torque components
        force = torch.zeros((batch, 3), device=self.device)
        force[:, 2] = alloc[:, 0]  # Only Z-axis thrust in body frame
        torque = alloc[:, 1:4]     # x, y, z torques

        # Update visualization data if debug_viz is enabled
        if self.debug_viz:
            current_time = time.time()

            # Create a snapshot of the current state for visualization
            viz_snapshot = {
                'timestamp': current_time,
                'num_envs': state.shape[0],
                'state': {
                    'pos': state[:, 0:3].detach().cpu().numpy().tolist(),
                    'quat': state[:, 3:7].detach().cpu().numpy().tolist(),
                    'vel': state[:, 7:10].detach().cpu().numpy().tolist(),
                    'ang_vel': state[:, 10:13].detach().cpu().numpy().tolist()
                },
                'cmd': {
                    'roll': cmd[:, 0].detach().cpu().numpy().tolist(),
                    'pitch': cmd[:, 1].detach().cpu().numpy().tolist(),
                    'yaw_rate': cmd[:, 2].detach().cpu().numpy().tolist(),
                    'thrust': cmd[:, 3].detach().cpu().numpy().tolist(),
                    'roll_des': roll_des.detach().cpu().numpy().tolist(),
                    'pitch_des': pitch_des.detach().cpu().numpy().tolist(),
                    'yaw_rate_des': yaw_rate_des.detach().cpu().numpy().tolist(),
                },
                'control': {
                    'force': force.detach().cpu().numpy().tolist(),
                    'torque': torque.detach().cpu().numpy().tolist(),
                    'gain_scale': gain_scale.detach().cpu().numpy().tolist(),
                    'delta_T_max_pos': delta_T_max_pos.detach().cpu().numpy().tolist(),
                    'delta_T_max_neg': delta_T_max_neg.detach().cpu().numpy().tolist(),
                }
            }

            # Update the current visualization data
            self.viz_data = viz_snapshot
            # Also add to history
            self.viz_history.append(viz_snapshot)

        return force, torque