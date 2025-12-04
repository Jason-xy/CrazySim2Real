"""
ROS2 visualization bridge for CrazySim2Real.

Polls the simulator REST API for controller debug data and republishes to ROS2
topics for tools like PlotJuggler. Keeps timestamps aligned by sampling the ROS2
clock once per publish and reusing it for all messages.
"""
import argparse
import sys
import time
from typing import Optional

import requests
from rclpy.time import Time

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rcl_interfaces.msg import SetParametersResult
    from geometry_msgs.msg import Vector3Stamped
    from std_msgs.msg import Float32MultiArray
except ImportError as exc:
    print("rclpy and ROS2 messages are required. Run inside the ROS2 environment.", file=sys.stderr)
    raise


class Ros2VisClient(Node):
    """Polls REST API and publishes controller telemetry to ROS2."""

    def __init__(self, api_url: str, hz: float):
        super().__init__("cf_ros2_vis_client")
        self.api_url = api_url.rstrip("/")
        self.period = 1.0 / hz if hz > 0 else 0.01
        self.session = requests.Session()
        self.timeout = 0.05

        qos = rclpy.qos.QoSProfile(depth=10)
        self.pub_att_sp = self.create_publisher(Vector3Stamped, "attitude_setpoint", qos)
        self.pub_att = self.create_publisher(Vector3Stamped, "attitude", qos)
        self.pub_gyro = self.create_publisher(Vector3Stamped, "gyro", qos)
        self.pub_rate_cmd = self.create_publisher(Vector3Stamped, "rate_cmd", qos)
        self.pub_rate_sp = self.create_publisher(Vector3Stamped, "rate_setpoint", qos)
        self.pub_thrust = self.create_publisher(Float32MultiArray, "thrust_pwm", qos)
        self.pub_motor = self.create_publisher(Float32MultiArray, "motor_pwm", qos)
        self.pub_force = self.create_publisher(Vector3Stamped, "force_body", qos)
        self.pub_torque = self.create_publisher(Vector3Stamped, "torque_body", qos)
        self.pub_pos_sp = self.create_publisher(Vector3Stamped, "position_setpoint", qos)
        self.pub_pos = self.create_publisher(Vector3Stamped, "position", qos)
        self.pub_vel_sp = self.create_publisher(Vector3Stamped, "velocity_setpoint", qos)
        self.pub_vel = self.create_publisher(Vector3Stamped, "velocity", qos)

        self.timer = self.create_timer(self.period, self._tick)

        # ROS parameters for gain tuning via rqt
        self._declare_pid_params()
        self._sync_params_from_api()
        self.add_on_set_parameters_callback(self._on_params_updated)

    def _vec_msg(self, vec, stamp):
        msg = Vector3Stamped()
        msg.header.stamp = stamp
        msg.vector.x = float(vec[0])
        msg.vector.y = float(vec[1])
        msg.vector.z = float(vec[2])
        return msg

    def _array_msg(self, arr):
        msg = Float32MultiArray()
        msg.data = [float(x) for x in arr]
        return msg

    def _fetch_debug(self) -> Optional[dict]:
        try:
            resp = self.session.get(f"{self.api_url}/controller/debug", timeout=self.timeout)
            if resp.status_code != 200:
                return None
            data = resp.json()
            return data if data else None
        except requests.RequestException:
            return None

    def _fetch_params(self) -> Optional[dict]:
        try:
            resp = self.session.get(f"{self.api_url}/controller/params", timeout=self.timeout)
            if resp.status_code != 200:
                return None
            return resp.json()
        except requests.RequestException:
            return None

    def _declare_pid_params(self):
        """Declare ROS parameters for PID gains."""
        defaults = {
            "roll_rate_kp": 0.0, "roll_rate_ki": 0.0, "roll_rate_kd": 0.0, "roll_rate_i_limit": 0.0,
            "pitch_rate_kp": 0.0, "pitch_rate_ki": 0.0, "pitch_rate_kd": 0.0, "pitch_rate_i_limit": 0.0,
            "yaw_rate_kp": 0.0, "yaw_rate_ki": 0.0, "yaw_rate_kd": 0.0, "yaw_rate_i_limit": 0.0,
            "roll_kp": 0.0, "roll_ki": 0.0, "roll_kd": 0.0, "roll_i_limit": 0.0,
            "pitch_kp": 0.0, "pitch_ki": 0.0, "pitch_kd": 0.0, "pitch_i_limit": 0.0,
            "yaw_kp": 0.0, "yaw_ki": 0.0, "yaw_kd": 0.0, "yaw_i_limit": 0.0,
        }
        for name, val in defaults.items():
            self.declare_parameter(name, val)

    def _sync_params_from_api(self):
        """Fetch current gains from REST and push into ROS parameters."""
        params = self._fetch_params()
        if not params:
            return

        def set_param(name, value):
            if value is None:
                return
            self.set_parameters([Parameter(name, value=value)])

        pid_map = {
            "pid_roll_rate": "roll_rate",
            "pid_pitch_rate": "pitch_rate",
            "pid_yaw_rate": "yaw_rate",
            "pid_roll": "roll",
            "pid_pitch": "pitch",
            "pid_yaw": "yaw",
        }
        for key, prefix in pid_map.items():
            gains = params.get(key, {})
            set_param(f"{prefix}_kp", gains.get("kp"))
            set_param(f"{prefix}_ki", gains.get("ki"))
            set_param(f"{prefix}_kd", gains.get("kd"))
            set_param(f"{prefix}_i_limit", gains.get("i_limit"))

    def _on_params_updated(self, params):
        """Push modified params back to simulator REST."""
        payload = {}
        for p in params:
            if not isinstance(p.value, (int, float)):
                continue
            name = p.name
            for prefix in ["roll_rate", "pitch_rate", "yaw_rate", "roll", "pitch", "yaw"]:
                if name.startswith(prefix + "_"):
                    field = name[len(prefix) + 1 :]
                    payload.setdefault(prefix, {})[field] = float(p.value)
                    break

        if payload:
            try:
                self.session.post(f"{self.api_url}/controller/params", json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                self.get_logger().warn(f"Failed to push params to simulator: {exc}")
        return SetParametersResult(successful=True)

    def _tick(self):
        dbg = self._fetch_debug()
        if not dbg:
            return

        sim_ts = dbg.get("timestamp") if isinstance(dbg, dict) else None
        if sim_ts is None:
            stamp = self.get_clock().now().to_msg()
        else:
            stamp = Time(seconds=float(sim_ts)).to_msg()

        att_sp = dbg.get("attitude_desired")
        att = dbg.get("attitude")
        gyro = dbg.get("gyro")
        roll_cmd = dbg.get("roll_cmd")
        pitch_cmd = dbg.get("pitch_cmd")
        yaw_cmd = dbg.get("yaw_cmd")
        thrust_pwm = dbg.get("thrust_pwm")
        motor_pwm = dbg.get("motor_pwm")
        force = dbg.get("force")
        torque = dbg.get("torque")
        rate_sp = dbg.get("rate_desired")
        rate_actual = dbg.get("rate_actual")
        pos_sp = dbg.get("position_setpoint")
        pos = dbg.get("position")
        vel_sp = dbg.get("velocity_setpoint")
        vel = dbg.get("velocity")

        try:
            # Controller debug data is already in degrees/deg/s; publish without conversion
            self.pub_att_sp.publish(self._vec_msg(att_sp[0], stamp))
            self.pub_att.publish(self._vec_msg(att[0], stamp))
            self.pub_gyro.publish(self._vec_msg(gyro[0], stamp))
            if rate_sp:
                self.pub_rate_sp.publish(self._vec_msg(rate_sp[0], stamp))
            if rate_actual:
                self.pub_rate_cmd.publish(self._vec_msg(rate_actual[0], stamp))
            else:
                self.pub_rate_cmd.publish(self._vec_msg([roll_cmd[0], pitch_cmd[0], yaw_cmd[0]], stamp))
            self.pub_thrust.publish(self._array_msg([thrust_pwm[0]]))
            self.pub_motor.publish(self._array_msg(motor_pwm[0]))
            self.pub_force.publish(self._vec_msg(force[0], stamp))
            self.pub_torque.publish(self._vec_msg(torque[0], stamp))
            if pos_sp:
                self.pub_pos_sp.publish(self._vec_msg(pos_sp[0], stamp))
            if pos:
                self.pub_pos.publish(self._vec_msg(pos[0], stamp))
            if vel_sp:
                self.pub_vel_sp.publish(self._vec_msg(vel_sp[0], stamp))
            if vel:
                self.pub_vel.publish(self._vec_msg(vel[0], stamp))
        except (TypeError, IndexError):
            # Skip malformed data silently
            return


def main():
    parser = argparse.ArgumentParser(description="ROS2 viz client for CrazySim2Real")
    parser.add_argument("--host", default="http://localhost:8000", help="Simulator REST base URL")
    parser.add_argument("--hz", type=float, default=300.0, help="Publish rate (Hz)")
    args = parser.parse_args()

    rclpy.init(args=None)
    node = Ros2VisClient(api_url=args.host, hz=args.hz)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
