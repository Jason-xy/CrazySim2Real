"""Simulator flight references and tracking math, independent of Isaac Lab."""

from dataclasses import dataclass
import math

import numpy as np


GRAVITY = 9.81
TRAJECTORIES = {
    "circle": (0.8, 1.2, 1.5),
    "figure8": (0.55, 0.8, 1.0),
    "helix": (0.8, 1.2, 1.5),
    "shuttle": (0.6, 0.9, 1.2),
}
KP = np.array((2.0, 2.0, 4.0))
KD = np.array((2.8, 2.8, 3.0))
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(16)


def wrap_degrees(value):
    return (value + 180.0) % 360.0 - 180.0


def smooth_step(t, start, end):
    u = np.clip((np.asarray(t) - start) / (end - start), 0, 1)
    return 10 * u**3 - 15 * u**4 + 6 * u**5, 30 * u**2 * (1 - u)**2 / (end - start)


@dataclass
class Reference:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    yaw: float
    yaw_rate: float = 0.0

    @classmethod
    def hover(cls, position, yaw):
        return cls(np.asarray(position, dtype=float), np.zeros(3), np.zeros(3), float(yaw))


class Trajectory:
    """Analytic derivatives of seeded paths with smoothly scheduled phase speed."""

    def __init__(self, name, duration=48.0, height=4.0, seed=0, origin=(0, 0, 0)):
        if name not in TRAJECTORIES or not math.isfinite(duration) or duration < 24:
            raise ValueError("Trajectories require a known name and at least 24 seconds")
        if not math.isfinite(height) or height < 1.5:
            raise ValueError("Agile height must be at least 1.5 m")
        self.name, self.duration, self.height = name, float(duration), float(height)
        self.seed, self.origin = int(seed), tuple(origin)
        rng = np.random.default_rng(seed)
        self.rotation = float(rng.uniform(-math.pi, math.pi))
        self.direction = int(rng.choice((-1, 1)))
        c, s = math.cos(self.rotation), math.sin(self.rotation)
        self.rotation_matrix = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
        self.levels = TRAJECTORIES[name]
        a, b = duration / 3, 2 * duration / 3
        self.ramps = ((0, self.levels[0], 0, 4),
                      (self.levels[0], self.levels[1], a, a + 4),
                      (self.levels[1], self.levels[2], b, b + 4),
                      (self.levels[2], 0, duration - 4, duration))
        self.knots = sorted({float(t) for ramp in self.ramps for t in ramp[2:]})
        self.integrals = [0.0]
        for lo, hi in zip(self.knots, self.knots[1:]):
            self.integrals.append(self.integrals[-1] + self._integrate_speed(lo, hi))

    def speed(self, t):
        value, derivative = np.zeros_like(np.asarray(t), dtype=float), np.zeros_like(np.asarray(t), dtype=float)
        for a, b, start, end in self.ramps:
            h, dh = smooth_step(t, start, end)
            value += (b - a) * h
            derivative += (b - a) * dh
        if self.name == "shuttle":
            frequency = 2 * math.pi / 8
            modulation = 1 + 0.2 * np.sin(frequency * np.asarray(t))
            derivative = derivative * modulation + value * 0.2 * frequency * np.cos(frequency * np.asarray(t))
            value *= modulation
        return self.direction * value, self.direction * derivative

    def _integrate_speed(self, lo, hi):
        if hi <= lo:
            return 0.0
        times = lo + (hi - lo) * (_NODES + 1) / 2
        # Quadrature is split at every polynomial boundary; derivatives remain analytic.
        return float((hi - lo) / 2 * np.dot(_WEIGHTS, self.speed(times)[0]))

    def phase(self, t):
        t = float(np.clip(t, 0, self.duration))
        index = min(len(self.knots) - 2, max(0, int(np.searchsorted(self.knots, t, side="right")) - 1))
        return self.integrals[index] + self._integrate_speed(self.knots[index], t)

    def sample(self, t):
        t = float(np.clip(t, 0, self.duration))
        theta, (w, dw) = self.phase(t), self.speed(t)
        s, c = math.sin(theta), math.cos(theta)
        if self.name == "figure8":
            p = np.array((2 * s, math.sin(2 * theta), 0.0))
            d = np.array((2 * c, 2 * math.cos(2 * theta), 0.0))
            dd = np.array((-2 * s, -4 * math.sin(2 * theta), 0.0))
        elif self.name == "shuttle":
            axis = np.array((math.sqrt(2), math.sqrt(2), 0))
            p, d, dd = axis * s, axis * c, -axis * s
        else:
            p, d, dd = np.array((2 * (c - 1), 2 * s, 0.0)), np.array((-2 * s, 2 * c, 0.0)), np.array((-2 * c, -2 * s, 0.0))
            if self.name == "helix":
                p[2], d[2], dd[2] = 0.75 * math.sin(theta / 2), 0.375 * math.cos(theta / 2), -0.1875 * math.sin(theta / 2)
        position = self.rotation_matrix @ p + np.array((*self.origin[:2], self.height))
        velocity = self.rotation_matrix @ (d * w)
        acceleration = self.rotation_matrix @ (dd * w * w + d * dw)
        yaw, yaw_rate = self.origin[2], 0.0
        if self.name == "shuttle":
            enter, denter = smooth_step(t, 0, 4)
            leave, dleave = smooth_step(t, self.duration - 4, self.duration)
            envelope = enter * (1 - leave)
            derivative = denter * (1 - leave) - enter * dleave
            frequency = 2 * math.pi * 0.25
            yaw += 60 * envelope * math.sin(frequency * t)
            yaw_rate = 60 * (derivative * math.sin(frequency * t) + envelope * frequency * math.cos(frequency * t))
        return Reference(position, velocity, acceleration, float(yaw), float(yaw_rate))

    def metadata(self):
        return {"name": self.name, "duration_s": self.duration, "height_m": self.height,
                "origin": {"x": self.origin[0], "y": self.origin[1], "yaw_deg": self.origin[2]},
                "seed": self.seed, "rotation_rad": self.rotation, "direction": self.direction,
                "phase_speeds_rad_s": list(self.levels), "speed_intervals_s": [
                    [i * self.duration / 3, (i + 1) * self.duration / 3] for i in range(3)]}


class QuinticTransition:
    """C2 reference handoff matching position, velocity and acceleration."""

    def __init__(self, start, end, duration=4.0):
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Transition duration must be positive")
        self.duration = float(duration)
        p0 = np.r_[start.position, start.yaw]
        p1 = np.r_[end.position, start.yaw + wrap_degrees(end.yaw - start.yaw)]
        v0, v1 = np.r_[start.velocity, start.yaw_rate], np.r_[end.velocity, end.yaw_rate]
        a0, a1 = np.r_[start.acceleration, 0.0], np.r_[end.acceleration, 0.0]
        c0, c1, c2 = p0, v0 * duration, a0 * duration**2 / 2
        rhs = np.stack((p1 - c0 - c1 - c2, v1 * duration - c1 - 2 * c2,
                        a1 * duration**2 - 2 * c2))
        high = np.linalg.solve(np.array(((1, 1, 1), (3, 4, 5), (6, 12, 20))), rhs)
        self.coefficients = np.vstack((c0, c1, c2, high))

    def sample(self, t):
        u = float(np.clip(t / self.duration, 0, 1))
        c = self.coefficients
        p = np.array([u**i for i in range(6)]) @ c
        v = np.array([0, 1, 2 * u, 3 * u**2, 4 * u**3, 5 * u**4]) @ c / self.duration
        a = np.array([0, 0, 2, 6 * u, 12 * u**2, 20 * u**3]) @ c / self.duration**2
        return Reference(p[:3], v[:3], a[:3], float(p[3]), float(v[3]))


def force_angles(acceleration, yaw_deg):
    ax, ay, az = acceleration
    yaw = math.radians(yaw_deg)
    fx = math.cos(yaw) * ax + math.sin(yaw) * ay
    fy = -math.sin(yaw) * ax + math.cos(yaw) * ay
    fz = GRAVITY + az
    return math.degrees(math.atan2(-fy, math.hypot(fx, fz))), math.degrees(math.atan2(fx, fz))


def tracking_command(reference, state, mass, motor_thrust_max):
    p = np.array([state["position"][axis] for axis in "xyz"])
    v = np.array([state["velocity"][axis] for axis in "xyz"])
    a = reference.acceleration + KP * (reference.position - p) + KD * (reference.velocity - v)
    angles = state["orientation"]
    roll, pitch = force_angles(a, angles["yaw"])
    yaw_rate = reference.yaw_rate + 1.5 * wrap_degrees(reference.yaw - angles["yaw"])
    tilt = math.cos(math.radians(angles["roll"])) * math.cos(math.radians(angles["pitch"]))
    thrust = mass * max(0.0, GRAVITY + a[2]) / (4 * motor_thrust_max * max(1e-6, tilt))
    return {"roll": float(np.clip(roll / 30, -1, 1)), "pitch": float(np.clip(pitch / 30, -1, 1)),
            "yaw_rate": float(np.clip(yaw_rate / 120, -1, 1)), "thrust": float(np.clip(thrust, 0, 1))}


def predicted_clearance(state, mass, motor_thrust_max):
    tilt = math.cos(math.radians(state["orientation"]["roll"])) * math.cos(math.radians(state["orientation"]["pitch"]))
    braking = min(3.0, 4 * motor_thrust_max / mass * max(0, tilt) - GRAVITY)
    down = max(0.0, -state["velocity"]["z"])
    if braking <= 0:
        return -math.inf
    return state["position"]["z"] - 0.5 * down - down**2 / (2 * braking)


def validate_reference(path, mass, motor_thrust_max, minimum_height=0.5):
    """Check nominal references, not an assertion of closed-loop flight safety."""
    maximum_tilt, maximum_thrust, minimum = 0.0, 0.0, math.inf
    for t in np.linspace(0, path.duration, max(2, math.ceil(path.duration * 100) + 1)):
        ref = path.sample(t)
        values = np.r_[ref.position, ref.velocity, ref.acceleration, ref.yaw, ref.yaw_rate]
        if not np.isfinite(values).all():
            raise ValueError("Reference has nonfinite values")
        roll, pitch = force_angles(ref.acceleration, ref.yaw)
        force = mass * np.linalg.norm(ref.acceleration + np.array((0, 0, GRAVITY)))
        minimum = min(minimum, float(ref.position[2]))
        maximum_tilt = max(maximum_tilt, abs(roll), abs(pitch))
        maximum_thrust = max(maximum_thrust, float(force / (4 * motor_thrust_max)))
        if minimum < minimum_height - 1e-8:
            raise ValueError("Reference violates the ground clearance margin")
        if GRAVITY + ref.acceleration[2] <= 0 or maximum_tilt > 30 + 1e-6 or abs(ref.yaw_rate) > 120 + 1e-6:
            raise ValueError("Reference exceeds the attitude interface")
        if maximum_thrust >= 1:
            raise ValueError("Reference has insufficient thrust margin")
    return {"minimum_height_m": minimum, "maximum_angle_deg": maximum_tilt,
            "maximum_thrust_fraction": maximum_thrust}
