"""
Configuration parameters for the Crazyflie simulator.
"""

# Simulation parameters
SIM_DT = 0.01  # Simulation time step in seconds
SIM_FREQUENCY = 100  # 1/dt Hz

# Controller parameters
POSITION_CONTROLLER = {
    "Kp": {"x": 1.0, "y": 1.0, "z": 1.0},
    "Ki": {"x": 0.0, "y": 0.0, "z": 0.1},
    "Kd": {"x": 0.4, "y": 0.4, "z": 0.4},
    "max_thrust": 1.0,  # Maximum thrust output (normalized)
    "min_thrust": 0.0,  # Minimum thrust output (normalized)
}

ATTITUDE_CONTROLLER = {
    "Krp_ang": [7.5, 7.5],  # Roll/pitch P gain
    "Kdrp_ang": [0.05, 0.05],   # Roll/pitch D gain
    "Kinv_ang_vel_tau": [20.0, 20.0, 5.0]  # Angular velocity inverse time constants
}

# Physics parameters
PHYSICS = {
    "mass": 0.049,  # kg
    "arm_length": 0.046,  # m
    "inertia": [2.16e-5, 2.16e-5, 4.33e-5],  # kg*m^2
    "gravity": 9.81  # m/s^2
}

# HTTP Server settings
SERVER = {
    "host": "localhost",
    "port": 8000
}

# Logging settings
LOGGING = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "format": "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    "file": "crazyflie_sim.log"
} 