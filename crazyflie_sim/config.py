"""
Configuration for Crazyflie simulator.

Physical parameters are imported from the CF2.1 BL controller config
to ensure consistency between simulation and controller.
"""
from crazyflie_sim.controllers.cf_controller import config as cf_config

# Simulation parameters
SIM_DT = cf_config.ATTITUDE_UPDATE_DT  # Simulation timestep (s)

# Physics parameters (from CF2.1 BL firmware)
PHYSICS = {
    "mass": cf_config.CF_MASS,
    "arm_length": cf_config.ARM_LENGTH,
    "inertia": [cf_config.INERTIA_XX, cf_config.INERTIA_YY, cf_config.INERTIA_ZZ],
    "gravity": cf_config.GRAVITY,
}

# HTTP Server
SERVER = {
    "host": "localhost",
    "port": 8000,
}

# Logging
LOGGING = {
    "level": "INFO",
    "format": "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
}
