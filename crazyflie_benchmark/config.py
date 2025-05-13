"""
Configuration module for the Crazyflie Sweeper package.

Defines configuration parameters for flights, tests, and safety limits.
Supports loading configuration from JSON or YAML files.
"""
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Union, Any

import yaml

# Try importing cflib - required for real hardware
try:
    from cflib.utils import uri_helper
    CFLIB_AVAILABLE = True
except ImportError:
    CFLIB_AVAILABLE = False
    uri_helper = None

logger = logging.getLogger(__name__)


@dataclass
class FlightConfig:
    """
    Configuration parameters for Crazyflie flights and tests.
    
    All angles are specified in degrees and all angular rates in degrees/second
    throughout the entire codebase, regardless of whether using the real hardware
    or the simulator. Any necessary unit conversions are handled internally.
    """
    
    # Connection parameters
    connection_type: str = "cflib"  # 'cflib' for real hardware, 'simulator' for simulation
    uri: str = field(default_factory=lambda: uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E6') if CFLIB_AVAILABLE else "")
    
    # Simulator connection parameters (used when connection_type is 'simulator')
    sim_host: str = "localhost"  # Simulator host
    sim_port: int = 8000  # Simulator port
    # When using simulator, angles from the simulator (which may be in radians)
    # are automatically converted to degrees for consistency with the real hardware
    
    # Flight parameters
    hover_thrust: int = 46641  # Initial hover thrust (0-65535)
    takeoff_thrust_initial: int = 48141  # Initial takeoff thrust, drone should lift off
    
    # Maneuver parameters
    default_yaw_rate: float = 0.0  # Default yaw rate (deg/s)
    test_roll_angle_deg: float = 8.0  # Test roll angle in degrees
    test_pitch_angle_deg: float = 8.0  # Test pitch angle in degrees
    test_thrust_increment: int = 4000  # Test thrust increment/decrement
    step_duration_s: float = 0.75  # Duration of step input
    hold_neutral_duration_s: float = 1.5  # Duration to hold neutral after a step
    inter_maneuver_delay_s: float = 2.0  # Delay between different maneuvers
    
    # Safety limits
    # The code will automatically hover in place if these limits are exceeded
    safety_max_roll_pitch_deg: float = 15.0  # Max roll/pitch angle in degrees
    safety_max_thrust: int = 48000  # Max thrust
    safety_min_thrust_flight: int = 20000  # Min thrust during flight
    safety_radius_m: float = 2.5  # Max allowed distance from takeoff (m)
    safety_max_height_m: float = 2.0  # Max allowed height (m)
    
    # Logging parameters
    log_rate_hz: int = 100  # Logging frequency in Hz
    
    # Variable name mappings for different firmware versions
    position_var_mapping: Dict[str, List[str]] = field(default_factory=lambda: {
        "x": ["kalman.stateX", "stateEstimate.x"],
        "y": ["kalman.stateY", "stateEstimate.y"],
        "z": ["kalman.stateZ", "stateEstimate.z"]
    })
    
    velocity_var_mapping: Dict[str, List[str]] = field(default_factory=lambda: {
        "vx": ["kalman.statePX", "stateEstimate.vx"],
        "vy": ["kalman.statePY", "stateEstimate.vy"],
        "vz": ["kalman.statePZ", "stateEstimate.vz"]
    })
    
    orientation_var_mapping: Dict[str, List[str]] = field(default_factory=lambda: {
        "roll": ["stabilizer.roll"],
        "pitch": ["stabilizer.pitch"],
        "yaw": ["stabilizer.yaw"]
    })
    
    # Test parameters
    # Note: These parameters define the maximum amplitudes.
    # The actual test amplitudes will be reduced by a safety factor (0.7)
    # in main.py to ensure the drone stays within safety limits.
    
    # Step test parameters
    step_test_amplitude_roll: float = 5.0  # Roll step amplitude (degrees)
    step_test_amplitude_pitch: float = 5.0  # Pitch step amplitude (degrees)
    step_test_amplitude_thrust: int = 2000  # Thrust step amplitude
    step_test_duration: float = 1.5  # Step test duration (seconds)
    
    # Impulse test parameters
    impulse_amplitude_roll: float = 6.0  # Roll impulse amplitude (degrees)
    impulse_amplitude_pitch: float = 6.0  # Pitch impulse amplitude (degrees)
    impulse_amplitude_thrust: int = 2500  # Thrust impulse amplitude
    impulse_duration: float = 0.2  # Impulse duration (seconds)
    
    # Sine sweep test parameters
    sine_sweep_duration: float = 8.0  # Sine sweep duration (seconds)
    sine_start_freq: float = 0.5  # Start frequency (Hz)
    sine_end_freq: float = 4.0  # End frequency (Hz)
    sine_amplitude_roll: float = 3.0  # Roll sine amplitude (degrees)
    sine_amplitude_pitch: float = 3.0  # Pitch sine amplitude (degrees)
    sine_amplitude_thrust: int = 1200  # Thrust sine amplitude
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure takeoff_thrust_initial is derived from hover_thrust if not explicitly set
        if self.takeoff_thrust_initial == 45500 and self.hover_thrust != 44000:
            self.takeoff_thrust_initial = self.hover_thrust + 1500
            
        # Validate connection type
        if self.connection_type.lower() not in ["cflib", "simulator"]:
            logger.warning(f"Invalid connection_type: {self.connection_type}. Using default: cflib")
            self.connection_type = "cflib"
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'FlightConfig':
        """
        Load configuration from a JSON or YAML file.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            FlightConfig instance with parameters from the file
        """
        if not os.path.exists(file_path):
            logger.warning(f"Configuration file {file_path} not found. Using default configuration.")
            return cls()
        
        try:
            with open(file_path, 'r') as f:
                if file_path.endswith('.json'):
                    config_dict = json.load(f)
                elif file_path.endswith(('.yaml', '.yml')):
                    config_dict = yaml.safe_load(f)
                else:
                    logger.warning(f"Unsupported file format: {file_path}. Using default configuration.")
                    return cls()
            
            return cls(**config_dict)
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            logger.error(f"Error parsing configuration file {file_path}: {e}")
            return cls()
        except Exception as e:
            logger.error(f"Error loading configuration from {file_path}: {e}")
            return cls()
    
    def save_to_file(self, file_path: str) -> bool:
        """
        Save configuration to a JSON or YAML file.
        
        Args:
            file_path: Path to save the configuration file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'w') as f:
                if file_path.endswith('.json'):
                    json.dump(asdict(self), f, indent=2)
                elif file_path.endswith(('.yaml', '.yml')):
                    yaml.dump(asdict(self), f, default_flow_style=False)
                else:
                    logger.warning(f"Unsupported file format: {file_path}. Using JSON format.")
                    json.dump(asdict(self), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving configuration to {file_path}: {e}")
            return False 