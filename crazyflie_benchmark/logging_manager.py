"""
Logging manager for the Crazyflie Sweeper package.

Handles configuration and management of data logging from the Crazyflie drone.
Provides a thread-safe structure for storing and accessing log data.
"""
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger

from .config import FlightConfig
from .connection import ConnectionManager
from .utils import get_var_from_data

logger = logging.getLogger(__name__)


class LoggingManager:
    """
    Manages data logging from the Crazyflie drone.
    
    Configures logging, handles data callbacks, and provides
    thread-safe access to log data.
    """
    
    def __init__(self, connection: ConnectionManager, config: FlightConfig):
        """
        Initialize the logging manager.
        
        Args:
            connection: Connection manager for the Crazyflie
            config: Configuration parameters for logging
        """
        self.connection = connection
        self.config = config
        self.log_configs: List[LogConfig] = [] # Store all active log configs
        
        # Thread safety for data access
        self._lock = threading.RLock()
        
        # Storage for historical log data, organized by maneuver, then by variable name
        # self._log_data[maneuver_name][variable_name] = [(timestamp, value), ...]
        self._log_data: Dict[str, Dict[str, List[Tuple[int, Any]]]] = {}
        
        # Storage for the latest value of each logged variable (for real-time state)
        self._latest_log_values: Dict[str, Any] = {}
        
        # Current maneuver/test being executed
        self._current_maneuver: str = "init"
        
        # Detection of firmware variable names
        self.active_position_vars: Dict[str, str] = {}
        self.active_velocity_vars: Dict[str, str] = {}
        self.active_orientation_vars: Dict[str, str] = {} # Though roll/pitch/yaw are standard
        self.firmware_vars_detected = False # Overall detection status
        
        # Current position tracking (most recent values from _latest_log_values)
        self.current_position: Dict[str, float] = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.current_velocity: Dict[str, float] = {'vx': 0.0, 'vy': 0.0, 'vz': 0.0}
        
        # Hover estimation data
        self.hover_data: List[Dict[str, Any]] = []
        self.z_velocity_data: List[float] = [] # For hover estimation
        self.altitude_data: List[float] = []   # For hover estimation

        # Initialize for the "init" maneuver
        self.set_current_maneuver("init")
        
    def setup_logging(self) -> bool:
        """
        Set up the logging configuration for data collection.
        
        Returns:
            True if logging was set up successfully, False otherwise
        """
        if not self.connection.is_connected:
            logger.error("Cannot set up logging: Not connected.")
            return False
        
        # Clear previous configs if any
        self.log_configs = [] 
        
        # Check connection type and handle accordingly
        if hasattr(self.connection, 'scf') and self.connection.scf:
            # This is a CFLibConnection - use cflib logging
            return self._setup_cflib_logging()
        else:
            # This is likely a SimulatorConnection - use simulator logging
            return self._setup_simulator_logging()
    
    def _setup_cflib_logging(self) -> bool:
        """
        Set up logging for real Crazyflie hardware using cflib.
        
        Returns:
            True if logging was set up successfully, False otherwise
        """
        try:
            # Helper to create and start a log configuration
            def add_and_start_log_config(name: str, variables: Dict[str, str]) -> Optional[LogConfig]:
                try:
                    log_conf = LogConfig(name=name, period_in_ms=1000 // self.config.log_rate_hz)
                    for var_name, var_type in variables.items():
                        log_conf.add_variable(var_name, var_type)
                    
                    self.connection.scf.cf.log.add_config(log_conf)
                    log_conf.data_received_cb.add_callback(self._log_data_callback)
                    log_conf.error_cb.add_callback(self._log_error_callback)
                    log_conf.start()
                    self.log_configs.append(log_conf)
                    logger.info(f"Logging for '{name}' started with variables: {list(variables.keys())}")
                    return log_conf
                except Exception as e:
                    logger.warning(f"Could not set up logging for '{name}': {str(e)}. Variables: {list(variables.keys())}")
                    return None

            # 1. Basic attitude (roll, pitch, yaw)
            attitude_vars = {
                "stabilizer.roll": "float", 
                "stabilizer.pitch": "float", 
                "stabilizer.yaw": "float"
            }
            if not add_and_start_log_config("attitude_log", attitude_vars):
                logger.error("Failed to set up critical attitude logging. Aborting setup.")
                return False # Critical, so abort
            
            # 2. Thrust
            thrust_vars = {"stabilizer.thrust": "uint16_t"}
            add_and_start_log_config("thrust_log", thrust_vars)
            
            # 3. Gyroscope data
            gyro_vars = {"gyro.x": "float", "gyro.y": "float", "gyro.z": "float"}
            add_and_start_log_config("gyro_log", gyro_vars)
            
            # 4. Accelerometer data
            acc_vars = {"acc.x": "float", "acc.y": "float", "acc.z": "float"}
            add_and_start_log_config("acc_log", acc_vars)
            
            # 5. Position data - try one variable from each axis's list
            position_logging_success = False
            for axis, possible_vars in self.config.position_var_mapping.items():
                if possible_vars:
                    var_to_log = possible_vars[0] 
                    if add_and_start_log_config(f"pos_{axis}_log", {var_to_log: "float"}):
                        position_logging_success = True
            
            if not position_logging_success:
                logger.warning("Could not set up any position logging. Position-dependent features may fail.")
            
            # 6. Velocity data - try one variable from each axis's list
            velocity_logging_success = False
            for axis, possible_vars in self.config.velocity_var_mapping.items():
                if possible_vars:
                    var_to_log = possible_vars[0]
                    if add_and_start_log_config(f"vel_{axis}_log", {var_to_log: "float"}):
                        velocity_logging_success = True
            
            if not velocity_logging_success:
                logger.warning("Could not set up any velocity logging.")
            
            logger.info("Logging setup process completed for real hardware.")
            return True
            
        except Exception as e:
            logger.error(f"Unexpected critical error during cflib logging setup: {str(e)}", exc_info=True)
            return False
            
    def _setup_simulator_logging(self) -> bool:
        """
        Set up logging for simulator connection.
        For the simulator, we'll use a polling approach to get state data.
        
        Returns:
            True if logging was set up successfully, False otherwise
        """
        try:
            # Start a thread to periodically poll state from the simulator
            self._simulator_polling = True
            self._simulator_polling_thread = threading.Thread(
                target=self._simulator_polling_loop,
                daemon=True
            )
            self._simulator_polling_thread.start()
            
            # Set up the active variable mappings for simulator
            self.active_position_vars = {
                "x": "position.x",
                "y": "position.y",
                "z": "position.z"
            }
            
            self.active_velocity_vars = {
                "vx": "velocity.x",
                "vy": "velocity.y",
                "vz": "velocity.z"
            }
            
            self.active_orientation_vars = {
                "roll": "stabilizer.roll",
                "pitch": "stabilizer.pitch",
                "yaw": "stabilizer.yaw"
            }
            
            self.firmware_vars_detected = True
            
            logger.info("Logging setup process completed for simulator.")
            return True
            
        except Exception as e:
            logger.error(f"Unexpected error during simulator logging setup: {str(e)}", exc_info=True)
            self._simulator_polling = False
            return False
            
    def _simulator_polling_loop(self):
        """
        Background thread that polls the simulator for state updates.
        
        All attitude angles are expected to be in degrees, matching the real Crazyflie's
        convention. If the simulator returns angles in radians, they'll be automatically 
        converted to degrees in the SimulatorConnection.get_state method.
        """
        poll_interval = 1.0 / self.config.log_rate_hz
        
        while self._simulator_polling and self.connection.is_connected:
            try:
                # Get the current state from the simulator
                # The SimulatorConnection.get_state() should convert any radians to degrees
                # for consistent data handling with real hardware
                state = self.connection.get_state()
                
                if state:
                    timestamp = int(time.time() * 1000)  # Current time in ms
                    
                    # Process the state data like we would in the callback
                    with self._lock:
                        # Flatten the nested dictionary to match cflib format
                        flattened_data = {}
                        
                        # Process position
                        if "position" in state:
                            for axis in ["x", "y", "z"]:
                                if axis in state["position"]:
                                    flattened_data[f"position.{axis}"] = state["position"][axis]
                        
                        # Process velocity
                        if "velocity" in state:
                            for axis in ["x", "y", "z"]:
                                if axis in state["velocity"]:
                                    flattened_data[f"velocity.{axis}"] = state["velocity"][axis]
                        
                        # Process attitude
                        if "attitude" in state:
                            for angle in ["roll", "pitch", "yaw"]:
                                if angle in state["attitude"]:
                                    flattened_data[f"stabilizer.{angle}"] = state["attitude"][angle]
                        
                        # Process gyroscope data
                        if "gyro" in state:
                            for axis in ["x", "y", "z"]:
                                if axis in state["gyro"]:
                                    flattened_data[f"gyro.{axis}"] = state["gyro"][axis]
                        
                        # Process accelerometer data
                        if "acc" in state:
                            for axis in ["x", "y", "z"]:
                                if axis in state["acc"]:
                                    flattened_data[f"acc.{axis}"] = state["acc"][axis]
                        
                        # Update latest values
                        self._latest_log_values.update(flattened_data)
                        
                        # Store historical data for the current maneuver
                        if self._current_maneuver not in self._log_data:
                            self._log_data[self._current_maneuver] = {}
                            
                        current_maneuver_logs = self._log_data[self._current_maneuver]
                        for var_name, value in flattened_data.items():
                            if var_name not in current_maneuver_logs:
                                current_maneuver_logs[var_name] = []
                            current_maneuver_logs[var_name].append((timestamp, value))
                            
                        # Update current position and velocity
                        position_updated = False
                        for axis, var_name in self.active_position_vars.items():
                            if var_name in flattened_data:
                                try:
                                    self.current_position[axis] = float(flattened_data[var_name])
                                    position_updated = True
                                except (ValueError, TypeError):
                                    pass
                                    
                        velocity_updated = False
                        for axis, var_name in self.active_velocity_vars.items():
                            if var_name in flattened_data:
                                try:
                                    self.current_velocity[axis] = float(flattened_data[var_name])
                                    velocity_updated = True
                                except (ValueError, TypeError):
                                    pass
            
            except Exception as e:
                logger.warning(f"Error in simulator polling loop: {e}")
                
            # Sleep until next poll
            time.sleep(poll_interval)
        
        logger.info("Simulator polling thread stopped.")
    
    def _log_data_callback(self, timestamp: int, data: Dict[str, Any], logconf: LogConfig) -> None:
        """
        Callback when data is received from the logging framework.
        
        Args:
            timestamp: Timestamp of the data point (ms)
            data: Dictionary of variable names and values for THIS logconf
            logconf: The log configuration that produced this data
        """
        with self._lock:
            # Update the consolidated latest values for real-time state
            self._latest_log_values.update(data)

            # Detect firmware variables if not already done, using the full latest data
            if not self.firmware_vars_detected:
                self._detect_firmware_variables(self._latest_log_values.copy()) 
            
            # Store historical data for the current maneuver
            # For each variable in the incoming 'data', append (timestamp, value)
            if self._current_maneuver not in self._log_data:
                # This case should ideally be handled by set_current_maneuver
                self._log_data[self._current_maneuver] = {}

            current_maneuver_logs = self._log_data[self._current_maneuver]
            for var_name, value in data.items():
                if var_name not in current_maneuver_logs:
                    current_maneuver_logs[var_name] = []
                current_maneuver_logs[var_name].append((timestamp, value))

            # Update current_position and current_velocity using detected active variables
            # These are derived from _latest_log_values for immediate state
            position_updated = False
            for axis, var_name in self.active_position_vars.items():
                if var_name in self._latest_log_values:
                    try:
                        self.current_position[axis] = float(self._latest_log_values[var_name])
                        position_updated = True
                    except (ValueError, TypeError):
                        logger.debug(f"Could not convert position value {self._latest_log_values[var_name]} for {var_name} to float.")

            velocity_updated = False
            vz_for_hover = 0.0 # Default value
            for axis, var_name in self.active_velocity_vars.items():
                if var_name in self._latest_log_values:
                    try:
                        self.current_velocity[axis] = float(self._latest_log_values[var_name])
                        velocity_updated = True
                        if axis == 'vz': 
                            vz_for_hover = float(self._latest_log_values[var_name])
                    except (ValueError, TypeError):
                        logger.debug(f"Could not convert velocity value {self._latest_log_values[var_name]} for {var_name} to float.")
            
            if velocity_updated and 'stabilizer.thrust' in self._latest_log_values:
                self.z_velocity_data.append(vz_for_hover)
                if self._current_maneuver == "hover_stabilization":
                    hover_point = {
                        'vz': vz_for_hover,
                        'thrust': self._latest_log_values.get('stabilizer.thrust'),
                        'z': self.current_position.get('z') 
                    }
                    if all(val is not None for val in hover_point.values()):
                        self.hover_data.append(hover_point)
            
            if position_updated:
                if 'z' in self.current_position:
                     self.altitude_data.append(self.current_position['z'])
            
            if not self.active_position_vars and timestamp % 5000 == 0: 
                logger.warning("Position variables (e.g., stateEstimate.x) have not been detected in logs. "
                               "Ensure your Crazyflie has a working positioning system (Flowdeck, LPS, Mocap) "
                               "and the correct variables are configured in config.py.")
    
    def _log_error_callback(self, logconf: LogConfig, msg: str) -> None:
        logger.error(f"Error in logging configuration '{logconf.name}': {msg}")
    
    def _detect_firmware_variables(self, current_data_snapshot: Dict[str, Any]) -> None:
        position_vars_found_this_call = {}
        for axis, possible_vars in self.config.position_var_mapping.items():
            if axis not in self.active_position_vars: 
                for var_name in possible_vars:
                    if var_name in current_data_snapshot:
                        position_vars_found_this_call[axis] = var_name
                        logger.info(f"Detected position variable for {axis}: {var_name}")
                        break
        self.active_position_vars.update(position_vars_found_this_call)

        velocity_vars_found_this_call = {}
        for axis, possible_vars in self.config.velocity_var_mapping.items():
            if axis not in self.active_velocity_vars: 
                for var_name in possible_vars:
                    if var_name in current_data_snapshot:
                        velocity_vars_found_this_call[axis] = var_name
                        logger.info(f"Detected velocity variable for {axis}: {var_name}")
                        break
        self.active_velocity_vars.update(velocity_vars_found_this_call)

        orientation_vars_found_this_call = {}
        for axis, possible_vars in self.config.orientation_var_mapping.items():
             if axis not in self.active_orientation_vars:
                for var_name in possible_vars: 
                    if var_name in current_data_snapshot:
                        orientation_vars_found_this_call[axis] = var_name
                        break
        self.active_orientation_vars.update(orientation_vars_found_this_call)
        
        all_pos_axes_found = all(axis in self.active_position_vars for axis in self.config.position_var_mapping.keys())
        all_vel_axes_found = all(axis in self.active_velocity_vars for axis in self.config.velocity_var_mapping.keys())
        
        if all_pos_axes_found and all_vel_axes_found : 
            logger.info("All expected position and velocity firmware variables detected.")
            self.firmware_vars_detected = True
        elif (len(self.active_position_vars) > 0 or len(self.active_velocity_vars) > 0) and not self.firmware_vars_detected :
             logger.info(f"Firmware variable detection ongoing: Pos found: {list(self.active_position_vars.keys())}, Vel found: {list(self.active_velocity_vars.keys())}")

    def get_latest_values(self) -> Dict[str, Any]:
        with self._lock:
            return self._latest_log_values.copy()

    def set_current_maneuver(self, maneuver: str) -> None:
        with self._lock:
            if self._current_maneuver == "landing" and maneuver == "data_processing":
                time.sleep(0.1)
            self._current_maneuver = maneuver
            logger.info(f"Current maneuver set to: {maneuver}")
            if maneuver not in self._log_data:
                self._log_data[maneuver] = {} # Initialize as empty dict for variables
    
    def get_log_data(self) -> Dict[str, Dict[str, List[Tuple[int, Any]]]]:
        """
        Get a copy of all historical logged data.
        Format: {maneuver_name: {var_name: [(timestamp, value), ...], ...}, ...}
        """
        with self._lock:
            # Return a deep copy
            return {
                maneuver: {
                    var_name: var_data_list[:] # Copy the list of tuples
                    for var_name, var_data_list in var_data_dict.items()
                }
                for maneuver, var_data_dict in self._log_data.items()
            }
    
    def get_log_data_for_maneuver(self, maneuver: str) -> Dict[str, List[Tuple[int, Any]]]:
        """
        Get logged data for a specific maneuver.
        Format: {var_name: [(timestamp, value), ...], ...}
        """
        with self._lock:
            if maneuver not in self._log_data:
                return {}
            # Return a deep copy for the specific maneuver
            return {
                var_name: var_data_list[:]
                for var_name, var_data_list in self._log_data[maneuver].items()
            }
    
    def get_current_position(self) -> Dict[str, float]:
        with self._lock:
            return self.current_position.copy()
    
    def get_current_velocity(self) -> Dict[str, float]:
        with self._lock:
            return self.current_velocity.copy()
    
    def reset_log_data(self) -> None:
        with self._lock:
            self._log_data = {}
            self._latest_log_values = {} 
            self.hover_data = []
            self.z_velocity_data = []
            self.altitude_data = []
            self.active_position_vars = {}
            self.active_velocity_vars = {}
            self.active_orientation_vars = {}
            self.firmware_vars_detected = False
            logger.info("Log data, latest values, and firmware variable detection reset.")
            # Re-initialize for the "init" maneuver after reset
            self.set_current_maneuver("init")

    def stop_logging(self) -> None:
        # Stop cflib logging if active
        if self.log_configs:
            logger.info(f"Stopping {len(self.log_configs)} active log configurations.")
            for log_conf in self.log_configs:
                try:
                    if hasattr(log_conf, 'running') and log_conf.running:
                        log_conf.stop()
                        logger.info(f"Stopped log configuration: {log_conf.name}")
                    else:
                        try:
                            log_conf.stop()
                            logger.info(f"Stopped log configuration: {log_conf.name}")
                        except AttributeError:
                            logger.debug(f"Log configuration {log_conf.name} may have already been stopped or lacks 'running' attribute.")
                except Exception as e:
                    logger.warning(f"Error stopping log configuration {log_conf.name}: {str(e)}")
            self.log_configs = [] 
            logger.info("All logging configurations instructed to stop.")
        
        # Stop simulator polling if active
        if hasattr(self, '_simulator_polling') and self._simulator_polling:
            logger.info("Stopping simulator polling thread.")
            self._simulator_polling = False
            if hasattr(self, '_simulator_polling_thread'):
                try:
                    self._simulator_polling_thread.join(timeout=2.0)
                    logger.info("Simulator polling thread stopped.")
                except Exception as e:
                    logger.warning(f"Error stopping simulator polling thread: {e}")

    def log_control_command(self, roll: float, pitch: float, yaw_rate: float, thrust: float) -> None:
        """
        Log a control command that was sent to the drone.
        
        This allows recording the stimulus signals for analysis alongside the drone response.
        
        Args:
            roll: Roll angle in degrees
            pitch: Pitch angle in degrees
            yaw_rate: Yaw rate in degrees/s
            thrust: Thrust value (0-65535 for real hardware, 0-1 for simulator)
        """
        timestamp = int(time.time() * 1000)  # Current time in ms
        
        with self._lock:
            if self._current_maneuver not in self._log_data:
                self._log_data[self._current_maneuver] = {}
            
            # Log each command component separately
            for name, value in [
                ("command.roll", roll),
                ("command.pitch", pitch),
                ("command.yaw_rate", yaw_rate),
                ("command.thrust", thrust)
            ]:
                if name not in self._log_data[self._current_maneuver]:
                    self._log_data[self._current_maneuver][name] = []
                
                self._log_data[self._current_maneuver][name].append((timestamp, value))
                
            # Update latest values
            self._latest_log_values["command.roll"] = roll
            self._latest_log_values["command.pitch"] = pitch
            self._latest_log_values["command.yaw_rate"] = yaw_rate
            self._latest_log_values["command.thrust"] = thrust
