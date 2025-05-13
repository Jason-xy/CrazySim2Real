"""
Connection manager for the Crazyflie Sweeper package.

Handles establishing and maintaining connections to the Crazyflie drone.
Implements a context manager interface for clean resource management.
"""
import abc
import logging
import time
import os
import sys
from typing import Callable, Dict, Optional, Any, Union, Tuple, Type

# Try importing cflib - required for real hardware, optional for simulation
try:
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    from cflib.utils import uri_helper
    CFLIB_AVAILABLE = True
except ImportError:
    CFLIB_AVAILABLE = False

# Try importing requests - required for simulation client
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class DroneConnectionBase(abc.ABC):
    """
    Abstract base class for drone connections.
    
    Provides a common interface for connecting to different types of drones
    (real hardware or simulation) and implements a context manager interface.
    """
    
    def __init__(self):
        """Initialize the base connection."""
        self.is_connected = False
        
        # Public event callbacks that can be set by clients
        self.on_connected: Optional[Callable[[str], None]] = None
        self.on_disconnected: Optional[Callable[[str], None]] = None
        self.on_connection_failed: Optional[Callable[[str, str], None]] = None
        self.on_connection_lost: Optional[Callable[[str, str], None]] = None
        
    @abc.abstractmethod
    def connect(self) -> bool:
        """
        Connect to the drone.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the drone."""
        pass
    
    @abc.abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current drone state.
        
        Returns:
            Dictionary containing drone state
        """
        pass
    
    @abc.abstractmethod
    def reset_estimator(self) -> bool:
        """
        Reset the state estimator.
        
        Returns:
            True if the command was sent successfully, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def send_setpoint(self, roll: float, pitch: float, yaw_rate: float, thrust: float) -> bool:
        """
        Send a control setpoint.
        
        Args:
            roll: Roll angle in degrees
            pitch: Pitch angle in degrees
            yaw_rate: Yaw rate in degrees/s
            thrust: Thrust as a normalized value (0-1 for simulator, 0-65535 for real drone)
            
        Returns:
            True if successful, False otherwise
        """
        pass

    @abc.abstractmethod
    def send_position_setpoint(self, x: float, y: float, z: float, yaw: float = 0.0) -> bool:
        """
        Send a position setpoint.
        
        Args:
            x: X position in meters
            y: Y position in meters
            z: Z position in meters
            yaw: Yaw angle in degrees
            
        Returns:
            True if successful, False otherwise
        """
        pass

    @abc.abstractmethod
    def send_hover_setpoint(self, vx: float, vy: float, yaw_rate: float, z: float) -> bool:
        """
        Send a hover setpoint.
        
        Args:
            vx: X velocity in m/s
            vy: Y velocity in m/s
            yaw_rate: Yaw rate in degrees/s
            z: Z position in meters
            
        Returns:
            True if successful, False otherwise
        """
        pass

    @abc.abstractmethod
    def send_stop_setpoint(self) -> bool:
        """
        Send a stop setpoint.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    def __enter__(self) -> 'DroneConnectionBase':
        """
        Enter the context manager.
        
        Returns:
            Self reference for use in 'with' statement
        
        Raises:
            RuntimeError: If connection fails
        """
        if not self.connect():
            raise RuntimeError(f"Failed to connect to drone")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the context manager.
        
        Args:
            exc_type: Exception type, if any
            exc_val: Exception value, if any
            exc_tb: Exception traceback, if any
        """
        self.disconnect()


class CFLibConnection(DroneConnectionBase):
    """
    Connection manager for real Crazyflie drones using CFLib.
    
    Implements a context manager interface to ensure proper initialization
    and cleanup of the connection to the drone.
    """
    
    def __init__(self, uri: str):
        """
        Initialize the connection manager.
        
        Args:
            uri: URI for connection to the Crazyflie
        """
        super().__init__()
        
        if not CFLIB_AVAILABLE:
            raise ImportError("CFLib is not available. Install it with 'pip install cflib'")
        
        self.uri = uri
        self.cf = Crazyflie(rw_cache='./cache')
        self.scf: Optional[SyncCrazyflie] = None
        
        # Bind connection callbacks
        self.cf.connected.add_callback(self._connected)
        self.cf.disconnected.add_callback(self._disconnected)
        self.cf.connection_failed.add_callback(self._connection_failed)
        self.cf.connection_lost.add_callback(self._connection_lost)
    
    def _connected(self, link_uri: str) -> None:
        """
        Callback for successful connection.
        
        Args:
            link_uri: URI of the connected device
        """
        logger.info(f"Successfully connected to {link_uri}")
        self.is_connected = True
        if self.on_connected:
            self.on_connected(link_uri)
    
    def _disconnected(self, link_uri: str) -> None:
        """
        Callback for disconnection.
        
        Args:
            link_uri: URI of the disconnected device
        """
        logger.info(f"Disconnected from {link_uri}")
        self.is_connected = False
        if self.on_disconnected:
            self.on_disconnected(link_uri)
    
    def _connection_failed(self, link_uri: str, msg: str) -> None:
        """
        Callback for connection failure.
        
        Args:
            link_uri: URI of the device
            msg: Error message
        """
        logger.error(f"Connection to {link_uri} failed: {msg}")
        self.is_connected = False
        if self.on_connection_failed:
            self.on_connection_failed(link_uri, msg)
    
    def _connection_lost(self, link_uri: str, msg: str) -> None:
        """
        Callback for connection loss.
        
        Args:
            link_uri: URI of the device
            msg: Error message
        """
        logger.error(f"Connection to {link_uri} lost: {msg}")
        self.is_connected = False
        if self.on_connection_lost:
            self.on_connection_lost(link_uri, msg)
            
    def connect(self) -> bool:
        """
        Connect to the Crazyflie.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Initialize the drivers
            cflib.crtp.init_drivers()
            
            # Create the SyncCrazyflie object
            self.scf = SyncCrazyflie(self.uri, cf=self.cf)
            
            # Connect to the Crazyflie
            self.scf.open_link()
            
            # Wait for the connection callback to confirm successful connection
            if not self.is_connected:
                logger.warning("Connection not confirmed by callback.")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error connecting to {self.uri}: {e}")
            self.is_connected = False
            return False
            
    def disconnect(self) -> None:
        """
        Disconnect from the Crazyflie.
        """
        if self.scf:
            try:
                self.scf.close_link()
            except Exception as e:
                logger.error(f"Error disconnecting from {self.uri}: {e}")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current drone state.
        
        Returns:
            Dictionary containing drone state (limited information for real drones)
        """
        # Real drones don't have a direct "get_state" method - this would typically
        # be handled by the logging framework in an actual implementation
        return {
            "is_connected": self.is_connected
        }
    
    def reset_estimator(self) -> bool:
        """
        Reset the Kalman estimator on the Crazyflie.
        
        Returns:
            True if the command was sent successfully, False otherwise.
        """
        if not self.is_connected or not self.scf:
            logger.error("Cannot reset estimator: Not connected.")
            return False
            
        try:
            logger.info("Resetting Kalman estimator...")
            # The parameter to reset the Kalman filter is 'kalman.resetEstimation'
            # Setting it to 1 triggers the reset.
            self.scf.cf.param.set_value('kalman.resetEstimation', '1')
            # Adding a short delay to allow the Crazyflie to process the reset
            time.sleep(0.2) # 200ms delay
            logger.info("Kalman estimator reset command sent.")
            return True
        except Exception as e:
            logger.error(f"Failed to send Kalman estimator reset command: {e}", exc_info=True)
            return False
    
    def send_setpoint(self, roll: float, pitch: float, yaw_rate: float, thrust: float) -> bool:
        """
        Send a control setpoint to the Crazyflie.
        
        Args:
            roll: Roll angle in degrees
            pitch: Pitch angle in degrees
            yaw_rate: Yaw rate in degrees/s
            thrust: Thrust as a normalized value (0-1), will be converted to 0-65535
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected or not self.scf:
            logger.error("Cannot send setpoint: Not connected.")
            return False
            
        try:
            # Convert normalized thrust (0-1) to Crazyflie thrust (0-65535)
            cf_thrust = int(thrust * 65535)
            
            self.scf.cf.commander.send_setpoint(roll, pitch, yaw_rate, cf_thrust)
            return True
        except Exception as e:
            logger.error(f"Failed to send setpoint: {e}", exc_info=True)
            return False

    def send_position_setpoint(self, x: float, y: float, z: float, yaw: float = 0.0) -> bool:
        """
        Send a position setpoint to the Crazyflie.
        
        Args:
            x: X position in meters
            y: Y position in meters
            z: Z position in meters
            yaw: Yaw angle in degrees
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected or not self.scf:
            logger.error("Cannot send position setpoint: Not connected.")
            return False
            
        try:
            self.scf.cf.commander.send_position_setpoint(x, y, z, yaw)
            return True
        except Exception as e:
            logger.error(f"Failed to send position setpoint: {e}", exc_info=True)
            return False

    def send_hover_setpoint(self, vx: float, vy: float, yaw_rate: float, z: float) -> bool:
        """
        Send a hover setpoint to the Crazyflie.
        
        Args:
            vx: X velocity in m/s
            vy: Y velocity in m/s
            yaw_rate: Yaw rate in degrees/s
            z: Z position in meters
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected or not self.scf:
            logger.error("Cannot send hover setpoint: Not connected.")
            return False
            
        try:
            self.scf.cf.commander.send_hover_setpoint(vx, vy, yaw_rate, z)
            return True
        except Exception as e:
            logger.error(f"Failed to send hover setpoint: {e}", exc_info=True)
            return False

    def send_stop_setpoint(self) -> bool:
        """
        Send a stop setpoint to the Crazyflie.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected or not self.scf:
            logger.error("Cannot send stop setpoint: Not connected.")
            return False
            
        try:
            self.scf.cf.commander.send_stop_setpoint()
            return True
        except Exception as e:
            logger.error(f"Failed to send stop setpoint: {e}", exc_info=True)
            return False


class SimulatorConnection(DroneConnectionBase):
    """
    Connection to a simulator implementing a REST API interface.
    
    Uses REST API to communicate with the simulator. The simulator should:
    1. Accept normalized attitude commands in the range [-1, 1] for attitude control
    2. Return all angular values in degrees (not radians) to match real Crazyflie conventions
    3. Implement appropriate endpoints for control and state retrieval
    """
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        """
        Initialize the simulator connection.
        
        Args:
            host: Server hostname or IP address
            port: Server port
        """
        super().__init__()
        
        if not REQUESTS_AVAILABLE:
            raise ImportError("Requests library is not available. Install it with 'pip install requests'")
        
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        
        # Current state and command tracking
        self.state = {}
        self.attitude_cmd = {
            "roll": 0.0,
            "pitch": 0.0,
            "yaw_rate": 0.0,
            "thrust": 0.5  # Default to mid-thrust
        }
    
    def connect(self) -> bool:
        """
        Connect to the simulator.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Test connection by getting state
            response = requests.get(f"{self.base_url}/state", timeout=1.0)
            response.raise_for_status()
            
            self.is_connected = True
            if self.on_connected:
                self.on_connected(self.base_url)
            
            logger.info(f"Successfully connected to simulator at {self.base_url}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to simulator at {self.base_url}: {e}")
            self.is_connected = False
            
            if self.on_connection_failed:
                self.on_connection_failed(self.base_url, str(e))
                
            return False
    
    def disconnect(self) -> None:
        """
        Disconnect from the simulator.
        
        Note: HTTP connections are stateless, so this just resets the connection flag.
        """
        self.is_connected = False
        
        if self.on_disconnected:
            self.on_disconnected(self.base_url)
        
        logger.info(f"Disconnected from simulator at {self.base_url}")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current drone state from the simulator.
        
        The simulator is expected to return all angular values (roll, pitch, yaw, 
        angular velocities) already in degrees to match real Crazyflie conventions.
        No unit conversion is performed on the received values.
        
        Returns:
            Dictionary containing drone state with all angular values in degrees
        """
        if not self.is_connected:
            logger.warning("Cannot get state: Not connected to simulator.")
            return {}
            
        try:
            response = requests.get(f"{self.base_url}/state", timeout=1.0)
            response.raise_for_status()
            self.state = response.json()
            
            # Convert state to match expected format for logging_manager.py
            # and standardize with CFLibConnection's format
            if self.state and "orientation" in self.state:
                orientation = self.state["orientation"]
                
                # Map to stabilizer values to match real Crazyflie's log naming
                if all(key in orientation for key in ["roll", "pitch", "yaw"]):
                    # Assume all values from simulator are already in degrees
                    roll_deg = orientation["roll"]
                    pitch_deg = orientation["pitch"]
                    yaw_deg = orientation["yaw"]
                    
                    # Store in attitude format
                    self.state["attitude"] = {
                        "roll": roll_deg,
                        "pitch": pitch_deg,
                        "yaw": yaw_deg
                    }
                    
                    # Add stabilizer variables for compatibility with LoggingManager
                    if "stabilizer" not in self.state:
                        self.state["stabilizer"] = {}
                    self.state["stabilizer"]["roll"] = roll_deg
                    self.state["stabilizer"]["pitch"] = pitch_deg
                    self.state["stabilizer"]["yaw"] = yaw_deg
            
            # Add gyroscope data from angular velocity
            if self.state and "angular_velocity" in self.state:
                angular_velocity = self.state["angular_velocity"]
                
                # Assume angular velocity is already in degrees/sec
                if "gyro" not in self.state:
                    self.state["gyro"] = {}
                
                if "x" in angular_velocity:
                    self.state["gyro"]["x"] = angular_velocity["x"]
                if "y" in angular_velocity:
                    self.state["gyro"]["y"] = angular_velocity["y"]
                if "z" in angular_velocity:
                    self.state["gyro"]["z"] = angular_velocity["z"]
            
            return self.state
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting state from simulator: {e}")
            
            # Handle disconnection
            if not isinstance(e, requests.exceptions.Timeout):
                self.is_connected = False
                if self.on_connection_lost:
                    self.on_connection_lost(self.base_url, str(e))
            
            return {}
    
    def reset_estimator(self) -> bool:
        """
        Reset the simulator (equivalent to estimator reset for real drones).
        
        Returns:
            True if the command was sent successfully, False otherwise.
        """
        if not self.is_connected:
            logger.error("Cannot reset simulator: Not connected.")
            return False
            
        try:
            # Use the reset endpoint of the simulator
            response = requests.post(f"{self.base_url}/reset", timeout=1.0)
            response.raise_for_status()
            logger.info("Simulator reset command sent.")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send simulator reset command: {e}")
            return False
    
    def send_setpoint(self, roll: float, pitch: float, yaw_rate: float, thrust: float) -> bool:
        """
        Send an attitude command to the simulator.
        
        This method performs normalization of the input values:
        - Roll and pitch angles (in degrees) are converted to normalized [-1, 1] range
          using a +/- 30 degrees = +/- 1.0 scaling (standard for Crazyflie firmware)
        - Yaw rate (in degrees/s) is converted to normalized [-1, 1] range
          using a +/- 200 degrees/s = +/- 1.0 scaling
        - Thrust is normalized to [0, 1] range
        
        Args:
            roll: Roll angle in degrees (will be converted to [-1, 1] for simulator)
            pitch: Pitch angle in degrees (will be converted to [-1, 1] for simulator)
            yaw_rate: Yaw rate in degrees/s (will be converted to [-1, 1] for simulator)
            thrust: Thrust normalized [0, 1]
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected:
            logger.warning("Cannot send setpoint: Not connected to simulator.")
            return False
        
        try:
            # Convert from degrees to normalized values expected by simulator
            # This scaling matches the Crazyflie firmware (±30 degrees maps to ±1.0)
            # See https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/sensor-to-control/controllers/
            MAX_ROLL_PITCH_DEG = 30.0  # ±30 degrees maps to ±1.0
            MAX_YAW_RATE_DEG = 200.0   # ±200 deg/s maps to ±1.0
            
            # Input angles are in degrees, convert to normalized [-1, 1] range
            roll_norm = roll / MAX_ROLL_PITCH_DEG
            pitch_norm = pitch / MAX_ROLL_PITCH_DEG
            yaw_rate_norm = yaw_rate / MAX_YAW_RATE_DEG
            
            # Clamp values to ensure they're within the expected range
            roll_norm = max(-1.0, min(1.0, roll_norm))
            pitch_norm = max(-1.0, min(1.0, pitch_norm))
            yaw_rate_norm = max(-1.0, min(1.0, yaw_rate_norm))
            thrust_norm = max(0.0, min(1.0, thrust))
            
            # Update command values
            self.attitude_cmd = {
                "roll": roll_norm,
                "pitch": pitch_norm,
                "yaw_rate": yaw_rate_norm,
                "thrust": thrust_norm
            }
            
            # Send command to simulator
            response = requests.post(
                f"{self.base_url}/control/attitude",
                json=self.attitude_cmd,
                timeout=1.0
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending attitude command to simulator: {e}")
            return False

    def send_position_setpoint(self, x: float, y: float, z: float, yaw: float = 0.0) -> bool:
        """
        Send a position setpoint to the simulator.
        
        Args:
            x: X position in meters
            y: Y position in meters
            z: Z position in meters
            yaw: Yaw angle in degrees
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected:
            logger.warning("Cannot send position setpoint: Not connected to simulator.")
            return False
        
        try:
            # API expects yaw in degrees, which is then converted to radians internally if needed
            # This matches the behavior of real Crazyflie where yaw is in degrees
            cmd = {"x": x, "y": y, "z": z, "yaw": yaw}
            response = requests.post(
                f"{self.base_url}/control/position",
                json=cmd,
                timeout=1.0
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending position command to simulator: {e}")
            return False

    def send_hover_setpoint(self, vx: float, vy: float, yaw_rate: float, z: float) -> bool:
        """
        Send a hover setpoint to the simulator.
        
        In the simulator, this is implemented by calculating a new position based
        on current position and desired velocity.
        
        Args:
            vx: X velocity in m/s
            vy: Y velocity in m/s
            yaw_rate: Yaw rate in degrees/s
            z: Z position in meters
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected:
            logger.warning("Cannot send hover setpoint: Not connected to simulator.")
            return False
        
        try:
            # First get current state to determine position
            current_state = self.get_state()
            if not current_state:
                logger.error("Cannot send hover setpoint: Failed to get current state.")
                return False
            
            # Extract current position
            position = current_state.get("position", {"x": 0.0, "y": 0.0, "z": 0.0})
            
            # Get current yaw from state (already in degrees after get_state processing)
            current_yaw = 0.0
            if "stabilizer" in current_state and "yaw" in current_state["stabilizer"]:
                current_yaw = current_state["stabilizer"]["yaw"]
            
            # Calculate new position based on velocity (simple Euler integration)
            # In a real system, the drone's controller would handle this internally
            dt = 0.1  # Assume a short time step for integration
            new_x = position["x"] + vx * dt
            new_y = position["y"] + vy * dt
            
            # Calculate new yaw based on yaw rate
            new_yaw = current_yaw + yaw_rate * dt
            
            # Send position command with the calculated position and desired height
            return self.send_position_setpoint(new_x, new_y, z, new_yaw)
        except Exception as e:
            logger.error(f"Error sending hover setpoint to simulator: {e}")
            return False

    def send_stop_setpoint(self) -> bool:
        """
        Send a stop setpoint to the simulator.
        
        For the simulator, this is implemented by sending a zero attitude command.
        
        Returns:
            True if successful, False otherwise
        """
        return self.send_setpoint(0.0, 0.0, 0.0, 0.0)


class ConnectionManager:
    """
    Factory class for creating drone connections.
    
    Provides a unified interface for creating connections to different types of drones.
    """
    
    @staticmethod
    def create_connection(connection_type: str, **kwargs) -> DroneConnectionBase:
        """
        Create a connection of the specified type.
        
        Args:
            connection_type: Type of connection ('cflib' or 'simulator')
            **kwargs: Additional arguments to pass to the connection constructor
            
        Returns:
            A connection instance of the specified type
            
        Raises:
            ValueError: If the connection type is not supported
            ImportError: If required dependencies are missing
        """
        if connection_type.lower() == 'cflib':
            if not CFLIB_AVAILABLE:
                raise ImportError("CFLib is not available. Install it with 'pip install cflib'")
            
            uri = kwargs.get('uri', uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E6') if 'uri_helper' in globals() else None)
            if uri is None:
                raise ValueError("URI must be provided for CFLib connection")
            
            return CFLibConnection(uri)
        
        elif connection_type.lower() == 'simulator':
            if not REQUESTS_AVAILABLE:
                raise ImportError("Requests library is not available. Install it with 'pip install requests'")
            
            host = kwargs.get('host', 'localhost')
            port = kwargs.get('port', 8000)
            
            return SimulatorConnection(host, port)
        
        else:
            raise ValueError(f"Unsupported connection type: {connection_type}")
    
    @staticmethod
    def create_connection_from_config(config) -> DroneConnectionBase:
        """
        Create a connection based on configuration.
        
        Args:
            config: Configuration object with connection parameters
            
        Returns:
            A connection instance based on the configuration
            
        Raises:
            ValueError: If the configuration is invalid
        """
        # Get connection type from config
        connection_type = getattr(config, 'connection_type', None)
        
        if connection_type is None:
            # Try to infer connection type from URI format
            uri = getattr(config, 'uri', None)
            if uri and isinstance(uri, str):
                if uri.startswith(('radio://', 'usb://')):
                    connection_type = 'cflib'
                elif uri.startswith(('http://', 'https://')):
                    connection_type = 'simulator'
        
        if connection_type is None:
            # Default to cflib if available, otherwise simulator
            connection_type = 'cflib' if CFLIB_AVAILABLE else 'simulator'
            logger.info(f"No connection type specified, defaulting to {connection_type}")
        
        # Create connection based on type
        if connection_type.lower() == 'cflib':
            uri = getattr(config, 'uri', None)
            return ConnectionManager.create_connection('cflib', uri=uri)
        
        elif connection_type.lower() == 'simulator':
            host = getattr(config, 'sim_host', 'localhost')
            port = getattr(config, 'sim_port', 8000)
            return ConnectionManager.create_connection('simulator', host=host, port=port)
        
        else:
            raise ValueError(f"Unknown connection type in config: {connection_type}") 