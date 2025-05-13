"""
HTTP API server for the Crazyflie simulator.

All angles in the API responses (orientation: roll, pitch, yaw) are in degrees (not radians).
All angular velocities (angular_velocity: x, y, z) are in degrees/second.
This ensures compatibility with the real Crazyflie's conventions.
"""
import logging
import json
import threading
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from http.server import HTTPServer
from typing import Dict, Any, Callable

logger = logging.getLogger(__name__)

class SimulatorAPIHandler(BaseHTTPRequestHandler):
    """
    HTTP request handler for the simulator API.
    
    All orientation angles (roll, pitch, yaw) in responses are in degrees.
    All angular velocities are in degrees/second.
    """
    
    # Class attribute to store a reference to the simulation manager
    simulation_manager = None
    
    def _set_response(self, status_code=200, content_type='application/json'):
        """Set the response headers."""
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')  # Allow CORS
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def _send_json(self, data: Dict[str, Any], status_code=200):
        """Send a JSON response."""
        self._set_response(status_code)
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def _read_json_body(self) -> Dict[str, Any]:
        """Read and parse JSON request body."""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        return json.loads(post_data)
    
    def log_message(self, format, *args):
        """Override to use our logger."""
        logger.debug(f"{self.client_address[0]} - {format%args}")
    
    def do_OPTIONS(self):
        """Handle OPTIONS request for CORS preflight."""
        self._set_response()
    
    def do_GET(self):
        """Handle GET requests."""
        try:
            if self.path == '/state':
                # Get drone state
                state = self.simulation_manager.get_state()
                self._send_json(state)
            
            elif self.path == '/controller/params':
                # Get controller parameters
                params = self.simulation_manager.get_controller_params()
                self._send_json(params)
            
            else:
                # Route not found
                self._send_json({"error": f"Route not found: {self.path}"}, 404)
        
        except Exception as e:
            logger.error(f"Error handling GET request: {e}")
            self._send_json({"error": str(e)}, 500)
    
    def do_POST(self):
        """Handle POST requests."""
        try:
            if self.path == '/control/position':
                # Position control command
                data = self._read_json_body()
                
                # Extract position setpoint
                x = data.get('x', 0.0)
                y = data.get('y', 0.0)
                z = data.get('z', 0.0)
                
                # Note: If yaw is provided, it's expected to be in degrees
                # It will be converted to radians internally in the simulator if needed
                yaw = data.get('yaw', None)
                
                # Send command to simulation manager
                if yaw is not None:
                    # Store original yaw in degrees to pass to simulation manager
                    self.simulation_manager.enqueue_position_cmd(x, y, z, yaw)
                else:
                    self.simulation_manager.enqueue_position_cmd(x, y, z)
                
                # Send success response
                self._send_json({"status": "success", "message": "Position command accepted"})
            
            elif self.path == '/control/attitude':
                # Attitude control command
                data = self._read_json_body()
                
                # Extract attitude setpoint
                # Note: All command input values are normalized [-1, 1]
                # The actual angles in degrees are calculated internally by the simulator
                roll = data.get('roll', 0.0)
                pitch = data.get('pitch', 0.0)
                yaw_rate = data.get('yaw_rate', 0.0)
                thrust = data.get('thrust', 0.5)
                
                # Send command to simulation manager
                self.simulation_manager.enqueue_attitude_cmd(roll, pitch, yaw_rate, thrust)
                
                # Send success response
                self._send_json({"status": "success", "message": "Attitude command accepted"})
            
            elif self.path == '/controller/params':
                # Update controller parameters
                data = self._read_json_body()
                
                # Extract PID gains
                kp = data.get('Kp')
                ki = data.get('Ki')
                kd = data.get('Kd')
                
                # Update controller parameters
                self.simulation_manager.update_pid_params(kp, ki, kd)
                
                # Send success response
                self._send_json({"status": "success", "message": "Controller parameters updated"})
            
            elif self.path == '/reset':
                # Reset simulation
                self.simulation_manager.reset()
                
                # Send success response
                self._send_json({"status": "success", "message": "Simulation reset"})
            
            else:
                # Route not found
                self._send_json({"error": f"Route not found: {self.path}"}, 404)
        
        except json.JSONDecodeError:
            logger.error("Invalid JSON received")
            self._send_json({"error": "Invalid JSON format"}, 400)
        
        except Exception as e:
            logger.error(f"Error handling POST request: {e}")
            self._send_json({"error": str(e)}, 500)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    pass


class SimulatorAPIServer:
    """
    HTTP server for the simulator API.
    """
    
    def __init__(self, host: str, port: int, simulation_manager):
        """
        Initialize the API server.
        
        Args:
            host: Server hostname or IP address
            port: Server port
            simulation_manager: Reference to the SimulationManager instance
        """
        self.host = host
        self.port = port
        self.server = None
        self.server_thread = None
        
        # Set the simulation_manager as a class attribute of the handler
        SimulatorAPIHandler.simulation_manager = simulation_manager
        
        logger.info(f"Initializing API server on {host}:{port}")
    
    def start(self):
        """Start the HTTP server in a daemon thread."""
        if self.server is not None:
            logger.warning("Server already running")
            return
        
        try:
            self.server = ThreadedHTTPServer((self.host, self.port), SimulatorAPIHandler)
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            logger.info(f"API server started on http://{self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            raise
    
    def stop(self):
        """Stop the HTTP server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            logger.info("API server stopped") 