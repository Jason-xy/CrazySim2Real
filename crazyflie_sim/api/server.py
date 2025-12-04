"""
HTTP API server for Crazyflie simulator.

All angles in responses are in degrees.
All angular velocities are in degrees/second.
"""
import logging
import json
import threading
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from http.server import HTTPServer
from typing import Dict, Any

logger = logging.getLogger(__name__)


class SimulatorAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for simulator API."""

    simulation_manager = None

    def _send_json(self, data: Dict[str, Any], status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _read_json(self) -> Dict[str, Any]:
        """Read JSON from request body."""
        length = int(self.headers["Content-Length"])
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def log_message(self, format, *args):
        logger.debug(f"{self.client_address[0]} - {format % args}")

    def do_OPTIONS(self):
        self._send_json({})

    def do_GET(self):
        try:
            if self.path == "/state":
                self._send_json(self.simulation_manager.get_state())
            elif self.path == "/controller/params":
                self._send_json(self.simulation_manager.get_controller_params())
            elif self.path == "/controller/debug":
                self._send_json(self.simulation_manager.get_controller_debug())
            else:
                self._send_json({"error": f"Not found: {self.path}"}, 404)
        except Exception as e:
            logger.error(f"GET error: {e}")
            self._send_json({"error": str(e)}, 500)

    def do_POST(self):
        try:
            if self.path == "/control/position":
                data = self._read_json()
                self.simulation_manager.enqueue_position_cmd(
                    data.get("x", 0.0),
                    data.get("y", 0.0),
                    data.get("z", 0.0),
                    data.get("yaw", 0.0),
                )
                self._send_json({"status": "ok"})

            elif self.path == "/control/velocity":
                data = self._read_json()
                self.simulation_manager.enqueue_velocity_cmd(
                    data.get("vx", 0.0),
                    data.get("vy", 0.0),
                    data.get("vz", 0.0),
                    data.get("yaw_rate", 0.0),
                )
                self._send_json({"status": "ok"})

            elif self.path == "/control/attitude":
                data = self._read_json()
                self.simulation_manager.enqueue_attitude_cmd(
                    data.get("roll", 0.0),
                    data.get("pitch", 0.0),
                    data.get("yaw_rate", 0.0),
                    data.get("thrust", 0.5),
                )
                self._send_json({"status": "ok"})

            elif self.path == "/reset":
                self.simulation_manager.reset()
                self._send_json({"status": "ok"})

            elif self.path == "/controller/params":
                data = self._read_json()
                self.simulation_manager.update_controller_params(data)
                self._send_json({"status": "ok"})

            else:
                self._send_json({"error": f"Not found: {self.path}"}, 404)

        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, 400)
        except Exception as e:
            logger.error(f"POST error: {e}")
            self._send_json({"error": str(e)}, 500)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server."""
    pass


class SimulatorAPIServer:
    """API server for simulator control."""

    def __init__(self, host: str, port: int, simulation_manager):
        self.host = host
        self.port = port
        self.server = None
        self.thread = None
        SimulatorAPIHandler.simulation_manager = simulation_manager
        logger.info(f"API server configured: {host}:{port}")

    def start(self):
        if self.server:
            return
        self.server = ThreadedHTTPServer((self.host, self.port), SimulatorAPIHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        logger.info(f"API server started: http://{self.host}:{self.port}")

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            logger.info("API server stopped")
