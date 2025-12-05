import logging
import time
import threading
import pandas as pd
from typing import Dict, List, Any, Optional
from .config import FlightConfig
from ..core.connection import DroneConnectionBase, CFLibConnection, SimulatorConnection

# Try importing cflib
try:
    from cflib.crazyflie.log import LogConfig
    CFLIB_AVAILABLE = True
except ImportError:
    CFLIB_AVAILABLE = False

logger = logging.getLogger(__name__)

class FlightLogger:
    def __init__(self, connection: DroneConnectionBase, config: FlightConfig):
        self.connection = connection
        self.config = config
        self._data: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._state: str = "init"
        self._t0_cf: Optional[float] = None  # first controller/sim timestamp (s)
        self._latest_cf_time: Optional[float] = None
        self._latest_host_time: Optional[float] = None
        self._is_logging = False

        # For simulator polling
        self._polling_thread = None

        # For cflib
        self._log_configs = []

    def start(self):
        self._data = []
        self._is_logging = True
        self._t0_cf = None
        self._latest_cf_time = None
        self._latest_host_time = None

        if isinstance(self.connection, SimulatorConnection):
            self._start_sim_logging()
        elif isinstance(self.connection, CFLibConnection) and CFLIB_AVAILABLE:
            self._start_cflib_logging()
        else:
            logger.warning("Unknown connection type or cflib not available. Logging might not work.")

    def stop(self):
        self._is_logging = False
        if self._polling_thread:
            self._polling_thread.join(timeout=1.0)

        if self._log_configs:
            for conf in self._log_configs:
                conf.stop()
            self._log_configs = []

    def log_command(self, roll: float, pitch: float, yaw_rate: float, thrust: float):
        """Log control command."""
        timestamp = self._current_time()
        with self._state_lock:
            state = self._state
        with self._lock:
            self._data.append({
                'timestamp': timestamp,
                'cmd_roll': roll,
                'cmd_pitch': pitch,
                'cmd_yaw_rate': yaw_rate,
                'cmd_thrust': thrust,
                'type': 'command',
                'state': state
            })

    def get_time(self) -> float:
        return self._current_time()

    def get_dataframe(self) -> pd.DataFrame:
        with self._lock:
            df = pd.DataFrame(self._data)
        return df

    def _start_sim_logging(self):
        self._polling_thread = threading.Thread(target=self._sim_poll_loop, daemon=True)
        self._polling_thread.start()

    def _sim_poll_loop(self):
        while self._is_logging:
            try:
                state = self.connection.get_state()
                if state:
                    self._process_sim_state(state)
            except Exception as e:
                logger.error(f"Sim polling error: {e}")
            time.sleep(1.0 / self.config.log_rate_hz)

    def _process_sim_state(self, state: Dict[str, Any]):
        ts_cf = state.get("timestamp")
        if ts_cf is None:
            # Fallback: keep previous or 0.0 if simulator didn't provide time
            ts_cf = self._latest_cf_time if self._latest_cf_time is not None else 0.0

        if self._t0_cf is None:
            self._t0_cf = ts_cf
        self._latest_cf_time = ts_cf
        self._latest_host_time = time.time()
        timestamp = ts_cf - self._t0_cf
        with self._state_lock:
            cur_state = self._state
        entry = {'timestamp': timestamp, 'type': 'telemetry', 'state': cur_state}

        # Flatten state
        # Expected state structure from SimulatorConnection:
        # {'stabilizer': {'roll': ...}, 'position': {'x': ...}, ...}

        if 'stabilizer' in state:
            for k, v in state['stabilizer'].items():
                entry[f'stabilizer.{k}'] = v

        if 'position' in state:
            for k, v in state['position'].items():
                entry[f'position.{k}'] = v

        if 'velocity' in state:
            for k, v in state['velocity'].items():
                entry[f'velocity.{k}'] = v

        if 'gyro' in state:
            for k, v in state['gyro'].items():
                entry[f'gyro.{k}'] = v

        if 'acc' in state:
            for k, v in state['acc'].items():
                entry[f'acc.{k}'] = v

        with self._lock:
            self._data.append(entry)

    def _start_cflib_logging(self):
        # Setup cflib log configs
        # This mirrors the original LoggingManager logic but simplified

        # Attitude
        self._add_log_config("Attitude", {
            "stabilizer.roll": "float",
            "stabilizer.pitch": "float",
            "stabilizer.yaw": "float",
            "stabilizer.thrust": "uint16_t"
        })

        # Position (try to find valid vars from config)
        pos_vars = {}
        for axis, candidates in self.config.position_var_mapping.items():
            if candidates:
                pos_vars[candidates[0]] = "float" # Just pick the first one for now
        if pos_vars:
            self._add_log_config("Position", pos_vars)

        # Velocity
        vel_vars = {}
        for axis, candidates in self.config.velocity_var_mapping.items():
            if candidates:
                vel_vars[candidates[0]] = "float"
        if vel_vars:
            self._add_log_config("Velocity", vel_vars)

    def _add_log_config(self, name: str, vars: Dict[str, str]):
        if not self.connection.scf:
            return

        conf = LogConfig(name=name, period_in_ms=1000 // self.config.log_rate_hz)
        for v, t in vars.items():
            conf.add_variable(v, t)

        conf.data_received_cb.add_callback(self._cflib_data_callback)
        self.connection.scf.cf.log.add_config(conf)
        conf.start()
        self._log_configs.append(conf)

    def _cflib_data_callback(self, timestamp, data, logconf):
        # timestamp: Crazyflie firmware time in ms since boot
        ts_cf = timestamp / 1000.0
        if self._t0_cf is None:
            self._t0_cf = ts_cf
        self._latest_cf_time = ts_cf
        self._latest_host_time = time.time()
        ts = ts_cf - self._t0_cf
        with self._state_lock:
            cur_state = self._state
        entry = {'timestamp': ts, 'type': 'telemetry', 'state': cur_state}
        entry.update(data)

        with self._lock:
            self._data.append(entry)

    def get_latest_telemetry(self) -> Dict[str, Any]:
        with self._lock:
            if self._data:
                # Find last telemetry entry
                for i in range(len(self._data)-1, -1, -1):
                    if self._data[i].get('type') == 'telemetry':
                        return self._data[i].copy()
        return {}

    def get_current_position(self) -> Dict[str, float]:
        telem = self.get_latest_telemetry()
        pos = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        # Try to find position vars
        # Check config mapping first if possible, but here we just try common names
        for axis in ['x', 'y', 'z']:
            # Try standard names found in simulator or cflib
            candidates = [
                f'position.{axis}',
                f'kalman.state{axis.upper()}',
                f'stateEstimate.{axis}'
            ]
            for c in candidates:
                if c in telem:
                    try:
                        pos[axis] = float(telem[c])
                        break
                    except (ValueError, TypeError):
                        pass
        return pos

    def _current_time(self) -> float:
        """
        Current time on the vehicle/simulator clock relative to first sample.
        Falls back to 0.0 if no telemetry has arrived yet.
        """
        if self._t0_cf is None or self._latest_cf_time is None:
            return 0.0
        base = self._latest_cf_time - self._t0_cf
        if self._latest_host_time is None:
            return base
        elapsed_host = time.time() - self._latest_host_time
        return base + elapsed_host

    def set_state(self, state: str) -> None:
        """Update the current flight/test state for logging."""
        with self._state_lock:
            self._state = state
