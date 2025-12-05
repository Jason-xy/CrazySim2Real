import argparse
import logging
import time
import os
import sys
import signal
import threading
from typing import List, Tuple, Optional, Dict, Any

# Allow running as a script
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from crazyflie_benchmark.core.config import FlightConfig
from crazyflie_benchmark.core.connection import ConnectionManager
from crazyflie_benchmark.core.logger import FlightLogger
from crazyflie_benchmark.core.safety import SafetyMonitor
from crazyflie_benchmark.core.metrics import calculate_step_metrics
from crazyflie_benchmark.controllers.base import FlightController
from crazyflie_benchmark.controllers.sim import SimFlightController
from crazyflie_benchmark.controllers.real import RealFlightController
from crazyflie_benchmark.tests.step import StepTest
from crazyflie_benchmark.tests.impulse import ImpulseTest
from crazyflie_benchmark.tests.sine_sweep import SineSweepTest
from crazyflie_benchmark.tools.thrust_estimation import measure_hover_thrust

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BenchmarkRunner:
    def __init__(self, args):
        self.args = args
        self.config = FlightConfig.load_from_file(args.config)

        self.connection = ConnectionManager.create_connection(
            self.config.connection_type,
            uri=self.config.uri,
            host=self.config.sim_host,
            port=self.config.sim_port
        )

        if self.config.connection_type == 'simulator':
            self.controller = SimFlightController(self.connection, self.config)
        else:
            self.controller = RealFlightController(self.connection, self.config)

        self.logger = FlightLogger(self.connection, self.config)
        self.safety_monitor = SafetyMonitor(self.config)
        self.stop_event = threading.Event()

    def setup_signal_handlers(self):
        def signal_handler(sig, frame):
            logger.warning("Emergency stop signal received!")
            self.stop_event.set()
            self.controller.stop()

        signal.signal(signal.SIGINT, signal_handler)

    def load_test_sequence(self, plan_file: str) -> List[Dict[str, Any]]:
        import yaml
        plan_path = self._find_plan_file(plan_file)
        if not plan_path:
            logger.error(f"Test plan file {plan_file} not found (searched current dir and test_plans/)")
            return []

        with open(plan_path, 'r') as f:
            data = yaml.safe_load(f)

        sequence = []
        for item in data.get('tests', []):
            sequence.append(item)
        return sequence

    def _find_plan_file(self, plan_file: str) -> Optional[str]:
        """
        Locate a plan file by checking the given path first, then falling back to
        crazyflie_benchmark/test_plans/.
        """
        # Absolute or relative path provided and exists
        if os.path.exists(plan_file):
            return plan_file

        # Fall back to test_plans next to this file
        candidate = os.path.join(os.path.dirname(__file__), 'test_plans', plan_file)
        if os.path.exists(candidate):
            return candidate

        return None

    def run_tests(self, sequence: List[Dict[str, Any]]):
        results = {}

        for item in sequence:
            test_type = item.get('type')
            channel = item.get('channel')
            amp = item.get('amplitude')
            dur = item.get('duration')
            start_freq = item.get('start_freq')
            end_freq = item.get('end_freq')
            if self.stop_event.is_set():
                break

            self.logger.set_state("hover_measure")
            # Measure hover thrust fresh before each test
            measured = measure_hover_thrust(self.controller, self.logger)
            if measured is None:
                logger.error("Stopping: hover thrust measurement failed before test run.")
                break
            self.config.hover_thrust = measured

            state_name = f"test:{test_type}:{channel}"
            self.logger.set_state(state_name)
            logger.info(f"Running {test_type} test on {channel}")

            # Record start time for metrics
            start_time = self.logger.get_time()

            if test_type == 'step':
                test = StepTest(
                    self.controller,
                    self.config,
                    self.safety_monitor,
                    self.logger,
                    channel,
                    amp,
                    dur
                )

                if test.execute():
                    # Calculate metrics
                    time.sleep(0.1) # Wait for logs to flush
                    df = self.logger.get_dataframe()
                    metrics = calculate_step_metrics(df, start_time, dur, channel)
                    results[f"{test_type}_{channel}"] = metrics
                    logger.info(f"Metrics for {channel}: {metrics}")
                else:
                    logger.warning(f"Test {test.test_name} failed")

            elif test_type == 'impulse':
                test = ImpulseTest(
                    self.controller,
                    self.config,
                    self.safety_monitor,
                    self.logger,
                    channel,
                    amp,
                    dur
                )

                if test.execute():
                    results[f"{test_type}_{channel}"] = {"status": "ok"}
                else:
                    logger.warning(f"Test {test.test_name} failed")

            elif test_type == 'sine_sweep':
                test = SineSweepTest(
                    self.controller,
                    self.config,
                    self.safety_monitor,
                    self.logger,
                    channel,
                    amp,
                    dur,
                    start_freq,
                    end_freq
                )
                if test.execute():
                    results[f"{test_type}_{channel}"] = {"status": "ok"}
                else:
                    logger.warning(f"Test {test.test_name} failed")
            else:
                logger.error(f"Unsupported test type: {test_type}")

            # Inter-test delay
            time.sleep(1.0)

        return results

    def run(self):
        self.setup_signal_handlers()

        logger.info(f"Connecting to {self.config.connection_type}...")
        self.logger.set_state("connecting")
        if not self.connection.connect():
            logger.error("Failed to connect")
            return

        self.logger.start()

        # Reset simulator if applicable
        if self.config.connection_type == 'simulator':
            logger.info("Resetting simulator...")
            self.connection.reset_estimator()
            time.sleep(1.0)

        try:
            self.logger.set_state("arming")
            logger.info("Arming...")
            if not self.controller.arm():
                logger.error("Failed to arm")
                return

            self.logger.set_state("takeoff")
            logger.info("Taking off...")
            if not self.controller.take_off():
                logger.error("Takeoff failed")
                return

            # Stabilize
            time.sleep(2.0)

            # Record takeoff pos
            pos = self.logger.get_current_position()
            self.safety_monitor.record_takeoff_position(pos)

            # Load test sequence
            sequence = self.load_test_sequence(self.args.plan)

            # Run tests
            results = self.run_tests(sequence)

            # Print summary
            print("\n=== Test Results ===")
            for name, metrics in results.items():
                print(f"\n{name}:")
                for k, v in metrics.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")
            print("==================\n")

            logger.info("Landing...")
            self.controller.land()

        except Exception as e:
            logger.error(f"Error during execution: {e}", exc_info=True)
            self.controller.stop()
        finally:
            self.logger.stop()
            self.connection.disconnect()

            # Save logs
            log_dir = os.path.join(os.path.dirname(__file__), 'logs')
            os.makedirs(log_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(log_dir, f"flight_log_{timestamp}.csv")

            df = self.logger.get_dataframe()
            if not df.empty:
                df.to_csv(filename, index=False)
                logger.info(f"Logs saved to {filename}")
            else:
                logger.warning("No data logged")

def list_tests():
    print("Available tests:")
    print("  - step: Step response test (roll, pitch, yaw, thrust)")
    print("  - impulse: Impulse response test")
    print("  - sine_sweep: Frequency sweep test")

def main():
    parser = argparse.ArgumentParser(description='Crazyflie Benchmark Runner')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--sim', action='store_true', help='Use simulator')
    parser.add_argument('--list-tests', action='store_true', help='List available tests')
    parser.add_argument('--plan', type=str, help='Path to test plan YAML file (required)')

    # Analysis arguments
    parser.add_argument('--analyze', type=str, help='Analyze a flight log file')
    parser.add_argument('--compare', type=str, nargs='+', help='Compare two flight log files')
    parser.add_argument('--save', action='store_true', help='Save plots to file instead of showing them')
    parser.add_argument('--output', '-o', help='Output directory for plots (only used with --save)')
    parser.add_argument('--label1', default='Log 1', help='Label for first log (comparison)')
    parser.add_argument('--label2', default='Log 2', help='Label for second log (comparison)')

    args = parser.parse_args()

    if args.list_tests:
        list_tests()
        return

    # Handle Analysis Tools
    if args.analyze:
        from crazyflie_benchmark.tools.analyze_logs import analyze_log
        analyze_log(args.analyze, args.save, args.output)
        return

    if args.compare:
        if len(args.compare) != 2:
            print("Error: --compare requires exactly two log files")
            return
        from crazyflie_benchmark.tools.compare_logs import compare_logs
        compare_logs(args.compare[0], args.compare[1], args.label1, args.label2, args.save, args.output)
        return

    if args.config and args.plan:
        runner = BenchmarkRunner(args)
        runner.run()
        return

if __name__ == "__main__":
    main()
