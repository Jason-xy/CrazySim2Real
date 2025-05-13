"""
Main entry point for the Crazyflie Sweeper package.

Provides a CLI interface for running tests and wires together the various components.
"""
import argparse
import logging
import os
import sys
import time
import threading
import signal
import atexit
import yaml
import glob
import json
from typing import Dict, List, Optional, Tuple, Callable

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now use absolute imports from the project's perspective
from crazyflie_benchmark.config import FlightConfig
from crazyflie_benchmark.connection import ConnectionManager
from crazyflie_benchmark.controller import FlightController
from crazyflie_benchmark.safety import SafetyMonitor
from crazyflie_benchmark.logging_manager import LoggingManager
from crazyflie_benchmark.data_processor import DataProcessor
from crazyflie_benchmark.plotter import Plotter
from crazyflie_benchmark.tests import StepTest, ImpulseTest, SineSweepTest
from crazyflie_benchmark.tests.base import TestStrategy

# For keyboard monitoring
try:
    import msvcrt  # Windows
    _windows_keyboard = True
except ImportError:
    try:
        import termios  # Linux/Mac
        import tty
        import select
        _windows_keyboard = False
    except ImportError:
        # No keyboard monitoring available
        _windows_keyboard = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('CrazySim2Real.log')
    ]
)

logger = logging.getLogger(__name__)

# Global state
_emergency_stop_requested = False
_emergency_stop_callback = None
_keyboard_thread = None
_original_terminal_settings = None

# Register cleanup handler to restore terminal settings on exit
def _restore_terminal_settings():
    """Restore terminal settings when program exits."""
    global _original_terminal_settings
    if not _windows_keyboard and _original_terminal_settings:
        try:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSAFLUSH, _original_terminal_settings)
            logger.debug("Terminal settings restored by atexit handler")
        except Exception as e:
            # Don't log here as logging system might be shut down
            pass

# Register the cleanup handler
atexit.register(_restore_terminal_settings)

def set_emergency_callback(callback: Callable[[], None]):
    """
    Set the callback function to be called when an emergency stop is triggered.
    
    Args:
        callback: Function to call on emergency stop
    """
    global _emergency_stop_callback
    _emergency_stop_callback = callback

def trigger_emergency_stop():
    """
    Trigger the emergency stop sequence.
    
    This function is called directly when a spacebar is detected,
    or can be called from signal handlers or other code paths.
    """
    global _emergency_stop_requested
    
    if _emergency_stop_requested:
        return  # Prevent multiple triggers
        
    _emergency_stop_requested = True
    logger.warning("EMERGENCY STOP TRIGGERED")
    
    # Call the callback if registered
    if _emergency_stop_callback:
        try:
            _emergency_stop_callback()
        except Exception as e:
            logger.error(f"Error in emergency stop callback: {e}")

def keyboard_listener():
    """
    Listen for keyboard input in a separate thread.
    When spacebar is detected, immediately triggers the emergency stop.
    """
    global _emergency_stop_requested, _windows_keyboard, _original_terminal_settings
    
    logger.info("Keyboard listener started - Press SPACEBAR for emergency stop")
    
    try:
        if _windows_keyboard:
            # Windows version
            while not _emergency_stop_requested:
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8')
                    if key == ' ':  # spacebar
                        trigger_emergency_stop()
                time.sleep(0.05)  # Reduced sleep time for faster response
        else:
            # Linux/Mac version using non-blocking input
            # We'll use a separate file descriptor to avoid interfering with stdin
            # This preserves normal terminal behavior for other inputs
            fd = sys.stdin.fileno()
            _original_terminal_settings = termios.tcgetattr(fd)
            
            try:
                # Set non-canonical mode but don't disable echoing to keep input visible
                new_settings = termios.tcgetattr(fd)
                new_settings[3] = new_settings[3] & ~termios.ICANON  # Turn off canonical mode
                new_settings[3] = new_settings[3] & ~termios.ECHO  # Disable echo for emergency key only
                termios.tcsetattr(fd, termios.TCSANOW, new_settings)
                
                # Use select to check for input without blocking
                while not _emergency_stop_requested:
                    # Very short timeout to remain responsive but not hog CPU
                    ready, _, _ = select.select([fd], [], [], 0.05)
                    if ready:
                        # Read a single character
                        key = os.read(fd, 1).decode('utf-8', errors='ignore')
                        if key == ' ':  # spacebar
                            trigger_emergency_stop()
                            # Write the character back to stdout
                            sys.stdout.write(key)
                            sys.stdout.flush()
            finally:
                # Always restore terminal settings before exiting the thread
                if _original_terminal_settings:
                    termios.tcsetattr(fd, termios.TCSAFLUSH, _original_terminal_settings)
                    logger.debug("Terminal settings restored in keyboard listener thread")
    except Exception as e:
        logger.error(f"Error in keyboard listener: {e}")
        # Try to restore terminal settings if an exception occurs
        if not _windows_keyboard and '_original_terminal_settings' in globals() and _original_terminal_settings:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSAFLUSH, _original_terminal_settings)
                logger.debug("Terminal settings restored after error")
            except Exception as e2:
                logger.error(f"Error restoring terminal settings: {e2}")
    
    logger.info("Keyboard listener stopped")

def signal_handler(sig, frame):
    """Handle system signals like SIGINT (Ctrl+C) and SIGTERM."""
    logger.warning(f"Received signal {sig}, triggering emergency stop")
    trigger_emergency_stop()
    
    # Restore terminal settings immediately for better user experience on signals
    if not _windows_keyboard and '_original_terminal_settings' in globals() and _original_terminal_settings:
        try:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSAFLUSH, _original_terminal_settings)
            logger.debug("Terminal settings restored in signal handler")
        except Exception as e:
            logger.error(f"Error restoring terminal settings in signal handler: {e}")

def start_keyboard_listener():
    """Start the keyboard listener thread and set up signal handlers."""
    global _keyboard_thread, _emergency_stop_requested, _original_terminal_settings
    
    # Reset emergency stop flag
    _emergency_stop_requested = False
    
    # Initialize terminal settings reference
    if not _windows_keyboard and _windows_keyboard is not None:
        _original_terminal_settings = None
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    # Create and start thread if keyboard support is available
    if _windows_keyboard is not None:  # None means no keyboard support
        _keyboard_thread = threading.Thread(target=keyboard_listener)
        _keyboard_thread.daemon = True
        _keyboard_thread.start()
        logger.info("Emergency stop: Press SPACEBAR at any time to trigger emergency stop")
    else:
        logger.warning("Keyboard monitoring not available - emergency stop via spacebar disabled")

def stop_keyboard_listener():
    """Stop the keyboard listener thread."""
    global _keyboard_thread, _emergency_stop_requested, _original_terminal_settings
    
    _emergency_stop_requested = True
    
    # Ensure terminal settings are restored first
    if not _windows_keyboard and '_original_terminal_settings' in globals() and _original_terminal_settings:
        try:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSAFLUSH, _original_terminal_settings)
            logger.debug("Terminal settings restored in stop_keyboard_listener")
        except Exception as e:
            logger.error(f"Error restoring terminal settings: {e}")
    
    if _keyboard_thread and _keyboard_thread.is_alive():
        # Wait for thread to stop
        _keyboard_thread.join(timeout=1.0)
        if _keyboard_thread.is_alive():
            logger.warning("Keyboard listener thread did not terminate cleanly")
        _keyboard_thread = None

def is_emergency_stop_requested():
    """
    Check if emergency stop has been requested.
    
    Returns:
        True if emergency stop requested, False otherwise
    """
    return _emergency_stop_requested

def emergency_stop(controller: FlightController) -> None:
    """
    Execute emergency stop procedure.
    
    Args:
        controller: Flight controller to send stop command
    """
    logger.warning("EXECUTING EMERGENCY STOP")
    
    try:
        if hasattr(controller, 'disarm') and callable(controller.disarm):
            # Try to disarm immediately for fastest motor stop
            if not controller.disarm():
                # If disarm fails, try the full stop sequence
                controller.stop()
        else:
            # Fallback to stop method
            controller.stop()
    except Exception as e:
        logger.error(f"Error during emergency stop: {e}")

def patch_test_strategy():
    """
    Patch the TestStrategy class to check for emergency stop.
    """
    original_check_safety = TestStrategy.check_safety
    
    def check_safety_with_emergency_stop(self):
        """Check safety and emergency stop."""
        # First check for emergency stop
        if is_emergency_stop_requested():
            logger.warning(f"Emergency stop requested during {self.test_name}")
            self.aborted = True
            return False
            
        # Then run the original safety check
        return original_check_safety(self)
    
    # Replace the check_safety method to include emergency stop check
    TestStrategy.check_safety = check_safety_with_emergency_stop

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Crazyflie Sweeper - A comprehensive testing tool for the Crazyflie drone")
    parser.add_argument('--config', '-c', type=str, default='hardware_config.yaml', 
                        help="Path to configuration file (default: hardware_config.yaml or simulator_config.yaml)")
    parser.add_argument('--list-tests', '-l', action='store_true',
                        help="List available test plans and exit")
    parser.add_argument('--test-plan', '-t', type=str, default='',
                        help="Path to test plan file (relative to test_plans/ directory or absolute path)")
    parser.add_argument('--custom', action='store_true',
                        help="Create a custom test plan interactively")
    parser.add_argument('--output-dir', '-o', type=str, default='logs',
                        help='Directory for saving results (default: logs)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="Enable verbose logging")
    parser.add_argument('--analyze', '-a', type=str, required=False,
                        help="Run data analysis on specified log directory (required, e.g. logs/20230401_120000)")
    
    return parser.parse_args()

def get_test_plans_directory():
    """
    Get the path to the test plans directory.
    
    Returns:
        Path to the test plans directory
    """
    # Test plans directory is in the same directory as this script
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_plans")

def list_available_test_plans():
    """
    List all available test plans in the test_plans directory.
    
    Returns:
        List of (file_path, name, description) tuples for each test plan
    """
    test_plans_dir = get_test_plans_directory()
    test_plans = []
    
    # Find all YAML and JSON files in the test plans directory
    plan_files = glob.glob(os.path.join(test_plans_dir, "*.yaml"))
    plan_files.extend(glob.glob(os.path.join(test_plans_dir, "*.yml")))
    plan_files.extend(glob.glob(os.path.join(test_plans_dir, "*.json")))
    
    for file_path in sorted(plan_files):
        try:
            with open(file_path, 'r') as f:
                if file_path.endswith('.json'):
                    plan_data = json.load(f)
                else:  # YAML
                    plan_data = yaml.safe_load(f)
                
                name = plan_data.get('name', os.path.basename(file_path))
                description = plan_data.get('description', 'No description available')
                test_plans.append((file_path, name, description))
        except Exception as e:
            logger.error(f"Error loading test plan from {file_path}: {e}")
    
    return test_plans

def load_test_plan(file_path: str, config: FlightConfig) -> List[Tuple[str, str, float, float]]:
    """
    Load a test plan from a file.
    
    Args:
        file_path: Path to the test plan file
        config: Configuration parameters
        
    Returns:
        List of tuples (test_type, channel, amplitude, duration)
    """
    if not os.path.exists(file_path):
        logger.error(f"Test plan file not found: {file_path}")
        return []
    
    try:
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                plan_data = json.load(f)
            else:  # YAML
                plan_data = yaml.safe_load(f)
        
        # Extract tests from the plan
        tests = plan_data.get('tests', [])
        test_sequence = []
        
        for test in tests:
            test_type = test.get('type')
            channel = test.get('channel')
            amplitude = test.get('amplitude')
            duration = test.get('duration')
            
            # Validate test parameters
            if not all([test_type, channel, amplitude is not None, duration is not None]):
                logger.warning(f"Invalid test in plan: {test}")
                continue
            
            # Use the amplitude directly from the test plan
            # Safety checks will be applied at execution time
            test_sequence.append((test_type, channel, amplitude, duration))
        
        return test_sequence
    
    except Exception as e:
        logger.error(f"Error loading test plan from {file_path}: {e}")
        return []

def select_test_plan_and_create_sequence(config):
    """
    Select a test plan and create a test sequence.
    
    Args:
        config: Configuration parameters
        
    Returns:
        List of tests to execute
    """
    # Get available test plans
    test_plans = list_available_test_plans()
    
    if not test_plans:
        logger.error("No test plans found. Please create test plans in the 'test_plans' directory.")
        return []
    
    print("\n" + "="*60)
    print("TEST PLAN SELECTION")
    print("="*60)
    
    # Display available test plans
    for i, (file_path, name, description) in enumerate(test_plans, 1):
        print(f"{i}. {name}")
        print(f"   {description}")
        print(f"   File: {os.path.basename(file_path)}")
        print()
    
    print(f"{len(test_plans) + 1}. Custom Test Sequence - Define your own tests")
    
    # Get user selection
    try:
        selection = int(safe_input("\nSelect test plan: "))
        if selection < 1 or selection > len(test_plans) + 1:
            print(f"Invalid selection. Using first test plan: {test_plans[0][1]}")
            selection = 1
    except ValueError:
        print(f"Invalid input. Using first test plan: {test_plans[0][1]}")
        selection = 1
    
    if selection == len(test_plans) + 1:
        # Custom test sequence
        return prompt_for_custom_tests(config)
    else:
        # Load selected test plan
        file_path, name, _ = test_plans[selection - 1]
        print(f"\nLoading test plan: {name}")
        return load_test_plan(file_path, config)

def prompt_for_custom_tests(config: FlightConfig) -> List[Tuple[str, str, float, float]]:
    """
    Prompt the user to create custom tests.
    
    Args:
        config: Configuration parameters
        
    Returns:
        List of tuples (test_type, channel, amplitude, duration)
    """
    print("\n" + "="*60)
    print("CUSTOM TEST CONFIGURATION")
    print("="*60)
    
    tests = []
    
    try:
        test_count = int(safe_input("\nHow many tests do you want to run? (1-9): "))
        if test_count < 1 or test_count > 9:
            print("Invalid input. Defaulting to 3 tests.")
            test_count = 3
    except ValueError:
        print("Invalid input. Defaulting to 3 tests.")
        test_count = 3
    
    for i in range(1, test_count + 1):
        print(f"\n--- Test {i} ---")
        
        # Get test type
        print("1. Step Test (constant input)")
        print("2. Impulse Test (short pulse)")
        print("3. Sine Sweep Test (frequency response)")
        try:
            test_type_num = int(safe_input("Test type (1=Step, 2=Impulse, 3=Sine Sweep): "))
            if test_type_num == 1:
                test_type = "step"
            elif test_type_num == 2:
                test_type = "impulse"
            elif test_type_num == 3:
                test_type = "sine_sweep"
            else:
                print("Invalid input. Defaulting to Step Test.")
                test_type = "step"
        except ValueError:
            print("Invalid input. Defaulting to Step Test.")
            test_type = "step"
        
        # Get channel
        print("\n1. Roll (x-axis rotation)")
        print("2. Pitch (y-axis rotation)")
        print("3. Thrust (z-axis force)")
        try:
            channel_num = int(safe_input("Channel (1=Roll, 2=Pitch, 3=Thrust): "))
            if channel_num == 1:
                channel = "roll"
            elif channel_num == 2:
                channel = "pitch"
            elif channel_num == 3:
                channel = "thrust"
            else:
                print("Invalid input. Defaulting to roll.")
                channel = "roll"
        except ValueError:
            print("Invalid input. Defaulting to roll.")
            channel = "roll"
        
        # Use standard default amplitude values based on test type and channel
        if channel == "roll":
            if test_type == "step":
                amplitude = 5.0  # Standard default for roll step (degrees)
            elif test_type == "impulse":
                amplitude = 6.0  # Standard default for roll impulse (degrees)
            else:  # sine_sweep
                amplitude = 3.0  # Standard default for roll sine sweep (degrees)
        elif channel == "pitch":
            if test_type == "step":
                amplitude = 5.0  # Standard default for pitch step (degrees)
            elif test_type == "impulse":
                amplitude = 6.0  # Standard default for pitch impulse (degrees)
            else:  # sine_sweep
                amplitude = 3.0  # Standard default for pitch sine sweep (degrees)
        else:  # thrust
            if test_type == "step":
                amplitude = 2000  # Standard default for thrust step
            elif test_type == "impulse":
                amplitude = 2500  # Standard default for thrust impulse
            else:  # sine_sweep
                amplitude = 1200  # Standard default for thrust sine sweep
                
        # Let user override the default amplitude
        try:
            amplitude_input = safe_input(f"Amplitude [{amplitude}]: ")
            if amplitude_input:
                amplitude = float(amplitude_input)
        except ValueError:
            print(f"Invalid input. Using default amplitude: {amplitude}")
        
        # Use standard default duration values based on test type
        if test_type == "step":
            duration = 1.5  # Standard default for step test duration (seconds)
        elif test_type == "impulse":
            duration = 0.2  # Standard default for impulse test duration (seconds)
        else:  # sine_sweep
            duration = 8.0  # Standard default for sine sweep duration (seconds)
            
        # Let user override the default duration
        try:
            duration_input = safe_input(f"Duration [{duration}]: ")
            if duration_input:
                duration = float(duration_input)
        except ValueError:
            print(f"Invalid input. Using default duration: {duration}")
            
        # Add test to list
        tests.append((test_type, channel, amplitude, duration))
        
    return tests


def create_test_strategy(test_type: str, 
                      channel: str, 
                      amplitude: float, 
                      duration: float,
                      controller: FlightController,
                      config: FlightConfig,
                      safety_monitor: SafetyMonitor,
                      logging_manager: LoggingManager):
    """
    Create a test strategy based on the specified parameters.
    
    Args:
        test_type: Type of test ('step', 'impulse', or 'sine_sweep')
        channel: Channel to test ('roll', 'pitch', or 'thrust')
        amplitude: Test amplitude
        duration: Test duration
        controller: Flight controller
        config: Configuration parameters
        safety_monitor: Safety monitor
        logging_manager: Logging manager
        
    Returns:
        Test strategy instance
    """
    if test_type == "step":
        return StepTest(controller, config, safety_monitor, logging_manager, 
                      channel, amplitude, duration)
    elif test_type == "impulse":
        return ImpulseTest(controller, config, safety_monitor, logging_manager, 
                         channel, amplitude, duration)
    elif test_type == "sine_sweep":
        return SineSweepTest(controller, config, safety_monitor, logging_manager, 
                           channel, amplitude, duration)
    else:
        logger.error(f"Unknown test type: {test_type}")
        return None


def run_tests(test_sequence: List[Tuple[str, str, float, float]],
            controller: FlightController,
            config: FlightConfig,
            safety_monitor: SafetyMonitor,
            logging_manager: LoggingManager) -> bool:
    """
    Run a sequence of tests.
    
    Args:
        test_sequence: List of tuples (test_type, channel, amplitude, duration)
        controller: Flight controller
        config: Configuration parameters
        safety_monitor: Safety monitor
        logging_manager: Logging manager
        
    Returns:
        Whether all tests completed successfully
    """
    all_tests_successful = True
    
    for test_num, (test_type, channel, amplitude, duration) in enumerate(test_sequence, 1):
        # Apply safety limits to amplitude values
        safe_amplitude = amplitude
        if channel == "roll" or channel == "pitch":
            # Limit roll/pitch angles to safety maximum
            safe_amplitude = min(amplitude, config.safety_max_roll_pitch_deg)
        elif channel == "thrust":
            # Ensure thrust is within safe limits
            if test_type == "step":
                # For step tests, ensure the resulting thrust doesn't exceed safety limits
                max_safe_thrust_increment = config.safety_max_thrust - config.hover_thrust
                min_safe_thrust_increment = config.safety_min_thrust_flight - config.hover_thrust
                safe_amplitude = max(min(amplitude, max_safe_thrust_increment), min_safe_thrust_increment)
            else:
                # For other tests, just make sure the amplitude itself isn't too large
                max_safe_amplitude = (config.safety_max_thrust - config.safety_min_thrust_flight) / 2
                safe_amplitude = min(amplitude, max_safe_amplitude)
        
        # Log the original and safe amplitude values
        if safe_amplitude != amplitude:
            logger.warning(f"Amplitude for {channel} reduced from {amplitude} to {safe_amplitude} for safety")
        
        logger.info(f"Running test {test_num}/{len(test_sequence)}: {test_type} on {channel} (amplitude={safe_amplitude}, duration={duration}s)")
        logging_manager.set_current_maneuver(f"test_{test_num}")
        
        # Check for emergency stop request
        if is_emergency_stop_requested():
            logger.warning("Emergency stop requested, aborting test sequence")
            emergency_stop(controller)
            return False
        
        # Create the test strategy with the safe amplitude
        test_strategy = create_test_strategy(test_type, channel, safe_amplitude, duration, 
                                           controller, config, safety_monitor, logging_manager)
                                           
        if test_strategy:
            # Execute the test
            print(f"\n--- Running Test {test_num}/{len(test_sequence)}: {test_type.upper()} on {channel.upper()} ---")
            if safe_amplitude != amplitude:
                print(f"NOTE: Amplitude reduced from {amplitude} to {safe_amplitude} for safety")
            
            test_successful = test_strategy.execute()
            
            # Check for safety boundary violation
            if safety_monitor.is_boundary_exceeded():
                logger.warning("Safety boundary exceeded, aborting test sequence")
                return False
            
            # Return to origin and stabilize between tests
            if test_num < len(test_sequence):
                return_duration_s = 2.0 # Time allocated to reach the origin point
                hover_duration_s = max(0.1, config.inter_maneuver_delay_s - return_duration_s) # Time to hover at origin

                # 1. Go to origin hover point
                logging_manager.set_current_maneuver(f"return_to_origin_{test_num}")
                logger.info(f"Returning to origin (0, 0, 0.5) for {return_duration_s}s...")
                # Use the position_control method identified earlier
                controller.position_control(x=0.0, y=0.0, z=0.5, yaw=0.0, duration=return_duration_s)

                # Check for emergency stop after returning
                if is_emergency_stop_requested():
                    logger.warning("Emergency stop requested during return to origin, aborting test sequence")
                    return False # Abort if emergency stop was triggered

                # 2. Maintain hover at origin
                logging_manager.set_current_maneuver(f"inter_test_hover_{test_num}")
                logger.info(f"Stabilizing at origin hover for {hover_duration_s}s before next test...")
                controller.maintain_hover(hover_duration_s, target_height=0.5) # Ensure hover maintains the target height
            
            # Update overall success status
            all_tests_successful = all_tests_successful and test_successful
            
            if not test_successful:
                logger.warning("Test failed, continuing with next test...")
        else:
            logger.error(f"Failed to create test: {test_type} on {channel}")
            all_tests_successful = False
    
    return all_tests_successful


def display_sensor_preview(connection: ConnectionManager, logging_manager: LoggingManager) -> bool:
    """
    Display a real-time preview of sensor data and attitude estimation.
    
    Args:
        connection: Connection manager for the Crazyflie
        logging_manager: Logging manager with configured loggers
        
    Returns:
        True if user confirms data looks good, False otherwise
    """
    if not connection.is_connected:
        logger.error("Cannot display sensor preview: Not connected.")
        return False
    
    print("\n" + "="*60)
    print("== SENSOR AND ATTITUDE DATA PREVIEW ==")
    print("="*60)
    print("Collecting real-time data for a few seconds...")
    
    # Wait for data to stabilize
    time.sleep(2)  # Allow some time to gather data
    
    # Show multiple samples to verify data
    for _ in range(3):
        latest_data = logging_manager.get_latest_values()
        
        if not latest_data:
            print("\nNo sensor data received. Check connection and try again.")
            return False
        
        # Function to format values with handling for None
        def format_value(value, format_str, width=7):
            if value is None:
                return "N/A".center(width)
            try:
                return format_str.format(value)
            except (ValueError, TypeError):
                return str(value).center(width)
        
        # Clear screen for each update
        print("\n" + "="*68)
        
        # Attitude (stabilizer)
        print("│ ATTITUDE (degrees):                                                  │")
        roll_val = latest_data.get('stabilizer.roll')
        pitch_val = latest_data.get('stabilizer.pitch')
        yaw_val = latest_data.get('stabilizer.yaw')
        
        roll_str = format_value(roll_val, "{:>9.2f}", width=9)
        pitch_str = format_value(pitch_val, "{:>9.2f}", width=9)
        yaw_str = format_value(yaw_val, "{:>9.2f}", width=9)
        
        print(f"│   Roll: {roll_str}    Pitch: {pitch_str}    Yaw: {yaw_str}   │")
        
        # Gyroscope
        print("│ GYROSCOPE (deg/s):                                                    │")
        gyro_x_val = latest_data.get('gyro.x')
        gyro_y_val = latest_data.get('gyro.y')
        gyro_z_val = latest_data.get('gyro.z')
        
        gyro_x_str = format_value(gyro_x_val, "{:>9.2f}", width=9)
        gyro_y_str = format_value(gyro_y_val, "{:>9.2f}", width=9)
        gyro_z_str = format_value(gyro_z_val, "{:>9.2f}", width=9)
        
        print(f"│   X: {gyro_x_str}    Y: {gyro_y_str}    Z: {gyro_z_str}   │")
        
        # Accelerometer
        print("│ ACCELEROMETER (g):                                                    │")
        acc_x_val = latest_data.get('acc.x')
        acc_y_val = latest_data.get('acc.y')
        acc_z_val = latest_data.get('acc.z')
        
        acc_x_str = format_value(acc_x_val, "{:>9.2f}", width=9)
        acc_y_str = format_value(acc_y_val, "{:>9.2f}", width=9)
        acc_z_str = format_value(acc_z_val, "{:>9.2f}", width=9)
        
        print(f"│   X: {acc_x_str}    Y: {acc_y_str}    Z: {acc_z_str}   │")
        
        # Position (if available)
        position_vars = {}
        for axis in ['x', 'y', 'z']:
            # Try all possible variable names for this axis
            for var_name in logging_manager.config.position_var_mapping.get(axis, []):
                if var_name in latest_data:
                    position_vars[axis] = latest_data[var_name]
                    break
        
        if position_vars:
            print("│ POSITION (m):                                                        │")
            pos_x_val = position_vars.get('x')
            pos_y_val = position_vars.get('y')
            pos_z_val = position_vars.get('z')
            
            pos_x_str = format_value(pos_x_val, "{:>9.2f}", width=9)
            pos_y_str = format_value(pos_y_val, "{:>9.2f}", width=9)
            pos_z_str = format_value(pos_z_val, "{:>9.2f}", width=9)
            
            print(f"│   X: {pos_x_str}    Y: {pos_y_str}    Z: {pos_z_str}   │")
        
        print("="*68)
        
        time.sleep(1)  # Pause between updates
    
    # Check for missing sensor data
    print("\nSensor Data Status:")
    
    if not all(latest_data.get(f'stabilizer.{axis}') is not None for axis in ['roll', 'pitch', 'yaw']):
        print("\n⚠️  WARNING: Attitude data is missing.")
    
    if not all(latest_data.get(f'gyro.{axis}') is not None for axis in ['x', 'y', 'z']):
        print("\n⚠️  WARNING: Gyroscope data is missing.")
    
    if not all(latest_data.get(f'acc.{axis}') is not None for axis in ['x', 'y', 'z']):
        print("\n⚠️  WARNING: Accelerometer data is missing.")
    
    # Ask for confirmation
    response = safe_input("\nDoes the sensor data look correct and complete? (y/n): ")
    return response.lower() in ['y', 'yes']

def setup_environment(args):
    """
    Set up the environment based on command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configuration object
    """
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if args.config:
        config = FlightConfig.load_from_file(args.config)
    else:
        config = FlightConfig()
    
    return config

def display_welcome_banner(config):
    """Display welcome banner with configuration information."""
    print("\n" + "="*60)
    print("== CRAZYFLIE RESPONSE TESTING TOOL ==")
    print("="*60)
    print(f"Connection: {'Simulator' if config.connection_type == 'simulator' else 'Real Hardware'}")
    if config.connection_type == 'simulator':
        print(f"Simulator: {config.sim_host}:{config.sim_port}")
    else:
        print(f"Using URI: {config.uri}")
        print(f"CONFIGURED HOVER_THRUST: {config.hover_thrust}")
        print("PLEASE ENSURE THIS HOVER_THRUST IS CORRECT FOR YOUR DRONE!")
    print(f"SAFETY BOUNDS: Radius {config.safety_radius_m}m, Height {config.safety_max_height_m}m")
    
    print("\nTEST PLAN SYSTEM:")
    print("- Test plans are loaded from YAML/JSON files in the 'test_plans' directory")
    print("- You can select a test plan interactively or specify one with --test-plan")
    print("- Use --list-tests to see all available test plans")
    print("- Create custom test sequences with --custom")
    
    print("\nThe drone will take off and perform the tests defined in the selected plan.")
    print("BE READY TO PRESS SPACEBAR TO INITIATE EMERGENCY LANDING.")
    print("Ensure you have at least 2x2 meters of clear, flat space.")
    print("For a 5x5m area, stay vigilant for drift.")

def initialize_components(config, args):
    """
    Initialize all system components.
    
    Args:
        config: Configuration object
        args: Command line arguments
        
    Returns:
        Tuple of (connection, controller, safety_monitor, logging_manager, data_processor, plotter)
    """
    # Create the connection using the configuration
    connection = ConnectionManager.create_connection_from_config(config)
    
    # Connect to Crazyflie without using 'with' statement to keep connection open
    if not connection.connect():
        logger.error("Failed to connect to drone, aborting")
        return None
    
    # Initialize components
    controller = FlightController(connection, config)
    safety_monitor = SafetyMonitor(config)
    logging_manager = LoggingManager(connection, config)
    data_processor = DataProcessor(config)
    
    # Set up output directory
    plotter = Plotter(config)
    plotter.setup_output_directory(args.output_dir)
    
    if not logging_manager.setup_logging():
        logger.error("Failed to set up logging, aborting")
        connection.disconnect()
        return None
    
    # Register emergency stop callback
    set_emergency_callback(lambda: emergency_stop(controller))
    
    return connection, controller, safety_monitor, logging_manager, data_processor, plotter

def run_flight_sequence(controller, safety_monitor, logging_manager, config, test_sequence):
    """
    Run the flight sequence and tests.
    
    Args:
        controller: Flight controller
        safety_monitor: Safety monitor
        logging_manager: Logging manager
        config: Configuration parameters
        test_sequence: List of tests to execute
        
    Returns:
        True if all tests completed successfully, False otherwise
    """
    # Register emergency stop callback
    set_emergency_callback(lambda: emergency_stop(controller))
    
    # Show test plan
    print("\n" + "="*60)
    print("TEST PLAN SUMMARY:")
    print("="*60)
    for i, (test_type, channel, amplitude, duration) in enumerate(test_sequence, 1):
        if test_type == "step":
            test_desc = "Step Test"
        elif test_type == "impulse":
            test_desc = "Impulse Test"
        elif test_type == "sine_sweep":
            test_desc = "Sine Sweep Test"
        else:
            test_desc = f"Unknown test type: {test_type}"
            
        print(f"{i}. {test_desc} on {channel.title()}, "
              f"amplitude={amplitude}{' deg' if channel in ['roll', 'pitch'] else ''}, "
              f"duration={duration}s")
    
    # Show safety information
    print("\n" + "="*60)
    print("SAFETY INFORMATION:")
    print(f"- Safety boundary: {config.safety_radius_m}m radius, {config.safety_max_height_m}m height")
    print("- If safety boundary is exceeded: Test will abort and drone will hover in place")
    print("- EMERGENCY STOP: Press SPACEBAR at any time to stop motors immediately")
    print("="*60)
    
    # Confirm test parameters
    safe_input("\nPress Enter to start the sequence or Ctrl+C to abort...")
    
    # Check for emergency stop request
    if is_emergency_stop_requested():
        logger.warning("Emergency stop requested before takeoff")
        return False
    
    # Take off to 0.5m
    logging_manager.set_current_maneuver("takeoff")
    
    if not controller.take_off(target_height=0.5, logging_manager=logging_manager):
        logger.error("Takeoff failed, aborting")
        return False
    
    # Record the takeoff position for safety monitoring
    safety_monitor.record_takeoff_position(logging_manager.get_current_position())
    
    # Immediately start sending hover commands to maintain position
    logger.info("Ensuring hover after takeoff...")
    controller.send_hover_setpoint(0, 0, 0, 0.5)
    
    # Check for emergency stop request
    if is_emergency_stop_requested():
        logger.warning("Emergency stop requested after takeoff")
        return False
    
    # Wait for stabilization
    logging_manager.set_current_maneuver("hover_stabilization")
    logger.info("Stabilizing at hover before starting tests...")
    controller.maintain_hover(config.inter_maneuver_delay_s)
    
    # Monitor height during tests
    print("\n--- HEIGHT MONITORING DURING TESTS ---")
    print("Current height: {:.3f}m".format(logging_manager.get_current_position()['z']))
    print("Safety height limit: {:.1f}m".format(config.safety_max_height_m))
    print("---------------------------------------")
    
    # Run test sequence
    test_success = run_tests(test_sequence, controller, config, safety_monitor, logging_manager)
    
    # Check for emergency stop request
    if is_emergency_stop_requested():
        logger.warning("Emergency stop requested after tests")
        return False
    
    # Return to position [0,0,0.5]
    logging_manager.set_current_maneuver("return_to_origin")
    logger.info("Returning to origin position...")
    try:
        controller.position_control(0, 0, 0.5, 0, 3.0)
    except Exception as e:
        logger.warning(f"Return to origin position failed: {e}")
    
    # Land the drone
    logging_manager.set_current_maneuver("landing")
    logger.info("Landing drone before data processing...")
    landing_successful = False
    try:
        landing_successful = controller.land()
        if not landing_successful:
            logger.error("Controlled landing failed, attempting emergency stop")
            controller.stop()
    except Exception as e:
        logger.error(f"Landing procedure failed with error: {e}")
        logger.warning("Connection may have been lost during landing")
    
    return test_success

def process_and_save_data(logging_manager, data_processor, plotter):
    """
    Process and save the flight data.
    
    Args:
        logging_manager: Logging manager
        data_processor: Data processor
        plotter: Plotter
        
    Returns:
        True if processing successful, False otherwise
    """
    logging_manager.set_current_maneuver("data_processing")
    logger.info("Processing data and preparing results...")
    
    try:
        # Get raw data
        raw_data = logging_manager.get_log_data()
        
        # Save raw data for later analysis
        for maneuver, maneuver_data in raw_data.items():
            if not maneuver or maneuver == 'data_processing':
                continue
                
            # Save maneuver data as JSON
            json_path = os.path.join(plotter.output_dir, f"{maneuver}.json")
            with open(json_path, 'w') as f:
                json.dump(maneuver_data, f)
        
        # Save hover data if available
        if len(logging_manager.hover_data) > 10:
            hover_data_path = os.path.join(plotter.output_dir, "hover_data.json")
            with open(hover_data_path, 'w') as f:
                json.dump(logging_manager.hover_data, f)
        
        # Process data and generate results
        results = data_processor.process(raw_data)
        
        # Create analysis subdirectory
        analysis_dir = os.path.join(plotter.output_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Store original output directory and set to analysis directory
        original_output_dir = plotter.output_dir
        plotter.output_dir = analysis_dir
        
        # Generate plots and reports
        plotter.generate_summary(results)
        plotter.generate_plots(results)
        
        # Generate hover analysis plot if available
        if len(logging_manager.hover_data) > 10:
            plotter.generate_hover_analysis_plot(logging_manager.hover_data)
        
        # Export data for simulation comparison
        plotter.export_data_for_simulation(results)
        
        # Restore original output directory
        plotter.output_dir = original_output_dir
        
        logger.info(f"Raw data saved to: {plotter.output_dir}")
        logger.info(f"Analysis results saved to: {analysis_dir}")
        print(f"\nRaw data saved to: {plotter.output_dir}")
        print(f"Analysis results saved to: {analysis_dir}")
        return True
    except Exception as e:
        logger.error(f"Error during data processing: {e}", exc_info=True)
        print(f"\nERROR: Data processing failed: {e}")
        return False
    finally:
        # Stop logging even if connection is lost
        try:
            logging_manager.stop_logging()
        except Exception as e:
            logger.warning(f"Error stopping logging: {e}")

# Safe input function that temporarily restores terminal settings
def safe_input(prompt=""):
    """
    Get user input while ensuring terminal settings are properly handled.
    
    This function temporarily restores normal terminal settings while waiting
    for user input, then resumes the keyboard listener settings afterward.
    
    Args:
        prompt: Input prompt to display
        
    Returns:
        User input string
    """
    global _original_terminal_settings
    
    # If we're not in Linux/Mac or not using keyboard monitoring, just use regular input
    if _windows_keyboard or _windows_keyboard is None:
        return input(prompt)
    
    # Save current terminal settings if keyboard listener is active
    current_settings = None
    if _original_terminal_settings:
        try:
            fd = sys.stdin.fileno()
            current_settings = termios.tcgetattr(fd)
            # Restore original settings to get normal input behavior
            termios.tcsetattr(fd, termios.TCSAFLUSH, _original_terminal_settings)
        except Exception as e:
            logger.error(f"Error preparing terminal for input: {e}")
    
    try:
        # Get input with normal terminal settings
        result = input(prompt)
        return result
    finally:
        # Restore the previous terminal settings if needed
        if current_settings:
            try:
                fd = sys.stdin.fileno()
                termios.tcsetattr(fd, termios.TCSANOW, current_settings)
            except Exception as e:
                logger.error(f"Error restoring terminal settings after input: {e}")

def run_analysis_mode(args):
    """
    Run data analysis only mode.
    
    Args:
        args: Command-line arguments
        
    Returns:
        0 if successful, 1 otherwise
    """
    logger.info("Running in data analysis mode")
    
    # Setup base environment
    config = setup_environment(args)
    
    print("\n" + "="*60)
    print("DATA ANALYSIS MODE")
    print("="*60)
    
    # Get specified data directory
    data_dir = args.analyze
    
    if not data_dir:
        logger.error("Error: A log directory must be specified with --analyze")
        print("\nError: A log directory must be specified with --analyze")
        print("Example: python main.py --analyze logs/20230401_120000")
        print("\nAvailable log directories:")
        
        # Show available directories to help the user
        logs_base_dir = os.path.join(os.getcwd(), 'logs')
        if os.path.exists(logs_base_dir):
            log_dirs = [d for d in os.listdir(logs_base_dir) 
                       if os.path.isdir(os.path.join(logs_base_dir, d))]
            
            if log_dirs:
                # Sort by modification time (newest first)
                log_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(logs_base_dir, x)), reverse=True)
                for i, log_dir in enumerate(log_dirs, 1):
                    dir_path = os.path.join(logs_base_dir, log_dir)
                    mod_time = os.path.getmtime(dir_path)
                    mod_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))
                    print(f"{i}. {log_dir} (Modified: {mod_time_str})")
            else:
                print("No log directories found.")
        else:
            print("Logs directory not found.")
            
        return 1
    
    # If relative path, make absolute relative to logs directory
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(os.getcwd(), 'logs', data_dir)
    
    # Check if the directory exists
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        print(f"\nError: Data directory not found: {data_dir}")
        return 1
    
    # Check if this is a base directory with multiple test directories
    if os.path.isdir(data_dir):
        subdirs = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d)) and 
                  any(f.endswith('.json') for f in os.listdir(os.path.join(data_dir, d)))]
        
        if len(subdirs) > 1:
            # This appears to be a base logs directory
            logger.error(f"Specified directory contains multiple test directories: {data_dir}")
            print(f"\nError: The specified directory contains multiple test directories")
            print("Please specify a specific test directory, for example:")
            for subdir in subdirs[:5]:  # Show up to 5 examples
                print(f"  python main.py --analyze {os.path.join(os.path.relpath(data_dir), subdir)}")
            return 1
    
    return process_single_data_directory(data_dir, config)

def process_single_data_directory(data_dir, config):
    """
    Process a single data directory.
    
    Args:
        data_dir: Directory containing log data
        config: Configuration parameters
        
    Returns:
        0 if successful, 1 otherwise
    """
    logger.info(f"Analyzing data from: {data_dir}")
    print(f"\nAnalyzing data from: {data_dir}")
    
    # Set up data processor and plotter
    data_processor = DataProcessor(config)
    plotter = Plotter(config)
    
    # Create a directory for analysis results inside the data directory
    analysis_dir = os.path.join(data_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    plotter.output_dir = analysis_dir
    
    try:
        # Load raw data from the specified directory
        raw_data = load_raw_data_from_directory(data_dir)
        
        if not raw_data:
            logger.error("No valid data found in the specified directory")
            return 1
            
        # Process data
        results = data_processor.process(raw_data)
        
        # Generate plots and reports
        plotter.generate_summary(results)
        plotter.generate_plots(results)
        
        # Generate hover analysis plot if available
        hover_data_path = os.path.join(data_dir, 'hover_data.json')
        if os.path.exists(hover_data_path):
            try:
                with open(hover_data_path, 'r') as f:
                    hover_data = json.load(f)
                if hover_data and len(hover_data) > 10:
                    plotter.generate_hover_analysis_plot(hover_data)
            except Exception as e:
                logger.warning(f"Error loading hover data: {e}")
        
        # Export data for simulation comparison
        plotter.export_data_for_simulation(results)
        
        logger.info(f"Analysis results saved to: {analysis_dir}")
        print(f"Analysis results saved to: {analysis_dir}")
        return 0
    except Exception as e:
        logger.error(f"Error during data analysis: {e}", exc_info=True)
        print(f"ERROR: Data analysis failed: {e}")
        return 1

def load_raw_data_from_directory(data_dir):
    """
    Load raw log data from files in the specified directory.
    
    Args:
        data_dir: Directory containing log data
        
    Returns:
        Raw log data organized by maneuver/test
    """
    raw_data = {}
    
    # Look for JSON data files
    for filename in os.listdir(data_dir):
        if filename.endswith('.json') and filename != 'hover_data.json':
            try:
                with open(os.path.join(data_dir, filename), 'r') as f:
                    maneuver_data = json.load(f)
                    
                # Extract maneuver name from filename (e.g., step_roll_01.json -> step_roll_01)
                maneuver_name = os.path.splitext(filename)[0]
                raw_data[maneuver_name] = maneuver_data
                logger.info(f"Loaded data for maneuver: {maneuver_name}")
            except Exception as e:
                logger.warning(f"Error loading data from {filename}: {e}")
    
    return raw_data

def main():
    """Main entry point for the application."""
    # Parse command-line arguments
    args = parse_args()
    
    # Just list test plans if requested
    if args.list_tests:
        test_plans = list_available_test_plans()
        if test_plans:
            print("\nAvailable Test Plans:")
            for i, (file_path, name, description) in enumerate(test_plans, 1):
                print(f"{i}. {name}")
                print(f"   {description}")
                print(f"   File: {os.path.basename(file_path)}")
                print()
        else:
            print("\nNo test plans found in the 'test_plans' directory.")
        return 0
    
    # Check if analysis mode is requested
    if args.analyze is not None:
        return run_analysis_mode(args)
    
    # Setup environment
    config = setup_environment(args)
    
    # Display welcome message
    display_welcome_banner(config)
    
    # Initialize components
    components = initialize_components(config, args)
    if components is None:
        return 1
        
    connection, controller, safety_monitor, logging_manager, data_processor, plotter = components
    
    # Ensure emergency stop is registered
    set_emergency_callback(lambda: emergency_stop(controller))
    
    # Start keyboard monitoring for emergency stop
    start_keyboard_listener()
    
    try:
        # Determine which test plan to use
        if args.test_plan:
            # Use specified test plan
            test_plan_path = args.test_plan
            # If not an absolute path, look in the test_plans directory
            if not os.path.isabs(test_plan_path):
                test_plan_path = os.path.join(get_test_plans_directory(), test_plan_path)
            
            if not os.path.exists(test_plan_path):
                logger.error(f"Test plan not found: {test_plan_path}")
                return 1
            
            test_sequence = load_test_plan(test_plan_path, config)
            if not test_sequence:
                logger.error(f"Failed to load test plan from {test_plan_path}")
                return 1
                
            logger.info(f"Using test plan from {test_plan_path}")
        elif args.custom:
            # Create custom test plan
            test_sequence = prompt_for_custom_tests(config)
        else:
            # Select test plan interactively
            test_sequence = select_test_plan_and_create_sequence(config)
        
        if not test_sequence:
            logger.error("No tests selected or test plan is empty.")
            return 1
        
        # Run the flight sequence
        success = run_flight_sequence(controller, safety_monitor, logging_manager, config, test_sequence)
        
        # Only process data if no emergency stop was requested
        if success and not is_emergency_stop_requested():
            success = process_and_save_data(logging_manager, data_processor, plotter)
        
        return 0 if success else 1
    
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        trigger_emergency_stop()
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        trigger_emergency_stop()
        return 1
    finally:
        # Always stop keyboard listener and restore terminal
        stop_keyboard_listener()
        
        # Clean shutdown
        try:
            if connection:
                logger.info("Disconnecting from drone...")
                # Check if is_connected is a property or method
                if hasattr(connection, 'is_connected'):
                    if callable(connection.is_connected):
                        # If it's a method
                        if connection.is_connected():
                            connection.disconnect()
                    else:
                        # If it's a property
                        if connection.is_connected:
                            connection.disconnect()
                else:
                    # If the attribute doesn't exist, try to disconnect anyway
                    connection.disconnect()
        except Exception as e:
            logger.error(f"Error during disconnection: {e}")
        
        logger.info("Exiting Crazyflie Sweeper.")

if __name__ == "__main__":
    sys.exit(main()) 