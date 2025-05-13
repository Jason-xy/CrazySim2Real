#!/usr/bin/env python3
"""
Crazyflie Hover Thrust Estimator

This script performs autonomous thrust estimation for Crazyflie drones.
It takes off to a specified height, maintains hover using the built-in 
hover setpoint API, collects thrust data, and calculates the optimal
hover thrust value.

All control is done through the hover setpoint API while actual thrust 
data is collected from the stabilizer.thrust log variable.
"""
import time
import json
import logging
import argparse
import sys
from datetime import datetime
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crtp.crtpstack import CRTPPacket, CRTPPort

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('thrust_estimation.log')
    ]
)

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_URI = 'radio://0/80/2M/E7E7E7E7E6'
DEFAULT_HEIGHT = 0.5  # meters
DEFAULT_DURATION = 15  # seconds
DEFAULT_OUTPUT_FILE = 'hover_thrust_config.json'
CONTROL_RATE_HZ = 50
CONTROL_PERIOD = 1.0 / CONTROL_RATE_HZ

class ThrustEstimator:
    """
    Thrust estimator for Crazyflie drones.
    
    Uses the hover setpoint API for control while collecting
    thrust data from the stabilizer log.
    """
    
    def __init__(self, uri=DEFAULT_URI):
        """Initialize the thrust estimator."""
        # Connection parameters
        self.uri = uri
        self.cf = None
        self.scf = None
        self.is_connected = False
        self.is_armed = False
        self._setpoint_initialized = False
        
        # Data collection
        self.log_conf = None
        self.thrust_data = []
        self.current_height = 0.0
        
    #--------------------------------------------------------------------------
    # Connection Management
    #--------------------------------------------------------------------------
    
    def connect(self):
        """Connect to the Crazyflie."""
        logger.info(f"Connecting to Crazyflie at {self.uri}")
        
        try:
            # Initialize drivers
            cflib.crtp.init_drivers()
            
            # Create and connect to Crazyflie
            self.cf = Crazyflie(rw_cache='./cache')
            self.scf = SyncCrazyflie(self.uri, cf=self.cf)
            
            # Set up callback to track connection state
            self.cf.connected.add_callback(lambda uri: setattr(self, 'is_connected', True))
            self.cf.disconnected.add_callback(lambda uri: setattr(self, 'is_connected', False))
            
            # Open the link - blocks until connected
            self.scf.open_link()
            
            # Wait for connection to initialize fully
            time.sleep(1.0)
            
            # Verify connection is active
            if hasattr(self.scf, 'is_link_open') and self.scf.is_link_open():
                self.is_connected = True
                logger.info("Successfully connected to Crazyflie")
                return True
            else:
                logger.error("Failed to establish connection")
                return False
                
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from the Crazyflie."""
        if self.is_connected and self.scf:
            try:
                # Send zero thrust commands before disconnecting
                for _ in range(5):
                    self.send_setpoint(0, 0, 0, 0)
                    time.sleep(0.02)
                
                # Close the link
                self.scf.close_link()
                self.is_connected = False
                logger.info("Disconnected from Crazyflie")
            except Exception as e:
                logger.error(f"Error during disconnection: {str(e)}")
    
    #--------------------------------------------------------------------------
    # Motor Control
    #--------------------------------------------------------------------------
    
    def arm(self):
        """Arm the Crazyflie motors."""
        if not self.is_connected:
            logger.error("Cannot arm: Not connected")
            return False
            
        # Check if already armed
        if self.check_arm_state():
            logger.info("Motors already armed")
            return True
            
        logger.info("Arming motors...")
        
        try:
            # Send ARM command
            pk = CRTPPacket()
            pk.port = CRTPPort.PLATFORM
            pk.channel = 0
            pk.data = bytes([1, 1])  # ARM command
            self.cf.send_packet(pk)
            
            # Set armed state
            self.is_armed = True
            time.sleep(0.3)
            logger.info("Motors armed")
            return True
            
        except Exception as e:
            logger.error(f"Arming failed: {str(e)}")
            return False
    
    def disarm(self):
        """Disarm the Crazyflie motors."""
        if not self.is_connected:
            logger.error("Cannot disarm: Not connected")
            return False
            
        logger.info("Disarming motors...")
        
        try:
            # Send DISARM command
            pk = CRTPPacket()
            pk.port = CRTPPort.PLATFORM
            pk.channel = 0
            pk.data = bytes([1, 0])  # DISARM command
            self.cf.send_packet(pk)
            
            # Set armed state
            self.is_armed = False
            time.sleep(0.2)
            logger.info("Motors disarmed")
            return True
            
        except Exception as e:
            logger.error(f"Disarming failed: {str(e)}")
            return False
    
    def check_arm_state(self):
        """Check if the drone is already armed by reading parameters."""
        if not self.is_connected or not self.scf:
            return False
            
        try:
            # Try to read the motor interlock state parameter (if available)
            try:
                interlock_param = self.scf.cf.param.get_value('motorPowerSet.enable')
                if int(interlock_param) == 1:
                    logger.info("Motors already enabled via motorPowerSet.enable")
                    self.is_armed = True
                    return True
            except:
                # Parameter not available, continue with normal check
                pass
                
            # For firmware without specific arming parameters, 
            # we assume the drone is not armed on connection
            return False
            
        except Exception as e:
            logger.error(f"Error checking arm state: {str(e)}")
            return False
    
    #--------------------------------------------------------------------------
    # Command Interface
    #--------------------------------------------------------------------------
    
    def initialize_setpoint(self):
        """Initialize the setpoint interface."""
        if not self.is_connected:
            logger.error("Cannot initialize setpoint: Not connected")
            return False
            
        try:
            logger.info("Initializing setpoint interface")
            self.cf.commander.send_setpoint(0, 0, 0, 0)
            self._setpoint_initialized = True
            time.sleep(0.1)
            return True
        except Exception as e:
            logger.error(f"Setpoint initialization failed: {str(e)}")
            return False
    
    def send_setpoint(self, roll, pitch, yaw_rate, thrust):
        """Send a setpoint command to the Crazyflie."""
        if not self.is_connected:
            return False
            
        # Initialize setpoint interface if needed
        if not self._setpoint_initialized:
            if not self.initialize_setpoint():
                return False
        
        try:
            self.cf.commander.send_setpoint(roll, pitch, yaw_rate, thrust)
            return True
        except Exception as e:
            logger.error(f"Error sending setpoint: {str(e)}")
            return False
            
    def send_hover_setpoint(self, vx, vy, yaw_rate, height):
        """Send a hover setpoint command to the Crazyflie."""
        if not self.is_connected:
            return False
            
        # Initialize setpoint interface if needed
        if not self._setpoint_initialized:
            if not self.initialize_setpoint():
                return False
        
        try:
            self.cf.commander.send_hover_setpoint(vx, vy, yaw_rate, height)
            return True
        except Exception as e:
            logger.error(f"Error sending hover setpoint: {str(e)}")
            return False
    
    def initialize_motors(self):
        """Initialize motors with warm-up sequence."""
        if not self.is_armed:
            logger.error("Cannot initialize motors: Not armed")
            return False
            
        logger.info("Starting motor initialization sequence")
        
        try:
            # Ensure setpoint interface is initialized
            if not self._setpoint_initialized:
                if not self.initialize_setpoint():
                    return False
            
            # Step 1: Send 20 setpoints at low thrust (10000)
            logger.info("Sending low thrust commands")
            for _ in range(20):
                self.send_setpoint(0, 0, 0, 10000)
                time.sleep(0.02)  # 50 Hz
            
            # Step 2: Send 20 setpoints at medium thrust (20000)
            logger.info("Sending medium thrust commands")
            for _ in range(20):
                self.send_setpoint(0, 0, 0, 20000)
                time.sleep(0.02)  # 50 Hz
            
            logger.info("Motor initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Motor initialization error: {str(e)}")
            return False
    
    #--------------------------------------------------------------------------
    # Data Collection
    #--------------------------------------------------------------------------
    
    def setup_logging(self):
        """Configure logging for height and thrust data."""
        if not self.is_connected or not self.scf or not self.cf:
            logger.error("Cannot setup logging: Not connected or CF objects not ready")
            return False
            
        try:
            # Create log configuration at 50Hz
            self.log_conf = LogConfig(name="ThrustEstimation", period_in_ms=20)
            
            # Add variables to log - stabilizer.thrust is crucial for thrust estimation
            self.log_conf.add_variable("stabilizer.thrust", "uint16_t")  # Get the actual thrust value
            self.log_conf.add_variable("stateEstimate.z", "float")       # Get the current height
            
            # For better accuracy, add more variables if available
            try:
                self.log_conf.add_variable("controller.actuatorThrust", "float")  # May exist in some firmware
            except Exception:
                logger.debug("controller.actuatorThrust not available")
                
            # Register callback
            self.log_conf.data_received_cb.add_callback(self._log_data_callback)
            
            # Add config to the CF
            self.scf.cf.log.add_config(self.log_conf)
            
            # Start logging
            self.log_conf.start()
            logger.info("Logging started")
            
            # Wait for first log data
            timeout = time.time() + 2.0
            while self.current_height == 0.0 and time.time() < timeout:
                time.sleep(0.1)
                
            if self.current_height == 0.0:
                logger.warning("No height data received yet, proceeding anyway")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup logging: {str(e)}")
            return False
    
    def _log_data_callback(self, timestamp, data, logconf):
        """Process incoming log data."""
        try:
            # Extract height data if available
            if 'stateEstimate.z' in data:
                self.current_height = data['stateEstimate.z']
                
            # Extract actual thrust data if available
            actual_thrust = None
            if 'stabilizer.thrust' in data:
                actual_thrust = data['stabilizer.thrust']
                
                # Store thrust data for estimation
                if actual_thrust is not None:
                    self.thrust_data.append({
                        'time': time.time(),
                        'height': self.current_height,
                        'thrust': actual_thrust
                    })
                
            # Periodic logging of received data (every ~1 second)
            if len(self.thrust_data) % 50 == 0:
                logger.debug(f"Log data: height={self.current_height:.2f}m, thrust={actual_thrust}")
                
        except Exception as e:
            logger.error(f"Error in log callback: {str(e)}")
    
    #--------------------------------------------------------------------------
    # Flight Operations
    #--------------------------------------------------------------------------
    
    def take_off(self, target_height):
        """Execute takeoff to specified height using hover setpoint."""
        if not self.is_connected or not self.is_armed:
            logger.error("Cannot take off: Not connected or not armed")
            return False
            
        logger.info(f"Taking off to {target_height}m")
        
        # Initialize motors
        if not self.initialize_motors():
            return False
            
        # Linear ramp-up to target height using 50Hz control
        velocity = 0.3  # m/s
        max_time = target_height / velocity
        start_time = time.time()
        next_control_time = time.time()
        
        while time.time() - start_time < max_time + 1.0:  # Add 1s for stabilization
            # Calculate progress
            elapsed = time.time() - start_time
            progress = min(elapsed / max_time, 1.0)
            current_target = target_height * progress
            
            # Send hover setpoint (no velocity in x/y)
            if not self.send_hover_setpoint(0, 0, 0, current_target):
                logger.error("Failed to send hover setpoint")
                return False
            
            # Log status periodically
            if int(elapsed * 2) % 2 == 0:
                logger.info(f"Takeoff: h={self.current_height:.2f}m, target={current_target:.2f}m")
            
            # Precise timing for 50Hz control
            next_control_time += CONTROL_PERIOD
            sleep_time = next_control_time - time.time()
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Reset timing if we're behind
                if -sleep_time > 0.1:
                    next_control_time = time.time() + CONTROL_PERIOD
        
        logger.info(f"Reached target height: {self.current_height:.2f}m")
        return True
    
    def hover(self, duration, target_height):
        """Maintain hover for specified duration using hover setpoint API."""
        if not self.is_connected or not self.is_armed:
            logger.error("Cannot hover: Not connected or not armed")
            return False
            
        logger.info(f"Hovering at {target_height}m for {duration}s")
        
        # Clear thrust data for a fresh collection
        self.thrust_data = []
        
        # Hover control loop using hover setpoint
        start_time = time.time()
        next_control_time = time.time()
        consecutive_failures = 0
        
        while time.time() - start_time < duration:
            # Send hover setpoint command (maintain position)
            if not self.send_hover_setpoint(0, 0, 0, target_height):
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    logger.error("Too many consecutive failures, aborting hover")
                    return False
            else:
                consecutive_failures = 0
            
            # Log status periodically
            elapsed = time.time() - start_time
            if int(elapsed * 5) % 5 == 0:
                logger.info(f"Hover: h={self.current_height:.2f}m, target={target_height:.2f}m")
            
            # Precise timing for 50Hz control
            next_control_time += CONTROL_PERIOD
            sleep_time = next_control_time - time.time()
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Reset timing if we're behind
                if -sleep_time > 0.1:
                    next_control_time = time.time() + CONTROL_PERIOD
        
        logger.info("Hover completed")
        return True
    
    def land(self):
        """Execute landing sequence using hover setpoint."""
        if not self.is_connected:
            return False
            
        logger.info("Landing...")
        
        try:
            # Gentle descent from current height
            current_height = self.current_height
            descent_velocity = 0.2  # m/s descent rate
            
            # Step 1: Descend to near ground using hover setpoint
            while current_height > 0.05:
                # Calculate height step
                height_step = descent_velocity * CONTROL_PERIOD
                current_height = max(0.05, current_height - height_step)
                
                # Send hover setpoint
                self.send_hover_setpoint(0, 0, 0, current_height)
                
                # Sleep for control period
                time.sleep(CONTROL_PERIOD)
            
            # Step 2: Final approach with reducing thrust using direct setpoint
            logger.info("Final approach: reducing thrust for landing")
            for _ in range(10):
                self.send_setpoint(0, 0, 0, 0)
                time.sleep(CONTROL_PERIOD)
            
            logger.info("Landing complete")
            return True
            
        except Exception as e:
            logger.error(f"Landing error: {str(e)}")
            # Always try to disarm for safety
            self.disarm()
            return False
    
    #--------------------------------------------------------------------------
    # Thrust Estimation
    #--------------------------------------------------------------------------
    
    def estimate_hover_thrust(self):
        """Estimate hover thrust from collected data."""
        if not self.thrust_data:
            logger.error("No thrust data available for estimation")
            return -1
            
        logger.info(f"Analyzing {len(self.thrust_data)} data points for hover thrust estimation")
        
        try:
            # Extract all thrust values
            all_thrusts = [data['thrust'] for data in self.thrust_data if 'thrust' in data]
            
            if not all_thrusts:
                logger.error("No valid thrust values found in data")
                return -1
                
            # Remove outliers if enough data points
            if len(all_thrusts) > 20:
                # Remove top and bottom 10%
                sorted_thrusts = sorted(all_thrusts)
                cutoff = int(len(sorted_thrusts) * 0.1)
                filtered_thrusts = sorted_thrusts[cutoff:-cutoff]
                logger.info(f"Removed {len(sorted_thrusts) - len(filtered_thrusts)} outliers")
                all_thrusts = filtered_thrusts
            
            # Calculate mean thrust
            estimated_thrust = int(sum(all_thrusts) / len(all_thrusts))
            logger.info(f"Estimated hover thrust = {estimated_thrust}")
            
            return estimated_thrust
                
        except Exception as e:
            logger.error(f"Error estimating hover thrust: {str(e)}")
            return -1
    
    def save_results(self, estimated_thrust, output_file):
        """Save estimation results to file."""
        data = {
            'hover_thrust': estimated_thrust,
            'takeoff_thrust_initial': estimated_thrust + 1500,
            'estimated_at': datetime.now().isoformat(),
            'uri': self.uri,
            'data_points': len(self.thrust_data)
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
    
    #--------------------------------------------------------------------------
    # Main Estimation Sequence
    #--------------------------------------------------------------------------
    
    def run_estimation(self, target_height, duration, output_file):
        """Run the full estimation procedure."""
        success = False
        estimated_thrust = -1
        
        try:
            # Step 1: Connect and setup
            if not self.connect():
                logger.error("Failed to connect to Crazyflie")
                return False, -1
            
            # Step 2: Initialize setpoint interface
            if not self.initialize_setpoint():
                logger.error("Failed to initialize setpoint interface")
                self.disconnect()
                return False, -1
                
            # Step 3: Setup logging
            if not self.setup_logging():
                logger.error("Failed to setup logging")
                self.disconnect()
                return False, -1
                
            # Step 4: Arm motors
            if not self.arm():
                logger.error("Failed to arm motors")
                self.disconnect()
                return False, -1
                
            # Step 5: Take off
            if not self.take_off(target_height):
                logger.error("Takeoff failed")
                self.land()
                self.disarm()
                self.disconnect()
                return False, -1
                
            # Step 6: Hover and collect data
            if not self.hover(duration, target_height):
                logger.error("Hover failed")
                self.land()
                self.disarm()
                self.disconnect()
                return False, -1
                
            # Step 7: Estimate hover thrust
            estimated_thrust = self.estimate_hover_thrust()
            success = estimated_thrust > 0
            
            # Step 8: Save results if successful
            if success:
                self.save_results(estimated_thrust, output_file)
            
            # Step 9: Land and cleanup    
            self.land()
            self.disarm()
            self.disconnect()
            
            return success, estimated_thrust
            
        except Exception as e:
            logger.error(f"Error during estimation: {str(e)}")
            # Always try to land and disconnect safely
            try:
                self.land()
                self.disarm()
                self.disconnect()
            except Exception as cleanup_error:
                logger.error(f"Additional error during cleanup: {str(cleanup_error)}")
            return False, -1


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Crazyflie Hover Thrust Estimator')
    
    parser.add_argument('--uri', type=str, default=DEFAULT_URI,
                        help=f'Crazyflie URI (default: {DEFAULT_URI})')
    
    parser.add_argument('--height', type=float, default=DEFAULT_HEIGHT,
                        help=f'Target hover height in meters (default: {DEFAULT_HEIGHT}m)')
    
    parser.add_argument('--duration', type=float, default=DEFAULT_DURATION,
                        help=f'Hover duration in seconds (default: {DEFAULT_DURATION}s)')
    
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_FILE,
                        help=f'Output JSON file (default: {DEFAULT_OUTPUT_FILE})')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print banner
    print("\n" + "="*62)
    print(" CRAZYFLIE HOVER THRUST ESTIMATOR ".center(62))
    print("="*62)
    print(f"URI:      {args.uri}")
    print(f"Height:   {args.height}m")
    print(f"Duration: {args.duration}s")
    print(f"Output:   {args.output}")
    print("="*62 + "\n")
    
    # Create and run estimator
    estimator = ThrustEstimator(uri=args.uri)
    success, thrust = estimator.run_estimation(
        target_height=args.height,
        duration=args.duration,
        output_file=args.output
    )
    
    # Display results
    if success:
        print("\n" + "="*62)
        print(" ESTIMATION RESULTS ".center(62))
        print("="*62)
        print(f"Estimated hover thrust: {thrust}")
        print(f"Takeoff initial thrust: {thrust + 1500}")
        print(f"Results saved to: {args.output}")
        print("="*62 + "\n")
        return 0
    else:
        print("\n" + "="*62)
        print(" ESTIMATION FAILED ".center(62))
        print("="*62 + "\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
