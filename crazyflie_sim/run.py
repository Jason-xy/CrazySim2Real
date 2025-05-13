#!/usr/bin/env python3
"""
Main script to run the Crazyflie simulator with HTTP API.
"""
import os
import sys
import argparse
import logging
import time
import signal
from isaaclab.app import AppLauncher

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crazyflie_sim.config import (
    SIM_DT, POSITION_CONTROLLER, ATTITUDE_CONTROLLER, 
    PHYSICS, SERVER, LOGGING
)

# Parse command line arguments
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Crazyflie Simulator")
    
    # Server settings
    parser.add_argument("--host", type=str, default=SERVER.get("host"),
                       help=f"Server host (default: {SERVER.get('host')})")
    parser.add_argument("--port", type=int, default=SERVER.get("port"),
                       help=f"Server port (default: {SERVER.get('port')})")
    
    # Simulation settings
    parser.add_argument("--dt", type=float, default=SIM_DT,
                       help=f"Simulation time step in seconds (default: {SIM_DT})")
    
    # Append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    
    return parser.parse_args()
args = parse_args()

# Initialize Omniverse app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Use absolute imports
from crazyflie_sim.sim.simulation_manager import SimulationManager
from crazyflie_sim.api.server import SimulatorAPIServer

# Add global running flag
running = True

def setup_logging():
    """Set up logging configuration."""
    logging_format = LOGGING.get("format", "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    logging_level = getattr(logging, LOGGING.get("level", "INFO"))
    logging_file = LOGGING.get("file", "crazyflie_sim.log")
    
    # Create formatter
    formatter = logging.Formatter(logging_format)
    
    # Setup file handler
    file_handler = logging.FileHandler(logging_file)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Logging initialized at level {logging_level}")

def setup_signal_handlers(api_server):
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(sig, frame):
        global running
        logging.info("Received shutdown signal, cleaning up...")
        running = False  # Set the global flag
        api_server.stop()
        logging.info("Shutdown complete")
        # Don't exit immediately to allow main loop to terminate cleanly
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Main entry point."""
    
    # Setup logging
    setup_logging()
    
    logging.info("Starting Crazyflie Simulator")
    
    try:
        # Initialize simulation manager
        sim_manager = SimulationManager(
            simulation_app=simulation_app,
            dt=args.dt,
            position_controller_params=POSITION_CONTROLLER,
            attitude_controller_params=ATTITUDE_CONTROLLER,
            physics_params=PHYSICS
        )
        
        # Initialize HTTP API server
        api_server = SimulatorAPIServer(
            host=args.host,
            port=args.port,
            simulation_manager=sim_manager
        )
        
        # Start the API server
        api_server.start()
        
        # Set up signal handlers
        setup_signal_handlers(api_server)
        
        logging.info(f"Simulator running. API server available at http://{args.host}:{args.port}")
        
        # Main simulation loop - run directly in main thread
        while running:  # Check the global running flag
            # Run a single simulation step
            should_continue = sim_manager.step()
            
            # Check if we should exit
            if not should_continue:
                break
        
        # Cleanup when done
        logging.info("Simulation ended")
        api_server.stop()
        
    except Exception as e:
        logging.error(f"Error running simulator: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 