#!/usr/bin/env python3
"""
Client for controlling the Crazyflie simulator with a keyboard.
"""
import os
import sys
import argparse
import logging
import time
import json
import threading
import enum
from typing import Dict, Any
from pynput import keyboard
import requests

SERVER = {
    "host": "localhost",
    "port": 8000
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("crazyflie_client.log")
    ]
)
logger = logging.getLogger("crazyflie_client")

class ControlMode(enum.Enum):
    """Control modes available for the drone."""
    ATTITUDE = 1  # Direct attitude control (roll, pitch, yaw rate, thrust)
    POSITION = 2  # Position setpoint control (x, y, z coordinates)
    VELOCITY = 3  # Velocity setpoint control (vx, vy, vz)
    ASSISTED = 4  # Assisted attitude control with auto-leveling

    def __str__(self):
        """Return a user-friendly string representation of the mode."""
        return self.name.capitalize()

class CrazyflieClient:
    """
    Client for controlling the Crazyflie simulator with a keyboard.
    """
    
    def __init__(self, host: str, port: int):
        """
        Initialize the client.
        
        Args:
            host: Server hostname or IP address
            port: Server port
        """
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        
        # Current drone state
        self.state = {}
        
        # Current attitude command
        self.attitude_cmd = {
            "roll": 0.0,     # Roll angle normalized [-1, 1] (represents approx. ±30° after scaling)
            "pitch": 0.0,    # Pitch angle normalized [-1, 1] (represents approx. ±30° after scaling)
            "yaw_rate": 0.0, # Yaw rate normalized [-1, 1] (represents approx. ±200°/s after scaling)
            "thrust": 0.5    # Thrust normalized [0, 1] (Default to mid-thrust)
        }
        
        # Position and velocity setpoints
        self.position_setpoint = {"x": 0.0, "y": 0.0, "z": 1.0}
        self.velocity_setpoint = {"x": 0.0, "y": 0.0, "z": 0.0}
        
        # Keyboard control parameters - smaller values for smoother control
        self.control_increment = 0.05  # Reduced from 0.1
        self.thrust_increment = 0.02   # Reduced from 0.05
        self.position_increment = 0.1  # Meters
        self.velocity_increment = 0.2  # m/s
        
        # Set of currently pressed keys
        self.pressed_keys = set()
        
        # Last key press time for rate limiting
        self.last_key_press_time = {}
        self.key_repeat_delay = 0.1  # Seconds between repeated actions
        
        # Control flags
        self.running = True
        self.control_mode = ControlMode.ATTITUDE  # Default to attitude control
        
        # Take-off and land heights
        self.takeoff_height = 1.0
        self.land_height = 0.0
        
        # Continuous control for arrow keys
        self.continuous_control_active = False
        self.continuous_control_thread = None
        
        # Check if terminal supports ANSI colors
        self.use_colors = True
        if "TERM" in os.environ:
            if os.environ["TERM"] == "dumb" or os.environ["TERM"] == "":
                self.use_colors = False
        
        logger.info(f"Client initialized for server at {self.base_url}")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current drone state from the server.
        
        Returns:
            Dictionary containing drone state
        """
        try:
            response = requests.get(f"{self.base_url}/state", timeout=1.0)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting state: {e}")
            return {}
    
    def send_attitude_cmd(self, roll: float = None, pitch: float = None, 
                          yaw_rate: float = None, thrust: float = None):
        """
        Send an attitude command to the server.
        
        Args:
            roll: Roll angle normalized [-1, 1]
            pitch: Pitch angle normalized [-1, 1]
            yaw_rate: Yaw rate normalized [-1, 1]
            thrust: Thrust normalized [0, 1]
        """
        # Update only provided values
        if roll is not None:
            self.attitude_cmd["roll"] = max(-1.0, min(1.0, roll))
        if pitch is not None:
            self.attitude_cmd["pitch"] = max(-1.0, min(1.0, pitch))
        if yaw_rate is not None:
            self.attitude_cmd["yaw_rate"] = max(-1.0, min(1.0, yaw_rate))
        if thrust is not None:
            self.attitude_cmd["thrust"] = max(0.0, min(1.0, thrust))
        
        try:
            response = requests.post(
                f"{self.base_url}/control/attitude",
                json=self.attitude_cmd,
                timeout=1.0
            )
            response.raise_for_status()
            logger.debug(f"Attitude command sent: {self.attitude_cmd}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending attitude command: {e}")
    
    def send_position_cmd(self, x: float, y: float, z: float, yaw: float = None):
        """
        Send a position command to the server.
        
        Args:
            x: X position (m)
            y: Y position (m) 
            z: Z position (m)
            yaw: Yaw angle in degrees (optional)
        """
        try:
            cmd = {"x": x, "y": y, "z": z}
            if yaw is not None:
                cmd["yaw"] = yaw
                
            response = requests.post(
                f"{self.base_url}/control/position",
                json=cmd,
                timeout=1.0
            )
            response.raise_for_status()
            
            # Update position setpoint
            self.position_setpoint = cmd
            
            # Log the message
            log_msg = f"Position command sent: x={x:.2f}, y={y:.2f}, z={z:.2f}"
            if yaw is not None:
                log_msg += f", yaw={yaw:.2f}°"
            logger.info(log_msg)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending position command: {e}")
    
    def send_velocity_cmd(self, vx: float, vy: float, vz: float):
        """
        Send a velocity command to the server.
        
        Args:
            vx: X velocity (m/s)
            vy: Y velocity (m/s)
            vz: Z velocity (m/s)
        """
        # This is a simplified implementation - in a real system, you'd have a 
        # velocity control endpoint or convert velocity to position increments
        # For now, we'll just adjust the position setpoint based on velocity
        self.velocity_setpoint = {"x": vx, "y": vy, "z": vz}
        
        # Get current position
        position = self.state.get("position", {"x": 0.0, "y": 0.0, "z": 0.0})
        
        # Integrate velocity to get new position setpoint (simplified)
        new_x = position["x"] + vx * 0.1  # Integrate over 0.1 seconds
        new_y = position["y"] + vy * 0.1
        new_z = position["z"] + vz * 0.1
        
        # Send position command
        self.send_position_cmd(new_x, new_y, new_z)
        logger.info(f"Velocity command sent: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}")
    
    def reset_simulation(self):
        """Reset the simulation."""
        try:
            response = requests.post(f"{self.base_url}/reset", timeout=1.0)
            response.raise_for_status()
            logger.info("Simulation reset")
            
            # Reset local attitude command
            self.attitude_cmd = {
                "roll": 0.0,
                "pitch": 0.0,
                "yaw_rate": 0.0,
                "thrust": 0.5
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Error resetting simulation: {e}")
    
    def take_off(self, yaw: float = 0.0):
        """
        Send take-off command (position control).
        
        Args:
            yaw: Yaw angle in degrees (default: 0.0)
        """
        self.send_position_cmd(0.0, 0.0, self.takeoff_height, yaw)
        logger.info(f"Take-off command sent: height={self.takeoff_height}m, yaw={yaw}°")
    
    def land(self, yaw: float = None):
        """
        Send landing command (position control).
        
        Args:
            yaw: Yaw angle in degrees (optional - keeps current yaw if None)
        """
        self.send_position_cmd(0.0, 0.0, self.land_height, yaw)
        log_msg = "Landing command sent"
        if yaw is not None:
            log_msg += f" with yaw={yaw}°"
        logger.info(log_msg)
    
    def adjust_attitude(self, axis: str, value: float):
        """
        Adjust attitude command value for the specified axis.
        
        Args:
            axis: Command axis ('roll', 'pitch', 'yaw_rate', or 'thrust')
            value: Increment value
        """
        if axis not in self.attitude_cmd:
            logger.error(f"Invalid attitude axis: {axis}")
            return
        
        current_value = self.attitude_cmd[axis]
        new_value = current_value + value
        
        # Clamp to valid range
        if axis == 'thrust':
            new_value = max(0.0, min(1.0, new_value))
        else:
            new_value = max(-1.0, min(1.0, new_value))
        
        # Send updated command
        self.send_attitude_cmd(**{axis: new_value})
    
    def switch_control_mode(self):
        """Cycle to the next control mode."""
        # Get list of all control modes
        all_modes = list(ControlMode)
        
        # Find current mode index
        current_index = all_modes.index(self.control_mode)
        
        # Switch to next mode (cycle back to first if at the end)
        next_index = (current_index + 1) % len(all_modes)
        self.control_mode = all_modes[next_index]
        
        # Reset control values
        if self.control_mode == ControlMode.ATTITUDE:
            self.attitude_cmd = {
                "roll": 0.0,
                "pitch": 0.0,
                "yaw_rate": 0.0,
                "thrust": 0.5
            }
        elif self.control_mode == ControlMode.POSITION:
            # Set position setpoint to current position plus a small height
            position = self.state.get("position", {"x": 0.0, "y": 0.0, "z": 0.0})
            self.position_setpoint = {
                "x": position["x"],
                "y": position["y"],
                "z": position["z"] + 0.1  # Small height increase for safety
            }
        elif self.control_mode == ControlMode.VELOCITY:
            self.velocity_setpoint = {"x": 0.0, "y": 0.0, "z": 0.0}
        elif self.control_mode == ControlMode.ASSISTED:
            self.attitude_cmd = {
                "roll": 0.0,
                "pitch": 0.0,
                "yaw_rate": 0.0,
                "thrust": 0.5
            }
            
        # Display new mode prominently
        if self.use_colors:
            self._log_message(f"Switched to {self._color_text(str(self.control_mode), '1;33')} control mode")
        else:
            self._log_message(f"Switched to {self.control_mode} control mode")
    
    def continuous_control_loop(self):
        """Thread for continuous control updates for keys that need continuous updates"""
        logger.debug("Starting continuous control loop")
        last_control_time = time.time()
        
        while self.running and self.continuous_control_active:
            current_time = time.time()
            
            # Update controls at ~50Hz (increased from 20Hz) for more responsive control
            if current_time - last_control_time >= 0.02:
                # Process keys based on current control mode
                if self.control_mode == ControlMode.ATTITUDE:
                    self._process_attitude_control_keys()
                elif self.control_mode == ControlMode.POSITION:
                    self._process_position_control_keys()
                elif self.control_mode == ControlMode.VELOCITY:
                    self._process_velocity_control_keys()
                elif self.control_mode == ControlMode.ASSISTED:
                    self._process_assisted_control_keys()
                
                last_control_time = current_time
            
            time.sleep(0.01)  # 100Hz polling for more responsive controls
        
        logger.debug("Exiting continuous control loop")
    
    def _color_text(self, text, color_code=""):
        """
        Apply ANSI color code to text if colors are enabled.
        
        Args:
            text: Text to format
            color_code: ANSI color code (e.g., "33" for yellow)
            
        Returns:
            Colored text if colors are enabled, otherwise plain text
        """
        if not self.use_colors:
            return text
        
        if not color_code:
            return text
            
        return f"\033[{color_code}m{text}\033[0m"
    
    def clear_terminal(self):
        """Clear the terminal screen using ANSI escape codes."""
        if self.use_colors:
            # ANSI escape code to clear screen and move cursor to top-left
            print("\033[2J\033[H", end="")
        else:
            # Fall back to printing newlines
            print("\n" * 50)
    
    def _process_attitude_control_keys(self):
        """Process keys for attitude control mode."""
        # Process keys that need continuous updates
        processed_pitch = False
        processed_roll = False
        processed_yaw = False
        
        for key_name in self.pressed_keys:
            # Pitch control (W/S)
            if key_name == 'w' and not processed_pitch:
                self._log_message("Continuously applying forward pitch")
                self.adjust_attitude('pitch', self.control_increment)
                processed_pitch = True
            elif key_name == 's' and not processed_pitch:
                self._log_message("Continuously applying backward pitch")
                self.adjust_attitude('pitch', -self.control_increment)
                processed_pitch = True
            
            # Roll control (A/D)
            elif key_name == 'a' and not processed_roll:
                self._log_message("Continuously applying left roll")
                self.adjust_attitude('roll', -self.control_increment)
                processed_roll = True
            elif key_name == 'd' and not processed_roll:
                self._log_message("Continuously applying right roll")
                self.adjust_attitude('roll', self.control_increment)
                processed_roll = True
            
            # Yaw control (Q/E)
            elif key_name == 'q' and not processed_yaw:
                self._log_message("Continuously applying left yaw")
                self.adjust_attitude('yaw_rate', -self.control_increment)
                processed_yaw = True
            elif key_name == 'e' and not processed_yaw:
                self._log_message("Continuously applying right yaw")
                self.adjust_attitude('yaw_rate', self.control_increment)
                processed_yaw = True
            
            # Thrust control (Up/Down arrows)
            elif key_name == 'up':
                self._log_message("Continuously increasing thrust")
                self.adjust_attitude('thrust', self.thrust_increment)
            elif key_name == 'down':
                self._log_message("Continuously decreasing thrust")
                self.adjust_attitude('thrust', -self.thrust_increment)
    
    def _process_position_control_keys(self):
        """Process keys for position control mode."""
        # Extract current position setpoint
        x = self.position_setpoint.get("x", 0.0)
        y = self.position_setpoint.get("y", 0.0)
        z = self.position_setpoint.get("z", 0.0)
        # Get current yaw if it exists in the setpoint, otherwise use 0
        yaw = self.position_setpoint.get("yaw", 0.0)
        
        # Flag to track if setpoint has changed
        position_changed = False
        yaw_changed = False
        
        # Yaw increment in degrees
        yaw_increment = 5.0  # 5 degrees per key press
        
        for key_name in self.pressed_keys:
            # Forward/Backward movement (W/S)
            if key_name == 'w':
                x += self.position_increment
                position_changed = True
                self._log_message(f"Position control: forward to x={x:.2f}")
            elif key_name == 's':
                x -= self.position_increment
                position_changed = True
                self._log_message(f"Position control: backward to x={x:.2f}")
            
            # Left/Right movement (A/D)
            elif key_name == 'a':
                y += self.position_increment  # Note: Y axis is different in many coordinate systems
                position_changed = True
                self._log_message(f"Position control: left to y={y:.2f}")
            elif key_name == 'd':
                y -= self.position_increment
                position_changed = True
                self._log_message(f"Position control: right to y={y:.2f}")
            
            # Up/Down movement (Up/Down arrows)
            elif key_name == 'up':
                z += self.position_increment
                position_changed = True
                self._log_message(f"Position control: up to z={z:.2f}")
            elif key_name == 'down':
                z -= self.position_increment
                position_changed = True
                self._log_message(f"Position control: down to z={z:.2f}")
            
            # Yaw rotation (Q/E)
            elif key_name == 'q':
                yaw -= yaw_increment
                # Normalize to -180 to 180 degrees
                yaw = ((yaw + 180) % 360) - 180
                yaw_changed = True
                self._log_message(f"Position control: rotate left to yaw={yaw:.2f}°")
            elif key_name == 'e':
                yaw += yaw_increment
                # Normalize to -180 to 180 degrees
                yaw = ((yaw + 180) % 360) - 180
                yaw_changed = True
                self._log_message(f"Position control: rotate right to yaw={yaw:.2f}°")
        
        # Send position command if setpoint has changed
        if position_changed or yaw_changed:
            self.position_setpoint = {"x": x, "y": y, "z": z, "yaw": yaw}
            self.send_position_cmd(x, y, z, yaw)
            
    def _log_message(self, message):
        """Log message and display it in the system message area."""
        logger.debug(message)
        # Update the system message area
        if self.use_colors:
            print("\033[19;1H\033[K", end="")  # Move to message area and clear line
            print(self._color_text(message, "35"))  # Magenta text for messages
        else:
            print(message)
    
    def _process_velocity_control_keys(self):
        """Process keys for velocity control mode."""
        # Extract current velocity setpoint
        vx = 0.0
        vy = 0.0
        vz = 0.0
        
        # Flag to track if setpoint has changed
        velocity_changed = False
        
        for key_name in self.pressed_keys:
            # Forward/Backward velocity (W/S)
            if key_name == 'w':
                vx = self.velocity_increment
                velocity_changed = True
                self._log_message(f"Velocity control: forward at vx={vx:.2f}")
            elif key_name == 's':
                vx = -self.velocity_increment
                velocity_changed = True
                self._log_message(f"Velocity control: backward at vx={vx:.2f}")
            
            # Left/Right velocity (A/D)
            elif key_name == 'a':
                vy = self.velocity_increment
                velocity_changed = True
                self._log_message(f"Velocity control: left at vy={vy:.2f}")
            elif key_name == 'd':
                vy = -self.velocity_increment
                velocity_changed = True
                self._log_message(f"Velocity control: right at vy={vy:.2f}")
            
            # Up/Down velocity (Up/Down arrows)
            elif key_name == 'up':
                vz = self.velocity_increment
                velocity_changed = True
                self._log_message(f"Velocity control: up at vz={vz:.2f}")
            elif key_name == 'down':
                vz = -self.velocity_increment
                velocity_changed = True
                self._log_message(f"Velocity control: down at vz={vz:.2f}")
        
        # Send velocity command if setpoint has changed
        if velocity_changed:
            self.velocity_setpoint = {"x": vx, "y": vy, "z": vz}
            self.send_velocity_cmd(vx, vy, vz)
        elif any(k in self.pressed_keys for k in {'w', 's', 'a', 'd', 'up', 'down'}):
            # If control keys are pressed but no change (e.g., conflicting keys),
            # send zero velocity to stop movement
            self.velocity_setpoint = {"x": 0.0, "y": 0.0, "z": 0.0}
            self.send_velocity_cmd(0.0, 0.0, 0.0)
            self._log_message("Stopping all movement due to conflicting keys")
    
    def _process_assisted_control_keys(self):
        """Process keys for assisted attitude control mode with auto-leveling."""
        # Assisted mode uses attitude control but with simplified inputs
        # and automatic leveling when keys are released
        processed_pitch = False
        processed_roll = False
        processed_yaw = False
        
        for key_name in self.pressed_keys:
            # Pitch control (W/S) - with lower sensitivity
            if key_name == 'w' and not processed_pitch:
                self._log_message("Assisted mode: forward tilt")
                self.adjust_attitude('pitch', self.control_increment * 0.7)
                processed_pitch = True
            elif key_name == 's' and not processed_pitch:
                self._log_message("Assisted mode: backward tilt")
                self.adjust_attitude('pitch', -self.control_increment * 0.7)
                processed_pitch = True
            
            # Roll control (A/D) - with lower sensitivity
            elif key_name == 'a' and not processed_roll:
                self._log_message("Assisted mode: left tilt")
                self.adjust_attitude('roll', -self.control_increment * 0.7)
                processed_roll = True
            elif key_name == 'd' and not processed_roll:
                self._log_message("Assisted mode: right tilt")
                self.adjust_attitude('roll', self.control_increment * 0.7)
                processed_roll = True
            
            # Yaw control (Q/E) - standard sensitivity
            elif key_name == 'q' and not processed_yaw:
                self._log_message("Assisted mode: left rotation")
                self.adjust_attitude('yaw_rate', -self.control_increment)
                processed_yaw = True
            elif key_name == 'e' and not processed_yaw:
                self._log_message("Assisted mode: right rotation")
                self.adjust_attitude('yaw_rate', self.control_increment)
                processed_yaw = True
            
            # Thrust control (Up/Down arrows)
            elif key_name == 'up':
                self._log_message("Assisted mode: increase height")
                self.adjust_attitude('thrust', self.thrust_increment * 1.2)  # Slightly stronger thrust
            elif key_name == 'down':
                self._log_message("Assisted mode: decrease height")
                self.adjust_attitude('thrust', -self.thrust_increment)
        
        # Auto-leveling: gradually reduce pitch and roll to zero if not being controlled
        if not processed_pitch and abs(self.attitude_cmd["pitch"]) > 0.01:
            # Gradually reduce pitch toward zero for auto-leveling
            if self.attitude_cmd["pitch"] > 0:
                self.attitude_cmd["pitch"] -= min(0.02, self.attitude_cmd["pitch"])
            else:
                self.attitude_cmd["pitch"] += min(0.02, abs(self.attitude_cmd["pitch"]))
            self.send_attitude_cmd(pitch=self.attitude_cmd["pitch"])
            if abs(self.attitude_cmd["pitch"]) < 0.02:
                self._log_message("Auto-leveling: pitch stabilized")
        
        if not processed_roll and abs(self.attitude_cmd["roll"]) > 0.01:
            # Gradually reduce roll toward zero for auto-leveling
            if self.attitude_cmd["roll"] > 0:
                self.attitude_cmd["roll"] -= min(0.02, self.attitude_cmd["roll"])
            else:
                self.attitude_cmd["roll"] += min(0.02, abs(self.attitude_cmd["roll"]))
            self.send_attitude_cmd(roll=self.attitude_cmd["roll"])
            if abs(self.attitude_cmd["roll"]) < 0.02:
                self._log_message("Auto-leveling: roll stabilized")
        
        if not processed_yaw and abs(self.attitude_cmd["yaw_rate"]) > 0.01:
            # Gradually reduce yaw_rate toward zero
            self.send_attitude_cmd(yaw_rate=0.0)
            self._log_message("Auto-leveling: yaw rate stabilized")

    def display_state_loop(self):
        """Continuously poll and display drone state."""
        last_display_time = time.time()
        # Clear terminal at startup
        self.clear_terminal()
        
        # Print static header once - using colors
        print(self._color_text("Crazyflie Simulator Control Interface", "1;36")) # Cyan, bold
        print(self._color_text("====================================", "36"))     # Cyan
        print()
        print(self._color_text("Controls:", "1;33"))  # Yellow, bold
        print(f"  {self._color_text('W/S', '33')}: Forward/Backward")
        print(f"  {self._color_text('A/D', '33')}: Left/Right")
        print(f"  {self._color_text('Q/E', '33')}: Rotate Left/Right")
        print(f"  {self._color_text('Up/Down', '33')}: Increase/Decrease Altitude")
        print(f"  {self._color_text('M', '33')}: Switch control mode")
        print(f"  {self._color_text('T', '33')}: Take off")
        print(f"  {self._color_text('L', '33')}: Land")
        print(f"  {self._color_text('R', '33')}: Reset simulation")
        print(f"  {self._color_text('ESC', '33')}: Exit program")
        print()
        print(self._color_text("Drone State:", "1;32"))  # Green, bold
        print()  # Leave lines for state data
        print()
        print()
        print()
        print()
        print()
        print()
        print()
        print(self._color_text("System Messages:", "1;35"))  # Magenta, bold
        
        # Store the last state to avoid unnecessary refreshes
        last_state_hash = ""
        
        while self.running:
            current_time = time.time()
            
            # Update at ~10 Hz
            if current_time - last_display_time >= 0.1:
                self.state = self.get_state()
                
                # Only refresh display if state has changed
                current_hash = hash(str(self.state) + str(self.control_mode))
                if current_hash != last_state_hash:
                    self.display_state()
                    last_state_hash = current_hash
                
                last_display_time = current_time
            
            time.sleep(0.02)  # 50Hz polling for smoother display
    
    def display_state(self):
        """Display current drone state in the console with fixed positioning."""
        try:
            # Get current state
            state = self.get_state()
            
            if not state:
                # Move cursor to drone state section and print message
                if self.use_colors:
                    print("\033[13;1H\033[K", end="")  # Move to line 13, column 1 and clear line
                    print(self._color_text("No state data available.", "31"))  # Red text
                else:
                    print("No state data available.")
                return
            
            # Extract values with safe defaults
            position = state.get('position', {"x": 0.0, "y": 0.0, "z": 0.0})
            velocity = state.get('velocity', {"x": 0.0, "y": 0.0, "z": 0.0})
            orientation = state.get('orientation', {"roll": 0.0, "pitch": 0.0, "yaw": 0.0})
            angular_velocity = state.get('angular_velocity', {"x": 0.0, "y": 0.0, "z": 0.0})
            
            # Prepare display strings
            if self.use_colors:
                # Move cursor to drone state section
                print("\033[13;1H", end="")  # Move to line 13, column 1
                
                # Format display strings with clear line after each line and color
                print("\033[K{} {}".format(
                    self._color_text('Control Mode:', '1;33'), 
                    self.control_mode
                ))
                
                # Position display with formatted values
                pos_x = self._color_text("{:.2f}".format(position.get('x', 0.0)), '1')
                pos_y = self._color_text("{:.2f}".format(position.get('y', 0.0)), '1')
                pos_z = self._color_text("{:.2f}".format(position.get('z', 0.0)), '1')
                print("\033[K{} x={}, y={}, z={} m".format(
                    self._color_text('Position:', '32'), 
                    pos_x, pos_y, pos_z
                ))
                
                # Velocity display
                vel_x = self._color_text("{:.2f}".format(velocity.get('x', 0.0)), '1')
                vel_y = self._color_text("{:.2f}".format(velocity.get('y', 0.0)), '1')
                vel_z = self._color_text("{:.2f}".format(velocity.get('z', 0.0)), '1')
                print("\033[K{} vx={}, vy={}, vz={} m/s".format(
                    self._color_text('Velocity:', '32'), 
                    vel_x, vel_y, vel_z
                ))
                
                # Attitude display
                roll = self._color_text("{:.1f}".format(orientation.get('roll', 0.0)), '1')
                pitch = self._color_text("{:.1f}".format(orientation.get('pitch', 0.0)), '1')
                yaw = self._color_text("{:.1f}".format(orientation.get('yaw', 0.0)), '1')
                print("\033[K{} roll={}°, pitch={}°, yaw={}°".format(
                    self._color_text('Attitude:', '32'), 
                    roll, pitch, yaw
                ))
                
                # Angular velocity display
                ang_x = self._color_text("{:.1f}".format(angular_velocity.get('x', 0.0)), '1')
                ang_y = self._color_text("{:.1f}".format(angular_velocity.get('y', 0.0)), '1')
                ang_z = self._color_text("{:.1f}".format(angular_velocity.get('z', 0.0)), '1')
                print("\033[K{} x={}°/s, y={}°/s, z={}°/s".format(
                    self._color_text('Angular velocity:', '32'), 
                    ang_x, ang_y, ang_z
                ))
                
                # Display current command based on mode
                print("\033[K{} ".format(self._color_text('Current command:', '32')), end="")
                
                if self.control_mode == ControlMode.ATTITUDE or self.control_mode == ControlMode.ASSISTED:
                    roll = self._color_text("{:.2f}".format(self.attitude_cmd['roll']), '1')
                    pitch = self._color_text("{:.2f}".format(self.attitude_cmd['pitch']), '1')
                    yaw_rate = self._color_text("{:.2f}".format(self.attitude_cmd['yaw_rate']), '1')
                    thrust = self._color_text("{:.2f}".format(self.attitude_cmd['thrust']), '1')
                    print("roll={}, pitch={}, yaw_rate={}, thrust={}".format(
                        roll, pitch, yaw_rate, thrust
                    ))
                elif self.control_mode == ControlMode.POSITION:
                    pos = self.position_setpoint
                    x = self._color_text("{:.2f}".format(pos.get('x', 0.0)), '1')
                    y = self._color_text("{:.2f}".format(pos.get('y', 0.0)), '1')
                    z = self._color_text("{:.2f}".format(pos.get('z', 0.0)), '1')
                    yaw_str = ""
                    if 'yaw' in pos:
                        yaw = self._color_text("{:.2f}".format(pos.get('yaw', 0.0)), '1')
                        yaw_str = ", yaw={}°".format(yaw)
                    print("x={}, y={}, z={}{}".format(x, y, z, yaw_str))
                elif self.control_mode == ControlMode.VELOCITY:
                    vel = self.velocity_setpoint
                    vx = self._color_text("{:.2f}".format(vel.get('x', 0.0)), '1')
                    vy = self._color_text("{:.2f}".format(vel.get('y', 0.0)), '1')
                    vz = self._color_text("{:.2f}".format(vel.get('z', 0.0)), '1')
                    print("vx={}, vy={}, vz={}".format(vx, vy, vz))
            else:
                # Simplified output for terminals without ANSI support
                print(f"Control Mode: {self.control_mode}")
                print(f"Position: x={position.get('x', 0.0):.2f}, y={position.get('y', 0.0):.2f}, z={position.get('z', 0.0):.2f} m")
                print(f"Velocity: vx={velocity.get('x', 0.0):.2f}, vy={velocity.get('y', 0.0):.2f}, vz={velocity.get('z', 0.0):.2f} m/s")
                print(f"Attitude: roll={orientation.get('roll', 0.0):.1f}°, pitch={orientation.get('pitch', 0.0):.1f}°, yaw={orientation.get('yaw', 0.0):.1f}°")
                print(f"Angular velocity: x={angular_velocity.get('x', 0.0):.1f}°/s, y={angular_velocity.get('y', 0.0):.1f}°/s, z={angular_velocity.get('z', 0.0):.1f}°/s")
                
                # Display current command based on mode
                if self.control_mode == ControlMode.ATTITUDE or self.control_mode == ControlMode.ASSISTED:
                    print(f"Current command: roll={self.attitude_cmd['roll']:.2f}, pitch={self.attitude_cmd['pitch']:.2f}, " +
                          f"yaw_rate={self.attitude_cmd['yaw_rate']:.2f}, thrust={self.attitude_cmd['thrust']:.2f}")
                elif self.control_mode == ControlMode.POSITION:
                    pos = self.position_setpoint
                    yaw_str = f", yaw={pos.get('yaw', 0.0):.2f}°" if 'yaw' in pos else ""
                    print(f"Current command: x={pos.get('x', 0.0):.2f}, y={pos.get('y', 0.0):.2f}, z={pos.get('z', 0.0):.2f}{yaw_str}")
                elif self.control_mode == ControlMode.VELOCITY:
                    vel = self.velocity_setpoint
                    print(f"Current command: vx={vel.get('x', 0.0):.2f}, vy={vel.get('y', 0.0):.2f}, vz={vel.get('z', 0.0):.2f}")
            
        except Exception as e:
            # Move to message section
            if self.use_colors:
                print("\033[19;1H\033[K", end="")  # Move to line 19, column 1 and clear line
                print(self._color_text(f"Error displaying state: {e}", "31"))  # Red text
            else:
                print(f"Error displaying state: {e}")
    
    def keyboard_control_loop(self):
        """Handle keyboard input for real-time control."""
        # Control loop already running, no need to show help text again
        
        try:
            # Define callbacks for keyboard events
            def on_press(key):
                # Extract key_name with proper handling for special keys
                try:
                    # For regular keys (letters, numbers, etc.)
                    key_name = key.char.lower()
                    logger.debug(f"Regular key pressed: {key_name}")
                except AttributeError:
                    # For special keys (arrows, shift, etc.)
                    key_name = str(key).replace('Key.', '')
                    logger.debug(f"Special key pressed: {key_name}")
                
                # Add key to pressed keys set
                self.pressed_keys.add(key_name)
                
                # Start continuous control thread if needed
                control_keys = {'w', 's', 'a', 'd', 'q', 'e', 'up', 'down'}
                if (any(k in self.pressed_keys for k in control_keys) and 
                    not self.continuous_control_active):
                    self.continuous_control_active = True
                    self.continuous_control_thread = threading.Thread(target=self.continuous_control_loop)
                    self.continuous_control_thread.daemon = True
                    self.continuous_control_thread.start()
                    self._log_message("Starting continuous control")
                
                # Process special commands immediately
                if key_name == 'm':
                    self._log_message("Switching control mode")
                    self.switch_control_mode()
                elif key_name == 't':
                    self._log_message("Taking off")
                    self.take_off()
                    # Set mode to position for takeoff
                    if self.control_mode != ControlMode.POSITION:
                        self.control_mode = ControlMode.POSITION
                        self._log_message(f"Switched to {self.control_mode} control mode for takeoff")
                elif key_name == 'l':
                    self._log_message("Landing")
                    self.land()
                    # Set mode to position for landing
                    if self.control_mode != ControlMode.POSITION:
                        self.control_mode = ControlMode.POSITION
                        self._log_message(f"Switched to {self.control_mode} control mode for landing")
                elif key_name == 'r':
                    self._log_message("Resetting simulation")
                    self.reset_simulation()
                    # Reset to default attitude control mode
                    self.control_mode = ControlMode.ATTITUDE
                    self._log_message(f"Reset to {self.control_mode} control mode")
                elif key_name == 'esc':
                    self._log_message("Exiting")
                    self.running = False
                    # Stop listener
                    return False
                
                # Update message in fixed message area with key pressed
                self._log_message(f"Key pressed: {key_name}")
                
                # Continue listening for events
                return True
            
            def on_release(key):
                try:
                    key_name = key.char.lower()
                except AttributeError:
                    key_name = str(key).replace('Key.', '')
                
                logger.debug(f"Key released: {key_name}")
                
                # Remove key from pressed keys set
                if key_name in self.pressed_keys:
                    self.pressed_keys.remove(key_name)
                    
                    # Check if we need to stop the continuous control thread
                    control_keys = {'w', 's', 'a', 'd', 'q', 'e', 'up', 'down'}
                    if key_name in control_keys and not any(k in self.pressed_keys for k in control_keys):
                        self.continuous_control_active = False
                        # Wait for thread to terminate
                        if self.continuous_control_thread and self.continuous_control_thread.is_alive():
                            self.continuous_control_thread.join(0.2)
                        self.continuous_control_thread = None
                        logger.debug("Stopping continuous control thread")
                    
                    # Handle control release based on mode
                    if self.control_mode == ControlMode.ATTITUDE:
                        # Reset specific controls when keys are released (except thrust)
                        if key_name == 'w' or key_name == 's':
                            self._log_message("Resetting pitch to 0")
                            self.send_attitude_cmd(pitch=0.0)
                        elif key_name == 'a' or key_name == 'd':
                            self._log_message("Resetting roll to 0")
                            self.send_attitude_cmd(roll=0.0)
                        elif key_name == 'q' or key_name == 'e':
                            self._log_message("Resetting yaw_rate to 0")
                            self.send_attitude_cmd(yaw_rate=0.0)
                    elif self.control_mode == ControlMode.VELOCITY:
                        # Stop movement in the released direction
                        if key_name in {'w', 's'}:
                            self.velocity_setpoint["x"] = 0.0
                            self._log_message("Stopping forward/backward velocity")
                            self.send_velocity_cmd(
                                0.0, 
                                self.velocity_setpoint["y"], 
                                self.velocity_setpoint["z"]
                            )
                        elif key_name in {'a', 'd'}:
                            self.velocity_setpoint["y"] = 0.0
                            self._log_message("Stopping left/right velocity")
                            self.send_velocity_cmd(
                                self.velocity_setpoint["x"], 
                                0.0, 
                                self.velocity_setpoint["z"]
                            )
                        elif key_name in {'up', 'down'}:
                            self.velocity_setpoint["z"] = 0.0
                            self._log_message("Stopping vertical velocity")
                            self.send_velocity_cmd(
                                self.velocity_setpoint["x"], 
                                self.velocity_setpoint["y"], 
                                0.0
                            )
                
                # Continue listening for events
                return True
            
            # Setup the listener
            with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                logger.info("Keyboard listener started")
                
                # Keep thread alive until self.running is False
                while self.running and listener.running:
                    time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error in keyboard control loop: {e}", exc_info=True)
    
    def run(self):
        """Run the client."""
        try:
            # Start keyboard control thread
            keyboard_thread = threading.Thread(target=self.keyboard_control_loop)
            keyboard_thread.daemon = True
            keyboard_thread.start()
            
            # Start state display thread
            display_thread = threading.Thread(target=self.display_state_loop)
            display_thread.daemon = True
            display_thread.start()
            
            # Keep main thread alive
            while self.running:
                time.sleep(0.1)
            
            logger.info("Client stopped")
            
        except KeyboardInterrupt:
            logger.info("Client interrupted by user")
            self.running = False
        except Exception as e:
            logger.error(f"Error running client: {e}")
        finally:
            # Sleep briefly to allow threads to terminate gracefully
            time.sleep(0.5)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Crazyflie Client")
    
    # Server settings
    parser.add_argument("--host", type=str, default=SERVER.get("host"),
                       help=f"Server host (default: {SERVER.get('host')})")
    parser.add_argument("--port", type=int, default=SERVER.get("port"),
                       help=f"Server port (default: {SERVER.get('port')})")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()
    
    logger.info(f"Starting Crazyflie client for server at {args.host}:{args.port}")
    
    try:
        # Initialize client
        client = CrazyflieClient(args.host, args.port)
        
        # Run the client
        client.run()
        
    except Exception as e:
        logger.error(f"Error running client: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 