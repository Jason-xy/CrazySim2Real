"""
Plotter for the Crazyflie Sweeper package.

Generates and saves matplotlib figures and text reports based on flight data,
including time-domain responses, hover analysis, and test summaries.
"""
import csv
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Callable

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .config import FlightConfig
from .utils import make_output_dir, current_timestamp

logger = logging.getLogger(__name__)


class Plotter:
    """
    Generates and saves plots and reports from flight data.
    
    Creates visualizations of flight data, including time-domain responses,
    hover analysis, and test summaries.
    """
    
    def __init__(self, config: FlightConfig):
        """
        Initialize the plotter.
        
        Args:
            config: Configuration parameters
        """
        self.config = config
        self.output_dir = None
        
    def setup_output_directory(self, base_name: str = "logs") -> str:
        """
        Set up the output directory for plots and reports.
        
        Args:
            base_name: Base name for the directory
            
        Returns:
            Path to the created directory
        """
        # Ensure base directory exists
        base_dir = os.path.join(os.getcwd(), base_name)
        os.makedirs(base_dir, exist_ok=True)
        
        # Create a timestamped directory for this run
        timestamp = current_timestamp()
        run_dir = os.path.join(base_dir, timestamp)
        os.makedirs(run_dir, exist_ok=True)
        
        self.output_dir = run_dir
        logger.info(f"Output directory created: {self.output_dir}")
        return self.output_dir
        
    def generate_summary(self, results: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate a text summary of the test results.
        
        Args:
            results: Dictionary of processed metrics
            
        Returns:
            Path to the generated summary file
        """
        if not self.output_dir:
            self.setup_output_directory()
            
        summary_path = os.path.join(self.output_dir, "summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("==== Crazyflie Sweep Test Results ====\n\n")
            f.write(f"Test performed on: {current_timestamp().replace('_', ' ')}\n")
            f.write(f"Hover thrust: {self.config.hover_thrust}\n")
            f.write(f"Roll/Pitch test values: {self.config.test_roll_angle_deg}/{self.config.test_pitch_angle_deg} degrees\n")
            f.write(f"Thrust increment test value: {self.config.test_thrust_increment}\n\n")
            
            f.write("== Performance Metrics ==\n\n")
            
            for maneuver, data in results.items():
                f.write(f"Maneuver: {maneuver}\n")
                
                # Write basic metrics for all tests
                for metric in ['max_roll', 'max_pitch', 'max_gyro_x', 'max_gyro_y', 'max_gyro_z']:
                    if metric in data:
                        f.write(f"  {metric.replace('_', ' ').title()}: {data[metric]:.2f}\n")
                
                # Write specific metrics based on test type
                test_type = data.get('test_type', '')
                
                if test_type == 'step':
                    for metric, format_str in [
                        ('rise_time', '  Rise time: {:.3f} seconds\n'),
                        ('settling_time', '  Settling time: {:.3f} seconds\n'),
                        ('overshoot', '  Overshoot: {:.2f}%\n')
                    ]:
                        if metric in data and data[metric] is not None:
                            f.write(format_str.format(data[metric]))
                
                elif test_type == 'impulse':
                    for metric, format_str in [
                        ('recovery_time', '  Recovery time: {:.3f} seconds\n'),
                        ('peak_value', '  Peak value: {:.2f}\n')
                    ]:
                        if metric in data and data[metric] is not None:
                            f.write(format_str.format(data[metric]))
                
                elif test_type == 'sine_sweep':
                    if 'max_amplitude' in data:
                        f.write(f"  Maximum amplitude: {data['max_amplitude']:.2f}\n")
                    if 'frequency_range' in data:
                        f.write(f"  Frequency range: {data['frequency_range']}\n")
                
                f.write("\n")
                
        logger.info(f"Summary file generated: {summary_path}")
        return summary_path
                
    def generate_plots(self, results: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Generate plots for each test/maneuver.
        
        Args:
            results: Dictionary of processed metrics
            
        Returns:
            List of paths to generated plot files
        """
        if not self.output_dir:
            self.setup_output_directory()
            
        plot_paths = []
        
        # Generate individual plots for each maneuver
        for maneuver, data in results.items():
            if 'times' not in data or not data['times']:
                logger.warning(f"No time data for maneuver: {maneuver}, skipping plot")
                continue

            # Select the appropriate plot type based on test_type
            test_type = data.get('test_type', '')
            plot_path = self._create_plot(maneuver, data, test_type)
            if plot_path:
                plot_paths.append(plot_path)
                
        return plot_paths
    
    def _create_plot(self, maneuver: str, data: Dict[str, Any], test_type: str) -> Optional[str]:
        """
        Create a plot based on the test type with proper error handling.
        
        Args:
            maneuver: Name of the maneuver
            data: Processed data for the maneuver
            test_type: Type of test ('step', 'impulse', 'sine_sweep', or other)
            
        Returns:
            Path to the generated plot file, or None if generation failed
        """
        try:
            # Create generic figure
            plt.figure(figsize=(12, 10))
            
            # Identify channel-specific data
            channel = data.get('channel', 'unknown')
            times = data['times']
            
            # Common plot data
            self._add_attitude_plot(plt.subplot(3, 1, 1), times, data, maneuver, test_type)
            self._add_gyro_plot(plt.subplot(3, 1, 2), times, data)
            
            # Plot specific to test type
            ax3 = plt.subplot(3, 1, 3)
            
            if test_type == 'step':
                self._add_step_response_plot(ax3, times, data, channel)
            elif test_type == 'impulse':
                self._add_impulse_response_plot(ax3, times, data, channel)
            elif test_type == 'sine_sweep':
                self._add_sine_sweep_plot(ax3, times, data, channel)
            else:
                # If we only have two plots for general case, adjust layout
                plt.figure(figsize=(12, 8))
                self._add_attitude_plot(plt.subplot(2, 1, 1), times, data, maneuver, "General")
                self._add_gyro_plot(plt.subplot(2, 1, 2), times, data)
            
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(self.output_dir, f"{maneuver.replace(' ', '_')}.png")
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"{test_type.capitalize() if test_type else 'General'} plot generated: {plot_path}")
            return plot_path
            
        except Exception as e:
            logger.error(f"Error generating plot for {maneuver}: {e}")
            plt.close()
            return None
    
    def _add_attitude_plot(self, ax, times, data, maneuver, plot_type):
        """Add attitude (roll/pitch) plot to the given axis."""
        # Plot response signals (solid lines)
        ax.plot(times, data['roll'], 'r-', label='Roll')
        ax.plot(times, data['pitch'], 'b-', label='Pitch')
        
        # Add command signals (dashed lines) if available
        if 'command_roll' in data:
            ax.plot(times, data['command_roll'], 'r--', label='Command Roll')
        if 'command_pitch' in data:
            ax.plot(times, data['command_pitch'], 'b--', label='Command Pitch')
            
        ax.grid(True)
        ax.set_title(f"{plot_type} Response: {maneuver}" if plot_type else f"Maneuver: {maneuver}")
        ax.set_ylabel('Angle (degrees)')
        ax.legend()
    
    def _add_gyro_plot(self, ax, times, data):
        """Add gyroscope data plot to the given axis."""
        ax.plot(times, data['gyro_x'], 'r-', label='Gyro X')
        ax.plot(times, data['gyro_y'], 'g-', label='Gyro Y')
        ax.plot(times, data['gyro_z'], 'b-', label='Gyro Z')
        
        # Add yaw rate command if available (typically matches with gyro z data)
        if 'command_yaw_rate' in data:
            ax.plot(times, data['command_yaw_rate'], 'b--', label='Command Yaw Rate')
            
        ax.grid(True)
        ax.set_ylabel('Angular velocity (degrees/s)')
        ax.legend()
    
    def _add_step_response_plot(self, ax, times, data, channel):
        """Add step response specifics to the given axis."""
        # Get channel data
        channel_data = data.get(channel, data.get('roll' if channel == 'roll' else 
                                      'pitch' if channel == 'pitch' else 'gyro_z'))
        
        # Plot channel data as solid line
        line_color = 'k'  # Default black
        if channel == 'roll':
            line_color = 'r'
        elif channel == 'pitch':
            line_color = 'b'
        elif channel == 'thrust':
            line_color = 'g'
            
        ax.plot(times, channel_data, f'{line_color}-', linewidth=2, label=f'{channel.capitalize()}')
        
        # Add command signal as dashed line with the same color
        command_key = None
        if channel == 'roll':
            command_key = 'command_roll'
        elif channel == 'pitch':
            command_key = 'command_pitch'
        elif channel == 'thrust':
            command_key = 'command_thrust'
            
        if command_key and command_key in data:
            ax.plot(times, data[command_key], f'{line_color}--', linewidth=1.5, 
                   label=f'Command {channel.capitalize()}')
        
        # Add metrics
        for metric, color, label_fmt in [
            ('rise_time', 'g', 'Rise time: {:.3f}s'),
            ('settling_time', 'b', 'Settling time: {:.3f}s')
        ]:
            if metric in data and data[metric] is not None:
                metric_value = data[metric]
                metric_idx = min(range(len(times)), key=lambda i: abs(times[i] - metric_value))
                ax.axvline(x=metric_value, color=color, linestyle='--', 
                         label=label_fmt.format(metric_value))
                ax.plot(metric_value, channel_data[metric_idx], f'{color}o', markersize=6)
        
        # Add overshoot annotation
        if 'overshoot' in data and data['overshoot'] > 0:
            ax.annotate(f'Overshoot: {data["overshoot"]:.2f}%', 
                       xy=(0.7, 0.9), xycoords='axes fraction',
                       bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
        
        ax.grid(True)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel(f'{channel.capitalize()} response')
        ax.legend()
    
    def _add_impulse_response_plot(self, ax, times, data, channel):
        """Add impulse response specifics to the given axis."""
        # Get channel data
        channel_data = data.get(channel, data.get('roll' if channel == 'roll' else 
                                      'pitch' if channel == 'pitch' else 'gyro_z'))
        
        # Plot channel data as solid line
        line_color = 'k'  # Default black
        if channel == 'roll':
            line_color = 'r'
        elif channel == 'pitch':
            line_color = 'b'
        elif channel == 'thrust':
            line_color = 'g'
            
        ax.plot(times, channel_data, f'{line_color}-', linewidth=2, label=f'{channel.capitalize()}')
        
        # Add command signal as dashed line with the same color
        command_key = None
        if channel == 'roll':
            command_key = 'command_roll'
        elif channel == 'pitch':
            command_key = 'command_pitch'
        elif channel == 'thrust':
            command_key = 'command_thrust'
            
        if command_key and command_key in data:
            ax.plot(times, data[command_key], f'{line_color}--', linewidth=1.5, 
                   label=f'Command {channel.capitalize()}')
        
        # Find and mark peak
        if len(channel_data) > 0:
            peak_idx = max(range(len(channel_data)), key=lambda i: abs(channel_data[i]))
            ax.plot(times[peak_idx], channel_data[peak_idx], 'ro', markersize=8, 
                  label=f'Peak: {channel_data[peak_idx]:.2f}')
        
        # Add recovery time marker
        if 'recovery_time' in data and data['recovery_time'] is not None:
            recovery_time = data['recovery_time']
            recovery_idx = min(range(len(times)), key=lambda i: abs(times[i] - recovery_time))
            ax.axvline(x=recovery_time, color='g', linestyle='--', 
                     label=f'Recovery time: {recovery_time:.3f}s')
            ax.plot(recovery_time, channel_data[recovery_idx], 'go', markersize=6)
        
        ax.grid(True)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel(f'{channel.capitalize()} response')
        ax.legend()
    
    def _add_sine_sweep_plot(self, ax, times, data, channel):
        """Add sine sweep specifics to the given axis."""
        # Get channel data
        channel_data = data.get(channel, data.get('roll' if channel == 'roll' else 
                                      'pitch' if channel == 'pitch' else 'gyro_z'))
        
        # Plot channel data as solid line
        line_color = 'k'  # Default black
        if channel == 'roll':
            line_color = 'r'
        elif channel == 'pitch':
            line_color = 'b'
        elif channel == 'thrust':
            line_color = 'g'
            
        ax.plot(times, channel_data, f'{line_color}-', linewidth=2, label=f'{channel.capitalize()}')
        
        # Add command signal as dashed line with the same color
        command_key = None
        if channel == 'roll':
            command_key = 'command_roll'
        elif channel == 'pitch':
            command_key = 'command_pitch'
        elif channel == 'thrust':
            command_key = 'command_thrust'
            
        if command_key and command_key in data:
            ax.plot(times, data[command_key], f'{line_color}--', linewidth=1.5, 
                   label=f'Command {channel.capitalize()}')
        
        # Add annotations
        for i, (key, fmt) in enumerate([
            ('frequency_range', 'Frequency range: {}'),
            ('max_amplitude', 'Max amplitude: {:.2f}')
        ]):
            if key in data:
                ax.annotate(fmt.format(data[key]), 
                           xy=(0.7, 0.9-0.1*i), xycoords='axes fraction',
                           bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
        
        ax.grid(True)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel(f'{channel.capitalize()} response')
        ax.legend()
        
    def generate_hover_analysis_plot(self, hover_data: List[Dict[str, Any]]) -> Optional[str]:
        """
        Generate a hover analysis plot from hover estimation data.
        
        Args:
            hover_data: List of hover data points
            
        Returns:
            Path to the generated plot file, or None if generation failed
        """
        if not hover_data or len(hover_data) < 10:
            logger.warning("Insufficient hover data for analysis plot")
            return None
            
        if not self.output_dir:
            self.setup_output_directory()
            
        try:
            plt.figure(figsize=(10, 8))
            
            # Plot 1: Thrust vs Z-Velocity scatter plot
            ax1 = plt.subplot(2, 1, 1)
            vz_values = [d['vz'] for d in hover_data]
            thrust_values = [d['thrust'] for d in hover_data]
            
            ax1.scatter(vz_values, thrust_values, alpha=0.5)
            
            # Add regression line if possible
            if len(vz_values) >= 2:
                # Calculate regression
                z = np.polyfit(vz_values, thrust_values, 1)
                p = np.poly1d(z)
                
                # Add the regression line
                vz_line = np.linspace(min(vz_values), max(vz_values), 100)
                ax1.plot(vz_line, p(vz_line), "r--", 
                       label=f'Fit: thrust = {z[0]:.1f}*vz + {z[1]:.1f}')
                
                # Highlight the zero velocity thrust (hover thrust)
                ax1.axhline(y=z[1], color='g', linestyle='--', 
                          label=f'Est. hover thrust: {z[1]:.0f}')
            
            ax1.set_xlabel('Z-Velocity (m/s)')
            ax1.set_ylabel('Thrust')
            ax1.set_title('Thrust vs Z-Velocity During Hover')
            ax1.grid(True)
            ax1.legend()
            
            # Plot 2: Height and Z-Velocity over time
            ax2 = plt.subplot(2, 1, 2)
            time_points = range(len(hover_data))
            ax2.plot(time_points, [d['z'] for d in hover_data], 'b-', label='Height (m)')
            ax2.plot(time_points, [d['vz'] for d in hover_data], 'g-', label='Z-Velocity (m/s)')
            
            ax2.set_xlabel('Sample Number')
            ax2.set_ylabel('Value')
            ax2.set_title('Height and Z-Velocity During Hover')
            ax2.grid(True)
            ax2.legend()
            
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(self.output_dir, "hover_thrust_analysis.png")
            plt.savefig(plot_path)
            plt.close()
            
            # Save the hover data as CSV for further analysis
            csv_path = os.path.join(self.output_dir, "hover_thrust_data.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['z_velocity', 'thrust', 'height'])
                for point in hover_data:
                    writer.writerow([point['vz'], point['thrust'], point['z']])
            
            logger.info(f"Hover analysis plot generated: {plot_path}")
            logger.info(f"Hover data saved to CSV: {csv_path}")
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error generating hover analysis plot: {e}")
            plt.close()
            return None
            
    def export_data_for_simulation(self, results: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """
        Export processed data for comparison with simulation.
        
        Args:
            results: Dictionary of processed metrics
            
        Returns:
            Path to the export directory, or None if export failed
        """
        if not self.output_dir:
            self.setup_output_directory()
            
        export_dir = os.path.join(self.output_dir, "sim_comparison_data")
        os.makedirs(export_dir, exist_ok=True)
        
        try:
            # Export each test as a separate CSV file
            for maneuver, data in results.items():
                if 'times' not in data or not data['times']:
                    continue
                    
                # Prepare data for CSV
                headers = ['time', 'roll', 'pitch', 'gyro_x', 'gyro_y', 'gyro_z']
                rows = []
                
                for i in range(len(data['times'])):
                    row = [data['times'][i]]
                    for field in headers[1:]:  # Skip 'time' which we already added
                        if field in data and i < len(data[field]):
                            row.append(data[field][i])
                        else:
                            row.append(0.0)
                    rows.append(row)
                
                # Write CSV file
                csv_path = os.path.join(export_dir, f"{maneuver.replace(' ', '_')}.csv")
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                    writer.writerows(rows)
                
                # Also save test parameters as JSON
                params = {k: v for k, v in data.items() 
                        if k not in ('times', 'roll', 'pitch', 'gyro_x', 'gyro_y', 'gyro_z')}
                params['test_name'] = maneuver
                
                json_path = os.path.join(export_dir, f"{maneuver.replace(' ', '_')}_params.json")
                with open(json_path, 'w') as f:
                    json.dump(params, f, indent=2)
            
            # Save config parameters
            config_path = os.path.join(export_dir, "config.json")
            with open(config_path, 'w') as f:
                # Convert dataclass to dict, excluding complex objects
                config_dict = {
                    k: v for k, v in vars(self.config).items() 
                    if isinstance(v, (int, float, str, bool, list, dict))
                }
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Data exported for simulation comparison: {export_dir}")
            return export_dir
            
        except Exception as e:
            logger.error(f"Error exporting data for simulation: {e}")
            return None 