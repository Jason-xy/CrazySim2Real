"""
Advanced metrics calculation for analyzing Crazyflie test results.

This module provides methods to calculate detailed metrics for different test types
beyond what the basic data_processor provides.
"""
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """
    Calculate advanced performance metrics for drone tests.
    
    This class extends the basic metrics from DataProcessor with more detailed
    performance characteristics for each test type.
    """
    
    def __init__(self):
        """Initialize the performance metrics calculator."""
        # Common metrics for all test types
        self.common_metrics = [
            'energy_consumption', 
            'control_effort', 
            'trajectory_error',
            'stability_score'
        ]
        
        # Test-specific metrics
        self.step_metrics = [
            'rise_time', 
            'settling_time', 
            'overshoot', 
            'steady_state_error',
            'response_time',
            'damping_ratio'
        ]
        
        self.impulse_metrics = [
            'recovery_time',
            'peak_value',
            'decay_time',
            'oscillation_frequency',
            'oscillation_count'
        ]
        
        self.sine_sweep_metrics = [
            'resonance_frequency',
            'bandwidth',
            'phase_margin',
            'amplitude_ratio',
            'transfer_function'
        ]
        
        self.hover_metrics = [
            'position_variance',
            'attitude_variance',
            'power_efficiency',
            'drift_rate'
        ]

    def compute_metrics(self, 
                        test_type: str, 
                        channel: str, 
                        data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute all applicable metrics for a specific test type and channel.
        
        Args:
            test_type: Type of test ('step', 'impulse', 'sine_sweep', 'hover')
            channel: Control channel ('roll', 'pitch', 'yaw', 'thrust', 'height')
            data: Processed data containing time series and relevant variables
            
        Returns:
            Dictionary of computed metrics
        """
        metrics_result = {}
        
        # Compute common metrics for all test types
        self._compute_common_metrics(metrics_result, test_type, channel, data)
        
        # Compute test-specific metrics
        if test_type == 'step':
            self._compute_step_metrics(metrics_result, channel, data)
        elif test_type == 'impulse':
            self._compute_impulse_metrics(metrics_result, channel, data)
        elif test_type == 'sine_sweep':
            self._compute_sine_sweep_metrics(metrics_result, channel, data)
        elif test_type == 'hover':
            self._compute_hover_metrics(metrics_result, channel, data)
        
        return metrics_result
    
    def _compute_common_metrics(self, 
                               metrics: Dict[str, Any], 
                               test_type: str, 
                               channel: str, 
                               data: Dict[str, Any]):
        """Compute metrics common to all test types."""
        times = np.array(data.get('times', []))
        
        if len(times) < 2:
            logger.warning(f"Insufficient time data for calculating common metrics")
            return
        
        # Calculate energy consumption (approximation based on thrust command)
        if 'command_thrust' in data and len(data['command_thrust']) >= 2:
            thrust_cmd = np.array(data['command_thrust'])
            thrust_energy = np.trapz(thrust_cmd**2, times)
            metrics['energy_consumption'] = float(thrust_energy)
        
        # Calculate control effort (sum of absolute control input changes)
        control_signals = {
            'roll': 'command_roll',
            'pitch': 'command_pitch',
            'yaw': 'command_yaw_rate',
            'thrust': 'command_thrust'
        }
        
        total_control_effort = 0.0
        for ctrl_name, data_key in control_signals.items():
            if data_key in data and len(data[data_key]) >= 2:
                ctrl_signal = np.array(data[data_key])
                # Calculate sum of absolute differences (total control movement)
                ctrl_changes = np.abs(np.diff(ctrl_signal))
                total_control_effort += np.sum(ctrl_changes)
        
        metrics['control_effort'] = float(total_control_effort)
        
        # Calculate trajectory error based on channel
        if channel in data and f'command_{channel}' in data:
            response = np.array(data[channel])
            command = np.array(data[f'command_{channel}'])
            
            # Make arrays the same length if needed
            min_len = min(len(response), len(command))
            if min_len >= 2:
                response = response[:min_len]
                command = command[:min_len]
                
                # Calculate RMS error
                error = response - command
                rms_error = np.sqrt(np.mean(error**2))
                metrics['trajectory_error'] = float(rms_error)
        
        # Calculate stability score (inverse of variance in steady state)
        if channel in data and len(data[channel]) >= 10:
            # Use last 30% of the data to estimate stability
            response = np.array(data[channel])
            steady_state_idx = int(0.7 * len(response))
            steady_state_data = response[steady_state_idx:]
            
            if len(steady_state_data) >= 5:
                variance = np.var(steady_state_data)
                # Convert to a 0-100 score with higher being better
                stability_score = 100.0 * np.exp(-variance)
                metrics['stability_score'] = float(stability_score)
    
    def _compute_step_metrics(self, 
                             metrics: Dict[str, Any], 
                             channel: str, 
                             data: Dict[str, Any]):
        """Compute metrics specific to step response tests."""
        times = np.array(data.get('times', []))
        
        if channel not in data or len(data[channel]) < 10:
            logger.warning(f"Insufficient data for channel {channel} in step test")
            return
        
        response = np.array(data[channel])
        
        # Most metrics are already computed by the DataProcessor
        # Let's add a few more advanced ones
        
        # Calculate steady state error
        if f'command_{channel}' in data:
            command = np.array(data[f'command_{channel}'])
            # Use the last 20% of the test data to calculate steady state error
            steady_state_idx = int(0.8 * len(response))
            
            if steady_state_idx < len(response) and steady_state_idx < len(command):
                steady_state_response = np.mean(response[steady_state_idx:])
                steady_state_command = np.mean(command[steady_state_idx:])
                steady_state_error = abs(steady_state_response - steady_state_command)
                metrics['steady_state_error'] = float(steady_state_error)
        
        # Estimate damping ratio from overshoot
        if 'overshoot' in data and data['overshoot'] is not None:
            overshoot_percent = data['overshoot']
            if overshoot_percent > 0:
                # Calculate damping ratio using logarithmic decrement approximation
                damping_ratio = -np.log(overshoot_percent / 100.0) / np.sqrt(np.pi**2 + np.log(overshoot_percent / 100.0)**2)
                metrics['damping_ratio'] = float(damping_ratio)
            else:
                # If no overshoot, system is critically damped or overdamped
                metrics['damping_ratio'] = 1.0
    
    def _compute_impulse_metrics(self, 
                                metrics: Dict[str, Any], 
                                channel: str, 
                                data: Dict[str, Any]):
        """Compute metrics specific to impulse response tests."""
        times = np.array(data.get('times', []))
        
        if channel not in data or len(data[channel]) < 10:
            logger.warning(f"Insufficient data for channel {channel} in impulse test")
            return
        
        response = np.array(data[channel])
        
        # Calculate oscillation count and frequency
        # First find zero crossings
        if len(response) >= 3:
            # Get mean value for baseline (should be close to zero for impulse response)
            baseline = np.mean(response[-int(len(response)*0.2):])  # Use last 20% for baseline
            centered_response = response - baseline
            
            # Find zero crossings
            zero_crossings = np.where(np.diff(np.signbit(centered_response)))[0]
            
            if len(zero_crossings) >= 2:
                metrics['oscillation_count'] = len(zero_crossings) // 2  # Divide by 2 to count complete oscillations
                
                # Calculate oscillation frequency
                if len(zero_crossings) >= 4 and times[zero_crossings[-1]] > times[zero_crossings[0]]:
                    # Calculate average period from multiple oscillations
                    period = (times[zero_crossings[-1]] - times[zero_crossings[0]]) / (len(zero_crossings) - 1) * 2
                    metrics['oscillation_frequency'] = float(1.0 / period if period > 0 else 0.0)
        
        # Calculate decay time (time to reach 10% of peak value)
        if 'peak_value' in data and data['peak_value'] is not None:
            peak_value = data['peak_value']
            peak_idx = np.argmax(np.abs(response))
            
            if peak_idx < len(response) - 1:
                # Look for first time response drops below 10% of peak after the peak
                threshold = 0.1 * abs(peak_value)
                decay_indices = np.where(np.abs(response[peak_idx:]) < threshold)[0]
                
                if len(decay_indices) > 0:
                    decay_idx = decay_indices[0] + peak_idx
                    metrics['decay_time'] = float(times[decay_idx] - times[peak_idx])
    
    def _compute_sine_sweep_metrics(self, 
                                   metrics: Dict[str, Any], 
                                   channel: str, 
                                   data: Dict[str, Any]):
        """Compute metrics specific to sine sweep frequency response tests."""
        times = np.array(data.get('times', []))
        
        if channel not in data or len(data[channel]) < 10:
            logger.warning(f"Insufficient data for channel {channel} in sine sweep test")
            return
        
        # These metrics require frequency domain analysis
        # For now, we'll include placeholder calculations
        # A full implementation would use FFT and transfer function estimation
        
        response = np.array(data[channel])
        
        # Find resonance frequency (frequency with maximum amplitude)
        # This is a simplified approach - real implementation would use FFT
        if f'command_{channel}' in data:
            command = np.array(data[f'command_{channel}'])
            
            # Calculate amplitude ratio at each time point
            ratio = []
            for i in range(len(response)):
                if i < len(command) and abs(command[i]) > 1e-6:
                    ratio.append(abs(response[i] / command[i]))
            
            if ratio:
                max_ratio_idx = np.argmax(ratio)
                if max_ratio_idx < len(times):
                    # In a real sine sweep, we'd have the frequency at each time point
                    # For now, we're setting a placeholder
                    metrics['resonance_frequency'] = f"At time={times[max_ratio_idx]:.2f}s"
                    metrics['amplitude_ratio'] = float(ratio[max_ratio_idx])
        
        # Add placeholder for bandwidth calculation
        metrics['bandwidth'] = "Requires frequency domain analysis"
        metrics['phase_margin'] = "Requires frequency domain analysis"
    
    def _compute_hover_metrics(self, 
                              metrics: Dict[str, Any], 
                              channel: str, 
                              data: Dict[str, Any]):
        """Compute metrics specific to hover performance."""
        times = np.array(data.get('times', []))
        
        # Calculate position variance
        if 'height' in data and len(data['height']) > 10:
            height = np.array(data['height'])
            metrics['position_variance'] = float(np.var(height))
        
        # Calculate attitude variance
        attitude_vars = {}
        for attitude in ['roll', 'pitch']:
            if attitude in data and len(data[attitude]) > 10:
                attitude_data = np.array(data[attitude])
                attitude_vars[attitude] = float(np.var(attitude_data))
        
        if attitude_vars:
            metrics['attitude_variance'] = attitude_vars
        
        # Calculate drift rate (change in position over time)
        if 'height' in data and len(data['height']) > 10 and len(times) > 10:
            height = np.array(data['height'])
            # Use linear regression to find drift rate
            try:
                slope, _ = np.polyfit(times, height, 1)
                metrics['drift_rate'] = float(slope)  # Units: position units / second
            except Exception as e:
                logger.warning(f"Could not calculate drift rate: {e}")
        
        # Calculate power efficiency (height maintained per unit of thrust)
        if 'height' in data and 'command_thrust' in data:
            if len(data['height']) > 10 and len(data['command_thrust']) > 10:
                height = np.array(data['height'])
                thrust = np.array(data['command_thrust'])
                
                # Make arrays the same length
                min_len = min(len(height), len(thrust))
                height = height[:min_len]
                thrust = thrust[:min_len]
                
                # Calculate average height / average thrust
                if np.mean(thrust) > 0:
                    metrics['power_efficiency'] = float(np.mean(height) / np.mean(thrust))


def calculate_test_metrics(processed_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate detailed metrics for all tests in processed data.
    
    Args:
        processed_data: Dictionary of processed test data
        
    Returns:
        Dictionary of calculated metrics by test name
    """
    metrics_calculator = PerformanceMetrics()
    metrics_results = {}
    
    for test_name, test_data in processed_data.items():
        test_type = test_data.get('test_type', 'unknown')
        channel = test_data.get('channel', 'unknown')
        
        # Skip tests with unknown type or channel
        if test_type == 'unknown' or channel == 'unknown':
            logger.warning(f"Skipping metrics calculation for test {test_name}: Unknown test type or channel")
            continue
        
        # Calculate metrics
        try:
            metrics = metrics_calculator.compute_metrics(test_type, channel, test_data)
            metrics_results[test_name] = metrics
        except Exception as e:
            logger.error(f"Error calculating metrics for test {test_name}: {e}")
            metrics_results[test_name] = {'error': str(e)}
    
    return metrics_results 