"""
Data processor for the Crazyflie Sweeper package.

Processes raw log data to extract flight metrics, including rise time, settling time, and overshoot calculations.
"""
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .config import FlightConfig

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Processes raw log data to extract flight metrics.
    
    Computes time-domain metrics such as rise time, settling time, and overshoot,
    as well as basic frequency-domain placeholders for future analysis.
    """
    MIN_DATA_POINTS = 5

    def __init__(self, config: FlightConfig):
        """
        Initialize the data processor.
        
        Args:
            config: Configuration parameters
        """
        self.config = config
        self.vars_to_extract = {
            'roll': 'stabilizer.roll',
            'pitch': 'stabilizer.pitch',
            'gyro_x': 'gyro.x',
            'gyro_y': 'gyro.y',
            'gyro_z': 'gyro.z',  # Often used for yaw rate
            'thrust': 'stabilizer.thrust',  # Commanded thrust or similar
            'height': ['stateEstimate.z', 'kalman.stateZ'],
            'command_roll': 'command.roll',
            'command_pitch': 'command.pitch',
            'command_yaw_rate': 'command.yaw_rate',
            'command_thrust': 'command.thrust'
        }
        self.primary_var_candidates = {
            'roll': ['stabilizer.roll'],
            'pitch': ['stabilizer.pitch'],
            'thrust': ['stabilizer.thrust'],
            'height': ['stateEstimate.z', 'kalman.stateZ'],
            'yaw': ['gyro.z'] # Yaw analysis often uses gyro.z
        }
        self.fallback_primary_vars = ['stabilizer.roll', 'stabilizer.pitch', 'stateEstimate.z', 'kalman.stateZ', 'gyro.z']

    def process(self, raw_log_data: Dict[str, Dict[str, List[Tuple[int, Any]]]]) -> Dict[str, Dict[str, Any]]:
        """
        Process raw log data to extract metrics for each valid maneuver.
        
        Args:
            raw_log_data: Raw log data organized by maneuver/test. 
                          Each variable within a maneuver has a list of (timestamp, value) tuples.
            
        Returns:
            Dictionary of processed metrics organized by maneuver/test
        """
        if not raw_log_data:
            logger.warning("No data provided to process.")
            return {}
        
        results = {}
        for maneuver, maneuver_data in raw_log_data.items():
            if self._should_skip_maneuver(maneuver):
                continue

            try:
                results[maneuver] = self._process_maneuver(maneuver, maneuver_data)
            except Exception as e:
                logger.error(f"Unexpected error processing maneuver {maneuver}: {e}", exc_info=True)
                results[maneuver] = {'status': 'processing_error', 'error': str(e)}

        return results
    
    def _should_skip_maneuver(self, maneuver: str) -> bool:
        """Check if a maneuver should be skipped based on its name."""
        return not maneuver or 'return_to_neutral' in maneuver or maneuver == 'data_processing'

    def _process_maneuver(self, maneuver: str, maneuver_data: Dict[str, List[Tuple[int, Any]]]) -> Dict[str, Any]:
        """Process a single maneuver's data."""
        test_type = self._determine_test_type(maneuver)
        channel = self._determine_channel(maneuver)

        # 1. Find and prepare primary data
        primary_var_name = self._find_primary_variable(maneuver, maneuver_data, channel, test_type)
        if not primary_var_name:
            logger.warning(f"Could not determine a valid primary variable for maneuver: {maneuver}")
            return {'status': 'no_primary_variable', 'test_type': test_type, 'channel': channel}

        primary_data_result = self._prepare_primary_data(maneuver, maneuver_data, primary_var_name)
        if 'error' in primary_data_result:
             return {
                 'status': primary_data_result['status'],
                 'test_type': test_type,
                 'channel': channel,
                 'primary_variable': primary_var_name,
                 'error': primary_data_result['error']
             }

        rel_times_sec_primary = primary_data_result['times']
        start_time_ms = primary_data_result['start_time_ms']

        # 2. Extract and interpolate all relevant data series
        extracted_data = self._extract_and_interpolate_data(
            maneuver, maneuver_data, rel_times_sec_primary, start_time_ms
        )

        # 3. Build initial metrics dictionary
        metrics = self._build_initial_metrics(maneuver, test_type, channel, rel_times_sec_primary, extracted_data)

        # 4. Calculate specific metrics based on test type and channel
        try:
            self._calculate_dynamic_metrics(metrics, test_type, channel, extracted_data)
        except Exception as e:
            logger.warning(f"Error calculating specific metrics for {maneuver}: {e}", exc_info=True)
            metrics['calculation_error'] = str(e)

        return metrics

    def _find_primary_variable(self, maneuver: str, maneuver_data: Dict[str, List[Tuple[int, Any]]], channel: str, test_type: str) -> Optional[str]:
        """Find the best primary variable name for the maneuver."""
        primary_var_name: Optional[str] = None

        # Try channel-specific candidates first
        if channel in self.primary_var_candidates:
            for candidate in self.primary_var_candidates[channel]:
                 if candidate in maneuver_data and len(maneuver_data[candidate]) >= self.MIN_DATA_POINTS:
                     primary_var_name = candidate
                     break

        # If not found, try fallback variables
        if not primary_var_name:
            for fallback_var in self.fallback_primary_vars:
                if fallback_var in maneuver_data and len(maneuver_data[fallback_var]) >= self.MIN_DATA_POINTS:
                    primary_var_name = fallback_var
                    logger.info(f"Maneuver '{maneuver}' (Type: {test_type}, Channel: {channel}) using fallback primary variable: '{primary_var_name}'")
                    break

        return primary_var_name

    def _prepare_primary_data(self, maneuver: str, maneuver_data: Dict[str, List[Tuple[int, Any]]], primary_var_name: str) -> Dict[str, Any]:
        """Prepare primary times and values, checking for validity."""
        primary_data_tuples = maneuver_data[primary_var_name]

        if len(primary_data_tuples) < self.MIN_DATA_POINTS:
            return {
                'status': 'insufficient_data',
                'error': f"Insufficient data points ({len(primary_data_tuples)}) for primary var '{primary_var_name}'"
            }

        try:
            times_ms = np.array([int(t) for t, v in primary_data_tuples])
            values_list = []
            for _, v_raw in primary_data_tuples:
                try:
                    values_list.append(float(v_raw))
                except (ValueError, TypeError):
                    values_list.append(np.nan) # Use NaN for unparseable values

            values = np.array(values_list)
            valid_mask = ~np.isnan(values)

            if np.sum(valid_mask) < self.MIN_DATA_POINTS:
                raise ValueError(f"Not enough valid numeric primary values after conversion ({np.sum(valid_mask)} found).")

            # Interpolate NaN values if necessary (simple linear interpolation)
            if np.any(~valid_mask):
                 values = np.interp(np.arange(len(values)), np.where(valid_mask)[0], values[valid_mask])
                 logger.debug(f"Interpolated NaN values in primary variable '{primary_var_name}' for maneuver '{maneuver}'.")

            start_time_ms = times_ms[0]
            rel_times_sec = (times_ms - start_time_ms) / 1000.0

            return {
                'status': 'ok',
                'times': rel_times_sec.tolist(),
                'values': values.tolist(),
                'start_time_ms': start_time_ms
            }

        except Exception as e:
            logger.error(f"Error processing primary data for {maneuver} ({primary_var_name}): {e}", exc_info=True)
            return {'status': 'primary_data_error', 'error': str(e)}

    def _extract_and_interpolate_data(self,
                                    maneuver: str,
                                    maneuver_data: Dict[str, List[Tuple[int, Any]]],
                                    primary_times_sec: List[float],
                                    primary_start_time_ms: int) -> Dict[str, Any]:
        """Extracts and interpolates all required data series onto the primary timeline."""
        extracted_data: Dict[str, Any] = {}
        primary_times_sec_np = np.array(primary_times_sec)

        for key, source_vars in self.vars_to_extract.items():
            data_tuples: Optional[List[Tuple[int, Any]]] = None
            actual_var_name: Optional[str] = None

            var_names_list = [source_vars] if isinstance(source_vars, str) else source_vars
            for v_name in var_names_list:
                if v_name in maneuver_data: # Check presence first
                    if len(maneuver_data[v_name]) >= 1: # Then check length
                        data_tuples = maneuver_data[v_name]
                        actual_var_name = v_name
                        break

            if not data_tuples:
                logger.debug(f"Data for '{key}' (source vars: {source_vars}) not found or empty in maneuver '{maneuver}'. Filling with zeros.")
                extracted_data[key] = np.zeros_like(primary_times_sec_np).tolist()
                extracted_data[f'{key}_original_rel_times_sec'] = []
                extracted_data[f'{key}_original_values'] = []
                continue

            try:
                current_times_ms_list = []
                current_values_list = []
                for t, v_raw in data_tuples:
                    try:
                        time_val = int(t)
                        current_times_ms_list.append(time_val)
                        # Attempt conversion, use NaN on failure for interpolation handling
                        float_val = float(v_raw)
                        current_values_list.append(float_val)
                    except (ValueError, TypeError):
                         current_times_ms_list.append(int(t)) # Keep time even if value is bad initially
                         current_values_list.append(np.nan) # Mark bad values as NaN
                
                if not current_times_ms_list:
                    raise ValueError("No time data found.")

                current_times_ms_np = np.array(current_times_ms_list)
                current_values_np = np.array(current_values_list)

                # Store original data before interpolation/cleaning
                original_start_time_ms = current_times_ms_np[0] if len(current_times_ms_np) > 0 else 0
                original_rel_times_sec = (current_times_ms_np - original_start_time_ms) / 1000.0
                extracted_data[f'{key}_original_rel_times_sec'] = original_rel_times_sec.tolist()
                # Store original values including NaNs
                original_vals_with_none = [v if not np.isnan(v) else None for v in current_values_np.tolist()]
                extracted_data[f'{key}_original_values'] = original_vals_with_none

                # Prepare for interpolation: handle NaNs and sort
                valid_mask = ~np.isnan(current_values_np)
                num_valid_points = np.sum(valid_mask)

                if num_valid_points < 1: # Need at least one valid point to interpolate
                    logger.warning(f"Not enough valid numeric data points for '{key}' (var: {actual_var_name}) in maneuver '{maneuver}'. Filling with zeros.")
                    interpolated_values = np.zeros_like(primary_times_sec_np)
                elif num_valid_points == 1: # Only one valid point, fill with that value
                     fill_value = current_values_np[valid_mask][0]
                     interpolated_values = np.full_like(primary_times_sec_np, fill_value)
                     logger.debug(f"Only one valid data point for '{key}' (var: {actual_var_name}) in maneuver '{maneuver}'. Filling with constant value {fill_value}.")
                else:
                    # Timestamps for THIS variable, relative to ITS OWN START, for interpolation source (FP)
                    start_time_for_var_rel_timeline = current_times_ms_np[valid_mask][0] if len(current_times_ms_np[valid_mask]) > 0 else 0
                    current_var_times_rel_to_own_start_sec = (current_times_ms_np - start_time_for_var_rel_timeline) / 1000.0
                    
                    valid_own_rel_times = current_var_times_rel_to_own_start_sec[valid_mask]
                    valid_values = current_values_np[valid_mask]

                    # Ensure times are monotonically increasing for interpolation
                    sort_indices = np.argsort(valid_own_rel_times)
                    sorted_valid_own_rel_times = valid_own_rel_times[sort_indices] # FP for interp
                    sorted_valid_values = valid_values[sort_indices]          # F for interp

                    # Handle potential duplicate times after sorting
                    unique_fp_times, unique_indices = np.unique(sorted_valid_own_rel_times, return_index=True)
                    unique_f_values = sorted_valid_values[unique_indices]

                    if len(unique_fp_times) < 2: # Need at least two unique points for interp
                        fill_value = unique_f_values[0] if len(unique_f_values)>0 else 0.0
                        interpolated_values = np.full_like(primary_times_sec_np, fill_value)
                        logger.debug(f"Fewer than 2 unique valid time points for '{key}' (var: {actual_var_name}) after cleaning for its own timeline. Filling with constant value {fill_value}.")
                    else:
                        interpolated_values = np.interp(
                            primary_times_sec_np,
                            unique_fp_times,
                            unique_f_values,
                            left=unique_f_values[0],
                            right=unique_f_values[-1]
                        )

                extracted_data[key] = interpolated_values.tolist()

            except Exception as e:
                logger.error(f"Error interpolating data for '{key}' (var: {actual_var_name}) in maneuver '{maneuver}': {e}", exc_info=True)
                extracted_data[key] = np.zeros_like(primary_times_sec_np).tolist() # Fill with zeros on error
                # Ensure original keys exist even on error
                if f'{key}_original_rel_times_sec' not in extracted_data:
                    extracted_data[f'{key}_original_rel_times_sec'] = []
                if f'{key}_original_values' not in extracted_data:
                    extracted_data[f'{key}_original_values'] = []


        return extracted_data

    def _build_initial_metrics(self, maneuver: str, test_type: str, channel: str,
                             rel_times_sec: List[float], extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build the basic metrics dictionary including extracted data."""
        metrics: Dict[str, Any] = {
            'status': 'processed',
            'maneuver': maneuver,
            'test_type': test_type,
            'channel': channel,
            'times': rel_times_sec,
            **{key: extracted_data.get(key, []) for key in self.vars_to_extract.keys()},
            # Include original data keys
            **{f'{key}_original_times': extracted_data.get(f'{key}_original_rel_times_sec', []) for key in self.vars_to_extract.keys()},
            **{f'{key}_original_values': extracted_data.get(f'{key}_original_values', []) for key in self.vars_to_extract.keys()},
        }

        # Calculate basic max values safely
        for key in self.vars_to_extract.keys():
             data = metrics.get(key, [])
             if data:
                 try:
                      metrics[f'max_{key}'] = max(data, key=abs)
                 except TypeError: # Handle potential non-numeric data if error occurred upstream
                      metrics[f'max_{key}'] = None
                      logger.warning(f"Could not calculate max value for '{key}' in maneuver '{maneuver}' due to non-numeric data.")
                 except Exception as e:
                      metrics[f'max_{key}'] = None
                      logger.warning(f"Error calculating max value for '{key}' in maneuver '{maneuver}': {e}")
             else:
                 metrics[f'max_{key}'] = 0.0

        return metrics

    def _calculate_dynamic_metrics(self, metrics: Dict[str, Any], test_type: str, channel: str, extracted_data: Dict[str, Any]):
        """Calculate metrics like rise time, settling time based on test type and channel."""

        data_for_metrics, times_for_metrics = self._select_data_by_channel(
            channel,
            metrics['times'],
            extracted_data # Pass the whole dict now
        )

        if not data_for_metrics or not times_for_metrics:
            logger.warning(f"No data selected for channel '{channel}' in maneuver '{metrics['maneuver']}' for metric calculation.")
            return # No data, no metrics

        if test_type == 'step':
            metrics.update({
                'rise_time': self._calculate_rise_time(times_for_metrics, data_for_metrics),
                'settling_time': self._calculate_settling_time(times_for_metrics, data_for_metrics),
                'overshoot': self._calculate_overshoot(data_for_metrics),
            })
        elif test_type == 'impulse':
            metrics.update({
                'recovery_time': self._calculate_recovery_time(times_for_metrics, data_for_metrics),
                'peak_value': max(data_for_metrics, key=abs) if data_for_metrics else 0.0,
            })
        elif test_type == 'sine_sweep':
            metrics.update({
                'max_amplitude': max(data_for_metrics, key=abs) if data_for_metrics else 0.0,
                # Keep frequency range info if config available
                'frequency_range': f"{self.config.sine_start_freq}-{self.config.sine_end_freq} Hz" if hasattr(self.config, 'sine_start_freq') and hasattr(self.config, 'sine_end_freq') else "N/A",
            })
        # else: unknown test type, no specific metrics calculated

    def _determine_test_type(self, maneuver: str) -> str:
        """
        Determine the test type from the maneuver name.
        
        Args:
            maneuver: Maneuver name
            
        Returns:
            Test type ('step', 'impulse', 'sine_sweep', or 'unknown')
        """
        if 'step_' in maneuver:
            return 'step'
        elif 'impulse_' in maneuver:
            return 'impulse'
        elif 'sine_sweep_' in maneuver:
            return 'sine_sweep'
        else:
            return 'unknown'
    
    def _determine_channel(self, maneuver: str) -> str:
        """
        Determine the channel from the maneuver name.
        
        Args:
            maneuver: Maneuver name
            
        Returns:
            Channel ('roll', 'pitch', 'thrust', 'height', or 'unknown')
        """
        if 'roll' in maneuver:
            return 'roll'
        elif 'pitch' in maneuver:
            return 'pitch'
        elif 'thrust' in maneuver: # Could be step_thrust, impulse_thrust
            return 'thrust'
        elif 'height' in maneuver or 'pos_z' in maneuver or 'z_axis' in maneuver : # e.g. step_height
            return 'height'
        elif 'yaw' in maneuver or 'psi' in maneuver or 'gyroZ' in maneuver: # Often, gyro_z is used for yaw response
             # For yaw, the primary data might be gyro.z.
             # Let's map 'yaw' channel to use gyro_z data in _select_data_by_channel.
            return 'yaw'
        else:
            return 'unknown'
    
    def _select_data_by_channel(self, channel: str,
                             times_sec: List[float], # This is the primary (interpolated) timeline
                             extracted_data: Dict[str, Any] # Use the extracted data dict
                             ) -> Tuple[List[float], List[float]]:
        """
        Select the appropriate data series based on the channel from extracted data.

        Args:
            channel: Channel name ('roll', 'pitch', 'thrust', 'yaw', 'height', or 'unknown')
            times_sec: The common time vector for all provided data series.
            extracted_data: Dictionary containing the interpolated data series.

        Returns:
            A tuple containing the selected data list and its corresponding time list.
            Returns empty lists if the channel is unknown or data is missing.
        """
        data_key_map = {
            'roll': 'roll',
            'pitch': 'pitch',
            'thrust': 'thrust', # Thrust channel should use extracted thrust data
            'yaw': 'gyro_z',   # Yaw analysis uses gyro_z data
            'height': 'height'
        }

        data_key = data_key_map.get(channel)

        if data_key and data_key in extracted_data:
             selected_data = extracted_data[data_key]
             # Ensure data is list (it should be, but safety check)
             if isinstance(selected_data, list):
                 return selected_data, times_sec
             else:
                  logger.warning(f"Data for key '{data_key}' (channel '{channel}') is not a list. Returning empty.")
                  return [], []
        else:
            logger.debug(f"No suitable data found for channel '{channel}' in extracted data.")
            # Return empty list for unknown channel or if data is not applicable/missing
            return [], []

    def _calculate_rise_time(self, times: List[float], values: List[float]) -> Optional[float]:
        """
        Calculate rise time (time to reach 90% of final value).
        
        Args:
            times: List of time points
            values: List of measured values
            
        Returns:
            Rise time in seconds or None if it cannot be calculated
        """
        if not times or not values or len(times) < 2:
            return None

        times_np = np.array(times)
        values_np = np.array(values)

        # Get the reference (initial) value
        initial_value = values_np[0]

        # Find the end of active test period - use only the step portion
        # Use config if available, else default (e.g., 1.5s)
        active_test_duration = getattr(self.config, 'step_duration', 1.5) # seconds
        active_mask = times_np <= active_test_duration
        if not np.any(active_mask): # If all times are beyond duration, use all data (edge case)
             active_mask = np.ones_like(times_np, dtype=bool)
             logger.warning("Rise time: All time points exceed active_test_duration. Using full dataset.")

        active_times = times_np[active_mask]
        active_values = values_np[active_mask]

        if len(active_values) < 2: # Need points within duration
             return None

        # Compute steady state value using the end of the active period
        # Use last 10% of active period or at least MIN_DATA_POINTS points within active period
        steady_state_window_size = max(self.MIN_DATA_POINTS, int(len(active_values) * 0.1))
        # Ensure window doesn't exceed available points
        steady_state_window_size = min(steady_state_window_size, len(active_values))
        # Take window from the end of the *active* values
        final_value = np.mean(active_values[-steady_state_window_size:])

        # If no significant change, return None
        change = final_value - initial_value
        # Use a small tolerance relative to the change magnitude or an absolute minimum
        tolerance = max(abs(change) * 0.01, 1e-3)
        if abs(change) < tolerance:
            logger.debug("Rise time calculation skipped: No significant change detected.")
            return None

        # Calculate the 10% and 90% targets (standard definition often uses 10%-90%)
        # target_10 = initial_value + 0.1 * change
        target_90 = initial_value + 0.9 * change

        # Find the first time when the value crosses the 90% target within the active period
        target_time_90 = None
        if change > 0: # Increasing step
             indices_above_90 = np.where(active_values >= target_90)[0]
             if indices_above_90.size > 0:
                 target_time_90 = active_times[indices_above_90[0]]
        else: # Decreasing step
            indices_below_90 = np.where(active_values <= target_90)[0]
            if indices_below_90.size > 0:
                target_time_90 = active_times[indices_below_90[0]]

        # Traditionally, rise time is t_90 - t_10. If we only calculate t_90:
        # return target_time_90 # This would be time *to* 90%

        # Let's stick to the original definition: time *at which* 90% is reached
        return target_time_90

    def _calculate_settling_time(self, times: List[float], values: List[float],
                                threshold_percent: float = 0.05) -> Optional[float]:
        """
        Calculate settling time (time to stay within a given percentage of final value).
        
        Args:
            times: List of time points
            values: List of measured values
            threshold_percent: Percentage threshold for settling (default: 5%)
            
        Returns:
            Settling time in seconds or None if it cannot be calculated
        """
        if not times or not values or len(times) < 2:
            return None

        times_np = np.array(times)
        values_np = np.array(values)

        # Find the end of active test period
        active_test_duration = getattr(self.config, 'step_duration', 1.5) # seconds
        active_mask = times_np <= active_test_duration
        if not np.any(active_mask):
             active_mask = np.ones_like(times_np, dtype=bool)
             logger.warning("Settling time: All time points exceed active_test_duration. Using full dataset.")


        active_times = times_np[active_mask]
        active_values = values_np[active_mask]

        if len(active_values) < self.MIN_DATA_POINTS: # Need enough points within duration
             logger.debug("Settling time calculation skipped: Not enough data points in active duration.")
             return None

        # Compute steady state value using the end of the active period
        steady_state_window_size = max(self.MIN_DATA_POINTS, int(len(active_values) * 0.1))
        steady_state_window_size = min(steady_state_window_size, len(active_values))
        final_value = np.mean(active_values[-steady_state_window_size:])


        # Calculate the threshold band relative to the steady state value
        # Use absolute threshold if final_value is near zero
        threshold = abs(threshold_percent * final_value) if abs(final_value) > 1e-6 else threshold_percent

        # Find the first time after which the signal *stays* within the band [final - threshold, final + threshold]
        # Search within the *active* period
        settling_time = None
        for i in range(len(active_values) - 1, -1, -1): # Search backwards from end of active period
            if abs(active_values[i] - final_value) > threshold:
                # This is the last point *outside* the band within the active duration.
                # The settling time is the time of the *next* point, if it exists.
                if i + 1 < len(active_values):
                    settling_time = active_times[i + 1]
                else:
                     # If the last point itself was outside, it hasn't settled within active duration
                     settling_time = None
                break
        else:
             # If the loop completes without break, all points were inside the band (or only 1 point).
             # Settling time is the time of the first point considered (or maybe the first point overall?).
             # Let's assume it settled immediately if all points are within the band.
             settling_time = active_times[0]


        # Alternative: Skip initial rise phase? Original code did int(max_active_idx * 0.1)
        # This can be complex. Sticking to "last time outside band" definition is common.

        return settling_time

    def _calculate_overshoot(self, values: List[float]) -> float:
        """
        Calculate percentage overshoot from a list of response values.
        
        Args:
            values: List of measured values
            
        Returns:
            Percentage overshoot (0.0 if no overshoot)
        """
        if not values or len(values) < 2:
            return 0.0

        values_np = np.array(values)
        active_test_duration = getattr(self.config, 'step_duration', 1.5) # seconds
        active_indices = int(len(values_np) * 0.75)
        active_indices = max(active_indices, self.MIN_DATA_POINTS) # Ensure minimum points
        active_indices = min(active_indices, len(values_np)) # Don't exceed length
        active_values = values_np[:active_indices]


        if len(active_values) < 2:
             return 0.0

        initial_value = values_np[0]

        # Get the steady-state value (average of last few points of *entire* series for stability)
        steady_state_window = max(self.MIN_DATA_POINTS, int(len(values_np) * 0.1))
        steady_state_window = min(steady_state_window, len(values_np))
        final_value = np.mean(values_np[-steady_state_window:])

        # If no significant change, return 0
        change = final_value - initial_value
        tolerance = max(abs(change) * 0.01, 1e-3)
        if abs(change) < tolerance:
             return 0.0

        # Determine if the response is increasing or decreasing
        increasing = change > 0

        # Find the peak value within the considered 'active' part
        peak_value = np.max(active_values) if increasing else np.min(active_values)

        # Calculate overshoot relative to the total change
        if abs(change) < 1e-9: # Avoid division by zero
             return 0.0

        if increasing:
            overshoot_val = peak_value - final_value
        else:
            overshoot_val = final_value - peak_value

        # Overshoot percentage = (overshoot value / total change) * 100
        # Ensure overshoot is positive (we are interested in magnitude exceeding final value)
        overshoot_percent = max(0.0, (overshoot_val / abs(change)) * 100.0)

        return overshoot_percent

    def _calculate_recovery_time(self, times: List[float], values: List[float],
                               threshold_percent: float = 0.1) -> Optional[float]:
        """
        Calculate recovery time after an impulse (time to return to within
        a given percentage of the initial value).
        
        Args:
            times: List of time points
            values: List of measured values
            threshold_percent: Percentage threshold for recovery (default: 10%)
            
        Returns:
            Recovery time in seconds or None if it cannot be calculated
        """
        if not times or not values or len(times) < 2:
            return None

        times_np = np.array(times)
        values_np = np.array(values)

        # Get the initial value (assuming it's the reference equilibrium before impulse)
        initial_value = values_np[0]

        # Calculate the threshold band relative to the initial value
        # Ensure threshold isn't too small if initial value is near zero
        threshold = abs(threshold_percent * initial_value) if abs(initial_value) > 1e-6 else threshold_percent
        min_threshold = 0.05 # Example minimum absolute threshold
        threshold = max(threshold, min_threshold)


        # Find the peak deviation index (maximum absolute difference from initial)
        deviation = np.abs(values_np - initial_value)
        peak_idx = np.argmax(deviation)

        # Start searching *after* the peak index
        search_start_idx = peak_idx + 1
        if search_start_idx >= len(times_np):
            # Peak was the last point, cannot recover after it.
            return None

        # Find the first index after the peak where the value is back within the threshold
        indices_within_threshold = np.where(deviation[search_start_idx:] <= threshold)[0]

        if indices_within_threshold.size > 0:
            # Found recovery point(s). Get the index relative to the start of the *sliced* array.
            first_recovery_idx_relative = indices_within_threshold[0]
            # Convert back to index in the original array
            first_recovery_idx_absolute = search_start_idx + first_recovery_idx_relative
            return times_np[first_recovery_idx_absolute]
        else:
            # Never recovered within the threshold after the peak
            return None

    def extract_frequency_response(self,
                                 maneuver: str,
                                 processed_data: Dict[str, Any]) -> Dict[str, Any]: # Changed input
        """
        Extract frequency response data from sine sweep using processed data.

        Placeholder for future implementation of frequency domain analysis.

        Args:
            maneuver: Name of the maneuver (used for logging).
            processed_data: The dictionary of processed metrics and data for this maneuver.

        Returns:
            Dictionary with frequency response data (currently placeholder).
        """
        if 'sine_sweep_' not in maneuver:
            logger.warning(f"'{maneuver}' is not a sine sweep test. Skipping frequency response extraction.")
            return {}

        logger.info(f"Frequency response analysis for {maneuver} is not yet implemented.")

        # Potential future implementation inputs from processed_data:
        # input_signal = processed_data.get(f'command_{processed_data["channel"]}', [])
        # output_signal = processed_data.get(processed_data['channel'], [])
        # times = processed_data.get('times', [])
        # Fs = 1 / np.mean(np.diff(times)) if times else None # Sample rate

        # Perform FFT, calculate gain/phase, etc.

        return {
            'frequencies': [],
            'gains': [],
            'phases': [],
            'note': 'Frequency response analysis is not yet implemented'
        } 