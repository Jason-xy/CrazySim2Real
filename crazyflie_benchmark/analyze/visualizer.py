"""
Advanced visualization tools for drone test data.

This module provides specialized visualization capabilities for test data,
including radar charts for comparing metrics, transfer function plots, and
comparative time series visualization.
"""
import logging
import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

# Import matplotlib only when needed to avoid loading GUI dependencies
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

logger = logging.getLogger(__name__)

class AdvancedVisualizer:
    """
    Advanced visualization tools for test data analysis.
    
    Generates specialized plots beyond the basic time-series visualizations,
    including performance radar charts, FFT analyses, and comparative plots.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = output_dir
        self.color_map = {
            'roll': 'red',
            'pitch': 'blue',
            'yaw': 'green',
            'thrust': 'purple',
            'height': 'orange'
        }
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_radar_chart(self, 
                           metrics: Dict[str, float], 
                           test_name: str,
                           test_type: str,
                           channel: str) -> str:
        """
        Generate a radar chart for a set of performance metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            test_name: Name of the test
            test_type: Type of test ('step', 'impulse', etc.)
            channel: Control channel ('roll', 'pitch', etc.)
            
        Returns:
            Path to the saved chart image
        """
        # Filter metrics to include only numeric values within a reasonable range
        filtered_metrics = {}
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if abs(value) < 1e6:  # Avoid extreme values
                    filtered_metrics[name] = value
        
        if len(filtered_metrics) < 3:
            logger.warning(f"Not enough metrics for radar chart: {test_name}")
            return ""
        
        # Select the most relevant metrics (up to 8) based on test type
        relevant_metrics = self._select_relevant_metrics(filtered_metrics, test_type)
        
        if len(relevant_metrics) < 3:
            logger.warning(f"Not enough relevant metrics for radar chart: {test_name}")
            return ""
        
        # Normalize metrics to 0-1 scale for radar chart
        normalized_metrics = self._normalize_metrics(relevant_metrics)
        
        # Create the radar chart
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Number of variables
        N = len(normalized_metrics)
        
        # Define angles for each metric (evenly spaced around the circle)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Get metric names and values, with values repeated to close the loop
        metric_names = list(normalized_metrics.keys())
        values = list(normalized_metrics.values())
        values += values[:1]  # Close the loop
        
        # Draw the plot
        ax.plot(angles, values, linewidth=2, linestyle='solid', 
                color=self.color_map.get(channel, 'blue'))
        ax.fill(angles, values, alpha=0.25, 
                color=self.color_map.get(channel, 'blue'))
        
        # Add metric labels with original values
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f"{name}\n({list(relevant_metrics.values())[i]:.2f})" 
                           for i, name in enumerate(metric_names)])
        
        # Add gridlines and adjust appearance
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"])
        ax.grid(True)
        
        # Set title
        plt.title(f"Performance Metrics: {test_name}\nTest Type: {test_type}, Channel: {channel}", 
                 size=15, color='black', y=1.1)
        
        # Save the plot
        output_path = os.path.join(self.output_dir, f"{test_name}_radar.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        return output_path
    
    def _select_relevant_metrics(self, 
                               metrics: Dict[str, float], 
                               test_type: str) -> Dict[str, float]:
        """
        Select the most relevant metrics for a radar chart based on test type.
        
        Args:
            metrics: Dictionary of metrics
            test_type: Type of test
            
        Returns:
            Dictionary containing selected metrics (up to 8)
        """
        # Define priority metrics for each test type
        priority_metrics = {
            'step': [
                'rise_time', 'settling_time', 'overshoot', 'stability_score',
                'control_effort', 'energy_consumption', 'trajectory_error',
                'steady_state_error', 'damping_ratio'
            ],
            'impulse': [
                'recovery_time', 'peak_value', 'stability_score', 'control_effort',
                'oscillation_frequency', 'oscillation_count', 'decay_time',
                'energy_consumption', 'trajectory_error'
            ],
            'sine_sweep': [
                'amplitude_ratio', 'bandwidth', 'stability_score', 'control_effort',
                'energy_consumption', 'trajectory_error'
            ],
            'hover': [
                'position_variance', 'attitude_variance', 'power_efficiency',
                'drift_rate', 'stability_score', 'control_effort', 'energy_consumption'
            ]
        }
        
        # Get the priority list for this test type
        priority_list = priority_metrics.get(test_type, list(metrics.keys()))
        
        # Filter metrics to keep only those in priority list, in priority order, up to 8
        selected_metrics = {}
        for metric in priority_list:
            if metric in metrics and len(selected_metrics) < 8:
                selected_metrics[metric] = metrics[metric]
        
        return selected_metrics
    
    def _normalize_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize metrics to 0-1 scale for visualization.
        
        Handles metrics where higher is better versus lower is better.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Dictionary of normalized metrics
        """
        normalized = {}
        
        # Define metrics where higher is better
        higher_is_better = [
            'stability_score', 
            'power_efficiency',
            'damping_ratio',
            'bandwidth'
        ]
        
        for name, value in metrics.items():
            # For metrics where higher is better, normalize directly
            if name in higher_is_better:
                # Map to 0.2-1.0 range to ensure visibility even for low scores
                normalized[name] = 0.2 + (min(value, 100) / 100) * 0.8
            # For metrics where lower is better, invert the normalization
            else:
                # Map to 0.2-1.0 range, with lower values being better
                # Use a sigmoid-like curve to handle varying ranges
                normalized[name] = 0.2 + 0.8 / (1 + value * value / 100)
        
        return normalized
    
    def generate_comparison_radar(self, 
                                dataset1_metrics: Dict[str, float],
                                dataset2_metrics: Dict[str, float],
                                test_name: str,
                                test_type: str,
                                channel: str,
                                label1: str = "Dataset 1",
                                label2: str = "Dataset 2") -> str:
        """
        Generate a radar chart comparing metrics from two datasets.
        
        Args:
            dataset1_metrics: Metrics from first dataset
            dataset2_metrics: Metrics from second dataset
            test_name: Name of the test
            test_type: Type of test
            channel: Control channel
            label1: Label for first dataset
            label2: Label for second dataset
            
        Returns:
            Path to the saved chart image
        """
        # Get common metrics
        common_metrics = {}
        for name, value in dataset1_metrics.items():
            if (name in dataset2_metrics and 
                isinstance(value, (int, float)) and 
                isinstance(dataset2_metrics[name], (int, float)) and
                not isinstance(value, bool) and
                not isinstance(dataset2_metrics[name], bool) and
                abs(value) < 1e6 and abs(dataset2_metrics[name]) < 1e6):
                common_metrics[name] = value
        
        if len(common_metrics) < 3:
            logger.warning(f"Not enough common metrics for comparison radar chart: {test_name}")
            return ""
        
        # Select relevant metrics
        relevant_metrics = self._select_relevant_metrics(common_metrics, test_type)
        
        if len(relevant_metrics) < 3:
            logger.warning(f"Not enough relevant common metrics for radar chart: {test_name}")
            return ""
        
        # Create a dictionary with both datasets' values for each metric
        metrics_to_plot = {}
        for name in relevant_metrics.keys():
            metrics_to_plot[name] = {
                'dataset1': dataset1_metrics[name],
                'dataset2': dataset2_metrics[name]
            }
        
        # Normalize all metrics to 0-1 range
        normalized_metrics = {
            'dataset1': {},
            'dataset2': {}
        }
        
        higher_is_better = [
            'stability_score', 
            'power_efficiency',
            'damping_ratio',
            'bandwidth'
        ]
        
        for name, values in metrics_to_plot.items():
            # Get the min and max across both datasets
            min_val = min(values['dataset1'], values['dataset2'])
            max_val = max(values['dataset1'], values['dataset2'])
            
            # Ensure there's a range to normalize
            if abs(max_val - min_val) < 1e-6:
                # Almost identical values, set both to 0.6
                normalized_metrics['dataset1'][name] = 0.6
                normalized_metrics['dataset2'][name] = 0.6
            else:
                # For metrics where higher is better, normalize directly
                if name in higher_is_better:
                    # Map to 0.2-1.0 range
                    for ds in ['dataset1', 'dataset2']:
                        normalized_metrics[ds][name] = 0.2 + (values[ds] / 100) * 0.8
                # For metrics where lower is better, invert the normalization
                else:
                    # Lower values are better, use inverse normalization
                    for ds in ['dataset1', 'dataset2']:
                        normalized_metrics[ds][name] = 0.2 + 0.8 / (1 + values[ds] * values[ds] / 100)
        
        # Create the radar chart
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Number of variables
        N = len(metrics_to_plot)
        
        # Define angles for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Get metric names
        metric_names = list(metrics_to_plot.keys())
        
        # Plot dataset1
        values1 = [normalized_metrics['dataset1'][m] for m in metric_names]
        values1 += values1[:1]  # Close the loop
        ax.plot(angles, values1, 'o-', linewidth=2, label=label1, color='blue')
        ax.fill(angles, values1, alpha=0.1, color='blue')
        
        # Plot dataset2
        values2 = [normalized_metrics['dataset2'][m] for m in metric_names]
        values2 += values2[:1]  # Close the loop
        ax.plot(angles, values2, 'o-', linewidth=2, label=label2, color='green')
        ax.fill(angles, values2, alpha=0.1, color='green')
        
        # Add metric labels with both values
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f"{name}\n{label1}: {metrics_to_plot[name]['dataset1']:.2f}\n{label2}: {metrics_to_plot[name]['dataset2']:.2f}" 
                           for name in metric_names])
        
        # Add gridlines and adjust appearance
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"])
        ax.grid(True)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Set title
        plt.title(f"Performance Comparison: {test_name}\nTest Type: {test_type}, Channel: {channel}", 
                 size=15, color='black', y=1.1)
        
        # Save the plot
        output_path = os.path.join(self.output_dir, f"{test_name}_comparison_radar.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        return output_path
    
    def generate_comparison_timeseries(self,
                                     data1: Dict[str, Any],
                                     data2: Dict[str, Any],
                                     test_name: str,
                                     label1: str = "Dataset 1",
                                     label2: str = "Dataset 2") -> str:
        """
        Generate a comparison plot of time series data from two datasets.
        
        Args:
            data1: Time series data from first dataset
            data2: Time series data from second dataset
            test_name: Name of the test
            label1: Label for first dataset
            label2: Label for second dataset
            
        Returns:
            Path to the saved chart image
        """
        # Check if we have valid time series data
        if ('times' not in data1 or 'times' not in data2 or
            len(data1['times']) < 2 or len(data2['times']) < 2):
            logger.warning(f"Not enough time series data for comparison plot: {test_name}")
            return ""
        
        # Get test type and channel
        test_type = data1.get('test_type', data2.get('test_type', 'unknown'))
        channel = data1.get('channel', data2.get('channel', 'unknown'))
        
        # Create a multi-panel figure
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(3, 1, figure=fig)
        
        # Panel 1: Primary channel response
        ax1 = fig.add_subplot(gs[0, 0])
        self._add_comparison_trace(ax1, data1, data2, channel, channel, label1, label2)
        ax1.set_title(f"Comparison: {test_name} - {test_type.capitalize()} Test ({channel})")
        
        # Panel 2: Gyro data
        ax2 = fig.add_subplot(gs[1, 0])
        if channel == 'roll':
            self._add_comparison_trace(ax2, data1, data2, 'gyro_x', 'Angular Rate (X)', label1, label2)
        elif channel == 'pitch':
            self._add_comparison_trace(ax2, data1, data2, 'gyro_y', 'Angular Rate (Y)', label1, label2)
        elif channel == 'yaw':
            self._add_comparison_trace(ax2, data1, data2, 'gyro_z', 'Angular Rate (Z)', label1, label2)
        else:
            # For non-attitude channels, show all gyro data
            self._add_comparison_trace(ax2, data1, data2, 'gyro_x', 'Angular Rate (X)', label1, label2)
            self._add_comparison_trace(ax2, data1, data2, 'gyro_y', 'Angular Rate (Y)', label1, label2, alt_linestyle='--')
            self._add_comparison_trace(ax2, data1, data2, 'gyro_z', 'Angular Rate (Z)', label1, label2, alt_linestyle=':')
        
        # Panel 3: Commands
        ax3 = fig.add_subplot(gs[2, 0])
        command_key = f'command_{channel}'
        if command_key in data1 and command_key in data2:
            self._add_comparison_trace(ax3, data1, data2, command_key, f'Command ({channel})', label1, label2)
        else:
            # If specific command not available, show thrust command as it's usually available
            self._add_comparison_trace(ax3, data1, data2, 'command_thrust', 'Command (thrust)', label1, label2)
        
        # Add overall legend
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        
        # Adjust layout and save
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f"{test_name}_comparison_timeseries.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        return output_path
    
    def _add_comparison_trace(self,
                            ax: plt.Axes,
                            data1: Dict[str, Any],
                            data2: Dict[str, Any],
                            key: str,
                            label: str,
                            dataset1_label: str,
                            dataset2_label: str,
                            alt_linestyle: str = '-'):
        """
        Add a comparison time series trace to a matplotlib axes.
        
        Args:
            ax: Matplotlib axes to plot on
            data1: First dataset
            data2: Second dataset
            key: Data key to plot
            label: Label for the data
            dataset1_label: Label for first dataset
            dataset2_label: Label for second dataset
            alt_linestyle: Alternative line style for multiple traces
        """
        if key in data1 and key in data2 and 'times' in data1 and 'times' in data2:
            # Plot dataset 1
            if len(data1[key]) > 1 and len(data1['times']) > 1:
                ax.plot(data1['times'], data1[key], 
                        color='blue', linestyle=alt_linestyle,
                        label=f"{label} - {dataset1_label}")
            
            # Plot dataset 2
            if len(data2[key]) > 1 and len(data2['times']) > 1:
                ax.plot(data2['times'], data2[key], 
                        color='green', linestyle=alt_linestyle,
                        label=f"{label} - {dataset2_label}")
            
            ax.grid(True)
            ax.set_ylabel(label)
            ax.set_xlabel('Time (s)')
        else:
            ax.text(0.5, 0.5, f"Data not available for {key}", 
                   ha='center', va='center', transform=ax.transAxes)
    
    def generate_performance_summary(self, 
                                   all_metrics: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate a summary visualization of all test performance.
        
        Creates a grid of performance indicators colored by performance level.
        
        Args:
            all_metrics: Dictionary of metrics for all tests
            
        Returns:
            Path to the saved chart image
        """
        # Group tests by type
        tests_by_type = {}
        for test_name, metrics in all_metrics.items():
            test_type = metrics.get('test_type', 'unknown')
            if test_type not in tests_by_type:
                tests_by_type[test_type] = []
            tests_by_type[test_type].append((test_name, metrics))
        
        # Skip if no valid test types
        if not tests_by_type or all(t == 'unknown' for t in tests_by_type.keys()):
            logger.warning("No valid test types found for performance summary")
            return ""
        
        # Create a figure with a grid for each test type
        fig = plt.figure(figsize=(15, 10))
        
        # Calculate grid dimensions
        num_test_types = len(tests_by_type)
        rows = max(1, (num_test_types + 1) // 2)
        cols = min(2, num_test_types)
        
        # Define a color map for performance scores
        cmap = plt.cm.RdYlGn  # Red (poor) to Yellow (average) to Green (good)
        
        # Plot each test type in its own subplot
        for i, (test_type, tests) in enumerate(sorted(tests_by_type.items())):
            if test_type == 'unknown':
                continue
                
            ax = fig.add_subplot(rows, cols, i+1)
            
            # Get key metrics for this test type
            key_metrics = self._get_key_metrics_for_type(test_type)
            
            # Prepare data for heatmap
            metric_names = []
            test_names = []
            performance_scores = []
            
            for test_name, metrics in sorted(tests):
                test_names.append(test_name)
                
                # Get scores for each key metric
                metric_scores = []
                for metric in key_metrics:
                    if metric not in metric_names:
                        metric_names.append(metric)
                    
                    # Get score or default to middle value (0.5)
                    if metric in metrics and isinstance(metrics[metric], (int, float)):
                        # Normalize to 0-1 range (1 = good, 0 = bad)
                        score = self._normalize_metric_for_heatmap(metric, metrics[metric])
                        metric_scores.append(score)
                    else:
                        metric_scores.append(0.5)  # Neutral score if missing
                
                performance_scores.append(metric_scores)
            
            # Convert to numpy array
            performance_array = np.array(performance_scores)
            
            # Create heatmap
            im = ax.imshow(performance_array, cmap=cmap, vmin=0, vmax=1, aspect='auto')
            
            # Add test names on y-axis
            ax.set_yticks(np.arange(len(test_names)))
            ax.set_yticklabels([t[:20] + '...' if len(t) > 20 else t for t in test_names])
            
            # Add metric names on x-axis
            ax.set_xticks(np.arange(len(metric_names)))
            ax.set_xticklabels(metric_names, rotation=45, ha='right')
            
            # Add color bar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Performance (0=Poor, 1=Excellent)')
            
            # Add title
            ax.set_title(f"{test_type.capitalize()} Test Performance")
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "performance_summary.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        return output_path
    
    def _get_key_metrics_for_type(self, test_type: str) -> List[str]:
        """
        Get the key metrics for a specific test type.
        
        Args:
            test_type: Test type
            
        Returns:
            List of key metric names
        """
        # Define key metrics for each test type
        key_metrics = {
            'step': [
                'rise_time', 'settling_time', 'overshoot', 'steady_state_error',
                'stability_score', 'control_effort'
            ],
            'impulse': [
                'recovery_time', 'peak_value', 'oscillation_count',
                'stability_score', 'control_effort', 'decay_time'
            ],
            'sine_sweep': [
                'bandwidth', 'amplitude_ratio', 'stability_score',
                'control_effort', 'energy_consumption'
            ],
            'hover': [
                'position_variance', 'drift_rate', 'power_efficiency',
                'stability_score', 'control_effort', 'energy_consumption'
            ]
        }
        
        return key_metrics.get(test_type, ['stability_score', 'control_effort'])
    
    def _normalize_metric_for_heatmap(self, metric: str, value: float) -> float:
        """
        Normalize a metric value to 0-1 range for heatmap visualization.
        
        Args:
            metric: Metric name
            value: Metric value
            
        Returns:
            Normalized value between 0 and 1
        """
        # Metrics where higher is better
        higher_is_better = [
            'stability_score', 
            'power_efficiency',
            'damping_ratio',
            'bandwidth'
        ]
        
        if metric in higher_is_better:
            # For stability score, it's already 0-100
            if metric == 'stability_score':
                return min(value / 100.0, 1.0)
            
            # For power efficiency, higher is better
            if metric == 'power_efficiency':
                return min(value / 5.0, 1.0)  # Assume 5.0 is excellent
            
            # For damping ratio, 0.7-1.0 is typically ideal
            if metric == 'damping_ratio':
                if value > 1.0:
                    return 0.8  # Overdamped but still good
                elif value >= 0.7:
                    return 1.0  # Optimally damped
                else:
                    return 0.7 * value  # Underdamped
            
            # For bandwidth, higher is generally better
            if metric == 'bandwidth':
                return min(value / 10.0, 1.0)  # Assume 10 Hz is excellent
        
        # Metrics where lower is better
        else:
            # For rise time, lower is better (0.1s is excellent)
            if metric == 'rise_time':
                return max(0, 1.0 - value / 2.0)  # 0-2s range
            
            # For settling time, lower is better (0.5s is excellent)
            if metric == 'settling_time':
                return max(0, 1.0 - value / 5.0)  # 0-5s range
            
            # For overshoot, lower is better (0% is excellent)
            if metric == 'overshoot':
                return max(0, 1.0 - value / 50.0)  # 0-50% range
            
            # For steady state error, lower is better (0 is excellent)
            if metric == 'steady_state_error':
                return max(0, 1.0 - value / 10.0)  # 0-10 range
            
            # For control effort, lower is better (0 is excellent)
            if metric == 'control_effort':
                return max(0, 1.0 - value / 1000.0)  # 0-1000 range
            
            # For energy consumption, lower is better (0 is excellent)
            if metric == 'energy_consumption':
                return max(0, 1.0 - value / 10000.0)  # 0-10000 range
            
            # For position variance, lower is better (0 is excellent)
            if metric == 'position_variance':
                return max(0, 1.0 - value / 0.1)  # 0-0.1 range
            
            # For drift rate, lower is better (0 is excellent)
            if metric == 'drift_rate':
                return max(0, 1.0 - abs(value) / 0.1)  # -0.1 to 0.1 range
            
            # For oscillation count, lower is better (0 is excellent)
            if metric == 'oscillation_count':
                return max(0, 1.0 - value / 10.0)  # 0-10 range
            
            # For decay time, lower is better (0 is excellent)
            if metric == 'decay_time':
                return max(0, 1.0 - value / 3.0)  # 0-3s range
        
        # Default normalization for other metrics
        return 0.5  # Neutral score


def generate_visualizations(metrics_data: Dict[str, Dict[str, Any]], 
                          output_dir: str) -> List[str]:
    """
    Generate all available visualizations for a set of metrics.
    
    Args:
        metrics_data: Dictionary of test metrics
        output_dir: Directory to save visualizations
        
    Returns:
        List of paths to generated visualizations
    """
    visualizer = AdvancedVisualizer(output_dir)
    generated_paths = []
    
    # Generate radar charts for each test
    for test_name, metrics in metrics_data.items():
        test_type = metrics.get('test_type', 'unknown')
        channel = metrics.get('channel', 'unknown')
        
        if test_type != 'unknown' and channel != 'unknown':
            radar_path = visualizer.generate_radar_chart(metrics, test_name, test_type, channel)
            if radar_path:
                generated_paths.append(radar_path)
    
    # Generate performance summary
    summary_path = visualizer.generate_performance_summary(metrics_data)
    if summary_path:
        generated_paths.append(summary_path)
    
    return generated_paths


def generate_comparison_visualizations(dataset1: Dict[str, Dict[str, Any]],
                                     dataset2: Dict[str, Dict[str, Any]],
                                     output_dir: str,
                                     label1: str = "Dataset 1",
                                     label2: str = "Dataset 2") -> List[str]:
    """
    Generate comparison visualizations for two datasets.
    
    Args:
        dataset1: First dataset metrics
        dataset2: Second dataset metrics
        output_dir: Directory to save visualizations
        label1: Label for first dataset
        label2: Label for second dataset
        
    Returns:
        List of paths to generated visualizations
    """
    visualizer = AdvancedVisualizer(output_dir)
    generated_paths = []
    
    # Find common tests
    common_tests = set(dataset1.keys()) & set(dataset2.keys())
    
    # Generate comparison visualizations for each common test
    for test_name in common_tests:
        data1 = dataset1[test_name]
        data2 = dataset2[test_name]
        
        test_type = data1.get('test_type', data2.get('test_type', 'unknown'))
        channel = data1.get('channel', data2.get('channel', 'unknown'))
        
        if test_type != 'unknown' and channel != 'unknown':
            # Generate radar comparison
            radar_path = visualizer.generate_comparison_radar(
                data1, data2, test_name, test_type, channel, label1, label2
            )
            if radar_path:
                generated_paths.append(radar_path)
            
            # Generate time series comparison
            timeseries_path = visualizer.generate_comparison_timeseries(
                data1, data2, test_name, label1, label2
            )
            if timeseries_path:
                generated_paths.append(timeseries_path)
    
    return generated_paths 