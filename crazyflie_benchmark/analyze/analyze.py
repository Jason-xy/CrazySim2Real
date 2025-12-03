"""
Main entry point for advanced data analysis.

This module provides the main functions for running detailed metrics calculations,
comparison analyses, and visualizations.
"""
import logging
import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union

from ..config import FlightConfig
from ..data_processor import DataProcessor
from .metrics import calculate_test_metrics
from .comparator import compare_datasets
from .visualizer import generate_visualizations, generate_comparison_visualizations

logger = logging.getLogger(__name__)

def run_metrics_analysis(data_dir: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Run advanced metrics analysis on a data directory.
    
    Args:
        data_dir: Path to data directory containing test logs
        output_dir: Output directory for reports and visualizations (default: data_dir/analysis)
        
    Returns:
        Dictionary with analysis results and output paths
    """
    logger.info(f"Running advanced metrics analysis on {data_dir}")
    
    # Set default output directory if not specified
    if output_dir is None:
        output_dir = os.path.join(data_dir, 'analysis')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data and calculate metrics
    try:
        from ..main import load_raw_data_from_directory
        
        # Load raw data
        raw_data = load_raw_data_from_directory(data_dir)
        if not raw_data:
            logger.error(f"No valid data found in {data_dir}")
            return {'error': f"No valid data found in {data_dir}"}
        
        # Process data with standard processor
        config = FlightConfig()
        processor = DataProcessor(config)
        processed_data = processor.process(raw_data)
        
        # Calculate advanced metrics
        metrics = calculate_test_metrics(processed_data)
        
        # Save metrics to JSON file
        metrics_path = os.path.join(output_dir, 'advanced_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Generate visualizations
        visualization_paths = generate_visualizations(metrics, output_dir)
        
        # Generate summary report
        summary_path = _generate_summary_report(metrics, output_dir)
        
        return {
            'status': 'success',
            'data_dir': data_dir,
            'output_dir': output_dir,
            'metrics_file': metrics_path,
            'summary_file': summary_path,
            'visualizations': visualization_paths,
            'metrics_data': metrics
        }
    
    except Exception as e:
        logger.error(f"Error running metrics analysis: {e}", exc_info=True)
        return {'error': str(e)}

def run_comparison_analysis(data_dir1: str, 
                          data_dir2: str, 
                          output_dir: Optional[str] = None,
                          label1: Optional[str] = None,
                          label2: Optional[str] = None) -> Dict[str, Any]:
    """
    Run comparison analysis between two data directories.
    
    Args:
        data_dir1: Path to first data directory
        data_dir2: Path to second data directory
        output_dir: Output directory for comparison reports (default: logs/comparison_<timestamp>)
        label1: Label for first dataset (default: directory name)
        label2: Label for second dataset (default: directory name)
        
    Returns:
        Dictionary with comparison results and output paths
    """
    logger.info(f"Running comparison analysis between {data_dir1} and {data_dir2}")
    
    # Set default output directory if not specified
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join('logs', f'comparison_{timestamp}')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default labels if not specified
    if label1 is None:
        label1 = os.path.basename(os.path.normpath(data_dir1))
    
    if label2 is None:
        label2 = os.path.basename(os.path.normpath(data_dir2))
    
    try:
        # Run comparison
        comparison_results = compare_datasets(data_dir1, data_dir2, output_dir, label1, label2)
        
        # Check for errors
        if 'error' in comparison_results:
            return comparison_results
        
        return {
            'status': 'success',
            'comparison_results': comparison_results,
            'output_dir': output_dir
        }
    
    except Exception as e:
        logger.error(f"Error running comparison analysis: {e}", exc_info=True)
        return {'error': str(e)}

def _generate_summary_report(metrics: Dict[str, Dict[str, Any]], output_dir: str) -> str:
    """
    Generate a summary report of the metrics analysis.
    
    Args:
        metrics: Dictionary of metrics by test name
        output_dir: Output directory for the report
        
    Returns:
        Path to the generated report file
    """
    summary_path = os.path.join(output_dir, 'metrics_summary.txt')
    
    try:
        with open(summary_path, 'w') as f:
            f.write("===== ADVANCED METRICS ANALYSIS =====\n\n")
            f.write(f"Analysis performed on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Group tests by type
            tests_by_type = {}
            for test_name, test_metrics in metrics.items():
                test_type = test_metrics.get('test_type', 'unknown')
                if test_type not in tests_by_type:
                    tests_by_type[test_type] = []
                tests_by_type[test_type].append((test_name, test_metrics))
            
            # Write summary by test type
            for test_type, tests in sorted(tests_by_type.items()):
                if test_type == 'unknown':
                    continue
                
                f.write(f"\n== {test_type.upper()} TESTS ==\n")
                
                for test_name, test_metrics in sorted(tests):
                    f.write(f"\n{test_name}:\n")
                    f.write(f"  Channel: {test_metrics.get('channel', 'unknown')}\n")
                    
                    # Write key metrics based on test type
                    key_metrics = _get_key_metrics_for_summary(test_type, test_metrics)
                    
                    for metric_name, metric_value in key_metrics:
                        if isinstance(metric_value, (int, float)):
                            f.write(f"  {metric_name}: {metric_value:.4f}\n")
                        else:
                            f.write(f"  {metric_name}: {metric_value}\n")
                    
                    # Add blank line between tests
                    f.write("\n")
            
            f.write("\nDetailed metrics saved in 'advanced_metrics.json'\n")
        
        logger.info(f"Summary report generated at {summary_path}")
        return summary_path
    
    except Exception as e:
        logger.error(f"Error generating summary report: {e}")
        return ""

def _get_key_metrics_for_summary(test_type: str, metrics: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """
    Get the key metrics to display in the summary report for a specific test type.
    
    Args:
        test_type: Test type
        metrics: Test metrics
        
    Returns:
        List of (metric_name, metric_value) tuples
    """
    # Common metrics for all test types
    common_metrics = [
        ('stability_score', metrics.get('stability_score', 'N/A')),
        ('control_effort', metrics.get('control_effort', 'N/A')),
        ('energy_consumption', metrics.get('energy_consumption', 'N/A'))
    ]
    
    # Test-specific metrics
    if test_type == 'step':
        specific_metrics = [
            ('rise_time', metrics.get('rise_time', 'N/A')),
            ('settling_time', metrics.get('settling_time', 'N/A')),
            ('overshoot', metrics.get('overshoot', 'N/A')),
            ('steady_state_error', metrics.get('steady_state_error', 'N/A')),
            ('damping_ratio', metrics.get('damping_ratio', 'N/A'))
        ]
    elif test_type == 'impulse':
        specific_metrics = [
            ('recovery_time', metrics.get('recovery_time', 'N/A')),
            ('peak_value', metrics.get('peak_value', 'N/A')),
            ('decay_time', metrics.get('decay_time', 'N/A')),
            ('oscillation_frequency', metrics.get('oscillation_frequency', 'N/A')),
            ('oscillation_count', metrics.get('oscillation_count', 'N/A'))
        ]
    elif test_type == 'sine_sweep':
        specific_metrics = [
            ('resonance_frequency', metrics.get('resonance_frequency', 'N/A')),
            ('bandwidth', metrics.get('bandwidth', 'N/A')),
            ('amplitude_ratio', metrics.get('amplitude_ratio', 'N/A')),
            ('phase_margin', metrics.get('phase_margin', 'N/A'))
        ]
    elif test_type == 'hover':
        specific_metrics = [
            ('position_variance', metrics.get('position_variance', 'N/A')),
            ('drift_rate', metrics.get('drift_rate', 'N/A')),
            ('power_efficiency', metrics.get('power_efficiency', 'N/A'))
        ]
    else:
        specific_metrics = []
    
    # Return test-specific metrics first, then common metrics
    return specific_metrics + common_metrics 