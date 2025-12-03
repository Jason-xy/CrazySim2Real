"""
Comparator module for analyzing differences between test datasets.

This module provides tools to compare performance metrics between two sets of test data,
such as simulation vs. real-world or before vs. after controller tuning.
"""
import logging
import numpy as np
import os
import json
import csv
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class DataComparator:
    """
    Compare performance metrics between two datasets.
    
    Provides tools to quantify differences in performance metrics, generate 
    comparison reports, and identify statistically significant changes.
    """
    
    def __init__(self):
        """Initialize the data comparator."""
        # Define which metrics to compare by test type
        self.comparison_metrics = {
            'step': [
                'rise_time', 
                'settling_time', 
                'overshoot', 
                'steady_state_error',
                'damping_ratio',
                'energy_consumption',
                'control_effort',
                'trajectory_error',
                'stability_score'
            ],
            'impulse': [
                'recovery_time',
                'peak_value',
                'decay_time',
                'oscillation_frequency',
                'oscillation_count',
                'energy_consumption',
                'control_effort',
                'trajectory_error',
                'stability_score'
            ],
            'sine_sweep': [
                'resonance_frequency',
                'bandwidth',
                'amplitude_ratio',
                'energy_consumption',
                'control_effort',
                'trajectory_error',
                'stability_score'
            ],
            'hover': [
                'position_variance',
                'attitude_variance',
                'power_efficiency',
                'drift_rate',
                'energy_consumption',
                'control_effort',
                'stability_score'
            ]
        }
        
        # Metrics where higher values are better (used for scoring)
        self.higher_is_better = [
            'stability_score', 
            'power_efficiency',
            'damping_ratio',
            'bandwidth'
        ]
        
        # Metrics where lower values are better (used for scoring)
        self.lower_is_better = [
            'rise_time',
            'settling_time',
            'recovery_time',
            'trajectory_error',
            'overshoot',
            'steady_state_error',
            'control_effort',
            'energy_consumption',
            'position_variance',
            'drift_rate',
            'oscillation_count', 
            'decay_time'
        ]
    
    def compare_datasets(self, 
                        dataset1: Dict[str, Dict[str, Any]], 
                        dataset2: Dict[str, Dict[str, Any]],
                        label1: str = "Dataset 1",
                        label2: str = "Dataset 2") -> Dict[str, Any]:
        """
        Compare two sets of performance data.
        
        Args:
            dataset1: First dataset (baseline)
            dataset2: Second dataset to compare against baseline
            label1: Label for the first dataset
            label2: Label for the second dataset
            
        Returns:
            Dictionary containing comparison results and summary metrics
        """
        comparison_results = {
            'metadata': {
                'dataset1_label': label1,
                'dataset2_label': label2,
                'common_tests': [],
                'only_in_dataset1': [],
                'only_in_dataset2': [],
                'summary': {}
            },
            'test_comparisons': {},
            'overall_score': {}
        }
        
        # Find common tests in both datasets
        common_tests = set(dataset1.keys()) & set(dataset2.keys())
        only_in_dataset1 = set(dataset1.keys()) - set(dataset2.keys())
        only_in_dataset2 = set(dataset2.keys()) - set(dataset1.keys())
        
        comparison_results['metadata']['common_tests'] = list(common_tests)
        comparison_results['metadata']['only_in_dataset1'] = list(only_in_dataset1)
        comparison_results['metadata']['only_in_dataset2'] = list(only_in_dataset2)
        
        # Compare metrics for each common test
        for test_name in common_tests:
            test_result = self._compare_test(
                test_name, 
                dataset1[test_name], 
                dataset2[test_name]
            )
            comparison_results['test_comparisons'][test_name] = test_result
        
        # Calculate overall comparison metrics
        comparison_results['overall_score'] = self._calculate_overall_scores(
            comparison_results['test_comparisons']
        )
        
        # Generate summary of key findings
        comparison_results['metadata']['summary'] = self._generate_summary(
            comparison_results
        )
        
        return comparison_results
    
    def _compare_test(self, 
                     test_name: str, 
                     data1: Dict[str, Any], 
                     data2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare metrics for a single test between two datasets.
        
        Args:
            test_name: Name of the test
            data1: Metrics from first dataset
            data2: Metrics from second dataset
            
        Returns:
            Dictionary with comparison metrics
        """
        test_type = data1.get('test_type', 'unknown')
        if test_type == 'unknown':
            test_type = data2.get('test_type', 'unknown')
        
        channel = data1.get('channel', 'unknown')
        if channel == 'unknown':
            channel = data2.get('channel', 'unknown')
        
        comparison = {
            'test_name': test_name,
            'test_type': test_type,
            'channel': channel,
            'metrics': {},
            'scores': {}
        }
        
        # Get relevant metrics for this test type
        metrics_to_compare = self.comparison_metrics.get(test_type, [])
        
        # Compare each metric
        for metric in metrics_to_compare:
            # Check if both datasets have this metric
            if metric in data1 and metric in data2:
                value1 = data1[metric]
                value2 = data2[metric]
                
                # Skip non-numeric metrics or None values
                if (not isinstance(value1, (int, float)) or 
                    not isinstance(value2, (int, float))):
                    continue
                
                # Calculate the difference and percent change
                diff = value2 - value1
                
                # Avoid division by zero
                if abs(value1) > 1e-6:
                    percent_change = (diff / abs(value1)) * 100.0
                else:
                    percent_change = float('inf') if diff > 0 else float('-inf') if diff < 0 else 0.0
                
                comparison['metrics'][metric] = {
                    'value1': value1,
                    'value2': value2,
                    'difference': diff,
                    'percent_change': percent_change
                }
                
                # Calculate a score for this metric (-100 to 100)
                # Positive means dataset2 is better, negative means dataset1 is better
                score = self._calculate_metric_score(metric, value1, value2)
                comparison['scores'][metric] = score
        
        # Calculate overall test score (average of metric scores)
        if comparison['scores']:
            comparison['overall_score'] = sum(comparison['scores'].values()) / len(comparison['scores'])
        else:
            comparison['overall_score'] = 0.0
        
        return comparison
    
    def _calculate_metric_score(self, 
                               metric: str, 
                               value1: float, 
                               value2: float) -> float:
        """
        Calculate a score for comparison between two metric values.
        
        Positive score means dataset2 is better, negative means dataset1 is better.
        Score range is -100 to 100 with 0 meaning no difference.
        
        Args:
            metric: Name of the metric
            value1: Value from first dataset
            value2: Value from second dataset
            
        Returns:
            Score between -100 and 100
        """
        # Handle zero and near-zero values for division
        if abs(value1) < 1e-6 and abs(value2) < 1e-6:
            return 0.0  # Both values are effectively zero, no difference
        
        # Calculate ratio and limit extreme values
        if abs(value1) < 1e-6:
            ratio = 10.0 if value2 > 0 else -10.0  # Cap at 1000% improvement/degradation
        else:
            ratio = value2 / value1
        
        # For comparison, transform to percent difference
        percent_diff = (ratio - 1.0) * 100.0
        
        # Cap at +/- 100%
        percent_diff = max(min(percent_diff, 100.0), -100.0)
        
        # Invert if lower is better
        if metric in self.lower_is_better:
            percent_diff = -percent_diff
        
        # For metrics neither in higher_is_better nor lower_is_better,
        # just show the change without scoring it as better or worse
        if metric not in self.higher_is_better and metric not in self.lower_is_better:
            return 0.0
        
        return percent_diff
    
    def _calculate_overall_scores(self, 
                                 test_comparisons: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate overall comparison scores across all tests.
        
        Args:
            test_comparisons: Dictionary of test comparisons
            
        Returns:
            Dictionary with overall scores by category
        """
        # Initialize counters
        total_score = 0.0
        score_count = 0
        
        # Group scores by test type
        scores_by_type = {}
        
        for test_name, comparison in test_comparisons.items():
            test_type = comparison.get('test_type', 'unknown')
            
            if test_type not in scores_by_type:
                scores_by_type[test_type] = {
                    'count': 0,
                    'total_score': 0.0,
                    'metrics': {}
                }
            
            # Add to overall and by-type scores
            test_score = comparison.get('overall_score', 0.0)
            total_score += test_score
            score_count += 1
            
            scores_by_type[test_type]['total_score'] += test_score
            scores_by_type[test_type]['count'] += 1
            
            # Track scores by metric
            for metric, score in comparison.get('scores', {}).items():
                if metric not in scores_by_type[test_type]['metrics']:
                    scores_by_type[test_type]['metrics'][metric] = {
                        'count': 0,
                        'total_score': 0.0
                    }
                
                scores_by_type[test_type]['metrics'][metric]['total_score'] += score
                scores_by_type[test_type]['metrics'][metric]['count'] += 1
        
        # Calculate averages
        overall_average = total_score / score_count if score_count > 0 else 0.0
        
        # Calculate averages for each test type
        for test_type, data in scores_by_type.items():
            if data['count'] > 0:
                data['average_score'] = data['total_score'] / data['count']
            else:
                data['average_score'] = 0.0
            
            # Calculate averages for each metric
            for metric, metric_data in data['metrics'].items():
                if metric_data['count'] > 0:
                    metric_data['average_score'] = metric_data['total_score'] / metric_data['count']
                else:
                    metric_data['average_score'] = 0.0
        
        return {
            'overall_average': overall_average,
            'by_test_type': scores_by_type
        }
    
    def _generate_summary(self, 
                         comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of key findings from the comparison.
        
        Args:
            comparison_results: Full comparison results
            
        Returns:
            Dictionary with summary information
        """
        overall_score = comparison_results['overall_score']['overall_average']
        
        summary = {
            'overall_verdict': "",
            'improved_metrics': [],
            'degraded_metrics': [],
            'significant_changes': []
        }
        
        # Overall verdict
        if abs(overall_score) < 5.0:
            summary['overall_verdict'] = "The two datasets show similar overall performance."
        elif overall_score > 0:
            summary['overall_verdict'] = f"{comparison_results['metadata']['dataset2_label']} shows better overall performance than {comparison_results['metadata']['dataset1_label']} (Score: +{overall_score:.1f})."
        else:
            summary['overall_verdict'] = f"{comparison_results['metadata']['dataset1_label']} shows better overall performance than {comparison_results['metadata']['dataset2_label']} (Score: {overall_score:.1f})."
        
        # Find most improved and degraded metrics
        all_metric_changes = []
        
        for test_name, test_comp in comparison_results['test_comparisons'].items():
            for metric, score in test_comp.get('scores', {}).items():
                if abs(score) > 10.0:  # Only include significant changes
                    all_metric_changes.append({
                        'test': test_name,
                        'metric': metric,
                        'score': score,
                        'change': test_comp['metrics'][metric]['percent_change']
                    })
        
        # Sort by score (highest to lowest)
        all_metric_changes.sort(key=lambda x: x['score'], reverse=True)
        
        # Get top 5 improvements
        for change in all_metric_changes[:5]:
            if change['score'] > 10.0:
                summary['improved_metrics'].append(
                    f"{change['metric']} in {change['test']}: {change['score']:.1f}% better"
                )
        
        # Get top 5 degradations (from the end of the list)
        for change in all_metric_changes[-5:]:
            if change['score'] < -10.0:
                summary['degraded_metrics'].append(
                    f"{change['metric']} in {change['test']}: {abs(change['score']):.1f}% worse"
                )
        
        # Find biggest changes overall (regardless of better/worse)
        biggest_changes = sorted(all_metric_changes, key=lambda x: abs(x['score']), reverse=True)
        
        for change in biggest_changes[:5]:
            direction = "better" if change['score'] > 0 else "worse"
            summary['significant_changes'].append(
                f"{change['metric']} in {change['test']}: {abs(change['score']):.1f}% {direction}"
            )
        
        return summary

    def save_comparison_report(self, 
                              comparison_results: Dict[str, Any], 
                              output_dir: str) -> str:
        """
        Save comparison results to files.
        
        Args:
            comparison_results: Comparison results from compare_datasets
            output_dir: Directory to save files
            
        Returns:
            Path to the directory containing the report files
        """
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Save full comparison data as JSON
        json_path = os.path.join(output_dir, "comparison_results.json")
        with open(json_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        # Generate a CSV with the key metrics
        csv_path = os.path.join(output_dir, "comparison_metrics.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                "Test", "Metric", 
                comparison_results['metadata']['dataset1_label'],
                comparison_results['metadata']['dataset2_label'],
                "Difference", "Percent Change", "Score"
            ])
            
            # Write data for each test and metric
            for test_name, test_comp in comparison_results['test_comparisons'].items():
                for metric, data in test_comp.get('metrics', {}).items():
                    writer.writerow([
                        test_name, metric,
                        data['value1'], data['value2'],
                        data['difference'], data['percent_change'],
                        test_comp['scores'].get(metric, 'N/A')
                    ])
        
        # Generate a text summary
        txt_path = os.path.join(output_dir, "comparison_summary.txt")
        with open(txt_path, 'w') as f:
            f.write("===== DATASET COMPARISON SUMMARY =====\n\n")
            
            dataset1 = comparison_results['metadata']['dataset1_label']
            dataset2 = comparison_results['metadata']['dataset2_label']
            
            f.write(f"Comparing {dataset2} against baseline {dataset1}\n\n")
            
            # Write overall verdict
            f.write("OVERALL VERDICT:\n")
            f.write(comparison_results['metadata']['summary']['overall_verdict'] + "\n\n")
            
            # Write improved metrics
            f.write("TOP IMPROVEMENTS:\n")
            if comparison_results['metadata']['summary']['improved_metrics']:
                for metric in comparison_results['metadata']['summary']['improved_metrics']:
                    f.write(f"* {metric}\n")
            else:
                f.write("No significant improvements detected\n")
            f.write("\n")
            
            # Write degraded metrics
            f.write("TOP DEGRADATIONS:\n")
            if comparison_results['metadata']['summary']['degraded_metrics']:
                for metric in comparison_results['metadata']['summary']['degraded_metrics']:
                    f.write(f"* {metric}\n")
            else:
                f.write("No significant degradations detected\n")
            f.write("\n")
            
            # Write by test type summary
            f.write("SCORES BY TEST TYPE:\n")
            for test_type, data in comparison_results['overall_score']['by_test_type'].items():
                f.write(f"* {test_type.capitalize()}: {data['average_score']:.1f}\n")
            f.write("\n")
            
            # Write missing tests
            if comparison_results['metadata']['only_in_dataset1']:
                f.write(f"Tests only in {dataset1}:\n")
                for test in comparison_results['metadata']['only_in_dataset1']:
                    f.write(f"* {test}\n")
                f.write("\n")
            
            if comparison_results['metadata']['only_in_dataset2']:
                f.write(f"Tests only in {dataset2}:\n")
                for test in comparison_results['metadata']['only_in_dataset2']:
                    f.write(f"* {test}\n")
                f.write("\n")
            
            f.write(f"Detailed results saved to {json_path} and {csv_path}\n")
        
        logger.info(f"Comparison report saved to {output_dir}")
        return output_dir


def compare_datasets(data_dir1: str, 
                    data_dir2: str, 
                    output_dir: str,
                    label1: str = None, 
                    label2: str = None) -> Dict[str, Any]:
    """
    Compare two data directories containing test results.
    
    Args:
        data_dir1: Path to first data directory
        data_dir2: Path to second data directory
        output_dir: Output directory for comparison reports
        label1: Label for first dataset (default: derived from dir name)
        label2: Label for second dataset (default: derived from dir name)
        
    Returns:
        Comparison results dictionary
    """
    # Set default labels if not provided
    if label1 is None:
        label1 = os.path.basename(os.path.normpath(data_dir1))
    
    if label2 is None:
        label2 = os.path.basename(os.path.normpath(data_dir2))
    
    # Load datasets from both directories
    # First load raw data from both directories
    try:
        from ..main import load_raw_data_from_directory
        from ..data_processor import DataProcessor
        from .metrics import calculate_test_metrics
        
        # Check if analysis results already exist and load them
        dataset1_metrics = _load_metrics_from_directory(data_dir1)
        dataset2_metrics = _load_metrics_from_directory(data_dir2)
        
        # If metrics not found, calculate them
        if not dataset1_metrics or not dataset2_metrics:
            logger.info("Generating metrics for comparison...")
            
            # Load configuration
            from ..config import FlightConfig
            config = FlightConfig()
            
            # Process dataset 1
            if not dataset1_metrics:
                logger.info(f"Processing dataset from {data_dir1}")
                raw_data1 = load_raw_data_from_directory(data_dir1)
                if raw_data1:
                    processor = DataProcessor(config)
                    processed_data1 = processor.process(raw_data1)
                    dataset1_metrics = calculate_test_metrics(processed_data1)
                else:
                    logger.error(f"No data found in {data_dir1}")
                    return {'error': f"No data found in {data_dir1}"}
            
            # Process dataset 2
            if not dataset2_metrics:
                logger.info(f"Processing dataset from {data_dir2}")
                raw_data2 = load_raw_data_from_directory(data_dir2)
                if raw_data2:
                    processor = DataProcessor(config)
                    processed_data2 = processor.process(raw_data2)
                    dataset2_metrics = calculate_test_metrics(processed_data2)
                else:
                    logger.error(f"No data found in {data_dir2}")
                    return {'error': f"No data found in {data_dir2}"}
        
        # Compare the datasets
        comparator = DataComparator()
        comparison_results = comparator.compare_datasets(
            dataset1_metrics, dataset2_metrics, label1, label2
        )
        
        # Save comparison report
        comparator.save_comparison_report(comparison_results, output_dir)
        
        return comparison_results
        
    except Exception as e:
        logger.error(f"Error comparing datasets: {e}")
        return {'error': str(e)}


def _load_metrics_from_directory(data_dir: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Load precomputed metrics from a data directory if available.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Dictionary of metrics by test name, or None if not found
    """
    # Check for metrics file in the analysis directory
    metrics_path = os.path.join(data_dir, 'analysis', 'advanced_metrics.json')
    
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            logger.info(f"Loaded pre-computed metrics from {metrics_path}")
            return metrics
        except Exception as e:
            logger.warning(f"Error loading metrics from {metrics_path}: {e}")
    
    return None 