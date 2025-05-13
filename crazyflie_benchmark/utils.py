"""
Utility functions for the Crazyflie Sweeper package.

Provides helper functions for common operations such as timestamp generation,
path creation, and value clamping.
"""
import os
from datetime import datetime
from typing import Any, Optional, TypeVar, Union, Dict, List

T = TypeVar('T')  # Generic type for comparable values


def clamp(value: T, min_val: T, max_val: T) -> T:
    """
    Clamp a value between a minimum and maximum.
    
    Args:
        value: The value to clamp
        min_val: The minimum allowed value
        max_val: The maximum allowed value
        
    Returns:
        The clamped value
    """
    return max(min(value, max_val), min_val)


def current_timestamp() -> str:
    """
    Get the current timestamp as a formatted string.
    
    Returns:
        String timestamp in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_output_dir(base_name: str = "logs") -> str:
    """
    Create a timestamped output directory for results.
    
    Args:
        base_name: Base name for the directory
        
    Returns:
        Path to the created directory
    """
    # Ensure base directory exists
    os.makedirs(base_name, exist_ok=True)
    
    # Create a timestamped subdirectory
    timestamp = current_timestamp()
    output_dir = os.path.join(base_name, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir


def get_var_from_data(data: Dict[str, Any], possible_names: List[str]) -> Optional[Any]:
    """
    Try to get a variable from data by trying multiple possible variable names.
    
    Args:
        data: The data dictionary to search in
        possible_names: List of possible variable names
        
    Returns:
        The variable value or None if not found
    """
    for name in possible_names:
        if name in data:
            return data[name]
    return None


def calculate_distance(point1: Dict[str, float], point2: Dict[str, float]) -> float:
    """
    Calculate Euclidean distance between two 3D points.
    
    Args:
        point1: Dictionary with x, y, z keys
        point2: Dictionary with x, y, z keys
        
    Returns:
        Euclidean distance between the points
    """
    dx = point1['x'] - point2['x']
    dy = point1['y'] - point2['y']
    dz = point1['z'] - point2['z']
    return (dx*dx + dy*dy + dz*dz) ** 0.5


def horizontal_distance(point1: Dict[str, float], point2: Dict[str, float]) -> float:
    """
    Calculate horizontal (x-y plane) distance between two points.
    
    Args:
        point1: Dictionary with x, y keys
        point2: Dictionary with x, y keys
        
    Returns:
        Horizontal distance between the points
    """
    dx = point1['x'] - point2['x']
    dy = point1['y'] - point2['y']
    return (dx*dx + dy*dy) ** 0.5 