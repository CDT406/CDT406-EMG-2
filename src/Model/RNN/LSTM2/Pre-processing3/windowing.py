import numpy as np
import pandas as pd

# Default parameters
DEFAULT_WINDOW_SIZE = 200  # Window size in milliseconds
DEFAULT_OVERLAP = 0.5     # 50% overlap

def create_windows(data, window_size_ms=DEFAULT_WINDOW_SIZE, overlap_percentage=DEFAULT_OVERLAP):
    """
    Create overlapping windows from EMG data.
    
    Args:
        data (pd.DataFrame): DataFrame with 'voltage' and 'label' columns
        window_size_ms (int): Window size in milliseconds
        overlap_percentage (float): Overlap between windows (0-1)
    
    Returns:
        list: List of tuples (window_data, window_label)
    """
    # Calculate window parameters
    samples_per_window = int(window_size_ms)  # at 1000Hz, 1 sample = 1ms
    step_size = int(samples_per_window * (1 - overlap_percentage))
    
    windows = []
    for start in range(0, len(data) - samples_per_window + 1, step_size):
        end = start + samples_per_window
        window_data = data.iloc[start:end].copy()
        
        # Get most common label in window
        window_label = window_data['label'].mode().iloc[0]
        
        windows.append((window_data, window_label))
    
    return windows