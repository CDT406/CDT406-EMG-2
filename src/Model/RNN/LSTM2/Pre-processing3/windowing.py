import numpy as np
import pandas as pd

def create_windows(signal_data, window_size_ms, overlap_percentage, sampling_rate=1000):
    """
    Create windows from the EMG signal with specified size and overlap.
    
    Args:
        signal_data (pd.DataFrame): DataFrame with columns [time, voltage, label]
        window_size_ms (int): Window size in milliseconds
        overlap_percentage (float): Window overlap percentage (0-1)
        sampling_rate (int): Sampling rate in Hz
    
    Returns:
        list: List of windows, each containing (window_data, window_label)
    """
    # Convert window size from ms to samples
    window_size_samples = int((window_size_ms / 1000) * sampling_rate)
    
    # Calculate step size based on overlap
    step_size = int(window_size_samples * (1 - overlap_percentage))
    
    windows = []
    
    for start_idx in range(0, len(signal_data) - window_size_samples + 1, step_size):
        end_idx = start_idx + window_size_samples
        
        # Extract window
        window_data = signal_data.iloc[start_idx:end_idx].copy()
        
        # Get the most common label in the window
        window_label = window_data['label'].mode().iloc[0]
        
        # Normalize the window
        window_data['voltage'] = normalize_window(window_data['voltage'])
        
        windows.append((window_data, window_label))
    
    return windows

def normalize_window(window_data):
    """
    Normalize a window of EMG data.
    
    Args:
        window_data (pd.Series): Window of EMG data
    
    Returns:
        pd.Series: Normalized window data
    """
    # Remove DC offset
    window_data = window_data - window_data.mean()
    
    # Normalize to unit variance
    if window_data.std() != 0:
        window_data = window_data / window_data.std()
    
    return window_data 