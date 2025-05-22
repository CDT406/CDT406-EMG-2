import numpy as np
import pandas as pd

SAMPLING_RATE = 1000  # Target sampling rate in Hz

def downsample_signal(signal_data, original_fs=5000, target_fs=SAMPLING_RATE):
    """
    Downsample the EMG signal from original_fs to target_fs.
    
    Args:
        signal_data (pd.DataFrame): DataFrame with columns [time, voltage, label]
        original_fs (int): Original sampling frequency (default: 5000 Hz)
        target_fs (int): Target sampling frequency (default: 1000 Hz)
    
    Returns:
        pd.DataFrame: Downsampled signal data
    """
    # Calculate the downsampling factor
    factor = original_fs // target_fs
    
    # Downsample the signal
    downsampled_data = signal_data.iloc[::factor].copy()
    
    # Reset the time values to be continuous
    downsampled_data.iloc[:, 0] = np.arange(len(downsampled_data)) / target_fs
    
    return downsampled_data

def load_and_downsample(file_path, original_fs=5000, target_fs=SAMPLING_RATE):
    """
    Load a CSV file and downsample the signal.
    
    Args:
        file_path (str): Path to the CSV file
        original_fs (int): Original sampling frequency
        target_fs (int): Target sampling frequency
    
    Returns:
        pd.DataFrame: Downsampled signal data
    """
    # Load the data with headers
    data = pd.read_csv(file_path)
    
    # Rename columns to match expected format
    data.columns = ['time', 'voltage', 'label']
    if 'measurement' in data.columns:
        data = data.rename(columns={'measurement': 'voltage'})
    
    # Downsample
    return downsample_signal(data, original_fs, target_fs)