import numpy as np

def extract_features(window_data):
    """
    Extract features from an EMG window.
    
    Args:
        window_data (pd.Series): Window of EMG data
    
    Returns:
        dict: Dictionary containing the extracted features
    """
    features = {
        'MAV': calculate_mav(window_data),
        'WL': calculate_wl(window_data),
        'WAMP': calculate_wamp(window_data),
        'MAVS': calculate_mavs(window_data)
    }
    return features

def calculate_mav(signal):
    """
    Calculate Mean Absolute Value (MAV).
    
    Args:
        signal (pd.Series): EMG signal window
    
    Returns:
        float: MAV value
    """
    return np.mean(np.abs(signal))

def calculate_wl(signal):
    """
    Calculate Waveform Length (WL).
    
    Args:
        signal (pd.Series): EMG signal window
    
    Returns:
        float: WL value
    """
    return np.sum(np.abs(np.diff(signal)))

def calculate_wamp(signal, threshold=0.01):
    """
    Calculate Wilson Amplitude (WAMP).
    
    Args:
        signal (pd.Series): EMG signal window
        threshold (float): Threshold for amplitude difference
    
    Returns:
        int: WAMP value
    """
    diff = np.abs(np.diff(signal))
    return np.sum(diff > threshold)

def calculate_mavs(signal):
    """
    Calculate Mean Absolute Value Slope (MAVS).
    
    Args:
        signal (pd.Series): EMG signal window
    
    Returns:
        float: MAVS value
    """
    # Calculate MAV for each segment
    segment_size = len(signal) // 4
    mavs = []
    
    for i in range(0, len(signal) - segment_size, segment_size):
        segment = signal[i:i + segment_size]
        mavs.append(calculate_mav(segment))
    
    # Calculate the slope between consecutive MAV values
    return np.mean(np.diff(mavs)) 