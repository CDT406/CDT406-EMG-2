import numpy as np
import pandas as pd
from collections import Counter
from filtering import bandpass_filter
from feature_extraction import extract_features

def get_majority_label_2state(labels):
    """Maps labels to binary classes: 0 for rest, 1 for any other state (grip/hold/release)"""
    mapped = [0 if l == 0 else 1 for l in labels]
    return Counter(mapped).most_common(1)[0][0]

def get_majority_label_4state(labels):
    """Maps labels to 4 states: 0=rest, 1=grip, 2=hold, 3=release"""
    return Counter(labels).most_common(1)[0][0]

def process_file(file_path, window_size, overlap, fs, lowcut, highcut, order, wamp_threshold, four_state=False):
    # Read the CSV file
    df = pd.read_csv(file_path, header=None, names=['time', 'voltage', 'label'])
    
    # Extract signal and labels from the DataFrame
    signal = df['voltage'].values
    labels = df['label'].values
    
    # Call the process_data function with appropriate state configuration
    X, y = process_data(signal, labels, window_size, overlap, fs, lowcut, highcut, order, wamp_threshold, four_state)
    
    # Return processed features (X) and labels (y)
    return X, y

def process_data_from_queue(data_queue, window_size, overlap, fs, lowcut, highcut, order, wamp_threshold, four_state=False, stop_event=None):
    """
    Process data as it comes from the queue.
    
    data_queue: A queue object where each item is a tuple (signal, labels).
    window_size: Size of the sliding window for feature extraction.
    overlap: Amount of overlap between consecutive windows.
    fs: Sampling frequency for filtering.
    lowcut: Low cutoff frequency for bandpass filter.
    highcut: High cutoff frequency for bandpass filter.
    order: Order of the bandpass filter.
    wamp_threshold: Threshold for feature extraction.
    four_state: If True, use 4-state classification (rest/grip/hold/release)
    stop_event: An optional event to stop processing (used for clean exit).
    
    Returns:
    Processed features (X) and labels (y) in each iteration.
    """
    while True:
        try:
            # Wait for data to arrive in the queue
            signal, labels = data_queue.get(timeout=1)  # Timeout ensures we can periodically check for stop_event

            # Bandpass filter the signal
            signal = bandpass_filter(signal, lowcut, highcut, fs, order)
            
            # Calculate window step size
            step = window_size - overlap
            num_windows = (len(signal) - window_size) // step + 1

            X, y = [], []
            for i in range(num_windows):
                start = i * step
                end = start + window_size
                window = signal[start:end]
                label_window = labels[start:end]

                if len(window) == window_size:
                    # Normalize the window
                    window = (window - np.mean(window)) / (np.std(window) + 1e-8)
                    
                    # Extract features for this window
                    features = extract_features(window, wamp_threshold=wamp_threshold)
                    majority_label = get_majority_label_4state(label_window) if four_state else get_majority_label_2state(label_window)
                    X.append(np.array(features, dtype=np.float32))
                    y.append(majority_label)

            # Convert lists to arrays, handling NaNs or infinities
            X = np.nan_to_num(np.array(X), nan=0.0, posinf=0.0, neginf=0.0)
            y = np.array(y)
            
            # Return processed features and labels for this batch
            yield X, y
            
            # Check for stop event (if passed) to break the loop
            if stop_event and stop_event.is_set():
                break
        except queue.Empty:
            # Handle case where no data is available in the queue for a given timeout
            if stop_event and stop_event.is_set():
                break
            continue

def process_data(signal, labels, window_size, overlap, fs, lowcut, highcut, order, wamp_threshold, four_state=False):
    # Apply bandpass filter to the signal
    signal = bandpass_filter(signal, lowcut, highcut, fs, order)
    
    # Calculate step size
    step = window_size - overlap
    num_windows = (len(signal) - window_size) // step + 1
    
    X, y = [], []
    for i in range(num_windows):
        start = i * step
        end = start + window_size
        window = signal[start:end]
        label_window = labels[start:end]

        if len(window) == window_size:
            # Normalize the window
            window = (window - np.mean(window)) / (np.std(window) + 1e-8)
            
            # Extract features from the window
            features = extract_features(window, wamp_threshold=wamp_threshold)
            majority_label = get_majority_label_4state(label_window) if four_state else get_majority_label_2state(label_window)
            X.append(np.array(features, dtype=np.float32))
            y.append(majority_label)

    # Convert features and labels to numpy arrays, replacing NaNs and infinities
    X = np.nan_to_num(np.array(X), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y)
    
    return X, y
