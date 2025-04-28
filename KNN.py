import pandas as pd
import numpy as np
import os

# ----------- Parameters -----------
window_size = 256  # samples
overlap = 128      # samples
sampling_rate = 1000  # Hz

# Your data folder
data_folder = r"Wyoflex/3 gester/VOLTAGE DATA"

# Mapping gestures to labels
gesture_labels = {
    "M1": 0,
    "M2": 1,
    "M6": 2
}

# ----------- Helper Functions -----------

def load_signal(file_path):
    """ Load and transpose sEMG signal from file. """
    data = pd.read_csv(file_path, header=None).T
    return data.squeeze()

def sliding_window(signal, window_size=256, overlap=128):
    """ Segment signal into overlapping windows. """
    windows = []
    start = 0
    while start + window_size <= len(signal):
        windows.append(signal[start:start+window_size])
        start += window_size - overlap
    return np.array(windows)

def extract_features(window):
    """ Extract MAV, WL, ZC, SSC features from a window. """
    mav = np.mean(np.abs(window))  # Mean Absolute Value
    wl = np.sum(np.abs(np.diff(window)))  # Waveform Length
    zc = np.sum(np.diff(np.sign(window)) != 0)  # Zero Crossing
    ssc = np.sum(np.diff(np.sign(np.diff(window))) != 0)  # Slope Sign Change
    return [mav, wl, zc, ssc]

def extract_features_all(windows):
    """ Extract features from all windows. """
    features = []
    for window in windows:
        features.append(extract_features(window))
    return np.array(features)

# ----------- Main Dataset Building -----------

X = []  # All feature vectors
y = []  # All labels

# Walk through all files
for root, dirs, files in os.walk(data_folder):
    for file in files:
        for gesture, label in gesture_labels.items():
            if gesture in file:
                file_path = os.path.join(root, file)
                print(f"Processing {file_path} with label {label}")
                signal = load_signal(file_path)
                windows = sliding_window(signal, window_size, overlap)
                features = extract_features_all(windows)
                
                X.append(features)
                y.extend([label] * len(features))  # Repeat label for each window
                break  # Found matching gesture, no need to check other labels

# Stack everything
if X:  # If there is at least one sample
    X = np.vstack(X)
    y = np.array(y)

    print(f"\n✅ Finished dataset preparation!")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels vector shape: {y.shape}")
else:
    print("❌ No files processed! Please check your data folder and filenames.")
