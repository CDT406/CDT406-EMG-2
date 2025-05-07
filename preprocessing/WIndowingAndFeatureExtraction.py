import pandas as pd
import numpy as np
import os
import joblib

# ----------- Parameters -----------
window_size = 256  # samples
overlap = 128      # samples
sampling_rate = 1000  # Hz

# Your data folder (update this if needed)
data_folder = r"Wyoflex/3 gester/VOLTAGE DATA"

# Folder to save processed features
save_folder = r"Wyoflex/3 gester/processed"
os.makedirs(save_folder, exist_ok=True)

# Mapping gestures to labels
gesture_labels = {
    "M1": 0,
    "M2": 1,
    "M6": 2
}

# Limit number of files for faster testing (set to None for all)
max_files = None  # e.g., 50 for testing, or None for full dataset

# ----------- Helper Functions -----------

def load_signal(file_path):
    """Load and transpose sEMG signal from file, and trim to 5–10 seconds."""
    data = pd.read_csv(file_path, header=None).T
    signal = data.squeeze()

    # Keep only 5s to 10s part (sampling rate = 1000Hz → samples 5000 to 10000)
    signal = signal[5500:8500]

    return signal

def sliding_window(signal, window_size=256, overlap=128):
    """Segment signal into overlapping windows."""
    starts = np.arange(0, len(signal) - window_size + 1, window_size - overlap)
    windows = np.array([signal[start:start+window_size] for start in starts])
    return windows

def fast_extract_features(windows):
    """Fast extraction of MAV, WL, ZC, SSC features for all windows using numpy."""
    mav = np.mean(np.abs(windows), axis=1)
    wl = np.sum(np.abs(np.diff(windows, axis=1)), axis=1)
    zc = np.sum(np.diff(np.sign(windows), axis=1) != 0, axis=1)
    diff1 = np.diff(windows, axis=1)
    ssc = np.sum(np.diff(np.sign(diff1), axis=1) != 0, axis=1)
    features = np.stack((mav, wl, zc, ssc), axis=1)
    return features

# ----------- Main Processing and Saving -----------

X = []  # All feature vectors
y = []  # All labels

file_counter = 0

for root, dirs, files in os.walk(data_folder):
    for file in files:
        for gesture, label in gesture_labels.items():
            if gesture in file and "O2" in file:  # <--- Only use O2 signals
                file_path = os.path.join(root, file)
                print(f"Processing {file_path} with label {label}")
                signal = load_signal(file_path)  # <--- Load only 5–10s
                windows = sliding_window(signal, window_size, overlap)
                features = fast_extract_features(windows)
                
                X.append(features)
                y.extend([label] * len(features))
                
                file_counter += 1
                if max_files is not None and file_counter >= max_files:
                    break
        if max_files is not None and file_counter >= max_files:
            break
    if max_files is not None and file_counter >= max_files:
        break

# Stack everything
if X:
    X = np.vstack(X)
    y = np.array(y)

    print(f"\n✅ Finished dataset preparation!")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels vector shape: {y.shape}")

    # Save the dataset
    save_path = os.path.join(save_folder, "features_labels_5_5to8_5s_O2.pkl")  # <--- updated filename
    joblib.dump((X, y), save_path)
    print(f"✅ Saved features and labels to {save_path}")

else:
    print("❌ No files processed! Please check your data folder and filenames.")
