import pandas as pd
import numpy as np
import os
import joblib
import re

# ----------- Parameters -----------
window_size = 64  # samples
overlap = 32      # samples
sampling_rate = 1000  # Hz

# Peak detection window
before_peak_samples = 200  # 0.2s
after_peak_samples = 600   # 0.6s
total_movement_samples = before_peak_samples + after_peak_samples

# Paths
data_folder = r"Wyoflex/3 gester/VOLTAGE DATA"
save_folder = r"Wyoflex/3 gester/DynamicProcessed/4channel"
os.makedirs(save_folder, exist_ok=True)

# Output file
combined_filename = f"features_labels_ALL_4channels_dynamic_O2_W{window_size}.pkl"

# Gesture labels
gesture_labels = {
    "M1": 0,
    "M2": 1,
    "M6": 2
}

# Optional file limit
max_files = None  # e.g., 50 for testing

# ----------- Helper Functions -----------

def load_signal(file_path):
    """Load signal and extract movement segment around peak."""
    data = pd.read_csv(file_path, header=None).T
    signal = data.squeeze()

    if len(signal) < total_movement_samples:
        print(f"⚠️ {file_path} too short.")
        return None

    search_start = before_peak_samples
    search_end = len(signal) - after_peak_samples
    if search_end <= search_start:
        print(f"⚠️ Invalid search range in {file_path}")
        return None

    sub_signal = signal[search_start:search_end]
    peak_relative = np.argmax(np.abs(sub_signal))
    peak_index = search_start + peak_relative

    start = peak_index - before_peak_samples
    end = peak_index + after_peak_samples
    return signal[start:end]

def sliding_window(signal, window_size, overlap):
    starts = np.arange(0, len(signal) - window_size + 1, window_size - overlap)
    return np.array([signal[start:start + window_size] for start in starts])

def extract_features(windows):
    """Extract MAV, WL, ZC, SSC, RMS, Variance, IEMG from windows."""
    mav = np.mean(np.abs(windows), axis=1)
    wl = np.sum(np.abs(np.diff(windows, axis=1)), axis=1)
    zc = np.sum(np.diff(np.sign(windows), axis=1) != 0, axis=1)
    ssc = np.sum(np.diff(np.sign(np.diff(windows, axis=1)), axis=1) != 0, axis=1)
    rms = np.sqrt(np.mean(windows**2, axis=1))
    variance = np.var(windows, axis=1)
    iemg = np.sum(np.abs(windows), axis=1)
    return np.stack((mav, wl, zc, ssc, rms, variance, iemg), axis=1)

# ----------- Combined Dataset Storage -----------

X_all = []
y_all = []
file_counter = 0

# Walk through all files and process matching ones
for root, dirs, files in os.walk(data_folder):
    for file in files:
        subject_match = re.match(r"^(P\d+)", file)
        for gesture, label in gesture_labels.items():
            if (
                subject_match and
                gesture in file and
                "O2" in file and
                "S1" in file  # Only start from S1 files
            ):
                base_filename = file  # ✅ keep full filename, don't delete anything

                # Try to load all 4 channels
                signals = []
                missing_channel = False
                for sensor in ["S1", "S2", "S3", "S4"]:
                    sensor_file = base_filename.replace("S1", sensor)
                    full_path = os.path.join(root, sensor_file)
                    if not os.path.exists(full_path):
                        print(f"⚠️ Missing sensor file: {full_path}")
                        missing_channel = True
                        break
                    signal = load_signal(full_path)
                    if signal is None or len(signal) < window_size:
                        missing_channel = True
                        break
                    signals.append(signal)

                if missing_channel:
                    print(f"⚠️ Skipping {file} (missing one of S1–S4)")
                    continue

                # Now signals = [S1_signal, S2_signal, S3_signal, S4_signal]
                channel_windows = []
                for sig in signals:
                    windows = sliding_window(sig, window_size, overlap)
                    features = extract_features(windows)
                    channel_windows.append(features)

                # Stack channels along new dimension (channels axis)
                channels_stacked = np.stack(channel_windows, axis=2)  # (num_windows, features, channels)

                # Move axes to (num_windows, 4 channels, 7 features)
                channels_stacked = np.moveaxis(channels_stacked, 2, 1)

                X_all.append(channels_stacked)
                y_all.extend([label] * channels_stacked.shape[0])

                file_counter += 1

                # ✅ Add progress print after every file
                print(f"✅ Finished {file_counter} files — Latest: {file} — {channels_stacked.shape[0]} windows extracted")

                if max_files is not None and file_counter >= max_files:
                    break
        if max_files is not None and file_counter >= max_files:
            break
    if max_files is not None and file_counter >= max_files:
        break

# ----------- Save Combined Dataset -----------

if X_all:
    X_all = np.vstack(X_all)
    y_all = np.array(y_all)

    print(f"\n✅ Finished processing all subjects (4 channels)")
    print(f"Total files processed: {file_counter}")
    print(f"Feature matrix shape: {X_all.shape}")
    print(f"Labels vector shape: {y_all.shape}")

    save_path = os.path.join(save_folder, combined_filename)
    joblib.dump((X_all, y_all), save_path)
    print(f"✅ Saved combined dataset to {save_path}")

else:
    print("❌ No data was processed.")
