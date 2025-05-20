import pandas as pd
import numpy as np
import os
import joblib
import re

# ----------- Parameters -----------
window_size = 256  # samples
overlap = 128      # samples
sampling_rate = 1000  # Hz

# Paths
data_folder = r"Wyoflex/3 gester/VOLTAGE DATA"
save_folder = r"Wyoflex/3 gester/FixedSegmentProcessed/4channel_byfile"
os.makedirs(save_folder, exist_ok=True)

# Output filename
save_path = os.path.join(save_folder, "features_by_file_5to9s_W256.pkl")

# Gesture labels
gesture_labels = {
    "M1": 0,
    "M2": 1,
    "M6": 2
}

# Optional limit for testing
max_files = None

# ----------- Helper Functions -----------

def load_signal(file_path):
    """Load sEMG signal and extract 5–9s segment (samples 5000 to 9000)."""
    data = pd.read_csv(file_path, header=None).T
    signal = data.squeeze()

    if len(signal) < 9000:
        print(f"⚠️ {file_path} too short for 9 seconds.")
        return None

    return signal[5000:9000]  # 5 to 9 seconds

def sliding_window(signal, window_size, overlap):
    starts = np.arange(0, len(signal) - window_size + 1, window_size - overlap)
    return np.array([signal[start:start + window_size] for start in starts])

def extract_features(windows):
    """Extract 7 time-domain features from each window."""
    mav = np.mean(np.abs(windows), axis=1)
    wl = np.sum(np.abs(np.diff(windows, axis=1)), axis=1)
    zc = np.sum(np.diff(np.sign(windows), axis=1) != 0, axis=1)
    ssc = np.sum(np.diff(np.sign(np.diff(windows, axis=1)), axis=1) != 0, axis=1)
    rms = np.sqrt(np.mean(windows**2, axis=1))
    var = np.var(windows, axis=1)
    iemg = np.sum(np.abs(windows), axis=1)
    return np.stack((mav, wl, zc, ssc, rms, var, iemg), axis=1)

# ----------- Main Processing -----------

data_by_file = {}
file_counter = 0

for root, dirs, files in os.walk(data_folder):
    for file in files:
        subject_match = re.match(r"^(P\d+)", file)
        for gesture, label in gesture_labels.items():
            if (
                subject_match and
                gesture in file and
                "O2" in file and
                "S1" in file
            ):
                base_filename = file
                signals = []
                missing = False

                for sensor in ["S1", "S2", "S3", "S4"]:
                    sensor_file = base_filename.replace("S1", sensor)
                    full_path = os.path.join(root, sensor_file)
                    if not os.path.exists(full_path):
                        print(f"⚠️ Missing sensor file: {full_path}")
                        missing = True
                        break
                    signal = load_signal(full_path)
                    if signal is None or len(signal) < window_size:
                        missing = True
                        break
                    signals.append(signal)

                if missing:
                    print(f"⚠️ Skipping {file} (incomplete or short channels)")
                    continue

                # Extract features for each channel
                channel_windows = []
                for sig in signals:
                    windows = sliding_window(sig, window_size, overlap)
                    features = extract_features(windows)
                    channel_windows.append(features)

                # Stack: (num_windows, 4, 7)
                features_stacked = np.stack(channel_windows, axis=2)
                features_stacked = np.moveaxis(features_stacked, 2, 1)

                label_array = np.full(features_stacked.shape[0], label)
                data_by_file[base_filename] = (features_stacked, label_array)
                file_counter += 1
                print(f"✅ Processed {file_counter}: {base_filename} => {features_stacked.shape[0]} windows")

                if max_files and file_counter >= max_files:
                    break
        if max_files and file_counter >= max_files:
            break
    if max_files and file_counter >= max_files:
        break

# ----------- Save Final Dataset -----------

if data_by_file:
    joblib.dump(data_by_file, save_path)
    print(f"\n✅ Saved dataset (by file) to: {save_path}")
    print(f"Total files processed: {file_counter}")
else:
    print("❌ No data was processed.") 
