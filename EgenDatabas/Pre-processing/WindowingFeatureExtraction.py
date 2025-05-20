import numpy as np
import pandas as pd
import os
from collections import Counter
import joblib

# ----------- Parameters -----------
window_size = 200        # e.g., 200 ms at 1 kHz
overlap = 100            # 50% overlap
wamp_threshold = 0.02    # WAMP threshold

# Input and output paths
input_folder = "datasets/official"
output_dir = "processed_output"
os.makedirs(output_dir, exist_ok=True)

output_file = f"features_labels_structured_W{window_size}_O{overlap}_WAMPth{int(wamp_threshold*1000)}.pkl"
output_path = os.path.join(output_dir, output_file)

# ----------- Feature Functions -----------

def compute_mav(window):
    return np.mean(np.abs(window))

def compute_wl(window):
    return np.sum(np.abs(np.diff(window)))

def compute_wamp(window, threshold=wamp_threshold):
    return np.sum(np.abs(np.diff(window)) > threshold)

def compute_mavs(window):
    half = len(window) // 2
    mav1 = compute_mav(window[:half])
    mav2 = compute_mav(window[half:])
    return np.abs(mav2 - mav1)

def extract_features(window):
    return [
        compute_mav(window),
        compute_wl(window),
        compute_wamp(window),
        compute_mavs(window)
    ]

def get_majority_label(labels):
    return Counter(labels).most_common(1)[0][0]

# ----------- Signal Processing -----------

def process_file(file_path):
    df = pd.read_csv(file_path, header=None, names=['time', 'voltage', 'label'])
    signal = df['voltage'].values
    labels = df['label'].values

    step = window_size - overlap
    num_windows = (len(signal) - window_size) // step + 1

    X = []
    y = []

    for i in range(num_windows):
        start = i * step
        end = start + window_size
        window = signal[start:end]
        label_window = labels[start:end]

        if len(window) == window_size:
            features = extract_features(window)
            majority_label = get_majority_label(label_window)
            X.append(features)
            y.append(majority_label)

    return np.array(X), np.array(y)

# ----------- Main Loop -----------

records = []
file_counter = 0

for person_folder in os.listdir(input_folder):
    person_path = os.path.join(input_folder, person_folder)
    if not os.path.isdir(person_path):
        continue

    try:
        person_id = int(person_folder)
    except ValueError:
        print(f"Skipping folder {person_folder} (not a person ID)")
        continue

    for fname in os.listdir(person_path):
        if fname.endswith(".csv"):
            file_path = os.path.join(person_path, fname)
            cycle_id = fname.split("-")[0]  # e.g., "0205" from "0205-132514record.csv"

            try:
                X, y = process_file(file_path)
            except Exception as e:
                print(f"⚠️ Error processing {file_path}: {e}")
                continue

            for features, label in zip(X, y):
                records.append({
                    "features": features,
                    "label": label,
                    "person_id": person_id,
                    "cycle_id": cycle_id
                })

            file_counter += 1
            print(f"✅ Processed file #{file_counter}: {fname} ({len(X)} windows)")

# ----------- Save Structured Dataset -----------

if records:
    joblib.dump(records, output_path)
    print(f"\n✅ Saved {len(records)} windows to: {output_path}")
else:
    print("❌ No data was processed.")
