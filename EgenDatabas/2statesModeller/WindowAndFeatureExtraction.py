import numpy as np
import pandas as pd
import os
import joblib
import re
from collections import Counter

# ----------- Parameters -----------
window_size = 200        # samples per window
overlap = 100            # 50% overlap
wamp_threshold = 0.02    # WAMP threshold

# ----------- Paths -----------
input_folder = "datasets/augmented_data"  # folder with subfolders per person
output_dir = "processed_output"
os.makedirs(output_dir, exist_ok=True)

# Generate filename based on parameters
output_file = f"features_labels_W{window_size}_O{overlap}_WAMPth{int(wamp_threshold * 1000)}_true_cycles.pkl"
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
    return np.abs(compute_mav(window[:half]) - compute_mav(window[half:]))

def extract_features(window):
    return [
        compute_mav(window),
        compute_wl(window),
        compute_wamp(window),
        compute_mavs(window)
    ]

def get_majority_label(labels):
    mapped = [0 if l == 0 else 1 for l in labels]
    return Counter(mapped).most_common(1)[0][0]

# ----------- Signal Processor -----------

def process_file(file_path):

    df = pd.read_csv(file_path, skiprows=1, header=None, names=['timestamp', 'frequency', 'label'])

    # df = pd.read_csv(file_path, header=None, names=['time', 'voltage', 'label'])
    signal = df['frequency'].values
    labels = df['label'].values

    step = window_size - overlap
    num_windows = (len(signal) - window_size) // step + 1

    X, y = [], []
    if num_windows <= 0:
        print(f"⚠️ {file_path} too short for any windows.")
        return np.array([]), np.array([])

    for i in range(num_windows):
        start = i * step
        end = start + window_size
        window = signal[start:end]
        label_window = labels[start:end]

        if len(window) == window_size:
            features = np.array(extract_features(window), dtype=np.float32)
            majority_label = get_majority_label(label_window)
            X.append(features)
            y.append(majority_label)

    X = np.nan_to_num(np.array(X), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y)
    return X, y

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
        print(f"Skipping non-numeric folder: {person_folder}")
        continue

    csv_files = sorted([f for f in os.listdir(person_path) if f.endswith(".csv")])
    for cycle_id, fname in enumerate(csv_files):
        file_path = os.path.join(person_path, fname)

        try:
            X, y = process_file(file_path)
        except Exception as e:
            print(f"⚠️ Error processing {file_path}: {e}")
            continue

        if len(X) == 0:
            continue

        for features, label in zip(X, y):
            records.append({
                "features": features,
                "label": label,
                "person_id": person_id,
                "cycle_id": cycle_id
            })

        file_counter += 1
        print(f"✅ Processed file #{file_counter}: Person {person_id}, Cycle {cycle_id}, Windows: {len(X)}")

# ----------- Save Output -----------

if records:
    joblib.dump(records, output_path)
    print(f"\n✅ Saved {len(records)} windows to: {output_path}")
    cycle_counts = Counter((r['person_id'], r['cycle_id']) for r in records)
    print("Cycle counts per person:", cycle_counts)
else:
    print("❌ No data processed.")