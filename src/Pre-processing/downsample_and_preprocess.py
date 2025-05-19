import numpy as np
import pandas as pd
from scipy import signal
import os
import joblib
from collections import Counter
from filtering import bandpass_filter
from feature_extraction import extract_features

def downsample_signal(signal_data, original_fs, target_fs):
    """
    Downsample signal from original_fs to target_fs
    """
    # Calculate the downsampling factor
    factor = original_fs // target_fs
    
    # Downsample using scipy's decimate
    return signal.decimate(signal_data, factor)

def process_file(file_path, window_size, overlap, fs, lowcut, highcut, order, wamp_threshold, four_state=False):
    # Read the CSV file
    df = pd.read_csv(file_path, header=None, names=['time', 'voltage', 'label'])
    
    # Extract signal and labels from the DataFrame
    original_signal = df['voltage'].values
    original_labels = df['label'].values
    
    # Downsample signal from 5000Hz to 1000Hz
    downsampled_signal = downsample_signal(original_signal, 5000, 1000)
    
    # Downsample labels (take every 5th label)
    downsampled_labels = original_labels[::5]
    
    # Ensure lengths match
    min_len = min(len(downsampled_signal), len(downsampled_labels))
    downsampled_signal = downsampled_signal[:min_len]
    downsampled_labels = downsampled_labels[:min_len]
    
    # Process the downsampled data
    X, y = process_data(downsampled_signal, downsampled_labels, window_size, overlap, fs, lowcut, highcut, order, wamp_threshold, four_state)
    
    return X, y

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

    # Convert features and labels to numpy arrays, replacing NaNs or infinities
    X = np.nan_to_num(np.array(X), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y)
    
    return X, y

def get_majority_label_2state(labels):
    """Maps labels to binary classes: 0 for rest, 1 for any other state (grip/hold/release)"""
    mapped = [0 if l == 0 else 1 for l in labels]
    return Counter(mapped).most_common(1)[0][0]

def get_majority_label_4state(labels):
    """Maps labels to 4 states: 0=rest, 1=grip, 2=hold, 3=release"""
    return Counter(labels).most_common(1)[0][0]

def run_preprocessing(
    input_folder,
    output_dir,
    window_sizes,  # List of window sizes in milliseconds
    overlap_ratio,  # Overlap as a ratio (e.g., 0.5 for 50% overlap)
    sampling_rate,
    low_cut,
    high_cut,
    filter_order,
    wamp_threshold,
    four_state=False
):
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert window sizes from milliseconds to samples
    window_sizes_samples = [int(ms * sampling_rate / 1000) for ms in window_sizes]
    
    for window_size in window_sizes_samples:
        overlap = int(window_size * overlap_ratio)
        print(f"\n🔁 Processing window size: {window_size} samples ({window_size*1000/sampling_rate:.1f}ms)")
        
        # Add state type to output filename
        state_type = "4state" if four_state else "2state"
        output_file = f"{state_type}_features_labels_W{window_size}_O{overlap}_WAMPth{int(wamp_threshold * 1000)}.pkl"
        output_path = os.path.join(output_dir, output_file)
        
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
                    X, y = process_file(
                        file_path,
                        window_size=window_size,
                        overlap=overlap,
                        fs=sampling_rate,
                        lowcut=low_cut,
                        highcut=high_cut,
                        order=filter_order,
                        wamp_threshold=wamp_threshold,
                        four_state=four_state
                    )
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
                
        if records:
            joblib.dump(records, output_path)
            print(f"\n✅ Saved {len(records)} windows to: {output_path}")
            
            # Print label distribution
            labels = [r['label'] for r in records]
            label_counts = Counter(labels)
            if four_state:
                print("\nLabel distribution (4-state):")
                print("Rest:", label_counts.get(0, 0))
                print("Grip:", label_counts.get(1, 0))
                print("Hold:", label_counts.get(2, 0))
                print("Release:", label_counts.get(3, 0))
            else:
                print("\nLabel distribution (2-state):")
                print("Rest:", label_counts.get(0, 0))
                print("Hold:", label_counts.get(1, 0))
                
            cycle_counts = Counter((r['person_id'], r['cycle_id']) for r in records)
            print("\nCycle counts per person:", cycle_counts)
        else:
            print("❌ No data processed.")

if __name__ == "__main__":
    # ----------- Constants -----------
    input_folder = "datasets/official/unprocessed/relabeled_old_dataset"
    output_dir_2state = "output/ProcessedData/NormalizedData1000Hz/2state"
    output_dir_4state = "output/ProcessedData/NormalizedData1000Hz/4state"
    
    # Test window sizes from 30ms to 300ms
    window_sizes_ms = list(range(30, 301, 30))  # [30, 60, 90, ..., 300] ms
    overlap_ratio = 0.5  # 50% overlap
    
    sampling_rate = 1000  # Hz (downsampled from 5000Hz)
    low_cut = 20
    high_cut = 450
    filter_order = 4
    wamp_threshold = 0.02
    
    # Process 2-state classification
    print("\n--- Processing 2-state classification (rest vs hold) ---")
    run_preprocessing(
        input_folder=input_folder,
        output_dir=output_dir_2state,
        window_sizes=window_sizes_ms,
        overlap_ratio=overlap_ratio,
        sampling_rate=sampling_rate,
        low_cut=low_cut,
        high_cut=high_cut,
        filter_order=filter_order,
        wamp_threshold=wamp_threshold,
        four_state=False
    )
    
    # Process 4-state classification
    print("\n--- Processing 4-state classification (rest/grip/hold/release) ---")
    run_preprocessing(
        input_folder=input_folder,
        output_dir=output_dir_4state,
        window_sizes=window_sizes_ms,
        overlap_ratio=overlap_ratio,
        sampling_rate=sampling_rate,
        low_cut=low_cut,
        high_cut=high_cut,
        filter_order=filter_order,
        wamp_threshold=wamp_threshold,
        four_state=True
    ) 