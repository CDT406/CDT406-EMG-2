# run_preprocess.py

import os
import joblib
from collections import Counter
from windowing import process_file

def run_preprocessing(
    input_folder,
    output_dir,
    window_size,
    overlap,
    sampling_rate,
    low_cut,
    high_cut,
    filter_order,
    wamp_threshold
):
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"features_labels_W{window_size}_O{overlap}_WAMPth{int(wamp_threshold * 1000)}.pkl"
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
                    wamp_threshold=wamp_threshold
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
        cycle_counts = Counter((r['person_id'], r['cycle_id']) for r in records)
        print("Cycle counts per person:", cycle_counts)
    else:
        print("❌ No data processed.")

    return output_path  # useful for downstream code
