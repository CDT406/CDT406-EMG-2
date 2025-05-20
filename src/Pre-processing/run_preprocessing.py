# run_preprocess.py

import os
import joblib
import json
import sqlite3
from collections import Counter
from windowing import process_file
from feature_normalization import normalize_features  # Assuming this is the correct import

def save_config_to_db(db_path, config_params):
    """Save preprocessing configuration to SQLite database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create config table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS config (
            parameter TEXT PRIMARY KEY,
            value TEXT,
            data_type TEXT
        )
    ''')
    
    # Insert configuration parameters
    params_to_save = {
        'sampling_rate': (config_params['sampling_rate'], 'int'),
        'low_cut': (config_params['low_cut'], 'float'),
        'high_cut': (config_params['high_cut'], 'float'),
        'filter_order': (config_params['filter_order'], 'int'),
        'wamp_threshold': (config_params['wamp_threshold'], 'float'),
        'sequence_length': (3, 'int'),
        'window_size': (config_params['window_size_ms'], 'int'),
        'window_overlap': (config_params['overlap_percentage'], 'float'),
        'features': (','.join(config_params['features']), 'list'),
        'normalization_type': ('feature_wise', 'str')
    }
    
    for param, (value, dtype) in params_to_save.items():
        cursor.execute('''
            INSERT OR REPLACE INTO config (parameter, value, data_type)
            VALUES (?, ?, ?)
        ''', (param, str(value), dtype))
    
    conn.commit()
    conn.close()

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

    # After feature extraction, normalize features
    print("Normalizing features...")
    processed_data, feature_stats = normalize_features(processed_data)
    
    # Save processed data
    with open(output_path, 'w') as f:
        json.dump(processed_data, f, indent=2)
        
    # Save feature stats to SQLite
    stats_db_path = os.path.join(output_dir, 'feature_stats.db')
    conn = sqlite3.connect(stats_db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feature_stats (
            feature_name TEXT PRIMARY KEY,
            mean REAL,
            std REAL
        )
    ''')
    
    for feat_name, stats in feature_stats.items():
        cursor.execute('''
            INSERT OR REPLACE INTO feature_stats (feature_name, mean, std)
            VALUES (?, ?, ?)
        ''', (feat_name, stats['mean'], stats['std']))
    
    conn.commit()
    conn.close()

    # Create config dictionary
    config_params = {
        'sampling_rate': sampling_rate,
        'low_cut': low_cut,
        'high_cut': high_cut,
        'filter_order': filter_order,
        'wamp_threshold': wamp_threshold,
        'window_size_ms': window_size,
        'overlap_percentage': overlap,
        'features': ['MAV', 'WL', 'WAMP', 'MAVS']
    }
    
    # Save config to SQLite
    db_path = os.path.join(output_dir, 'model_config.db')
    save_config_to_db(db_path, config_params)

    return output_path  # useful for downstream code
