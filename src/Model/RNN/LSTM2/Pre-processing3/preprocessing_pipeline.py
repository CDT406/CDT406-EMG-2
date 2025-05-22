import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import json
import sqlite3
import inspect
import toml

from downsample import load_and_downsample, SAMPLING_RATE
from windowing import create_windows, DEFAULT_WINDOW_SIZE, DEFAULT_OVERLAP
from feature_extraction import extract_features, WAMP_THRESHOLD  # Remove filter imports

def get_module_params():
    """Collect parameters from all preprocessing modules"""
    params = {
        # From downsample.py
        'sampling_rate': (SAMPLING_RATE, 'int'),
        
        # From windowing.py
        'window_size': (DEFAULT_WINDOW_SIZE, 'int'),
        'window_overlap': (DEFAULT_OVERLAP, 'float'),
        'sequence_length': (3, 'int'),  # LSTM sequence length
        'windows_count': (3, 'int'),
        
        # Feature configuration
        'wamp_threshold': (WAMP_THRESHOLD, 'float'),
        'features': (','.join(['MAV', 'WL', 'WAMP', 'MAVS']), 'list'),
        'normalization': ('MeanStd', 'str'),
    }
    return params

def calculate_global_window_stats(data_dir):
    """Calculate global mean and std across all EMG windows."""
    all_values = []
    file_pattern = os.path.join(data_dir, '**', '*.csv')
    
    print("Calculating global window statistics...")
    for file_path in glob.glob(file_pattern, recursive=True):
        # Load and downsample
        downsampled_data = load_and_downsample(file_path)
        # Get windows
        windows = create_windows(downsampled_data, DEFAULT_WINDOW_SIZE, DEFAULT_OVERLAP)
        # Collect all voltage values
        for window_data, _ in windows:
            all_values.extend(window_data['voltage'])
    
    # Calculate global stats
    global_mean = np.mean(all_values)
    global_std = np.std(all_values)
    print(f"Global window stats - Mean: {global_mean:.4f}, Std: {global_std:.4f}")
    
    return global_mean, global_std

# Add this function after get_module_params() and before EMGPreprocessor class
def normalize_features(features_dict):
    """Normalize each feature type across all windows and all persons"""
    # Initialize feature stats
    feature_stats = {
        'MAV': {'mean': None, 'std': None},
        'WL': {'mean': None, 'std': None},
        'WAMP': {'mean': None, 'std': None},
        'MAVS': {'mean': None, 'std': None}
    }
    
    # Collect all values for each feature type
    feature_values = {feat: [] for feat in ['MAV', 'WL', 'WAMP', 'MAVS']}
    
    # Gather all values for each feature type
    for person_id, cycles in features_dict.items():
        for cycle_id, windows in cycles.items():
            for window in windows:
                feature_values['MAV'].append(window['MAV'])
                feature_values['WL'].append(window['WL'])
                feature_values['WAMP'].append(window['WAMP'])
                feature_values['MAVS'].append(window['MAVS'])
    
    # Calculate stats and normalize
    for feat_name in ['MAV', 'WL', 'WAMP', 'MAVS']:
        values = np.array(feature_values[feat_name])
        mean = np.mean(values)
        std = np.std(values)
        feature_stats[feat_name]['mean'] = float(mean)
        feature_stats[feat_name]['std'] = float(std)
        
        # Normalize all values for this feature
        for person_id, cycles in features_dict.items():
            for cycle_id, windows in cycles.items():
                for window in windows:
                    window[feat_name] = (window[feat_name] - mean) / std
    
    return features_dict, feature_stats

def save_feature_stats(stats, output_dir):
    """Save feature statistics to SQLite database"""
    db_path = os.path.join(output_dir, 'feature_stats.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feature_stats (
            feature_name TEXT PRIMARY KEY,
            mean REAL,
            std REAL
        )
    ''')
    
    # Insert stats
    for feat_name, values in stats.items():
        cursor.execute('''
            INSERT OR REPLACE INTO feature_stats (feature_name, mean, std)
            VALUES (?, ?, ?)
        ''', (feat_name, values['mean'], values['std']))
    
    conn.commit()
    conn.close()

# Also add this helper function
def convert_to_serializable(data):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(data, (np.int64, np.int32, np.int16, np.int8)):
        return int(data)
    elif isinstance(data, (np.float64, np.float32)):
        return float(data)
    elif isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    return data

def normalize_window(window_data, global_mean, global_std):
    """Normalize window using global statistics."""
    if global_std == 0:
        return window_data['voltage']
    return (window_data['voltage'] - global_mean) / global_std

class EMGPreprocessor:
    def __init__(self, data_dir, output_dir, window_size_ms=200, overlap_percentage=0.5):
        """
        Initialize the EMG preprocessor.
        
        Args:
            data_dir (str): Directory containing the raw EMG data
            output_dir (str): Directory to save processed data
            window_size_ms (int): Window size in milliseconds
            overlap_percentage (float): Window overlap percentage (0-1)
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.window_size_ms = window_size_ms
        self.overlap_percentage = overlap_percentage
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate global stats once
        self.global_mean, self.global_std = calculate_global_window_stats(data_dir)
    
    def process_file(self, file_path):
        """Process a single EMG data file."""
        person_id = int(Path(file_path).parent.name)
        print(f"Processing file: {file_path}")
        
        # Load and downsample the data
        downsampled_data = load_and_downsample(file_path)
        
        # Create windows
        windows = create_windows(
            downsampled_data,
            self.window_size_ms,
            self.overlap_percentage
        )
        
        # Extract features using global normalization
        processed_data = []
        for window_data, window_label in windows:
            # Normalize window using global stats
            normalized_window = normalize_window(window_data, self.global_mean, self.global_std)
            window_data['voltage'] = normalized_window
            features = extract_features(window_data['voltage'])
            
            # Map label 4 to 0
            label = int(window_label)
            if label == 4:
                label = 0
            features['label'] = label
            
            processed_data.append(features)
        
        print(f"  - Extracted features for all windows")
        return person_id, processed_data
    
    def process_all_files(self):
        """
        Process all EMG data files and save the results.
        """
        print("\nStarting EMG data preprocessing...")
        print(f"Window size: {self.window_size_ms}ms")
        print(f"Overlap: {self.overlap_percentage*100}%")
        print(f"Output directory: {self.output_dir}\n")
        
        # Get all CSV files in the data directory
        file_pattern = os.path.join(self.data_dir, '**', '*.csv')
        all_files = glob.glob(file_pattern, recursive=True)
        print(f"Found {len(all_files)} files to process\n")
        
        # Process each file
        all_processed_data = {}
        cycle_counts = {}  # Keep track of cycle numbers for each person
        
        for i, file_path in enumerate(sorted(all_files), 1):
            print(f"\nProcessing file {i}/{len(all_files)}")
            person_id, processed_data = self.process_file(file_path)
            
            # Initialize person's data if not exists
            if person_id not in all_processed_data:
                all_processed_data[person_id] = {}
                cycle_counts[person_id] = 1
            
            # Add the processed data with sequential cycle number
            all_processed_data[person_id][cycle_counts[person_id]] = processed_data
            cycle_counts[person_id] += 1
        
        print("\nNormalizing features...")
        normalized_data, feature_stats = normalize_features(all_processed_data)
        
        print("Saving preprocessing configuration and feature statistics...")
        params = get_module_params()
        params.update({
            'window_size': (self.window_size_ms, 'int'),
            'window_overlap': (self.overlap_percentage, 'float'),
        })
        save_combined_config(self.output_dir, params, feature_stats, (self.global_mean, self.global_std))
        
        print("\nConverting data to JSON format...")
        # Convert normalized data to JSON serializable format
        serializable_data = convert_to_serializable(normalized_data)
        
        # Save the processed data
        output_file = os.path.join(self.output_dir, 'processed_data.json')
        print(f"Saving processed data to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        # Convert metadata to serializable format before saving
        metadata = convert_to_serializable({
            'window_size_ms': self.window_size_ms,
            'overlap_percentage': self.overlap_percentage,
            'sampling_rate': 1000,
            'num_persons': len(all_processed_data),
            'cycles_per_person': {str(person_id): len(cycles) 
                                for person_id, cycles in all_processed_data.items()},
            'features': ['MAV', 'WL', 'WAMP', 'MAVS'],
            'normalization': {
                'type': 'feature-wise',
                'config_location': 'preprocessing_config.toml'
            }
        })
        
        # Save metadata
        metadata_file = os.path.join(self.output_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\nPreprocessing completed successfully!")
        print(f"Processed data for {len(all_processed_data)} persons")
        for person_id, cycles in all_processed_data.items():
            print(f"Person {person_id}: {len(cycles)} cycles")
    
    def save_preprocessing_config(self):
        """Save all preprocessing parameters to SQLite"""
        db_path = os.path.join(self.output_dir, 'preprocessing_config.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS preprocessing_config (
                parameter TEXT PRIMARY KEY,
                value TEXT,
                data_type TEXT
            )
        ''')
        
        # Get parameters from modules
        params = get_module_params()
        
        # Override with instance-specific values
        params.update({
            'window_size': (self.window_size_ms, 'int'),
            'window_overlap': (self.overlap_percentage, 'float'),
        })
        
        # Save parameters
        for param, (value, dtype) in params.items():
            cursor.execute('''
                INSERT OR REPLACE INTO preprocessing_config (parameter, value, data_type)
                VALUES (?, ?, ?)
            ''', (param, str(value), dtype))
        
        conn.commit()
        conn.close()

def save_combined_config(output_dir, preprocessing_params, feature_stats, global_window_stats):
    """Save config including global window statistics."""
    save_dir = os.path.join('src', 'Model', 'RNN', 'LSTM2', 'Saved_models')
    os.makedirs(save_dir, exist_ok=True)
    
    config = {
        'window_normalization': {
            'global_mean': float(global_window_stats[0]),
            'global_std': float(global_window_stats[1])
        },
        'feature_stats': {
            'MAV': [float(feature_stats['MAV']['mean']), float(feature_stats['MAV']['std'])],
            'WL': [float(feature_stats['WL']['mean']), float(feature_stats['WL']['std'])],
            'WAMP': [float(feature_stats['WAMP']['mean']), float(feature_stats['WAMP']['std'])],
            'MAVS': [float(feature_stats['MAVS']['mean']), float(feature_stats['MAVS']['std'])]
        },
        'preprocessing_config': {
            'sampling_rate': preprocessing_params['sampling_rate'][0],
            'window_size': preprocessing_params['window_size'][0],
            'window_overlap': preprocessing_params['window_overlap'][0],
            'sequence_length': preprocessing_params['sequence_length'][0],
            'windows_count': preprocessing_params['windows_count'][0],
            'wamp_threshold': preprocessing_params['wamp_threshold'][0],
            'features': preprocessing_params['features'][0].split(','),
            'normalization': preprocessing_params['normalization'][0]
        }
    }
    
    # Save to TOML file
    toml_path = os.path.join(save_dir, 'preprocessing_config.toml')
    with open(toml_path, 'w') as f:
        toml.dump(config, f)
    
    print(f"Saved configuration to {toml_path}")

def main():
    # Update path to new dataset
    data_dir = 'datasets/OsirisPrime CDT406-Smart-Gripper master data-processed/firm_final_labeled_data'
    output_dir = 'src/Model/RNN/LSTM2/Pre-processing3/processed_data'
    
    # Create and run preprocessor
    preprocessor = EMGPreprocessor(
        data_dir=data_dir,
        output_dir=output_dir,
        window_size_ms=200,  # 200ms windows
        overlap_percentage=0.5  # 50% overlap
    )

    preprocessor.process_all_files()

if __name__ == '__main__':
    main()