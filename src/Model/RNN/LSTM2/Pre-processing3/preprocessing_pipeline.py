import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import json

from downsample import load_and_downsample
from windowing import create_windows
from feature_extraction import extract_features

def convert_to_serializable(obj):
    """
    Convert NumPy types to Python native types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

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
    
    def process_file(self, file_path):
        """
        Process a single EMG data file.
        
        Args:
            file_path (str): Path to the EMG data file
        
        Returns:
            tuple: (person_id, processed_data)
        """
        # Extract person_id from the folder name
        person_id = int(Path(file_path).parent.name)
        
        print(f"Processing file: {file_path}")
        
        # Load and downsample the data
        downsampled_data = load_and_downsample(file_path)
        print(f"  - Downsampled from {len(downsampled_data)} samples")
        
        # Create windows
        windows = create_windows(
            downsampled_data,
            self.window_size_ms,
            self.overlap_percentage
        )
        print(f"  - Created {len(windows)} windows")
        
        # Extract features for each window
        processed_data = []
        for window_data, window_label in windows:
            features = extract_features(window_data['voltage'])
            features['label'] = int(window_label)  # Convert label to Python int
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
        
        print("\nConverting data to JSON format...")
        # Convert data to JSON serializable format
        serializable_data = convert_to_serializable(all_processed_data)
        
        # Save the processed data
        output_file = os.path.join(self.output_dir, 'processed_data.json')
        print(f"Saving processed data to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        # Save metadata
        metadata = {
            'window_size_ms': self.window_size_ms,
            'overlap_percentage': self.overlap_percentage,
            'sampling_rate': 1000,  # After downsampling
            'num_persons': len(all_processed_data),
            'cycles_per_person': {str(person_id): len(cycles) 
                                for person_id, cycles in all_processed_data.items()},
            'features': ['MAV', 'WL', 'WAMP', 'MAVS']
        }
        
        metadata_file = os.path.join(self.output_dir, 'metadata.json')
        print(f"Saving metadata to {metadata_file}...")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\nPreprocessing completed successfully!")
        print(f"Processed data for {len(all_processed_data)} persons")
        for person_id, cycles in all_processed_data.items():
            print(f"Person {person_id}: {len(cycles)} cycles")

def main():
    # Define paths
    data_dir = 'datasets/official/unprocessed/relabeled_old_dataset'
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