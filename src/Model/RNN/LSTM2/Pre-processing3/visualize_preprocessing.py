import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from downsample import load_and_downsample
from windowing import create_windows
from feature_extraction import extract_features

def plot_signal_comparison(file_path, original_fs=5000, target_fs=1000):
    """
    Plot original and downsampled signals for comparison.
    """
    # Load original data
    original_data = pd.read_csv(file_path, header=None)
    original_data.columns = ['time', 'voltage', 'label']
    
    # Get downsampled data
    downsampled_data = load_and_downsample(file_path, original_fs, target_fs)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot original signal
    ax1.plot(original_data['time'], original_data['voltage'], 'b-', label='Original Signal')
    ax1.set_title('Original Signal (5kHz)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Voltage')
    ax1.grid(True)
    ax1.legend()
    
    # Plot downsampled signal
    ax2.plot(downsampled_data['time'], downsampled_data['voltage'], 'r-', label='Downsampled Signal')
    ax2.set_title('Downsampled Signal (1kHz)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Voltage')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    return fig

def plot_windows_and_features(file_path, window_size_ms=200, overlap_percentage=0.5):
    """
    Plot windows and their extracted features.
    """
    # Load and downsample data
    downsampled_data = load_and_downsample(file_path)
    
    # Create windows
    windows = create_windows(downsampled_data, window_size_ms, overlap_percentage)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot signal with window boundaries
    ax1.plot(downsampled_data['time'], downsampled_data['voltage'], 'b-', label='Signal')
    
    # Plot window boundaries and labels
    colors = ['r', 'g', 'b', 'y']  # Different colors for different labels
    for i, (window_data, window_label) in enumerate(windows):
        start_time = window_data['time'].iloc[0]
        end_time = window_data['time'].iloc[-1]
        ax1.axvspan(start_time, end_time, alpha=0.2, color=colors[int(window_label)])
        if i % 5 == 0:  # Label every 5th window to avoid overcrowding
            ax1.text((start_time + end_time)/2, ax1.get_ylim()[1], f'L:{int(window_label)}',
                    horizontalalignment='center')
    
    ax1.set_title('Signal with Windows and Labels')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Voltage')
    ax1.grid(True)
    ax1.legend()
    
    # Plot features for each window
    features = []
    for window_data, _ in windows:
        feat = extract_features(window_data['voltage'])
        features.append(feat)
    
    # Convert features to DataFrame for easier plotting
    features_df = pd.DataFrame(features)
    
    # Plot features
    for feature in ['MAV', 'WL', 'WAMP', 'MAVS']:
        ax2.plot(features_df[feature], label=feature)
    
    ax2.set_title('Extracted Features')
    ax2.set_xlabel('Window Index')
    ax2.set_ylabel('Feature Value')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    return fig

def main():
    # Define paths
    data_dir = 'datasets/official/unprocessed/relabeled_old_dataset'
    output_dir = 'src/Model/RNN/LSTM2/Pre-processing3/visualization_output'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get first file from each person's directory
    for person_dir in sorted(Path(data_dir).glob('*')):
        if person_dir.is_dir():
            person_id = person_dir.name
            print(f"\nProcessing person {person_id}")
            
            # Get first file for this person
            files = list(person_dir.glob('*.csv'))
            if not files:
                continue
                
            file_path = str(files[0])
            print(f"Visualizing file: {file_path}")
            
            # Plot signal comparison
            fig1 = plot_signal_comparison(file_path)
            fig1.savefig(os.path.join(output_dir, f'person_{person_id}_signal_comparison.png'))
            plt.close(fig1)
            
            # Plot windows and features
            fig2 = plot_windows_and_features(file_path)
            fig2.savefig(os.path.join(output_dir, f'person_{person_id}_windows_and_features.png'))
            plt.close(fig2)
            
            print(f"Saved visualizations for person {person_id}")

if __name__ == '__main__':
    main() 