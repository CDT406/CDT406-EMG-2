from file_processor import FileProcessor
from config import Config
import time
import numpy as np
import matplotlib.pyplot as plt

def visualize_predictions(signal, predictions, window_starts, window_size, fs=1000):
    """Visualize signal with predictions"""
    t = np.arange(len(signal)) / fs
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1])
    
    # Signal plot
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, signal, 'k-', alpha=0.3, label='Signal')
    
    # Define 2-state colors and names (swapped to match actual predictions)
    label_colors = {
        0: '#3498DB',  # Active - blue (was Rest)
        1: '#7F8C8D',  # Rest - gray (was Active)
    }
    label_names = {
        0: 'Active',   # Swapped
        1: 'Rest'      # Swapped
    }
    
    # Plot predictions
    for i, (start, pred) in enumerate(zip(window_starts, predictions)):
        end = min(start + window_size, len(signal))
        
        # Plot the prediction background
        pred_color = label_colors[pred]
        ax1.axvspan(t[start], t[end-1], alpha=0.2, color=pred_color)
        
        # Add markers for predictions
        mid_point = start + (end - start) // 2
        if mid_point < len(t):
            ax1.plot(t[mid_point], 0, 'g.', markersize=10, alpha=0.7)
    
    # Add legend
    handles = []
    for state in range(len(label_colors)):
        handles.append(plt.plot([], [], color=label_colors[state], 
                              label=f'{label_names[state]}', alpha=0.2)[0])
    
    ax1.set_title('EMG Signal with Predictions (2-state Model)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Voltage')
    ax1.grid(True)
    ax1.legend(handles=handles, loc='upper right')
    
    # Add prediction distribution plot
    ax2 = fig.add_subplot(gs[1])
    predictions = np.array(predictions)
    
    # Calculate prediction distribution
    pred_counts = {}
    for state in range(len(label_colors)):
        count = np.sum(predictions == state)
        pred_counts[label_names[state]] = count
    
    # Create distribution bar plot
    states = list(pred_counts.keys())
    counts = list(pred_counts.values())
    bars = ax2.bar(states, counts)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}', ha='center', va='bottom')
    
    ax2.set_title('Prediction Distribution')
    ax2.set_ylabel('Count')
    ax2.grid(True, axis='y')
    
    # Print prediction distribution
    print("\nPrediction Distribution:")
    for state, count in pred_counts.items():
        print(f"{state}: {count}")
    
    plt.tight_layout()
    plt.savefig('real_time_predictions.png')
    plt.close()

def print_status(processor, last_prediction, predictions):
    """Print processor status and performance metrics"""
    status = processor.get_status()
    
    print("\033[2J\033[H")  # Clear screen and move to top
    print("=== File Processing Status ===")
    print(f"Progress: {status['progress']}")
    print(f"File Position: {status['file_position']}")
    print(f"Model Ready: {status['model_ready']}")
    print(f"Normalization Active: {status['normalization_active']}")
    print("\n=== Performance Metrics ===")
    print(f"Total Samples: {status['total_samples']}")
    print(f"Dropped Samples: {status['dropped_samples']} ({status['drop_rate']:.2f}%)")
    print(f"Avg Processing Time: {status['avg_processing_time_ms']:.2f} ms")
    print(f"Max Processing Time: {status['max_processing_time_ms']:.2f} ms")
    print(f"\nLast Prediction: {last_prediction}")
    
    if predictions:
        # Count occurrences of each prediction
        unique_preds, counts = np.unique(predictions, return_counts=True)
        print("\n=== Prediction Statistics ===")
        for pred, count in zip(unique_preds, counts):
            print(f"Class {pred}: {count} ({(count/len(predictions))*100:.1f}%)")

def main():
    # Initialize components
    config = Config()
    
    # Override sampling rate for 1kHz data
    config.sampling_rate = 1000
    
    # Create processor instance
    processor = FileProcessor(
        config=config,
        model_path="output/SavedModels/RNN/NormalizedData1000Hz/WindowTest50%overlap/2state/2state_features_labels_W180_O90_WAMPth20.tflite",
        file_path="src/inference/output.csv"
    )
    
    last_prediction = None
    last_status_time = time.time()
    status_interval = 1.0  # Update status every second
    predictions = []  # Store all predictions
    window_starts = []  # Store window start positions
    
    try:
        # Start processing
        processor.start()
        print("Started file processing. Press Ctrl+C to stop.")
        
        # Main loop
        while processor.running:
            # Get latest prediction
            prediction = processor.get_latest_prediction()
            
            # Store prediction if we have one
            if prediction is not None:
                last_prediction = prediction
                predictions.append(prediction)
                # Calculate window start position
                window_start = len(predictions) * (processor.window_size - processor.overlap)
                window_starts.append(window_start)
            
            # Update status periodically
            current_time = time.time()
            if current_time - last_status_time >= status_interval:
                print_status(processor, last_prediction, predictions)
                last_status_time = current_time
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Final status update
        print("\nProcessing Complete!")
        print_status(processor, last_prediction, predictions)
        
        # Save predictions to file
        if predictions:
            np.savetxt('predictions.csv', predictions, fmt='%d')
            print("\nPredictions saved to predictions.csv")
            
            # Visualize predictions
            print("\nGenerating visualization...")
            visualize_predictions(
                processor.data,  # Raw signal
                predictions,
                window_starts,
                processor.window_size,
                processor.sampling_rate
            )
            print("Visualization saved to real_time_predictions.png")

if __name__ == "__main__":
    main() 