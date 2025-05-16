from file_processor import FileProcessor
from config import Config
import time
import numpy as np

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
        model_path=config.model,
        file_path="src/inference/output.csv"
    )
    
    last_prediction = None
    last_status_time = time.time()
    status_interval = 1.0  # Update status every second
    predictions = []  # Store all predictions
    
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

if __name__ == "__main__":
    main() 