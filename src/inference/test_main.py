from test_processor import TestProcessor
from config import Config
import time
import signal
import sys
import os
import numpy as np

def main():
    # Initialize configuration
    config = Config()
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create processor with test data
    processor = TestProcessor(
        config,
        os.path.join(script_dir, config.model),
        os.path.join(script_dir, 'output.csv')
    )
    
    def signal_handler(sig, frame):
        print("\nStopping real-time processing...")
        processor.stop()
        sys.exit(0)
    
    # Register Ctrl+C handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start processing
    print("Started processing test data...")
    processor.start()
    
    # Collect all predictions and features
    predictions = []
    features_list = []
    sequences = []  # Store feature sequences used for predictions
    raw_outputs = []  # Store raw model outputs
    
    # Main loop to display status and collect predictions
    try:
        while not processor.is_finished():
            status = processor.get_status()
            
            # Get latest prediction and features from queue if available
            try:
                prediction, raw_output, feature_sequence = processor.get_latest_prediction_with_debug()
                if prediction is not None:
                    predictions.append(prediction)
                    raw_outputs.append(raw_output)
                    sequences.append(feature_sequence)
                
                # Get the latest features
                features = processor.get_latest_features()
                if features is not None:
                    features_list.append(features)
            except:
                pass
            
            # Update status every second
            if status['total_samples'] % 1000 == 0:
                print(f"Processed {status['total_samples']} samples...")
            
            time.sleep(0.001)  # Small sleep to prevent CPU hogging
            
        # Convert features list to numpy array for analysis
        features_array = np.array(features_list)
        sequences_array = np.array(sequences)
        raw_outputs_array = np.array(raw_outputs)
        
        # Print final results
        print("\n=== Processing Complete ===")
        print(f"Total samples processed: {status['total_samples']}")
        print(f"Number of predictions: {len(predictions)}")
        print(f"Unique predictions: {sorted(set(predictions))}")
        print(f"Average processing time: {status['avg_processing_time_ms']:.2f} ms")
        print(f"Maximum processing time: {status['max_processing_time_ms']:.2f} ms")
        
        # Print feature statistics
        print("\n=== Feature Statistics ===")
        feature_names = ['MAV', 'WL', 'WAMP', 'MAVS']
        for i, name in enumerate(feature_names):
            feature_values = features_array[:, i]
            print(f"\n{name}:")
            print(f"  Min: {np.min(feature_values):.4f}")
            print(f"  Max: {np.max(feature_values):.4f}")
            print(f"  Mean: {np.mean(feature_values):.4f}")
            print(f"  Std: {np.std(feature_values):.4f}")
        
        # Print model output statistics
        print("\n=== Model Output Statistics ===")
        print(f"Raw output shape: {raw_outputs_array.shape}")
        print("\nRaw output statistics per class:")
        for i in range(raw_outputs_array.shape[1]):
            values = raw_outputs_array[:, i]
            print(f"Class {i}:")
            print(f"  Min: {np.min(values):.4f}")
            print(f"  Max: {np.max(values):.4f}")
            print(f"  Mean: {np.mean(values):.4f}")
            print(f"  Std: {np.std(values):.4f}")
        
        # Print sequence statistics
        print("\n=== Sequence Statistics ===")
        print(f"Number of sequences: {len(sequences)}")
        print(f"Sequence shape: {sequences_array.shape}")
        print("\nFeature ranges across all sequences:")
        for i, name in enumerate(feature_names):
            values = sequences_array[:, :, i].flatten()
            print(f"\n{name}:")
            print(f"  Min: {np.min(values):.4f}")
            print(f"  Max: {np.max(values):.4f}")
            print(f"  Mean: {np.mean(values):.4f}")
            print(f"  Std: {np.std(values):.4f}")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted...")
        processor.stop()

if __name__ == "__main__":
    main() 