import numpy as np
from preprocessing_pipeline import EMGPreprocessor

def batch_preprocess(window_sizes):
    """Run preprocessing for multiple window sizes."""
    base_data_dir = 'datasets/OsirisPrime CDT406-Smart-Gripper master data-processed/firm_final_labeled_data'
    
    for window_size in window_sizes:
        print(f"\nProcessing with window size: {window_size}ms")
        
        # Create output directory for this window size
        output_dir = f'src/Model/RNN/LSTM2/Pre-processing3/processed_data_{window_size}ms'
        model_save_dir = f'src/Model/RNN/LSTM2/Saved_models_{window_size}ms'
        
        # Create and run preprocessor
        preprocessor = EMGPreprocessor(
            data_dir=base_data_dir,
            output_dir=output_dir,
            window_size_ms=window_size,
            overlap_percentage=0.5
        )
        
        preprocessor.process_all_files()

if __name__ == '__main__':
    # Define window sizes to test (30ms to 300ms)
    window_sizes = np.arange(30, 301, 30)  # [30, 60, 90, ..., 300]
    batch_preprocess(window_sizes)