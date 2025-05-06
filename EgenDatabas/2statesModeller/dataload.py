import numpy as np
import pandas as pd
import glob, os


class DataLoader:

    # Parameters
    window_size = 200
    overlap = 100
    wamp_threshold = 0.02
    sequence_length = 3
    

    def __init__(self, window_size=200, overlap=100, wamp_threshold=0.02, sequence_length=3):
        self.window_size = window_size
        self.overlap = overlap
        self.wamp_threshold = wamp_threshold
        self.sequence_length = sequence_length


    # ----------- Feature Functions -----------
    def compute_mav(self, window):
        return np.mean(np.abs(window))


    def compute_wl(self, window):
        return np.sum(np.abs(np.diff(window)))


    def compute_wamp(self, window, threshold=wamp_threshold):
        return np.sum(np.abs(np.diff(window)) > threshold)


    def compute_mavs(self, window):
        half = len(window) // 2
        return np.abs(self.compute_mav(window[:half]) - self.compute_mav(window[half:]))


    def extract_features(self, window):
        return [
            self.compute_mav(window),
            self.compute_wl(window),
            self.compute_wamp(window),
            self.compute_mavs(window)
        ]
        
        
    def extract_features_from_csv(self, file_path):
        df = pd.read_csv(file_path, skiprows=1, header=None, names=['timestamp', 'frequency', 'label'])
        signal = df['frequency'].values
        step = self.window_size - self.overlap
        num_windows = (len(signal) - self.window_size) // step + 1

        X = []
        window_positions = []  # Save start indices for plotting
        for i in range(num_windows):
            start = i * step
            end = start + self.window_size
            window = signal[start:end]

            if len(window) == self.window_size:
                features = np.array(self.extract_features(window), dtype=np.float32)
                X.append(features)
                window_positions.append((start, end))

        X = np.nan_to_num(np.array(X), nan=0.0, posinf=0.0, neginf=0.0)
        return signal, X, window_positions


    def make_sequences(self, X, window_positions, seq_length=3):
        sequences = []
        sequence_positions = []
        for i in range(len(X) - seq_length + 1):
            seq = X[i:i+seq_length]
            sequences.append(seq)
            sequence_positions.append(window_positions[i + seq_length // 2])  # Use center window for position
        return np.array(sequences, dtype=np.float32), sequence_positions


    def get_data_file_paths(self):
        data_dir_path = "./datasets/augmented_data/*/"
        
        data_type = 'aug'
        emg_data = glob.glob(os.path.join(data_dir_path, f'{data_type}*.csv'))
        
        return emg_data


    def get_data(self, sequence_length=3):
        file_path = "datasets/augmented_data/9/aug_1.csv"
        signal, X_infer, window_positions = self.extract_features_from_csv(file_path)
        X_seq, seq_positions = self.make_sequences(X_infer, window_positions, sequence_length)
        
        return X_seq, seq_positions
    
    
if __name__ == "__main__":
    data_loader = DataLoader()
    emgs = data_loader.get_data_file_paths()
    print("EMG Data Files:", emgs)