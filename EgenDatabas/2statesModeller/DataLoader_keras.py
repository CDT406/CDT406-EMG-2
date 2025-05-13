# semg_loader_keras.py
import numpy as np
import os

class SEMGDataLoaderKeras:
    def __init__(self, data_dir, window_size=200, step_size=3):
        self.data_dir = data_dir
        self.window_size = window_size
        self.step_size = step_size
        self.samples = []
        self._load_data()

    def _load_data(self):
        for file in os.listdir(self.data_dir):
            if file.endswith('.npy'):
                label = int(file.split('_')[0])
                signal = np.load(os.path.join(self.data_dir, file))  # [channels, time]
                num_windows = (signal.shape[1] - self.window_size) // self.step_size + 1
                for i in range(num_windows):
                    start = i * self.step_size
                    end = start + self.window_size
                    window = signal[:, start:end].T  # Transpose: [window_size, channels]
                    self.samples.append((window, label))

    def __len__(self):
        return len(self.samples)

    def get_batch(self, batch_size, idx):
        batch = self.samples[idx * batch_size : (idx + 1) * batch_size]
        X = np.array([x[0] for x in batch], dtype=np.float32)  # [B, window_size, channels]
        y = np.array([x[1] for x in batch], dtype=np.int32)
        return X, y
