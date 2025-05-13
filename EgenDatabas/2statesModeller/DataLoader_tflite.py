# semg_loader_tflite.py
import numpy as np
import os

class SEMGDataLoaderTFLite:
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
                    window = signal[:, start:end].T.astype(np.float32)  # [window_size, channels]
                    self.samples.append((window, label))

    def __len__(self):
        return len(self.samples)

    def get_input(self, idx):
        window, _ = self.samples[idx]
        return np.expand_dims(window, axis=0)  # [1, window_size, channels]

    def get_label(self, idx):
        return self.samples[idx][1]
