import numpy as np
import pandas as pd
from collections import Counter
from filtering import bandpass_filter
from feature_extraction import extract_features

def get_majority_label(labels):
    mapped = [0 if l == 0 else 1 for l in labels]
    return Counter(mapped).most_common(1)[0][0]

def process_file(file_path, window_size, overlap, fs, lowcut, highcut, order, wamp_threshold):
    df = pd.read_csv(file_path, header=None, names=['time', 'voltage', 'label'])
    signal = df['voltage'].values
    labels = df['label'].values

    signal = bandpass_filter(signal, lowcut, highcut, fs, order)
    step = window_size - overlap
    num_windows = (len(signal) - window_size) // step + 1

    X, y = [], []
    for i in range(num_windows):
        start = i * step
        end = start + window_size
        window = signal[start:end]
        label_window = labels[start:end]

        if len(window) == window_size:
            features = extract_features(window, wamp_threshold=wamp_threshold)
            majority_label = get_majority_label(label_window)
            X.append(np.array(features, dtype=np.float32))
            y.append(majority_label)

    X = np.nan_to_num(np.array(X), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y)
    return X, y
