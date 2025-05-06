import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

# ----------- Parameters (must match training) -----------
window_size = 200
overlap = 100
wamp_threshold = 0.02
sequence_length = 3

# ----------- Feature Functions -----------
def compute_mav(window):
    return np.mean(np.abs(window))

def compute_wl(window):
    return np.sum(np.abs(np.diff(window)))

def compute_wamp(window, threshold=wamp_threshold):
    return np.sum(np.abs(np.diff(window)) > threshold)

def compute_mavs(window):
    half = len(window) // 2
    return np.abs(compute_mav(window[:half]) - compute_mav(window[half:]))

def extract_features(window):
    return [
        compute_mav(window),
        compute_wl(window),
        compute_wamp(window),
        compute_mavs(window)
    ]

# ----------- Load Model -----------
model = keras.models.load_model("model.keras")
print("✅ Model loaded.")

# ----------- Feature Extraction -----------
def extract_features_from_csv(file_path):
    df = pd.read_csv(file_path, skiprows=1, header=None, names=['timestamp', 'frequency', 'label'])
    signal = df['frequency'].values
    step = window_size - overlap
    num_windows = (len(signal) - window_size) // step + 1

    X = []
    window_positions = []  # Save start indices for plotting
    for i in range(num_windows):
        start = i * step
        end = start + window_size
        window = signal[start:end]

        if len(window) == window_size:
            features = np.array(extract_features(window), dtype=np.float32)
            X.append(features)
            window_positions.append((start, end))

    X = np.nan_to_num(np.array(X), nan=0.0, posinf=0.0, neginf=0.0)
    return signal, X, window_positions

# ----------- Sequence Maker -----------
def make_sequences(X, window_positions, seq_length=3):
    sequences = []
    sequence_positions = []
    for i in range(len(X) - seq_length + 1):
        seq = X[i:i+seq_length]
        sequences.append(seq)
        sequence_positions.append(window_positions[i + seq_length // 2])  # Use center window for position
    return np.array(sequences, dtype=np.float32), sequence_positions

# ----------- Run Inference and Plot -----------
file_path = "datasets/augmented_data/9/aug_1.csv"
signal, X_infer, window_positions = extract_features_from_csv(file_path)

if len(X_infer) < sequence_length:
    print(f"⚠️ Not enough data to form sequences.")
else:
    X_seq, seq_positions = make_sequences(X_infer, window_positions, sequence_length)
    predictions = model.predict(X_seq)
    predicted_classes = predictions.argmax(axis=-1)

    # ----------- Plotting -----------
    plt.figure(figsize=(15, 5))
    plt.plot(signal, color='gray', linewidth=1, label='sEMG Signal')

    for (start, end), cls in zip(seq_positions, predicted_classes):
        color = 'red' if cls == 0 else 'green'
        plt.axvspan(start, end, facecolor=color, alpha=0.3)

    plt.title("sEMG Signal with Predicted Classes")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()
