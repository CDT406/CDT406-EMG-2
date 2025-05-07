import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scipy.signal import butter, filtfilt

# ----------- Paths -----------
model_path = "EgenDatabas/HoldRelease/Modeller/SparadeModeller/lstm_holdrest_model.keras"
file_path = r"D:\Programmering\CDT\CDT406-EMG-2\datasets\official\1\0205-132514record.csv"

# ----------- Parameters -----------
sampling_rate = 5000  # ✅ Corrected to 5 kHz
window_size = 200
overlap = 100
sequence_length = 3
wamp_threshold = 0.02

# ----------- Load Model -----------
model = load_model(model_path)

# ----------- Bandpass Filter -----------
def bandpass_filter(signal, lowcut=20, highcut=450, fs=5000, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# ----------- Feature Extraction -----------
def compute_mav(window): return np.mean(np.abs(window))
def compute_wl(window): return np.sum(np.abs(np.diff(window)))
def compute_wamp(window, threshold=wamp_threshold): return np.sum(np.abs(np.diff(window)) > threshold)
def compute_mavs(window):
    half = len(window) // 2
    return np.abs(compute_mav(window[:half]) - compute_mav(window[half:]))

def extract_features(window):
    return np.array([
        compute_mav(window),
        compute_wl(window),
        compute_wamp(window),
        compute_mavs(window)
    ], dtype=np.float32)

# ----------- Load & Preprocess Signal -----------
df = pd.read_csv(file_path, header=None, names=['time', 'voltage', 'label'])
signal_raw = df['voltage'].values
labels = df['label'].values
time = df['time'].values / 1000  # ms → s

signal = bandpass_filter(signal_raw)

step = window_size - overlap
X_seq = []
pred_indices = []
true_labels = []

for i in range(0, len(signal) - sequence_length * step - window_size + 1, step):
    seq_features = []
    for j in range(sequence_length):
        start = i + j * step
        window = signal[start:start + window_size]
        label_window = labels[start:start + window_size]
        if len(window) == window_size:
            features = extract_features(window)
            seq_features.append(features)
    if len(seq_features) == sequence_length:
        X_seq.append(seq_features)
        pred_indices.append(i + sequence_length * step)
        majority_label = 1 if np.count_nonzero(label_window) > (window_size // 2) else 0
        true_labels.append(majority_label)

X_seq = np.array(X_seq)
true_labels = np.array(true_labels)

# ----------- Predict -----------
y_pred = model.predict(X_seq, verbose=0)
y_pred_labels = np.argmax(y_pred, axis=1)

# ----------- Plot: Signal + Prediction + Ground Truth -----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6), sharex=True)

# -- Top: EMG signal with correctness
ax1.plot(time, signal, label='Filtered EMG', color='black', alpha=0.5)
for idx, true, pred in zip(pred_indices, true_labels, y_pred_labels):
    t = time[idx]
    color = 'green' if pred == true else 'red'
    ax1.axvline(t, color=color, linestyle='--', alpha=0.3)

ax1.set_title("Filtered EMG Signal with Prediction Results (Green = Correct, Red = Incorrect)")
ax1.set_ylabel("Voltage")

# -- Bottom: Ground-truth label timeline
label_line = np.zeros_like(time)
label_line[:len(labels)] = np.where(labels > 0, 1, 0)
ax2.plot(time, label_line, color='blue')
ax2.set_title("Ground Truth Labels (Rest = 0, Hold/Grip/Release = 1)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Label")

plt.tight_layout()
plt.show()
