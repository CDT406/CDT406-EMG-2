import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ----------- Parameters -----------
signal_path = "datasets/official/1/0205-132514record.csv"  # ⬅ update path
fs = 5000  # Hz
lowcut = 20
highcut = 500
order = 4

label_colors = {
    0: '#7F8C8D',  # Rest – cool gray
    1: '#E67E22',  # Grip – orange
    2: '#3498DB',  # Hold – blue
    3: '#E74C3C'   # Release – red/pink
}
label_names = {
    0: 'Rest',
    1: 'Grip',
    2: 'Hold',
    3: 'Release'
}

# ----------- Bandpass Filter -----------
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# ----------- Load Data -----------
df = pd.read_csv(signal_path, header=None, names=['time', 'voltage', 'label'])
signal_raw = df['voltage'].values
labels = df['label'].values
t = np.arange(len(signal_raw)) / fs

# ----------- Filter Signal -----------
signal_filtered = bandpass_filter(signal_raw, lowcut, highcut, fs, order)

# ----------- Plot Raw and Filtered Signals (with colored labels) -----------

fig, axs = plt.subplots(2, 1, figsize=(15, 6), sharex=True)

# Helper function to plot segments
def plot_labeled_segments(ax, t, signal, labels, title):
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[start] or i == len(labels) - 1:
            label = labels[start]
            end = i if labels[i] != label else i + 1
            ax.plot(t[start:end], signal[start:end],
                    color=label_colors.get(label, 'black'),
                    label=label_names.get(label, str(label)))
            start = i
    ax.set_title(title)
    ax.set_ylabel("Voltage")
    handles, labels_ = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))  # deduplicate
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")
    ax.grid(True)

# Plot raw
plot_labeled_segments(axs[0], t, signal_raw, labels, "Raw EMG Signal (Colored by Label)")
# Plot filtered
plot_labeled_segments(axs[1], t, signal_filtered, labels, "Filtered EMG Signal (20–500 Hz)")

axs[1].set_xlabel("Time (s)")
plt.tight_layout()
plt.show()
