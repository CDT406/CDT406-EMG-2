import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch

# ----------- Parameters -----------
file_path = r"datasets/official/4/0205-135538record.csv"
sampling_rate = 5000  # Hz
low_cut_1 = 20
high_cut_1 = 450
low_cut_2 = 20
high_cut_2 = 150
filter_order = 4
notch_freq = 50
quality_factor = 30

# ----------- Filters -----------
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def notch_filter(signal, freq, fs, q=30):
    nyquist = 0.5 * fs
    w0 = freq / nyquist
    b, a = iirnotch(w0, q)
    return filtfilt(b, a, signal)

# ----------- Load Data -----------
df = pd.read_csv(file_path, header=None, names=['time', 'voltage', 'label'])
signal = df['voltage'].values
labels = df['label'].values
time = df['time'].values / 1000  # ms to seconds

# ----------- Filtered Signals -----------
bandpass_20_450 = bandpass_filter(signal, low_cut_1, high_cut_1, sampling_rate, filter_order)
notched = notch_filter(signal, notch_freq, sampling_rate, quality_factor)
notch_plus_bandpass_20_450 = bandpass_filter(notched, low_cut_1, high_cut_1, sampling_rate, filter_order)
bandpass_20_150 = bandpass_filter(signal, low_cut_2, high_cut_2, sampling_rate, filter_order)

# ----------- Color Mapping (Rest = 0, Hold/Grip/Release = 1/2/3 → blue) -----------
label_colors = {
    0: 'gray',   # Rest
    1: 'blue',   # Grip
    2: 'blue',   # Hold
    3: 'blue'    # Release
}

def plot_colored_signal(ax, time, signal, labels, title):
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[start] or i == len(labels) - 1:
            end = i
            label = labels[start]
            color = label_colors.get(label, 'red')  # fallback
            ax.plot(time[start:end], signal[start:end], color=color, linewidth=1)
            start = i
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage")

# ----------- Plot All Versions with Label Coloring -----------
fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)

plot_colored_signal(axes[0], time, signal, labels, "Raw EMG Signal")
plot_colored_signal(axes[1], time, bandpass_20_450, labels, "Bandpass Filtered (20–450 Hz)")
plot_colored_signal(axes[2], time, notch_plus_bandpass_20_450, labels, "Notch @ 50 Hz + Bandpass (20–450 Hz)")
plot_colored_signal(axes[3], time, bandpass_20_150, labels, "Bandpass Filtered (20–150 Hz)")

# Add custom legend
legend_handles = [
    plt.Line2D([0], [0], color='gray', lw=2, label='Rest'),
    plt.Line2D([0], [0], color='blue', lw=2, label='Hold/Grip/Release')
]
axes[0].legend(handles=legend_handles)

plt.tight_layout()
plt.show()
