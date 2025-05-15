import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ----------- Parameters -----------
signal_path = "datasets/official/unprocessed/relabeled_slow/0205-132514record.csv"  # ← Update to your path
fs = 5000  # Hz
order = 4
bands = [(i, i + 10) for i in range(20, 500, 10)]  # 10 Hz steps → 48 bands
bands_per_figure = 6  # plots per page

# Label colors and names
label_colors = {
    0: '#7F8C8D',  # Rest – gray
    1: '#E67E22',  # Grip – orange
    2: '#3498DB',  # Hold – blue
    3: '#E74C3C'   # Release – red
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
signal = df['voltage'].values
labels = df['label'].values
t = np.arange(len(signal)) / fs

# ----------- Helper: plot colored segments -----------
def plot_labeled_signal(ax, t, signal, labels, title):
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[start] or i == len(labels) - 1:
            label = labels[start]
            end = i if labels[i] != label else i + 1
            ax.plot(t[start:end], signal[start:end],
                    color=label_colors.get(label, 'black'),
                    label=label_names.get(label, str(label)),
                    linewidth=0.8)
            start = i
    ax.set_title(title)
    ax.set_ylabel("Voltage")
    ax.grid(True)

# ----------- Main Loop: Paginated Plots -----------
for fig_index in range(0, len(bands), bands_per_figure):
    fig, axs = plt.subplots(bands_per_figure, 1, figsize=(15, 2.8 * bands_per_figure), sharex=True)
    axs = np.atleast_1d(axs)

    for i, (low, high) in enumerate(bands[fig_index:fig_index + bands_per_figure]):
        band_index = fig_index + i
        filtered = bandpass_filter(signal, low, high, fs, order)
        title = f"Band {low}–{high} Hz"
        plot_labeled_signal(axs[i], t, filtered, labels, title)

    axs[-1].set_xlabel("Time (s)")

    # Deduplicate legend
    handles, lbls = axs[0].get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    axs[0].legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.tight_layout()
    plt.show()
