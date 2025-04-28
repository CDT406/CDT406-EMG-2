import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt

def get_emg():
    """
    Load EMG data from the NinaPro dataset.
    """
    # Load the MATLAB file
    annotation = loadmat('NinaPro_DB5/s1/s1/S1_E3_A1.mat')

    # Get the keys (excluding private keys that start with "__")
    labels = [key for key in annotation.keys() if not key.startswith('__')]

    # Extract the data for each label
    data_dict = {}
    for label in labels:
        data_dict[label] = np.squeeze(annotation[label])

    # Get EMG data
    emg_data = data_dict['emg']
    num_channels = emg_data.shape[1]
    print(f"Number of EMG channels: {num_channels}")

    # Create DataFrame
    column_names = [f"EMG_{i+1}" for i in range(num_channels)]
    emg_df = pd.DataFrame(data=emg_data, columns=column_names)

    return emg_df, data_dict

def plot_emg_channel(emg_df, channel=5):
    """
    Plot a specific EMG channel.

    Args:
        emg_df (DataFrame): DataFrame containing EMG data
        channel (int): Channel number to plot (default: 5)
    """
    plt.figure(figsize=(12, 6))
    plt.plot(emg_df[f'EMG_{channel}'], label=f'EMG Channel {channel}')
    plt.ylabel("Amplitude")
    plt.xlabel("Sample Index")
    plt.title(f"EMG Signal - Channel {channel}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    emg_df, data_dict = get_emg()
    plot_emg_channel(emg_df, channel=5)
