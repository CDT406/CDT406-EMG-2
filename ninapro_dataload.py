import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt

training_data = ['NinaPro_DB5/s1/s1/S1_E2_A1.mat',
                 'NinaPro_DB5/s2/s2/S2_E2_A1.mat',
                 'NinaPro_DB5/s3/s3/S3_E2_A1.mat']

testing_data = 'NinaPro_DB5/s4/S4_E2_A1.mat'

def get_emg(file_name='NinaPro_DB5/s1/s1/S1_E3_A1.mat'):
    """
    Load EMG data from the NinaPro dataset.
    """
    # Load the MATLAB file
    annotation = loadmat(file_name)

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
    plt.ylabel("Frequency")
    plt.xlabel("Sample Index")
    plt.title(f"EMG Signal - Channel {channel}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
def print_info(data_dict):
    print(data_dict)
    
    
def get_training_data():
    """
    Load training data from the NinaPro dataset.
    """
    training_data_list = []
    for file_name in training_data:
        emg_df, data_dict = get_emg(file_name)
        training_data_list.append(emg_df)
    return training_data_list


def get_testing_data():
    """
    Load testing data from the NinaPro dataset.
    """
    emg_df, data_dict = get_emg(testing_data)
    return emg_df


# Example usage
if __name__ == "__main__":
    emg_df, data_dict = get_emg()
    print_info(data_dict)
    plot_emg_channel(emg_df, channel=4)
