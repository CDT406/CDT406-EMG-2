# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# # Set the path to the file you want to visualize
# base_path = r"official\1\0205-132514record.csv"
# participant = "1"
# file_name = "0205-132514record.csv"  # Replace with the actual file name

# file_path = os.path.join(base_path)

# # Read the CSV
# df = pd.read_csv(file_path)

# # Ensure the columns are labeled correctly
# df.columns = ['timestamp', 'frequency', 'label']  # Adjust if actual column names differ

# # Plot timestamp vs frequency
# plt.figure(figsize=(12, 6))
# plt.scatter(df['timestamp'], df['frequency'], c=df['label'], cmap='viridis', s=5)
# plt.title(f"sEMG Data - Participant {participant} - {file_name}")
# plt.xlabel("Timestamp")
# plt.ylabel("Frequency")
# plt.colorbar(label="Label")
# plt.tight_layout()
# plt.show()


#######################################################################################

#######################################################################################


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import random
# from scipy.interpolate import CubicSpline

# # Load one file
# file_path = os.path.join(r"official\1\0205-132514record.csv")  # Update to actual file name
# df = pd.read_csv(file_path)
# df.columns = ['timestamp', 'frequency', 'label']
# signal = df['frequency'].values

# # Augmentation Functions

# def jitter(signal, noise_level=0.002):
#     noise = np.random.normal(loc=0.0, scale=noise_level, size=signal.shape)
#     return signal + noise

# def scaling(signal, scale_range=(0.8, 1.2)):
#     scale = np.random.uniform(scale_range[0], scale_range[1])
#     return signal * scale

# def time_warp(signal, sigma=0.2):
#     # Create random smooth curve to warp time
#     orig_steps = np.arange(len(signal))
#     random_warp = np.random.normal(loc=1.0, scale=sigma, size=len(signal))
#     warped_steps = np.cumsum(random_warp)
#     warped_steps = warped_steps * (len(signal) / warped_steps[-1])
#     cs = CubicSpline(warped_steps, signal, extrapolate=True)
#     return cs(orig_steps)

# # Apply Augmentations
# aug_jitter = jitter(signal)
# aug_scaling = scaling(signal)
# aug_timewarp = time_warp(signal)

# # Plotting
# plt.figure(figsize=(15, 8))

# plt.subplot(4, 1, 1)
# plt.plot(signal, label="Original")
# plt.title("Original Signal")
# plt.legend()

# plt.subplot(4, 1, 2)
# plt.plot(aug_jitter, label="Jittering", color='orange')
# plt.title("Jittering")
# plt.legend()

# plt.subplot(4, 1, 3)
# plt.plot(aug_scaling, label="Scaling", color='green')
# plt.title("Scaling")
# plt.legend()

# plt.subplot(4, 1, 4)
# plt.plot(aug_timewarp, label="Time Warping", color='purple')
# plt.title("Time Warping")
# plt.legend()

# plt.tight_layout()
# plt.show()



###################################################################################

####################################################################################


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import random
# from scipy.interpolate import CubicSpline


# # -------- Define Augmentations -------- #
# def jitter(signal, noise_level=0.005):
#     return signal + np.random.normal(0, noise_level, size=signal.shape)

# def scaling(signal, scale_range=(0.8, 1.2)):
#     scale = np.random.uniform(*scale_range)
#     return signal * scale

# def time_warp(signal, sigma=0.2):
#     orig_steps = np.arange(len(signal))
#     random_warp = np.random.normal(1.0, sigma, len(signal))
#     warped_steps = np.cumsum(random_warp)
#     warped_steps = warped_steps * (len(signal) / warped_steps[-1])
#     cs = CubicSpline(warped_steps, signal, extrapolate=True)
#     return cs(orig_steps)

# # -------- Combine Augmentations -------- #
# def apply_augmentations(signal, augmentations):
#     for aug in augmentations:
#         signal = aug(signal)
#     return signal

# # -------- Paths -------- #
# input_base = "official"
# output_base = "augmented_data"
# os.makedirs(output_base, exist_ok=True)

# # -------- Augment and Save All -------- #
# combined_augmentations = [jitter, scaling, time_warp]

# for participant_id in range(8, 10):  # folders 1 to 9
#     input_folder = os.path.join(input_base, str(participant_id))
#     output_folder = os.path.join(output_base, str(participant_id))
#     os.makedirs(output_folder, exist_ok=True)

#     for filename in os.listdir(input_folder):
#         if filename.endswith(".csv"):
#             input_path = os.path.join(input_folder, filename)
#             df = pd.read_csv(input_path)
#             df.columns = ['timestamp', 'frequency', 'label']

#             # Apply augmentation
#             original_signal = df['frequency'].values
#             augmented_signal = apply_augmentations(original_signal.copy(), combined_augmentations)
#             df['frequency'] = augmented_signal

#             # Save to output folder
#             output_path = os.path.join(output_folder, f"aug_{filename}")
#             df.to_csv(output_path, index=False)
#             print(f"Saved: {output_path}")



#####################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import CubicSpline

# -------- Augmentation Functions -------- #
def jitter(signal, noise_level=0.002):
    return signal + np.random.normal(loc=0.0, scale=noise_level, size=signal.shape)

def scaling(signal, scale_range=(0.8, 1.2)):
    scale = np.random.uniform(*scale_range)
    return signal * scale

def time_warp(signal, sigma=0.2, knot=4):
    orig_steps = np.arange(len(signal))
    knot_x = np.linspace(0, len(signal) - 1, knot)
    knot_y = np.random.normal(loc=1.0, scale=sigma, size=knot)
    warping_curve = CubicSpline(knot_x, knot_y)(orig_steps)
    return signal * warping_curve

def save_augmented(df_original, aug_signal, output_path):
    df_aug = df_original.copy()
    df_aug['frequency'] = aug_signal
    df_aug.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

# -------- Process all files -------- #
input_base = "official"
output_base = "augmented_data"

for participant in os.listdir(input_base):
    participant_path = os.path.join(input_base, participant)
    if not os.path.isdir(participant_path):
        continue

    output_participant_dir = os.path.join(output_base, participant)
    os.makedirs(output_participant_dir, exist_ok=True)

    for filename in os.listdir(participant_path):
        if not filename.endswith(".csv"):
            continue

        file_path = os.path.join(participant_path, filename)
        df = pd.read_csv(file_path)
        df.columns = ['timestamp', 'frequency', 'label']
        signal = df['frequency'].values

        # Apply each augmentation
        aug_jitter = jitter(signal)
        aug_scaling = scaling(signal)
        aug_timewarp = time_warp(signal)

        # Save augmented files
        save_augmented(df, aug_jitter, os.path.join(output_participant_dir, f"jitter_{filename}"))
        save_augmented(df, aug_scaling, os.path.join(output_participant_dir, f"scaling_{filename}"))
        save_augmented(df, aug_timewarp, os.path.join(output_participant_dir, f"timewarp_{filename}"))