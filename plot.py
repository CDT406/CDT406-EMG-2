import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Input data
df = pd.read_csv("data/raw/M6 Dataset/subject #1/cycle #1/P1C1S1M6F1O1", header=None)
# Assuming that each sample has multiple features and represents a time-series of sEMG data
# Plot the first few rows (signals) for visualization
x_train = df.values.flatten()

plt.figure(figsize=(12, 6))
plt.plot(x_train)  # Plot the mean of each window as an example
plt.title("Samples")
plt.xlabel("Sample index")
plt.ylabel("Sample Value")
plt.legend()
plt.show()