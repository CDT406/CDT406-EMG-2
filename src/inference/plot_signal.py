import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read the signal
signal = pd.read_csv('output.csv', header=None).values.flatten()

# Create time array (assuming 1000 Hz sampling rate)
time = np.arange(len(signal)) / 1000  # Convert to seconds

# Normalize the signal (zero mean, unit variance)
signal_normalized = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

# Create the plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

# Plot raw signal
ax1.plot(time, signal, 'b-', linewidth=1, alpha=0.8)
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Amplitude')
ax1.set_title('Raw EMG Signal')

# Add statistics to raw signal plot
ax1.axhline(y=np.mean(signal), color='r', linestyle='--', alpha=0.5, 
            label=f'Mean: {np.mean(signal):.2f}')
ax1.axhline(y=np.mean(signal) + np.std(signal), color='g', linestyle=':', alpha=0.5, 
            label=f'Mean ± Std: {np.std(signal):.2f}')
ax1.axhline(y=np.mean(signal) - np.std(signal), color='g', linestyle=':', alpha=0.5)
ax1.legend()

# Plot normalized signal
ax2.plot(time, signal_normalized, 'b-', linewidth=1, alpha=0.8)
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Normalized Amplitude')
ax2.set_title('Normalized EMG Signal (Zero Mean, Unit Variance)')

# Add statistics to normalized signal plot
ax2.axhline(y=np.mean(signal_normalized), color='r', linestyle='--', alpha=0.5, 
            label=f'Mean: {np.mean(signal_normalized):.2f}')
ax2.axhline(y=np.mean(signal_normalized) + np.std(signal_normalized), color='g', linestyle=':', alpha=0.5, 
            label=f'Mean ± Std: {np.std(signal_normalized):.2f}')
ax2.axhline(y=np.mean(signal_normalized) - np.std(signal_normalized), color='g', linestyle=':', alpha=0.5)
ax2.legend()

# Print statistics
print(f"Raw Signal Statistics:")
print(f"Length: {len(signal)} samples ({len(signal)/1000:.2f} seconds)")
print(f"Mean: {np.mean(signal):.2f}")
print(f"Std: {np.std(signal):.2f}")
print(f"Min: {np.min(signal):.2f}")
print(f"Max: {np.max(signal):.2f}")

print(f"\nNormalized Signal Statistics:")
print(f"Mean: {np.mean(signal_normalized):.2e}")
print(f"Std: {np.std(signal_normalized):.2f}")
print(f"Min: {np.min(signal_normalized):.2f}")
print(f"Max: {np.max(signal_normalized):.2f}")

plt.tight_layout()
plt.savefig('signal_plot_comparison.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'signal_plot_comparison.png'") 