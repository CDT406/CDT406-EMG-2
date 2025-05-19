import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tflite_runtime.interpreter import Interpreter
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """Bandpass filter the signal"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def compute_mav(window):
    """Mean Absolute Value"""
    return np.mean(np.abs(window))

def compute_wl(window):
    """Waveform Length"""
    return np.sum(np.abs(np.diff(window)))

def compute_wamp(window, threshold=0.02):
    """Willison Amplitude"""
    return np.sum(np.abs(np.diff(window)) > threshold)

def compute_mavs(window):
    """MAV Slope"""
    half = len(window) // 2
    return np.abs(compute_mav(window[:half]) - compute_mav(window[half:]))

def extract_features(window, wamp_threshold=0.02):
    """Extract all features from a window"""
    features = []
    features.append(compute_mav(window))
    features.append(compute_wl(window))
    features.append(compute_wamp(window, threshold=wamp_threshold))
    features.append(compute_mavs(window))
    return np.array(features, dtype=np.float32)

def process_signal(signal, window_size, overlap, fs, lowcut, highcut, order, wamp_threshold):
    """Process signal into windows and extract features"""
    # Apply bandpass filter
    signal = bandpass_filter(signal, lowcut, highcut, fs, order)
    
    # Calculate step size
    step = window_size - overlap
    num_windows = (len(signal) - window_size) // step + 1
    
    X = []
    window_starts = []
    
    for i in range(num_windows):
        start = i * step
        end = start + window_size
        window = signal[start:end]
        
        if len(window) == window_size:
            features = extract_features(window, wamp_threshold=wamp_threshold)
            X.append(features)
            window_starts.append(start)
    
    return np.array(X), window_starts

def create_sequences(features, sequence_length):
    """Create sequences for LSTM input"""
    if len(features) < sequence_length:
        return None
    
    X = []
    for i in range(len(features) - sequence_length + 1):
        X.append(features[i:i + sequence_length])
    
    return np.array(X)

def load_and_predict(model_path, signal_path, window_size, overlap, sequence_length,
                    fs, lowcut, highcut, order, wamp_threshold):
    """Load model and make predictions on a signal"""
    # Load signal (raw values only)
    signal = np.loadtxt(signal_path, delimiter=",", dtype=np.float32)
    
    # Convert ADC values to voltage (assuming 12-bit ADC with 1.8V reference)
    signal = (signal / 4095.0) * 1.8
    
    # Process signal into windows and extract features
    X, window_starts = process_signal(signal, window_size, overlap, fs, lowcut, highcut, order, wamp_threshold)
    
    # Create sequences for LSTM
    X = create_sequences(X, sequence_length)
    
    if X is None:
        raise ValueError("Signal too short for sequence creation")
    
    # Load and run model
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    predictions = []
    for i in range(len(X)):
        interpreter.set_tensor(input_details[0]['index'], X[i:i+1].astype(np.float32))
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(np.argmax(pred[0]))
    
    # Pad predictions to match window_starts (due to sequence creation)
    predictions = [predictions[0]] * (sequence_length - 1) + predictions
    
    # For visualization, we'll use predictions as both true and predicted labels
    # since we don't have ground truth
    return signal, predictions, predictions, window_starts

def visualize_predictions(signal, true_labels, predictions, window_starts, window_size, fs=1000, is_2state=True):
    """Visualize signal with predictions"""
    t = np.arange(len(signal)) / fs
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1])
    
    # Signal plot
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, signal, 'k-', alpha=0.3, label='Signal')
    
    if is_2state:
        # Define 2-state colors and names (swapped to match actual predictions)
        label_colors = {
            0: '#3498DB',  # Active - blue (was Rest)
            1: '#7F8C8D',  # Rest - gray (was Active)
        }
        label_names = {
            0: 'Active',   # Swapped
            1: 'Rest'      # Swapped
        }
    else:
        # Define 4-state colors and names
        label_colors = {
            0: '#7F8C8D',  # Rest - gray
            1: '#E67E22',  # Grip - orange
            2: '#3498DB',  # Hold - blue
            3: '#E74C3C'   # Release - red
        }
        label_names = {
            0: 'Rest',
            1: 'Grip',
            2: 'Hold',
            3: 'Release'
        }
    
    # Plot predictions
    for i, (start, pred) in enumerate(zip(window_starts, predictions)):
        end = min(start + window_size, len(signal))
        
        # Plot the prediction background
        pred_color = label_colors[pred]
        ax1.axvspan(t[start], t[end-1], alpha=0.2, color=pred_color)
        
        # Add markers for predictions
        mid_point = start + (end - start) // 2
        if mid_point < len(t):
            ax1.plot(t[mid_point], 0, 'g.', markersize=10, alpha=0.7)
    
    # Add legend
    handles = []
    for state in range(len(label_colors)):
        handles.append(plt.plot([], [], color=label_colors[state], 
                              label=f'{label_names[state]}', alpha=0.2)[0])
    
    ax1.set_title(f'EMG Signal with Predictions ({("2-state" if is_2state else "4-state")} Model)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Voltage')
    ax1.grid(True)
    ax1.legend(handles=handles, loc='upper right')
    
    # Add prediction distribution plot
    ax2 = fig.add_subplot(gs[1])
    predictions = np.array(predictions)
    
    # Calculate prediction distribution
    pred_counts = {}
    for state in range(len(label_colors)):
        count = np.sum(predictions == state)
        pred_counts[label_names[state]] = count
    
    # Create distribution bar plot
    states = list(pred_counts.keys())
    counts = list(pred_counts.values())
    bars = ax2.bar(states, counts)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}', ha='center', va='bottom')
    
    ax2.set_title('Prediction Distribution')
    ax2.set_ylabel('Count')
    ax2.grid(True, axis='y')
    
    # Print prediction distribution
    print("\nPrediction Distribution:")
    for state, count in pred_counts.items():
        print(f"{state}: {count}")
    
    plt.tight_layout()
    # Save the plot instead of showing it
    plt.savefig('emg_predictions.png')
    plt.close()

if __name__ == "__main__":
    # Configuration
    USE_2STATE = True  # Set to False for 4-state model
    if USE_2STATE:
        MODEL_PATH = "output/SavedModels/RNN/NormalizedData1000Hz/WindowTest50%overlap/2state/2state_features_labels_W180_O90_WAMPth20.tflite"
    else:
        MODEL_PATH = "output/SavedModels/RNN/NormalizedData1000Hz/WindowTest50%overlap/4state/4state_features_labels_W180_O90_WAMPth20.tflite"
    
    SIGNAL_PATH = "src/inference/output.csv"  # Updated to use our recorded signal
    
    # Parameters matching the training configuration
    WINDOW_SIZE = 180      # 180 samples at 1000Hz = 180ms window
    OVERLAP = 90          # 50% overlap
    SEQUENCE_LENGTH = 3
    FS = 1000            # Updated to 1000Hz
    LOWCUT = 20
    HIGHCUT = 450
    ORDER = 4
    WAMP_THRESHOLD = 0.02
    
    # Run prediction and visualization
    signal, true_labels, predictions, window_starts = load_and_predict(
        MODEL_PATH, SIGNAL_PATH, WINDOW_SIZE, OVERLAP, SEQUENCE_LENGTH,
        FS, LOWCUT, HIGHCUT, ORDER, WAMP_THRESHOLD
    )
    
    visualize_predictions(signal, true_labels, predictions, window_starts, WINDOW_SIZE, FS, USE_2STATE) 