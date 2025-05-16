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

def load_and_predict(model_path, signal_path, window_size=224, overlap=112, sequence_length=3,
                    fs=5000, lowcut=20, highcut=450, order=4, wamp_threshold=0.02):
    """Load model and make predictions on a signal"""
    # Load and process signal
    df = pd.read_csv(signal_path, header=None, names=['time', 'voltage', 'label'])
    signal = df['voltage'].values
    true_labels = df['label'].values
    
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
    
    return signal, true_labels, predictions, window_starts

def visualize_predictions(signal, true_labels, predictions, window_starts, window_size, fs=5000, is_2state=True):
    """Visualize signal with correct/incorrect predictions highlighted"""
    t = np.arange(len(signal)) / fs
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1])
    
    # Signal plot
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, signal, 'k-', alpha=0.3, label='Signal')
    
    if is_2state:
        # Define 2-state colors and names
        label_colors = {
            0: '#7F8C8D',  # Rest - gray
            1: '#3498DB',  # Active - blue
        }
        label_names = {
            0: 'Rest',
            1: 'Active'
        }
        # Map 4-state labels to 2-state for true labels
        true_labels = np.where(true_labels > 0, 1, 0)
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
    
    # Collect true labels and predictions for each window
    window_true_labels = []
    window_predictions = []
    
    # Plot predictions with actual state colors
    for i, (start, pred) in enumerate(zip(window_starts, predictions)):
        end = min(start + window_size, len(signal))
        window_labels = true_labels[start:end]
        majority_true = np.argmax(np.bincount(window_labels))
        window_true_labels.append(majority_true)
        window_predictions.append(pred)
        
        # Plot the true label background
        true_color = label_colors[majority_true]
        ax1.axvspan(t[start], t[end-1], alpha=0.2, color=true_color)
        
        # Add markers for predictions
        mid_point = start + (end - start) // 2
        if mid_point < len(t):
            if pred == majority_true:
                ax1.plot(t[mid_point], 0, 'g.', markersize=10, alpha=0.7)
            else:
                ax1.plot(t[mid_point], 0, 'rx', markersize=10, alpha=0.7)
    
    # Add legend
    handles = []
    for state in range(len(label_colors)):
        handles.append(plt.plot([], [], color=label_colors[state], 
                              label=f'True {label_names[state]}', alpha=0.2)[0])
    handles.extend([
        plt.plot([], [], 'g.', label='Correct Prediction', markersize=10, alpha=0.7)[0],
        plt.plot([], [], 'rx', label='Incorrect Prediction', markersize=10, alpha=0.7)[0]
    ])
    
    ax1.set_title(f'EMG Signal with Predictions ({("2-state" if is_2state else "4-state")} Model)\n(Background: True Labels, Markers: Predictions)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Voltage')
    ax1.grid(True)
    ax1.legend(handles=handles, loc='upper right')
    
    # Add confusion matrix visualization
    ax2 = fig.add_subplot(gs[1])
    window_true_labels = np.array(window_true_labels)
    window_predictions = np.array(window_predictions)
    
    # Calculate class-wise accuracy
    accuracies = {}
    for state in range(len(label_colors)):
        state_mask = window_true_labels == state
        if np.sum(state_mask) > 0:
            state_acc = np.mean(window_predictions[state_mask] == state) * 100
            accuracies[label_names[state]] = state_acc
    
    # Create accuracy bar plot
    states = list(accuracies.keys())
    accs = list(accuracies.values())
    bars = ax2.bar(states, accs)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    ax2.set_title('Prediction Accuracy by State')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, axis='y')
    ax2.set_ylim(0, 100)
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    for true_state in range(len(label_colors)):
        for pred_state in range(len(label_colors)):
            mask = (window_true_labels == true_state) & (window_predictions == pred_state)
            count = np.sum(mask)
            print(f"True {label_names[true_state]}, Predicted {label_names[pred_state]}: {count}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Configuration
    USE_2STATE = True  # Set to False for 4-state model
    if USE_2STATE:
        MODEL_PATH = "output/SavedModels/RNN/WindowTest50%overlap/2state/2state_features_labels_W224_O112_WAMPth20.tflite"
    else:
        MODEL_PATH = "output/SavedModels/RNN/WindowTest50%overlap/4state/4state_features_labels_W224_O112_WAMPth20.tflite"
    
    SIGNAL_PATH = "datasets/official/unprocessed/relabeled_old_dataset/1/0205-132514record.csv"
    
    # Parameters matching the training configuration
    WINDOW_SIZE = 224
    OVERLAP = 112
    SEQUENCE_LENGTH = 3
    FS = 5000
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