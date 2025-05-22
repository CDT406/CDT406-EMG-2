import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from lstm_model import build_model, create_sequences, load_processed_data

def calculate_per_class_average_accuracy(y_true, y_pred):
    """Calculate the average of per-class accuracies (balanced accuracy)."""
    classes = np.unique(y_true)
    class_accuracies = []
    
    for c in classes:
        # Get indices for this class
        class_mask = (y_true == c)
        if np.sum(class_mask) > 0:  # Only calculate if we have samples
            # Calculate accuracy for this class
            class_acc = np.mean(y_true[class_mask] == y_pred[class_mask])
            class_accuracies.append(class_acc)
    
    # Return average of per-class accuracies
    return np.mean(class_accuracies)

def train_all_window_sizes(mode='4state'):
    """Train models for all preprocessed window sizes."""
    results = {}
    base_dir = 'src/Model/RNN/LSTM2/Pre-processing3'
    
    # Define suffix at the start of function
    suffix = '_2state' if mode == '2state' else ''
    
    # Find all processed_data directories
    data_dirs = [d for d in os.listdir(base_dir) 
                if d.startswith('processed_data_') and d.endswith('ms')]
    
    for data_dir in sorted(data_dirs):
        window_size = int(data_dir.split('_')[-1].replace('ms', ''))
        print(f"\nTraining model for window size: {window_size}ms in {mode} mode")
        
        # Load data from original processed data directory
        data_path = os.path.join(base_dir, data_dir, 'processed_data.json')
        data = load_processed_data(data_path)
        
        # Split data
        train_data = {str(pid): data[str(pid)] for pid in data.keys() 
                     if int(pid) not in [1, 8]}
        val_data = {str(pid): data[str(pid)] for pid in data.keys() 
                   if int(pid) in [1, 8]}
        
        # Create sequences with mode-specific label mapping
        X_train, y_train = create_sequences(train_data, mode=mode)
        X_val, y_val = create_sequences(val_data, mode=mode)
        
        # Build model with correct number of classes
        num_classes = 2 if mode == '2state' else 4
        model = build_model(num_classes=num_classes)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            verbose=1
        )
        
        # Evaluate with balanced metrics
        y_pred = model.predict(X_val).argmax(axis=1)
        balanced_accuracy = calculate_per_class_average_accuracy(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average='macro')
        conf_matrix = confusion_matrix(y_val, y_pred)
        
        # Store results
        results[window_size] = {
            'balanced_accuracy': float(balanced_accuracy),
            'macro_f1': float(macro_f1),
            'confusion_matrix': conf_matrix.tolist(),
            'val_accuracy': history.history['val_accuracy'][-1],
            'val_loss': history.history['val_loss'][-1]
        }
    
    # Plot performance comparison
    plot_performance_comparison(results, mode)
    
    # Save results with mode-specific paths (now suffix is defined)
    results_path = f'src/Model/RNN/LSTM2/window_size_comparison{suffix}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining completed for {mode}! Results saved to:", results_path)

def save_confusion_matrix(conf_matrix, window_size, save_dir, class_names):
    """Save confusion matrix as heatmap."""
    plt.figure(figsize=(6, 5) if len(class_names) == 2 else (8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'Confusion Matrix - Window Size {window_size}ms')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def plot_performance_comparison(results, mode):
    """Plot window sizes vs performance metrics."""
    window_sizes = sorted(list(results.keys()))
    accuracies = [results[ws]['balanced_accuracy'] for ws in window_sizes]
    f1_scores = [results[ws]['macro_f1'] for ws in window_sizes]
    
    plt.figure(figsize=(12, 6))
    plt.plot(window_sizes, accuracies, 'b-o', label='Balanced Accuracy')
    plt.plot(window_sizes, f1_scores, 'r-o', label='Macro F1')
    plt.xlabel('Window Size (ms)')
    plt.ylabel('Score')
    plt.title(f'Model Performance vs Window Size ({mode})')
    plt.grid(True)
    plt.legend()
    
    # Save with mode-specific path
    suffix = '_2state' if mode == '2state' else ''
    plt.savefig(f'src/Model/RNN/LSTM2/performance_metrics{suffix}.png')
    plt.close()

if __name__ == '__main__':
    # Train 2-state first, then 4-state
    for mode in ['2state', '4state']:
        print(f"\nStarting {mode} training...")
        train_all_window_sizes(mode)