import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from lstm_model import build_model, create_sequences, load_processed_data

def train_all_window_sizes():
    """Train models for all preprocessed window sizes."""
    results = {}
    base_dir = 'src/Model/RNN/LSTM2/Pre-processing3'
    
    # Find all processed_data directories
    data_dirs = [d for d in os.listdir(base_dir) if d.startswith('processed_data_')]
    
    for data_dir in sorted(data_dirs):
        window_size = int(data_dir.split('_')[-1].replace('ms', ''))
        print(f"\nTraining model for window size: {window_size}ms")
        
        # Load data
        data_path = os.path.join(base_dir, data_dir, 'processed_data.json')
        data = load_processed_data(data_path)
        
        # Split data
        train_data = {str(pid): data[str(pid)] for pid in data.keys() 
                     if int(pid) not in [1, 8]}
        val_data = {str(pid): data[str(pid)] for pid in data.keys() 
                   if int(pid) in [1, 8]}
        
        # Create sequences
        X_train, y_train = create_sequences(train_data)
        X_val, y_val = create_sequences(val_data)
        
        # Build and train model
        model = build_model()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            verbose=1
        )
        
        # Evaluate with additional metrics
        y_pred = model.predict(X_val).argmax(axis=1)
        macro_accuracy = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average='macro')
        conf_matrix = confusion_matrix(y_val, y_pred)
        
        # Save confusion matrix
        save_dir = f'src/Model/RNN/LSTM2/Saved_models_{window_size}ms'
        os.makedirs(save_dir, exist_ok=True)
        save_confusion_matrix(conf_matrix, window_size, save_dir)
        
        # Save results
        results[window_size] = {
            'macro_accuracy': float(macro_accuracy),
            'macro_f1': float(macro_f1),
            'confusion_matrix': conf_matrix.tolist(),
            'val_accuracy': history.history['val_accuracy'][-1],
            'val_loss': history.history['val_loss'][-1]
        }
        
        # Save model
        model.save(os.path.join(save_dir, 'model.keras'))
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(os.path.join(save_dir, 'model.tflite'), 'wb') as f:
            f.write(tflite_model)
    
    # Plot performance comparison
    plot_performance_comparison(results)
    
    # Save overall results and print best performances
    results_path = 'src/Model/RNN/LSTM2/window_size_comparison.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nTraining completed! Results saved to:", results_path)
    best_acc_size = max(results.keys(), key=lambda x: results[x]['macro_accuracy'])
    best_f1_size = max(results.keys(), key=lambda x: results[x]['macro_f1'])
    print(f"Best accuracy window size: {best_acc_size}ms (Accuracy: {results[best_acc_size]['macro_accuracy']:.4f})")
    print(f"Best F1 window size: {best_f1_size}ms (F1: {results[best_f1_size]['macro_f1']:.4f})")

def save_confusion_matrix(conf_matrix, window_size, save_dir):
    """Save confusion matrix as heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Window Size {window_size}ms')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def plot_performance_comparison(results):
    """Plot window sizes vs performance metrics."""
    window_sizes = sorted(list(results.keys()))
    accuracies = [results[ws]['macro_accuracy'] for ws in window_sizes]
    f1_scores = [results[ws]['macro_f1'] for ws in window_sizes]
    
    plt.figure(figsize=(12, 6))
    plt.plot(window_sizes, accuracies, 'b-o', label='Macro Accuracy')
    plt.plot(window_sizes, f1_scores, 'r-o', label='Macro F1')
    plt.xlabel('Window Size (ms)')
    plt.ylabel('Score')
    plt.title('Model Performance vs Window Size')
    plt.grid(True)
    plt.legend()
    plt.savefig('src/Model/RNN/LSTM2/performance_metrics.png')
    plt.close()

if __name__ == '__main__':
    train_all_window_sizes()