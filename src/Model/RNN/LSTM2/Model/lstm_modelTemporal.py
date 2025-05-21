import os
import json
import numpy as np
import tensorflow as tf
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

def load_processed_data(data_path):
    """Load and parse the processed EMG data."""
    with open(data_path, 'r') as f:
        return json.load(f)

def create_sequences(data, sequence_length=3):
    """Create sequences of 3 consecutive windows with their features and track cycle boundaries."""
    X, y = [], []
    cycle_ends = []  # Track where each cycle ends
    current_pos = 0
    
    for person_id, cycles in data.items():
        for cycle_id, windows in cycles.items():
            # Get features and labels for this cycle
            features = [[w['MAV'], w['WL'], w['WAMP'], w['MAVS']] for w in windows]
            labels = [w['label'] for w in windows]
            
            # Only create sequences within the same cycle
            cycle_sequences = []
            cycle_labels = []
            
            for i in range(len(features) - sequence_length + 1):
                # Don't create sequences that cross cycle boundaries
                seq = features[i:i + sequence_length]
                cycle_sequences.append(np.array(seq).flatten())
                cycle_labels.append(labels[i + sequence_length - 1])
            
            if cycle_sequences:  # Only add if cycle has sequences
                X.extend(cycle_sequences)
                y.extend(cycle_labels)
                current_pos += len(cycle_sequences)
                cycle_ends.append(current_pos)
    
    return np.array(X), np.array(y), cycle_ends

class CycleResetCallback(tf.keras.callbacks.Callback):
    """Reset LSTM states at cycle boundaries."""
    def __init__(self, cycle_ends, batch_size):
        super().__init__()
        self.cycle_ends = cycle_ends
        self.batch_size = batch_size
        
    def on_batch_end(self, batch, logs=None):
        current_pos = (batch + 1) * self.batch_size
        if any(end <= current_pos for end in self.cycle_ends):
            self.model.get_layer('lstm').reset_states()

class ValidationCycleResetCallback(tf.keras.callbacks.Callback):
    """Reset LSTM states at cycle boundaries during validation"""
    def __init__(self, cycle_ends, batch_size):
        super().__init__()
        self.cycle_ends = cycle_ends
        self.batch_size = batch_size
    
    def on_test_batch_end(self, batch, logs=None):
        current_pos = (batch + 1) * self.batch_size
        if any(end <= current_pos for end in self.cycle_ends):
            self.model.get_layer('lstm').reset_states()

def normalize_features_by_type(X_train, X_val):
    """Normalize each feature type using all values across windows"""
    # Reshape arrays to separate windows and features
    train_reshaped = X_train.reshape(-1, 3, 4)  # (samples, windows, features)
    val_reshaped = X_val.reshape(-1, 3, 4)
    
    feature_stats = {
        'MAV': {'mean': None, 'std': None},
        'WL': {'mean': None, 'std': None},
        'WAMP': {'mean': None, 'std': None},
        'MAVS': {'mean': None, 'std': None}
    }
    
    # Process each feature type
    for feat_idx, feat_name in enumerate(['MAV', 'WL', 'WAMP', 'MAVS']):
        # Get all values for this feature
        train_feature = train_reshaped[:, :, feat_idx].ravel()
        
        # Calculate stats
        mean = np.mean(train_feature)
        std = np.std(train_feature)
        feature_stats[feat_name]['mean'] = mean
        feature_stats[feat_name]['std'] = std
        
        # Normalize both train and val using same stats
        for window in range(3):
            train_reshaped[:, window, feat_idx] = (train_reshaped[:, window, feat_idx] - mean) / std
            val_reshaped[:, window, feat_idx] = (val_reshaped[:, window, feat_idx] - mean) / std
    
    return (train_reshaped.reshape(-1, 12), 
            val_reshaped.reshape(-1, 12), 
            feature_stats)

def build_model(num_features=12, num_classes=4, batch_size=32):
    """Build the LSTM model with stateful configuration."""
    model = Sequential([
        tf.keras.layers.Input(batch_shape=(batch_size, 3, 4), name='input'),
        LSTM(16,  # Reduced to 16 neurons
             return_sequences=False, 
             stateful=True,
             unroll=True,
             implementation=1,
             name='lstm'),
        Dense(32, activation='tanh', name='dense1'),  # Changed to tanh activation
        Dense(num_classes, activation='softmax', name='output')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def save_feature_stats_to_db(feature_stats, db_path):
    """Save feature statistics to SQLite database."""
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feature_stats (
            feature TEXT PRIMARY KEY,
            mean REAL,
            std REAL
        )
    ''')
    
    # Insert or replace feature statistics
    for feature, stats in feature_stats.items():
        cursor.execute('''
            INSERT OR REPLACE INTO feature_stats (feature, mean, std)
            VALUES (?, ?, ?)
        ''', (feature, stats['mean'], stats['std']))
    
    # Commit and close
    conn.commit()
    conn.close()

def predict_with_cycle_resets(model, X, cycle_ends, batch_size):
    """Make predictions with proper cycle boundary handling."""
    predictions = []
    model.get_layer('lstm').reset_states()  # Initial reset
    
    # Process each batch
    for i in range(0, len(X), batch_size):
        batch_x = X[i:i + batch_size]
        
        # Pad last batch if needed
        if len(batch_x) < batch_size:
            pad_size = batch_size - len(batch_x)
            batch_x = np.pad(batch_x, ((0, pad_size), (0, 0)), mode='constant')
        
        # Make prediction
        batch_pred = model.predict(batch_x, batch_size=batch_size)
        predictions.append(batch_pred[:len(batch_x)])
        
        # Check if we need to reset states
        current_pos = i + batch_size
        if any(end <= current_pos for end in cycle_ends):
            model.get_layer('lstm').reset_states()
    
    return np.vstack(predictions)

def main():
    # Create directories if they don't exist
    base_dir = os.path.join('src', 'Model', 'RNN', 'LSTM2', 'Saved_models')
    os.makedirs(base_dir, exist_ok=True)
    
    # Paths with proper extensions and using os.path.join
    data_path = os.path.join('src', 'Model', 'RNN', 'LSTM2', 'Pre-processing3', 
                            'processed_data', 'processed_data.json')
    model_save_path = os.path.join(base_dir, 'model.keras')  # .keras extension
    tflite_save_path = os.path.join(base_dir, 'model.tflite')
    scaler_save_path = os.path.join(base_dir, 'scaler.pkl')
    
    # Load data
    print("Loading data...")
    data = load_processed_data(data_path)
    
    # Split data by person
    train_data = {str(pid): data[str(pid)] for pid in data.keys() 
                 if int(pid) not in [1, 8]}
    val_data = {str(pid): data[str(pid)] for pid in data.keys() 
                if int(pid) in [1, 8]}
    
    # Create sequences with cycle boundaries
    print("Creating sequences...")
    X_train, y_train, train_cycle_ends = create_sequences(train_data)
    X_val, y_val, val_cycle_ends = create_sequences(val_data)
    
    # Make sure number of samples is divisible by batch_size
    batch_size = 32
    samples_per_epoch = (len(X_train) // batch_size) * batch_size
    
    model = build_model(batch_size=batch_size)
    
    model.fit(
        X_train[:samples_per_epoch].reshape(-1, 3, 4),
        y_train[:samples_per_epoch],
        validation_data=(
            X_val[:(len(X_val) // batch_size) * batch_size].reshape(-1, 3, 4),
            y_val[:(len(X_val) // batch_size) * batch_size]
        ),
        epochs=50,
        batch_size=batch_size,
        shuffle=False,  # Important for stateful LSTM
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            # Reset at training cycle boundaries
            CycleResetCallback(train_cycle_ends, batch_size),
            # Reset at validation cycle boundaries
            ValidationCycleResetCallback(val_cycle_ends, batch_size),
            # Reset at epoch end
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: model.get_layer('lstm').reset_states()
            )
        ]
    )
    
    # Evaluate model performance
    print("\nEvaluating model performance per class...")
    
    # Prepare validation data
    val_samples = (len(X_val) // batch_size) * batch_size
    X_val_reshaped = X_val[:val_samples].reshape(-1, 3, 4)
    y_val_trimmed = y_val[:val_samples]
    
    # Get predictions using cycle-aware prediction
    predictions = predict_with_cycle_resets(model, X_val_reshaped, val_cycle_ends, batch_size)
    y_pred = np.argmax(predictions, axis=1)
    
    # Calculate F1 scores
    weighted_f1 = f1_score(y_val_trimmed, y_pred, average='weighted')
    macro_f1 = f1_score(y_val_trimmed, y_pred, average='macro')
    
    # Print classification report
    class_names = ['Rest', 'Grip', 'Hold', 'Release']
    print("\nClassification Report:")
    print(classification_report(y_val_trimmed, y_pred, target_names=class_names))
    
    # Print F1 scores
    print(f"\nMacro F1-Score: {macro_f1:.4f}")
    print(f"Weighted F1-Score: {weighted_f1:.4f}")
    
    # Create confusion matrix using trimmed validation data
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_val_trimmed, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save confusion matrix plot
    confusion_matrix_path = os.path.join(base_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    print(f"\nConfusion matrix saved to: {confusion_matrix_path}")
    
    # Save model with .keras extension
    print("\nSaving model...")
    model.save(model_save_path)
    
    # Convert to TFLite
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Save TFLite model
    with open(tflite_save_path, 'wb') as f:
        f.write(tflite_model)
    
    print("Done!")

if __name__ == '__main__':
    main()