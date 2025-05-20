import os
import json
import numpy as np
import tensorflow as tf
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
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
    """Create sequences of 3 consecutive windows with their features."""
    X, y = [], []
    
    for person_id, cycles in data.items():
        for cycle_id, windows in cycles.items():
            features = [[w['MAV'], w['WL'], w['WAMP'], w['MAVS']] for w in windows]
            labels = [w['label'] for w in windows]
            
            # Create sequences
            for i in range(len(features) - sequence_length + 1):
                seq = features[i:i + sequence_length]
                X.append(np.array(seq).flatten())  # Flatten 3x4 into 12 features
                y.append(labels[i + sequence_length - 1])  # Label of newest window
    
    return np.array(X), np.array(y)

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

def build_model(num_features=12, num_classes=4):
    """Build the LSTM model."""
    model = Sequential([
        tf.keras.layers.Input(shape=(12,), name='input'),
        tf.keras.layers.Reshape((3, 4), name='reshape'),
        LSTM(64, 
             return_sequences=False, 
             stateful=False,
             unroll=True,  # Added unroll parameter
             implementation=1,
             name='lstm'),
        Dense(32, activation='relu', name='dense1'),
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
    
    # Create sequences
    print("Creating sequences...")
    X_train, y_train = create_sequences(train_data)
    X_val, y_val = create_sequences(val_data)
    
    # Build and train model (no normalization needed)
    print("Building and training model...")
    model = build_model()
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            )
        ]
    )
    
    # Evaluate model performance per class
    print("\nEvaluating model performance per class...")
    
    # Get predictions
    y_pred = np.argmax(model.predict(X_val), axis=1)
    
    # Print classification report
    class_names = ['Rest', 'Grip', 'Hold', 'Release']
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=class_names))
    
    # Create confusion matrix plot with updated labels
    plt.figure(figsize=(8, 6))  # Adjusted size for 4x4 matrix
    cm = confusion_matrix(y_val, y_pred)
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