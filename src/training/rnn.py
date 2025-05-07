# --- train_rnn.py ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter

# --- Check and Configure GPU ---
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("GPU is available and will be used")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
else:
    print("No GPU available, using CPU")

# --- Set Path to Your Dataset ---
dataset_path = r'NinaPro_DB5'

# --- Let user choose which subjects to include ---
print("Available subjects: s1, s2, s3, s4, s5, s6, s7, s8, s9, s10")
selected_subjects = input("Enter subjects to include (comma-separated, e.g., 's1,s2,s3'): ").strip().split(',')
selected_subjects = [s.strip() for s in selected_subjects]

# Validate input
valid_subjects = [f's{i}' for i in range(1, 11)]
for subject in selected_subjects:
    if subject not in valid_subjects:
        raise ValueError(f"Invalid subject: {subject}. Valid subjects are: {', '.join(valid_subjects)}")

print(f"Selected subjects: {', '.join(selected_subjects)}")

# --- Load and Analyze Data ---
X_list = []
y_list = []

for subject in selected_subjects:
    subject_path = os.path.join(dataset_path, subject, subject)
    if not os.path.exists(subject_path):
        print(f"Warning: Directory {subject_path} not found, skipping...")
        continue
        
    print(f"\nLoading data from {subject_path}")
    for file in os.listdir(subject_path):
        if file.endswith('.mat'):
            file_path = os.path.join(subject_path, file)
            print(f"Loading {file_path}")
            data = scipy.io.loadmat(file_path)
            
            # Print data structure for analysis
            print("\nData structure in .mat file:")
            for key in data.keys():
                if not key.startswith('__'):
                    print(f"{key}: shape {data[key].shape}")
            
            # Extract EMG and stimulus data
            X = data['emg']
            y = data['stimulus']
            
            # Print data statistics
            print(f"\nEMG data statistics for {file}:")
            print(f"Shape: {X.shape}")
            print(f"Min value: {X.min()}")
            print(f"Max value: {X.max()}")
            print(f"Mean value: {X.mean()}")
            print(f"Std value: {X.std()}")
            
            print(f"\nStimulus data statistics for {file}:")
            print(f"Shape: {y.shape}")
            print(f"Unique classes: {np.unique(y)}")
            print(f"Class distribution: {Counter(y.flatten())}")
            
            X_list.append(X)
            y_list.append(y)

if not X_list:
    raise ValueError("No .mat files found in the selected subject directories!")

# --- Combine and Analyze Data ---
X = np.vstack(X_list)
y = np.vstack(y_list)

print('\n=== Combined Data Analysis ===')
print('Total X shape:', X.shape)
print('Total y shape:', y.shape)
print('Number of unique classes:', len(np.unique(y)))
print('Class distribution:', Counter(y.flatten()))

# --- Improved Data Preprocessing ---
# Standardize the data with better scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add min-max scaling to ensure values are between -1 and 1
X = np.clip(X, -3, 3)  # Clip outliers
X = (X - X.min()) / (X.max() - X.min()) * 2 - 1  # Scale to [-1, 1]

# --- Prepare Sequences with Overlap ---
timesteps = 100  # Increased sequence length for better temporal patterns
features = X.shape[1]

# Create sequences with 50% overlap for more training data
step_size = timesteps // 2
X_new = np.array([X[i:i+timesteps] for i in range(0, len(X) - timesteps, step_size)])
y_new = np.array([y[i+timesteps-1] for i in range(0, len(X) - timesteps, step_size)])

print('\n=== Prepared Data ===')
print('X_new shape:', X_new.shape)
print('y_new shape:', y_new.shape)
print('Number of sequences:', len(X_new))
print('Sequence length:', timesteps)
print('Features per timestep:', features)

# --- One-hot Encode Labels ---
from tensorflow.keras.utils import to_categorical

y_new = y_new.flatten()
y_new = to_categorical(y_new - 1)  # Make labels start from 0

# --- Split data into train and validation sets ---
X_train, X_val, y_train, y_val = train_test_split(X_new, y_new, test_size=0.2, random_state=42, stratify=y_new)

# --- Build Enhanced LSTM Model ---
model = Sequential([
    # First LSTM layer
    LSTM(256, activation='tanh', return_sequences=True, input_shape=(timesteps, features)),
    BatchNormalization(),
    Dropout(0.2),
    
    # Second LSTM layer
    LSTM(128, activation='tanh'),
    BatchNormalization(),
    Dropout(0.2),
    
    # Dense layers
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(y_new.shape[1], activation='softmax')
])

# --- Compile Model with optimized settings ---
initial_learning_rate = 0.001  # Reduced learning rate for stability

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
print('\n=== Model Summary ===')
print(f'Initial learning rate: {initial_learning_rate}')
model.summary()

# --- Add Improved Callbacks ---
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    mode='max'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    mode='max'
)

# --- Train Model with optimized epochs ---
print('\n=== Training Model ===')
history = model.fit(
    X_train, 
    y_train,
    epochs=50,  # Increased epochs for better learning
    batch_size=128,  # Reduced batch size for better generalization
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# --- Plot Training Results ---
plt.figure(figsize=(12, 4))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_results.png')
plt.show()

# --- Save Model ---
model.save('rnn_model.h5')
print("\n=== Results ===")
print("✅ Model trained and saved as 'rnn_model.h5'")
print("✅ Training results plotted and saved as 'training_results.png'")

# --- Print Final Accuracy ---
final_accuracy = max(history.history['val_accuracy'])
print(f"✅ Best Validation Accuracy: {final_accuracy:.4f}")

# --- Print Class-wise Accuracy ---
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_val, axis=1)

from sklearn.metrics import classification_report
print("\n=== Classification Report ===")
print(classification_report(y_true_classes, y_pred_classes))
