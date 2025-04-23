import tensorflow as tf
import numpy as np
import keras
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split


@keras.saving.register_keras_serializable()
class ELMModel(tf.keras.Model):
    def __init__(self, input_dim, hidden_units, num_classes):
        super(ELMModel, self).__init__()
        # Initialize the hidden layer weights randomly
        self.hidden_weights = tf.Variable(np.random.randn(input_dim, hidden_units), trainable=False, dtype=tf.float32)
        self.hidden_bias = tf.Variable(np.random.randn(hidden_units), trainable=False, dtype=tf.float32)
        
        # Output layer (trainable)
        self.output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')  # For classification

    def call(self, inputs):
        # Compute hidden layer outputs (no training, just random weights)
        hidden_output = tf.nn.relu(tf.matmul(inputs, self.hidden_weights) + self.hidden_bias)
        
        # Compute output layer
        output = self.output_layer(hidden_output)
        return output

# Function to calculate time-domain features for each window
def extract_time_features(signal, window_size=200):
    features = []
    # Slide over the signal with the specified window size
    num_windows = len(signal) // window_size  # Total number of windows
    for i in range(num_windows):
        window = signal[i * window_size: (i + 1) * window_size]
        
        # Calculate time-domain features
        mean = np.mean(window)
        variance = np.var(window)
        skewness = stats.skew(window)
        kurtosis = stats.kurtosis(window)
        
        features.append([mean, variance, skewness, kurtosis])
    
    return np.array(features)

# Example usage:
input_dim = 1  # Example input dimension
hidden_units = 10  # Number of hidden neurons
num_classes = 3  # 3 classes (0, 1, 2)
model = ELMModel(input_dim, hidden_units, num_classes)

# Input data
df = pd.read_csv("data/raw/M6 Dataset/subject #1/cycle #1/P1C1S1M6F1O2", header=None)
# Assuming that each sample has multiple features and represents a time-series of sEMG data
# Plot the first few rows (signals) for visualization

X_train = df.values

# Convert data from column vector to row vector (shape = 13000, 1)
X_train = df.values
X_train = X_train.flatten()  # Flatten to 1D array if it's 2D column vector
X_features = extract_time_features(X_train)

Y_train = np.zeros(X_features.shape[0])  # Initialize with zeros (13000 samples, now in windows)
Y_train[30:60] = 1  # Set labels 1 for samples from window 30 to 60 (6000 to 12000 index range)
Y_train[60:] = 2    # Set labels 2 for the remaining windows (12000 to 13000 index range)

# Ensure the data has the correct shape, it's a single row with 13000 features
print("Shape of data:", X_train.shape)

plt.figure(figsize=(12, 6))
plt.plot(X_features[:200, 0])  # Plot the mean of each window as an example
plt.title("Extracted Feature: Mean of Each Window")
plt.xlabel("Window Index")
plt.ylabel("Mean Value")
plt.legend()
plt.show()

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y_train, test_size=0.2, random_state=42)

# Build the model
input_dim = X_train.shape[1]  # Number of features for each sample
hidden_units = 10  # Number of hidden units (neurons)
num_classes = 3  # Number of classes
model = ELMModel(input_dim, hidden_units, num_classes)

# Compile the model
model.compile(optimizer=tf.optimizers.SGD(learning_rate=0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=1)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f"Test accuracy: {test_acc}")

converter = tf.lite.TFLiteConverter.from_keras_model(model) 
tflite_model = converter.convert()

with open('elm_model.tflite', 'wb') as f:     
  f.write(tflite_model)