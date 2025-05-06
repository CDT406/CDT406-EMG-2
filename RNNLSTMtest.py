import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from keras2tflite import convert_keras_to_tflite

# ----------- Parameters -----------
sequence_length = 3  # How many windows to stack into one sequence
num_classes = 3  # M1, M2, M6

# ----------- Load Features and Labels -----------
load_path = r"datasets/Wyoflex/3 gester/processed/features_labels_5_5to8_5s_O2.pkl"

X, y = joblib.load(load_path)
print(f"Loaded feature matrix shape: {X.shape}")
print(f"Loaded label vector shape: {y.shape}")

# ----------- Feature Engineering (Stronger Features) -----------

def calculate_stronger_features(X_raw):
    """Create stronger feature set: MAV, WL, ZC, SSC, RMS, Variance, IEMG."""
    mav = X_raw[:, 0]
    wl = X_raw[:, 1]
    zc = X_raw[:, 2]
    ssc = X_raw[:, 3]

    rms = np.sqrt(np.mean(np.abs(X_raw)**2, axis=1))
    variance = np.var(X_raw, axis=1)
    iemg = np.sum(np.abs(X_raw), axis=1)

    features = np.vstack((mav, wl, zc, ssc, rms, variance, iemg)).T
    return features

X = calculate_stronger_features(X)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ----------- Create Sequences -----------

# Function to create sequences
def create_sequences(X, y, sequence_length):
    X_seq = []
    y_seq = []
    for i in range(len(X) - sequence_length):
        # Take a sequence of windows
        X_seq.append(X[i:i+sequence_length])
        # Use label of last window in the sequence
        y_seq.append(y[i+sequence_length-1])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X, y, sequence_length)
print(f"Created sequences shape: {X_seq.shape}")
print(f"Corresponding labels shape: {y_seq.shape}")

# One-hot encode labels
y_seq_categorical = to_categorical(y_seq, num_classes=num_classes)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq_categorical, test_size=0.2, random_state=42, stratify=y_seq
)

print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

# ----------- Build LSTM Model -----------

model = Sequential([
    LSTM(64, input_shape=(sequence_length, X_seq.shape[2]), unroll=True),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ----------- Train the Model -----------

history = model.fit(
    X_train, y_train,
    epochs=1,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ----------- Evaluate -----------

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ LSTM Test Accuracy: {test_accuracy*100:.2f}%")

# Convert the Keras model to TFLite
output_tflite_path = "rnn.tflite"
tflite_model = convert_keras_to_tflite(model, output_tflite_path)
# Save the model.

with open(output_tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model saved to {output_tflite_path}")

# ----------- Plot Training History -----------

plt.figure(figsize=(10,4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy (LSTM)')
plt.grid(True)
plt.show()

# ----------- Confusion Matrix -----------

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["M1", "M2", "M6"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - LSTM Classifier")
plt.show()
