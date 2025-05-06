import joblib
import numpy as np
from collections import defaultdict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import tensorflow as tf

def convert_keras_to_tflite(keras_model, output_path):
    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    return tflite_model

# ----------- Parameters -----------
sequence_length = 3
data_path = "EgenDatabas/HoldRelease/Pre-processing/processed_output/features_labels_W200_O100_WAMPth20_true_cycles.pkl"

# ----------- Load Processed Data -----------
records = joblib.load(data_path)

# ----------- Group by Person and Cycle -----------
grouped = defaultdict(lambda: defaultdict(list))
for rec in records:
    features = np.array(rec["features"], dtype=np.float32)
    grouped[rec["person_id"]][rec["cycle_id"]].append((features, rec["label"]))

# ----------- Sequence Builder -----------
def create_sequences(samples, seq_len=1):
    if len(samples) < seq_len:
        return None, None
    features, labels = zip(*samples)
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i:i + seq_len])
        y.append(labels[i + seq_len - 1])
    return np.array(X), np.array(y)

# ----------- Split Persons: 1–7 train, 8–9 validate -----------
X_train, y_train, X_val, y_val = [], [], [], []

for person_id, cycles in grouped.items():
    # Sort for consistency (not required but clean)
    if person_id <= 7:
        for samples in cycles.values():
            Xt, yt = create_sequences(samples, sequence_length)
            if Xt is not None:
                X_train.append(Xt)
                y_train.append(yt)
    elif person_id >= 8:
        for samples in cycles.values():
            Xv, yv = create_sequences(samples, sequence_length)
            if Xv is not None:
                X_val.append(Xv)
                y_val.append(yv)

# ----------- Combine & Encode -----------
if not X_train or not X_val:
    raise ValueError("❌ No valid training or validation data found.")

X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)
X_val = np.concatenate(X_val)
y_val = np.concatenate(y_val)

y_train_cat = to_categorical(y_train, num_classes=2)
y_val_cat = to_categorical(y_val, num_classes=2)

# ----------- Define LSTM Model -----------
model = Sequential([
    InputLayer(input_shape=(sequence_length, X_train.shape[2]), unroll=True),
    LSTM(15, unroll=True),
    Dense(6, activation='relu'),
    Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ----------- Train Model -----------
history = model.fit(
    X_train, y_train_cat,
    epochs=15,
    batch_size=32,
    validation_data=(X_val, y_val_cat),
    verbose=1
)

# ----------- Evaluate Model -----------
loss, accuracy = model.evaluate(X_val, y_val_cat, verbose=0)
print(f"\n✅ Validation Accuracy: {accuracy * 100:.2f}%")

# ----------- Classification Report -----------
y_pred = model.predict(X_val, verbose=0)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_val_cat, axis=1)

print("\n--- Classification Report (Rest vs Hold) ---")
print(classification_report(y_true_labels, y_pred_labels, target_names=["Rest", "Hold"]))

# Convert the Keras model to TFLite
output_tflite_path = "rnn.tflite"
tflite_model = convert_keras_to_tflite(model, output_tflite_path)
# Save the model.

with open(output_tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model saved to {output_tflite_path}")
