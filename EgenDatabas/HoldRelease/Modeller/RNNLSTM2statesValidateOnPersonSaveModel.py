import joblib
import numpy as np
import os
from collections import defaultdict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import save_model
from sklearn.metrics import classification_report

# ----------- Parameters -----------
sequence_length = 3
data_path = "EgenDatabas/HoldRelease/Pre-processing/processed_output/features_labels_W200_O100_WAMPth20_true_cycles_filtered.pkl"
model_save_path = "EgenDatabas/HoldRelease/Modeller/SparadeModeller/lstm_holdrest_model.keras"

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
    InputLayer(input_shape=(sequence_length, X_train.shape[2])),
    LSTM(15),
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

# ----------- Save Model -----------
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)
print(f"\n✅ Model saved to: {model_save_path}")
