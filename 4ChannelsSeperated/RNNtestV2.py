import joblib
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Flatten, InputLayer, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# ----------- Parameters -----------
sequence_length = 3
num_classes = 3  # M1, M2, M6 

# ----------- Load Preprocessed Data -----------
load_path = r"datasets/Wyoflex/3 gester/FixedSegmentProcessed/4channel_byfile/features_by_file_5to9s_W256.pkl"
data_by_file = joblib.load(load_path)

# ----------- Split by Cycle -----------
train_data = []
val_data = []

for fname, (X_file, y_file) in data_by_file.items():
    if "C2" in fname:
        val_data.append((X_file, y_file))
    elif "C1" in fname or "C3" in fname:
        train_data.append((X_file, y_file))

print(f"✅ Training files: {len(train_data)}, Validation files: {len(val_data)}")

# ----------- Create Sequences Function -----------

def create_sequences(X, y, sequence_length):
    X_seq = []
    y_seq = []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length-1])
    return np.array(X_seq), np.array(y_seq)

# ----------- Normalize Across All Training Data -----------

# Flatten and gather all training windows
X_train_all = np.concatenate([X for X, _ in train_data])
X_train_flat = X_train_all.reshape(X_train_all.shape[0], -1)

scaler = StandardScaler()
X_train_flat = scaler.fit_transform(X_train_flat)
X_train_all = X_train_flat.reshape(X_train_all.shape[0], 4, 7)

# Reassign normalized X back to train_data
i = 0
for idx in range(len(train_data)):
    X, y = train_data[idx]
    n = X.shape[0]
    train_data[idx] = (X_train_all[i:i+n], y)
    i += n

# Normalize validation set using same scaler
X_val_all = np.concatenate([X for X, _ in val_data])
X_val_flat = X_val_all.reshape(X_val_all.shape[0], -1)
X_val_flat = scaler.transform(X_val_flat)
X_val_all = X_val_flat.reshape(X_val_all.shape[0], 4, 7)

# Reassign normalized X back to val_data
i = 0
for idx in range(len(val_data)):
    X, y = val_data[idx]
    n = X.shape[0]
    val_data[idx] = (X_val_all[i:i+n], y)
    i += n

# ----------- Build Final Datasets -----------

X_train_seq, y_train = [], []
X_val_seq, y_val = [], []

for X_file, y_file in train_data:
    Xs, ys = create_sequences(X_file, y_file, sequence_length)
    X_train_seq.append(Xs)
    y_train.append(ys)

for X_file, y_file in val_data:
    Xs, ys = create_sequences(X_file, y_file, sequence_length)
    X_val_seq.append(Xs)
    y_val.append(ys)

X_train = np.concatenate(X_train_seq)
y_train = np.concatenate(y_train)
X_val = np.concatenate(X_val_seq)
y_val = np.concatenate(y_val)

print(f"✅ Final X_train: {X_train.shape}, X_val: {X_val.shape}")

# ----------- One-hot encode labels -----------
y_train_cat = to_categorical(y_train, num_classes)
y_val_cat = to_categorical(y_val, num_classes)

# ----------- Build LSTM Model -----------

model = Sequential([
    InputLayer(input_shape=(sequence_length, 4, 7)),
    TimeDistributed(Flatten()),
    LSTM(32),
    Dropout(0.4),
    Dense(16, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ----------- Train Model -----------

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train_cat,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val_cat),
    callbacks=[early_stop],
    verbose=1
)

# ----------- Evaluate -----------

val_loss, val_accuracy = model.evaluate(X_val, y_val_cat, verbose=0)
print(f"\n✅ Validation Accuracy (C2): {val_accuracy*100:.2f}%")

# ----------- Plot Training History -----------

plt.figure(figsize=(10, 4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# ----------- Confusion Matrix -----------

y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_val_cat, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["M1", "M2", "M6"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Validation (Cycle C2)")
plt.show()
