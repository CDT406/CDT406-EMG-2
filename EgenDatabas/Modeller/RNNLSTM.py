import joblib
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# ----------- Load Training Data -----------
train_path = "../Pre-processing/processed_output/features_labels_structured_W200_O100_WAMPth20.pkl"
train_records = joblib.load(train_path)
X_train_raw = np.array([rec['features'] for rec in train_records])
y_train_raw = np.array([rec['label'] for rec in train_records])

# ----------- Load Validation/Test Data -----------
val_path = "../Pre-processing/processed_output/original_features_labels_structured_W200_O100_WAMPth20.pkl"
val_records = joblib.load(val_path)
X_val_raw = np.array([rec['features'] for rec in val_records])
y_val_raw = np.array([rec['label'] for rec in val_records])

# ----------- Create Sequences -----------
sequence_length = 5

def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i + seq_len - 1])
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = create_sequences(X_train_raw, y_train_raw, sequence_length)
X_val_seq, y_val_seq = create_sequences(X_val_raw, y_val_raw, sequence_length)

# ----------- One-Hot Encode Labels -----------
y_train_cat = to_categorical(y_train_seq)
y_val_cat = to_categorical(y_val_seq)

# ----------- Define LSTM Model -----------
model = Sequential([
    InputLayer(input_shape=(sequence_length, X_train_raw.shape[1])),
    LSTM(32),
    Dense(32, activation='relu'),
    Dense(y_train_cat.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ----------- Train Model -----------
history = model.fit(
    X_train_seq, y_train_cat,
    epochs=15,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ----------- Evaluate Model -----------
loss, accuracy = model.evaluate(X_val_seq, y_val_cat)
print(f"\n✅ Validation Accuracy: {accuracy * 100:.2f}%")

y_pred = model.predict(X_val_seq)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_val_cat, axis=1)

print(classification_report(y_true_labels, y_pred_labels, target_names=["Rest", "Grip", "Hold", "Release"]))

model.save("semg_model.h5")