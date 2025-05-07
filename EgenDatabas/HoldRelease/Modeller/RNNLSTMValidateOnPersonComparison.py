import joblib
import numpy as np
from collections import defaultdict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score
from itertools import combinations

# ----------- Parameters -----------
sequence_length = 3
data_path = "EgenDatabas/HoldRelease/Pre-processing/processed_output/features_labels_W200_O100_WAMPth20_true_cycles_filtered.pkl"

# ----------- Load Data -----------
records = joblib.load(data_path)

# ----------- Group by Person and Cycle -----------
grouped = defaultdict(lambda: defaultdict(list))
for rec in records:
    features = np.array(rec["features"], dtype=np.float32)
    grouped[rec["person_id"]][rec["cycle_id"]].append((features, rec["label"]))

all_person_ids = sorted(grouped.keys())
results = []

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

# ----------- Loop Over All 2-Person Validation Splits -----------
for val_pair in combinations(all_person_ids, 2):
    train_ids = [pid for pid in all_person_ids if pid not in val_pair]
    val_ids = list(val_pair)

    X_train, y_train, X_val, y_val = [], [], [], []

    for person_id, cycles in grouped.items():
        all_samples = []
        for samples in cycles.values():
            all_samples.extend(samples)

        Xy = create_sequences(all_samples, sequence_length)
        if Xy is None:
            continue

        X, y = Xy
        if person_id in train_ids:
            X_train.append(X)
            y_train.append(y)
        elif person_id in val_ids:
            X_val.append(X)
            y_val.append(y)

    if not X_train or not X_val:
        print(f"Skipping split {val_ids} due to insufficient data.")
        continue

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_val = np.concatenate(X_val)
    y_val = np.concatenate(y_val)

    y_train_cat = to_categorical(y_train, num_classes=2)
    y_val_cat = to_categorical(y_val, num_classes=2)

    # ----------- Build and Train Model -----------
    model = Sequential([
        InputLayer(input_shape=(sequence_length, X_train.shape[2])),
        LSTM(3),
        Dense(2, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train_cat, epochs=15, batch_size=32, verbose=0)

    # ----------- Evaluate -----------
    y_pred = model.predict(X_val, verbose=0)
    y_pred_labels = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_val, y_pred_labels)

    print(f"\n🔁 Validation on persons {val_ids} → Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_val, y_pred_labels, target_names=["Rest", "Hold"]))

    results.append({
        "val_ids": val_ids,
        "accuracy": accuracy
    })

# ----------- Summary -----------
print("\n📊 Cross-validation summary:")
for r in results:
    print(f"Validation on {r['val_ids']} → Accuracy: {r['accuracy'] * 100:.2f}%")
