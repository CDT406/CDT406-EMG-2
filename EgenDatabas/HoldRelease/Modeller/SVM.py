import joblib
import numpy as np
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# ----------- Parameters -----------
data_path = "EgenDatabas/HoldRelease/Pre-processing/processed_output/features_labels_W200_O100_WAMPth20_true_cycles.pkl"
train_ids = set(range(1, 8))  # Persons 1–7
val_ids = set(range(8, 10))   # Persons 8–9

# ----------- Load Data -----------
records = joblib.load(data_path)

# ----------- Group by Person and Cycle -----------
grouped = defaultdict(lambda: defaultdict(list))
for rec in records:
    features = np.array(rec["features"], dtype=np.float32)
    grouped[rec["person_id"]][rec["cycle_id"]].append((features, rec["label"]))

# ----------- Split and Flatten Samples -----------
X_train, y_train, X_val, y_val = [], [], [], []

for person_id, cycles in grouped.items():
    all_samples = []
    for samples in cycles.values():
        all_samples.extend(samples)

    if person_id in train_ids:
        for feat, label in all_samples:
            X_train.append(feat)
            y_train.append(label)

    elif person_id in val_ids:
        for feat, label in all_samples:
            X_val.append(feat)
            y_val.append(label)

# ----------- Scale Features (Important for SVM) -----------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# ----------- Train SVM -----------
svm = SVC(kernel='rbf', C=1.0, gamma='scale')  # You can experiment with kernel, C, gamma
svm.fit(X_train, y_train)

# ----------- Predict & Evaluate -----------
y_pred = svm.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print(f"\n✅ Validation Accuracy: {accuracy * 100:.2f}%")
print("\n--- Classification Report (Rest vs Hold) ---")
print(classification_report(y_val, y_pred, target_names=["Rest", "Hold"]))
