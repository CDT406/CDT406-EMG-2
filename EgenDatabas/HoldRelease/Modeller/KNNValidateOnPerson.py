import joblib
import numpy as np
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# ----------- Parameters -----------
data_path = "EgenDatabas/HoldRelease/Pre-processing/processed_output/features_labels_W200_O100_WAMPth20_true_cycles.pkl"
train_ids = set(range(1, 8))  # Persons 1–7
val_ids = set(range(8, 10))   # Persons 8–9
k_values = list(range(1, 21))  # Try k from 1 to 20

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

# ----------- Tune k -----------
best_k = None
best_accuracy = 0.0
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    accuracies.append((k, acc))
    print(f"k={k:2d} → Validation Accuracy: {acc * 100:.2f}%")

    if acc > best_accuracy:
        best_accuracy = acc
        best_k = k

# ----------- Final Model with Best k -----------
print(f"\n✅ Best k = {best_k} with accuracy = {best_accuracy * 100:.2f}%")

best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)
y_pred_final = best_knn.predict(X_val)

print("\n--- Classification Report (Rest vs Hold) ---")
print(classification_report(y_val, y_pred_final, target_names=["Rest", "Hold"]))
