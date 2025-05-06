import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from numpy.linalg import pinv
import joblib
from collections import defaultdict

# ----------- Load Data -----------
data_path = "EgenDatabas/HoldRelease/Pre-processing/processed_output/features_labels_W200_O100_WAMPth20_true_cycles.pkl"
records = joblib.load(data_path)

# ----------- Split Persons -----------
grouped = defaultdict(lambda: defaultdict(list))
for rec in records:
    features = np.array(rec["features"], dtype=np.float32)
    grouped[rec["person_id"]][rec["cycle_id"]].append((features, rec["label"]))

train_ids = set(range(1, 8))
val_ids = set(range(8, 10))
X_train, y_train, X_val, y_val = [], [], [], []

for pid, cycles in grouped.items():
    all_samples = []
    for samples in cycles.values():
        all_samples.extend(samples)
    for feat, label in all_samples:
        if pid in train_ids:
            X_train.append(feat)
            y_train.append(label)
        elif pid in val_ids:
            X_val.append(feat)
            y_val.append(label)

X_train = np.array(X_train)
X_val = np.array(X_val)
y_train = np.array(y_train).reshape(-1, 1)
y_val = np.array(y_val).reshape(-1, 1)

# ----------- Normalize Features -----------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# ----------- One-Hot Encode Labels -----------
encoder = OneHotEncoder(sparse_output=False)
Y_train = encoder.fit_transform(y_train)

# ----------- Extreme Learning Machine -----------
input_dim = X_train.shape[1]
hidden_neurons = 100  # try 100, 200, etc.

# Random hidden layer weights and biases
W = np.random.randn(input_dim, hidden_neurons)
b = np.random.randn(hidden_neurons)

# Hidden layer output
def relu(x): return np.maximum(0, x)
H = relu(np.dot(X_train, W) + b)
H_val = relu(np.dot(X_val, W) + b)

# Solve output weights using Moore-Penrose pseudoinverse
beta = np.dot(pinv(H), Y_train)

# Predict
Y_pred_val = np.dot(H_val, beta)
y_pred = np.argmax(Y_pred_val, axis=1)
y_true = y_val.flatten()

# ----------- Results -----------
acc = accuracy_score(y_true, y_pred)
print(f"\n✅ ELM Validation Accuracy: {acc * 100:.2f}%")
print("\n--- Classification Report (Rest vs Hold) ---")
print(classification_report(y_true, y_pred, target_names=["Rest", "Hold"]))
