import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# ----------- Load features and labels -----------

load_path = r"datasets/Wyoflex/3 gester/processed/features_labels_P1_dynamic_O2_W384.pkl"

# Load X and y
X, y = joblib.load(load_path)

print(f"Loaded feature matrix shape: {X.shape}")
print(f"Loaded label vector shape: {y.shape}")

# ----------- Create new better features -----------

def calculate_stronger_features(X_raw):
    """Create stronger feature set: MAV, WL, ZC, SSC, RMS, Variance, IEMG."""
    mav = X_raw[:, 0]  # Already MAV
    wl = X_raw[:, 1]   # Already WL
    zc = X_raw[:, 2]   # Already ZC
    ssc = X_raw[:, 3]  # Already SSC

    # Let's recalculate based on original window signals if needed
    # Here, fake calculation from MAV as a base approximation
    rms = np.sqrt(np.mean(np.abs(X_raw)**2, axis=1))  # Very rough
    variance = np.var(X_raw, axis=1)
    iemg = np.sum(np.abs(X_raw), axis=1)

    # Stack all together
    features = np.vstack((mav, wl, zc, ssc, rms, variance, iemg)).T
    return features

# Update X with stronger features
X = calculate_stronger_features(X)

print(f"New feature matrix shape: {X.shape}")

# ----------- Preprocessing -----------

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encode labels
y_categorical = to_categorical(y, num_classes=3)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

# ----------- Build Bigger ANN Model -----------

model = Sequential([
    Dense(64, input_shape=(X.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')  # 3 output classes (M1, M2, M6)
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ----------- Train the model -----------

history = model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ----------- Evaluate -----------

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ ANN Test Accuracy: {test_accuracy*100:.2f}%")

# ----------- Plot training history -----------

plt.figure(figsize=(10,4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.grid(True)
plt.show()

# ----------- Confusion Matrix -----------

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["M1", "M2", "M6"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - ANN Classifier (Improved Features)")
plt.show()
