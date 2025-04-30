import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Flatten, InputLayer
from tensorflow.keras.utils import to_categorical

# ----------- Parameters -----------
sequence_length = 10   # How many windows to stack into one sequence
num_classes = 3  # M1, M2, M6

# ----------- Load Features and Labels -----------
load_path = r"Wyoflex/3 gester/DynamicProcessed/4channel/features_labels_ALL_4channels_dynamic_O2_W64.pkl"

X, y = joblib.load(load_path)
print(f"Loaded feature matrix shape: {X.shape}")
print(f"Loaded label vector shape: {y.shape}")

# ----------- Preprocessing -----------

# Flatten channels/features per window for scaling
n_windows, n_channels, n_features = X.shape
X_flat = X.reshape(n_windows, n_channels * n_features)

# Normalize features
scaler = StandardScaler()
X_flat = scaler.fit_transform(X_flat)

# Restore shape: (n_windows, channels, features)
X = X_flat.reshape(n_windows, n_channels, n_features)

# ----------- Create Sequences -----------

def create_sequences(X, y, sequence_length):
    X_seq = []
    y_seq = []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length-1])  # use label of last window
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
    InputLayer(input_shape=(sequence_length, n_channels, n_features)),
    TimeDistributed(Flatten()),   # flatten (4 channels × 7 features) into one vector per timestep
    LSTM(64),
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
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ----------- Evaluate -----------

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ LSTM Test Accuracy: {test_accuracy*100:.2f}%")

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
