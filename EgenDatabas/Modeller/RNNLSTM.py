import joblib
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ----------- Load Preprocessed Features and Labels -----------
data_path = "EgenDatabas/Pre-processing/processed_output/features_labels_structured_W200_O100_WAMPth20.pkl"
records = joblib.load(data_path)

X = np.array([rec['features'] for rec in records])
y = np.array([rec['label'] for rec in records])

# ----------- Create Sequences of Feature Vectors -----------
sequence_length = 5
X_seq, y_seq = [], []
for i in range(len(X) - sequence_length):
    X_seq.append(X[i:i+sequence_length])
    y_seq.append(y[i + sequence_length - 1])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)
y_seq_cat = to_categorical(y_seq)

# ----------- Split into Train/Test -----------
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq_cat, test_size=0.2, stratify=y_seq, random_state=42
)

# ----------- Define LSTM Model -----------
model = Sequential([
    InputLayer(input_shape=(sequence_length, X.shape[1])),
    LSTM(32),
    Dense(32, activation='relu'),
    Dense(y_seq_cat.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ----------- Train Model -----------
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ----------- Evaluate Model -----------
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")

y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print(classification_report(y_true_labels, y_pred_labels, target_names=["Rest", "Grip", "Hold", "Release"]))
