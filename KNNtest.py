import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ----------- Load features and labels -----------

load_path = r"datasets/Wyoflex/3 gester/processed/features_labels_5_5to8_5s_O2.pkl"

# Load X and y
X, y = joblib.load(load_path)

print(f"Loaded feature matrix shape: {X.shape}")
print(f"Loaded label vector shape: {y.shape}")

# ----------- Split into train and test sets -----------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

# ----------- Train KNN classifier -----------

knn = KNeighborsClassifier(n_neighbors=11)  # 5 neighbors
knn.fit(X_train, y_train)

# ----------- Evaluate -----------

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ KNN Test Accuracy: {accuracy*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["M1", "M2", "M6"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - KNN Classifier")
plt.show()
