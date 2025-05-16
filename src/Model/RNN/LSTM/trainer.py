import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

def get_class_names(num_classes):
    """Get class names based on number of classes"""
    if num_classes == 2:
        return ["Rest", "Hold"]
    else:
        return ["Rest", "Grip", "Hold", "Release"]

def train_and_evaluate(model, X_train, y_train, X_val, y_val, num_classes=2, epochs=15, batch_size=32):
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_val = np.concatenate(X_val)
    y_val = np.concatenate(y_val)

    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)

    model.fit(
        X_train, y_train_cat,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val_cat),
        verbose=1
    )

    loss, accuracy = model.evaluate(X_val, y_val_cat, verbose=0)
    print(f"\n✅ Validation Accuracy: {accuracy * 100:.2f}%")

    y_pred = model.predict(X_val, verbose=0)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_val_cat, axis=1)

    class_names = get_class_names(num_classes)
    print(f"\n--- Classification Report ({' vs '.join(class_names)}) ---")
    print(classification_report(y_true_labels, y_pred_labels, target_names=class_names))

    return accuracy, len(y_train), len(y_val)
