import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import ninapro_dataload as nd
from pprint import pp
import joblib


def load_data(load_path="Wyoflex/3 gester/processed/features_labels_5_5to8_5s_O2.pkl"):
    return joblib.load(load_path)


def preprocess_data(x, y):
    # Normalize features
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # One-hot encode labels
    y_categorical = to_categorical(y, num_classes=3)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y_categorical, test_size=0.2, random_state=42, stratify=y
    )
    return x_train, x_test, y_train, y_test


def build_model(input_shape):
    model = Sequential()
    model.add(GRU(units=32, return_sequences=True, input_shape=input_shape))
    model.add(GRU(units=16))
    model.add(Dense(units=3, activation='softmax'))  # Output layer with 3 units

    model.summary()
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['accuracy']
    )
    
    return model


def train_model(model, x_train, y_train, epochs=5, batch_size=16):
    return model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )


def evaluate_model(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=16)
    print(f"Test loss: {loss}")
    print(f"Test accuracy: {accuracy}")
    return loss, accuracy


def save_model(model, model_path="gru_model.h5"):
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    
def load_model(model_path="gru_model.h5"):
    from tensorflow.keras.models import load_model
    
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model


def plot_traning_history(history):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,4))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Load data
    x, y = load_data()

    # Preprocess data
    x_train, x_test, y_train, y_test = preprocess_data(x, y)

    # Reshape x for GRU input: (samples, timesteps, features)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # Build model
    model = build_model(input_shape=(x_train.shape[1], 1))

    # Train model
    history = train_model(model, x_train, y_train)
    
    plot_traning_history(history)

    # Evaluate model
    evaluate_model(model, x_test, y_test)

    # Save model
    save_model(model)
    
