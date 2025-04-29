import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import ninapro_dataload as nd
from pprint import pp


def normalize(df):
    scaled = MinMaxScaler(feature_range=(0, 1))
    return scaled.fit_transform(df)


def create_dataset(data, time_step=1):
    x, y = [], []

    for i in range(len(data) - time_step - 1):
        x.append(data[i:i + time_step, 0])
        y.append(data[i + time_step, 0])

    return np.array(x), np.array(y)


if __name__ == "__main__":
    emg_df, data_dict = nd.get_emg()
    test_emg = nd.get_testing_data()

    # Get EMG data
    scaled_data = normalize(emg_df)
    scaled_test_data = normalize(test_emg)
    
    # Create dataset
    x, y = create_dataset(scaled_data, 1)  # time_step = 100
    x_test, y_test = create_dataset(scaled_test_data, 100)
    
    # Reshape x for GRU input: (samples, timesteps, features)
    x = x.reshape((x.shape[0], x.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    
    # One-hot encode y for 3 classes
    y = to_categorical(y, num_classes=3)
    y_test = to_categorical(y_test, num_classes=3)

    model = Sequential()
    
    model.add(GRU(units=32, return_sequences=True, input_shape=(x.shape[1], 1)))
    model.add(GRU(units=16))
    model.add(Dense(units=3, activation='softmax'))  # Output layer with 3 units
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['accuracy']
        )
    
    model.fit(x, y, epochs=5, batch_size=16)
    
    loss = model.evaluate(x_test, y_test, batch_size=16)
    print(f"Test loss: {loss}")
    
