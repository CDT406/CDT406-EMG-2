import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from ninapro_dataload import get_emg
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
    emg_df, data_dict = get_emg()

    scaled_data = normalize(emg_df)
    x, y = create_dataset(scaled_data, 100)  # time_step = 100

    model = Sequential()
    model.add(GRU(units=50, return_sequences=True, input_shape=(x.shape[1], 1)))
    model.add(GRU(units=50))
    model.add(Dense(units=3)) 
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    model.fit(x, y, epochs=10, batch_size=32)
