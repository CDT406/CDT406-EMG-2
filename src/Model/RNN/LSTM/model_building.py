from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer

def build_lstm_model(input_shape):
    model = Sequential([
        InputLayer(input_shape=input_shape, unroll=True),
        LSTM(15, unroll=True),
        Dense(6, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
