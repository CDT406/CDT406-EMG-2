from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer

def build_lstm_model(input_shape, num_classes=2):
    """
    Build LSTM model for EMG classification
    
    Args:
        input_shape: Tuple of (sequence_length, num_features)
        num_classes: Number of output classes (2 for rest/hold, 4 for rest/grip/hold/release)
    """
    model = Sequential([
        InputLayer(input_shape=input_shape, unroll=True),
        LSTM(15, unroll=True),
        Dense(8, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
