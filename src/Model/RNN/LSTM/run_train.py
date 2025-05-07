import joblib
from model_building import build_lstm_model
from sequence_utils import split_by_person
from trainer import train_and_evaluate
from conversion import convert_keras_to_tflite

def run_training(data_path, sequence_length, tflite_output_path, val_ids={1, 2}):
    """
    Trains LSTM model using persons not in val_ids for training and val_ids for validation.
    """
    records = joblib.load(data_path)
    X_train, y_train, X_val, y_val = split_by_person(records, sequence_length, val_ids)

    if not X_train or not X_val:
        raise ValueError("❌ No valid training or validation data found.")

    input_shape = (sequence_length, X_train[0].shape[2])
    model = build_lstm_model(input_shape)

    accuracy, n_train, n_val = train_and_evaluate(model, X_train, y_train, X_val, y_val)

    convert_keras_to_tflite(model, tflite_output_path)
    return accuracy, n_train, n_val
