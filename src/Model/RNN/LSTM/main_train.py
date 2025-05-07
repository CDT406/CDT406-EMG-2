import joblib
from model_building import build_lstm_model
from sequence_utils import split_by_person
from trainer import train_and_evaluate
from conversion import convert_keras_to_tflite

# ----------- Parameters -----------
sequence_length = 3
data_path = "output/ProcessedData/features_labels_W200_O100_WAMPth20_true_cycles_filtered.pkl"
tflite_output_path = "output/SavedModels/RNN/rnn_lstm_model.tflite"

# ----------- Load Data -----------
records = joblib.load(data_path)

# ----------- Prepare Sequences -----------
X_train, y_train, X_val, y_val = split_by_person(records, sequence_length)

if not X_train or not X_val:
    raise ValueError("❌ No valid training or validation data found.")

# ----------- Build and Train Model -----------
input_shape = (sequence_length, X_train[0].shape[2])
model = build_lstm_model(input_shape)
model = train_and_evaluate(model, X_train, y_train, X_val, y_val)

# ----------- Convert to TFLite -----------
convert_keras_to_tflite(model, tflite_output_path)
