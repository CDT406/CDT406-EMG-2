import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, Activation, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Function to load Ninapro DB1 dataset for a specific subject
def load_ninapro_db1(subject_id):
    base_path = f'NinaPro_DB1/s{subject_id}/'
    emg_data = scipy.io.loadmat(base_path + f'S{subject_id}_A1_E1.mat')['emg']
    repetition_data = scipy.io.loadmat(base_path + f'S{subject_id}_A1_E1.mat')['rerepetition']
    stimulus_data = scipy.io.loadmat(base_path + f'S{subject_id}_A1_E1.mat')['restimulus']
    return emg_data, repetition_data, stimulus_data

# Function to segment the data into fixed-length sequences
def segment_data(emg_data, stimulus_data, sequence_length):
    X, y = [], []
    for i in range(0, len(emg_data) - sequence_length, sequence_length):
        X.append(emg_data[i:i + sequence_length])
        y.append(stimulus_data[i + sequence_length - 1])
    return np.array(X), np.array(y)

# Function to build the TCN model
def build_tcn_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=3, padding='causal', dilation_rate=1)(inputs)
    x = Activation('relu')(x)
    x = Dropout(0.05)(x)
    x = Conv1D(filters=64, kernel_size=3, padding='causal', dilation_rate=2)(x)
    x = Activation('relu')(x)
    x = Dropout(0.05)(x)
    x = Conv1D(filters=64, kernel_size=3, padding='causal', dilation_rate=4)(x)
    x = Activation('relu')(x)
    x = Dropout(0.05)(x)
    x = Conv1D(filters=64, kernel_size=3, padding='causal', dilation_rate=8)(x)
    x = Activation('relu')(x)
    x = Dropout(0.05)(x)
    x = GlobalAveragePooling1D()(x)  # Pooling to get a single output per sequence
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train and evaluate the model for each subject
def train_and_evaluate(subject_id):
    emg_data, repetition_data, stimulus_data = load_ninapro_db1(subject_id)
    
    # Segment the data into fixed-length sequences
    sequence_length = 300  # Example sequence length
    X, y = segment_data(emg_data, stimulus_data, sequence_length)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Convert labels to categorical format
    num_classes = len(np.unique(stimulus_data))
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    # Build the TCN model
    input_shape = (sequence_length, X_train.shape[2])
    tcn_model = build_tcn_model(input_shape, num_classes)

    # Train the model on the Ninapro dataset
    tcn_model.fit(X_train, y_train_cat, epochs=30, batch_size=128, validation_data=(X_test, y_test_cat))

    # Evaluate the model on the test set
    loss, accuracy = tcn_model.evaluate(X_test, y_test_cat)
    print(f"Subject {subject_id} - Test Loss: {loss}, Test Accuracy: {accuracy}")

    # Calculate top-1 and top-3 accuracy
    y_pred = tcn_model.predict(X_test)
    top1_accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test_cat, axis=1))
    top3_accuracy = np.mean([np.argmax(y_test_cat[i]) in np.argsort(y_pred[i])[-3:] for i in range(len(y_test_cat))])
    print(f"Subject {subject_id} - Top-1 Accuracy: {top1_accuracy}, Top-3 Accuracy: {top3_accuracy}")

# Train and evaluate the model for all subjects
num_subjects = 27  # Number of subjects in the Ninapro DB1 dataset
for subject_id in range(1, num_subjects + 1):
    train_and_evaluate(subject_id)
