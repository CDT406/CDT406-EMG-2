import tensorflow as tf
import numpy as np
import keras
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from os_elm.os_elm import OS_ELM
from keras.datasets import mnist
from keras.utils import to_categorical
import logging
import tqdm

def softmax(a):
    c = np.max(a, axis=-1).reshape(-1, 1)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a, axis=-1).reshape(-1, 1)
    return exp_a / sum_exp_a

def mean_absolute_value(window):
    return np.mean(np.abs(window))

def waveform_length(window):
    return np.sum(np.abs(np.diff(window)))

def zero_crossings(window):
    return np.sum(np.diff(np.sign(window)) != 0)

def slope_sign_changes(window):
    return np.sum(np.diff(np.sign(np.diff(window))) != 0)

# Function to calculate time-domain features for each window
def extract_time_features(signal, window_size):
    features = []
    # Slide over the signal with the specified window size
    num_windows = len(signal) // window_size  # Total number of windows
    for i in range(num_windows):
        window = signal[i * window_size: (i + 1) * window_size]
        
        # Calculate time-domain features
        mean = np.mean(window)
        variance = np.var(window)
        skewness = stats.skew(window)
        kurtosis = stats.kurtosis(window)
        mav = mean_absolute_value(window)
        wl = waveform_length(window)
        zc = zero_crossings(window)
        ssc = slope_sign_changes(window)

        
        #features.append([mean, variance, skewness, kurtosis, mav, wl, zc, ssc])
        features.append([mean, variance, skewness, kurtosis])
    
    return np.array(features)

def process_multiple_sensors(sensor_data, window_size):
    all_features = []
    for signal in sensor_data:
        all_features.append(extract_time_features(signal, window_size))
    
    # Concatenate features from all sensors along the feature axis
    combined_features = np.hstack(all_features)
    return combined_features

def process_multiple_persons(person_data, window_size):
    all_features = []
    for signal in person_data:
        all_features.append(process_multiple_sensors(signal, window_size))
    
    # Concatenate features from all persons along the feature axis
    combined_features = np.hstack(all_features)
    return combined_features

def load_wyoflex(person):
    # Load the data from the CSV file
    data = np.empty((4,), dtype=object)  # Initialize an empty array to hold the data
    data[0] = pd.read_csv(f"Wyoflex/3 gester/VOLTAGE DATA/P{person}C1S1M6F1O2", header=None).values.flatten()
    data[1] = pd.read_csv(f"Wyoflex/3 gester/VOLTAGE DATA/P{person}C1S2M6F1O2", header=None).values.flatten()
    data[2] = pd.read_csv(f"Wyoflex/3 gester/VOLTAGE DATA/P{person}C1S3M6F1O2", header=None).values.flatten()
    data[3] = pd.read_csv(f"Wyoflex/3 gester/VOLTAGE DATA/P{person}C1S4M6F1O2", header=None).values.flatten()
    return data

def load_wyoflex_all():
    data = np.empty((28,), dtype=object)  # Initialize an empty array to hold the data
    for i in range(1,29):
       data[i-1] = load_wyoflex(i)
    return data

def main():
    logging.basicConfig(level=logging.ERROR)

    window_size = 200

    # Load the data
    x_original = load_wyoflex_all()
    x_train = process_multiple_persons(x_original, window_size)

    # Input data
    #x_original = load_wyoflex(1)
    #x_train = process_multiple_sensors(x_original, window_size)  # Process data for all sensors
    
    #SINGLE SENSOR
    #x_train =  pd.read_csv(f"Wyoflex/3 gester/VOLTAGE DATA/P1C1S1M1F1O1", header=None).values.flatten()
    #x_train = extract_time_features(x_train, window_size)  # Process data for all sensors
    
    t_train = np.zeros(x_train.shape[0])  # Initialize with zeros (13000 samples, now in windows)
    t_train[30:60] = 1  # Set labels 1 for samples from window 30 to 60 (6000 to 12000 index range)
    t_train[60:] = 2    # Set labels 2 for the remaining windows (12000 to 13000 index range)
    
    #x_Plot x_orfirst few rows (signals) for visualization
    # Plot the original signal

    # Plot the original signal
    #plt.figure(figsize=(14, 8))
    #plt.plot(x_original, label='Original Signal', alpha=0.7)

    # Plot the features
    #mav_values = x_train[:, 0]
    #wl_values = x_train[:, 1]
    #zc_values = x_train[:, 2]
    #ssc_values = x_train[:, 3]

    ## Create x-axis positions for the features
    #x_positions = np.arange(window_size // 2, window_size // 2 + len(x_train) * window_size, window_size)

    ## Plot each feature
    #plt.plot(x_positions, mav_values, 'o-', label='MAV', color='red')
    #plt.plot(x_positions, wl_values, 'o-', label='WL', color='green')
    #plt.plot(x_positions, zc_values, 'o-', label='ZC', color='blue')
    #plt.plot(x_positions, ssc_values, 'o-', label='SSC', color='purple')

    #plt.title('Original Signal with Overlapping Features')
    #plt.xlabel('Sample Index')
    #plt.ylabel('Amplitude')
    #plt.legend()
    #plt.show()

    # Shuffling x_train and t_train in sync
    indices_train = np.random.permutation(x_train.shape[0])
    x_train = x_train[indices_train]
    t_train = t_train[indices_train]

    # Train/test split
    x_train, x_test, t_train, t_test = train_test_split(x_train, t_train, test_size=0.2, random_state=42)
    
    # ===========================================
    # Instantiate os-elm
    # ===========================================
    n_input_nodes = x_train.shape[1] # Number of features for each sample
    n_hidden_nodes = 400
    n_output_nodes = 3

    os_elm = OS_ELM(
        # the number of input nodes.
        n_input_nodes=n_input_nodes,
        # the number of hidden nodes.
        n_hidden_nodes=n_hidden_nodes,
        # the number of output nodes.
        n_output_nodes=n_output_nodes,
        # loss function.
        # the default value is 'mean_squared_error'.
        # for the other functions, we support
        # 'mean_absolute_error', 'categorical_crossentropy', and 'binary_crossentropy'.
        loss='mean_squared_error',
        # activation function applied to the hidden nodes.
        # the default value is 'sigmoid'.
        # for the other functions, we support 'linear' and 'tanh'.
        # NOTE: OS-ELM can apply an activation function only to the hidden nodes.
        activation='sigmoid',
    )

    # ===========================================
    # Prepare dataset
    # ===========================================
    n_classes = n_output_nodes

    # normalize values within [0, 1]
    x_train = x_train.reshape(-1, n_input_nodes) 
    x_train = x_train.astype(np.float32)
    x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
    x_test = x_test.reshape(-1, n_input_nodes)
    x_test = x_test.astype(np.float32)
    x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))

    # convert label data into one-hot-vector format data.
    t_train = to_categorical(t_train, num_classes=n_classes)
    t_test = to_categorical(t_test, num_classes=n_classes)
    t_train = t_train.astype(np.float32)
    t_test = t_test.astype(np.float32)


    # divide the training dataset into two datasets:
    # (1) for the initial training phase
    # (2) for the sequential training phase
    # NOTE: the number of training samples for the initial training phase
    # must be much greater than the number of the model's hidden nodes.
    # here, we assign int(1.5 * n_hidden_nodes) training samples
    # for the initial training phase.
    border = int(1.5 * n_hidden_nodes)
    x_train_init = x_train[:border]
    x_train_seq = x_train[border:]
    t_train_init = t_train[:border]
    t_train_seq = t_train[border:]


    # ===========================================
    # Training
    # ===========================================
    # the initial training phase
    pbar = tqdm.tqdm(total=len(x_train), desc='initial training phase')
    os_elm.init_train(x_train_init, t_train_init)
    pbar.update(n=len(x_train_init))

    # the sequential training phase
    pbar.set_description('sequential training phase')
    batch_size = 64
    for i in range(0, len(x_train_seq), batch_size):
        x_batch = x_train_seq[i:i+batch_size]
        t_batch = t_train_seq[i:i+batch_size]
        os_elm.seq_train(x_batch, t_batch)
        pbar.update(n=len(x_batch))
    pbar.close()

    # ===========================================
    # Prediction
    # ===========================================
    # sample 10 validation samples from x_test
    n = 10
    x = x_test
    t = t_test

    # 'predict' method returns raw values of output nodes.
    y = os_elm.predict(x)
    # apply softmax function to the output values.
    y = softmax(y)
    
    # check the answers.
    for i in range(n):
        max_ind = np.argmax(y[i])
        print('========== sample index %d ==========' % i)
        print('estimated answer: class %d' % max_ind)
        print('estimated probability: %.3f' % y[i,max_ind])
        print('true answer: class %d' % np.argmax(t[i]))

    # ===========================================
    # Evaluation
    # ===========================================
    # we currently support 'loss' and 'accuracy' for 'metrics'.
    # NOTE: 'accuracy' is valid only if the model assumes
    # to deal with a classification problem, while 'loss' is always valid.
    # loss = os_elm.evaluate(x_test, t_test, metrics=['loss']
    [loss, accuracy] = os_elm.evaluate(x_test, t_test, metrics=['loss', 'accuracy'])
    print('val_loss: %f, val_accuracy: %f' % (loss, accuracy))

    # ===========================================
    # Save model
    # ===========================================
    print('saving model parameters...')
    # Create a dummy input to trace the model
    x_dummy = np.random.rand(1, n_input_nodes).astype(np.float32)

    # Wrap the model's predict method with tf.function to generate a concrete function
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, n_input_nodes], dtype=tf.float32)])
    def predict_fn(x):
        return os_elm.predict(x)

    # Get the concrete function
    concrete_func = predict_fn.get_concrete_function()

    # Convert the concrete function to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], trackable_obj=os_elm)
    tflite_model = converter.convert()

    with open('elm_model.tflite', 'wb') as f:     
        f.write(tflite_model)



if __name__ == '__main__':
    main()