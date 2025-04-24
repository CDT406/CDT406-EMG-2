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
import tqdm

# Function to calculate time-domain features for each window
def extract_time_features(signal, window_size=200):
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
        
        features.append([mean, variance, skewness, kurtosis])
    
    return np.array(features)
# Ensure the data has the correct shape, it's a single row with 13000 features
#print("Shape of data:", x_train.shape)

#plt.figure(figsize=(12, 6))
#plt.plot(x_features[:200, 0])  # Plot the mean of each window as an example
#plt.title("Extracted Feature: Mean of Each Window")
#plt.xlabel("Window Index")
#plt.ylabel("Mean Value")
#plt.legend()
#plt.show()


def main():
    # Input data
    df = pd.read_csv("data/raw/M6 Dataset/subject #2/cycle #1/P2C1S1M6F1O2", header=None)
    # Assuming that each sample has multiple features and represents a time-series of sEMG data
    # Plot the first few rows (signals) for visualization

    x_train = df.values

    # Convert data from column vector to row vector (shape = 13000, 1)
    x_train = df.values
    x_train = x_train.flatten()  # Flatten to 1D array if it's 2D column vector
    x_features = extract_time_features(x_train)

    t_train = np.zeros(x_features.shape[0])  # Initialize with zeros (13000 samples, now in windows)
    t_train[30:60] = 1  # Set labels 1 for samples from window 30 to 60 (6000 to 12000 index range)
    t_train[60:] = 2    # Set labels 2 for the remaining windows (12000 to 13000 index range)
    # Train/test split
    x_train, x_test, t_train, t_test = train_test_split(x_features, t_train, test_size=0.2, random_state=42)
    
    # ===========================================
    # Instantiate os-elm
    # ===========================================
    n_input_nodes = x_train.shape[1] # Number of features for each sample
    n_hidden_nodes = 10
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

    # normalize images' values within [0, 1]
    #x_train = x_train.reshape(-1, n_input_nodes) / 255.
    #x_test = x_test.reshape(-1, n_input_nodes) / 255.
    #x_train = x_train.astype(np.float32)
    #x_test = x_test.astype(np.float32)

    # convert label data into one-hot-vector format data.
    #t_train = to_categorical(t_train, num_classes=n_classes)
    #t_test = to_categorical(t_test, num_classes=n_classes)
    #t_train = t_train.astype(np.float32)
    #t_test = t_test.astype(np.float32)

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
    x = x_test[:n]
    t = t_test[:n]

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
    converter = tf.lite.TFLiteConverter.from_keras_model(os_elm) 
    tflite_model = converter.convert()

    with open('elm_model.tflite', 'wb') as f:     
        f.write(tflite_model)



if __name__ == '__main__':
    main()