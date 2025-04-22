import tensorflow as tf
import numpy as np
import keras

@keras.saving.register_keras_serializable()
class ELMModel(tf.keras.Model):
    def __init__(self, input_dim, hidden_units):
        super(ELMModel, self).__init__()
        # Initialize the hidden layer weights randomly
        self.hidden_weights = tf.Variable(np.random.randn(input_dim, hidden_units), trainable=False, dtype=tf.float32)
        self.hidden_bias = tf.Variable(np.random.randn(hidden_units), trainable=False, dtype=tf.float32)
        
        # Output layer (trainable)
        self.output_layer = tf.keras.layers.Dense(1)  # For regression tasks (adjust for classification)

    def call(self, inputs):
        # Compute hidden layer outputs (no training, just random weights)
        hidden_output = tf.nn.relu(tf.matmul(inputs, self.hidden_weights) + self.hidden_bias)
        
        # Compute output layer
        output = self.output_layer(hidden_output)
        return output

keras_model = tf.keras.models.load_model('elm_model.keras')
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model) 

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model_quantized = converter.convert()

with open('elm_model.tflite', 'wb') as f:     
  f.write(tflite_model_quantized)