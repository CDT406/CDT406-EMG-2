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

# Example usage:
input_dim = 5  # Example input dimension
hidden_units = 10  # Number of hidden neurons
model = ELMModel(input_dim, hidden_units)

# Example input data (batch size of 1)
input_data = np.random.randn(1, input_dim).astype(np.float32)

# Inference
output = model(input_data)
print("ELM-like output:", output)

# Training data (for example)
X_train = np.random.randn(100, input_dim).astype(np.float32)  # 100 samples, 5 features
y_train = np.random.randn(100, 1).astype(np.float32)  # 100 target values

# Training loop
optimizer = tf.optimizers.SGD(learning_rate=0.01)

for epoch in range(100):
    with tf.GradientTape() as tape:
        # Forward pass
        predictions = model(X_train)
        loss = tf.reduce_mean(tf.square(predictions - y_train))  # Mean squared error loss

    # Compute gradients and update output layer
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")

model.save("elm_model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model) 
tflite_model_quantized = converter.convert()

with open('elm_model.tflite', 'wb') as f:     
  f.write(tflite_model_quantized)