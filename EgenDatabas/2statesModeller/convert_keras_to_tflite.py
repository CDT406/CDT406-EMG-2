import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model("model.keras")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model to disk
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Converted to TensorFlow Lite successfully!")