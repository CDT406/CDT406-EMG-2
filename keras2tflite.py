import tensorflow as tf
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python keras2tflite.py <keras_model_path> [output_tflite_path]")
    sys.exit(1)

keras_model_path = sys.argv[1]
output_tflite_path = sys.argv[2] if len(sys.argv) > 2 else "model.tflite"

# Load the given Keras model
model = tf.keras.models.load_model(keras_model_path)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open(output_tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model saved to {output_tflite_path}")