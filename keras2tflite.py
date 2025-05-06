import tensorflow as tf
import sys
import os

def convert_keras_to_tflite(keras_model, output_path):
    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    return tflite_model

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python keras2tflite.py <keras_model_path> [output_tflite_path]")
        sys.exit(1)

    keras_model_path = sys.argv[1]
    output_tflite_path = sys.argv[2] if len(sys.argv) > 2 else "model.tflite"

    # Load the given Keras model
    model = tf.keras.models.load_model(keras_model_path)
    # Convert the Keras model to TFLite
    tflite_model = convert_keras_to_tflite(model, output_tflite_path)
    # Save the model.

    with open(output_tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {output_tflite_path}")