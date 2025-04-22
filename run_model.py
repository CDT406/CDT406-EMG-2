import tflite_runtime.interpreter as tflite
import numpy as np

# Load the model
interpreter = tflite.Interpreter(model_path="elm_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Example input data (must match the input shape of your model)
input_data = np.random.randn(1, 5).astype(np.float32)  # Example input with 5 features

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Model output:", output_data)