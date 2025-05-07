import numpy as np
import tflite_runtime.interpreter as tflite
import unittest

RNNLSTM_PATH = 'model.tflite'
GRU_PATH = 'gru_model.tflite'
LSTM_PATH = 'LSTM_model.tflite'


# Model runner
def get_prediction(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare random data
    input_shape = input_details[0]['shape']
    input_data = np.random.random_sample(input_shape).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    interpreter.invoke()  # Run the model
    
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data


#  Test classes for every model

class LSTMTest(unittest.TestCase):
    model_path = LSTM_PATH
        
    def test_compare_with_tflite(self):
        tflite_y_predictions = get_prediction(self.model_path)
        print("Predictions for floating-point model:", tflite_y_predictions)
        
        
class GRUTest(unittest.TestCase):
    model_path = GRU_PATH
        
    def test_compare_with_tflite(self):
        tflite_y_predictions = get_prediction(self.model_path)
        print("Predictions for floating-point model:", tflite_y_predictions)
        
        
class RNNLSTMTest(unittest.TestCase):
    model_path = RNNLSTM_PATH
        
    def test_compare_with_tflite(self):
        tflite_y_predictions = get_prediction(self.model_path)
        print("Predictions for floating-point model:", tflite_y_predictions)


if __name__ == '__main__':
    unittest.main()

