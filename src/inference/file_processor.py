from real_time_processor import RealTimeProcessor
import numpy as np
import time

class FileProcessor(RealTimeProcessor):
    def __init__(self, config, model_path, file_path):
        super().__init__(config)
        self.file_path = file_path
        self.data = None
        self.current_index = 0
        
        # Load the data
        try:
            self.data = np.loadtxt(file_path, delimiter=",", dtype=np.float32)
            print(f"Loaded {len(self.data)} samples from {file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {file_path}: {e}")
            
        # Initialize TFLite model
        from tflite_runtime.interpreter import Interpreter
        try:
            self.interpreter = Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Verify input shape matches our sequence configuration
            expected_shape = (1, self.sequence_length, 4)  # batch_size, sequence_length, n_features
            actual_shape = tuple(self.input_details[0]['shape'])
            if actual_shape != expected_shape:
                raise ValueError(f"Model input shape {actual_shape} does not match expected shape {expected_shape}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {e}")
    
    def _read_adc(self):
        """Read from file instead of ADC"""
        if self.current_index >= len(self.data):
            self.running = False  # Stop when we reach the end of file
            return None
            
        value = self.data[self.current_index]
        # Convert ADC value to voltage (assuming 12-bit ADC with 1.8V reference)
        voltage = (value / 4095.0) * 1.8
        self.current_index += 1
        return voltage
    
    def _make_prediction(self, sequence):
        """Make prediction using TFLite model"""
        try:
            # Reshape sequence for model input (batch_size=1, sequence_length, features)
            input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
            
            # Debug output
            print("\nFeature values for prediction:")
            print("MAV, WL, WAMP, MAVS")
            print(sequence[-1])  # Print features for the last window
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get prediction
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            raw_output = output[0]
            print(f"Raw model output: {raw_output}")
            return np.argmax(output[0])  # Return predicted class
            
        except Exception as e:
            print(f"Warning: Prediction failed: {e}")
            return None
    
    def get_status(self):
        """Get processor status including performance stats"""
        stats = self.get_performance_stats()
        return {
            **stats,
            'normalization_active': self.feature_means is not None,
            'file_position': f"{self.current_index}/{len(self.data)}",
            'progress': f"{(self.current_index/len(self.data))*100:.1f}%",
            'model_ready': hasattr(self, 'interpreter')
        } 