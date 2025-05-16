from real_time_processor import RealTimeProcessor
import numpy as np
from tflite_runtime.interpreter import Interpreter
import json

class BeagleBoneProcessor(RealTimeProcessor):
    def __init__(self, config, model_path, analog_pin=0):
        super().__init__(config)
        self.analog_pin = analog_pin
        self.adc_file = None
        
        # Load normalization parameters if available
        try:
            with open(config.normalization_params_path, 'r') as f:
                params = json.load(f)
                self.set_normalization_params(
                    np.array(params['means'], dtype=np.float32),
                    np.array(params['stds'], dtype=np.float32)
                )
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load normalization parameters: {e}")
            print("Features will not be normalized!")
        
        # Initialize TFLite model
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
        """Read from BeagleBone's ADC"""
        if self.adc_file is None:
            try:
                self.adc_file = open(f"/sys/bus/iio/devices/iio:device0/in_voltage{self.analog_pin}_raw", "r")
            except FileNotFoundError:
                raise RuntimeError(f"ADC pin {self.analog_pin} not found")
        
        try:
            self.adc_file.seek(0)
            value = float(self.adc_file.read().strip())
            
            # Convert ADC value to voltage (assuming 12-bit ADC with 1.8V reference)
            voltage = (value / 4095.0) * 1.8
            return voltage
            
        except (IOError, ValueError) as e:
            print(f"Warning: ADC read failed: {e}")
            return 0.0
    
    def _make_prediction(self, sequence):
        """Make prediction using TFLite model"""
        try:
            # Reshape sequence for model input (batch_size=1, sequence_length, features)
            input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get prediction
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            return np.argmax(output[0])  # Return predicted class
            
        except Exception as e:
            print(f"Warning: Prediction failed: {e}")
            return None
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop()
        if self.adc_file:
            self.adc_file.close()
    
    def get_status(self):
        """Get processor status including performance stats"""
        stats = self.get_performance_stats()
        return {
            **stats,
            'normalization_active': self.feature_means is not None,
            'adc_ready': self.adc_file is not None,
            'model_ready': hasattr(self, 'interpreter')
        } 