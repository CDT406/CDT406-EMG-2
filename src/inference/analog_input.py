try:
    import Adafruit_BBIO.ADC as ADC
except ImportError:
    print("Running on PC - using mock ADC")
    from mock_adc import MockADC as ADC

import pandas as pd
import numpy as np
import io
import time
import threading
import queue


class FileInput:
    def __init__(self, file_name, frequency, window_size):
        self.data = pd.read_csv(file_name)
        self.frequency = frequency
        self.window_size = window_size
        self.current_index = 0
        
    def read_window(self):
        samples_per_window = int((self.window_size/1000) * self.frequency)
        if self.current_index + samples_per_window > len(self.data):
            return None
            
        window = self.data.iloc[self.current_index:self.current_index + samples_per_window]
        self.current_index += samples_per_window
        return window['voltage'].values


class SensorInput:
    def __init__(self, frequency, window_size):
        self.frequency = frequency
        self.window_size = window_size
        self.pin = "AIN0"
        ADC.setup()
    
    def read_window(self):
        samples = []
        sample_count = int((self.window_size/1000) * self.frequency)
        
        for _ in range(sample_count):
            value = ADC.read(self.pin)
            samples.append(value)
        
        return np.array(samples)