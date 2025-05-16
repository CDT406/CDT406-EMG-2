from beaglebone_processor import BeagleBoneProcessor
import numpy as np
import csv
import time
import queue

class TestProcessor(BeagleBoneProcessor):
    def __init__(self, config, model_path, test_file_path):
        super().__init__(config, model_path)
        self.test_file_path = test_file_path
        self.test_data = []
        self.current_index = 0
        self.finished = False
        self.latest_features = None
        self.feature_queue = queue.Queue(maxsize=100)  # Store latest features
        self._load_test_data()
        
    def _load_test_data(self):
        """Load test data from CSV file"""
        with open(self.test_file_path, 'r') as f:
            reader = csv.reader(f)
            self.test_data = [float(row[0]) for row in reader]
        print(f"Loaded {len(self.test_data)} samples from {self.test_file_path}")
        
    def _read_adc(self):
        """Override ADC reading with test data"""
        if self.current_index >= len(self.test_data):
            self.finished = True
            self.stop()  # Stop processing when we reach the end
            return 0.0
        
        value = self.test_data[self.current_index]
        self.current_index += 1
        return value
        
    def is_finished(self):
        """Check if we've processed all the test data"""
        return self.finished
        
    def _extract_features(self, window):
        """Override to store latest features"""
        features = super()._extract_features(window)
        try:
            # Store features in queue, removing old ones if necessary
            if self.feature_queue.full():
                self.feature_queue.get_nowait()
            self.feature_queue.put_nowait(features)
        except:
            pass
        return features
        
    def get_latest_features(self):
        """Get the latest features if available"""
        try:
            return self.feature_queue.get_nowait()
        except queue.Empty:
            return None 