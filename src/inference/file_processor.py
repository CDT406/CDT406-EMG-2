from real_time_processor import RealTimeProcessor
import numpy as np
from scipy.signal import butter, filtfilt
from tflite_runtime.interpreter import Interpreter

class FileProcessor(RealTimeProcessor):
    def __init__(self, config, model_path, file_path):
        super().__init__(config)
        self.lowcut = 20
        self.highcut = 450
        self.order = 4
        self.wamp_threshold = 0.02
        self.file_path = file_path
        self.data = None
        self.current_index = 0
        self.running = False
        self.all_features = []  # Store all features for sequence creation
        
        # Load and preprocess the entire file
        self._load_and_preprocess_file()
        
        # Initialize TFLite model
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def _load_and_preprocess_file(self):
        """Load and preprocess the entire file"""
        # Load signal (raw values only)
        self.data = np.loadtxt(self.file_path, delimiter=",", dtype=np.float32)
        
        # Convert ADC values to voltage (assuming 12-bit ADC with 1.8V reference)
        self.data = (self.data / 4095.0) * 1.8
        
        # Apply bandpass filter
        self.data = self._bandpass_filter(self.data)
        
        print(f"Loaded {len(self.data)} samples from {self.file_path}")
    
    def _bandpass_filter(self, signal):
        """Apply bandpass filter to signal"""
        nyquist = 0.5 * self.sampling_rate
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = butter(self.order, [low, high], btype='band')
        return filtfilt(b, a, signal)
    
    def _normalize_window(self, window):
        """Normalize window using mean and std"""
        return (window - np.mean(window)) / (np.std(window) + 1e-8)
    
    def _extract_features(self, window):
        """Extract features from window"""
        features = []
        features.append(self._compute_mav(window))
        features.append(self._compute_wl(window))
        features.append(self._compute_wamp(window))
        features.append(self._compute_mavs(window))
        return np.array(features, dtype=np.float32)
    
    def _compute_mav(self, window):
        """Mean Absolute Value"""
        return np.mean(np.abs(window))
    
    def _compute_wl(self, window):
        """Waveform Length"""
        return np.sum(np.abs(np.diff(window)))
    
    def _compute_wamp(self, window):
        """Willison Amplitude"""
        return np.sum(np.abs(np.diff(window)) > self.wamp_threshold)
    
    def _compute_mavs(self, window):
        """MAV Slope"""
        half = len(window) // 2
        return np.abs(self._compute_mav(window[:half]) - self._compute_mav(window[half:]))
    
    def start(self):
        """Start processing the file"""
        self.running = True
        self.current_index = 0
        self.all_features = []
    
    def stop(self):
        """Stop processing"""
        self.running = False
    
    def get_latest_prediction(self):
        """Get prediction for next window"""
        if not self.running or self.current_index >= len(self.data):
            return None
        
        # Get window
        end = min(self.current_index + self.window_size, len(self.data))
        window = self.data[self.current_index:end]
        
        if len(window) < self.window_size:
            self.running = False
            return None
        
        # Normalize window
        window = self._normalize_window(window)
        
        # Extract features
        features = self._extract_features(window)
        
        # Add to all features
        self.all_features.append(features)
        
        # Create sequence if we have enough features
        if len(self.all_features) >= self.sequence_length:
            # Create sequences by sliding over all features
            X = []
            for i in range(len(self.all_features) - self.sequence_length + 1):
                X.append(self.all_features[i:i + self.sequence_length])
            X = np.array(X, dtype=np.float32)
            
            # Get the last sequence
            last_sequence = X[-1:]
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], last_sequence)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get prediction
            pred = self.interpreter.get_tensor(self.output_details[0]['index'])
            prediction = np.argmax(pred[0])
            
            # Update metrics
            self.total_samples += 1
            
            # Move to next window
            self.current_index += (self.window_size - self.overlap)
            
            return prediction
        
        # Move to next window even if we don't have enough features yet
        self.current_index += (self.window_size - self.overlap)
        return None
    
    def get_status(self):
        """Get processor status including performance stats"""
        stats = self.get_performance_stats()
        return {
            **stats,
            'normalization_active': True,  # We normalize windows
            'file_position': f"{self.current_index}/{len(self.data)}",
            'progress': f"{(self.current_index/len(self.data))*100:.1f}%",
            'model_ready': hasattr(self, 'interpreter')
        } 