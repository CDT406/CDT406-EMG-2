import numpy as np
from collections import deque
import threading
import time
from scipy.signal import butter, sosfilt
import queue
from feature_normalization import normalize_features

class RealTimeProcessor:
    def __init__(self, config):
        # Sampling and window parameters
        self.sampling_rate = 5000  # Hz
        self.window_size = 224     # Samples
        self.overlap = 112         # 50% overlap
        self.sequence_length = 3   # Number of windows for LSTM
        
        # Buffer sizes
        self.circular_buffer_size = self.window_size + self.overlap  # Hold enough samples for processing
        self.feature_buffer_size = self.sequence_length  # Hold enough features for LSTM input
        
        # Initialize buffers
        self.sample_buffer = deque(maxlen=self.circular_buffer_size)
        self.feature_buffer = deque(maxlen=self.feature_buffer_size)
        
        # Initialize queues for thread communication
        self.sample_queue = queue.Queue(maxsize=self.sampling_rate)  # 1 second buffer
        self.feature_queue = queue.Queue(maxsize=100)  # Buffer for features
        self.prediction_queue = queue.Queue()
        
        # Bandpass filter parameters
        self.sos = self._create_bandpass_filter(20, 450, self.sampling_rate)
        
        # Performance monitoring
        self.processing_times = deque(maxlen=100)  # Track last 100 processing times
        self.dropped_samples = 0
        self.total_samples = 0
        
        # Threading control
        self.running = False
        self.acquisition_thread = None
        self.processing_thread = None
        self.inference_thread = None
    
    def _create_bandpass_filter(self, lowcut, highcut, fs, order=4):
        """Create bandpass filter coefficients"""
        nyquist = fs / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        return butter(order, [low, high], btype='band', output='sos')
    
    def _filter_signal(self, signal):
        """Apply bandpass filter and normalize signal"""
        # Apply bandpass filter
        filtered = sosfilt(self.sos, signal)
        # Normalize the filtered signal
        normalized = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
        return normalized
    
    def _extract_features(self, window):
        """Extract features from a window of samples"""
        start_time = time.perf_counter()
        
        # Mean Absolute Value
        mav = np.mean(np.abs(window))
        
        # Waveform Length
        wl = np.sum(np.abs(np.diff(window)))
        
        # Willison Amplitude
        wamp = np.sum(np.abs(np.diff(window)) > 0.02)
        
        # MAV Slope
        half = len(window) // 2
        mavs = np.abs(np.mean(np.abs(window[:half])) - np.mean(np.abs(window[half:])))
        
        features = np.array([mav, wl, wamp, mavs], dtype=np.float32)
        
        # Normalize features using training data statistics
        features = normalize_features(features)
        
        processing_time = time.perf_counter() - start_time
        self.processing_times.append(processing_time)
        
        return features
    
    def _acquisition_loop(self):
        """Continuous data acquisition loop"""
        last_time = time.perf_counter()
        sample_period = 1.0 / self.sampling_rate
        
        while self.running:
            current_time = time.perf_counter()
            if current_time - last_time >= sample_period:
                try:
                    # Read from ADC
                    sample = self._read_adc()
                    self.sample_queue.put_nowait(sample)
                    self.total_samples += 1
                    last_time = current_time
                except queue.Full:
                    self.dropped_samples += 1
                    # Skip this sample and adjust timing
                    last_time = current_time
    
    def _processing_loop(self):
        """Continuous signal processing loop"""
        samples_since_last_window = 0
        
        while self.running:
            try:
                # Process all available samples
                while True:
                    sample = self.sample_queue.get_nowait()
                    self.sample_buffer.append(sample)
                    samples_since_last_window += 1
                    
                    # Check if we have enough samples for a new window
                    if samples_since_last_window >= self.overlap and len(self.sample_buffer) >= self.window_size:
                        # Extract window
                        window = np.array(list(self.sample_buffer)[-self.window_size:])
                        
                        # Filter and normalize the window
                        filtered_window = self._filter_signal(window)
                        
                        # Extract features
                        features = self._extract_features(filtered_window)
                        
                        # Add to feature queue
                        try:
                            self.feature_queue.put_nowait(features)
                        except queue.Full:
                            # If feature queue is full, remove oldest feature
                            try:
                                self.feature_queue.get_nowait()
                                self.feature_queue.put_nowait(features)
                            except queue.Empty:
                                pass
                        
                        samples_since_last_window = 0
                        
            except queue.Empty:
                # No more samples to process, sleep briefly
                time.sleep(0.0001)
    
    def _inference_loop(self):
        """Separate thread for model inference"""
        feature_sequence = deque(maxlen=self.sequence_length)
        
        while self.running:
            try:
                # Get new features
                features = self.feature_queue.get(timeout=0.1)
                feature_sequence.append(features)
                
                # If we have enough features, make prediction
                if len(feature_sequence) >= self.sequence_length:
                    sequence = np.array(list(feature_sequence))
                    prediction = self._make_prediction(sequence)
                    
                    # Update prediction queue (remove old prediction if necessary)
                    try:
                        self.prediction_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.prediction_queue.put_nowait(prediction)
                    
            except queue.Empty:
                continue
    
    def _read_adc(self):
        """Read from ADC - implement for specific hardware"""
        raise NotImplementedError("Implement for specific ADC")
    
    def _make_prediction(self, sequence):
        """Make prediction using the model - implement for specific model"""
        raise NotImplementedError("Implement for specific model")
    
    def start(self):
        """Start real-time processing"""
        self.running = True
        self.acquisition_thread = threading.Thread(target=self._acquisition_loop, daemon=True)
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        
        self.acquisition_thread.start()
        self.processing_thread.start()
        self.inference_thread.start()
    
    def stop(self):
        """Stop real-time processing"""
        self.running = False
        if self.acquisition_thread:
            self.acquisition_thread.join()
        if self.processing_thread:
            self.processing_thread.join()
        if self.inference_thread:
            self.inference_thread.join()
    
    def get_latest_prediction(self):
        """Get the latest prediction if available"""
        try:
            return self.prediction_queue.get_nowait()
        except queue.Empty:
            return None
            
    def get_performance_stats(self):
        """Get performance statistics"""
        if len(self.processing_times) > 0:
            avg_processing_time = np.mean(self.processing_times) * 1000  # Convert to ms
            max_processing_time = np.max(self.processing_times) * 1000
        else:
            avg_processing_time = max_processing_time = 0
            
        return {
            'dropped_samples': self.dropped_samples,
            'total_samples': self.total_samples,
            'drop_rate': self.dropped_samples / max(1, self.total_samples) * 100,
            'avg_processing_time_ms': avg_processing_time,
            'max_processing_time_ms': max_processing_time
        } 