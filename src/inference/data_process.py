import joblib
import io, os.path
from feature_extraction import extract_features
from collections import deque



#  Read -> Windowing -> Filter -> Feature Extraction -> Model -> Output


def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    from scipy.signal import butter, filtfilt
    
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    
    return filtfilt(b, a, signal)


class DataProcess:

    def __init__(self, config, data_input):
        self.config = config
        self.data_input = data_input
        self.buffer = deque()
        self.processed_windows = deque()
        
        # Need multiple windows in buffer due to window overlap
        while (len(self.buffer) < self.config.pre_buffer_count):
            if (self.data_input.has_next()):  # non-blocking, busy-wait
                processed_window = self.filter_window(self.data_input.next())
                self.buffer.append(processed_window)
        
        self.process_windows()
        
        
    def filter_window(self, window):
        return bandpass_filter(
            signal=window,
            lowcut=self.config.low_cut,
            highcut=self.config.high_cut,
            fs=self.config.sampling_rate,
            order=self.config.filter_order
        )
        

    def get_next_window(self):
        while (not self.data_input.has_next()):
            continue
            
        self.buffer.popleft()    
        self.buffer.append(self.data_input.next())
            
        self.process_windows()

        self.buffer.remove(0)

        return self.buffer[-1]  # Always last element
            
            
    def process_windows(self):
        

       
        