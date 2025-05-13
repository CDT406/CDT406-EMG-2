import joblib
import io, os.path
from feature_extraction import extract_features



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
        self.buffer = []
        self.processed_windows = []
        

    def get_window(self):
        # Need multiple windows in buffer due to window overlap
        while (len(self.buffer) <= self.config.window_overlap):
            self.buffer.append(self.data_input.get_window())
            
        self.process_windows()

        self.buffer.remove(0)

        return self.buffer[-1]  # Always last element
            
            
    def process_windows(self):
        print("farts")

       
        