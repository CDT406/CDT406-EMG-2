import numpy as np
from feature_extraction import extract_features
from collections import deque



#  Read -> Windowing -> Filter -> Feature Extraction -> Model -> Output


class DataProcess:

    def __init__(self, config, data_input):
        self.config = config
        self.data_input = data_input
        self.buffer = deque(maxlen=config.window_overlap)
        self.step = config.window_size // config.window_overlap      
        
        
    def _filter_window(self, window):
        return bandpass_filter(
            signal=window,
            lowcut=self.config.low_cut,
            highcut=self.config.high_cut,
            fs=self.config.sampling_rate,
            order=self.config.filter_order
        )
        

    def get_next_window(self):
        if (self.buffer.count == self.buffer.maxlen):
            self.buffer.popleft()    
        
        # Need multiple windows due to window overlap
        while (len(self.buffer) < self.config.pre_buffer_count):
            if (self.data_input.has_next()):  # non-blocking, busy-wait
                next_window = self.data_input.next()
                self.buffer.append(next_window)  # TODO: Decide where to filter, here or in process_window?
            else:
                return None
        
        window = np.array(self.buffer).flatten()[self.step:self.step + self.config.window_size]
        processed_window = self._process_window(window) 
        
        self.buffer.popleft()    

        return processed_window  
            
            
    def _process_window(self, window):
        return window  # TODO: Add processing

       