import numpy as np
from feature_extraction import extract_features, bandpass_filter
from collections import deque
from config import Normalization


#  Read -> Windowing -> Filter -> Feature Extraction -> Model -> Output


class DataProcess:

    def __init__(self, config, data_input):
        self.config = config
        self.data_input = data_input
        self.buffer = deque(maxlen=config.buffer_count)
        self.finalized_data = deque(maxlen=config.windows_count)
        self.step = round(config.read_window_size * config.window_overlap)


    def _bandpass_filter(self, window):
        return bandpass_filter(
            signal=window,
            lowcut=self.config.low_cut,
            highcut=self.config.high_cut,
            fs=self.config.sampling_rate,
            order=self.config.filter_order
        )

    
    def _get_next_window(self):
        if (len(self.buffer) == self.buffer.maxlen):
            self.buffer.popleft()

        # Need multiple windows due to window overlap
        while (len(self.buffer) < self.config.buffer_count):
            if (self.data_input.has_next()):  # non-blocking, busy-wait
                next_window = self.data_input.next()

                #normalize the window
                if self.config.normalization == Normalization.No:
                    pass
                elif self.config.normalization == Normalization.MinMax:
                    next_window = (next_window - np.min(next_window)) / (np.max(next_window) - np.min(next_window))
                elif self.config.normalization == Normalization.MeanStd:
                    next_window = (next_window - np.mean(next_window)) / np.std(next_window)

                self.buffer.append(next_window)  # TODO: Decide where to filter, here or in process_window?
            else:
                return None

        window = np.array(self.buffer, dtype=np.float32).flatten()[self.config.read_window_size - self.step:]
        processed_window = self._process_window(window)
        return processed_window

    def get_next(self):
        while len(self.finalized_data) < self.config.windows_count:
            window = self._get_next_window()
            if window is None:
                return None
            self.finalized_data.append(window)

        f_data = np.array(self.finalized_data, dtype=np.float32)
        self.finalized_data.popleft()  # Reset for next batch
        return np.array([f_data])


    def _process_window(self, window):
        filtered_window = self._bandpass_filter(window)
        
        if (len(self.config.features) > 0):
            features = extract_features(
                window=filtered_window,
                features=self.config.features,
                wamp_threshold=self.config.wamp_threshold
            )

            #TODO How do we do normalization when the only have one of each feature?

            return features
        else:
            return filtered_window 
