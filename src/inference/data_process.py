import numpy as np
from feature_extraction import extract_features, bandpass_filter
from collections import deque
from config import Normalization


#  Read -> Windowing -> Filter -> Feature Extraction -> Model -> Output


class DataProcess:

    def __init__(self, config, data_input, logger=None):
        self.config = config
        self.data_input = data_input
        self.buffer = deque(maxlen=config.buffer_len)
        self.finalized_data = deque(maxlen=config.windows_count)
        self.step = round(config.read_window_size * config.window_overlap)
        self.index = 0
        self.logger = logger


    def _bandpass_filter(self, window):
        return bandpass_filter(
            signal=window,
            lowcut=self.config.low_cut,
            highcut=self.config.high_cut,
            fs=self.config.sampling_rate,
            order=self.config.filter_order
        )


    def _get_next_window(self):
        # Need multiple windows due to window overlap
        while len(self.buffer) < self.config.buffer_len:
            if self.data_input.has_next():  # non-blocking, busy-wait
                next_window = self.data_input.next()
                if self.logger is not None:
                    self.logger(next_window)
                self.buffer.append(next_window)  # TODO: Decide where to filter, here or in process_window?

            if self.data_input.is_done():
                return None

        window = np.array(self.buffer, dtype=np.float32).flatten()[self.index:self.index+self.config.read_window_size]

        self.index += self.step
        if self.index + self.config.read_window_size >= self.config.read_window_size * self.buffer.maxlen:
            self.index -= self.config.read_window_size
            if len(self.buffer) == self.buffer.maxlen:
                self.buffer.popleft()

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
        return np.array([f_data.flatten()])


    def _process_window(self, window):
        if (len(self.config.features) > 0):
            window = np.array(window)
           # window = np.array(self._bandpass_filter(window))

            #normalize the window
            if self.config.normalization == Normalization.No:
                pass
            elif self.config.normalization == Normalization.MinMax:
                window = (window - np.min(window)) / (np.max(window) - np.min(window))
                exit(-1) # TODO FIX THIS
            elif self.config.normalization == Normalization.MeanStd:
                window = (window - self.config.window_normalization['global_mean']) / self.config.window_normalization['global_std']

            #window = np.array(self._bandpass_filter(window))
            features = extract_features(
                window=window,
                features=self.config.features,
                wamp_threshold=self.config.wamp_threshold
            )

            normalized_features = []

            for feature_name, feature in zip(self.config.features, features):
                [mean, std] = self.config.feature_stats[feature_name]
                normalized_feature = (feature - mean) / std
                normalized_features.append(normalized_feature)

            return np.array(normalized_features, dtype=np.float32)
            # return normalized_features
        else:
            return window
