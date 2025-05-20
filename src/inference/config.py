from enum import Enum

class Normalization(Enum):
    No = 1
    MinMax = 2
    MeanStd = 3

class Config:

    def __init__(self):
        self.sampling_rate = 1000
        self.low_cut = 20
        self.high_cut = 450
        self.filter_order = 4
        self.wamp_threshold = 0.02
        self.buffer = []
        self.buffer_count = 2
        self.sequence_length = 3
        self.windows_count = 3
        self.window_overlap = 0.5  # 2 -> 50%
        self.read_window_size = 200
        self.frequency = 1000  # input frequency from semg
        self.model_type = 'default'
        self.model = "2state_features_labels_W180_O90_WAMPth20.tflite"
        self.file_path = 'src/inference/output.csv'
        self.features = ['mav', 'wl', 'wamp', 'mavs']
        self.normalization = Normalization.MeanStd
