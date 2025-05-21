from enum import Enum

class Normalization(Enum):
    No = 1
    MinMax = 2
    MeanStd = 3

class Config:

    def __init__(self):
        self.sampling_rate = 1000 # input frequency from semg
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
        self.model_type = 'default'
        self.model = "output/SavedModels/RNN/NormalizedData1000Hz/WindowTest50%overlap/2state/2state_features_labels_W180_O90_WAMPth20.tflite"
        self.file_path = 'official/unprocessed/slow/2/0205-133530record.csv'
        self.features = ['mav', 'wl', 'wamp', 'mavs']
        self.normalization = Normalization.MeanStd
        self.sqlite_path = f"output/{self.model_type}.db"
