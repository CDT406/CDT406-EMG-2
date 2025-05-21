from enum import Enum
import toml

class Normalization(Enum):
    No = 1
    MinMax = 2
    MeanStd = 3

def str2enum(s):
    if s == 'MinMax':
        return Normalization.MinMax
    if s == 'MeanStd':
        return Normalization.MeanStd
    else:
        return Normalization.No

class Config:
    def __init__(self, toml_path):
        data = toml.load(toml_path)
        fstats = data['feature_stats']
        model = data['model']
        self.sampling_rate = fstats['sampling_rate']
        self.read_window_size = fstats['window_size']
        self.window_overlap = fstats['window_overlap']
        self.sequence_length = fstats['sequence_length']
        self.windows_count = fstats['windows_count']
        self.low_cut = fstats['low_cut']
        self.high_cut = fstats['high_cut']
        self.filter_order = fstats['filter_order']
        self.wamp_threshold = fstats['wamp_threshold']
        self.features = fstats['features']
        self.normalization = str2enum(fstats['normalization'])
        self.model = model['model_path']
        self.file_path = model['test_file_path']
        self.preprocessing_stats = data['preprocessing_config']
        self.log_path = model['log_file_path']


# from enum import Enum

# class Normalization(Enum):
#     No = 1
#     MinMax = 2
#     MeanStd = 3


# def str2enum(str):
#     if str == 'MinMax':
#         return Normalization.MinMax
#     if str == 'MeanStd':
#         return Normalization.MeanStd
#     else:
#         return Normalization.No


# class Config:

#     def __init__(
#         self,
#         sampling_rate = 1000,
#         read_window_size = 200,
#         window_overlap = 0.5,
#         sequence_length = 3,
#         windows_count = 3,
#         low_cut = 20,
#         high_cut = 450,
#         filter_order = 4,
#         wamp_threshold = 0.02,
#         features = ['mav', 'wl', 'wamp', 'mavs'],
#         normalization_str = 'MeanStd'
#         ):
#         self.sampling_rate = sampling_rate # input frequency from semg
#         self.read_window_size = read_window_size
#         self.window_overlap = window_overlap  # 2 -> 50%
#         self.sequence_length = sequence_length
#         self.windows_count = windows_count
#         self.low_cut = low_cut
#         self.high_cut = high_cut
#         self.filter_order = filter_order
#         self.wamp_threshold = wamp_threshold
#         self.features = features
#         self.normalization_str = normalization_str
#         self.normalization = str2enum(normalization_str)
#         self.buffer = []
#         self.buffer_count = 2
#         self.model_type = 'default'
#         self.model = "2state_features_labels_W180_O90_WAMPth20.tflite"
#         self.file_path = 'src/inference/output.csv'
#         self.sqlite_path = f"output/{self.model_type}.db"


<<<<<<< Updated upstream
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
=======
>>>>>>> Stashed changes
