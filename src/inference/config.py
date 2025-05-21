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

