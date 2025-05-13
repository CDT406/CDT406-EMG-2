

class Config:

    def __init__(self):
        self.sampling_rate = 1000
        self.low_cut = 20
        self.high_cut = 450
        self.filter_order = 4
        self.wamp_threshold = 0.02
        self.buffer = []
        self.pre_buffer_count = 2
        self.sequence_length = 3
        self.window_overlap = 2  # 2 -> 50%
        self.window_size = 200
        self.frequency = 1000  # input frequency from semg
        self.model = "model.tflite"
