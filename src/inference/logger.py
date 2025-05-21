import numpy as np

class Logger:
    
    def __init__(self, path='output/output.csv'):
        self.path = path
        self.file = open(self.path, 'w')
        self.file.write("time, input_data, output_state, output_data\n")
        self.buffer = []
        print(f"Logger initialized at {self.path}")

    def __del__(self):
        self.file.close()
        print(f"Logger closed at {self.path}")

    def log_input_data(self, input_data):
        self.buffer.extend(input_data)

    # Input data is a 1D numpy array of raw input data
    # Output data is a 1D numpy array of percentage values
    def log_output_data(self, output_data, time): 
        input_data = self.buffer
        self.buffer = []
        output_data = output_data.tolist()
        output_state = [output_data.index(max(output_data))] * len(input_data)
        output_data = [output_data] * len(input_data)
        _time = [time] * len(input_data)

        for t, i, s, o in zip(_time, input_data, output_state, output_data):
            self.file.write(f"{t}, {i}, {s}, {','.join(map(str, o))}\n")
        self.file.flush()