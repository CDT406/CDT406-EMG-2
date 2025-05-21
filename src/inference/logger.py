import numpy as np

class Logger:
    
    def __init__(self, path='output/output.csv'):
        self.path = path
        self.file = open(self.path, 'w')
        self.file.write("input_data,output_data\n")
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
        breakpoint()
        output_data = output_data.tolist()
        output_state = [output_data.index(max(output_data))] * len(input_data)
        output_data = [output_data] * len(input_data)
        time = [time] * len(input_data)

        breakpoint()
        for t, i, s, o in zip(time, input_data, output_state, output_data):
            time_str = ','.join(map(str, t))
            input_str = ','.join(map(str, i))
            output_state_str = ','.join(map(str, s))
            output_str = ','.join(map(str, o))
            self.file.write(f"{time_str},{input_str},{output_state_str},{output_str}\n")
        self.file.flush()