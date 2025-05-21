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
        self.buffer.append(input_data)

    # Input data is a 1D numpy array of raw input data
    # Output data is a 1D numpy array of percentage values
    def log_output_data(self, time, output_data):
        input_data = input_data.flatten()
        #copy output len(input_data) times to output_data
        output_data = np.tile(output_data, (len(input_data), 1))
        time = np.tile(time, (len(input_data), 1))
        for i in zip(input_data, output_data):
            time_str = ','.join(map(str, i[0]))
            input_str = ','.join(map(str, i[0]))
            output_str = ','.join(map(str, i[1]))
            self.file.write(f"{time_str},{input_str},{output_str}\n")
        self.file.flush()