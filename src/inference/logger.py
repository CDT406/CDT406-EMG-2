import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Logger:
    
    def __init__(self, path='output/output.csv'):
        self.path = path
        self.file = open(self.path, 'w')
        self.buffer = []
        self.has_written_header = False
        print(f"Logger initialized at {self.path}")


    def __del__(self):
        self.file.close()
        self.plot_output_state()
        print(f"Logger closed at {self.path}")


    def log_input_data(self, input_data):
        self.buffer.extend(input_data)


    def plot_output_state(self, output_png='output/output_state_plot.png'):
        # Read the CSV file, using '/' as separator, and replace ',' with '.' for decimals
        with open(self.path, 'r') as f:
            lines = f.readlines()
        header = lines[0].strip().split('/')
        data = [line.strip().split('/') for line in lines[1:]]
        df = pd.DataFrame(data, columns=header)
        
        # Convert relevant columns to float
        df['time'] = df['time'].str.replace(',', '.').astype(float)
        df['output_state'] = df['output_state'].astype(int)
        
        # Plot with color based on output_state
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'black']
        plt.figure(figsize=(12, 4))
        for state in df['output_state'].unique():
            mask = df['output_state'] == str(state)
            plt.scatter(df['time'][mask], df['output_state'][mask], 
                        color=colors[int(state) % len(colors)], label=f'State {state}', s=10)
        plt.xlabel('Time')
        plt.ylabel('Output State')
        plt.title('Output State over Time')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_png)
        plt.close()


    # Input data is a 1D numpy array of raw input data
    # Output data is a 1D numpy array of percentage values
    def log_output_data(self, output_data, time): 
        input_data = self.buffer
        self.buffer = []
        output_data = output_data.tolist()
        
        
        if not self.has_written_header:
            #SHAME
            states = ["/rest","/grip","/hold","/release"]
            states = "".join([states[i] for i in range(len(output_data))])
            self.file.write(f"time/input_data/output_state{states}\n")
            self.has_written_header = True


        output_state = [output_data.index(max(output_data))] * len(input_data)
        output_data = [output_data] * len(input_data)
        _time = [time] * len(input_data)


        for t, i, s, o in zip(_time, input_data, output_state, output_data):
            self.file.write(f"{t}/{i}/{s}/{'/'.join(map(str, o))}\n".replace('.', ','))
        self.file.flush()