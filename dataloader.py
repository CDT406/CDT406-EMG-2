import pandas as pd
import numpy as np
import os

def load_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    
    data = pd.read_csv(file_path, header=4)
    return data.sort_index(inplace=False, ascending=False)


if __name__ == "__main__":
    file_path = "Testdata/testJ.csv"
    data = load_file(file_path)
    
    print(data.head())