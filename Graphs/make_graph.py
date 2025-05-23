import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_csv_data(input_file: str, output_file: str):
    """Reads CSV file, plots data, and saves the plot as PNG."""
    df = pd.read_csv(input_file)

    plt.figure(figsize=(10, 5))
    for column in df.columns[1:3]:  # Skipping the first column ('time' assumed)
        plt.plot(df[column], label=column)

    #plt.xlabel(df.columns[0])
    plt.ylabel("Values")
    plt.title("CSV Data Visualization")
    plt.legend()
    plt.grid()

    plt.savefig(output_file, dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot CSV data and save as PNG")
    parser.add_argument("input_file", type=str, help="Path to input CSV file")
    parser.add_argument("output_file", type=str, help="Path to save output PNG file")

    args = parser.parse_args()
    plot_csv_data(args.input_file, args.output_file)

