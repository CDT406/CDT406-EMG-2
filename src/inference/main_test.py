from led_control import LedControl
from analog_input import FileInput
from data_process import DataProcess
from config import Config
from model_loader import get_model
import os

if __name__ == '__main__':
    # Initialize configuration
    config = Config()
    
    # Override config settings for test
    config.file_path = os.path.join(
        "datasets", "official", "unprocessed",
        "relabeled_old_dataset", "1",
        "edited_0205-132606record_N.csv"
    )
    
    # Initialize components
    led_control = LedControl()
    data_input = FileInput(
        file_name=config.file_path,
        frequency=config.frequency,
        window_size=config.read_window_size
    )
    data_process = DataProcess(config=config, data_input=data_input)
    model = get_model(model_type=config.model_type, model_path=config.model)

    # Process and classify windows
    predictions = []
    while True:
        window = data_process.get_next()
        if window is None:
            break

        output_state = model.get_output_state(window)
        predictions.append(output_state)
        print(f"Predicted state: {output_state}")
        led_control.set_state(output_state)

    # Print summary
    print("\nPrediction Summary:")
    print(f"Total windows processed: {len(predictions)}")
    print("\nPredicted states distribution:")
    for state in range(4):  # 4 classes (0: Rest, 1: Grip, 2: Hold, 3: Release)
        count = predictions.count(state)
        percentage = (count / len(predictions)) * 100
        print(f"State {state}: {count} windows ({percentage:.1f}%)")