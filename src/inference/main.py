from led_control import LedControl
from analog_input import FileInput, SensorInput
from data_process import DataProcess
from logger import Logger
from config import Config
from model_loader import Model
import numpy as np
import time

def get_data_input(type='file'):
    if type == 'file':
        return FileInput(
                file_name=config.file_path,
                sampling_rate=config.sampling_rate,
                window_size=config.read_window_size
            )
    elif type == 'sensor':
        return SensorInput(sampling_rate=config.sampling_rate, window_size=config.read_window_size)
    else:
        raise Exception("Wrong input type")


if __name__ == '__main__':
    data_source = 'file'
    print(f"Using data from {data_source}")
    config = Config('output/SavedModels/RNN2/Saved_models_180ms/preprocessing_config.toml')
    logger = Logger(config.log_path, config.model_states)
    led_control = LedControl()
    data_input = get_data_input(data_source)
    data_process = DataProcess(config=config, data_input=data_input, logger=logger.log_input_data)
    model = Model(model_path=config.model_path, logger=logger.log_output_data)

    start_time = time.time()

    while 1:
        window = data_process.get_next()
        if window is None:
            continue

        if time.time() - start_time > config.timeout:
            break

        #print(window)
        output_state = model.get_output_state(window)
        fart_state = output_state.tolist().index(max(output_state.tolist()))
        print(["REST   ", "GRIP   ", "HOLD   ", "RELEASE"][fart_state], end='\r', flush=True)
        led_control.set_state(output_state)
