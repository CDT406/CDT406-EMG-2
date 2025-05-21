from led_control import LedControl
from analog_input import FileInput, SensorInput
from data_process import DataProcess
from logger import Logger
from config import Config
from model_loader import Model
import numpy as np


def get_data_input(type='File'):
    if type == 'File':
        return FileInput(
                file_name=config.file_path,
                sampling_rate=config.sampling_rate,
                window_size=config.read_window_size
            )
    elif type == 'Sensor':
        return SensorInput(sampling_rate=config.sampling_rate, window_size=config.read_window_size)
    else:
        raise Exception("Wrong input type")


if __name__ == '__main__':
    config = Config('output/SavedModels/RNN2/Saved_models/preprocessing_config.toml')
    breakpoint()
    logger = Logger(config.log_path)
    led_control = LedControl()
    data_input = get_data_input('File')
    data_process = DataProcess(config=config, data_input=data_input, logger=logger.log_input_data)
    model = Model(model_path=config.model_path, logger=logger.log_output_data)

    while 1:
        window = data_process.get_next()
        if window is None:
            break
        #print(window)
        output_state = model.get_output_state(window)
        #output_state = model.get_output_state(np.random.rand(1, 3, 4).astype(np.float32)*10)
        print(output_state)
        led_control.set_state(output_state)
