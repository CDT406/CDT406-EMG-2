from led_control import LedControl
from analog_input import SensorInput, FileInput
from data_process import DataProcess
from logger import Logger
from config import Config
from model_loader import get_model


if __name__ == '__main__':
    config = Config()
    logger = Logger(config.sqlite_path)
    led_control = LedControl()
    # data_input = FileInput(
    #     file_name=config.file_path,
    #     sampling_rate=config.sampling_rate,
    #     window_size=config.read_window_size
    #     )
    data_input = SensorInput(sampling_rate=config.sampling_rate, window_size=config.read_window_size)
    data_process = DataProcess(config=config, data_input=data_input)
    model = get_model(model_type=config.model_type, model_path=config.model, logger=logger.log_data)


    while 1:
        window = data_process.get_next()
        if window is None:
            break
        #print(window)

        output_state = model.get_output_state(window)
        print(output_state)
        led_control.set_state(output_state)
