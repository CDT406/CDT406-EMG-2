from led_control import LedControl
from analog_input import SensorInput, FileInput
from data_process import DataProcess
from config import Config
from model_loader import ModelLoader


if __name__ == '__main__':
    config = Config()
    led_control = LedControl()
    data_input = FileInput(
        file_name=config.file_path,
        frequency=config.frequency,
        window_size=config.read_window_size
        )
    # data_input = SensorInput(frequency=config.frequency, window_size=config.window_size)
    data_process = DataProcess(config=config, data_input=data_input)
    model = ModelLoader(config.model)
    
    
    while 1:           
        window = data_process.get_next()    
        if window is None:
            break
        #print(window)
          
        output_state = model.get_output_state(window)
        print(output_state)
        led_control.set_state(output_state)
        