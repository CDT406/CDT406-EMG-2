from led_control import LedControl
from analogInput import SensorInput
from data_process import DataLoader



if __name__ == '__main__':
    led_control = LedControl()
    analog_input = SensorInput()
    data_process = DataLoader()
    # model = ModelLoader()
    
    
    # while 1:
        # window = DataLoader.get_window()
        # model.set_input_window(window)
        # output_state = model.get_output_state()
        # led_control.setState(output_state)
        