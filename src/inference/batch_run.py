from led_control import LedControl
from analog_input import SensorInput, FileInput
from data_process import DataProcess
from config import Config
from model_loader import get_model
import glob

def run(led_control, config):
    data_input = FileInput
    (
        file_name=config.file_path,
        sampling_rate=config.sampling_rate,
        window_size=config.read_window_size
    )
    #data_input = SensorInput(sampling_rate=config.sampling_rate, window_size=config.read_window_size)
    data_process = DataProcess(config=config, data_input=data_input)
    model = get_model(model_type=config.model_type, model_path=config.model)

    data = np.array([], dtype=np.int32)
	#for i in range(time * sampling_rate):
	#	array = np.append(array, read_value(file))
	#	print("Reading value: ", array)
	#	time.sleep(1.0 / sampling_rate)
	#np.savetxt("output.csv", array, delimiter=",", fmt="%d")

    while 1:
        window = data_process.get_next()
        if window is None:
            break

        output_state = model.get_output_state(window)
        array = np.append(array, read_value(file))
        print(output_state)
        led_control.set_state(output_state)


# TODO : Implement get all models
def get_all_models():
    # Folder path with glob
    # TODO FIX PATH
    file_paths = glob.glob('NormalizedData/WindowTest50%overlap/2state/*.tflite')
    results = []

    for path in file_paths:
        filename = os.path.basename(path)  # Get just the file name
        parts = filename.replace('.tflite', '').split('_')
        
        # Extract values based on fixed positions
        window = int(parts[3][1:])          # 'W128' -> 128
        overlap = int(parts[4][1:])         # 'O64' -> 64
        wamp = int(parts[5][6:])            # 'WAMPth20' -> 20

        config = Config()
        config.model_path = file_name
        config.read_window_size = 200
        config.window_overlap = overlap / window
        config.wamp_threshold = wamp / 1000

        results.append(config)

    return results


if __name__ == '__main__':
    led_control = LedControl()
    models = get_all_models()
    for model in models:
        print("Running model: ", model)
        run(model, led_control)
