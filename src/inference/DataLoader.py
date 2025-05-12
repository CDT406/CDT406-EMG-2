import joblib
import io, os.path


#  Read -> Windowing -> Filter -> Feature Extraction -> Model -> Output


class DataLoader:


    def __init__(self, sequence_length=3):
        self.sequence_length = sequence_length
        self.analog_read_path = "/sys/bus/iio/devices/iio:device0/in_voltage0_raw"
        self.is_on_board = os.path.exists(self.analog_read_path)

        if (self.is_on_board):
            self.analog_stream = io.open(
                self.analog_read_path,
                mode = "r",
                buffering = -1,
                encoding = None,
                errors = None,
                newline = None,
                closefd = True
            )

    def get_training_data(self):



# import io
# import time
# stream = io.open("/sys/bus/iio/devices/iio:device0/in_voltage0_raw", mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True)
# while True:
# 	content = stream.read()
# 	stream.seek(0)
# 	# Process the binary content as needed
# 	print(content)
# 	time.sleep(0.001)
