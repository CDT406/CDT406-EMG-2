import io
import time
import threading
import queue
import numpy as np
import Adafruit_BBIO.ADC as ADC


def pop_front(array, n=200):
    if len(array) < n:
        raise ValueError("Not enough elements to pop.")
    front = array[:n]
    array = array[n:]  # new view
    return front, np.array(array, dtype=np.float32)


def read_sensor(queue, analog_pin="P9_33", sampling_rate=1000, window_size=200):
	assert(sampling_rate == 1000, "Frequency must be 1000Hz dumb dumb code")

	window = np.zeros(window_size, dtype=np.float32)
	sleep = 0.0003273
	ADC.setup()

	while True:
		for i in range(window_size):
			window[i] = ADC.read(analog_pin) * 2.95 # max output from sparkfun hearbeat sensor
			time.sleep(sleep)

		queue.put(window)

class SensorInput:
	def __init__(self, analog_pin="P9_33", sampling_rate=1000, window_size=200):
		self.queue = queue.Queue()
		self.analog_pin = analog_pin
		threading.Thread(target=read_sensor, daemon=True, args=[self.queue, analog_pin, sampling_rate, window_size]).start()

	def is_done(self):
		return False

	def has_next(self):
		return not self.queue.empty()

	def next(self):
		try:
			return self.queue.get_nowait()
		except queue.Empty:
			return None

class FileInput:
	def __init__(self, file_name="output.csv", sampling_rate=1000, window_size=200):
		self.data = np.loadtxt(file_name, delimiter=",", dtype=np.float32)[:,1]
		breakpoint()
		self.window_size = window_size

	def is_done(self):
		return self.has_next() == 0

	def has_next(self):
		return self.data.shape[0] > 0

	def next(self):
		try:
			window, self.data = pop_front(self.data, n=self.window_size)
			return window
		except queue.Empty:
			return None


def record_sensor_data(time =10, sampling_rate=1000):
	file = open_sensor()
	array = np.array([], dtype=np.int32)
	for i in range(time * sampling_rate):
		array = np.append(array, read_value(file))
		print("Reading value: ", array)
		time.sleep(1.0 / sampling_rate)
	np.savetxt("output.csv", array, delimiter=",", fmt="%d")


if __name__ == "__main__":
	f = FileInput()
	while f.has_next():
		window = f.next()
		if window is not None:
			print("Window: ", window)
		else:
			print("No more data.")
		time.sleep(1.0 / 1000)
