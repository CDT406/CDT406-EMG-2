import io
import time
import threading
import queue
import numpy as np


def open_sensor(analog_pin=0):
	try:
		file = open("/sys/bus/iio/devices/iio:device0/in_voltage{}_raw".format(analog_pin), mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True)
		return file
	except FileNotFoundError:
		print("Sensor not found. Please check the analog pin.")
		return None


def read_value(file):
	sensor_value = file.read()
	file.seek(0)
	return int(sensor_value)


def pop_front(array, n=200):
    if len(array) < n:
        raise ValueError("Not enough elements to pop.")
    front = array[:n]
    array = array[n:]  # new view
    return front, array


def read_sensor(queue, analog_pin=0, frequency=1000, window_size=200):
	window = []
	while True:
		file = open_sensor(analog_pin)
		window.append(read_value(file))
		if len(window) >= window_size:
			queue.put(window)
			window = []
		time.sleep(1.0 / frequency)


class SensorInput:
	def __init__(self, analog_pin=0, frequency=1000, window_size=200):
		self.queue = queue.Queue()
		self.analog_pin = analog_pin
		threading.Thread(target=read_sensor, daemon=True, args=[self.queue, analog_pin, frequency, window_size]).start()
		
	def has_next(self):
		return not self.queue.empty()

	def next(self):
		try:
			return self.queue.get_nowait()
		except queue.Empty:
			return None


class FileInput:
	def __init__(self, file_name="output.csv", frequency=1000, window_size=200):
		self.data = np.loadtxt(file_name, delimiter=",", dtype=np.int32)
		self.window_size = window_size
		
	def has_next(self):
		return self.data.shape[0] > 0

	def next(self):
		try:
			window, self.data = pop_front(self.data, n=self.window_size)
			return window
		except queue.Empty:
			return None


def record_sensor_data(time =10, frequency=1000):
	frequency = 1000
	file = open_sensor()
	array = np.array([], dtype=np.int32)
	for i in range(time * frequency):
		array = np.append(array, read_value(file))
		print("Reading value: ", array)
		time.sleep(1.0 / frequency)
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