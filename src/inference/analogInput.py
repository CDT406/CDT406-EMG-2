import io
import time
import threading
import queue

def read_sensor(queue, analog_pin, frequency=1000, window_size=200):
	window = []
	while True:
		with open("/sys/bus/iio/devices/iio:device0/in_voltage{}_raw".format(analog_pin), mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True) as file:
			sensor_value = file.read()
			file.seek(0)
			window.append(sensor_value)
			if len(window) >= window_size:
				queue.put(int(window))
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

if __name__ == "__main__":
	sensor = SensorInput()
	while True:
		if sensor.has_next():
			data = sensor.next()
			print(data)