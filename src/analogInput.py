import io
import time
stream = io.open("/sys/bus/iio/devices/iio:device0/in_voltage0_raw", mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True)
while True:
	content = stream.read()
	stream.seek(0)
	# Process the binary content as needed
	print(content)
	time.sleep(0.001)
