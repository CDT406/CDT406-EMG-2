import time
import os.path


class PrintControl:
    def __init__(self):
        self.LED_count = 4
        self.current_state = 0


    def set_state(self, state):
        self.current_state = state
        print(f"Current state: {self.current_state}", end="\r", flush=True)


class LedControl:
    def __init__(self):
        self.LED_count = 4
        self.LED_path = '/sys/class/leds/beaglebone:green:usr'
        self.leds = []
        self.current_state = 0
        self.is_on_board = os.path.exists(self.LED_path)

        if (not self.is_on_board):
            print("Not running on a Beaglebone")
            return

        # Turn off triggers
        for i in range(self.LED_count):
            with open(self.LED_path + str(i) + "/trigger", "w") as f:
                f.write("none")

        # Open a file for each led
        for i in range(self.LED_count):
            self.leds.append(open(self.LED_path + str(i) + "/brightness", "w"))


    def set_state(self, state=[0]):
        if (len(state) > self.LED_count):
            raise Exception("Too large state given, 4 max.")

        new_state = state.index(max(state))
        self.write_state(self.current_state, "0")
        self.write_state(new_state, "1")
        self.current_state = new_state


    def write_state(self, led, val):
        self.leds[led].seek(0)
        self.leds[led].write(val)
        self.leds[led].flush()


    def __del__(self):
        if (self.is_on_board):
            for led in self.leds:
                led.close()


if __name__ == '__main__':
    # led_c = LedControl()
    led_c = PrintControl()


    led_c.set_state([1, 2, 0, 0])
    time.sleep(1)
    led_c.set_state([1, 2, 3, 0])
    time.sleep(1)
    led_c.set_state([1, 2, 0, 4])
    time.sleep(1)
    led_c.set_state([1, 2, 0, 4, 5])
