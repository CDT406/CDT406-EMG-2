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


    def set_state(self, state):
        val = "1" if state else "0"
        self.write_state(self.current_state, "0")
        self.write_state(state, "0")
        self.current_state = state


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
    

    led_c.set_state(1)
    time.sleep(1)
    led_c.set_state(2)
    time.sleep(1)
    led_c.set_state(3)
