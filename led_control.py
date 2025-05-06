import time


class LedControl:
    def __init__(self):
        self.LED_count = 4
        self.LED_path = '/sys/class/leds/beaglebone:green:usr'
        self.leds = []

        # Turn off triggers
        for i in range(self.LED_count):
            with open(self.LED_path + str(i) + "/trigger", "w") as f:
                f.write("none")

        # Open a file for each led
        for i in range(self.LED_count):
            self.leds.append(open(self.LED_path + str(i) + "/brightness", "w"))

    def setLED(self, led=0, state=False):
        val = "1" if state else "0"
        self.leds[led].seek(0)
        self.leds[led].write(val)
        self.leds[led].flush()            
    

    def __del__(self):
        for led in self.leds:
            led.close()


class TestLedControl:
    def __init__(self):
        self.LED_count = 4
        self.current_state = [False] * self.LED_count
        

    def setLED(self, led=0, state=False):
        self.current_state[led] = state
        print(self.current_state, end="\r", flush=True)


if __name__ == '__main__':
    # led_c = LedControl()
    led_c = TestLedControl()
    led_c.setLED(state=True)
    time.sleep(3)
    led_c.setLED(state=False)
    