import time
import os.path


class LedControl:
    def __init__(self):
        self.LED_count = 4
        self.LED_path = '/sys/class/leds/beaglebone:green:usr'
        self.leds = []
        self.is_on_board = os.path.exists(self.LED_path)
        
        if (self.is_on_board):        
            # Turn off triggers
            for i in range(self.LED_count):
                with open(self.LED_path + str(i) + "/trigger", "w") as f:
                    f.write("none")

            # Open a file for each led
            for i in range(self.LED_count):
                self.leds.append(open(self.LED_path + str(i) + "/brightness", "w"))
        else:
            self.current_state = [False] * self.LED_count
            

    # Write to LED if on board, else print to terminal
    def setLED(self, led=0, state=False):
        if (self.is_on_board):  
            val = "1" if state else "0"
            self.leds[led].seek(0)
            self.leds[led].write(val)
            self.leds[led].flush()    
        else:
            self.current_state[led] = state
            print(self.current_state, end="\r", flush=True)   
    

    def __del__(self):
        if (self.is_on_board):
            for led in self.leds:
                led.close()


if __name__ == '__main__':
    led_c = LedControl()
    
    led_c.setLED(state=True)
    time.sleep(3)
    led_c.setLED(state=False)
    