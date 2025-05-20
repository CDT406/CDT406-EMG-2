class MockADC:
    @staticmethod
    def setup():
        print("Mock ADC setup")
    
    @staticmethod
    def read(pin):
        return 0.0  # Mock value