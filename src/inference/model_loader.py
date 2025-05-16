import tflite_runtime.interpreter as tflite


class Model:
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


    def _mupp_function(self):
        pass


    def get_output_state(self, input_data):
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        self._mupp_function()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data


class RNNLTSM(Model):
    def __init__(self, model_path):
        super().__init__(model_path=model_path)
        
        
    def _mupp_function(self):
        print("Me overriden!")