import tflite_runtime as tflite


class ModelLoader:
    def __init__(self,model_path):
        self.model = model_path
        self.input_details = None
        self.output_details = None
        self.interpreter = None

    def load_model(self):
        self.interpreter = tflite.Interpreter(model_path=self.model)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


    def get_output_state(self, input_data):
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data
