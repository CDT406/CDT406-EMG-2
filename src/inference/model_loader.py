#import tflite_runtime.interpreter as tflite
import tensorflow as tf
import time

def get_model(model_type, model_path, logger=None):
    if model_type == 'default':
        return Model(model_path=model_path, logger=logger)
    elif model_type == 'RNNLTSM':
        return RNNLTSM(model_path=model_path, logger=logger)
    else:
        raise Exception("Error in config")


class Model:
    def __init__(self, model_path, logger=None):
        #self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.logger = logger
        self.time = time.time()
        print("TFLite model loaded.")

    def _mupp_function(self):
        pass


    def get_output_state(self, input_data):
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        self._mupp_function()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        if not self.logger == None:
            #calculate elapsed time in seconds
            elapsed_time = time.time() - self.time
            self.logger(output_data, elapsed_time)

        return output_data


class RNNLTSM(Model):
    def __init__(self, model_path, logger=None):
        super().__init__(model_path=model_path)


    def _mupp_function(self):
        print("Me overriden!")
