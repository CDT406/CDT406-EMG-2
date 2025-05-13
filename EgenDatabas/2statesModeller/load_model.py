from tensorflow import keras

def load_semg_model(model_path):
    model = keras.models.load_model(model_path)
    model.summary()  # Optional: shows model structure
    return model