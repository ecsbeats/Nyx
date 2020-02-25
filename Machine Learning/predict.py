# Imports
import keras
from keras.models import load_model as load_model_keras

# load model function
def load_model(path='./models/Model_Saves/Final_Model.model'):
    model = load_model_keras(path)
    return model

# Shape image function
def shape_image(image):
    image = image.reshape((50, 50, 1))
    image = np.expand_dims(image, axis=0)
    return image

# Prediction function
def make_prediction(image, model):
    return model.predict(image)

# Utilized prediction function
def predict(image, model):
    shaped_image = shape_image(image)
    prediction = make_prediction(image, model)
    return prediction
