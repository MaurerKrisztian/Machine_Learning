import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

def load_model(filepath):
    # Load the model from a file
    model = tf.keras.models.load_model(filepath)
    return model

def show_image(image, predicted_digit):
     # Show the image and the predicted digit
    plt.imshow(image[0], cmap=plt.cm.binary)
    plt.title(f"Predicted digit: {predicted_digit}")
    plt.show()

def make_prediction(model, image):
    # Make a prediction on a single image
    image = np.expand_dims(image, axis=0)  # add a batch dimension
    prediction = model.predict(image)  # predict the class probabilities
    digit = np.argmax(prediction)  # get the digit with the highest probability
    
    show_image(image, digit)
    
    return digit



(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Load the model
model = load_model('model/digit_recognition_model.h5')

# Make a prediction
image = x_test[0]
digit = make_prediction(model, image)