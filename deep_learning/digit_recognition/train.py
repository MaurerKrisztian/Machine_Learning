import tensorflow as tf
from tensorflow.keras.datasets import mnist

def load_data():
    # Load the MNIST dataset and split it into training and testing sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize the pixel values to be between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    return x_train, y_train, x_test, y_test

def create_model():
    # Define the model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # flatten the 28x28 input image
        tf.keras.layers.Dense(128, activation='relu'),  # add a fully connected layer with 128 neurons
        tf.keras.layers.Dense(10, activation='softmax')  # add an output layer with 10 neurons (one for each digit) and softmax activation
    ])
    
    # Compile the model with categorical cross-entropy loss and Adam optimizer
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_model(model, x_train, y_train):
    # Train the model on the training set
    model.fit(x_train, y_train, epochs=3, validation_split=0.2)
    
def evaluate_model(model, x_test, y_test):
    # Evaluate the model on the testing set
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc * 100, "%")

def save_model(model, filepath):
    # Save the model to a file
    model.save(filepath)

# Load the data
x_train, y_train, x_test, y_test = load_data()

# Create the model
model = create_model()

# Train the model
train_model(model, x_train, y_train)

# Evaluate the model
evaluate_model(model, x_test, y_test)

model.save('model/digit_recognition_model.h5')