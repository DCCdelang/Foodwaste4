import numpy as np
import tensorflow as tf
from preprocessing import image_to_matrix
from tensorflow.keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

grayscalepics = image_to_matrix("20181201154841_646541e6-4df7-41a1-8076-bc3314746fdf.jpg")

# Set up the model
model = Sequential()

# Add convolutional layer with 3, 3 by 3 filters and a stride size of 1
# Set padding so that input size equals output size
model.add(Conv2D(6,2,input_shape=(1440, 1920)))

# Add relu activation to the layer 
model.add(Activation('relu'))

#Pooling
model.add(MaxPool2D(2))

#Fully connected layers
# Use Flatten to convert 3D data to 1D
model.add(Flatten())

# Add dense layer with 10 neurons
model.add(Dense(10))

# we use the softmax activation function for our last layer
model.add(Activation('softmax'))

"""Before the training process, we have to put together a learning process in a particular form. It consists of 3 elements: an optimiser, a loss function and a metric."""
model.compile(loss='sparse_categorical_crossentropy', optimizer = 'adam', metrics=['acc'])
# dataset with handwritten digits to train the model on
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train, y_train)

x_train = np.expand_dims(x_train,-1)
x_test = np.expand_dims(x_test,-1)
# model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test,y_test))
# model.summary

