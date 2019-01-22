import numpy as np
import tensorflow as tf
from preprocessing import image_to_matrix
from tensorflow.keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

x_train = []
pic = image_to_matrix("20181201154841_646541e6-4df7-41a1-8076-bc3314746fdf.jpg")
x_train.append(pic)
pic = image_to_matrix("20181201155022_41021fcc-070f-45f7-a73b-679f6b09259b.jpg")
x_train.append(pic)
pic = image_to_matrix("20181201155006_ecdfcf72-d369-4619-a979-3ceee3abaee6.jpg")
x_train.append(pic)

y_train = [2, 1, 2]

x_test = []
pic = image_to_matrix("20181201155220_862367f1-0dde-4679-b063-4dd8f0e4274f.jpg")
x_test.append(pic)
pic = image_to_matrix("20181201155233_6d39000c-2594-4904-a431-5c8e7f043396.jpg")
x_test.append(pic)
pic = image_to_matrix("20181201155328_d4ba3da3-8993-4d8d-876e-33066fe7d0f4.jpg")
x_test.append(pic)

y_test = [1, 1, 3]

# Set up the model
model = Sequential()

# Add convolutional layer with 3, 3 by 3 filters and a stride size of 1
# Set padding so that input size equals output size
model.add(Conv2D(6, 2, input_shape=(1440, 1920, 1)))

# Add relu activation to the layer 
model.add(Activation('relu'))

#Pooling
model.add(MaxPool2D(2))

#Fully connected layers
# Use Flatten to convert 3D data to 1D
model.add(Flatten())

# Add dense layer with 10 neurons
model.add(Dense(5))

# we use the softmax activation function for our last layer
model.add(Activation('softmax'))

"""Before the training process, we have to put together a learning process in a particular form. 
        It consists of 3 elements: an optimiser, a loss function and a metric."""
model.compile(loss='sparse_categorical_crossentropy', optimizer = 'adam', metrics=['acc'])
# dataset with handwritten digits to train the model on
# from keras.datasets import mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.expand_dims(x_train,-1)
x_test = np.expand_dims(x_test,-1)
model.fit(x_train, y_train, batch_size=1, epochs=1, validation_data=(x_test,y_test))
model.summary

