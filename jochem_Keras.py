from jochem_preprocessing import *
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

#create model
model = Sequential()

#add model layers
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(192,256, 1)))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(4, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_X = np.expand_dims(train_X, -1)
valid_X = np.expand_dims(valid_X, -1)

#train the model
model.fit(train_X, train_Y, validation_data=(valid_X, valid_Y), epochs=1)

#predict first 4 images in the test set
print(model.predict(valid_X[:20]))

#actual results for first 4 images in test set
print(valid_Y[:20])