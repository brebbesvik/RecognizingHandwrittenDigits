import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

# Read the data from CSV files. Mye experience is using pandas for reading is faster
X = pd.read_csv('../handwritten_digits_images.csv', header=None).values
y = pd.read_csv('../handwritten_digits_labels.csv', header=None).values

# Split the data set into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.3, random_state=2)

# Reshape the data in X. Both training and test sets. Put the channel to the end
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Convert to float and represent the data from 0-1
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# The output should be 10 neurons
Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)


# Set up the model with layers an neurons
model = Sequential() # initialize
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Inspect the created model
print(model.summary())

#Then you need to compile model (for speed):
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#           model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model (X is features, Y is output, this is for training data)
model.fit(X_train, Y_train, batch_size=200, epochs=10, validation_data=(X_test, Y_test), verbose=1)
#           model.fit(X_train, Y_train, epochs=150, batch_size=10)

# epochs are the number of iterations wanted.
# batch size the number of iterations before update
# (150 epoches, 10 batch size = 15 #updates).
# score that should be at least 95%
#           scores = model.evaluate(X_test, Y_test)
#           print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
