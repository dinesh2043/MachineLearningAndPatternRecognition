# Let's start by importing numpy and setting a seed for the computer's pseudorandom number generator. 
# This allows us to reproduce the results from our script:
import numpy as np
np.random.seed(123)  # for reproducibility

# Sequential model type from Keras is simply a linear stack of neural network layers, and 
# it's perfect for the type of feed-forward CNN
from keras.models import Sequential
# from keras import backend as K
# K.set_image_dim_ordering('th')

# let's import the "core" layers from Keras. These are the layers that are used in almost any neural network:
from keras.layers import Dense, Dropout, Activation, Flatten

# import the CNN layers from Keras. These are the convolutional layers that will help us efficiently train on image data:
from keras.layers import Convolution2D, MaxPooling2D

# import some utilities. This will help us transform our data later:
from keras.utils import np_utils

# The Keras library conveniently includes it already. We can load it like so:
from keras.datasets import mnist

# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# We can look at the shape of the dataset:
X_train.shape

y_train.shape

X_test.shape

from matplotlib import pyplot as plt
plt.imshow(X_train[0])
plt.show()

# Preprocess input data for Keras.
# When using the Theano backend, you must explicitly declare a dimension for the depth of the input image. For example, a 
# full-color image with all 3 RGB channels will have a depth of 3.
# Our MNIST images only have a depth of 1, but we must explicitly declare that.
# Reshape input data
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

# To confirm, we can print X_train's dimensions again:
X_train.shape

# The final preprocessing step for the input data is to convert our data type to float32 and normalize our data values to 
# the range [0, 1].
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

X_train[0]

# Preprocess class labels for Keras
y_train.shape

# We should have 10 different classes, one for each digit, but it looks like we only have a 1-dimensional array. 
# Let's take a look at the labels for the first 10 training samples:
y_train[:10]

uniques, ids = np.unique(y_test, return_inverse=True)
uniques, ids

# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

Y_test

# Define model architecture
# declaring a sequential model format:
model = Sequential()

# declare the CNN input layer:
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1,28,28)))
# model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1,28,28), 
#                       dim_ordering='th',data_format="channels_first",))

# But what do the first 3 parameters represent? They correspond to the number of convolution filters to use, the number 
# of rows in each convolution kernel, and the number of columns in each convolution kernel, respectively.
# *Note: The step size is (1,1) by default, and it can be tuned using the 'subsample' parameter.
print (model.output_shape)

# we can simply add more layers to our model like we're building legos:
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# So far, for model parameters, we've added two Convolution layers. To complete our model architecture, 
# let's add a fully connected layer and then the output layer:
# Fully connected Dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# For Dense layers, the first parameter is the output size of the layer. Keras automatically handles the connections between 
# layers.

# Note that the final layer has an output size of 10, corresponding to the 10 classes of digits.

# Also note that the weights from the Convolution layers must be flattened (made 1-dimensional) before passing them to the 
# fully connected Dense layer.

# Now all we need to do is define the loss function and the optimizer, and then we'll be ready to train it.
# We just need to compile the model and we'll be ready to train it. When we compile the model, we declare the 
# loss function and the optimizer (SGD, Adam, etc.).
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# To fit the model, all we have to do is declare the batch size and number of epochs to train for, then pass in 
# our training data.
history = model.fit(X_train, Y_train, validation_split=0.166666, 
          batch_size=32, epochs=10, verbose=1)

model.summary()

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

y_pred = model.predict(X_test)
y_pred.ravel()
reasult = y_pred.astype(int)
reasult = np.take(uniques,np.argmax(y_pred,1))
reasult

from pandas_ml import ConfusionMatrix
confusion_matrix = ConfusionMatrix(y_test, reasult)
print("Confusion matrix:\n%s" % confusion_matrix)

confusion_matrix.stats_class

# Evaluate model on test data.
# Finally, we can evaluate our model on the test data:
score = model.evaluate(X_test, Y_test, verbose=1)

score

sum = 0
for i in range (y_test.size):
    sum =sum + abs(reasult[i].astype(int)-y_test[i].astype(int))
    print (sum, i, reasult[i],y_test[i])

sum
MAE = sum/y_test.size
MAE


