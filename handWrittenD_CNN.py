# Convolutional Neural Network

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (28, 28, 1), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 10, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.datasets import mnist
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

classifier.fit(x=X_train, y=y_train,
                         batch_size = 32,
                         epochs = 100,
                         verbose = 1
                         )


score = classifier.evaluate(X_test, y_test, verbose=0)

##Prediction Part------------------------
##from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
##from skimage.color import rgb2gray
import matplotlib.pyplot as plt
#Pred from another pics (Not Working!!)
img_pred = image.load_img('D:/handWrittenDigits/4.jpg', target_size = (28, 28))
img_pred = img_pred.convert('L')
plt.imshow(img_pred)
img_pred = image.img_to_array(img_pred)
##img_pred = rgb2gray(img_pred)
##keras.backend.reshape(img_pred, (28,28,1))
##img_pred = img_pred.reshape(28, 28, 1)

#Predict from test_set
img_pred = np.expand_dims(img_pred, axis = 0)
rslt = classifier.predict(X_test)
plt.imshow(image.array_to_img(X_test[20]))


##Save model to json
import os
from keras.models import model_from_json

clssf = classifier.to_json()
with open("handWD.json", "w") as json_file:
    json_file.write(clssf)
classifier.save_weights("handWDweights.h5")
print("model saved to disk....")
