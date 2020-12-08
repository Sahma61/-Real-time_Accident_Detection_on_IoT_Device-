import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Dense, Flatten
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize

X = np.load("train_features.npy")
print("St 1")
Y = np.load("train_labels.npy")
print("St 2")
X = X / 255
print("St 3")
X = X.reshape(list(X.shape) + [1])
print("St 4")
print(X.shape)

#creating my model
model = Sequential();

#adding first layer i.e convolutional layer, passing input_shape only in first layer
model.add(Conv2D(32, (5 , 5), activation="relu", input_shape=(128, 128, 1)))

#adding a pooling layer
model.add(MaxPooling2D(pool_size = (2 , 2)))

#adding another convolutional layer
model.add(Conv2D(32, (5 , 5), activation="relu"))

#adding another pooling layer
model.add(MaxPooling2D(pool_size = (2 , 2)))

#adding flatteing layer i.e. fully connected layer
model.add(Flatten())

#Embedding nuerons using dense layer
model.add(Dense(1000, activation="relu"))

#Adding a dropout with 50% droupout rate
model.add(Dropout(0.5))

#Embedding nuerons using dense layer
model.add(Dense(500, activation="relu"))

#Adding a dropout with 50% droupout rate
model.add(Dropout(0.5))

#Embedding nuerons using dense layer
model.add(Dense(250, activation="relu"))

#Embedding nuerons using dense layer
model.add(Dense(2, activation="softmax"))

#compiling the model
model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

#training the model
train = model.fit(X, Y, batch_size = 256, epochs = 10, validation_split=0.2, shuffle=True)
model.save('model_train')

#plotting the model accuracy
plt.plot(train.history["accuracy"])
plt.plot(train.history["val_accuracy"])
plt.title("Model Accuracy Visualization")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(['train', 'val'], loc="upper right")
plt.show()
