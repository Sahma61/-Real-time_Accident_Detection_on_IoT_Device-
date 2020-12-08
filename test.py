import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Dense, Flatten
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize

train = tf.keras.models.load_model('/home/sahma61/Downloads/Proj_final/model_train')
categories = ["Accident", "Non-Accident"]

img_test = np.load("/home/sahma61/Downloads/Proj_final/test_features.npy")
label_test = np.load("/home/sahma61/Downloads/Proj_final/test_labels.npy")
img_test = img_test / 255
img_test = img_test.reshape(list(img_test.shape) + [1])
print(img_test.shape)
train.evaluate(img_test, label_test)[1]
print("index \t label  \t predicted")
classes = train.predict_classes(img_test)
for i in range(len(label_test)):
    print(i+1,"\t", categories[label_test[i]], "\t", categories[classes[i]])
