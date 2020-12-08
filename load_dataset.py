import numpy as np
import random
import matplotlib.pyplot as plt 
import os
import cv2
from skimage.transform import resize

Train_DIR = r"/home/sparsh/Desktop/Accident_Detection/Accident-Dataset/train"
Test_DIR = r"/home/sparsh/Desktop/Accident_Detection/Accident-Dataset/test"
categories = ["Accident", "Non-Accident"]
test_data = []
train_data = []

def getTrainData():
    i = 0
    for index, category in enumerate(categories):
        path = os.path.join(Train_DIR , category)
        for img in os.listdir(path):
            i += 1
            print(i)
            try:
                image = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                image = resize(image, (128, 128))
                train_data.append([image , index])   
            except Exception as e:
                pass



def getTestData():
    i = 0
    for index, category in enumerate(categories):
        path = os.path.join(Test_DIR , category)
        for img in os.listdir(path):
            i += 1
            print(i)
            try:
                image = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                image = resize(image, (128, 128))
                test_data.append([image , index])   
            except Exception as e:
                pass
print("Getting Train Data")
getTrainData()

print("Getting Test Data")
getTestData()

random.shuffle(train_data)
random.shuffle(test_data)

train_features = []
train_labels = []
test_features = []
test_labels = []

for feature, label in train_data:
    train_features.append(feature)
    train_labels.append(label)
    
train_features = np.array(train_features)
train_labels = np.array(train_labels)
print(type(train_features))

for feature, label in test_data:
    test_features.append(feature)
    test_labels.append(label)
    
test_features = np.array(test_features)
test_labels = np.array(test_labels)
print(type(test_features))

print("Saving Numpy Arrays")

np.save("train_features", train_features)
np.save("train_labels", train_labels)
np.save("test_features", test_features)
np.save("test_labels", test_labels)
