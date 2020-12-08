import cv2
import tensorflow as tf
import os
from skimage.transform import resize
train = tf.keras.models.load_model('/home/sahma61/Downloads/Proj_final/model_train')
categories = ["Accident", "Non-Accident"]

main_dir = "/home/sahma61/Downloads/Proj_final/output_frames"
for dirs in os.listdir(main_dir):
    test_dir = os.path.join(main_dir, dirs)
    print(test_dir)
    for img_t in os.listdir(test_dir):
        image = cv2.imread(os.path.join(test_dir, img_t), cv2.IMREAD_GRAYSCALE)
        image = resize(image, (128, 128))
        image = image / 255
        img_arr = image.reshape([1] + list(image.shape) + [1])
        predict_class = train.predict_classes(img_arr)
        print(img_t, "\t", categories[predict_class[0]])
