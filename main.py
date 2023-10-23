import os
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# I downloaded the images from the Kitti database: https://www.cvlibs.net/datasets/kitti/eval_road.php
# I set relative paths.
data_path = r'data_road\training'

image_folder = os.path.join(data_path, r'image_2')
label_folder = os.path.join(data_path, r'gt_image_2')

image_files = os.listdir(image_folder)
label_files = os.listdir(label_folder)

# Array for storing images
images = []
labels = []

# Load images and (image)labels into the array
for image_file, label_file in zip(image_files, label_files):
    image_path = os.path.join(image_folder, image_file)
    label_path = os.path.join(label_folder, label_file)

    image = cv2.imread(image_path)
    image = cv2.resize(image, (1242, 375))
    label = cv2.imread(label_path)
    label = cv2.resize(label, (1242, 375))

    if image is not None and label is not None:
        images.append(image)

        # Modify the (img)labels
        # Example: A (purple pixels) is [255, 0.255] -> 1, everything else -> 0
        target_color = [255, 0, 255]
        binary_mask = np.all(label == target_color, axis=-1).astype(np.uint8)
        labels.append(binary_mask)

# Convert images and labels to np tensors
images = np.array(images)
labels = np.array(labels)

# Train, Validation, Test data distribution (80%, 10%, 10%)
x_train, x_valid, y_train, y_valid = train_test_split(images, labels, test_size=0.2, random_state=42)
x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, test_size=0.5, random_state=42)

# Check dimensions
print(np.shape(x_train))
print(np.shape(y_train))
print(np.shape(x_valid))
print(np.shape(y_valid))
print(np.shape(x_test))
print(np.shape(y_test))

# Convert back the (image)label for the first image of the y_test
rgb_labels = np.zeros((labels[0].shape[0], labels[0].shape[1], 3), dtype=np.uint8)
rgb_labels[y_test[0] == 1] = [255, 0, 255]
rgb_labels[y_test[0] == 0] = [0, 0, 0]

# Visualize an input and output image
cv2.imshow('INPUT Image', x_test[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('OUTPUT Image', rgb_labels)
cv2.waitKey(0)
cv2.destroyAllWindows()