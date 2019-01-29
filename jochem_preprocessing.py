# Importing libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import glob
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

# Setting RGB to Grayscale via OpenCV library
def RGB2GRAY(Imagepath):
    image = cv2.imread(Imagepath)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Original image", image)
    # cv2.imshow("Grayscale image", gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return gray

def image_to_matrix(picture):
    img_dir = "Images"
    data_path = os.path.join(img_dir, picture)
    image_matrix = cv2.imread(data_path, 0)
    return np.array(image_matrix)

def create_matrix(csvfile):
    data = open(csvfile, 'rt')
    reader = csv.reader(data)
    next(reader, None)

    data_array = []
    for row in reader:
        row_data = []
        row_data.append(str(row[0]))
        for i in range(1, 5):
            if i == 1 and int(row[i]) == 1:
                row_data.append(np.array([1, 0, 0, 0]))

            elif i == 2 and int(row[i]) == 1:
                row_data.append(np.array([0, 1, 0, 0]))

            elif i == 3 and int(row[i]) == 1:
                row_data.append(np.array([0, 0, 1, 0]))

            elif i == 4 and int(row[i]) == 1:
                row_data.append(np.array([0, 0, 0, 1]))
        row_data.append(image_to_matrix(row[0]))
        data_array.append(row_data)
    return np.asarray(data_array)

# def validation_split(data, ratio):
#     np.random.shuffle(data)
#     split_index = int(round(ratio*len(data)))
#     set1 = data[:split_index]
#     set2 = data[split_index:]
#     return set1,set2

def split_data(data):
    filenames = []
    labels = []
    vectors = []

    for datarow in data:
        filenames.append(datarow[0])
        labels.append(datarow[1])
        vectors.append(datarow[2])
    return np.asarray(filenames), np.asarray(labels), np.asarray(vectors)

# data_matrix = create_matrix('labels.csv')
# training_data, validation_data = validation_split(data_matrix, 0.7)
# train_filenames, train_Y, train_X = split_data(training_data)
# valid_filenames, valid_Y, valid_X = split_data(validation_data)

# def transform_labels(data):
#     labels = []
#     for row in data:
#         for i in range(0, 4):
#             if i == 0 and int(row[i]) == 1:
#                 labels.append([1, 0, 0, 0])
#                 break
#             elif i == 1 and int(row[i]) == 1:
#                 labels.append([0, 1, 0, 0])
#                 break
#             elif i == 2 and int(row[i]) == 1:
#                 labels.append([0, 0, 1, 0])
#                 break    
#             elif i == 3 and int(row[i]) == 1:
#                 labels.append([0, 0, 0, 1])
#                 break

#     return np.asarray(labels)

def create_image_matrix(data):
    images = []
    for filename in data:
        images.append(image_to_matrix(filename))

    return np.asarray(images)


def validation_split(images, data, ratio):
    data_joined = list(zip(images, data))
    np.random.shuffle(data_joined)
    images, data = zip(*data_joined)

    split_index = int(round(ratio*len(images)))

    train_X = images[:split_index]
    train_Y = data[:split_index]

    valid_X = images[split_index:]
    valid_Y = data[split_index:]

    return np.asarray(train_X),np.asarray(train_Y), np.asarray(valid_X), np.asarray(valid_Y)

def transform_labels(labels):
    transformed = []

    for label in labels:
        trans_label = label.astype(float)
        transformed.append(trans_label)

    return np.asarray(transformed)

data_filename = np.genfromtxt('labels.csv', skip_header=True, delimiter=',', dtype=str, usecols=0)
data_labels = np.genfromtxt('labels.csv', skip_header=True, delimiter=',', dtype=float, usecols=(1,2,3,4))

data_images = create_image_matrix(data_filename)
data_matrix = np.column_stack((data_filename, data_labels))

train_X ,joined_train_Y, valid_X, joined_valid_Y = validation_split(data_images, data_matrix, 0.7)

train_Y = joined_train_Y[:,(1,2,3,4)]
valid_Y = joined_valid_Y[:,(1,2,3,4)]

train_Y = transform_labels(train_Y)
valid_Y = transform_labels(valid_Y)


training_data, validation_data = validation_split(data_matrix, 0.7)
train_filenames, train_labels, train_vectors = split_data(training_data)
valid_filenames, valid_labels, valid_vectors = split_data(validation_data)
