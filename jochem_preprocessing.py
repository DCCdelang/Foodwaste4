# Importing libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import glob

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
    img_dir = "20190106_dataset_zero_foodwaste_uva"
    data_path = os.path.join(img_dir, picture)
    image = cv2.imread(data_path, 0)
    matrix_image = cv2.bitwise_not(image)

    return matrix_image

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
                break
            elif i == 2 and int(row[i]) == 1:
                row_data.append(np.array([0, 1, 0, 0]))
                break

            elif i == 3 and int(row[i]) == 1:
                row_data.append(np.array([0, 0, 1, 0]))
                break
            
            elif i == 4 and int(row[i]) == 1:
                row_data.append(np.array([0, 0, 0, 1]))
        
        row_data.append(image_to_matrix(row[0]))
        data_array.append(row_data)
    return np.asarray(data_array)

def validation_split(data, ratio):
    np.random.shuffle(data)
    split_index = int(round(ratio*len(data)))
    set1 = data[:split_index]
    set2 = data[split_index:]
    return set1,set2

def split_data(data):
    
data_matrix = create_matrix('labels.csv')
print(data_matrix[0][1])
training_data, validation_data = validation_split(data_matrix, 0.7)
