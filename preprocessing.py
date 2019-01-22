# Importing libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

def image_to_matrix(picture):
    image = cv2.imread(picture, 0)
    img_reverted = cv2.bitwise_not(image)
    # matrix_image = np.round((img_reverted / 255), 2)
    return img_reverted

# Setting up csv to dictionary
def csv_to_array(csvfile):
    data = open(csvfile, 'rt')
    reader = csv.reader(data)
    data_array = []

    for row in reader:
        semi_data = []
        for i in range(len(row)):
            semi_data.append(row[i])
        data_array.append(semi_data)
        data = data_array[1:]

    return data

#maakt een array met voor elke foto een lijst met: naam foto, vier getallen en matrix van de foto
# directory is path dat gaat naar map waar de foto's in staan, csvfile is "labels.csv"
def find_pictures(directory, csvfile):
    # data = csv_to_array(csvfile)
    new_data = []

    i = 0

    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".jpg"): 
            matrix = image_to_matrix(filename)
            new_data.append(matrix)
            # print(i)
            i += 1
    return np.asarray(new_data)

final_data = find_pictures("/Users/NiekIJzerman/Desktop/Beta-Gamma/Jaar-3/Blok-3/Leren en beslissen/git/Foodwaste4/", "labels.csv")
print(np.shape(final_data))



# def find_pictures(directory, csvfile):
#     # data = csv_to_array(csvfile)
#     new_data = []

#     i = 0

#     for filename in sorted(os.listdir(directory)):
#         if filename.endswith(".jpg"): 
#             matrix = image_to_matrix(filename)
#             data[i].append(matrix)
#             print(i)
#             i += 1
#             return np.asarray(data)

# final_data = find_pictures("/Users/NiekIJzerman/Desktop/Beta-Gamma/Jaar-3/Blok-3/Leren en beslissen/git/Foodwaste4/", "labels.csv")
# print(final_data)