# Importing libraries
import cv2
import numpy as np
import os
# import matplotlib.pyplot as plt
# import csv
# import glob

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
=======
# given the name of a picture, finds the picture in the database, 
# converts this image to grayscale and finaly converts this to a 2D 
# matrix and returns this
def image_to_matrix(picture):
    img_dir = "20190106_dataset_zero_foodwaste_uva"
    data_path = os.path.join(img_dir, picture)
    imgage = cv2.imread(data_path)
    grayimg = cv2.cvtColor(imgage,cv2.COLOR_BGR2GRAY)
    img_reverted = cv2.bitwise_not(grayimg)
    matrix_image = np.round((img_reverted / 255), 2)
    return matrix_image

# Setting RGB to Grayscale via OpenCV library
# def RGB2GRAY(Imagepath):
    # image = cv2.imread(Imagepath)
    # gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     # cv2.imshow("Original image", image)
#     # cv2.imshow("Grayscale image", gray)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     return gray

# def all_images_to_matrix_list():
#     img_dir = "20190106_dataset_zero_foodwaste_uva"
#     data_path = os.path.join(img_dir,'*g')
#     files = glob.glob(data_path)
#     data = []
#     for f1 in files:
#         print(f1)
#         img = cv2.imread(f1, 0)
#         img_reverted = cv2.bitwise_not(img)
#         matrix_image = np.round((img_reverted / 255), 2)
#         data.append(matrix_image)
#     return data

# Setting up csv to dictionary
# def csv_to_array(csvfile):
#     data = open(csvfile, 'rt')
#     reader = csv.reader(data)
#     data_array = []
#     for row in reader:
#         semi_data = []
#         for i in range(len(row)):
#             semi_data.append(row[i])
#         data_array.append(semi_data)
#         data = data_array[1:]
#     return data

#maakt een array met voor elke foto een lijst met: naam foto, vier getallen en matrix van de foto
# directory is path dat gaat naar map waar de foto's in staan, csvfile is "labels.csv"
# def find_pictures(directory, csvfile):
#     data = csv_to_array(csvfile)
#     i = 0
#     for filename in sorted(os.listdir(directory)):
#         if filename.endswith(".jpg"):
#             matrix = image_to_matrix(filename)
#             data[i].append(matrix)
#             i += 1
#     return np.asarray(data)

# Setting up csv to dictionary
# def csv_to_dict(csvfile):
#     matrix_list = all_images_to_matrix_list()
#     with open(csvfile) as fh:
#         reader = csv.DictReader(fh, delimiter = ',')
#         dic = {}
#         counter = 0
#         for image in reader:
#             picture = image.get("Image")
#             dic[picture] = image.get("Plate Waste"), image.get("Empty Plate"), image.get("Kitchen Waste"),image.get("No Objects"), matrix_list[counter]
#             counter += 1
#     return dic

# def main():
#     return
>>>>>>> 6c94fe15042a0dd9b37952113e4171b8ad8edea2
