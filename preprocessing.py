# Importing libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

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

    imgage = cv2.imread(picture, 0)
    img_reverted = cv2.bitwise_not(imgage)
    matrix_image = np.round((img_reverted / 255), 2)

    return matrix_image

# Setting up csv to dictionary
def csv_to_dict(csvfile):
    with open(csvfile) as fh:
        reader = csv.DictReader(fh, delimiter = ',')
        dic = {}
        for image in reader:
            dic[image.get("Image")] = image.get("Plate Waste"), image.get("Empty Plate"), image.get("Kitchen Waste"),image.get("No Objects"), 
            image_to_matrix(picture)
    return dic

print(csv_to_dict("labels.csv")[0].get('Image'))


# def main():
#     return
