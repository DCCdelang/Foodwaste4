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
            dic[image.get("Image")] = image.get("Plate Waste"), image.get("Empty Plate"), image.get("Kitchen Waste"),image.get("No Objects")
    return dic

Dictionary = csv_to_dict("labels.csv")
Dictionary['20190105133933_5ff99ea4-30d1-4e65-87dc-b3ed496c8711.jpg'] = 2013
print(Dictionary.get('20190105133933_5ff99ea4-30d1-4e65-87dc-b3ed496c8711.jpg'))




def main(impath):
    gray = RGB2GRAY(impath)
    image_to_matrix(gray)

    return
