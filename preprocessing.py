# Foodwaste Group 4
# Dante de Lang
# Darius Barsony
# Niek Ijzerman
# Jochem Soons
# Jeroen van Wely

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
    reader = csv.DictReader(open(csvfile, 'r'))
    dictionary = []
    for image in reader:
        dictionary.append(image)
    return dictionary

print(csv_to_dict("labels.csv"))
