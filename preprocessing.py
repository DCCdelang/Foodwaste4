# Importing libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os


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
    imgage = cv2.imread(data_path)
    img_reverted = cv2.bitwise_not(imgage)
    matrix_image = np.round((img_reverted / 255), 2)

    return matrix_image

# Setting up csv to dictionary
def csv_to_dict(csvfile):
    with open(csvfile) as fh:
        reader = csv.DictReader(fh, delimiter = ',')
        dic = {}
        for image in reader:
            picture = image.get("Image")
            dic[picture] = image.get("Plate Waste"), image.get("Empty Plate"), image.get("Kitchen Waste"),image.get("No Objects"), image_to_matrix(picture)
    return dic

print(csv_to_dict("labels.csv"))


# img_dir = "20190106_dataset_zero_foodwaste_uva"
# data_path = os.path.join(img_dir,'*g')
# files = glob.glob(data_path)
# data = []
# for f1 in files:
#     if f1 == picture:
#         img = cv2.imread(f1)
#         data.append(img)
# print(data[0])



def main(impath):
    gray = RGB2GRAY(impath)
    image_to_matrix(gray)

    return
