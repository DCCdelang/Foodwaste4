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

#Setting RGB to Grayscale via OpenCV library
image = cv2.imread('C:/Users/djdcc_000/Documents/School/UNI/Beta-Gamma/2018-2019/Leren_en_Beslissen/Git/Foodwaste4/20190106_dataset_zero_foodwaste_uva/20181201155604_2e04dad4-f5a1-4400-93bb-9ce940ff7a23.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# cv2.imshow("Original image", image)
# cv2.imshow("Grayscale image", gray)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
