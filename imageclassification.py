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
def RGB2GRAY(Imagepath):
    image = cv2.imread(Imagepath)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Original image", image)
    # cv2.imshow("Grayscale image", gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return gray
