import cv2
import numpy as np
from matplotlib import pyplot as plt

def edge_detection(image):
    img = cv2.imread(image,0)
    edges = cv2.Canny(img,100,200)

    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()

def image_gradients(image):
    img = cv2.imread(image,0)

    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

    plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

    plt.show()
    sobelx = cv2.imwrite('sobelx.jpg', sobelx)
    return sobelx

def hough_transform(image):
    import cv2
    import numpy as np
 
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 75, 150)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=250)
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    plt.subplot(2, 2, 1),plt.imshow(gray, cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2),plt.imshow(edges,cmap = 'gray')
    plt.title('Edges'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3),plt.imshow(img,cmap = 'gray')
    plt.title('Lines'), plt.xticks([]), plt.yticks([])
    plt.show()

def filter_background(image):
    img = cv2.imread(image)
    mask = np.zeros(img.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    rect = (500,500,450,290)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]

    plt.imshow(img),plt.colorbar(),plt.show()

# # print(edge_detection('20190106_dataset_zero_foodwaste_uva/20181201154841_646541e6-4df7-41a1-8076-bc3314746fdf.jpg'))
# sobelx = image_gradients('20190106_dataset_zero_foodwaste_uva/20181201154841_646541e6-4df7-41a1-8076-bc3314746fdf.jpg')
# # print(hough_transform('20190106_dataset_zero_foodwaste_uva/20181201154841_646541e6-4df7-41a1-8076-bc3314746fdf.jpg'))
# # print(hough_transform('20190106_dataset_zero_foodwaste_uva/20181201155006_ecdfcf72-d369-4619-a979-3ceee3abaee6.jpg'))
# # print(filter_background('20190106_dataset_zero_foodwaste_uva/20190104183839_7d0b4a6c-7911-4227-8901-6147316b9bf3.jpg'))
# print(hough_transform('sobelx.jpg'))
