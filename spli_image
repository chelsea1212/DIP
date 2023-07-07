import cv2
import numpy as np
 

image = cv2.imread("/home/sahyadri/lab/flower.jpg")
(h, w) = image.shape[:2]
cv2.imshow('Original', image)
 
# compute the center coordinate of the image
(cX, cY) = (w // 2, h // 2)

# crop the image into four parts which will be labelled as
# top left, top right, bottom left, and bottom right.
TL = cv2.copyMakeBorder(
    image[0:cY, 0:cX], 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 255, 255])
TR = cv2.copyMakeBorder(
    image[0:cY, cX:w], 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 255, 255])
BL = cv2.copyMakeBorder(
    image[cY:h, 0:cX], 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 255, 255])
BR = cv2.copyMakeBorder(
    image[cY:h, cX:w], 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 255, 255])
 
# visualize the cropped regions
# cv2.imwrite("TL", topLeft)
# cv2.imwrite("TR", topRight)
# cv2.imwrite("BL", bottomLeft)
# cv2.imwrite("BR", bottomRight)

H1 = np.concatenate((TL, TR), axis=1)
H2 = np.concatenate((BL, BR), axis=1)
V = np.concatenate((H1, H2), axis=0)

cv2.imshow("Output",V)

cv2.waitKey(0)
