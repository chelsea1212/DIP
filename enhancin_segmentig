import cv2
import numpy as np
img = cv2.imread('sahyadri.jpg')
img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_equalize = cv2.equalizeHist(img_g)
result = np.hstack((img_g, img_equalize))
ret, img_thresh = cv2.threshold(img_equalize, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('Original Image', img)
cv2.imshow('Original Image Converted to Grayscale', img_g)
cv2.imshow('Enhanced Image', img_equalize)
cv2.imshow('original image & enhanced image', result)
cv2.imshow('Segmented Image', img_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
