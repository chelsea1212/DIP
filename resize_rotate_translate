import cv2 as cv
import numpy as np

img = cv.imread('/home/sahyadri/lab/flower.jpg')
cv.imshow('OriginalImage', img)
resized_img = cv.resize(img, None, fx=0.5, fy=0.5)
cv.imshow('scalled down Image', resized_img)

rotated_image = cv.rotate(resized_img, cv.ROTATE_90_CLOCKWISE)

cv.imshow('Rotated Image', rotated_image)
height, width = resized_img.shape[:2]
  
quarter_height, quarter_width = height / 4, width / 4
  
T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]])
  

img_translation = cv.warpAffine(resized_img, T, (width, height))
  

cv.imshow('Translation', img_translation)
cv.waitKey(0)
if cv.waitKey(1) & 0xFF == ord('q'):
# cv.waitKey(0)
    cv.destroyAllWindows()
