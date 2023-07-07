import cv2
import numpy as np
import matplotlib.pyplot as plt
image_original = cv2.imread('sample2.jpg')
image = cv2.resize(image_original, (400, 400))
kernel = np.ones((5, 5), np.uint8)
eroded_image = cv2.erode(image, kernel, iterations=1)
dilated_image = cv2.dilate(image, kernel, iterations=1)
eroded_edge_image = cv2.absdiff(image, eroded_image)
dilated_edge_image = cv2.absdiff(image, dilated_image)
plt.figure(figsize=(12, 8))
plt.subplot(331) #--> 3 rows , 3 columns , 1st index
plt.imshow(image, cmap='gray')# Displays the image using the grayscale color map which maps the pixel
plt.title('Original Image') #Provides brief descripton of the image
plt.subplot(332)
plt.imshow(eroded_image, cmap='gray')
plt.title('Eroded Image')
plt.subplot(333)
plt.imshow(eroded_edge_image, cmap='gray')
plt.title('Eroded Edge Image')
plt.subplot(335)
plt.imshow(dilated_edge_image, cmap='gray')
plt.title('Dilated Edge Image')
plt.subplot(334)
plt.imshow(dilated_image, cmap='gray')
plt.title('Dilated Image')
plt.tight_layout()
plt.show()
