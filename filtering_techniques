import cv2
import numpy as np
import matplotlib.pyplot as plt
image_original = cv2.imread('sample2.jpg')
image = cv2.resize(image_original, (400, 400))
edges = cv2.Canny(image, 100, 200)
blur = cv2.blur(image, (5, 5)) #Uses a 5x5 kernel to apply a simple averaging blur to the image.
laplacian = cv2.Laplacian(image, cv2.CV_64F) #The Laplacian is a 2nd order filter
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.imshow(image, cmap='gray')#displays the image using the grayscale color map which maps
plt.title('Original Image') 
plt.subplot(232)
plt.imshow(edges, cmap='gray')
plt.title('Edges')
plt.subplot(233)
plt.imshow(blur, cmap='gray')
plt.title('Blur')
plt.subplot(234)
plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian')
plt.subplot(235)
plt.imshow(sobel_x, cmap='gray')
plt.title('Sobel X')
plt.subplot(236)
plt.imshow(sobel_y, cmap='gray')
plt.title('Sobel Y')
plt.tight_layout()
plt.show()
