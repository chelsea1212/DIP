import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("/content/wallpaper1.jpg", cv2.IMREAD_GRAYSCALE)

# Define the standard deviations for the Gaussian blurring
sigma1 = 1.0
sigma2 = 2.0

# Apply Gaussian blurring
blur1 = cv2.GaussianBlur(image, (0, 0), sigma1)
blur2 = cv2.GaussianBlur(image, (0, 0), sigma2)

# Compute the Difference of Gaussians
dog = blur2 - blur1

# Thresholding
threshold = 30
_, thresholded = cv2.threshold(dog, threshold, 255, cv2.THRESH_BINARY)

# Display the original image and the DoG result
plt.figure(figsize=(10, 5))

plt.subplot(131)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(132)
plt.imshow(dog, cmap="gray")
plt.title("Difference of Gaussians")
plt.axis("off")

plt.subplot(133)
plt.imshow(thresholded, cmap="gray")
plt.title("Thresholded DoG")
plt.axis("off")

plt.tight_layout()
plt.show()
