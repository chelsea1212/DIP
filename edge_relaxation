import numpy as np
import cv2
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt

# Define the edge relaxation kernel
kernel = np.array([[0, 1, 0],
                   [1, 4, 1],
                   [0, 1, 0]])

# Define the edge relaxation threshold
threshold = 50


def edge_relaxation(image):
    # Convert image to grayscale if needed
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert image to float for calculations
    image = image.astype(np.float32)

    # Normalize image to range [0, 1]
    image /= 255.0

    # Perform edge relaxation iterations
    while True:
        # Apply the edge relaxation kernel
        smoothed_image = convolve(image, kernel, mode='constant')

        # Find the difference between the original and smoothed images
        diff = np.abs(image - smoothed_image)

        # Update the image with pixels that have a difference above the threshold
        image = np.where(diff > threshold, smoothed_image, image)

        # Check convergence
        if np.max(diff) <= threshold:
            break

    # Convert image back to the range [0, 255]
    image *= 255.0
    image = image.astype(np.uint8)

    return image


# Read the image
image = cv2.imread('image.jpg')

# Apply edge relaxation
relaxed_image = edge_relaxation(image)

# Convert images to RGB for compatibility with matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
relaxed_image_rgb = cv2.cvtColor(relaxed_image, cv2.COLOR_BGR2RGB)

# Display the original and relaxed images side by side using matplotlib
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image_rgb)
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow(relaxed_image_rgb)
axes[1].set_title('Relaxed Image')
axes[1].axis('off')
plt.show()
