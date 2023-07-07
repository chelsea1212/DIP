import numpy as np
import cv2
import heapq
import matplotlib.pyplot as plt

# Define 8-connectivity neighbors
neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]

def heuristic(node, goal):
    # Calculate Euclidean distance as the heuristic
    return np.linalg.norm(np.array(node) - np.array(goal))

def edge_linking(image):
    # Convert image to binary
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Get image dimensions
    height, width = binary_image.shape[:2]

    # Create a visited matrix to keep track of visited pixels
    visited = np.zeros((height, width), dtype=bool)

    # Create a list to store object boundaries
    object_boundaries = []

    # Iterate over each pixel in the image
    for y in range(height):
        for x in range(width):
            # Check if the pixel is an unvisited edge pixel
            if binary_image[y, x] == 255 and not visited[y, x]:
                # Start a new object boundary
                object_boundary = []

                # Perform A* algorithm to find connected edge pixels
                open_list = []
                heapq.heappush(open_list, (0, (y, x)))
                while open_list:
                    _, (cy, cx) = heapq.heappop(open_list)
                    visited[cy, cx] = True
                    object_boundary.append((cx, cy))

                    # Check neighboring pixels
                    for dy, dx in neighbors:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width and binary_image[ny, nx] == 255 and not visited[ny, nx]:
                            g = 1  # Cost of moving to a neighboring pixel
                            h = heuristic((ny, nx), (y, x))
                            f = g + h
                            heapq.heappush(open_list, (f, (ny, nx)))
                            visited[ny, nx] = True

                # Add the object boundary to the list
                object_boundaries.append(object_boundary)

    return object_boundaries

# Read the image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Perform edge linking with A* algorithm
object_boundaries = edge_linking(image)

# Create a color image for visualization
image_with_boundaries = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Draw the object boundaries on the image
for object_boundary in object_boundaries:
    for x, y in object_boundary:
        image_with_boundaries[y, x] = (0, 0, 255)  # Draw boundary pixels in red

# Convert images to RGB for compatibility with matplotlib
image_rgb = cv2.cvtColor(image_with_boundaries, cv2.COLOR_BGR2RGB)

# Display the original image and image with object boundaries side by side using matplotlib
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow(image_rgb)
axes[1].set_title('Image with Object Boundaries (A*)')
axes[1].axis('off')
plt.show()
