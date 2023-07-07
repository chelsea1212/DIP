import cv2
import numpy as np
from PIL import Image
 
# now we will be loading the image and converting it to grayscale
image = cv2.imread(r"/home/sahyadri/lab/flower.jpg")
img= Image.open("/home/sahyadri/lab/flower.jpg")
np_array_img = np.array(img)
 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# Compute the discrete Fourier Transform of the image
fourier = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
 
# Shift the zero-frequency component to the center of the spectrum
fourier_shift = np.fft.fftshift(fourier)
 
# calculate the magnitude of the Fourier Transform
magnitude = 20*np.log(cv2.magnitude(fourier_shift[:,:,0],fourier_shift[:,:,1]))
 
# Scale the magnitude for display
magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

np_array_mag = np.array(magnitude)
 
# Display the magnitude of the Fourier Transform
print("The matrix of original image is ")
print(np_array_img)
print()
print("The matrix of FT of image is ")
print(np_array_mag)
cv2.imshow('Original Image',image)
cv2.imshow('Fourier Transform', magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()
