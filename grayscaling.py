import cv2 # Import relevant libraries
import numpy as np

img = cv2.imread('img.jpg', 0) # Read in image

height = img.shape[0] # Get the dimensions
width = img.shape[1]

# Define mask
mask = 255*np.ones(img.shape, dtype='uint8')

# Draw circle at x = 100, y = 70 of radius 25 and fill this in with 0
x=width//2
y=height//2
radius=min(height,width)//2
cv2.circle(mask, (x, y), radius, 0, -1)