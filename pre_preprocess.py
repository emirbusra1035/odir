import cv2
import numpy as np
import logging
import os
from image_crop import Crop 

class Crop:
def __init__(self, source_folder, destination_folder, file_name):
self.logger = logging.getLogger('odir')
self.source_folder = source_folder
self.destination_folder = destination_folder
self.file_name = file_name
def remove_black_pixels(self):
file = os.path.join(self.source_folder, self.file_name)
image = cv2.imread(file)

# Mask of coloured pixels
mask = image > 0
# Coordinates of coloured pixels
coordinates = np.argwhere(mask)
# Binding box of non-black pixels
x0, y0, s0 = coordinates.min(axis=0)
# slices are exclusive at the top
x1, y1, s1 = coordinates.max(axis=0) + 1
# Get the contents of the bounding box
cropped = image[x0:x1, y0:y1]
# overwrite the same file
file_cropped = os.path.join(self.destination_folder, self.file_name)
cv2.imwrite(file_cropped, cropped)