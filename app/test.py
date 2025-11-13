import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.ndimage import maximum_filter

# Load image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MONSTER_DIR = os.path.join(BASE_DIR, "resources", "templates", "fight")
a = np.ones(1)
print(a)
print(a.any())

