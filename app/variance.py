import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.path.join(BASE_DIR, "resources", "test", "green_test_high_res.png")

# Load image (gray for simplicity)
img = cv2.imread(TEST_PATH).astype(np.float32)
print(img.shape)

# My way
# signed_std_deviation = img - cv2.blur(img, (15, 15))
# variance = np.sqrt(np.square(signed_std_deviation))
# variance = cv2.cvtColor(variance, cv2.COLOR_BGR2GRAY) * 4

# GPT way 

# mean = cv2.blur(img, (15, 15))
# sq_mean = cv2.blur(np.square(img), (15, 15))
# variance = np.sqrt((sq_mean - np.square(mean))) * 120
# variance = 255 - variance

# kernel = np.zeros((30, 30), np.uint8)
# cv2.fillConvexPoly(kernel, np.array([[14,7],[28,14],[14,21],[0,14]]), 1)
kernel = np.zeros((40, 40), np.uint8)
cv2.fillConvexPoly(kernel, np.array([[20,10],[40,20],[20,30],[0,20]]), 1)
kernel = kernel.astype(np.float32)
kernel /= kernel.sum()

mean = cv2.filter2D(img, -1, kernel)
sq_mean = cv2.filter2D(np.square(img), -1, kernel)
variance = sq_mean - np.square(mean)
_, flat_mask = cv2.threshold(variance, 5, 255, cv2.THRESH_BINARY_INV)
flat_mask = flat_mask.astype(np.uint8)


cv2.imshow("Flat regions", flat_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()