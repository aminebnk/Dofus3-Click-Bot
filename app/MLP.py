import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import os
from sklearn.neural_network import MLPClassifier
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GREEN_TEST_1_PATH = os.path.join(BASE_DIR, "resources", "test", "green_test.png")
GREEN_TEST_2_PATH = os.path.join(BASE_DIR, "resources", "test", "green_test_2.png")
MODEL_PATH = os.path.join(BASE_DIR, "resources", "tile_mlp.pkl")

img_1 = cv2.imread(GREEN_TEST_1_PATH)
img_1_hsv = cv2.cvtColor(img_1, cv2.COLOR_BGR2HSV)

img_2 = cv2.imread(GREEN_TEST_2_PATH)
img_2_hsv = cv2.cvtColor(img_2, cv2.COLOR_BGR2HSV)

mask_1 = np.empty(img_1.shape[:2])
mask_2 = np.empty(img_2.shape[:2])

for i, ligne in enumerate(img_1_hsv):
    for j, pixel in enumerate(ligne):
      if 370 < j < 665 and 460 < i < 620 and 40 < pixel[0] < 53:
            mask_1[i][j] = 1
      else:
          mask_1[i][j] = 0

for i, ligne in enumerate(img_2_hsv):
    for j, pixel in enumerate(ligne):
      if 624 < j < 925 and 375 < i < 533 and 40 < pixel[0] < 53:
            mask_2[i][j] = 1
      else:
          mask_2[i][j] = 0

# cv2.imshow("label 1", mask_1)
# cv2.imshow("label 2", mask_2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

labels = np.concatenate((mask_1.reshape(-1, 1), mask_2.reshape(-1, 1)), axis=0)
pixels = np.concatenate((img_1_hsv.reshape(-1, 3), img_2_hsv.reshape(-1, 3)), axis=0)

mlp = MLPClassifier(
    hidden_layer_sizes=(8, 4),  # two small hidden layers
    activation='relu',
    max_iter=300,
    random_state=42
)
mlp.fit(pixels, labels)

pixels_1 = img_1_hsv.reshape(-1, 3)
pixels_2 = img_2_hsv.reshape(-1, 3)
pred_1 = mlp.predict(pixels_1)
pred_2 = mlp.predict(pixels_2)
pred_1 = pred_1.reshape(img_1_hsv.shape[:2])
pred_2 = pred_2.reshape(img_2_hsv.shape[:2])

cv2.imshow("prediction 1", pred_1)
cv2.imshow("prediction 2", pred_2)
cv2.waitKey(0)
cv2.destroyAllWindows()

joblib.dump(mlp, MODEL_PATH)

      

