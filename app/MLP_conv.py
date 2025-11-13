import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import os
from sklearn.neural_network import MLPClassifier
import joblib
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GREEN_TEST_PATH = os.path.join(BASE_DIR, "resources", "test", "green_test_2.png")
MODEL_PATH = os.path.join(BASE_DIR, "resources", "tile_mlp.pkl")

img = cv2.imread(GREEN_TEST_PATH)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
pixels = img_hsv.reshape(-1, 3)

mlp = joblib.load(MODEL_PATH)

tic = time.time()
pred = mlp.predict(pixels)
tac = time.time()
print("prediction: " , (tac - tic)*10e3, "ms")
pred = pred.reshape(img_hsv.shape[:2]).astype(np.uint8) * 255

tic = time.time()
avg = cv2.blur(pred, (7, 7))
tac = time.time()
print("averaging: " , (tac - tic)*10e3, "ms")
_ , res = cv2.threshold(avg, 254, 255, cv2.THRESH_BINARY)

cv2.imshow("prediction", res)
cv2.waitKey(0)
cv2.destroyAllWindows()