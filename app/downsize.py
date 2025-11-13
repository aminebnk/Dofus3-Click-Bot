import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_LOADING_FOLDER = os.path.join(BASE_DIR, "resources", "training", "full_size", "raw")
PROCESSED_LOADING_FOLDER = os.path.join(BASE_DIR, "resources", "training", "full_size", "processed")
DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")
DOWNSIZE_RAW_FOLDER = os.path.join(BASE_DIR, "resources", "training", "down_size", "raw")
DOWNSIZE_PROCESSED_FOLDER = os.path.join(BASE_DIR, "resources", "training", "down_size", "processed")
FILE_NAME = "processed_screenshot_"


for filename in os.listdir(RAW_LOADING_FOLDER):
    path = os.path.join(RAW_LOADING_FOLDER, filename)
    img = cv2.imread(path)
    ## process image
    img = cv2.resize(img, (704, 448), interpolation=cv2.INTER_LINEAR)
    saving_path = os.path.join(DOWNSIZE_RAW_FOLDER, filename)
    cv2.imwrite(saving_path, img)

for filename in os.listdir(PROCESSED_LOADING_FOLDER):
    path = os.path.join(PROCESSED_LOADING_FOLDER, filename)
    img = cv2.imread(path)
    ## process image
    img = cv2.resize(img, (704, 448), interpolation=cv2.INTER_NEAREST)
    saving_path = os.path.join(DOWNSIZE_PROCESSED_FOLDER, filename)
    cv2.imwrite(saving_path, img)