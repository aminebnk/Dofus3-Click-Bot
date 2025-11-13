import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOADING_FOLDER = os.path.join(BASE_DIR, "resources", "training", "full_size", "raw")
DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")
DESTINATION_FOLDER = os.path.join(DESKTOP, "processed")
WRITE_NAME = "processed_"
LOAD_NAME = "raw_"

def process(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for ligne in hsv:
        for pixel in ligne:
            if 40 < pixel[0] < 51 and pixel[1] > 110 and pixel[2] > 100:
                pixel[:3] = 255
            else:
                pixel[:3] = 0
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return hsv

for i in range(19, 31):
    raw = cv2.imread(os.path.join(LOADING_FOLDER, LOAD_NAME + str(i) + ".png"))
    processed = process(raw)
    cv2.imwrite(os.path.join(DESTINATION_FOLDER, WRITE_NAME + str(i) + ".png"), processed)


# for filename in os.listdir(LOADING_FOLDER):
#     count += 1
#     path = os.path.join(LOADING_FOLDER, filename)
#     img = cv2.imread(path)
#     img = process(img)
#     saving_name = FILE_NAME + str(count) + ".png"
#     saving_path = os.path.join(DESTINATION_FOLDER, saving_name)
#     cv2.imwrite(saving_path, img)


