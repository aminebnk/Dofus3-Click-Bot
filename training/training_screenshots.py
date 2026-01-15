from pynput import keyboard
import mss
import numpy as np
import cv2
import os

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(TRAINING_DIR, os.pardir))
DESTINATION_FOLDER = os.path.join(BASE_DIR, "resources", "training","map")
FILE_NAME = "map_"

count = 0

def on_press(key):
    global requested
    if key == keyboard.KeyCode.from_char('n'):
        requested = True

listener = keyboard.Listener(on_press = on_press)
listener.start()

requested = False

while True:
    if requested:
        count += 1
        with mss.mss() as sct:
            sct_img = sct.grab(sct.monitors[1])
            img = np.array(sct_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        file = FILE_NAME + str(count) + ".png"
        file = os.path.join(DESTINATION_FOLDER, file)
        cv2.imwrite(file, img)
        requested = False