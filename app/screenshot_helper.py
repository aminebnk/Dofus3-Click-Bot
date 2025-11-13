from pynput import keyboard
import mss
import numpy as np
import cv2

def on_press(key):
    global requested
    if key == keyboard.KeyCode.from_char('n'):
        requested = True

listener = keyboard.Listener(on_press = on_press)
listener.start()

requested = False

while True:
    if requested:
        with mss.mss() as sct:
            sct_img = sct.grab(sct.monitors[1])
            img = np.array(sct_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        cv2.imwrite("screenshot.png", img)
        requested = False