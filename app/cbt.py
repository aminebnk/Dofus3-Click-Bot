import os
import cv2
import mss
import pytesseract
import Quartz.CoreGraphics as CG
import numpy as np
from pynput import keyboard
from pynput.keyboard import Controller as KeyboardController
import pyautogui
import time
from scipy.ndimage import maximum_filter
import random
import torch
from torch import nn

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MONSTER_DIR = os.path.join(BASE_DIR, "resources", "templates", "fight")

MONSTER_NORTHEAST = cv2.imread(MONSTER_DIR + "/monster_northeast.png", cv2.IMREAD_UNCHANGED)
MONSTER_NORTHWEST = cv2.imread(MONSTER_DIR + "/monster_northwest.png", cv2.IMREAD_UNCHANGED)
MONSTER_SOUTHEAST = cv2.imread(MONSTER_DIR + "/monster_southeast.png", cv2.IMREAD_UNCHANGED)
MONSTER_SOUTHWEST = cv2.imread(MONSTER_DIR + "/monster_southwest.png", cv2.IMREAD_UNCHANGED)
NORTHEAST_BGR = MONSTER_NORTHEAST[:, :, :3]
NORTHEAST_ALPHA = MONSTER_NORTHEAST[:, :, 3]
NORTHWEST_BGR = MONSTER_NORTHWEST[:, :, :3]
NORTHWEST_ALPHA = MONSTER_NORTHWEST[:, :, 3]
SOUTHEAST_BGR = MONSTER_SOUTHEAST[:, :, :3]
SOUTHEAST_ALPHA = MONSTER_SOUTHEAST[:, :, 3]
SOUTHWEST_BGR = MONSTER_SOUTHWEST[:, :, :3]
SOUTHWEST_ALPHA = MONSTER_SOUTHWEST[:, :, 3]
CHALLENGES = "CHOISIR LES\nCHALLENGES"
PRET = "PRET"
FIN = "FIN DE TOUR"
TOP_CORNER = (975, 835)
SIZE = (120, 28)

## U-Nets model definition

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(3, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(32, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv2 = DoubleConv(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv1 = DoubleConv(32, 16)

        self.final = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        c1 = self.down1(x)
        p1 = self.pool1(c1)
        c2 = self.down2(p1)
        p2 = self.pool2(c2)

        b = self.bottleneck(p2)

        u2 = self.up2(b)
        u2 = torch.cat([u2, c2], dim=1)
        c3 = self.conv2(u2)

        u1 = self.up1(c3)
        u1 = torch.cat([u1, c1], dim=1)
        c4 = self.conv1(u1)

        return torch.sigmoid(self.final(c4))
    
## Load model

model = UNet().to(device)
model.load_state_dict(torch.load("tile_segmentation_unet.pth", map_location=device))
model.eval()

def screenshot_high_res(x, y, w, h):
    """
    Capture une zone de l'écran en numpy (BGR).
    x, y = coordonnées du coin haut-gauche
    w, h = largeur et hauteur de la zone
    """
    display_id = CG.CGMainDisplayID()

    # Définir la zone à capturer
    rect = CG.CGRectMake(x, y, w, h)

    # Capture la zone
    image = CG.CGDisplayCreateImageForRect(display_id, rect)

    width = CG.CGImageGetWidth(image)
    height = CG.CGImageGetHeight(image)
    bytes_per_row = CG.CGImageGetBytesPerRow(image)

    provider = CG.CGImageGetDataProvider(image)
    data = CG.CGDataProviderCopyData(provider)

    buf = np.frombuffer(data, dtype=np.uint8)
    arr = np.ndarray(
        shape=(height, bytes_per_row),
        dtype=np.uint8,
        buffer=buf
    )

    arr = arr[:, :width*4]  # On garde seulement la partie utile
    arr = arr.reshape((height, width, 4))  # RGBA
    img = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)  # Conversion en BGR
    return img

def get_fighting_text(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for ligne in img:
        for pixel in ligne:
            if pixel[1] <= 0:
                pixel[0] = 0
                pixel[1] = 0
                pixel[2] = 0
            else:
                pixel[0] = 255
                pixel[1] = 255
                pixel[2] = 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3,3), 0.6)

    text = pytesseract.image_to_string(img, lang="fra")
    return text.strip()

def get_grey_fighting_text(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for ligne in img:
        for pixel in ligne:
            if pixel[1] <= 8:
                if pixel[2] >= 190:
                    pixel[0] = 0
                    pixel[1] = 0
                    pixel[2] = 0
                else:
                    pixel[0] = 255
                    pixel[1] = 255
                    pixel[2] = 255
            else:
                pixel[0] = 255
                pixel[1] = 255
                pixel[2] = 255
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3,3), 0.6)

    text = pytesseract.image_to_string(img, lang="fra")
    return text

def check_fight(fight_status):
    text = get_fighting_text(fight_status)
    grey_text = get_grey_fighting_text(fight_status)
    if text == FIN or text == PRET or text == CHALLENGES or grey_text == FIN:
        return True
    else:
        return False

def get_monster_pos(img):
    northwest_match = cv2.matchTemplate(img, NORTHWEST_BGR, cv2.TM_SQDIFF_NORMED, mask=NORTHWEST_ALPHA)
    northeast_match = cv2.matchTemplate(img, NORTHEAST_BGR, cv2.TM_SQDIFF_NORMED, mask=NORTHEAST_ALPHA)
    southeast_match = cv2.matchTemplate(img, SOUTHEAST_BGR, cv2.TM_SQDIFF_NORMED, mask=SOUTHEAST_ALPHA)
    southwest_match = cv2.matchTemplate(img, SOUTHWEST_BGR, cv2.TM_SQDIFF_NORMED, mask=SOUTHWEST_ALPHA)
    print(min(np.min(northeast_match), np.min(northwest_match), np.min(southeast_match), np.min(southwest_match)))
    threshold = 55e-3
    northwest_match = (northwest_match < threshold) & (-northwest_match == maximum_filter(-northwest_match, size=15))
    southwest_match = (southwest_match < threshold) & (-southwest_match == maximum_filter(-southwest_match, size=15))
    southeast_match = (southeast_match < threshold) & (-southeast_match == maximum_filter(-southeast_match, size=15))
    northeast_match = (northeast_match < threshold) & (-northeast_match == maximum_filter(-northeast_match, size=15))
    match = northwest_match + northeast_match + southeast_match + southwest_match
    monster_indices = np.argwhere(match)
    if monster_indices.any():
        return monster_indices[0] + [13, 11]
    else:
        return [-1,-1]
    
def close_in(img, monster_pos):
    # with mss.mss() as sct:
    #     img = sct.grab(sct.monitors[0])
    # img = np.array(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    #img_down = cv2.resize(img, (704, 448), interpolation=cv2.INTER_LINEAR)[:, :, ::-1] / 255
    img_tensor = torch.tensor(img[:, :, ::-1] / 255, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(img_tensor)[0, 0].cpu().numpy()
    mask = (pred > 0.5).astype(np.uint8) * 255
    mask = cv2.blur(mask, (9, 9))
    _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
    check = img.copy()
    check[mask == 255] = [255, 0, 0]
    #cv2.circle(check, (monster_pos[1], monster_pos[0]), 1, (0, 0, 255), -1)

    min_dist = float('inf')
    dest = None
    for i, ligne in enumerate(mask):
        for j, pixel in enumerate(ligne):
            if pixel > 0.5:
                if abs((monster_pos[0] - i)) + abs(monster_pos[1] - j) / 2 < min_dist:
                    min_dist = abs(monster_pos[0] - i) + abs(monster_pos[1] - j) / 2
                    dest = [i, j]
    if dest:
        #cv2.circle(check, (dest[1], dest[0]), 1, (0, 255, 0), -1)
        pyautogui.click(dest[1], dest[0])
    
    # cv2.imshow("check", check)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def attack(monster_pos):
    if monster_pos[0] != -1:
        time.sleep(0.6)
        pyautogui.click(500, 845)
        time.sleep(0.7)
        pyautogui.click(monster_pos[1], monster_pos[0])
        pyautogui.moveRel(random.randint(100, 200), random.randint(100, 200), 0.2)

def take_action(fight_status):
    text = get_fighting_text(fight_status)
    grey_text = get_grey_fighting_text(fight_status)
    if text == FIN and grey_text != FIN:
        with mss.mss() as sct:
            img = sct.grab(sct.monitors[0])
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        monster_pos = get_monster_pos(img)
        print(monster_pos)
        close_in(img, monster_pos)
        attack(monster_pos)
        time.sleep(0.7)
        attack(monster_pos)
        pyautogui.click(1070, 842)
        pyautogui.moveRel(random.randint(100, 200), random.randint(100, 200), 0.2)
    elif text == CHALLENGES or text == PRET:
        print("ready to fight")
        pyautogui.click(1070, 842)
        pyautogui.moveRel(random.randint(100, 200), random.randint(100, 200), 0.2)

if __name__=="__main__":

    def on_press(key):
        global requested
        if key == keyboard.KeyCode.from_char('n'):
            if requested:
                requested = False
            else:
                requested = True

    listener = keyboard.Listener(on_press = on_press)
    listener.start()

    requested = False

    while True:
        if requested:
            fight_status = screenshot_high_res(TOP_CORNER[0], TOP_CORNER[1], SIZE[0], SIZE[1])
            take_action(fight_status)
            # fight_status = screenshot_high_res(TOP_CORNER[0], TOP_CORNER[1], SIZE[0], SIZE[1])
            # while check_fight(fight_status) and requested:
            #     take_action(fight_status)
            #     time.sleep(2)
            #     fight_status = screenshot_high_res(TOP_CORNER[0], TOP_CORNER[1], SIZE[0], SIZE[1])
            requested = False
        