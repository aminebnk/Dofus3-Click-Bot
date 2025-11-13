import random
import pyautogui
import heapq
from pynput.mouse import Controller as MouseController
from pynput import keyboard
from pynput.keyboard import Key, Controller as KeyboardController
import mss
import cv2
import re
import numpy as np
import pytesseract
import sqlite3
from collections import defaultdict
import ast
import Quartz.CoreGraphics as CG
from scipy.ndimage import maximum_filter
import time
import os
import torch
from torch import nn
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

SCREENSHOT_SIZE = 350  ## Width of the window used to check wether a resource is available
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MONSTER_DIR = os.path.join(BASE_DIR, "resources", "templates", "fight")
# Fighting texts
CHALLENGES = "CHOISIR LES\nCHALLENGES\n"
PRET = "PRET\n"
FIN = "FIN DE TOUR\n"
# Coordinates for the map coordinates screenshot
TOP_CORNER_MAP = (1004, 826)
SIZE_MAP = (133, 32)
# Coordinates for the fithging button
TOP_CORNER_FIGHT = (975, 835)
SIZE_FIGHT = (120, 28)
# Monster templates
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

FIGHT_POPUP_TEMPLATE = cv2.imread(os.path.join(BASE_DIR, "resources", "templates", "fight", "victory_popup.png"), cv2.IMREAD_COLOR)


path_full = os.path.join(BASE_DIR, "resources", "templates", "inventory", "Full.png")
path_almost_full = os.path.join(BASE_DIR, "resources", "templates", "inventory", "Almost_full.png")
path_not_full = os.path.join(BASE_DIR, "resources", "templates", "inventory", "Not_full.png")
template_not_full = cv2.imread(path_not_full, cv2.IMREAD_COLOR)
template_full = cv2.imread(path_full, cv2.IMREAD_COLOR)
template_almost_full = cv2.imread(path_almost_full, cv2.IMREAD_COLOR)

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

## General function definitions

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

def astar(start, end, walls = []):
    def heuristics(a, b):
        # Manhattan distance
        return (abs(b[0] - a[0]) + abs(b[1] - a[1]))

    open_set = []
    heapq.heappush(open_set, (0, start))

    g_score = {start: 0}
    came_from = {}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == end: 
            path = [current]
            while current != start:
                path.append(came_from[current])
                current = came_from[current]
            return path
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbour = (current[0]+dx, current[1]+dy)
            if (neighbour, current) in walls:
                continue
            if g_score[current] + 1 < g_score.get(neighbour, float("inf")):
                g_score[neighbour] = g_score[current] + 1
                came_from[neighbour] = current
                h_score = heuristics(neighbour, end)
                heapq.heappush(open_set, (g_score[neighbour]+h_score, neighbour))

    return None

def move(dest):
    curr_pos = get_map()
    while curr_pos == None:
        time.sleep(0.1)
        curr_pos = get_map()
    count = 10
    while (curr_pos != dest):
        if count > 9:
            fighting_status = screenshot_high_res(TOP_CORNER_FIGHT[0], TOP_CORNER_FIGHT[1], SIZE_FIGHT[0], SIZE_FIGHT[1])
            while check_fight(fighting_status):
                take_action(fighting_status)
                time.sleep(2)
                check_popup()
                fighting_status = screenshot_high_res(TOP_CORNER_FIGHT[0], TOP_CORNER_FIGHT[1], SIZE_FIGHT[0], SIZE_FIGHT[1])
            (move_x, move_y) = (dest[0] - curr_pos[0], dest[1] - curr_pos[1])
            if move_x == 1:
                go_direction('E')
            elif move_x == -1:
                go_direction('O')
            elif move_y == 1:
                go_direction('S')
            elif move_y == -1:
                go_direction('N')
            count = 0
        else:
            time.sleep(0.4)
            count += 1
        curr_pos = get_map()
        while not curr_pos:
            curr_pos = get_map()
            time.sleep(0.1)

def go_to(destination, walls = []):
    start = get_map()
    while start == None:
        start = get_map()
    end = destination
    path = astar(start, end, walls)
    l = len(path)
    for i in range(l-1):
        move(path[-i-2])

def get_map():
    frame = screenshot_high_res(3, 104, 124, 22) # Le module mss ne fonctionne pas bien avec Retina, donc ici on prend un screenshot en utilisant CoreGraphics pour améliorer la détection de texte
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Passage en HSV pour ne garder que les pixels peu saturés (gris)

    # Si un pixel est proche du gris (valeur de saturation S proche de 0), alors on en fait un pixel noir, sinon on en fait un pixel blanc. L'image est donc un texte noir sur blanc fromat BGR
    for ligne in img:
        for pixel in ligne:
            if pixel[1] <= 0 and pixel[2] != 0:
                pixel[0] = 0
                pixel[1] = 0
                pixel[2] = 0
            else:
                pixel[0] = 255
                pixel[1] = 255
                pixel[2] = 255

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    config = r'--psm 7 -c tessedit_char_whitelist=0123456789O,-'  # Liste de caractères que l'algorithme s'attend à détecter
    text = pytesseract.image_to_string(img, lang="eng", config=config)
    text = text.replace("O", "0")
    match = re.search(r"(-?\d+)\s*,\s*(-?\d+)", text)

    if match:
        x, y = int(match.group(1)), int(match.group(2))
        return (x, y)
    else:
        return None

def get_resources(map_pos, zone="Douze", types = "All"):
    resource_path = os.path.join(BASE_DIR, "resources", "resources.db")
    conn = sqlite3.connect(resource_path)
    c = conn.cursor()
    resources = defaultdict(list)
    if types == "All":
        c.execute("SELECT type, pos_x, pos_y FROM resources WHERE map_x = ? AND map_y = ? AND zone = ?", (map_pos[0], map_pos[1], zone))
        for row in c.fetchall():
            resources[row[0]].append((row[1], row[2]))
    else:
        for type in types:
            c.execute("SELECT type, pos_x, pos_y FROM resources WHERE map_x = ? AND map_y = ? AND zone = ? AND type = ?", (map_pos[0], map_pos[1], zone, type))
            for row in c.fetchall():
                resources[row[0]].append((row[1], row[2]))
    conn.close()
    return dict(resources)

def get_route(route_file_name):
    route_path = os.path.join(BASE_DIR, "resources", "routes", route_file_name)
    with open(route_path, "r") as f:
        content = f.read().strip()
    elements = content.split(";")
    list_maps = [ast.literal_eval(e) for e in elements]
    return list_maps

def get_walls(walls_file_name):
    pathname = os.path.join(BASE_DIR, "resources", "walls", walls_file_name)
    with open(pathname, "r") as f:
        content = f.read().strip()
    elements = content.split(";")
    list_walls = [ast.literal_eval(e) for e in elements]
    return list_walls

def get_name_pos(name_template_file):
    template_path = os.path.join(BASE_DIR, "resources", "templates", "names", name_template_file)
    # recupère la position du nom du personnage pour savoir s'il bouge
    with mss.mss() as sct:
        sct_img = sct.grab(sct.monitors[1])
        screen = np.array(sct_img)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
    
    # comparer la capture d'ecran au template
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    res = cv2.matchTemplate(screen, template, cv2.TM_SQDIFF_NORMED)
    local_min = (-res == maximum_filter(-res, size=15)) & (res <= 0.20) # Le match n'est pas parfait car mss réagit bizarrement avec retina
    y, x = np.where(local_min)
    points = list(zip(x, y))
    return points

def go_direction(direction):
    keyboard = KeyboardController()
    if direction=='E':
        with keyboard.pressed(Key.shift):
            keyboard.press('d')
            time.sleep(0.03)
            keyboard.release('d')

    elif direction=='O':
        with keyboard.pressed(Key.shift):
            keyboard.press('a')
            time.sleep(0.03)
            keyboard.release('a')

    elif direction=='S':
        with keyboard.pressed(Key.shift):
            keyboard.press('s')
            time.sleep(0.03)
            keyboard.release('s')

    elif direction=='N':
        with keyboard.pressed(Key.shift):
            keyboard.press('w')
            time.sleep(0.03)
            keyboard.release('w')

def check_at(pos, template_file_name):
    template_path = os.path.join(BASE_DIR, "resources", "templates", "resources", template_file_name)
    # déplacer la souris à pos
    mouse = MouseController()
    #pos = mouse.position
    mouse.position = (pos[0], pos[1])
    time.sleep(0.3)
    # prendre une capture d'ecran autour de pos
    with mss.mss() as sct:
        region = {"top": pos[1] - SCREENSHOT_SIZE//2, "left": pos[0] - SCREENSHOT_SIZE//2, "width": SCREENSHOT_SIZE, "height": SCREENSHOT_SIZE}
        sct_img = sct.grab(region)
        img = np.array(sct_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # comparer la capture d'ecran au template
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
    local_min = (-res == maximum_filter(-res, size=15)) & (res <= 0.2) # Le match n'est pas parfait car mss réagit bizarrement avec retina
    y, x = np.where(local_min)
    points = list(zip(x, y))

    return bool(points)

def inventory_full():

    keyboard = KeyboardController()
    keyboard.press('i')
    time.sleep(0.03)
    keyboard.release('i')
    time.sleep(0.7)
    frame = screenshot_high_res(775, 755, 18, 14)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    keyboard.press('i')
    time.sleep(0.03)
    keyboard.release('i')

    full = cv2.matchTemplate(frame, template_full, cv2.TM_SQDIFF_NORMED)
    almost_full = cv2.matchTemplate(frame, template_almost_full, cv2.TM_SQDIFF_NORMED)
    not_full =  cv2.matchTemplate(frame, template_not_full, cv2.TM_SQDIFF_NORMED)
    if full[0][0] <= 2e-1 or almost_full[0][0] <= 2e-1:
        return True
    elif not_full[0][0] <= 2e-1:
        return False
    else:
        keyboard.press('i')
        time.sleep(0.03)
        keyboard.release('i')
        return inventory_full()

    
def empty_inventory(walls):
    go_to((4, -18), walls)
    pyautogui.click(875, 300)
    time.sleep(5)
    pyautogui.click(750, 325)
    time.sleep(2)
    pyautogui.click(1025, 380)
    time.sleep(2)
    pyautogui.click(644, 335)
    time.sleep(2)
    pyautogui.click(931, 253)
    time.sleep(1)
    pyautogui.click(1061, 327)
    time.sleep(1)
    pyautogui.click(936, 196)
    time.sleep(1)
    pyautogui.click(516, 620)
    time.sleep(1)
    
## Fighting functions

def get_fighting_text(img):
    text_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for ligne in text_img:
        for pixel in ligne:
            if pixel[1] <= 0:
                pixel[0] = 0
                pixel[1] = 0
                pixel[2] = 0
            else:
                pixel[0] = 255
                pixel[1] = 255
                pixel[2] = 255
    text_img = cv2.cvtColor(text_img, cv2.COLOR_BGR2GRAY)
    text_img = cv2.GaussianBlur(text_img, (3,3), 0.6)

    text = pytesseract.image_to_string(text_img, lang="fra")
    return text

def get_grey_fighting_text(img):
    text_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for ligne in text_img:
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
    
    text_img = cv2.cvtColor(text_img, cv2.COLOR_BGR2GRAY)
    text_img = cv2.GaussianBlur(text_img, (3,3), 0.6)

    text = pytesseract.image_to_string(text_img, lang="fra")
    return text
    
def check_fight(img):
    text = get_fighting_text(img)
    grey_text = get_grey_fighting_text(img)
    if text == CHALLENGES or text == FIN or text == PRET or grey_text == FIN:
        return True
    else:
        return False
    
def check_popup():
    with mss.mss() as sct:
                region = {"top": 400, "left": 500, "width": 400, "height": 150}
                img = sct.grab(region)
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    match = cv2.matchTemplate(img, FIGHT_POPUP_TEMPLATE, cv2.TM_SQDIFF_NORMED)
    if np.min(match) < 1e-3:
        keyboard = KeyboardController()
        keyboard.press(Key.esc)
        time.sleep(0.03)
        keyboard.release(Key.esc)
        return True
    return False
        
    
def get_monster_pos(img):
    northwest_match = cv2.matchTemplate(img, NORTHWEST_BGR, cv2.TM_SQDIFF_NORMED, mask=NORTHWEST_ALPHA)
    northeast_match = cv2.matchTemplate(img, NORTHEAST_BGR, cv2.TM_SQDIFF_NORMED, mask=NORTHEAST_ALPHA)
    southeast_match = cv2.matchTemplate(img, SOUTHEAST_BGR, cv2.TM_SQDIFF_NORMED, mask=SOUTHEAST_ALPHA)
    southwest_match = cv2.matchTemplate(img, SOUTHWEST_BGR, cv2.TM_SQDIFF_NORMED, mask=SOUTHWEST_ALPHA)
    #print(min(np.min(northeast_match), np.min(northwest_match), np.min(southeast_match), np.min(southwest_match)))
    threshold = 50e-3
    northwest_match = (northwest_match < threshold) & (-northwest_match == maximum_filter(-northwest_match, size=15))
    southwest_match = (southwest_match < threshold) & (-southwest_match == maximum_filter(-southwest_match, size=15))
    southeast_match = (southeast_match < threshold) & (-southeast_match == maximum_filter(-southeast_match, size=15))
    northeast_match = (northeast_match < threshold) & (-northeast_match == maximum_filter(-northeast_match, size=15))
    match = northwest_match + northeast_match + southeast_match + southwest_match
    monster_indices = np.argwhere(match)
    # for indice in monster_indices:
    #     cv2.circle(img, (indice[1] + 11, indice[0] + 13), 10, (0, 255, 0), -1)

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
        pyautogui.moveTo(dest[1], dest[0])
        time.sleep(0.4)
        pyautogui.click(dest[1], dest[0])
        time.sleep(0.4)
    
    # cv2.imshow("check", check)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def take_action(fight_status):
    text = get_fighting_text(fight_status)
    if text == FIN:
        with mss.mss() as sct:
            img = sct.grab(sct.monitors[0])
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        monster_pos = get_monster_pos(img)
        close_in(img, monster_pos)
        attack(monster_pos)
        time.sleep(0.7)
        if not check_popup():
            attack(monster_pos)
            pyautogui.click(1070, 842)
            pyautogui.moveRel(random.randint(100, 200), random.randint(100, 200), 0.2)
    elif text == CHALLENGES or text == PRET:
        pyautogui.click(1070, 842)
        pyautogui.moveRel(random.randint(100, 200), random.randint(100, 200), 0.2)
    
def attack(monster_pos):
    if monster_pos[0] != -1:
        pyautogui.click(500, 845)
        time.sleep(0.7)
        pyautogui.click(monster_pos[1], monster_pos[0])
        pyautogui.moveRel(random.randint(100, 200), random.randint(100, 200), 0.2)
    
def run_script(walls_name, route_name, resource_names, character_name, zone_name):
    run = False
    terminate = False

    bank_count = 0

    def on_press(key):
        nonlocal run
        nonlocal terminate
        if key == keyboard.KeyCode.from_char('n'):
            if not run:
                run = True
            else:
                run = False
                terminate = True

    listener = keyboard.Listener(on_press = on_press)
    listener.start()


    walls = get_walls(walls_name)
    path = get_route(route_name)
    while not terminate:
        time.sleep(1)
        while run:
            for destination in path:
                bank_count += 1
                if bank_count == 5:
                    bank_count = 0
                    if inventory_full():
                        empty_inventory(walls)
                go_to(destination, walls)
                if terminate:
                    break
                resources = get_resources(destination, zone_name, resource_names)
                for resource, positions in resources.items():
                    for pos in positions:
                        if check_at(pos, resource + ".png"):
                            time.sleep(0.1)
                            pyautogui.click(pos[0], pos[1])
                            time.sleep(0.1)
                            pyautogui.moveRel(random.randint(100, 200), random.randint(100, 200), 0.2)
                            moving = True
                            name_pos = get_name_pos(character_name + ".png")
                            time.sleep(1.5)
                            while(moving):
                                new_name_pos = get_name_pos(character_name + ".png")
                                if new_name_pos == name_pos:
                                    moving = False
                                    time.sleep(1.5)
                                else: 
                                    name_pos = new_name_pos
                                    time.sleep(1.5)
                            fighting_status = screenshot_high_res(TOP_CORNER_FIGHT[0], TOP_CORNER_FIGHT[1], SIZE_FIGHT[0], SIZE_FIGHT[1])
                            while check_fight(fighting_status):
                                take_action(fighting_status)
                                time.sleep(1)
                                check_popup()
                                fighting_status = screenshot_high_res(TOP_CORNER_FIGHT[0], TOP_CORNER_FIGHT[1], SIZE_FIGHT[0], SIZE_FIGHT[1])
        
        
if __name__=="__main__":
    def on_press(key):
        global requested
        if key == keyboard.KeyCode.from_char('n'):
            requested = True

    listener = keyboard.Listener(on_press = on_press)
    listener.start()

    requested = False

    while True:
        if requested:
            run_script("Amakna.txt", "Forêt d'Astrub 2.txt", ["Frêne", "Châtaigner"], "Charles-Martelo", "Amakna")
            #print(check_popup())
            # with mss.mss() as sct:
            #     img = sct.grab(sct.monitors[0])
            #     img = np.array(img)
            #     img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            # get_monster_pos(img)
            requested = False
