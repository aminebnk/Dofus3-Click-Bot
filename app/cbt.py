import cv2
import mss
import pytesseract
import Quartz.CoreGraphics as CG
import numpy as np
from pynput import keyboard
from pynput.keyboard import Key, Controller as KeyboardController
import pyautogui
import time
from scipy.ndimage import maximum_filter
import random
import torch
from torch import nn
import heapq
import torch.nn.functional as F
import re
import os, sys

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # Use the CPU to run the Convolutional Neurol Network

def resource_path(relative_path):
    """Get absolute path to resource, wether we are running the app as a pythion file or as an app made by PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller’s temporary folder
        base_path = sys._MEIPASS
    else:
        # Normal execution
        base_path = os.path.abspath(".")
    return os.path.join(base_path, "resources", relative_path)


BLUE_BORDER = cv2.imread(resource_path("templates/fight/blue_border.png"), cv2.IMREAD_UNCHANGED)
BLUE_BORDER_BGR = BLUE_BORDER[:, :, :3]
BLUE_BORDER_ALPHA = BLUE_BORDER[:, :, 3]

RED_BORDER = cv2.imread(resource_path("templates/fight/red_border.png"), cv2.IMREAD_UNCHANGED)
RED_BORDER_BGR = RED_BORDER[:, :, :3]
RED_BORDER_ALPHA = RED_BORDER[:, :, 3]

FIGHT_POPUP_TEMPLATE = cv2.imread(resource_path("templates/fight/victory_popup.png"), cv2.IMREAD_COLOR)
CHALLENGES = "CHOISIR LES\nCHALLENGES" # When a fight is preparing, the fight button says "CHOISIR LES CHALLENGS": chose your challenges
PRET = "PRET" # Then the button says "PRET": ready
FIN = "FIN DE TOUR" # When it's your turn to play, it says "FIN DE TOUR": end your turn
CORNER_FIGHT_BTN = (975, 835)
SIZE_FIGHT_BTN = (120, 28)
END_TURN_POS = (1070, 842)
SPELL_POS = (500, 845)
SPELL_RANGE = 5

# The fight is confined to a scene in the middle of the screen, so we only make the template matching there to save some computation

FIGHT_SCN_TOP = 175
FIGHT_SCN_LEFT = 220
FIGHT_SCN_WIDTH = 980
FIGHT_SCN_HEIGHT = 555

## Loading CNN classification model

"""
The class TileClassifier defines the architecture for a CNN (Convolutional Neurol Network)
used to determine wether a small 19x19 picture represents a tile the character can move to.
The model weigths are then loaded into a TileClassifier object.
"""

class TileClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        #self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(16 * 19 * 19, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        return self.fc2(x)

model = TileClassifier().to(device) # We use the CPU to go faster
model.load_state_dict(torch.load(resource_path("model/tile_cnn.pth"), map_location="cpu")) # We load the weights from a previous training
model.eval()

## Define the vector class used to get the tiles surrouding the character

"""
Because of the 3D projected to 2D perspective, the grid of the fight system is tilted and the y dimension is compressed.
We define a custom vector class suited to that geometry for the purpose of computing the tiles a character can move to 
and the distance in 'tiles' that separates two entities for spell range.
The distance corresponds to the Manhattan distance, but for the non orthonormal base of the grid: u1 = -2 * ux + uy, u2 = 2 * ux + uy 
"""

class vector:
    def __init__(self, x=-36.4, y=18.3):
        self.x = x
        self.y = y

    def rotate_hourly(self):
        if self.x * self.y > 0:
            self.x = - self.x
        elif self.x * self.y < 0:
            self.y = - self.y

    def __add__(self, other):
        return vector(self.x + other.x, self.y + other.y)
    
    def dist(self, other):
        return max(2 * abs(self.y - other.y), abs(self.x - other.x)) / 36.4 # We divide by 36 to get the distance in tiles. Tiles are separted roughly 36 pixels across in the x direction and 18 in the y direction

def screenshot_high_res(x, y, w, h):
    """
    Takes a high-definition screenshot using Core Graphics and converts it to numpy (BGR).
    x, y = coordinates of the top-left corner
    w, h = width and height of the zone

    Note: This is slower than the mss library but has better definition. Used for small areas where text recognition is needed.
    """
    display_id = CG.CGMainDisplayID()

    # Screenshot rectangle
    rect = CG.CGRectMake(x, y, w, h)

    # Capture
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

    arr = arr[:, :width*4]  # Taking a screenshot adds balck borders for some reason. We only keep the useful part of the image.
    arr = arr.reshape((height, width, 4))  # And reshape it
    img = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)  # And convert it to BGR, which is what cv2 works with by default
    return img

def get_fighting_text(img):

    """
    When a fight is ongoing or in the preparation step, a green button appears at the bottom right of the screen.
    We use text recognition to get the state of that button and determine wether a fight is ongoing, and if at what
    step we are.
    See the function get_map in bot_script.py for details on the text recognition.
    """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for ligne in img:
        for pixel in ligne:
            if pixel[1] == 0:
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
    """
    When it's currently the monster turn, the fight button turns grey, so we need to modify
    the previous function slightly to detect the text.
    """
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
    return text.strip()

def check_fight(fight_status):
    """
    We simply read the fight button text to know wether a fight is ongoing or in preparation
    """
    text = get_fighting_text(fight_status)
    grey_text = get_grey_fighting_text(fight_status)
    if text == FIN or text == PRET or text == CHALLENGES or grey_text == FIN:
        return True
    else:
        return False
    
def check_popup():
    """
    When the fight is over, a victory pop up pops up. 
    This function identifies it with template matching and closes it by pressing the esc. key"""
    with mss.mss() as sct:
                region = {"top": 400, "left": 500, "width": 400, "height": 150}
                img = sct.grab(region)
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    match = cv2.matchTemplate(img, FIGHT_POPUP_TEMPLATE, cv2.TM_SQDIFF_NORMED)
    if np.min(match) < 25e-3:
        keyboard = KeyboardController()
        keyboard.press(Key.esc)
        time.sleep(0.03)
        keyboard.release(Key.esc)
        time.sleep(0.1)
        keyboard.press('p')
        time.sleep(0.03)
        keyboard.release('p')
        return True
    return False

def get_monster_pos(img):
    """
    This function takes in an image and returns the position of the monsters identified by template matching.
    Returns [-1, -1] if no monsters were found.
    Four templates are necessary since the monster can be facing 4 different ways.
    There are two modes: 
    -first-found=True returns the first monster position found. This speeds up the function on average and is
    when we know there is one and only one monster to be found
    -first-found=False returns a list of the positions of all the matches.
    We only use the first-found=True version of this function for this version of the BOT.
    """

    border_match = np.nan_to_num(cv2.matchTemplate(img, RED_BORDER_BGR, cv2.TM_SQDIFF, mask=RED_BORDER_ALPHA), nan=np.inf)
    border_match = (border_match == np.min(border_match))

    pos = np.argwhere(border_match)
    if pos.any():
        return pos[0] + [5, 24]
    else:
        return [-1, -1]

    
def get_character_pos(img):

    """
    Exactly the same as the monster version. Note: there are 18 classes in the game. I produced templates for only 2 of them (Zobal and Sacrieur)
    to reduce my screentime.
    Sacrieur is the best for these kind of BOTs anyways.
    """

    border_match = np.nan_to_num(cv2.matchTemplate(img, BLUE_BORDER_BGR, cv2.TM_SQDIFF, mask=BLUE_BORDER_ALPHA), nan=np.inf)
    border_match = (border_match == np.min(border_match))

    pos = np.argwhere(border_match)
    if pos.any():
        return pos[0] + [9, 20]
    else:
        return [-1, -1]
    
def get_character_mp():

    """
    Captures the part of the screen corresponding to the character's movements points and reads it out using text recognition.
    See the function get_map in bot_script.py for details on the text recognition.
    """

    mp_img = screenshot_high_res(441, 863, 21, 18)
    img = cv2.cvtColor(mp_img, cv2.COLOR_BGR2HSV) # Passage en HSV pour ne garder que les pixels peu saturés (gris)
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
    config = r'--psm 7 -c tessedit_char_whitelist=0123456789O'  # Liste de caractères que l'algorithme s'attend à détecter
    text = pytesseract.image_to_string(img, lang="eng", config=config)
    text = text.replace("O", "0")
    match = re.search(r"\d+", text)

    if match:
        return int(match[0])
    else:
        return 3

def expand_character(character_pos, monster_pos, mp = 4):

    """
    Using the character position and his movement points, we determine all the positions the character can move to.
    Then we put all those position tuples in a pile ordered by incresing distance to the monster, so the closest tile comes out first.
    The tiles that are further away form the monster than the current position of the character are ignored.
    At this stage, it is still uknown wether those tiles are indeed available to the character, there could be walls in the way.

    To determine the tiles we can move to, we concatenate i vectors for all i <= mp, all those pointing southwest.
    All those vectors put together point to a possible tile. 
    Then, starting from the last vector, we rotate all of them hourly, one by one. This scans across possible tiles. 
    If we do this four times we get all possible tiles.
    """

    tiles = []
    character_vector = vector(character_pos[1], character_pos[0])
    monster_vector = vector(monster_pos[1], monster_pos[0])
    for l in range(1, mp+1): # For all l <= mp
        position_vectors = []
        for i in range(l):
            position_vectors.append(vector()) # We concatenate l vectors
        for i in range(4):
            for vector1 in position_vectors: # Get the position they point to
                new_tile = character_vector
                for vector2 in position_vectors:
                    new_tile += vector2
                tiles.append(new_tile)
                vector1.rotate_hourly() # Then rotate them hourly one by one
    
    min_dist = round(character_vector.dist(monster_vector)) - 0.2
    tiles_hq = []
    counter = 0
    for tile in tiles:
        dist = round(tile.dist(monster_vector))
        if dist < min_dist:
            heapq.heappush(tiles_hq, (dist, counter, tile))
            counter += 1
    return tiles_hq

def predict(tile):
    """
    We use the model we trained previously to predict wether our character can move to a tile. 
    The tiles our character is able to move to are highlighted in green by the game.
    THe model is a minimalist CNN trained for classification with two categories: tile or non tile.
    I made the training data from 640 tile screenshots.
    """
    tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
    tile = torch.tensor(tile, dtype=torch.float32).permute(2,0,1) / 255.0
    tile = tile.unsqueeze(0).to(device)
    logit = model(tile)
    return (logit > 0)

def close_in(img, monster_pos, character_pos, fight_corner = [FIGHT_SCN_TOP, FIGHT_SCN_LEFT], mp = 3, square_width = 9):
    """
    We expand the character position using expand_character. 
    Then we pop the possible tiles one by one. 
    Predict wether the current tile is available and if so, click on it.
    We return the new distance from our character to the monster.

    """
    tiles_hq = expand_character(character_pos, monster_pos, mp)
    while tiles_hq:
        _ , _ , tile = heapq.heappop(tiles_hq)
        tile_img = img[round(tile.y)-square_width:round(tile.y) + square_width + 1, round(tile.x)-square_width:round(tile.x) + square_width + 1, :]
        if predict(tile_img):
            pyautogui.moveTo(tile.x + fight_corner[1], tile.y + fight_corner[0])
            time.sleep(0.2)
            pyautogui.click(tile.x + fight_corner[1], tile.y + fight_corner[0])
            return round(tile.dist(vector(monster_pos[1], monster_pos[0])))
    character_vector = vector(character_pos[1], character_pos[0])
    monster_vector = vector(monster_pos[1], monster_pos[0])
    return round(character_vector.dist(monster_vector))

def attack(monster_pos):

    """
    We attack the monster by clicking our spell, then clicking the monster. 
    This game is quite simple.
    """

    if monster_pos[0] != -1:
        pyautogui.moveTo(SPELL_POS[0], SPELL_POS[1], duration=0.1)
        pyautogui.click(SPELL_POS[0], SPELL_POS[1])
        time.sleep(0.1)
        pyautogui.moveTo(monster_pos[1], monster_pos[0], duration=0.1)
        pyautogui.click(monster_pos[1], monster_pos[0])
        pyautogui.moveRel(random.randint(50, 100), random.randint(50, 100), 0.1)

def take_action(fight_status, fight_corner = [FIGHT_SCN_TOP, FIGHT_SCN_LEFT], fight_size = [FIGHT_SCN_WIDTH, FIGHT_SCN_HEIGHT]):

    """"
    This function puts everything together. 
    fight_status is the picture of the fighting button. 
    fight_corner and fight_size define the bounds of the screenshot (the fight doesn't use the whole screen).
    If we are in preparation mode, we simply click the fight button. 
    Else if it's out turn, we take a screenshot of the fight scene. 
    Then get the monster's positon and character's position
    Using those positions, we close in on the monster, and attack it if it's within range. Them pass our turn.
    Two attacks can be made, but it's possible the first attack kills the monster. To avoid unwanted clicks, 
    we check wether the fight is over before attacking again.
    """
    text = get_fighting_text(fight_status)
    if text == FIN: # Our turn to play
        # Get the screen shot
        with mss.mss() as sct:
            region = {
                "top": fight_corner[0],
                "left": fight_corner[1],
                "width": fight_size[0],
                "height": fight_size[1]
            }
            img = sct.grab(region)
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        # Get the positions
        monster_pos = get_monster_pos(img)
        character_pos = get_character_pos(img)
        # If the monster and character were found, close in on the monster
        if character_pos[0] != -1 and monster_pos[0] != -1:
            mp = get_character_mp()
            distance = close_in(img, monster_pos, character_pos, mp=mp)
            if distance <= SPELL_RANGE: # attack the monster once if in range
                attack(monster_pos + [fight_corner[0], fight_corner[1]])
                time.sleep(0.9)
                if not check_popup(): # if he's not dead, attack again then pass the turn
                    attack(monster_pos + [fight_corner[0], fight_corner[1]])
                    pyautogui.click(END_TURN_POS[0], END_TURN_POS[1])
                    pyautogui.moveRel(random.randint(100, 200), random.randint(100, 200), 0.1)
            else: # If not in range of the monster after moving, pass the turn
                pyautogui.click(END_TURN_POS[0], END_TURN_POS[1])
                pyautogui.moveRel(random.randint(100, 200), random.randint(100, 200), 0.1)
        else: # Warning in case the template matching didn't work somehow
            pyautogui.click(END_TURN_POS[0], END_TURN_POS[1])
            pyautogui.moveRel(random.randint(100, 200), random.randint(100, 200), 0.1)
    elif text == CHALLENGES or text == PRET: # If it's the preparation phase, start the fight
        pyautogui.click(END_TURN_POS[0], END_TURN_POS[1])
        pyautogui.moveRel(random.randint(100, 200), random.randint(100, 200), 0.1)

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
            # fight_status = screenshot_high_res(TOP_CORNER[0], TOP_CORNER[1], SIZE[0], SIZE[1])
            # take_action(fight_status)
            # fight_corner = [FIGHT_SCN_TOP, FIGHT_SCN_LEFT]
            # fight_size = [FIGHT_SCN_WIDTH, FIGHT_SCN_HEIGHT]
            # with mss.mss() as sct:
            #     region = {
            #         "top": fight_corner[0],
            #         "left": fight_corner[1],
            #         "width": fight_size[0],
            #         "height": fight_size[1]
            #     }
            #     img = sct.grab(region)
            #     img = np.array(img)
            #     img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            # monster_pos = get_monster_pos(img)
            # character_pos = get_character_pos(img)
            # cv2.circle(img, (monster_pos[1], monster_pos[0]), 10, (0, 0, 255), -1)
            # cv2.circle(img, (character_pos[1], character_pos[0]), 10, (0, 255, 0), -1)    
            # cv2.imshow("Matches", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            fight_status = screenshot_high_res(CORNER_FIGHT_BTN[0], CORNER_FIGHT_BTN[1], SIZE_FIGHT_BTN[0], SIZE_FIGHT_BTN[1])
            while check_fight(fight_status) and requested:
                take_action(fight_status)
                time.sleep(1)
                fight_status = screenshot_high_res(CORNER_FIGHT_BTN[0], CORNER_FIGHT_BTN[1], SIZE_FIGHT_BTN[0], SIZE_FIGHT_BTN[1])
            check_popup()

            requested = False
        
"""
ClickBot/
    app/
        main.py
        clickbot.py
        bot_script.py
        cbt.py
        routemaker.py
        screenselector.py
        resources/
            database/
                resources.db
            routes/
                some files
            templates/
                fight/
                    some files
                inventory/
                    some files
                names/
                    some files
                resources/
                    some files
            walls/
                some files
            model/
                some files
"""