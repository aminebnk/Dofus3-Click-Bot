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
import os, sys
import torch
from cbt import check_fight, take_action, check_popup, resource_path, CORNER_FIGHT_BTN, SIZE_FIGHT_BTN  ## cbt is the file that takes care of everything combat-related

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

SCREENSHOT_SIZE = 350  ## Width of the window used to check wether a resource is available

# Bounds for the map coordinates screenshot
TOP_CORNER_MAP = (3, 104)
SIZE_MAP = (124, 22)

# Paths and images to check wether the inventory is full or almost full 
path_full = resource_path("templates/inventory/Full.png")
path_almost_full = resource_path("templates/inventory/Almost_full.png")
path_not_full = resource_path("templates/inventory/Not_full.png")
template_not_full = cv2.imread(path_not_full, cv2.IMREAD_COLOR)
template_full = cv2.imread(path_full, cv2.IMREAD_COLOR)
template_almost_full = cv2.imread(path_almost_full, cv2.IMREAD_COLOR)


## General function definitions

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

def astar(start, end, walls = []):
    """
    This is the classic A-star algorithm covered by ai50 wusing the Manhattan distance defined by the maps of the game. 
    The walls argument is a list of tuples of coordinates ((x1, y1), (x2, y2)) that define the forbidden 
    paths from one map to another, i.e you can't go from map (x1, y1) to (x2, y2)"""

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
    """
    This is just a while loop to go from one map to another using the keyboard.
    Sometimes the character is busy or some other thing goes wrong so we need a while loop
    to ensure the move is made. This is only used to move the character from adjacent maps."""
    curr_pos = get_map()
    while curr_pos == None:
        time.sleep(0.1)
        curr_pos = get_map()
    count = 10
    while (curr_pos != dest):
        if count > 9:
            fight_status = screenshot_high_res(CORNER_FIGHT_BTN[0], CORNER_FIGHT_BTN[1], SIZE_FIGHT_BTN[0], SIZE_FIGHT_BTN[1])
            while check_fight(fight_status):
                take_action(fight_status)
                time.sleep(1)
                fight_status = screenshot_high_res(CORNER_FIGHT_BTN[0], CORNER_FIGHT_BTN[1], SIZE_FIGHT_BTN[0], SIZE_FIGHT_BTN[1])
            check_popup()
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

def go_direction(direction):
    """
    Press the button corresponding to the direction. direction is expected to be a cardinal point letter."""
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

def go_to(destination, walls = []):
    """
    Simply calls the A-star algorithm to figure out the path and executes the moves one-by-one."""
    start = get_map()
    while start == None:
        start = get_map()
    end = destination
    path = astar(start, end, walls)
    l = len(path)
    for i in range(l-1):
        move(path[-i-2])

def get_map():
    """
    Takes a high res screenshot of the part of the screen where map coordinates are displayed.
    The text is grey on a non-grey background, so we convert to HSV and make the text black on white
    by selecting only pixels with a saturation of 0. Inside buildings the background can be pure black,
    so we need to check for that too, otherwise the text would be black-on-black."""

    frame = screenshot_high_res(TOP_CORNER_MAP[0], TOP_CORNER_MAP[1], SIZE_MAP[0], SIZE_MAP[1]) # Screenshot of map coordinates
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert to HSV
    for ligne in img:
        for pixel in ligne:
            if pixel[1] == 0 and pixel[2] != 0: # If a pixel is grey ( Saturation = 0 ) but not black ( Value = 0 ), we make it black (it's the text)
                pixel[0] = 0
                pixel[1] = 0
                pixel[2] = 0
            else: # Otherwise we make it white ( it's the background )
                pixel[0] = 255
                pixel[1] = 255
                pixel[2] = 255

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to gray scale
    config = r'--psm 7 -c tessedit_char_whitelist=0123456789O,-'  # White list of characters to detect
    text = pytesseract.image_to_string(img, lang="eng", config=config) # Calls Google's open source text recognition on the image
    text = text.replace("O", "0") # Sometimes the 0 get confused with O
    match = re.search(r"(-?\d+)\s*,\s*(-?\d+)", text) # we match for coordinates (x, y). They can be negative

    if match:
        x, y = int(match.group(1)), int(match.group(2))
        return (x, y)
    else:
        return None

def get_resources(map_pos, zone="Amakna.txt", types = "All"):
    """
    We connect to the database containing the info on in-game collectable resources. 
    There are zones in the game that make coordinates degenerate, so we need to know for which zone we need the coordinates.
    Takes "All" or a list of strings as types argument, 
    and returns a dictionnary with the name of the resources as keys and a list of positions as values."""

    zone, _ = os.path.splitext(zone)
    db_path = resource_path("database/resources.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    resources = defaultdict(list) # Dictionary with empty list as default values so we can append the position tuples
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
    """
    Retrieves a path for the Bot to follow. A simple text file with the successive coordinate tuples separated by semicolons.
    Those files can be written using the RouteMaker inside the main app.
    """
    route_path = resource_path("routes/" + route_file_name)
    with open(route_path, "r") as f:
        content = f.read().strip()
    elements = content.split(";")
    list_maps = [ast.literal_eval(e) for e in elements]
    return list_maps

def get_walls(walls_file_name):
    """
    Retrieves the walls for a given zone from a text file. A simple text file with the successive tuples of coordinates ((x1, y1), (x2, y2)) separated by semicolons.
    The wall text files are created using the WallMapper app. Unlike the RouteMaker, it is not intended to be used by the user.
    The wall files are named after zone they correspond to.
    """
    pathname = resource_path("walls/" + walls_file_name)
    with open(pathname, "r") as f:
        content = f.read().strip()
    elements = content.split(";")
    list_walls = [ast.literal_eval(e) for e in elements]
    return list_walls

def get_name_pos(name_template_file):
    """
    Gets the position of the name tag of the character. Name tags can be activated in game by pressing 'p'. 
    Not necessary but improves the bot operation by detecting its movements.
    The user should take a screenshot of the name tag and add it to the templates/names folder using the name
    of the character as file name."""

    template_path = resource_path("templates/names/" + name_template_file)
    # Take a screenshot with mss library and convert it to numpy
    with mss.mss() as sct:
        sct_img = sct.grab(sct.monitors[1])
        screen = np.array(sct_img)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
    # Load template (name tag)
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    # Simple template matching. Returns only if a there is a match below the threshold
    res = cv2.matchTemplate(screen, template, cv2.TM_SQDIFF_NORMED)
    local_min = (-res == maximum_filter(-res, size=5)) & (res <= 0.20)
    y, x = np.where(local_min)
    points = list(zip(x, y))
    return points

def check_at(pos, template_file_name):
    """
    When the mouse hovers over a resource, a template appears in game indicating wether the resource is available. 
    Using template matching, we use this to infer availability, knowing the resource position already. 
    pos: position of the resource to check
    template_file_name: template to match for availability
    returns True if available, else False
    """

    template_path = resource_path("templates/resources/" + template_file_name)
    # Hover over the resource
    mouse = MouseController()
    mouse.position = (pos[0], pos[1])
    time.sleep(0.3)
    # Screenshot around the mouse
    with mss.mss() as sct:
        region = {"top": pos[1] - SCREENSHOT_SIZE//2, "left": pos[0] - SCREENSHOT_SIZE//2, "width": SCREENSHOT_SIZE, "height": SCREENSHOT_SIZE}
        sct_img = sct.grab(region)
        img = np.array(sct_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Template matching on the screenshot
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
    local_min = (-res == maximum_filter(-res, size=5)) & (res <= 0.3) # Le match n'est pas parfait car mss réagit bizarrement avec retina
    y, x = np.where(local_min)
    points = list(zip(x, y))

    return bool(points) # If there is a match, returns True.

def inventory_full():
    """
    Opens the inventory. Take screenshot of a resttricted region and checks for some templates to know wether the inventory is full.
    Then closes the inventory.
    It can happen that the inventory is already open, or that only one click is registered.
    If no template is found to match, we open/close the inventory only once and call the function recursivly.
    """

    keyboard = KeyboardController()
    # Open inventory
    keyboard.press('i')
    time.sleep(0.03)
    keyboard.release('i')
    time.sleep(0.7)
    # Screenshot
    frame = screenshot_high_res(770, 750, 28, 34)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    # Close inventory
    keyboard.press('i')
    time.sleep(0.03)
    keyboard.release('i')
    # Template matching
    full = cv2.matchTemplate(frame, template_full, cv2.TM_SQDIFF_NORMED)
    almost_full = cv2.matchTemplate(frame, template_almost_full, cv2.TM_SQDIFF_NORMED)
    not_full =  cv2.matchTemplate(frame, template_not_full, cv2.TM_SQDIFF_NORMED)

    if np.nanmin(full) <= 2e-1 or np.nanmin(almost_full) <= 2e-1:
        return True
    elif np.nanmin(not_full) <= 2e-1:
        return False
    else: # If no match is found, we press the inventory button only once and call the function recursively
        keyboard.press('i')
        time.sleep(0.03)
        keyboard.release('i')
        return inventory_full()

    
def empty_inventory(walls):
    """ 
    Criptic but simple: we go to the bank and make a series of clicks to empty our inventory. 
    Can sometimes go wrong. Could be improved by checking for states with screenshots but to litle benefit.
    """
    go_to((4, -18), walls)
    keyboard = KeyboardController()
    keyboard.press('p')
    time.sleep(0.03)
    keyboard.release('p')
    pyautogui.click(875, 300)
    time.sleep(5)
    pyautogui.click(750, 325)
    time.sleep(2)
    pyautogui.click(1025, 380)
    time.sleep(2)
    pyautogui.click(675, 330)
    time.sleep(2)
    pyautogui.click(970, 250)
    time.sleep(1)
    pyautogui.click(1061, 327)
    time.sleep(1)
    pyautogui.click(980, 195)
    time.sleep(1)
    pyautogui.click(516, 620)
    time.sleep(5)
    keyboard.press('p')
    time.sleep(0.03)
    keyboard.release('p')
    
def run_script(route_name, resource_names, character_name, zone_name="Amakna.txt"):
    """
    zone_name: name of the zone walls, i.e: "Amakna.txt"
    route_name: name of the path defined created by the user, i.e: "Astrub_forest.txt"
    resource_names: list of resources to collect along the way, i.e: ["Frêne", "Ortie", "Châtaigner"]
    character_name: name of the character. There should be an image of the name tag named "character_name.png" in the templates/names directory.

    Where the magic happens.
    Press 'n' to start/pause the BOT. Press 'z' to stop it definitely.
    The BOT will follow the route defined in the route_name file, and along that path collect the resources defined in resource_names.
    An inventory check is regularly performed, and the BOT goes to the bank to empty its inventory when full or almost full. 
    The BOT can be aggro'd by monster while collecting resources. 
    It checks for fights after collecting resources, and also inside the function 'move'. 
    When a fight is detected, the BOT goes through the fight loop until the fight is over. More info in the 'cbt.py' file.

    """
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
        elif key == keyboard.KeyCode.from_char('z'):
            terminate = True

    listener = keyboard.Listener(on_press = on_press)
    listener.start()


    walls = get_walls(zone_name)
    path = get_route(route_name)
    while True:
        time.sleep(1)
        while run:
            for destination in path: # Goes along the path one map after the other
                bank_count += 1
                if bank_count == 5:
                    bank_count = 0
                    if inventory_full():
                        empty_inventory(walls)
                go_to(destination, walls)
                if terminate:
                    break
                resources = get_resources(destination, zone_name, resource_names) # retrieve the positions of the resources to collect for the current map
                for resource, positions in resources.items():  # For each resource type, checks all potential positions on the screen. 
                    for pos in positions:
                        if check_at(pos, resource + ".png"): # If a resource is available, collects it.
                            time.sleep(0.1)
                            pyautogui.click(pos[0], pos[1]) # Click on the resource. The mouse is already on it from the check_at call
                            time.sleep(0.1)
                            pyautogui.moveRel(random.randint(100, 200), random.randint(100, 200), 0.2) # Moves the mouse out of the way
                            moving = True # The character gets moving towards the resource. We shall not click on another resource until he is done moving
                            name_pos = get_name_pos(character_name) # Get the position of the BOT form its name tag
                            time.sleep(1.5) # Give it some time to move
                            while(moving): # While loop to wait until the BOT is stationary. If the name tag doesn't appear on screen (user forgot to turn it on using 'p' or the BOT is in a fight) then the BOT will be considered immobile.
                                new_name_pos = get_name_pos(character_name)
                                if new_name_pos == name_pos: 
                                    moving = False
                                    time.sleep(1.5)
                                else: 
                                    name_pos = new_name_pos
                                    time.sleep(1.5)
                            fight_status = screenshot_high_res(CORNER_FIGHT_BTN[0], CORNER_FIGHT_BTN[1], SIZE_FIGHT_BTN[0], SIZE_FIGHT_BTN[1])
                            while check_fight(fight_status):
                                take_action(fight_status)
                                time.sleep(1)
                                fight_status = screenshot_high_res(CORNER_FIGHT_BTN[0], CORNER_FIGHT_BTN[1], SIZE_FIGHT_BTN[0], SIZE_FIGHT_BTN[1])
                            check_popup()
        
        
if __name__=="__main__":
    run_script("Forêt d'Astrub 2.txt", ["Frêne", "Châtaigner"], "Brigitte-Fardeau.png", "Amakna.txt")

    # # # # # # # # # # # # # # # # # # # # # # # 
    # Just some code to test specific functions #
    # # # # # # # # # # # # # # # # # # # # # # # 

    # def on_press(key):
    #     global requested
    #     if key == keyboard.KeyCode.from_char('n'):
    #         if requested:
    #             requested = False
    #         else:
    #             requested = True

    # listener = keyboard.Listener(on_press = on_press)
    # listener.start()

    # requested = False
    # while True:
    #     if requested:
    #         print(get_map())
    #         requested = False
