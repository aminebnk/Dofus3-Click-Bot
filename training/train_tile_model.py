import os
import cv2
import numpy as np
from scipy.ndimage import maximum_filter
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import shutil


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
TRAING_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(TRAING_DIR, os.pardir))

FIGHT_DIR = os.path.join(BASE_DIR, "resources", "templates", "fight")
TRAINING_TILES = os.path.join(BASE_DIR, "resources", "training", "tiles", "raw")
TRAINING_MAPS = os.path.join(BASE_DIR, "resources", "training", "maps")
TILE_DIR = os.path.join(BASE_DIR, "resources", "training", "tiles", "tile")
NON_TILE_DIR = os.path.join(BASE_DIR, "resources", "training", "tiles", "non_tile")
UNCLASSIFIED_DIR = os.path.join(BASE_DIR, "resources", "training", "tiles", "unclassified")
DATASET_DIR = os.path.join(BASE_DIR, "resources", "training", "tiles", "classified")

# Load templates for monster recognition
MONSTER_NORTHEAST = cv2.imread(FIGHT_DIR + "/monster_northeast.png", cv2.IMREAD_UNCHANGED)
MONSTER_NORTHWEST = cv2.imread(FIGHT_DIR + "/monster_northwest.png", cv2.IMREAD_UNCHANGED)
MONSTER_SOUTHEAST = cv2.imread(FIGHT_DIR + "/monster_southeast.png", cv2.IMREAD_UNCHANGED)
MONSTER_SOUTHWEST = cv2.imread(FIGHT_DIR + "/monster_southwest.png", cv2.IMREAD_UNCHANGED)
NORTHEAST_BGR = MONSTER_NORTHEAST[:, :, :3]
NORTHEAST_ALPHA = MONSTER_NORTHEAST[:, :, 3]
NORTHWEST_BGR = MONSTER_NORTHWEST[:, :, :3]
NORTHWEST_ALPHA = MONSTER_NORTHWEST[:, :, 3]
SOUTHEAST_BGR = MONSTER_SOUTHEAST[:, :, :3]
SOUTHEAST_ALPHA = MONSTER_SOUTHEAST[:, :, 3]
SOUTHWEST_BGR = MONSTER_SOUTHWEST[:, :, :3]
SOUTHWEST_ALPHA = MONSTER_SOUTHWEST[:, :, 3]

# Load templates for character recognition
ZOBAL_NORTHEAST = cv2.imread(FIGHT_DIR + "/zobal_northeast.png", cv2.IMREAD_UNCHANGED)
ZOBAL_NORTHWEST = cv2.imread(FIGHT_DIR + "/zobal_northwest.png", cv2.IMREAD_UNCHANGED)
ZOBAL_SOUTHEAST = cv2.imread(FIGHT_DIR + "/zobal_southeast.png", cv2.IMREAD_UNCHANGED)
ZOBAL_SOUTHWEST = cv2.imread(FIGHT_DIR + "/zobal_southwest.png", cv2.IMREAD_UNCHANGED)
Z_NORTHEAST_BGR = ZOBAL_NORTHEAST[:, :, :3]
Z_NORTHEAST_ALPHA = ZOBAL_NORTHEAST[:, :, 3]
Z_NORTHWEST_BGR = ZOBAL_NORTHWEST[:, :, :3]
Z_NORTHWEST_ALPHA = ZOBAL_NORTHWEST[:, :, 3]
Z_SOUTHEAST_BGR = ZOBAL_SOUTHEAST[:, :, :3]
Z_SOUTHEAST_ALPHA = ZOBAL_SOUTHEAST[:, :, 3]
Z_SOUTHWEST_BGR = ZOBAL_SOUTHWEST[:, :, :3]
Z_SOUTHWEST_ALPHA = ZOBAL_SOUTHWEST[:, :, 3]
CHALLENGES = "CHOISIR LES\nCHALLENGES"
PRET = "PRET"
FIN = "FIN DE TOUR"
TOP_CORNER = (975, 835)
SIZE = (120, 28)

## Data set definition

class TileDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        self.root = root

        self.classes = {
            "tile": 1,
            "non_tile": 0
        }

        for cls_name, label in self.classes.items():
            folder = os.path.join(root, cls_name)
            for fname in os.listdir(folder):
                self.samples.append((
                    os.path.join(folder, fname),
                    float(label)
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.tensor(img, dtype=torch.float32).permute(2,0,1) / 255.0

        return img, torch.tensor([label], dtype=torch.float32), path


## Classification model definition 

class TileClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(16 * 19 * 19, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        return self.fc2(x)

## Training loop

dataset = TileDataset(DATASET_DIR)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = TileClassifier().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 200

for epoch in range(epochs):
    total_loss = 0.0

    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

## Evaluation

correct = 0
total = 0

with torch.no_grad():
    for image, label, path in dataset:
        image = image.unsqueeze(0).to(device)

        logit = model(image)
        pred = 1 if logit > 0 else 0

        if pred == label[0]:
            correct += 1
        else:
            title = "tile" if pred == 1 else "non tile"
            image = image.squeeze(0).cpu().permute(1, 2, 0) * 255.0
            image = image.numpy().astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            #cv2.imshow(path + "classified as " + title, image)
        total += 1
print("Accuracy:", correct / total)

torch.save(model.state_dict(), "tile_cnn.pth")

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

def generate_tiles(img, map_number, mp=4, square_width = 9):
    pos_character = get_character_pos(img)
    if pos_character[0] != -1:
        positions = []
        pos_character = vector(pos_character[1], pos_character[0])
        for l in range(1, mp+1):
            position_vectors = []
            for i in range(l):
                position_vectors.append(vector())
            for i in range(4):
                for vector1 in position_vectors:
                    new_tile = pos_character
                    for vector2 in position_vectors:
                        new_tile += vector2
                    positions.append(new_tile)
                    vector1.rotate_hourly()
        for i, tile in enumerate(positions):
            cv2.imwrite(os.path.join(TRAINING_TILES, "map_" + str(map_number) + "_tile_" + str(i) + ".png"), img[round(tile.y)-square_width:round(tile.y) + square_width + 1, round(tile.x)-square_width:round(tile.x) + square_width + 1, :])
            # if i < 4:
            #     cv2.rectangle(img, (round(tile.x) - square_width, round(tile.y) - square_width), (round(tile.x) + square_width, round(tile.y) + square_width), (255, 0, 0), 1)
            # elif i < 12:
            #     cv2.rectangle(img, (round(tile.x) - square_width, round(tile.y) - square_width), (round(tile.x) + square_width, round(tile.y) + square_width), (0, 255, 0), 1)
            # elif i < 24:
            #     cv2.rectangle(img, (round(tile.x) - square_width, round(tile.y) - square_width), (round(tile.x) + square_width, round(tile.y) + square_width), (0, 0, 255), 1)

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
    
def get_character_pos(img):
    northwest_match = cv2.matchTemplate(img, Z_NORTHWEST_BGR, cv2.TM_SQDIFF_NORMED, mask=Z_NORTHWEST_ALPHA)
    northeast_match = cv2.matchTemplate(img, Z_NORTHEAST_BGR, cv2.TM_SQDIFF_NORMED, mask=Z_NORTHEAST_ALPHA)
    southeast_match = cv2.matchTemplate(img, Z_SOUTHEAST_BGR, cv2.TM_SQDIFF_NORMED, mask=Z_SOUTHEAST_ALPHA)
    southwest_match = cv2.matchTemplate(img, Z_SOUTHWEST_BGR, cv2.TM_SQDIFF_NORMED, mask=Z_SOUTHWEST_ALPHA)
    print(min(np.min(northeast_match), np.min(northwest_match), np.min(southeast_match), np.min(southwest_match)))
    threshold = 55e-3
    northwest_match = (northwest_match < threshold) & (-northwest_match == maximum_filter(-northwest_match, size=15))
    southwest_match = (southwest_match < threshold) & (-southwest_match == maximum_filter(-southwest_match, size=15))
    southeast_match = (southeast_match < threshold) & (-southeast_match == maximum_filter(-southeast_match, size=15))
    northeast_match = (northeast_match < threshold) & (-northeast_match == maximum_filter(-northeast_match, size=15))

    sw_indices = np.argwhere(southwest_match) + [31, 14]
    se_indices = np.argwhere(southeast_match) + [29, 14]
    nw_indices = np.argwhere(northwest_match) + [29, 12]
    ne_indices = np.argwhere(northeast_match) + [29, 12]
    character_indices = np.concatenate((sw_indices, se_indices, nw_indices, ne_indices), axis=0)
    # for indice in character_indices:
    #     cv2.circle(img, (indice[1], indice[0]), 3, (0, 255, 0), -1)
    # cv2.imshow("match", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if character_indices.any():
        return character_indices[0]
    else:
        return [-1,-1]
    
def classify_tiles(SRC_DIR, TILE_DIR, NON_TILE_DIR, UNCLASSIFIED_DIR):

    files = [f for f in os.listdir(SRC_DIR)]

    for filename in files:
        path = os.path.join(SRC_DIR, filename)
        img = cv2.imread(path)

        # show image (zoomed so it's easier to see if needed)
        zoom = 8   # enlarge 19x19 to ~150px
        img_big = cv2.resize(img, (img.shape[1]*zoom, img.shape[0]*zoom), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("Label (y = tile, n = non-tile, q = quit)", img_big)
        key = cv2.waitKey(0)

        if key == ord('y'):
            shutil.copy(path, os.path.join(TILE_DIR, filename))
            print(f"{filename} → tiles")
        elif key == ord('n'):
            shutil.copy(path, os.path.join(NON_TILE_DIR, filename))
            print(f"{filename} → non_tiles")
        elif key == ord('q'):
            print("Quitting early…")
            break
        else:
            shutil.copy(path, os.path.join(UNCLASSIFIED_DIR))
            print("Invalid key, moved to unclassified")

    cv2.destroyAllWindows()


#if __name__=="__main__":


    # classify_tiles(TRAINING_TILES, TILE_DIR, NON_TILE_DIR, UNCLASSIFIED_DIR)

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
    #         #fight_status = screenshot_high_res(TOP_CORNER[0], TOP_CORNER[1], SIZE[0], SIZE[1])
            #take_action(fight_status)
            # fight_status = screenshot_high_res(TOP_CORNER[0], TOP_CORNER[1], SIZE[0], SIZE[1])
            # while check_fight(fight_status) and requested:
            #     take_action(fight_status)
            #     time.sleep(2)
            #     fight_status = screenshot_high_res(TOP_CORNER[0], TOP_CORNER[1], SIZE[0], SIZE[1])
            # for i in range(1, 17):
            #     map = cv2.imread(os.path.join(TRAINING_MAPS, "map_" + str(i) + ".png"))
            #     generate_tiles(map, i)
            # requested = False