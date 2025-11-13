import torch
import cv2
import numpy as np
from torch import nn
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# 1. Device setup
# =========================================================

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# =========================================================
# 2. Model definition (same as during training)
# =========================================================

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


# =========================================================
# 3. Load model
# =========================================================

model = UNet().to(device)
model.load_state_dict(torch.load("tile_segmentation_unet.pth", map_location=device))
model.eval()

# =========================================================
# 4. Load your new screenshot
# =========================================================

IMG_PATH = os.path.join(BASE_DIR, "resources", "test", "screenshot.png")  # <-- change this
img = cv2.imread(IMG_PATH)[:, :, ::-1] / 255.0
tic = time.time()
img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
tac = time.time()
print((tac-tic) * 1000, "ms")

# =========================================================
# 5. Predict mask
# =========================================================

tic = time.time()
with torch.no_grad():
    pred = model(img_tensor)[0, 0].cpu().numpy()
tac = time.time()
print((tac - tic) * 1000, "ms")

# =========================================================
# 6. Threshold and visualize
# =========================================================

mask = (pred > 0.5).astype(np.uint8) * 255
mask = cv2.resize(mask, (1400, 900), interpolation=cv2.INTER_NEAREST)
# Overlay mask on image for visualization
img = cv2.resize(img, (1400, 900), interpolation=cv2.INTER_LINEAR)
overlay = img.copy()
overlay[mask == 255] = [1.0, 0.0, 0.0]  # red overlay


result = (overlay * 255).astype(np.uint8)[:, :, ::-1]  # back to BGR for OpenCV

cv2.imshow("Prediction Overlay", result)
cv2.waitKey(0)
cv2.destroyAllWindows()