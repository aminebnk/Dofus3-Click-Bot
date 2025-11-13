import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# 1. Use the M1 GPU if available
# =========================================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# =========================================================
# 2. Dataset
# =========================================================
class TileDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.raw_files = sorted(os.listdir(img_dir))
        self.processed_files = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.raw_files)

    def __getitem__(self, idx):
        rname = self.raw_files[idx]
        pname= self.processed_files[idx]
        img_path = os.path.join(self.img_dir, rname)
        mask_path = os.path.join(self.mask_dir, pname)

        # --- Load and normalize ---
        img = cv2.imread(img_path)[:, :, ::-1] / 255.0
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0

        # keep full 900x1400 resolution
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return img, mask


# =========================================================
# 3. Simple U-Net
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
# 4. Training setup
# =========================================================
img_dir = os.path.join(BASE_DIR, "resources", "training", "full_size", "raw")
mask_dir = os.path.join(BASE_DIR, "resources", "training", "full_size", "processed")

dataset = TileDataset(img_dir, mask_dir)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

model = UNet().to(device)
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=1e-3)

# =========================================================
# 5. Training loop
# =========================================================
for epoch in range(40):  # start with 1 epochs
    total_loss = 0.0
    model.train()

    for img, mask in train_loader:
        img, mask = img.to(device), mask.to(device)

        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# =========================================================
# 6. Save model
# =========================================================
torch.save(model.state_dict(), "tile_segmentation_unet.pth")
print("✅ Training done. Model saved as tile_segmentation_unet.pth")
