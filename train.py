import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.utils import save_image
import torchvision.io as tv_io
import time
import random
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Hyperparameters ---
epochs = 20
batch_size = 64
learning_rate = 5e-4

# ==========================================
# 1. BLAZING FAST DATALOADER (WITH CPU COMPRESSION)
# ==========================================
class FastCropDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.hf = None
        with h5py.File(self.h5_path, 'r') as hf:
            self.length = hf['crops'].shape[0]

    def __len__(self): return self.length

    def __getitem__(self, idx):
        if self.hf is None: self.hf = h5py.File(self.h5_path, 'r', swmr=True)
        crop_np = self.hf['crops'][idx]

        # High Res Ground Truth
        hr_tensor = torch.from_numpy(crop_np).permute(2, 0, 1).float() / 255.0

        # 1. mathematically downscale on CPU
        lr_tensor = F.interpolate(hr_tensor.unsqueeze(0), scale_factor=0.5, mode='bicubic', antialias=True).squeeze(0)

        # 2. Add compression noise 80% of the time to train the denoiser
        if random.random() < 0.8:
            quality = random.randint(30, 80)
            # Convert to uint8 for the C++ JPEG engine
            lr_uint8 = (lr_tensor * 255).clamp(0, 255).byte()
            # Compress and Decompress to simulate artifacts
            jpeg_bytes = tv_io.encode_jpeg(lr_uint8, quality=quality)
            lr_tensor = tv_io.decode_jpeg(jpeg_bytes).float() / 255.0

        return lr_tensor, hr_tensor

full_dataset = FastCropDataset(h5_path='./fast_crops_256.h5')
train_size = int(0.9 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_set, test_set = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True,
    num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4
)
test_loader = DataLoader(
    dataset=test_set, batch_size=batch_size, shuffle=False,
    num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4
)

# ==========================================
# 2. LITE CBAM ARCHITECTURE
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(self.pool(x))

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # 3x3 instead of 7x7 for massive iGPU speedup
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(y))

class CBAMResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        res = self.body(x)
        res = self.ca(res)
        res = self.sa(res)
        return res + x

class LightningUpscaler(nn.Module):
    def __init__(self, num_res_blocks=2, channels=8):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.conv_in = nn.Conv2d(12, channels, 3, padding=1)
        self.res_blocks = nn.Sequential(*[CBAMResBlock(channels) for _ in range(num_res_blocks)])
        self.conv_mid = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv_up = nn.Conv2d(channels, 48, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(4)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        feat = self.unshuffle(x)
        feat = F.leaky_relu(self.conv_in(feat), 0.2, inplace=True)
        res = self.conv_mid(self.res_blocks(feat)) + feat
        out = self.pixel_shuffle(self.conv_up(res))
        return out + base

# ==========================================
# 3. LOSS & OPTIMIZER
# ==========================================
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y)**2 + self.eps**2))

# Initialized with the lighter parameters
model = LightningUpscaler(num_res_blocks=2, channels=8).to(device)
criterion = CharbonnierLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
scaler = torch.amp.GradScaler('cuda')

# ==========================================
# 4. MULTI-ANCHOR SETUP
# ==========================================
os.makedirs('training_samples', exist_ok=True)
anchor_frames = []
"""with h5py.File('./super_res_data.h5', 'r') as hf:
    keys = list(hf.keys())
    k1, k2, k3 = keys[0], keys[len(keys)//2], keys[-1]
    anchor_frames.append({"name": "Animation", "data": hf[k1][min(120, hf[k1].shape[0]-1)]})
    anchor_frames.append({"name": "LiveAction_1", "data": hf[k2][hf[k2].shape[0]//2]})
    anchor_frames.append({"name": "LiveAction_2", "data": hf[k3][max(0, hf[k3].shape[0]-200)]})
"""
# ==========================================
# 5. TRAINING LOOP
# ==========================================
best_loss = float('inf')

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    total_data_time, total_forward_time, total_backward_time = 0, 0, 0

    pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", unit="batch")
    batch_start_time = time.perf_counter()

    for lr_batch, hr_batch in pbar:
        data_end_time = time.perf_counter()
        total_data_time += (data_end_time - batch_start_time)

        lr_batch = lr_batch.to(device)
        hr_batch = hr_batch.to(device)

        optimizer.zero_grad(set_to_none=True)

        if device.type == 'cuda': torch.cuda.synchronize()
        forward_start = time.perf_counter()

        with torch.amp.autocast('cuda'):
            outputs = model(lr_batch)
            loss = criterion(outputs, hr_batch)

        if device.type == 'cuda': torch.cuda.synchronize()
        forward_end = time.perf_counter()
        total_forward_time += (forward_end - forward_start)

        backward_start = time.perf_counter()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if device.type == 'cuda': torch.cuda.synchronize()
        backward_end = time.perf_counter()
        total_backward_time += (backward_end - backward_start)

        train_loss += loss.item()
        pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
        batch_start_time = time.perf_counter()

    avg_train_loss = train_loss / len(train_loader)
    avg_data_time = (total_data_time / len(train_loader)) * 1000
    avg_forward_time = (total_forward_time / len(train_loader)) * 1000
    avg_backward_time = (total_backward_time / len(train_loader)) * 1000

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for lr_batch, hr_batch in test_loader:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(lr_batch)
                loss = criterion(outputs, hr_batch)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    scheduler.step(avg_test_loss)

    tqdm.write(f"\n--- Epoch {epoch+1} Summary ---")
    tqdm.write(f"Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
    tqdm.write(f"Timings -> Load: {avg_data_time:.1f}ms | Forward: {avg_forward_time:.1f}ms | Backward: {avg_backward_time:.1f}ms")

    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        torch.save(model.half().state_dict(), "lightning_v3_best.pth")
        model.float()
        tqdm.write("  -> New best! Saved weights.")
    tqdm.write("-" * 25 + "\n")

    # Multi-Anchor Visual Check
    with torch.no_grad():
        for config in anchor_frames:
            hr_raw = torch.from_numpy(config["data"]).permute(2, 0, 1).float().unsqueeze(0)
            _, _, h, w = hr_raw.shape
            hr_raw = hr_raw[:, :, :h - (h % 4), :w - (w % 4)]

            lr_raw = F.interpolate(hr_raw, scale_factor=0.5, mode='bicubic', antialias=True)
            lr_uint8 = (lr_raw[0] * 255).clamp(0, 255).byte()
            jpeg_bytes = tv_io.encode_jpeg(lr_uint8, quality=40)
            lr_raw = tv_io.decode_jpeg(jpeg_bytes).float().unsqueeze(0).to(device) / 255.0

            hr_raw = hr_raw.to(device)

            with torch.amp.autocast('cuda'):
                out_raw = model(lr_raw).float()
            base_view = F.interpolate(lr_raw, scale_factor=2, mode='bilinear', align_corners=False)
            grid = torch.cat([base_view, out_raw, hr_raw], dim=3)
            save_image(grid, f'training_samples/ep{epoch+1}_{config["name"]}.png')
