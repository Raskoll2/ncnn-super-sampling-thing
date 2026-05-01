import torch
import torch.nn as nn
import torch.nn.functional as F
import pnnx

# ==========================================
# MODEL ARCHITECTURE
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
# EXPORT LOGIC
# ==========================================
if __name__ == "__main__":
    print("Instantiating model...")
    model = LightningUpscaler(num_res_blocks=2, channels=8)
    
    print("Loading weights...")
    state_dict = torch.load("lightning_v3_best.pth", map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.float()
    model.eval()

    # Create a dummy input tensor
    input_tensor = torch.rand(1, 3, 256, 256)

    print("Exporting with pnnx...")
    pnnx.export(model, "lightning_v3.pt", (input_tensor,))
    
    print("Done. Look for lightning_v3.ncnn.param and lightning_v3.ncnn.bin")
