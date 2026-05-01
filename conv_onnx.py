import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from onnxsim import simplify

# ==========================================
# MODEL ARCHITECTURE (Required for loading weights)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(self.pool(x))

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
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
            nn.SiLU(inplace=True),
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
    def __init__(self, num_res_blocks=6): 
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2) 
        self.conv_in = nn.Conv2d(12, 32, 3, padding=1)
        self.res_blocks = nn.Sequential(*[CBAMResBlock(32) for _ in range(num_res_blocks)])
        self.conv_mid = nn.Conv2d(32, 32, 3, padding=1)
        self.conv_up = nn.Conv2d(32, 48, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(4) 
        
    def forward(self, x):
        base = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        feat = self.unshuffle(x)
        feat = F.silu(self.conv_in(feat))
        res = self.conv_mid(self.res_blocks(feat)) + feat
        out = self.pixel_shuffle(self.conv_up(res))
        return out + base

# ==========================================
# EXPORT LOGIC
# ==========================================
def main():
    print("Instantiating model...")
    model = LightningUpscaler(num_res_blocks=4, num_)
    
    print("Loading weights...")
    # Your training script saves using model.half().state_dict()
    # We load them to CPU and convert back to float32 for clean ONNX export
    state_dict = torch.load("lightning_v3_best.pth", map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.float()
    model.eval()

    # Create a dummy input (Batch=1, Channels=3, H=256, W=256)
    dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32)

    onnx_path = "lightning_v3.onnx"
    onnx_sim_path = "lightning_v3_sim.onnx"

    print(f"Exporting raw ONNX to {onnx_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=14, # Opset 14 natively supports SiLU and complex interpolations
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'}, 
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )

    print(f"Simplifying ONNX model to {onnx_sim_path}...")
    onnx_model = onnx.load(onnx_path)
    model_simp, check = simplify(onnx_model)
    
    if not check:
        print("Warning: Simplified ONNX model could not be validated.")
    else:
        onnx.save(model_simp, onnx_sim_path)
        print(f"Success! Model simplified and saved to {onnx_sim_path}")
        print("\nNext step for ncnn:")
        print(f"onnx2ncnn {onnx_sim_path} upscaler.param upscaler.bin")

if __name__ == "__main__":
    main()
