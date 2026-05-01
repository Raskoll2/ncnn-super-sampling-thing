import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import time
import sys
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. CBAM ARCHITECTURE (Must match v3 training)
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
        base = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        feat = self.unshuffle(x)
        feat = F.silu(self.conv_in(feat))
        res = self.conv_mid(self.res_blocks(feat)) + feat
        out = self.pixel_shuffle(self.conv_up(res))
        return out + base

# ==========================================
# 2. INFERENCE ENGINE
# ==========================================
def upscale_video_v3(input_video, output_video, batch_size=32, weights_path="lightning_v3_best.pth"):
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        return

    print(f"Initializing CBAM Engine on {device}...")
    model = LightningUpscaler(num_res_blocks=6).to(device)
    
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print("Optimizing: Casting to bfloat16 and compiling kernels...")
    model = model.bfloat16()
    model = torch.compile(model)

    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_width, out_height = width * 2, height * 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (out_width, out_height))

    frames_batch = []
    frames_processed = 0
    total_inference_time = 0.0

    print(f"\nProcessing: {input_video}")
    print(f"Target: {width}x{height} -> {out_width}x{out_height} @ {fps:.2f}fps")
    print("-" * 50)

    dummy_input = torch.randn(1, 3, height, width, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        _ = model(dummy_input)

    start_global_time = time.perf_counter()

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if ret:
                frames_batch.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if len(frames_batch) == batch_size or (not ret and len(frames_batch) > 0):
                current_batch_size = len(frames_batch)
                
                batch_tensor = torch.from_numpy(np.array(frames_batch)).permute(0, 3, 1, 2)
                batch_tensor = batch_tensor.to(device, dtype=torch.bfloat16) / 255.0

                if device.type == 'cuda': torch.cuda.synchronize()
                batch_start = time.perf_counter()

                upscaled_batch = model(batch_tensor)

                if device.type == 'cuda': torch.cuda.synchronize()
                batch_end = time.perf_counter()
                
                batch_time = batch_end - batch_start
                total_inference_time += batch_time
                current_fps = current_batch_size / batch_time
                avg_fps = frames_processed / total_inference_time if frames_processed > 0 else current_fps

                upscaled_batch = (upscaled_batch.float() * 255).clamp(0, 255).byte()
                upscaled_batch = upscaled_batch.permute(0, 2, 3, 1).cpu().numpy()

                for upscaled_frame in upscaled_batch:
                    out.write(cv2.cvtColor(upscaled_frame, cv2.COLOR_RGB2BGR))

                frames_processed += current_batch_size
                sys.stdout.write(f"\rFrames: {frames_processed}/{total_frames} | Batch FPS: {current_fps:.1f} | Avg FPS: {avg_fps:.1f}  ")
                sys.stdout.flush()
                
                frames_batch = [] 

            if not ret: break

    cap.release()
    out.release()
    
    total_time = time.perf_counter() - start_global_time
    print(f"\n\nUpscale Complete!")
    print(f"Total Time: {total_time:.1f}s | Final Output FPS: {total_frames/total_time:.1f}")

if __name__ == "__main__":
    upscale_video_v3(
        input_video="test.mp4", 
        output_video="testUP.mp4", 
        batch_size=4,
        weights_path="lightning_v3_best.pth"
    )
