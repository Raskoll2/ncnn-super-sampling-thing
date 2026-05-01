import os
import cv2
import h5py
import numpy as np
import random
from tqdm import tqdm
import glob
import sys

# ==========================================
# CONFIGURATION
# ==========================================
VIDEO_DIR = '/home/raskoll/Downloads/Movies/toMove/'         
OUTPUT_DB = './fast_crops_256.h5'     
CROP_SIZE = 256                       
FRAME_SKIP = 48
BUFFER_SIZE = 200 # Smaller buffer = more frequent disk writes

def get_video_files(directory):
    extensions = ('*.mp4', '*.mkv', '*.avi', '*.mov')
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))
    return files

def build_direct_fast_dataset():
    video_files = get_video_files(VIDEO_DIR)
    if not video_files:
        print(f"Error: No videos found in {VIDEO_DIR}")
        return

    print(f"Found {len(video_files)} videos. Press Ctrl+C to stop and save at any time.")

    # libver='latest' helps with more robust writing
    with h5py.File(OUTPUT_DB, 'w', libver='latest') as hf_out:
        dataset = hf_out.create_dataset(
            'crops', 
            shape=(0, CROP_SIZE, CROP_SIZE, 3), 
            maxshape=(None, CROP_SIZE, CROP_SIZE, 3),
            dtype='uint8',
            chunks=(1, CROP_SIZE, CROP_SIZE, 3) 
        )

        crop_buffer = []
        total_crops_written = 0

        try:
            for video_path in video_files:
                cap = cv2.VideoCapture(video_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if width < CROP_SIZE or height < CROP_SIZE:
                    cap.release()
                    continue

                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    if frame_idx % FRAME_SKIP == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        top = random.randint(0, height - CROP_SIZE)
                        left = random.randint(0, width - CROP_SIZE)
                        crop = frame_rgb[top:top+CROP_SIZE, left:left+CROP_SIZE, :]
                        crop_buffer.append(crop)
                        
                        if len(crop_buffer) >= BUFFER_SIZE:
                            current_size = dataset.shape[0]
                            dataset.resize((current_size + len(crop_buffer), CROP_SIZE, CROP_SIZE, 3))
                            dataset[current_size:] = np.array(crop_buffer)
                            
                            # Force the Metadata and Data to the physical SSD
                            hf_out.flush() 
                            
                            total_crops_written += len(crop_buffer)
                            crop_buffer = []
                            sys.stdout.write(f"\rTotal Crops Saved: {total_crops_written} | File Size: {os.path.getsize(OUTPUT_DB)/1024**2:.1f} MB")
                            sys.stdout.flush()

                    frame_idx += 1
                cap.release()

        except KeyboardInterrupt:
            print("\n\nStopping early... saving buffer and closing file.")
            if len(crop_buffer) > 0:
                current_size = dataset.shape[0]
                dataset.resize((current_size + len(crop_buffer), CROP_SIZE, CROP_SIZE, 3))
                dataset[current_size:] = np.array(crop_buffer)
                total_crops_written += len(crop_buffer)

    print(f"\nFinished! Total crops in {OUTPUT_DB}: {total_crops_written}")

if __name__ == "__main__":
    build_direct_fast_dataset()
