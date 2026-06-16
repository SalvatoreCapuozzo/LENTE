import os
import glob
import time
import h5py
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Import from your existing L4S utilities
from LENTE_utils import LENTE, get_best_mask, scale_boxes

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_FOLDER = "/datadrive/LENTE/SAM3/SAM2/LandSlide4Sense/TestData/img" # Directory containing .h5 files
CHECKPOINT_NAME = "best_sam2tiny_adapter_h5.pth"
SAM_CKPT_PATH = "facebook/sam2-hiera-tiny"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SIZE = 1024  # SAM's expected input size
ADAPTER_DIM = 64

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def load_h5_tensor(filepath):
    """Loads a 14-channel H5 file and returns a (1, 14, H, W) tensor."""
    with h5py.File(filepath, 'r') as f:
        img_np = f['img'][:] # Shape: (128, 128, 14)
    
    # Permute from (H, W, C) to (C, H, W) before adding the batch dimension!
    img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor, img_np

def prep_visualization_frame(img_np, mask_np):
    """
    Extracts RGB (for OpenCV BGR), normalizes it, and overlays the mask.
    L4S bands: B4(red)=idx 3, B3(green)=idx 2, B2(blue)=idx 1.
    """
    # img_np is (H, W, C), so we slice the last dimension (no transpose needed)
    bgr = img_np[:, :, [1, 2, 3]]
    
    # Normalize to 0-255 uint8
    bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min() + 1e-8)
    bgr = (bgr * 255).astype(np.uint8)
    
    # Resize back to TARGET_SIZE for visualization clarity
    bgr_resized = cv2.resize(bgr, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask_np, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)
    
    # Create a red overlay for the mask
    overlay = bgr_resized.copy()
    overlay[mask_resized == 1] = [0, 0, 255] # Red in BGR
    
    # Blend the original image and the overlay (alpha=0.6, beta=0.4)
    blended = cv2.addWeighted(overlay, 0.4, bgr_resized, 0.6, 0)
    
    return blended

# ============================================================
# MAIN STREAMING LOOP
# ============================================================
def run_livestream():
    print(f"Initializing Live Stream on {DEVICE}...")
    
    # 1. Load Model
    model = LENTE(sam_ckpt_path=SAM_CKPT_PATH, adapter_dim=ADAPTER_DIM).to(DEVICE)
    if os.path.exists(CHECKPOINT_NAME):
        ckpt = torch.load(CHECKPOINT_NAME, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print("Weights loaded successfully.")
    else:
        print("WARNING: Checkpoint not found. Running with untrained weights.")
    
    model.eval()
    
    # 2. Get stream files
    h5_files = sorted(glob.glob(os.path.join(DATASET_FOLDER, "*.h5")))
    if not h5_files:
        print(f"No .h5 files found in {DATASET_FOLDER}")
        return
        
    print(f"Found {len(h5_files)} frames. Starting livestream simulation...")
    
    # Warmup GPU (Jetson requires a few passes to initialize CUDA context fully)
    dummy_input = torch.randn(1, 14, TARGET_SIZE, TARGET_SIZE, device=DEVICE)
    dummy_boxes = torch.tensor([[0., 0., float(TARGET_SIZE), float(TARGET_SIZE)]], device=DEVICE)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
        for _ in range(3):
            _ = model(dummy_input, dummy_boxes)
            
    # FPS tracking variables
    fps_smoothing = 0.9
    fps = 0.0

    # 3. Stream Loop
    for filepath in h5_files:
        loop_start = time.time()
        
        # Load Data
        img_tensor, img_np = load_h5_tensor(filepath)
        orig_h, orig_w = img_tensor.shape[-2], img_tensor.shape[-1]
        
        # Default full-image bounding box
        boxes = torch.tensor([[0., 0., float(TARGET_SIZE), float(TARGET_SIZE)]])
        
        # Move to GPU and resize
        img_tensor = img_tensor.to(DEVICE)
        boxes = boxes.to(DEVICE)
        
        img_resized = F.interpolate(img_tensor, (TARGET_SIZE, TARGET_SIZE), mode="bilinear", align_corners=False)
        boxes_scaled = scale_boxes(boxes, orig_h, orig_w, TARGET_SIZE)
        
        # Inference (FP16 via Autocast)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
            all_preds, scores = model(img_resized, boxes_scaled)
            preds_resized = get_best_mask(all_preds, scores)
            
            # Convert logits to binary mask map
            preds = F.interpolate(preds_resized.float(), (orig_h, orig_w), mode="bilinear", align_corners=False)
            prob_map = torch.sigmoid(preds).squeeze().cpu().numpy()
            pred_mask = (prob_map > 0.5).astype(np.uint8)
        
        # Prepare OpenCV visualization
        display_frame = prep_visualization_frame(img_np, pred_mask)
        
        # Calculate FPS
        infer_time = time.time() - loop_start
        current_fps = 1.0 / infer_time
        fps = (fps * fps_smoothing) + (current_fps * (1.0 - fps_smoothing))
        
        # Render FPS and info on frame
        filename = os.path.basename(filepath)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(display_frame, f"Frame: {filename}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        #cv2.putText(display_frame, "Target: Jetson Orin Nano (FP16)", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        
        # Display window
        cv2.imshow("LENTE - Landslide Detection Livestream", display_frame)
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Livestream terminated by user.")
            break

    cv2.destroyAllWindows()
    print("Livestream ended.")

if __name__ == "__main__":
    run_livestream()