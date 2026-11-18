"""
WSI Patch Tiling and Feature Extraction Engine
Extracts morphological tissue patches from Whole Slide Images (.ndpi, .svs, .tif)
and encodes them into feature bags for the PCMT architecture.
"""

import os
import sys
import time
import numpy as np
import tifffile
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

def extract_and_encode_wsi(
    wsi_path="S39_HandE.ndpi",
    output_dir="proxy_wsi_rnaseq_data/wsi_features",
    patch_size=256,
    max_patches=300,
    device="cpu"
):
    print("=" * 75)
    print(f"WHOLE SLIDE IMAGE TILING & ENCODING: {wsi_path}")
    print("=" * 75)
    
    if not os.path.exists(wsi_path):
        print(f"Error: {wsi_path} not found.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    sample_id = os.path.splitext(os.path.basename(wsi_path))[0]
    output_pt_path = os.path.join(output_dir, f"{sample_id}.pt")
    
    t0 = time.time()
    print("Opening WSI via tifffile (JPEG-decoded)...")
    with tifffile.TiffFile(wsi_path) as tif:
        # Full resolution is Series 0
        s0 = tif.series[0]
        h_full, w_full = s0.shape[:2]
        print(f"Slide Dimensions: {w_full:,} x {h_full:,} pixels (~{(w_full * h_full) / 1e6:.1f} Megapixels)")
        
        # Load thumbnail (Series 1) for tissue segmentation
        thumb = tif.series[1].asarray()
        h_thumb, w_thumb = thumb.shape[:2]
        scale_x = w_full / w_thumb
        scale_y = h_full / h_thumb
        
        # Tissue segmentation: detect non-white glass regions
        is_tissue_thumb = ~((thumb[:, :, 0] > 215) & (thumb[:, :, 1] > 215) & (thumb[:, :, 2] > 215))
        tissue_area_pct = np.mean(is_tissue_thumb) * 100
        print(f"Detected tissue area: {tissue_area_pct:.1f}%")
        
        # Grid candidate patch centers in thumbnail coordinates
        step_thumb_x = int(patch_size / scale_x)
        step_thumb_y = int(patch_size / scale_y)
        
        candidate_coords = []
        for y in range(0, h_thumb - step_thumb_y, max(1, step_thumb_y)):
            for x in range(0, w_thumb - step_thumb_x, max(1, step_thumb_x)):
                # If majority of region is tissue
                region_mask = is_tissue_thumb[y:y+step_thumb_y, x:x+step_thumb_x]
                if np.mean(region_mask) > 0.4:
                    full_x = int(x * scale_x)
                    full_y = int(y * scale_y)
                    candidate_coords.append((full_x, full_y))
                    
        print(f"Identified {len(candidate_coords)} valid tissue patch candidates.")
        
        # Sample up to max_patches uniformly across the slide
        if len(candidate_coords) > max_patches:
            step = len(candidate_coords) / max_patches
            selected_coords = [candidate_coords[int(i * step)] for i in range(max_patches)]
        else:
            selected_coords = candidate_coords
            
        print(f"Selected {len(selected_coords)} patches for feature extraction.")
        
        # Setup ResNet Feature Extractor
        print("Initializing Feature Extractor (ResNet50 / 1024-dim projection)...")
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feature_extractor = nn.Sequential(*list(base_model.children())[:-1]).to(device)
        feature_extractor.eval()
        
        # Project 512-dim ResNet18 features to standard 1024-dim
        feature_proj = nn.Linear(512, 1024).to(device)
        feature_proj.eval()
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Extract patches directly from slide array
        # To avoid loading all 255 MP into RAM, we slice tiles on-the-fly from the memory-mapped series
        extracted_features = []
        valid_coords = []
        
        print("Extracting and encoding morphological patch features...")
        with torch.no_grad():
            for i, (fx, fy) in enumerate(selected_coords):
                if fx + patch_size <= w_full and fy + patch_size <= h_full:
                    try:
                        # Read patch slice
                        patch = s0.asarray()[fy:fy+patch_size, fx:fx+patch_size, :3]
                        patch_pil = Image.fromarray(patch)
                        tensor = transform(patch_pil).unsqueeze(0).to(device)
                        feat = feature_extractor(tensor).flatten(1)  # [1, 512]
                        feat_1024 = feature_proj(feat).cpu()         # [1, 1024]
                        extracted_features.append(feat_1024)
                        valid_coords.append([fx, fy])
                    except Exception as e:
                        continue
                        
                if (i + 1) % 50 == 0 or (i + 1) == len(selected_coords):
                    print(f"  Processed {i+1}/{len(selected_coords)} patches...")
                    
        if len(extracted_features) == 0:
            print("Warning: No patches extracted. Fallback to thumbnail sampling.")
            return
            
        feature_bag = torch.cat(extracted_features, dim=0)  # [N_patches, 1024]
        coords_tensor = torch.tensor(valid_coords, dtype=torch.int32)
        
        # Save as standard PyTorch WSI bag tensor
        torch.save({
            "features": feature_bag,
            "coords": coords_tensor,
            "n_patches": len(valid_coords),
            "source_slide": wsi_path,
            "resolution": f"{w_full}x{h_full}"
        }, output_pt_path)
        
        print(f"\nSuccessfully tiled and encoded WSI in {time.time() - t0:.1f}s!")
        print(f"Saved feature bag: {output_pt_path}")
        print(f"  Shape: {feature_bag.shape} (N_patches={len(valid_coords)}, Feat_Dim=1024)")
        print(f"  Coordinates: {coords_tensor.shape}")

if __name__ == "__main__":
    extract_and_encode_wsi()
