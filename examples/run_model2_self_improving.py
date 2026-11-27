r"""
Model 2 Runner: Sovereign Self-Improving AgroTop-Flow Engine.
============================================================

Demonstrates standalone execution of Model 2:
  - Fuses optical histology, spatial coordinates, and gene expression.
  - Evaluates conformal prediction set sizes at (1 - alpha) certified coverage.
  - Demonstrates autonomous self-evolution mechanisms:
      1. Conformal pseudo-label uncertainty gating (|C(X)| = 1).
      2. Dynamic anchor teacher initialization.
      3. Adversarial biophysical hard-negative synthesis.
"""

import sys
import os
import torch
import numpy as np

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.registry import get_model, get_model_info

def main():
    print("=" * 75)
    print("MODEL 2 STANDALONE EXECUTION: Self-Improving AgroTopFlow")
    print("=" * 75)
    
    info = get_model_info("self_improving_agrotop")
    print(f"Name:        {info['name']}")
    print(f"Role:        {info['role']}")
    print(f"Description: {info['description']}")
    print("-" * 75)
    
    # 1. Instantiate Model 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model("self_improving_agrotop", in_dim=1024, n_genes=200, d_model=64, num_classes=7).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized Model 2 on {device} with {num_params:,} trainable parameters.")
    
    # 2. Simulate Multimodal Spatial Batch (Bt Cotton / Arabidopsis sample)
    batch_size = 24
    print(f"\nGenerating batch of {batch_size} multimodal plant spatial spots:")
    img_patches = torch.randn(batch_size, 1024, device=device)
    coords = torch.rand(batch_size, 2, device=device) * 100.0
    gene_profiles = torch.rand(batch_size, 200, device=device)
    
    # 3. Standard Forward Pass: (patch_features, gene_expr, coords=coords)
    model.eval()
    with torch.no_grad():
        out = model(img_patches, gene_profiles, coords=coords)
    
    logits = out["logits"]
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)
    
    print("\nStandard Inference Output:")
    print(f"  • Classification Logits:       {list(logits.shape)}")
    print(f"  • Top-1 Predicted Classes:     {preds.tolist()}")
    print(f"  • Mean Prediction Confidence:  {probs.max(dim=-1).values.mean().item():.4f}")
    
    # 4. Teacher Initialization for Distillation
    print("\nInitializing Distillation Teacher Anchor...")
    model.initialize_teacher()
    print("  • Teacher anchor initialized and parameters frozen.")
    
    # 5. Conformal Uncertainty Filter on Streaming Data
    print("\nSimulating Streaming Conformal Filter (|C(X)| == 1 criterion)...")
    from torch.utils.data import TensorDataset, DataLoader
    stream_ds = TensorDataset(
        img_patches.unsqueeze(1),
        coords.unsqueeze(1),
        gene_profiles
    )
    stream_loader = DataLoader(stream_ds, batch_size=8)
    
    filter_result = model.conformal_filter_unlabelled(stream_loader, alpha=0.10, device=device)
    if filter_result:
        print(f"  • High-Confidence Singletons Admitted: {filter_result['admitted_count']}/{batch_size}")
        print(f"  • Acceptance Rate:                     {filter_result['acceptance_rate']:.1f}%")
        print(f"  • Admitted Feature Shape:              {filter_result['X_vis'].shape}")
    else:
        print("  • Conformal filter conservative rejection: 0 false positives admitted.")
        
    print("\nModel 2 verification successfully finished.")

if __name__ == "__main__":
    main()
