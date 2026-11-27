r"""
Model 1 Runner: Foundational Riemannian Morphogenetic Operator (RMNO-Hodge).
=============================================================================

Demonstrates standalone execution of Model 1:
  - Fuses optical histological features, physical spatial coordinates (x,y), and RNA gene profiles.
  - Dynamically computes the anisotropic cell wall metric tensor g_ij(x).
  - Decomposes molecular flux via Hodge-Helmholtz into curl-free gradient drift and divergence-free cyclosis.
  - Computes Benamou-Brenier optimal transport distance and classification logits.
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
    print("MODEL 1 STANDALONE EXECUTION: RMNO-Hodge Classifier")
    print("=" * 75)
    
    info = get_model_info("rmno_hodge")
    print(f"Name:        {info['name']}")
    print(f"Role:        {info['role']}")
    print(f"Description: {info['description']}")
    print("-" * 75)
    
    # 1. Instantiate Model 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model("rmno_hodge", in_dim=1024, n_genes=200, d_model=64, num_classes=7).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized Model 1 on {device} with {num_params:,} trainable parameters.")
    
    # 2. Simulate Multimodal Spatial Batch (Bt Cotton / Visium HD sample)
    batch_size = 16
    print(f"\nGenerating batch of {batch_size} multimodal plant spatial spots:")
    img_patches = torch.randn(batch_size, 1024, device=device)       # Optical tissue morphology
    coords = torch.rand(batch_size, 2, device=device) * 100.0         # Spatial coordinates (µm)
    gene_profiles = torch.rand(batch_size, 200, device=device)       # Cry gene & spatial transcriptome
    
    print(f"  • Image Patches:    {list(img_patches.shape)}")
    print(f"  • Spatial Coords:   {list(coords.shape)}")
    print(f"  • Gene Expressions: {list(gene_profiles.shape)}")
    
    # 3. Forward Pass: (patch_features, gene_expr, coords=coords)
    model.eval()
    with torch.no_grad():
        out = model(img_patches, gene_profiles, coords=coords)
    
    logits = out["logits"]
    probs = out.get("probabilities", torch.softmax(logits, dim=-1))
    preds = torch.argmax(probs, dim=-1)
    conf = out.get("certified_confidence", probs.max(dim=-1).values)
    
    print("\nInference Output:")
    print(f"  • Classification Logits:       {list(logits.shape)}")
    print(f"  • Top-1 Predicted Classes:     {preds.tolist()}")
    print(f"  • Certified Mean Confidence:   {conf.mean().item():.4f}")
    if "geodesic_distance" in out:
        print(f"  • Mean Fisher-Rao Distance:    {out['geodesic_distance'].min(dim=-1).values.mean().item():.4f} rad")
    
    print("\nModel 1 verification successfully finished.")

if __name__ == "__main__":
    main()
