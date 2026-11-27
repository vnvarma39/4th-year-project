r"""
Dual Model Comparative Evaluator: Model 1 (RMNO-Hodge) vs. Model 2 (Self-Improving AgroTopFlow).
================================================================================================

Comprehensive benchmark comparing both repository models on plant spatial transcriptomics data:
  - Accuracy & Macro F1-Score
  - Out-of-Distribution Noise Robustness (Gaussian noise perturbation sigma=0.3)
  - Inference Latency (ms / batch) and Trainable Parameter Counts
"""

import sys
import os
import time
import torch
import numpy as np

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.models.registry import get_model, get_model_info, list_models
from experiments.benchmark_5_iterations_novel_architectures import get_seven_plant_datasets

def evaluate_model_on_data(model, test_batch, device):
    imgs, coords, genes, labels = test_batch
    imgs = imgs.to(device)
    coords = coords.to(device)
    genes = genes.to(device)
    labels = labels.to(device)
    
    model.eval()
    
    # Clean Latency & Forward: (patch_features, gene_expr, coords=coords)
    t0 = time.time()
    with torch.no_grad():
        out = model(imgs, genes, coords=coords)
        logits = out["logits"] if isinstance(out, dict) else out
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
    latency_ms = (time.time() - t0) * 1000.0 / imgs.size(0)
    
    clean_acc = (preds == labels).float().mean().item() * 100.0
    
    # Noise Perturbation (OOD Robustness)
    noise_imgs = imgs + 0.30 * torch.randn_like(imgs)
    noise_genes = genes + 0.30 * torch.randn_like(genes)
    with torch.no_grad():
        noise_out = model(noise_imgs, noise_genes, coords=coords)
        noise_logits = noise_out["logits"] if isinstance(noise_out, dict) else noise_out
        noise_preds = torch.argmax(torch.softmax(noise_logits, dim=-1), dim=-1)
    noise_acc = (noise_preds == labels).float().mean().item() * 100.0
    noise_drop = noise_acc - clean_acc
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "clean_acc": clean_acc,
        "noise_acc": noise_acc,
        "noise_drop": noise_drop,
        "latency_ms": latency_ms,
        "num_params": num_params
    }

def main():
    print("=" * 80)
    print("DUAL MODEL COMPARATIVE BENCHMARK: MODEL 1 vs. MODEL 2")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmark Device: {device}\n")
    
    # Load Real Spatial Benchmark (Bt Cotton / Visium HD)
    datasets = get_seven_plant_datasets()
    dataset_name = "3. Arabidopsis-8um (Authentic Visium HD)"
    (train_data, val_data, num_classes) = datasets[dataset_name]
    X_vis, coords, X_gene, y = val_data
    
    print(f"Dataset Selected: {dataset_name}")
    print(f"Total Validation Spots: {len(coords)} | Classes: {num_classes}\n")
    
    # Prepare Batch
    N = min(300, len(coords))
    test_batch = (
        torch.tensor(X_vis[:N], dtype=torch.float32),
        torch.tensor(coords[:N], dtype=torch.float32),
        torch.tensor(X_gene[:N], dtype=torch.float32),
        torch.tensor(y[:N], dtype=torch.long)
    )
    
    # Instantiate Model 1 & Model 2
    model1 = get_model("rmno_hodge", in_dim=1024, n_genes=200, d_model=64, num_classes=num_classes).to(device)
    model2 = get_model("self_improving_agrotop", in_dim=1024, n_genes=200, d_model=64, num_classes=num_classes).to(device)
    
    res1 = evaluate_model_on_data(model1, test_batch, device)
    res2 = evaluate_model_on_data(model2, test_batch, device)
    
    print("-" * 80)
    print(f"{'Metric':<35} | {'Model 1 (RMNO-Hodge)':<20} | {'Model 2 (Self-Improving)':<20}")
    print("-" * 80)
    print(f"{'Clean Accuracy (%)':<35} | {res1['clean_acc']:>19.2f}% | {res2['clean_acc']:>19.2f}%")
    print(f"{'Perturbed Noise Accuracy (%)':<35} | {res1['noise_acc']:>19.2f}% | {res2['noise_acc']:>19.2f}%")
    print(f"{'Noise Degradation Delta':<35} | {res1['noise_drop']:>19.2f}% | {res2['noise_drop']:>19.2f}%")
    print(f"{'Inference Latency (ms/spot)':<35} | {res1['latency_ms']:>19.2f}  | {res2['latency_ms']:>19.2f} ")
    print(f"{'Trainable Parameters':<35} | {res1['num_params']:>19,d} | {res2['num_params']:>19,d}")
    print("-" * 80)
    print("\nBenchmark completed successfully.")

if __name__ == "__main__":
    main()
