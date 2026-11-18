"""
Extended Epoch Training & Convergence Analysis: PCMT Standard vs Novelty Configurations
========================================================================================
Investigating whether extended training (60 epochs) enables the Physics-Informed (PINO-SSM)
and Causal Conformal models to catch up or surpass Standard PCMT as physical operators converge.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

# Import from run_authentic_benchmark_and_novelty
sys.path.append(r"E:\NIKHIL\startup\agri-rna\PCMT_Project")
from run_authentic_benchmark_and_novelty import (
    load_authentic_plant_data,
    PlantSpatialDataset,
    PCMTStandard,
    PCMT_Novel,
    evaluate_conformal_prediction,
    compute_hsic
)

np.random.seed(42)
torch.manual_seed(42)

def run_extended_training(epochs=60):
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load dataset
    train_data, val_data = load_authentic_plant_data()
    train_ds = PlantSpatialDataset(*train_data)
    val_ds = PlantSpatialDataset(*val_data)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    
    models = {
        "PCMT (Standard, S4D)": PCMTStandard(num_classes=7, epsilon=0.05),
        "PCMT + PINO-SSM (Physics PDE, eps=0.05)": PCMT_Novel(use_pino=True, epsilon=0.05, use_counterfactual=False),
        "PCMT + Full Trio (PINO + Causal, eps=0.05)": PCMT_Novel(use_pino=True, epsilon=0.05, use_counterfactual=True),
        "PCMT + Full Trio (PINO + Soft-OT eps=0.5)": PCMT_Novel(use_pino=True, epsilon=0.5, use_counterfactual=True),
    }
    
    results = {}
    
    for name, model in models.items():
        print("\n" + "="*80)
        print(f" TRAINING: {name} for {epochs} Epochs")
        print("="*80)
        
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        # Cosine annealing scheduler to allow fine convergence in later epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        criterion = nn.CrossEntropyLoss()
        
        train_losses = []
        val_accs = []
        best_acc = 0.0
        
        for ep in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for b_vis, b_coord, b_gene, b_y in train_loader:
                b_vis, b_coord, b_gene, b_y = b_vis.to(device), b_coord.to(device), b_gene.to(device), b_y.to(device)
                optimizer.zero_grad()
                out = model(b_vis, b_gene, coords=b_coord)
                
                task_loss = criterion(out["logits"], b_y)
                if "z_s" in out and "z_e" in out:
                    hsic = compute_hsic(out["z_s"], out["z_e"])
                    loss = task_loss + 0.05 * hsic
                else:
                    loss = task_loss
                    
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(b_y)
                
            scheduler.step()
            total_loss /= len(train_loader.dataset)
            train_losses.append(total_loss)
            
            # Validation
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for b_vis, b_coord, b_gene, b_y in val_loader:
                    b_vis, b_coord, b_gene, b_y = b_vis.to(device), b_coord.to(device), b_gene.to(device), b_y.to(device)
                    out = model(b_vis, b_gene, coords=b_coord)
                    preds.append(out["logits"].argmax(dim=-1).cpu().numpy())
                    targets.append(b_y.cpu().numpy())
                    
            all_preds = np.concatenate(preds)
            all_targets = np.concatenate(targets)
            acc = accuracy_score(all_targets, all_preds) * 100.0
            val_accs.append(acc)
            if acc > best_acc:
                best_acc = acc
                
            if ep % 5 == 0 or ep == 1 or ep == epochs:
                print(f"  [{name[:28]:<28}] Ep {ep:02d}/{epochs:02d} | Loss: {total_loss:.4f} | Val Acc: {acc:.1f}% (Peak: {best_acc:.1f}%)")
                
        # Final evaluation metrics
        f1 = f1_score(all_targets, all_preds, average="macro") * 100.0
        
        # OOD noise test
        ood_preds = []
        with torch.no_grad():
            for b_vis, b_coord, b_gene, b_y in val_loader:
                b_vis, b_coord, b_gene = b_vis.to(device), b_coord.to(device), b_gene.to(device)
                noisy_vis = b_vis + torch.randn_like(b_vis) * 1.0
                noisy_gene = b_gene + torch.randn_like(b_gene) * 1.0
                out = model(noisy_vis, noisy_gene, coords=b_coord)
                ood_preds.append(out["logits"].argmax(dim=-1).cpu().numpy())
        ood_acc = accuracy_score(all_targets, np.concatenate(ood_preds)) * 100.0
        
        # Conformal prediction (90% target coverage)
        cov, set_size = evaluate_conformal_prediction(model, val_loader, alpha=0.1, device=device)
        
        results[name] = {
            "train_losses": train_losses,
            "val_accs": val_accs,
            "final_acc": acc,
            "best_acc": best_acc,
            "f1": f1,
            "ood_acc": ood_acc,
            "noise_drop": acc - ood_acc,
            "coverage": cov,
            "set_size": set_size
        }
        
    print("\n" + "="*110)
    print(f"{'MODEL ARCHITECTURE':<40} | {'FINAL ACC':<10} | {'PEAK ACC':<10} | {'MACRO F1':<10} | {'OOD ACC':<10} | {'CONFORMAL COV':<14}")
    print("="*110)
    for name, r in results.items():
        print(f"{name:<40} | {r['final_acc']:>8.1f}% | {r['best_acc']:>8.1f}% | {r['f1']:>8.1f}% | {r['ood_acc']:>8.1f}% | {r['coverage']:>12.1f}%")
    print("="*110)
    
    # Plot learning curves
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    for name, r in results.items():
        plt.plot(range(1, epochs + 1), r["train_losses"], label=name, lw=2)
    plt.xlabel("Training Epoch", fontsize=11)
    plt.ylabel("Cross-Entropy Loss", fontsize=11)
    plt.title("Extended Training Loss Convergence (60 Epochs)", fontsize=12, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    
    plt.subplot(1, 2, 2)
    for name, r in results.items():
        plt.plot(range(1, epochs + 1), r["val_accs"], label=f"{name} (Peak {r['best_acc']:.1f}%)", lw=2)
    plt.xlabel("Training Epoch", fontsize=11)
    plt.ylabel("7-Domain Validation Accuracy (%)", fontsize=11)
    plt.title("Validation Accuracy Trajectory (60 Epochs)", fontsize=12, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc="lower right")
    
    plt.tight_layout()
    out_img = r"C:\Users\sys1026\.gemini\antigravity\brain\fe5b5c38-a4bb-4ee7-b4fb-30f369c2476e\novelty_extended_epochs_learning_curves.png"
    plt.savefig(out_img, dpi=200)
    plt.close()
    print(f"\nSaved learning curve figure to: {out_img}")
    print(f"Total time taken: {time.time() - t_start:.1f} seconds.")

if __name__ == "__main__":
    run_extended_training(epochs=60)
