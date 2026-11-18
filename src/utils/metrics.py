"""
Evaluation Metrics & Biophysical Loss Functions
===============================================
Includes:
  - compute_hsic: Hilbert-Schmidt Independence Criterion for dual-latent causal disentanglement.
  - evaluate_classification_metrics: Multiclass accuracy, macro F1, precision, recall.
  - evaluate_noise_robustness: Out-of-distribution optical scattering and amplification degradation.
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_hsic(z_s: torch.Tensor, z_e: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Computes the Hilbert-Schmidt Independence Criterion (HSIC) using an RBF kernel.
    HSIC = 0 if and only if z_s and z_e are statistically independent.
    
    Args:
        z_s: Invariant biological signal representation [B, d]
        z_e: Environmental confounder representation [B, d]
        sigma: Kernel bandwidth parameter
    """
    B = z_s.size(0)
    if B <= 1:
        return torch.tensor(0.0, device=z_s.device)
        
    H = torch.eye(B, device=z_s.device) - (1.0 / B) * torch.ones((B, B), device=z_s.device)
    
    # RBF Gram matrices
    dist_s = torch.cdist(z_s, z_s) ** 2
    dist_e = torch.cdist(z_e, z_e) ** 2
    
    K_s = torch.exp(-dist_s / (2.0 * (sigma ** 2) + 1e-6))
    K_e = torch.exp(-dist_e / (2.0 * (sigma ** 2) + 1e-6))
    
    HK_s = torch.matmul(H, K_s)
    HK_e = torch.matmul(H, K_e)
    
    hsic_val = torch.trace(torch.matmul(HK_s, HK_e)) / ((B - 1) ** 2 + 1e-6)
    return torch.clamp(hsic_val, min=0.0)

def evaluate_classification_metrics(y_true, y_pred):
    """
    Computes standard multiclass performance metrics.
    """
    acc = accuracy_score(y_true, y_pred) * 100.0
    f1 = f1_score(y_true, y_pred, average="macro") * 100.0
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    return {"accuracy": acc, "macro_f1": f1, "precision": prec, "recall": rec}
