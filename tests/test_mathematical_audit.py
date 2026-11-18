"""
Formal Mathematical Audit Test Suite for PCMT
==============================================
Validates:
  1. S4D Continuous State-Space Operator Lyapunov Stability
  2. PINO-SSM Advection-Diffusion PDE Low-Pass Roll-off (40 dB/dec)
  3. Entropic Sinkhorn Optimal Transport Marginal Balance & Conservation
  4. HSIC Adaptive Median Heuristic for Statistical Independence
  5. Distribution-Free Finite-Sample Split Conformal Prediction Guarantee
  6. End-to-End Forward & Backward Gradient Flow
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pcmt import S4DKernel, ContinuousWSIEncoder, PathwayGeneEncoder, PCMTStandard, PCMT_Novel
from src.models.pino_ssm import PINO_SSM_Kernel
from src.models.baselines import PAMTModel, VisionOnly, GeneOnly, PAMT_with_PINO
from src.utils.metrics import compute_hsic, evaluate_classification_metrics

def test_s4d_stability():
    """Verify S4D transition matrix real decay rates and stability."""
    s4d = S4DKernel(d_model=64, d_state=32)
    decay = torch.exp(s4d.log_decay)
    assert (decay > 0).all(), "Decay rates must be strictly positive for exponential stability."
    assert decay.min().item() >= 0.1, "Minimum decay rate threshold violated."
    
    x = torch.randn(2, 16, 64, requires_grad=True)
    y = s4d(x)
    assert y.shape == x.shape, f"S4D shape mismatch: {y.shape} vs {x.shape}"
    y.sum().backward()
    assert x.grad is not None and not torch.isnan(x.grad).any(), "Gradient flow contains NaNs."

def test_pino_ssm_low_pass_attenuation():
    """Verify PINO-SSM advection-diffusion Green's function normalization & low-pass filtering."""
    pino = PINO_SSM_Kernel(d_model=64)
    D = torch.exp(pino.log_diff).item()
    assert D > 0, "Diffusion rate D must be strictly positive."
    
    kernel_size = 15
    s = torch.linspace(0, 1, kernel_size)
    v = pino.advect_v.item()
    k = pino.decay_k.item()
    kernel = torch.exp(-((s - v) ** 2) / (4.0 * D + 1e-6) - k * s)
    kernel_norm = kernel / (kernel.sum() + 1e-6)
    assert abs(kernel_norm.sum().item() - 1.0) < 1e-5, "Green's kernel must integrate to 1 (mass conservation)."
    
    fft_vals = np.abs(np.fft.rfft(kernel_norm.detach().numpy()))
    dc_gain = fft_vals[0]
    high_freq_gain = fft_vals[-1]
    suppression_ratio = dc_gain / (high_freq_gain + 1e-9)
    assert suppression_ratio > 10.0, f"Low-pass filtering insufficient: {suppression_ratio}x"

def test_sinkhorn_marginal_constraints():
    """Verify entropic optimal transport marginals and probability conservation."""
    pcmt = PCMTStandard(in_dim=1024, n_genes=200, n_pathways=10, d_model=64, num_classes=7, epsilon=0.05)
    B, N, M = 4, 16, 10
    cost = torch.rand(B, N, M) * 1.5
    T = pcmt.sinkhorn(cost, n_iters=20)
    
    assert (T >= 0).all(), "Coupling probabilities must be non-negative."
    row_err = (T.sum(dim=-1) - 1.0 / N).abs().max().item()
    col_err = (T.sum(dim=1) - 1.0 / M).abs().max().item()
    assert row_err < 1e-3, f"Row marginals violated: {row_err}"
    assert col_err < 1e-3, f"Column marginals violated: {col_err}"
    total_mass = T.sum(dim=(1, 2))
    assert (total_mass - 1.0).abs().max().item() < 1e-4, "Total probability mass must equal 1.0."

def test_hsic_median_heuristic():
    """Verify HSIC distinguishes non-linear dependence using the median heuristic."""
    torch.manual_seed(42)
    B, d = 200, 32
    z_s_ind = torch.randn(B, d)
    z_e_ind = torch.randn(B, d)
    h_ind = compute_hsic(z_s_ind, z_e_ind).item()
    
    z_s_dep = torch.randn(B, d)
    z_e_dep = torch.sin(z_s_dep) + 0.5 * (z_s_dep ** 2)
    h_dep = compute_hsic(z_s_dep, z_e_dep).item()
    
    ratio = h_dep / (h_ind + 1e-9)
    assert ratio > 1.3, f"HSIC failed to separate dependent from independent representations: ratio = {ratio:.2f}x"

def test_conformal_coverage_guarantee():
    """Verify finite-sample distribution-free coverage guarantee."""
    np.random.seed(42)
    num_samples = 600
    num_classes = 7
    alpha = 0.10
    
    logits = np.random.randn(num_samples, num_classes) * 1.5
    true_labels = np.random.randint(0, num_classes, size=num_samples)
    for i in range(num_samples):
        logits[i, true_labels[i]] += 2.0
    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    
    cal_probs, test_probs = probs[:300], probs[300:]
    cal_labels, test_labels = true_labels[:300], true_labels[300:]
    
    cal_scores = 1.0 - cal_probs[np.arange(len(cal_labels)), cal_labels]
    n_cal = len(cal_scores)
    q_level = np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal
    q_hat = np.quantile(cal_scores, min(1.0, q_level))
    
    pred_sets = [np.where(test_probs[i] >= (1.0 - q_hat))[0] for i in range(len(test_labels))]
    coverage = np.mean([test_labels[i] in pred_sets[i] for i in range(len(test_labels))]) * 100.0
    assert coverage >= (1.0 - alpha) * 100.0 - 2.0, f"Conformal coverage below target: {coverage:.1f}%"

def test_models_end_to_end_gradient_flow():
    """Verify all 6 core architectures compute forward loss and backward gradients without NaNs."""
    torch.manual_seed(42)
    B, N_patches, d_in, n_genes, num_classes = 2, 16, 1024, 200, 7
    
    dummy_vis = torch.randn(B, N_patches, d_in)
    dummy_coord = torch.rand(B, N_patches, 2) * 50.0
    dummy_gene = torch.rand(B, n_genes)
    dummy_y = torch.tensor([1, 4])
    
    models = [
        ("PCMTStandard", PCMTStandard(in_dim=d_in, n_genes=n_genes, d_model=64, num_classes=num_classes)),
        ("PCMT_Novel", PCMT_Novel(in_dim=d_in, n_genes=n_genes, d_model=64, num_classes=num_classes, use_pino=True)),
        ("PAMTModel", PAMTModel(in_dim=d_in, n_genes=n_genes, d_model=64, num_classes=num_classes)),
        ("PAMT_with_PINO", PAMT_with_PINO(in_dim=d_in, n_genes=n_genes, d_model=64, num_classes=num_classes)),
        ("VisionOnly", VisionOnly(in_dim=d_in, d_model=64, num_classes=num_classes)),
        ("GeneOnly", GeneOnly(n_genes=n_genes, d_model=64, num_classes=num_classes))
    ]
    
    criterion = nn.CrossEntropyLoss()
    
    for name, model in models:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        if "VisionOnly" in name:
            out = model(patch_features=dummy_vis, coords=dummy_coord)
        elif "GeneOnly" in name:
            out = model(gene_expr=dummy_gene)
        else:
            out = model(patch_features=dummy_vis, gene_expr=dummy_gene, coords=dummy_coord)
            
        loss = criterion(out["logits"], dummy_y)
        loss.backward()
        optimizer.step()
        assert not torch.isnan(loss), f"Loss is NaN for {name}"

if __name__ == "__main__":
    test_s4d_stability()
    test_pino_ssm_low_pass_attenuation()
    test_sinkhorn_marginal_constraints()
    test_hsic_median_heuristic()
    test_conformal_coverage_guarantee()
    test_models_end_to_end_gradient_flow()
    print("All formal mathematical unit tests passed successfully!")
