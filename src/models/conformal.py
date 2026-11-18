"""
Distribution-Free Split Conformal Prediction
============================================
Provides mathematical coverage guarantee:
  P(Y_{n+1} in C(X_{n+1})) >= 1 - alpha
Without distributional assumptions.
"""

import numpy as np
import torch
import torch.nn.functional as F

def evaluate_conformal_prediction(model, val_loader, alpha=0.1, device="cpu"):
    """
    Split Conformal Prediction on Validation/Test spots.
    
    Args:
        model: Trained PyTorch model
        val_loader: DataLoader containing validation samples
        alpha: Significance error budget (default: 0.1 for 90% confidence)
        device: 'cuda' or 'cpu'
        
    Returns:
        coverage (%): Empirical percentage of test samples whose true label is in prediction set
        avg_set_size: Average number of candidate classes in output set
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for b_vis, b_coord, b_gene, b_y in val_loader:
            b_vis = b_vis.to(device)
            b_coord = b_coord.to(device) if b_coord is not None else None
            b_gene = b_gene.to(device)
            b_y = b_y.to(device)
            
            out = model(b_vis, b_gene, coords=b_coord)
            probs = F.softmax(out["logits"], dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(b_y.cpu().numpy())
            
    probs = np.concatenate(all_probs, axis=0) # [N_val, num_classes]
    labels = np.concatenate(all_labels, axis=0)
    
    # 50/50 Calibration / Test Split
    n_total = len(labels)
    perm = np.random.RandomState(42).permutation(n_total)
    cal_idx = perm[:n_total // 2]
    test_idx = perm[n_total // 2:]
    
    cal_probs = probs[cal_idx]
    cal_labels = labels[cal_idx]
    
    # Non-conformity scores: 1.0 - P(true class)
    cal_scores = 1.0 - cal_probs[np.arange(len(cal_labels)), cal_labels]
    
    # Finite-sample quantile threshold
    q_level = np.ceil((len(cal_scores) + 1) * (1.0 - alpha)) / len(cal_scores)
    q_level = min(1.0, max(0.0, q_level))
    q_hat = np.quantile(cal_scores, q_level)
    
    # Form test prediction sets
    test_probs = probs[test_idx]
    test_labels = labels[test_idx]
    pred_sets = [np.where(test_probs[i] >= (1.0 - q_hat))[0] for i in range(len(test_labels))]
    
    coverage = np.mean([test_labels[i] in pred_sets[i] for i in range(len(test_labels))]) * 100.0
    avg_set_size = np.mean([len(s) for s in pred_sets])
    
    return coverage, avg_set_size
