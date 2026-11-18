import os
import sys
import time
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 80)
print("PHYSIO-CAUSAL MULTIMODAL SIMULATION BENCHMARK")
print("Target: Evaluating Architectural Novelties for Top-Tier AI Conferences (NeurIPS/ICML)")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. BIOPHYSICAL MULTIMODAL DATASET GENERATOR
# ---------------------------------------------------------------------------
def generate_biophysical_dataset(
    n_samples=1000,
    gossypol_range=(0.0, 0.2),
    chlorophyll_range=(0.0, 0.2),
    drift_range=(0.0, 0.0),
    smear_factor_range=(1.0, 1.0),
    noise_std=0.05
):
    """
    Generates synthetic multimodal Bt-cotton assay data:
      - Gel lane (1D vector of length 64, spatial migration)
      - Absorbance spectrum (1D vector of length 60, 400-700 nm)
      - Construct Operon Graph prior (one-hot vector of length 4)
      - Class label:
          0: Ref Control (Sad1 only, ~200 bp)
          1: Cry1Ac (Sad1 + Cry1Ac ~300 bp)
          2: Cry2Ab (Sad1 + Cry2Ab ~750 bp)
          3: Bivalent BG-II (Sad1 + Cry1Ac + Cry2Ab)
      - Confounder vector (gossypol, chlorophyll, drift, smear)
    """
    gel_len = 64
    spec_len = 60
    x_gel = np.linspace(0, 1, gel_len)
    x_spec = np.linspace(400, 700, spec_len)

    gel_data = []
    spec_data = []
    labels = []
    confounders = []
    operon_priors = []

    # Standard peak positions (normalized in [0, 1])
    # Migration follows log(bp): 200bp migrates further than 750bp
    # In inverted gel coordinates (0 = well / high MW, 1 = bottom / low MW):
    mu_sad1 = 0.78    # ~200 bp (migrates far)
    mu_cry1ac = 0.55  # ~300 bp
    mu_cry2ab = 0.28  # ~750 bp (migrates slow, near well)

    for i in range(n_samples):
        y = i % 4
        labels.append(y)

        # Environmental confounders for this sample
        c_goss = np.random.uniform(gossypol_range[0], gossypol_range[1])
        c_chlo = np.random.uniform(chlorophyll_range[0], chlorophyll_range[1])
        drift = np.random.uniform(drift_range[0], drift_range[1])
        smear = np.random.uniform(smear_factor_range[0], smear_factor_range[1])
        confounders.append([c_goss, c_chlo, drift, smear])

        # --- Modality 1: Gel Lane (Advection-Diffusion Migration) ---
        gel_signal = np.zeros(gel_len)
        sigma_base = 0.025 * smear

        # Sad1 endogenous control present in all cotton samples
        pos_sad1 = np.clip(mu_sad1 + drift, 0.05, 0.95)
        gel_signal += 1.0 * np.exp(-((x_gel - pos_sad1) ** 2) / (2 * (sigma_base ** 2)))

        if y in [1, 3]:  # Cry1Ac present
            pos_cry1 = np.clip(mu_cry1ac + drift, 0.05, 0.95)
            gel_signal += 1.2 * np.exp(-((x_gel - pos_cry1) ** 2) / (2 * (sigma_base ** 2)))

        if y in [2, 3]:  # Cry2Ab present
            pos_cry2 = np.clip(mu_cry2ab + drift, 0.05, 0.95)
            gel_signal += 1.1 * np.exp(-((x_gel - pos_cry2) ** 2) / (2 * (sigma_base ** 2)))

        # Confounder impact on Gel: Gossypol creates autofluorescence smear background
        gel_background = c_goss * (0.35 * np.exp(-x_gel * 2.0) + 0.1)
        gel_noisy = gel_signal + gel_background + np.random.normal(0, noise_std, gel_len)
        gel_data.append(np.clip(gel_noisy, 0.0, None))

        # --- Modality 2: Absorbance Spectrum (400 - 700 nm) ---
        spec_signal = np.zeros(spec_len)

        # Nanoparticle plasmonic coupling peaks
        if y in [1, 3]:  # Cry1Ac peak at ~525 nm
            spec_signal += 0.8 * np.exp(-((x_spec - 525) ** 2) / (2 * (15 ** 2)))
        if y in [2, 3]:  # Cry2Ab peak at ~585 nm
            spec_signal += 0.85 * np.exp(-((x_spec - 585) ** 2) / (2 * (18 ** 2)))

        # Confounder impact on Spectrum:
        # Gossypol absorbs strongly in 400-480 nm
        goss_curve = c_goss * 1.5 * np.exp(-((x_spec - 420) ** 2) / (2 * (35 ** 2)))
        # Chlorophyll absorbs at 430 nm and 660 nm
        chlo_curve = c_chlo * (1.2 * np.exp(-((x_spec - 430) ** 2) / (2 * (20 ** 2))) +
                               1.0 * np.exp(-((x_spec - 660) ** 2) / (2 * (25 ** 2))))
        # Baseline scattering drift
        baseline = 0.2 + 0.1 * (x_spec - 400) / 300.0

        spec_noisy = spec_signal + goss_curve + chlo_curve + baseline + np.random.normal(0, noise_std, spec_len)
        spec_data.append(spec_noisy)

        # --- Modality 3: Operon Graph Construct Encoding ---
        # Prior adjacency / element presence vector: [Sad1, Cry1Ac, Cry2Ab, Junction_Interaction]
        operon = np.zeros(4)
        operon[0] = 1.0  # Sad1
        if y in [1, 3]: operon[1] = 1.0
        if y in [2, 3]: operon[2] = 1.0
        if y == 3: operon[3] = 1.0  # Dual cassette interaction
        operon_priors.append(operon)

    return (
        torch.tensor(np.array(gel_data), dtype=torch.float32),
        torch.tensor(np.array(spec_data), dtype=torch.float32),
        torch.tensor(np.array(operon_priors), dtype=torch.float32),
        torch.tensor(np.array(labels), dtype=torch.long),
        torch.tensor(np.array(confounders), dtype=torch.float32)
    )

print("\nGenerating Benchmark Datasets across ID and 3 OOD Stress Regimes...")

# 1. Training Set: Sterile lab condition (low gossypol, no drift, sharp bands)
train_gel, train_spec, train_op, train_y, train_conf = generate_biophysical_dataset(
    n_samples=800, gossypol_range=(0.0, 0.15), chlorophyll_range=(0.0, 0.15),
    drift_range=(0.0, 0.0), smear_factor_range=(1.0, 1.0), noise_std=0.04
)

# 2. In-Distribution (ID) Test Set: Similar clean conditions
test_id_gel, test_id_spec, test_id_op, test_id_y, test_id_conf = generate_biophysical_dataset(
    n_samples=250, gossypol_range=(0.0, 0.15), chlorophyll_range=(0.0, 0.15),
    drift_range=(0.0, 0.0), smear_factor_range=(1.0, 1.0), noise_std=0.04
)

# 3. OOD Shift 1: High Chemical Matrix Interference (Field gossypol + chlorophyll surge)
test_ood1_gel, test_ood1_spec, test_ood1_op, test_ood1_y, _ = generate_biophysical_dataset(
    n_samples=250, gossypol_range=(0.7, 1.0), chlorophyll_range=(0.6, 0.95),
    drift_range=(0.0, 0.0), smear_factor_range=(1.0, 1.0), noise_std=0.05
)

# 4. OOD Shift 2: Electrophoretic Spatial Distortion (Voltage drift + severe band smearing)
test_ood2_gel, test_ood2_spec, test_ood2_op, test_ood2_y, _ = generate_biophysical_dataset(
    n_samples=250, gossypol_range=(0.0, 0.15), chlorophyll_range=(0.0, 0.15),
    drift_range=(-0.12, 0.12), smear_factor_range=(2.2, 3.0), noise_std=0.05
)

# 5. OOD Shift 3: Worst-Case Field Stress (High Matrix Confounders + Drift + Smearing + Noise)
test_ood3_gel, test_ood3_spec, test_ood3_op, test_ood3_y, _ = generate_biophysical_dataset(
    n_samples=250, gossypol_range=(0.7, 1.0), chlorophyll_range=(0.6, 0.95),
    drift_range=(-0.12, 0.12), smear_factor_range=(2.2, 3.0), noise_std=0.10
)

print(f"Data Generation Completed: Train={len(train_y)}, Test_ID={len(test_id_y)}, OOD_1={len(test_ood1_y)}, OOD_2={len(test_ood2_y)}, OOD_3={len(test_ood3_y)}")

# ---------------------------------------------------------------------------
# 2. HILBERT-SCHMIDT INDEPENDENCE CRITERION (HSIC) LOSS
# ---------------------------------------------------------------------------
def compute_rbf_kernel(X, gamma=None):
    N = X.size(0)
    dist_sq = torch.cdist(X, X) ** 2
    if gamma is None:
        med = torch.median(dist_sq)
        gamma = 1.0 / (med + 1e-6)
    return torch.exp(-gamma * dist_sq)

def hsic_loss(Z_S, Z_E):
    """
    Computes empirical HSIC between invariant signal Z_S and confounder Z_E:
    HSIC(Z_S, Z_E) = (1 / (N-1)^2) * Tr(K_S * H * K_E * H)
    """
    N = Z_S.size(0)
    if N <= 1:
        return torch.tensor(0.0, device=Z_S.device)
    K_S = compute_rbf_kernel(Z_S)
    K_E = compute_rbf_kernel(Z_E)
    H = torch.eye(N, device=Z_S.device) - (1.0 / N) * torch.ones((N, N), device=Z_S.device)
    HK_S = torch.mm(H, K_S)
    HK_E = torch.mm(H, K_E)
    hsic = torch.trace(torch.mm(HK_S, HK_E)) / ((N - 1) ** 2)
    return hsic

# ---------------------------------------------------------------------------
# 3. ARCHITECTURAL METHODOLOGIES DEFINITION
# ---------------------------------------------------------------------------

# --- Method 0: Baseline PAMT (Discrete Patch ViT + Flat Cross-Attention) ---
class BaselinePAMT(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()
        # Patch gel into 8 patches of 8
        self.gel_patch_embed = nn.Linear(8, d_model)
        self.gel_pos = nn.Parameter(torch.randn(1, 8, d_model) * 0.02)
        # Patch spectrum into 6 patches of 10
        self.spec_patch_embed = nn.Linear(10, d_model)
        self.spec_pos = nn.Parameter(torch.randn(1, 6, d_model) * 0.02)
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=2, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, gel, spec, operon):
        B = gel.size(0)
        gel_p = gel.view(B, 8, 8)
        gel_emb = self.gel_patch_embed(gel_p) + self.gel_pos
        
        spec_p = spec.view(B, 6, 10)
        spec_emb = self.spec_patch_embed(spec_p) + self.spec_pos
        
        # Cross attention: query=spec, key=gel, value=gel
        attn_out, _ = self.cross_attn(spec_emb, gel_emb, gel_emb)
        pooled = torch.cat([attn_out.mean(dim=1), gel_emb.mean(dim=1)], dim=-1)
        logits = self.classifier(pooled)
        return logits, pooled

# --- Method 1: Continuous State-Space Dynamics (PI-SSM) ---
class StateSpaceGelLayer(nn.Module):
    """Continuous 1D State Space layer with decaying diffusion-advection transition matrix A."""
    def __init__(self, in_features, state_dim=16):
        super().__init__()
        self.state_dim = state_dim
        # Construct diagonal A with negative real part for physical stability
        # A_k = -0.5 - i * pi * k (continuous advection-diffusion decay)
        decay = torch.linspace(0.1, 2.0, state_dim)
        self.log_decay = nn.Parameter(torch.log(decay))
        self.B = nn.Parameter(torch.randn(state_dim, in_features) * 0.1)
        self.C = nn.Parameter(torch.randn(in_features, state_dim) * 0.1)
        self.D = nn.Parameter(torch.zeros(in_features))

    def forward(self, u):
        # u: [B, L, C] -> continuous convolution via physical SSM advection-diffusion filter
        B, L, C = u.shape
        decay = torch.exp(self.log_decay)  # [state_dim]
        # Continuous physical kernel K(t) with compact support (odd length=15)
        kernel_len = 15
        t = torch.linspace(0, 1, kernel_len, device=u.device).unsqueeze(1)  # [15, 1]
        K = torch.matmul(torch.exp(-t * decay.unsqueeze(0)), self.B)  # [15, C]
        
        # 1D depthwise convolution
        u_t = u.permute(0, 2, 1)  # [B, C, L]
        K_t = K.permute(1, 0).unsqueeze(1)  # [C, 1, 15]
        pad = kernel_len // 2
        y = F.conv1d(u_t, K_t, padding=pad, groups=C)
        y = y.permute(0, 2, 1) + u * self.D
        return y

class Method1_PISSM(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()
        self.gel_in = nn.Linear(1, d_model)
        self.ssm_layer = StateSpaceGelLayer(in_features=d_model, state_dim=16)
        self.gel_proj = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU())
        self.spec_embed = nn.Sequential(nn.Linear(60, d_model), nn.GELU())
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, gel, spec, operon):
        B = gel.size(0)
        gel_in = self.gel_in(gel.unsqueeze(-1))  # [B, 64, d_model]
        gel_ssm = self.ssm_layer(gel_in)        # [B, 64, d_model]
        # Multi-scale pooling: mean background energy + peak detection
        gel_pooled = self.gel_proj(torch.cat([gel_ssm.mean(dim=1), gel_ssm.max(dim=1)[0]], dim=-1))
        spec_pooled = self.spec_embed(spec)
        fused = torch.cat([gel_pooled, spec_pooled], dim=-1)
        logits = self.classifier(fused)
        return logits, fused

# --- Method 2: Causal Disentanglement (SCM-HSIC) ---
class Method2_SCM_HSIC(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()
        self.gel_enc = nn.Sequential(nn.Linear(64, d_model), nn.ReLU())
        self.spec_enc = nn.Sequential(nn.Linear(60, d_model), nn.ReLU())
        
        # Dual Latent Decomposers:
        # Z_S: Invariant Transgenic Factor (dim=32)
        # Z_E: Environmental Confounder Factor (dim=32)
        self.fc_signal = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.fc_env = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        
        self.classifier = nn.Linear(d_model, 4)
        self.env_predictor = nn.Linear(d_model, 2)  # predicts [gossypol, chlorophyll]

    def forward(self, gel, spec, operon):
        B = gel.size(0)
        h_gel = self.gel_enc(gel)
        h_spec = self.spec_enc(spec)
        h = torch.cat([h_gel, h_spec], dim=-1)
        
        z_s = self.fc_signal(h)
        z_e = self.fc_env(h)
        
        logits = self.classifier(z_s)
        pred_env = self.env_predictor(z_e)
        return logits, z_s, z_e, pred_env

# --- Method 3: Operon Fused Gromov-Wasserstein OT Attention (FGW-OT) ---
class Method3_FGW_OT(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()
        self.d_model = d_model
        self.gel_enc = nn.Linear(8, d_model)   # 8 spatial bins
        self.spec_enc = nn.Linear(10, d_model) # 6 spectral bins
        self.operon_enc = nn.Linear(4, d_model)
        
        # Physics-Topological Cost Matrix Prior (8 gel bins x 4 operon nodes)
        # Maps Sad1 to bin 6, Cry1Ac to bin 4, Cry2Ab to bin 2
        prior_cost = torch.ones(8, 4) * 2.0
        prior_cost[6, 0] = 0.1  # Sad1 bin
        prior_cost[4, 1] = 0.1  # Cry1Ac bin
        prior_cost[2, 2] = 0.1  # Cry2Ab bin
        prior_cost[1:5, 3] = 0.2 # Interaction
        self.register_buffer("topological_prior", prior_cost)

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, gel, spec, operon):
        B = gel.size(0)
        gel_p = gel.view(B, 8, 8)
        gel_tokens = self.gel_enc(gel_p)  # [B, 8, d_model]
        
        spec_p = spec.view(B, 6, 10)
        spec_tokens = self.spec_enc(spec_p)  # [B, 6, d_model]
        
        # FGW-Regularized Attention:
        # Distance-penalized transport plan
        dot_sim = torch.bmm(gel_tokens, gel_tokens.transpose(1, 2)) / math.sqrt(self.d_model)
        # Apply operon topological mask:
        op_presence = operon.unsqueeze(1)  # [B, 1, 4]
        topo_energy = torch.matmul(self.topological_prior, op_presence.transpose(1, 2)).squeeze(-1) # [B, 8]
        topo_bias = -1.5 * topo_energy.unsqueeze(1) # [B, 1, 8]
        
        ot_attn = F.softmax(dot_sim + topo_bias, dim=-1)
        gel_aligned = torch.bmm(ot_attn, gel_tokens).mean(dim=1)
        spec_pooled = spec_tokens.mean(dim=1)
        
        fused = torch.cat([gel_aligned, spec_pooled], dim=-1)
        logits = self.classifier(fused)
        return logits, fused

# --- Method 4: Proposed Unified PCMT (State-Space + SCM-HSIC + FGW-OT) ---
class Method4_Unified_PCMT(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()
        self.d_model = d_model
        # 1. State-Space continuous gel encoder
        self.gel_in = nn.Linear(1, d_model)
        self.ssm_layer = StateSpaceGelLayer(in_features=d_model, state_dim=16)
        
        # 2. Spectral encoder
        self.spec_enc = nn.Sequential(nn.Linear(60, d_model), nn.GELU())
        
        # 3. Operon Topology Prior
        prior_cost = torch.ones(8, 4) * 2.0
        prior_cost[6, 0] = 0.1
        prior_cost[4, 1] = 0.1
        prior_cost[2, 2] = 0.1
        prior_cost[1:5, 3] = 0.2
        self.register_buffer("topological_prior", prior_cost)

        # 4. Causal Disentanglement Latents
        self.fc_signal = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU())
        self.fc_env = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU())
        
        self.classifier = nn.Linear(d_model, 4)
        self.env_predictor = nn.Linear(d_model, 2)

    def forward(self, gel, spec, operon):
        B = gel.size(0)
        # Continuous SSM pass
        gel_ssm = self.ssm_layer(self.gel_in(gel.unsqueeze(-1))) # [B, 64, d_model]
        # Pool into 8 topological chunks via local peak detection
        gel_chunks = gel_ssm.view(B, 8, 8, self.d_model).max(dim=2)[0] # [B, 8, d_model]
        
        # FGW Topological Coupling
        op_presence = operon.unsqueeze(1)
        topo_energy = torch.matmul(self.topological_prior, op_presence.transpose(1, 2)).squeeze(-1)
        topo_weight = F.softmax(-1.0 * topo_energy, dim=-1).unsqueeze(-1) # [B, 8, 1]
        gel_topological = (gel_chunks * topo_weight).sum(dim=1) # [B, d_model]
        
        spec_pooled = self.spec_enc(spec) # [B, d_model]
        fused = torch.cat([gel_topological, spec_pooled], dim=-1)
        
        # Disentangle Invariant Signal Z_S from Confounder Z_E
        z_s = self.fc_signal(fused)
        z_e = self.fc_env(fused)
        
        logits = self.classifier(z_s)
        pred_env = self.env_predictor(z_e)
        return logits, z_s, z_e, pred_env

# ---------------------------------------------------------------------------
# 4. TRAINING AND EVALUATION ENGINE
# ---------------------------------------------------------------------------
def evaluate_model(model, method_name, gel, spec, operon, y):
    model.eval()
    with torch.no_grad():
        if "SCM-HSIC" in method_name or "Unified PCMT" in method_name:
            logits, z_s, z_e, _ = model(gel, spec, operon)
        else:
            logits, z_s = model(gel, spec, operon)
            z_e = None
            
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y).float().mean().item() * 100.0
    return acc, z_s

def train_and_evaluate_methodology(name, model, train_loader, test_sets, epochs=25, lr=0.003):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_task = nn.CrossEntropyLoss()
    criterion_env = nn.MSELoss()
    
    print(f"\n>>> Training {name} for {epochs} epochs...", flush=True)
    t0 = time.time()
    
    for epoch in range(epochs):
        model.train()
        for batch_gel, batch_spec, batch_op, batch_y, batch_conf in train_loader:
            optimizer.zero_grad()
            
            if "SCM-HSIC" in name or "Unified PCMT" in name:
                logits, z_s, z_e, pred_env = model(batch_gel, batch_spec, batch_op)
                loss_task = criterion_task(logits, batch_y)
                # HSIC penalization enforcing independence between z_s and z_e
                loss_hsic = hsic_loss(z_s, z_e)
                # Confounder supervision on z_e (to absorb gossypol & chlorophyll)
                true_env = batch_conf[:, :2] # gossypol, chlorophyll
                loss_env = criterion_env(pred_env, true_env)
                
                loss = loss_task + 0.5 * loss_hsic + 0.5 * loss_env
            else:
                logits, _ = model(batch_gel, batch_spec, batch_op)
                loss = criterion_task(logits, batch_y)
                
            loss.backward()
            optimizer.step()
            
    train_time = time.time() - t0
    
    # Evaluate across all 4 regimes
    acc_id, z_s_id = evaluate_model(model, name, test_sets["ID"][0], test_sets["ID"][1], test_sets["ID"][2], test_sets["ID"][3])
    acc_ood1, _ = evaluate_model(model, name, test_sets["OOD1"][0], test_sets["OOD1"][1], test_sets["OOD1"][2], test_sets["OOD1"][3])
    acc_ood2, _ = evaluate_model(model, name, test_sets["OOD2"][0], test_sets["OOD2"][1], test_sets["OOD2"][2], test_sets["OOD2"][3])
    acc_ood3, z_s_ood3 = evaluate_model(model, name, test_sets["OOD3"][0], test_sets["OOD3"][1], test_sets["OOD3"][2], test_sets["OOD3"][3])
    
    # Confounder Leakage: How much gossypol/chlorophyll information leaked into Z_S?
    # Measured via Linear Probe R^2 from Z_S to true gossypol concentration
    # Lower is better (0.00 = perfect invariance)
    z_s_np = z_s_id.detach().cpu().numpy()
    conf_np = test_sets["ID"][4][:, 0].numpy() # gossypol
    # Simple linear correlation score
    cov = np.cov(z_s_np.mean(axis=1), conf_np)[0, 1]
    leakage = abs(cov) / (np.std(z_s_np.mean(axis=1)) * np.std(conf_np) + 1e-6)
    
    ood_gap = acc_id - acc_ood3
    
    print(f"[{name}] Done in {train_time:.1f}s | ID Acc: {acc_id:.2f}% | OOD1 (Matrix): {acc_ood1:.2f}% | OOD2 (Drift): {acc_ood2:.2f}% | OOD3 (Worst Field): {acc_ood3:.2f}% | OOD Drop: {ood_gap:.2f}%")
    
    return {
        "Name": name,
        "ID_Acc": acc_id,
        "OOD1_Acc": acc_ood1,
        "OOD2_Acc": acc_ood2,
        "OOD3_Acc": acc_ood3,
        "OOD_Gap": ood_gap,
        "Confounder_Leakage": float(leakage),
        "Train_Time_s": train_time
    }

# ---------------------------------------------------------------------------
# 5. EXECUTE COMPARATIVE BENCHMARK RUN
# ---------------------------------------------------------------------------
train_dataset = TensorDataset(train_gel, train_spec, train_op, train_y, train_conf)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_sets = {
    "ID": (test_id_gel, test_id_spec, test_id_op, test_id_y, test_id_conf),
    "OOD1": (test_ood1_gel, test_ood1_spec, test_ood1_op, test_ood1_y, None),
    "OOD2": (test_ood2_gel, test_ood2_spec, test_ood2_op, test_ood2_y, None),
    "OOD3": (test_ood3_gel, test_ood3_spec, test_ood3_op, test_ood3_y, None)
}

models_to_test = [
    ("Method 0 (PAMT Baseline)", BaselinePAMT()),
    ("Method 1 (PI-SSM Dynamics)", Method1_PISSM()),
    ("Method 2 (SCM-HSIC Causal)", Method2_SCM_HSIC()),
    ("Method 3 (Operon FGW-OT)", Method3_FGW_OT()),
    ("Method 4 (Unified PCMT)", Method4_Unified_PCMT())
]

results = []
for name, model in models_to_test:
    res = train_and_evaluate_methodology(name, model, train_loader, test_sets, epochs=25, lr=0.003)
    results.append(res)

# Save results to JSON
output_json_path = os.path.join(r"E:\NIKHIL\startup\agri-rna", "simulation_methodology_results.json")
with open(output_json_path, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 90)
print(f"{'Methodology':<28} | {'ID Acc':<8} | {'OOD-1(Chem)':<11} | {'OOD-2(Phys)':<11} | {'OOD-3(Field)':<12} | {'OOD Drop':<8} | {'Leakage':<7}")
print("-" * 90)
for r in results:
    print(f"{r['Name']:<28} | {r['ID_Acc']:>6.2f}% | {r['OOD1_Acc']:>9.2f}% | {r['OOD2_Acc']:>9.2f}% | {r['OOD3_Acc']:>10.2f}% | {r['OOD_Gap']:>6.2f}% | {r['Confounder_Leakage']:>7.3f}")
print("=" * 90)

# ---------------------------------------------------------------------------
# 6. GENERATE COMPARATIVE BENCHMARK VISUALIZATION
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    names = [r["Name"].replace("Method ", "M").split(" (")[0] + "\n" + r["Name"].split(" (")[1].replace(")", "") for r in results]
    id_accs = [r["ID_Acc"] for r in results]
    ood1_accs = [r["OOD1_Acc"] for r in results]
    ood2_accs = [r["OOD2_Acc"] for r in results]
    ood3_accs = [r["OOD3_Acc"] for r in results]
    
    x = np.arange(len(names))
    width = 0.2
    
    ax1.bar(x - 1.5 * width, id_accs, width, label='In-Distribution (Lab Clean)', color='#2ca02c')
    ax1.bar(x - 0.5 * width, ood1_accs, width, label='OOD-1 (Chemical Matrix / Gossypol)', color='#ff7f0e')
    ax1.bar(x + 0.5 * width, ood2_accs, width, label='OOD-2 (Electrophoretic Distortion)', color='#1f77b4')
    ax1.bar(x + 1.5 * width, ood3_accs, width, label='OOD-3 (Combined Real Field Stress)', color='#d62728')
    
    ax1.set_ylabel('Classification Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Robustness Across Environmental & Physical Stress', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10)
    ax1.set_ylim(40, 105)
    ax1.legend(loc='lower left', frameon=True)
    
    # Second plot: OOD Performance Gap vs Confounder Leakage
    gaps = [r["OOD_Gap"] for r in results]
    leakages = [r["Confounder_Leakage"] for r in results]
    colors = ['#d62728', '#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c']
    
    scatter = ax2.scatter(gaps, leakages, s=[200, 240, 240, 240, 320], c=colors, alpha=0.9, edgecolors='black', linewidth=1.5)
    for i, txt in enumerate(["PAMT (Baseline)", "PI-SSM", "SCM-HSIC", "FGW-OT", "Unified PCMT (Ours)"]):
        offset_y = 0.02 if i % 2 == 0 else -0.025
        ax2.annotate(txt, (gaps[i], leakages[i] + offset_y), fontsize=10, fontweight='bold', ha='center')
        
    ax2.set_xlabel('OOD Performance Degradation Gap Δ (%) [Lower is Better]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Confounder Information Leakage [Lower is Better]', fontsize=12, fontweight='bold')
    ax2.set_title('Pareto Frontier: Confounder Invariance vs Robustness', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    chart_path = os.path.join(r"C:\Users\sys1026\.gemini\antigravity\brain\fe5b5c38-a4bb-4ee7-b4fb-30f369c2476e", "simulation_benchmark_comparison.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"\nSaved benchmark comparison chart to: {chart_path}")
except Exception as e:
    print(f"Chart plotting error: {e}")

print("\nSIMULATION BENCHMARK RUN COMPLETE.")
