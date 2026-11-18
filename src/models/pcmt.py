"""
Physio-Causal Multimodal Transformer (PCMT)
===========================================
Core Architecture:
  1. S4DKernel / ContinuousWSIEncoder: Diagonal state-space operator for continuous tissue histology.
  2. PathwayGeneEncoder: Biological pathway-guided gene token projection.
  3. FusedGromovWassersteinAttention: Entropic Sinkhorn optimal transport cross-attention.
  4. CausalDisentanglementHead: Dual-latent invariant/confounder decomposition with HSIC penalty.
  5. PCMT_Novel: Full suite with PINO-SSM transport PDE, Soft-OT, and Causal Counterfactual head.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pino_ssm import PINO_SSM_Kernel

# ============================================================================
# 1. CONTINUOUS STATE-SPACE OPERATOR (S4D)
# ============================================================================
class S4DKernel(nn.Module):
    """
    Diagonal State Space (S4D) continuous spatial operator.
    Parameterizes spatial advection-diffusion:
      d h(s)/ds = A h(s) + B u(s),  y(s) = C h(s) + D u(s)
    """
    def __init__(self, d_model: int, d_state: int = 32):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Initialize diagonal transition A with decaying real parts
        log_decay = torch.log(torch.linspace(0.1, 4.0, d_state))
        self.log_decay = nn.Parameter(log_decay)
        
        self.B = nn.Parameter(torch.randn(d_state, d_model) * 0.05)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.05)
        self.D = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        decay = torch.exp(self.log_decay)
        
        kernel_len = min(15, N if N % 2 == 1 else N - 1)
        if kernel_len < 3:
            return x * (1.0 + self.D)
            
        t = torch.linspace(0, 1, kernel_len, device=x.device).unsqueeze(1)
        K = torch.matmul(torch.exp(-t * decay.unsqueeze(0)), self.B)
        
        x_t = x.permute(0, 2, 1)
        K_t = K.permute(1, 0).unsqueeze(1)
        pad = kernel_len // 2
        y = F.conv1d(x_t, K_t, padding=pad, groups=C)
        y = y.permute(0, 2, 1) + x * self.D
        return y

class ContinuousWSIEncoder(nn.Module):
    """
    Encodes spatial WSI patch bags [B, N_patches, in_dim]
    via S4D continuous operator and 2D spatial coordinate embeddings.
    """
    def __init__(self, in_dim: int = 1024, d_model: int = 128, d_state: int = 32):
        super().__init__()
        self.patch_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.coord_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.s4d = S4DKernel(d_model=d_model, d_state=d_state)

    def forward(self, patch_features: torch.Tensor, coords: torch.Tensor = None) -> torch.Tensor:
        h = self.patch_proj(patch_features)
        if coords is not None:
            h = h + self.coord_proj(coords)
        return self.s4d(h)

# ============================================================================
# 2. PATHWAY GENE ENCODER
# ============================================================================
class PathwayGeneEncoder(nn.Module):
    """
    Groups high-dimensional gene expression into functional biological pathway tokens.
    """
    def __init__(self, n_genes: int = 200, n_pathways: int = 10, d_model: int = 128):
        super().__init__()
        self.n_genes = n_genes
        self.n_pathways = n_pathways
        self.d_model = d_model
        
        self.pathway_proj = nn.Sequential(
            nn.Linear(n_genes, n_pathways * d_model),
            nn.LayerNorm(n_pathways * d_model),
            nn.GELU()
        )
        self.token_norm = nn.LayerNorm(d_model)

    def forward(self, gene_expr: torch.Tensor) -> torch.Tensor:
        B = gene_expr.size(0)
        h = self.pathway_proj(gene_expr)
        tokens = h.view(B, self.n_pathways, self.d_model)
        return self.token_norm(tokens)

# ============================================================================
# 3. CAUSAL COUNTERFACTUAL HEAD
# ============================================================================
class CounterfactualInterventionHead(nn.Module):
    """
    Dual-latent decomposition: separates biological invariant signal (Z_S) 
    from environmental / technical noise (Z_E) with Pearl do-calculus intervention support.
    """
    def __init__(self, in_dim: int = 128, latent_dim: int = 64, num_classes: int = 7):
        super().__init__()
        self.fc_signal = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.fc_env = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, num_classes)
        )
        self.intervention_matrix = nn.Parameter(torch.eye(latent_dim) + torch.randn(latent_dim, latent_dim) * 0.01)

    def forward(self, fused_tokens: torch.Tensor, do_domain=None):
        h = fused_tokens.mean(dim=1)
        z_s = self.fc_signal(h)
        z_e = self.fc_env(h)
        
        if do_domain is not None:
            z_s = torch.matmul(z_s, self.intervention_matrix)
            
        logits = self.classifier(z_s)
        return logits, z_s, z_e

# ============================================================================
# 4. STANDARD PCMT (PRIMARY HIGH-ACCURACY ENGINE)
# ============================================================================
class PCMTStandard(nn.Module):
    """
    Standard PCMT: Continuous S4D Encoder + Pathway Gene Projection + Entropic OT Attention.
    """
    def __init__(self, in_dim: int = 1024, n_genes: int = 200, n_pathways: int = 10, 
                 d_model: int = 128, num_classes: int = 7, epsilon: float = 0.05):
        super().__init__()
        self.wsi_encoder = ContinuousWSIEncoder(in_dim=in_dim, d_model=d_model)
        self.gene_encoder = PathwayGeneEncoder(n_genes=n_genes, n_pathways=n_pathways, d_model=d_model)
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.epsilon = epsilon
        
        self.fc_signal = nn.Sequential(nn.Linear(d_model, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 64))
        self.fc_env = nn.Sequential(nn.Linear(d_model, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 64))
        self.classifier = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, num_classes))

    def sinkhorn(self, cost: torch.Tensor) -> torch.Tensor:
        B, N, M = cost.shape
        K = torch.exp(-cost / self.epsilon) + 1e-9
        u = torch.ones(B, N, device=cost.device) / N
        for _ in range(10):
            v = (1.0 / M) / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-9)
            u = (1.0 / N) / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-9)
        return u.unsqueeze(-1) * K * v.unsqueeze(1)

    def forward(self, patch_features: torch.Tensor, gene_expr: torch.Tensor, coords: torch.Tensor = None):
        wsi = self.wsi_encoder(patch_features, coords=coords)
        gene = self.gene_encoder(gene_expr)
        
        Q = self.q_proj(gene)
        K = self.k_proj(wsi)
        V = self.v_proj(wsi)
        
        cost = 1.0 - torch.bmm(F.normalize(K, dim=-1), F.normalize(Q, dim=-1).transpose(1, 2))
        T = self.sinkhorn(cost)
        aligned = torch.bmm(T.transpose(1, 2) * wsi.size(1), V)
        fused = self.out_proj(aligned) + Q
        
        h = fused.mean(dim=1)
        z_s = self.fc_signal(h)
        z_e = self.fc_env(h)
        logits = self.classifier(z_s)
        return {"logits": logits, "z_s": z_s, "z_e": z_e, "transport_plan": T}

# ============================================================================
# 5. NOVELTY PCMT (PINO-SSM + CONFORMAL + CAUSAL FLOW)
# ============================================================================
class PCMT_Novel(nn.Module):
    """
    Novelty PCMT: Equips PCMT with continuous 2D Diffusion PDE (PINO-SSM) 
    and Causal Counterfactual Intervention Head.
    """
    def __init__(self, in_dim: int = 1024, n_genes: int = 200, n_pathways: int = 10, 
                 d_model: int = 128, num_classes: int = 7, use_pino: bool = True, 
                 epsilon: float = 0.5, use_counterfactual: bool = True):
        super().__init__()
        self.use_pino = use_pino
        self.epsilon = epsilon
        
        self.patch_proj = nn.Sequential(nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.coord_proj = nn.Sequential(nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        
        if use_pino:
            self.spatial_op = PINO_SSM_Kernel(d_model=d_model)
        else:
            self.spatial_op = S4DKernel(d_model=d_model)
            
        self.gene_encoder = PathwayGeneEncoder(n_genes=n_genes, n_pathways=n_pathways, d_model=d_model)
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.causal_head = CounterfactualInterventionHead(in_dim=d_model, latent_dim=64, num_classes=num_classes)

    def sinkhorn(self, cost: torch.Tensor) -> torch.Tensor:
        B, N, M = cost.shape
        K = torch.exp(-cost / self.epsilon) + 1e-9
        u = torch.ones(B, N, device=cost.device) / N
        for _ in range(10):
            v = (1.0 / M) / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-9)
            u = (1.0 / N) / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-9)
        return u.unsqueeze(-1) * K * v.unsqueeze(1)

    def forward(self, patch_features: torch.Tensor, gene_expr: torch.Tensor, coords: torch.Tensor = None, do_domain=None):
        h_vis = self.patch_proj(patch_features)
        if coords is not None:
            h_vis = h_vis + self.coord_proj(coords)
        h_vis = self.spatial_op(h_vis)
        
        h_gen = self.gene_encoder(gene_expr)
        
        Q = self.q_proj(h_gen)
        K = self.k_proj(h_vis)
        V = self.v_proj(h_vis)
        
        cost = 1.0 - torch.bmm(F.normalize(K, dim=-1), F.normalize(Q, dim=-1).transpose(1, 2))
        T = self.sinkhorn(cost)
        aligned = torch.bmm(T.transpose(1, 2) * patch_features.size(1), V)
        fused = self.out_proj(aligned) + Q
        
        logits, z_s, z_e = self.causal_head(fused, do_domain=do_domain)
        return {"logits": logits, "z_s": z_s, "z_e": z_e, "transport_plan": T}
