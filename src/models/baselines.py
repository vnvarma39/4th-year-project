"""
Comparative Baseline Architectures
==================================
Includes:
  1. PAMTModel: Pathological-Gene Multimodal Transformer (Baseline Multi-Head Cross Attention).
  2. VisionOnly: Histology patch visual encoder without transcriptomic input.
  3. GeneOnly: Pathway gene token encoder without histological input.
  4. PAMT_with_PINO: PAMT baseline augmented with PINO-SSM continuous transport.
"""

import torch
import torch.nn as nn
from .pcmt import PathwayGeneEncoder
from .pino_ssm import PINO_SSM_Kernel

class PAMTModel(nn.Module):
    """
    PAMT: Pathological-Gene Multimodal Transformer (Standard Discrete Softmax Attention).
    """
    def __init__(self, in_dim: int = 1024, n_genes: int = 200, n_pathways: int = 10, 
                 d_model: int = 128, num_classes: int = 7):
        super().__init__()
        self.wsi_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.gene_encoder = PathwayGeneEncoder(n_genes=n_genes, n_pathways=n_pathways, d_model=d_model)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, patch_features: torch.Tensor, gene_expr: torch.Tensor, coords: torch.Tensor = None):
        vis = self.wsi_proj(patch_features)
        gen = self.gene_encoder(gene_expr)
        attn_out, _ = self.cross_attn(gen, vis, vis)
        h = attn_out.mean(dim=1)
        logits = self.classifier(h)
        return {"logits": logits}

class VisionOnly(nn.Module):
    """
    Unimodal Histology Patch Baseline.
    """
    def __init__(self, in_dim: int = 1024, d_model: int = 128, num_classes: int = 7):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, patch_features: torch.Tensor, gene_expr: torch.Tensor = None, coords: torch.Tensor = None):
        h = self.proj(patch_features).mean(dim=1)
        logits = self.classifier(h)
        return {"logits": logits}

class GeneOnly(nn.Module):
    """
    Unimodal Spot-Level Transcriptomics Baseline.
    """
    def __init__(self, n_genes: int = 200, n_pathways: int = 10, d_model: int = 128, num_classes: int = 7):
        super().__init__()
        self.gene_encoder = PathwayGeneEncoder(n_genes=n_genes, n_pathways=n_pathways, d_model=d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, patch_features: torch.Tensor = None, gene_expr: torch.Tensor = None, coords: torch.Tensor = None):
        tokens = self.gene_encoder(gene_expr)
        h = tokens.mean(dim=1)
        logits = self.classifier(h)
        return {"logits": logits}

class PAMT_with_PINO(nn.Module):
    """
    PAMT augmented with PINO-SSM continuous spatial diffusion operator.
    """
    def __init__(self, in_dim: int = 1024, n_genes: int = 200, n_pathways: int = 10, 
                 d_model: int = 128, num_classes: int = 7):
        super().__init__()
        self.wsi_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.pino_spatial = PINO_SSM_Kernel(d_model=d_model)
        self.gene_encoder = PathwayGeneEncoder(n_genes=n_genes, n_pathways=n_pathways, d_model=d_model)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, patch_features: torch.Tensor, gene_expr: torch.Tensor, coords: torch.Tensor = None):
        vis = self.pino_spatial(self.wsi_proj(patch_features))
        gen = self.gene_encoder(gene_expr)
        attn_out, _ = self.cross_attn(gen, vis, vis)
        h = attn_out.mean(dim=1)
        return {"logits": self.classifier(h)}
