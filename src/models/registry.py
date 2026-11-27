r"""
Model Registry and Architecture Factory for Agricultural Spatial Transcriptomics.
==================================================================================

Provides a unified registry interface for instantiating, configuring, and inspecting:
  1. 'rmno_hodge' (Model 1): Foundational Riemannian Morphogenetic Operator with Hodge-Helmholtz Flux.
  2. 'self_improving_agrotop' (Model 2): Grand Sovereign Unified Architecture with Autonomous Active Loop.
  3. Intermediate Mathematical Generations ('gen2_csno', 'gen3_shmo', 'gen4_wfr', 'gen5_agrotop_flow').
"""

from typing import Dict, Any, Type, List, Optional
import torch
import torch.nn as nn

from src.models.rmno_hodge import RMNO_Hodge_Classifier
from src.models.gen2_csno import Gen2_CSNO_Classifier
from src.models.gen3_shmo import Gen3_SHMO_Classifier
from src.models.gen4_wfr import Gen4_WFR_Classifier
from src.models.gen5_agrotop_flow import AgroTopFlowEngine
from src.models.self_improving_model import SelfImprovingAgroTopFlow

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # -------------------------------------------------------------
    # MODEL 1: Foundational Riemannian Morphogenetic Operator
    # -------------------------------------------------------------
    "rmno_hodge": {
        "class": RMNO_Hodge_Classifier,
        "name": "RMNO-Hodge (Foundational Architecture)",
        "role": "Model 1: Morphogenetic PDE Operator & Hodge-Helmholtz Flux Decomposition",
        "description": (
            "Models continuous intercellular morphogen transport across anisotropic cell wall "
            "Riemannian metrics g_ij(x). Decomposes molecular flux into curl-free gradient drift (nabla phi) "
            "and divergence-free rotational cyclosis (curl psi)."
        ),
        "default_kwargs": {
            "in_dim": 1024,
            "n_genes": 200,
            "d_model": 64,
            "num_classes": 7
        },
        "modalities": ["optical_patch", "spatial_coords", "gene_expression"],
        "category": "Foundational PDE"
    },
    
    # -------------------------------------------------------------
    # MODEL 2: Sovereign Self-Improving AgroTop-Flow Engine
    # -------------------------------------------------------------
    "self_improving_agrotop": {
        "class": SelfImprovingAgroTopFlow,
        "name": "Self-Improving AgroTopFlow (Autonomous Co-Evolution)",
        "role": "Model 2: Grand Sovereign Architecture with Conformal Pseudo-Labeling & Geodesic Distillation",
        "description": (
            "Integrates Gen-5 Sovereign AgroTopFlow with an autonomous self-improving loop: "
            "Distribution-free split conformal uncertainty gating (|C(X)|=1), morphogenetic hard-negative "
            "mining, Fisher-Rao geodesic self-distillation, and anchor-preserving model selection."
        ),
        "default_kwargs": {
            "in_dim": 1024,
            "n_genes": 200,
            "n_tokens": 10,
            "d_model": 64,
            "num_classes": 7,
            "k_neighbors": 8
        },
        "modalities": ["optical_patch", "spatial_coords", "gene_expression"],
        "category": "Autonomous Self-Improving"
    },
    
    # -------------------------------------------------------------
    # Intermediate Mathematical Generations
    # -------------------------------------------------------------
    "gen2_csno": {
        "class": Gen2_CSNO_Classifier,
        "name": "Gen-2 Cellular Sheaf Neural Operator (CSNO)",
        "role": "Cellular Sheaf Laplacians & Polar Auxin Directional Transport",
        "description": "Replaces graph Laplacians with cellular sheaf Laplacians to eliminate oversmoothing.",
        "default_kwargs": {"in_dim": 1024, "n_genes": 200, "d_model": 64, "num_classes": 7},
        "modalities": ["optical_patch", "spatial_coords", "gene_expression"],
        "category": "Mathematical Generation"
    },
    "gen3_shmo": {
        "class": Gen3_SHMO_Classifier,
        "name": "Gen-3 Symplectic Hamiltonian Morphogenetic Operator (SHMO)",
        "role": "Phase Space Energy-Preserving Cyclosis Dynamics",
        "description": "Models cytoplasmic streaming as symplectic Hamiltonian flow on cotangent bundle T*M.",
        "default_kwargs": {"in_dim": 1024, "n_genes": 200, "d_model": 64, "num_classes": 7},
        "modalities": ["optical_patch", "spatial_coords", "gene_expression"],
        "category": "Mathematical Generation"
    },
    "gen4_wfr": {
        "class": Gen4_WFR_Classifier,
        "name": "Gen-4 Unbalanced Wasserstein-Fisher-Rao Bridge (WFR)",
        "role": "Dynamic Mass-Varying Reactive Transport",
        "description": "Unbalanced optimal transport modeling continuous transcript generation and degradation.",
        "default_kwargs": {"in_dim": 1024, "n_genes": 200, "d_model": 64, "num_classes": 7},
        "modalities": ["optical_patch", "spatial_coords", "gene_expression"],
        "category": "Mathematical Generation"
    },
    "gen5_agrotop_flow": {
        "class": AgroTopFlowEngine,
        "name": "Gen-5 AgroTop-Flow Engine (Sovereign Architecture)",
        "role": "Grand Unified Physical-Algebraic-Symplectic Architecture",
        "description": "Unified 5-tier architecture combining Sheaf Laplacians, Symplectic Flow, and WFR Bridge.",
        "default_kwargs": {"in_dim": 1024, "n_genes": 200, "d_model": 64, "num_classes": 7},
        "modalities": ["optical_patch", "spatial_coords", "gene_expression"],
        "category": "Mathematical Generation"
    }
}

def list_models() -> List[str]:
    """Return all registered model keys."""
    return list(MODEL_REGISTRY.keys())

def get_model_info(model_key: str) -> Dict[str, Any]:
    """Retrieve metadata description for a registered model."""
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model key '{model_key}'. Available: {list_models()}")
    return MODEL_REGISTRY[model_key]

def get_model(model_key: str, **kwargs) -> nn.Module:
    """
    Factory function to instantiate any registered model with custom or default parameters.
    
    Supports alias mappings for convenience:
      - img_dim -> in_dim
      - gene_dim -> n_genes
      - hidden_dim -> d_model
    """
    info = get_model_info(model_key)
    model_cls = info["class"]
    model_kwargs = info["default_kwargs"].copy()
    
    # Handle common argument aliases
    if "img_dim" in kwargs:
        kwargs["in_dim"] = kwargs.pop("img_dim")
    if "gene_dim" in kwargs:
        kwargs["n_genes"] = kwargs.pop("gene_dim")
    if "hidden_dim" in kwargs:
        kwargs["d_model"] = kwargs.pop("hidden_dim")
        
    model_kwargs.update(kwargs)
    return model_cls(**model_kwargs)
