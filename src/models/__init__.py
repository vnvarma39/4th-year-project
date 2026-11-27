r"""
Novel Mathematical Model Implementations for Agricultural Spatial Transcriptomics.
===================================================================================

Two Core Architectural Pillars:
  1. RMNO_Hodge_Classifier:
     Foundational Riemannian Morphogenetic Operator with Hodge-Helmholtz Flux Decomposition,
     anisotropic cell wall metric tensor estimation, and continuous Benamou-Brenier transport.
  2. SelfImprovingAgroTopFlow / AgroTopFlowEngine:
     The Grand Sovereign Unified Architecture equipped with autonomous active learning,
     conformal risk-gated pseudo-label filtering, and Fisher-Rao self-distillation.
"""

from src.models.rmno_hodge import (
    RMNO_Hodge_Classifier,
    HodgeHelmholtzFluxLayer,
    RiemannianMetricEstimator,
    DynamicBenamouBrenierOT
)
from src.models.gen2_csno import Gen2_CSNO_Classifier, CellularSheafLaplacianLayer
from src.models.gen3_shmo import Gen3_SHMO_Classifier, SymplecticHamiltonianBlock
from src.models.gen4_wfr import Gen4_WFR_Classifier, WassersteinFisherRaoSinkhornBridge
from src.models.gen5_agrotop_flow import AgroTopFlowEngine
from src.models.self_improving_model import SelfImprovingAgroTopFlow

__all__ = [
    # 1. Foundational Riemannian Architecture (Model 1)
    "RMNO_Hodge_Classifier",
    "HodgeHelmholtzFluxLayer",
    "RiemannianMetricEstimator",
    "DynamicBenamouBrenierOT",
    
    # 2. Sovereign Self-Improving Architecture (Model 2)
    "SelfImprovingAgroTopFlow",
    "AgroTopFlowEngine",
    
    # Mathematical Intermediate Generations
    "Gen2_CSNO_Classifier",
    "CellularSheafLaplacianLayer",
    "Gen3_SHMO_Classifier",
    "SymplecticHamiltonianBlock",
    "Gen4_WFR_Classifier",
    "WassersteinFisherRaoSinkhornBridge"
]
