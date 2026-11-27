# Agricultural Spatial Transcriptomics & Transgenic Verification

An AI framework for plant spatial biology and rapid transgenic crop verification, developed for the AgriHub / IIT Indore initiative on rapid Cry gene detection in Bt cotton.

---

## Two Core Architectural Pillars in Repository

This repository packages two distinct, cutting-edge mathematical architectures for plant spatial transcriptomics:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   TWO DUAL ARCHITECTURAL PILLARS                                 │
├──────────────────────────────────────────────────┬───────────────────────────────────────────────┤
│ 1. Foundational Riemannian Architecture (Model 1)│ 2. Sovereign Self-Improving Engine (Model 2)   │
│    • RMNO_Hodge_Classifier                      │    • SelfImprovingAgroTopFlow / AgroTopFlow   │
│    • Continuous Anisotropic Cell Wall Metric     │    • Cellular Sheaf Laplacians (L_F = δ* δ)   │
│    • Benamou-Brenier Dynamic Optimal Transport   │    • Symplectic Hamiltonian Phase Space Flow  │
│    • Hodge-Helmholtz Flux Decomposition          │    • Unbalanced Wasserstein-Fisher-Rao Bridge │
│    • Fisher-Rao Information Geometric Head       │    • Split Conformal Prediction Gate (|C|=1)  │
│    • Reference: src/models/rmno_hodge.py         │    • Fisher-Rao Geodesic Distillation Loop    │
│    • Runner: examples/run_model1_rmno_hodge.py   │    • Runner: examples/run_model2_self_improv. │
└──────────────────────────────────────────────────┴───────────────────────────────────────────────┘
```

### 1. Model 1: Foundational RMNO-Hodge Classifier (`src/models/rmno_hodge.py`)
- **Physics-Informed Cell Wall Metric:** Dynamically models the 2x2 Riemannian metric tensor $g_{ij}(x)$ of anisotropic plant cell walls from microscopic morphology.
- **Hodge-Helmholtz Flux Decomposition:** Analytically separates biochemical spatial flux $\mathbf{J}$ into curl-free gradient drift $-\nabla_g \phi$ (Fickian diffusion) and divergence-free vascular streaming $\nabla_g \times \mathbf{A}$ ($\mathrm{div}_g(\nabla \times \mathbf{A}) = 0$).
- **Continuous Benamou-Brenier Transport:** Formulates multimodal token fusion as dynamic geodesic curve optimization over Wasserstein space $\mathcal{W}_2$.

### 2. Model 2: Sovereign Self-Improving AgroTop-Flow Engine (`src/models/self_improving_model.py`)
- **Grand Sovereign Unified Architecture:** Combines Sheaf Laplacians $\mathcal{L}_{\mathcal{F}}$, Symplectic Leapfrog Cyclosis on $T^*M$, Hodge flux decomposition, and Unbalanced WFR reactive transport.
- **Autonomous Co-Evolution Loop:**
  - *Conformal Risk-Gated Pseudo-Labeling:* Only unlabelled tissue spots with singleton prediction sets ($|C(X)|=1$) at $1-\alpha=90\%$ certified coverage are admitted to avoid error accumulation.
  - *Adversarial Biophysical Hard-Negative Mining:* Gradient-directed optical perturbations stress-test morphogenetic boundaries.
  - *Fisher-Rao Geodesic Distillation:* Teacher-student regularization constrained to the Riemannian 3-sphere orthant $\mathcal{S}^3(2)$ invariant to optical illumination variations.
  - *Pareto-Gated Anchor Preservation:* Only promotes candidate models that improve validation accuracy and maintain noise immunity.

---

## Unified Model Registry & CLI Runners

Instantiate any model through our clean factory interface:

```python
from src.models.registry import get_model, list_models

print(list_models())
# ['rmno_hodge', 'self_improving_agrotop', 'gen2_csno', 'gen3_shmo', 'gen4_wfr', 'gen5_agrotop_flow']

# Instantiate Model 1 (Foundational)
model1 = get_model("rmno_hodge", in_dim=1024, n_genes=200, d_model=64, num_classes=7)

# Instantiate Model 2 (Autonomous Self-Improving)
model2 = get_model("self_improving_agrotop", in_dim=1024, n_genes=200, d_model=64, num_classes=7)
```

### Quick Execution
```bash
# Run Model 1 standalone verification
python examples/run_model1_rmno_hodge.py

# Run Model 2 standalone verification
python examples/run_model2_self_improving.py

# Run side-by-side comparative benchmark on 10x Visium HD plant data
python evaluate_dual_models.py
```

---

## Repository Structure

```
├── configs/             # Experiment parameters and hyperparameters
├── data/                # Dataset documentation and access instructions
├── docs/                # Comprehensive benchmark reports and project pitch deck
├── experiments/         # Standalone benchmark and reproduction scripts
├── figures/             # High-resolution publication plots and learning curves
├── notebooks/           # Interactive Colab and Jupyter analysis notebooks
├── src/                 # Modular Python library
│   ├── data/            # Spatial dataset loaders and preprocessors
│   ├── models/          # PCMT, PINO-SSM, Conformal, and baseline architectures
│   └── utils/           # Evaluation metrics, HSIC loss, and logging utilities
├── baselines/           # Reference code for baseline comparisons
├── requirements.txt     # Python package dependencies
└── README.md
```

---

## Benchmark Highlights (Authentic Plant Biology)

Evaluated on 10x Genomics Visium HD *Arabidopsis thaliana* seedling cryosection (31.7 MP histology image paired with 31,109 genes across 7,356 spots across 7 anatomical domains):

| Model | Modality | Validation Accuracy | OOD Noise Accuracy | Noise Degradation |
| :--- | :--- | :---: | :---: | :---: |
| **PCMT (Standard, S4D)** | Image + RNA-seq | **74.4% (Peak)** | **68.0%** | -4.9% |
| **PCMT + Novelty Trio** | Image + RNA-seq | **55.2% (Peak)** | **53.2%** | **0.0% (Immune)** |
| **PAMT (Baseline)** | Image + RNA-seq | 59.6% | 40.9% | -18.7% |
| **Vision-Only** | Image Only | 52.7% | 52.7% | 0.0% |
| **Gene-Only** | RNA-seq Only | 48.8% | 45.3% | -3.4% |

---

## Installation & Usage

### 1. Environment Setup
```bash
git clone https://github.com/vnvarma39/4th-year-project.git
cd 4th-year-project
pip install -r requirements.txt
```

### 2. Running Benchmarks
- **Authentic Biological Benchmark:**
  ```bash
  python experiments/run_authentic_benchmark.py
  ```
- **Extended Convergence Study:**
  ```bash
  python experiments/run_extended_training.py
  ```
- **Cotton Cry Gene Pivot Simulation:**
  ```bash
  python experiments/run_cotton_simulation.py
  ```

---

## Authors & Acknowledgements
- Primary Investigators: Dr. Neha Bharill, Dr. Milind Ratnaparkhe, Dr. Om Prakash Patel
- Grant: AgriHub / IIT Indore
