# Physio-Causal Multimodal Transformer (PCMT)

An AI framework for plant spatial biology and rapid transgenic crop verification, developed for the AgriHub / IIT Indore initiative on rapid Cry gene detection in Bt cotton.

---

## Overview

PCMT fuses physical tissue images (high-resolution histology/morphology) with molecular readouts (spatial transcriptomics, PCR band intensities, or spectral curves). 

Unlike standard transformers that treat spatial biological tokens as discrete unordered sequences, PCMT integrates continuous physical transport dynamics:
1. **Continuous Spatial Operator (S4D / PINO-SSM):** Incorporates 2D diffusion-advection partial differential equations to model intercellular biological continuity.
2. **Pathway Projection:** Translates high-dimensional gene expressions into structured functional pathway tokens.
3. **Optimal Transport Alignment:** Uses regularized Sinkhorn distance to find optimal pairings between visual morphology and active gene pathways.
4. **Causal Disentanglement:** Isolates invariant biological signals from optical noise and technical batch artifacts using Hilbert-Schmidt Independence Criterion (HSIC).
5. **Conformal Safety Head:** Employs distribution-free split conformal prediction to output mathematically certified prediction sets at calibrated confidence levels (e.g., 90%).

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
