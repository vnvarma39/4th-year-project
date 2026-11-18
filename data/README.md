# Dataset Guide

This repository benchmarks multi-modal architectures using authentic 10x Genomics Visium HD Spatial Biology data, structured for subsequent direct deployment to dip-based PCR and spectral detection of Cry genes in Bt cotton.

## 1. 10x Genomics Visium HD Arabidopsis thaliana Dataset
- **Organism:** *Arabidopsis thaliana* (Col-0 seedling cryosection, 8-micron resolution).
- **Histology Image:** 31.7 Megapixel Whole Slide Image (`tissue_hires_image.png`, 5283 x 6000 px).
- **Spatial RNA-seq:** 31,109 genes across 7,356 tissue-associated spots.
- **Task:** 7-Class Ground-Truth Anatomical Domain Classification (Cotyledon, Hypocotyl, SAM, REZ, Root Core, Meristematic Zone, Seed Reserve).

### Download Links (Direct 10x Genomics CDN):
1. **Spatial Coordinates & WSI Slide (55.4 MB):**
   ```bash
   curl -O https://cf.10xgenomics.com/samples/spatial-exp/4.0.1/Visium_HD_3prime_Arabidopsis/Visium_HD_3prime_Arabidopsis_spatial.tar.gz
   tar -xzf Visium_HD_3prime_Arabidopsis_spatial.tar.gz
   ```
2. **Binned Feature Matrix & Cell Clustering (768.1 MB):**
   ```bash
   curl -O https://cf.10xgenomics.com/samples/spatial-exp/4.0.1/Visium_HD_3prime_Arabidopsis/Visium_HD_3prime_Arabidopsis_binned_outputs.tar.gz
   tar -xzf Visium_HD_3prime_Arabidopsis_binned_outputs.tar.gz
   ```

Place extracted folders into:
```
data/
└── arabidopsis_spatial/
    ├── spatial/
    │   └── tissue_hires_image.png
    └── binned_outputs/
        └── square_008um/
            ├── filtered_feature_bc_matrix.h5
            ├── spatial/tissue_positions.parquet
            └── analysis/clustering/gene_expression_graphclust/clusters.csv
```

## 2. Bt Cotton Cry Gene Pivot
When wet-lab samples are delivered:
- **Optical Modality:** Dip-strip band optical density / handheld reflectance spectra (400–1100 nm).
- **Genetic Modality:** Dip-based PCR gel band fluorescence / Raman molecular vibrational peaks.
- **Classes:** Non-Bt control (0), Cry1Ac positive (1), Cry2Ab positive (2), Dual-Bt stacked (3).
