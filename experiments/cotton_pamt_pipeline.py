"""
Cotton-PAMT: Multimodal Transformer for Rapid Cry Gene Detection in Bt Cotton
Adapted from PAMT (Pathway-Aware Multimodal Transformer) architecture
for Dip-PCR Gel Imaging + Spectral Analysis.

Project: Rapid Detection of Cry Genes in Bt Cotton Using Dip-Based PCR and AI-Assisted Spectral Analysis
Investigators: Dr. Neha Bharill, Dr. Milind Ratnaparkhe, Dr. Om Prakash Patel
Target Classes:
  0: Non-Bt Cotton (Negative for Cry genes)
  1: Cry1Ac Positive (Bollgard-I, ~300 bp amplicon)
  2: Cry2Ab Positive (~750 bp amplicon)
  3: Stacked Cry1Ac + Cry2Ab Positive (Bollgard-II, both 300 bp & 750 bp)
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. Synthetic Data Generation
# ==========================================

def generate_synthetic_bt_cotton_dataset(num_samples=200, seed=42):
    """
    Generates realistic paired Dip-PCR image band profiles and spectral curves.
    
    Image Modality (Dip-PCR Gel Lane):
      - 32 spatial height bins along the gel lane (representing 100 bp to 1500 bp)
      - Cry1Ac marker located around bin index 12 (~300 bp)
      - Cry2Ab marker located around bin index 22 (~750 bp)
      - Internal control (GhActin/Sad1) located around bin index 8 (~200-250 bp)
      - Channel dimension = 8 visual feature filters (simulating multi-channel patch features)
      
    Spectral Modality (Dip-PCR Absorbance / Fluorescence):
      - 60 spectral channels (400 nm - 700 nm in 5 nm increments)
      - Cry1Ac induces plasmonic shift peak around 530 nm (index 26)
      - Cry2Ab induces distinct peak around 585 nm (index 37)
      - Non-Bt shows baseline baseline absorbance without target-specific shifts
      
    Marker / Pathway Modality:
      - 5 target markers: [Cry1Ac, Cry2Ab, CaMV 35S promoter, NOS terminator, Sad1 cotton ref]
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    samples = []
    classes = [0, 1, 2, 3] # Non-Bt, Cry1Ac, Cry2Ab, Stacked
    
    wavelengths = np.linspace(400, 700, 60)
    
    for i in range(num_samples):
        label = np.random.choice(classes, p=[0.25, 0.25, 0.25, 0.25])
        
        # 1. Gel Band Lane Simulation: [32, 8]
        # Base background noise
        lane = np.random.normal(0.05, 0.02, (32, 8)).astype(np.float32)
        # Internal control band present in all viable plant extracts
        lane[8:10, :] += np.random.uniform(0.6, 0.9, (2, 8))
        
        # 2. Spectral Curve Simulation: [60]
        # Base baseline curve (smooth background Rayleigh scattering)
        spectrum = 0.15 + 0.1 * np.exp(-(wavelengths - 420)**2 / (2 * 50**2))
        spectrum += np.random.normal(0, 0.01, 60)
        
        # 3. Marker Vector: [5] (Cry1Ac, Cry2Ab, 35S, NOS, Sad1)
        markers = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32) # Sad1 is always 1.0
        
        if label == 1:  # Cry1Ac
            lane[11:13, :] += np.random.uniform(0.7, 1.0, (2, 8)) # 300 bp band
            spectrum += 0.5 * np.exp(-(wavelengths - 530)**2 / (2 * 15**2)) # 530 nm peak
            markers[0] = 1.0 # Cry1Ac
            markers[2] = 1.0 # 35S promoter
            markers[3] = 1.0 # NOS terminator
        elif label == 2: # Cry2Ab
            lane[21:23, :] += np.random.uniform(0.7, 1.0, (2, 8)) # 750 bp band
            spectrum += 0.45 * np.exp(-(wavelengths - 585)**2 / (2 * 18**2)) # 585 nm peak
            markers[1] = 1.0 # Cry2Ab
            markers[2] = 1.0 # 35S promoter
            markers[3] = 1.0 # NOS terminator
        elif label == 3: # Stacked Cry1Ac + Cry2Ab
            lane[11:13, :] += np.random.uniform(0.65, 0.95, (2, 8)) # 300 bp
            lane[21:23, :] += np.random.uniform(0.65, 0.95, (2, 8)) # 750 bp
            spectrum += 0.45 * np.exp(-(wavelengths - 530)**2 / (2 * 15**2))
            spectrum += 0.40 * np.exp(-(wavelengths - 585)**2 / (2 * 18**2))
            markers[0] = 1.0
            markers[1] = 1.0
            markers[2] = 1.0
            markers[3] = 1.0
        
        lane = np.clip(lane, 0, 1.5).astype(np.float32)
        spectrum = np.clip(spectrum, 0, 2.0).astype(np.float32)
        
        samples.append({
            "sample_id": f"COTTON_SMP_{i:04d}",
            "lane_image_features": lane,
            "spectrum": spectrum,
            "markers": markers,
            "label": label
        })
        
    return samples

class CottonBtDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        item = self.samples[idx]
        return (
            torch.tensor(item["lane_image_features"], dtype=torch.float32),
            torch.tensor(item["spectrum"], dtype=torch.float32),
            torch.tensor(item["markers"], dtype=torch.float32),
            torch.tensor(item["label"], dtype=torch.long)
        )

# ==========================================
# 2. Cotton-PAMT Architecture
# ==========================================

class SpectralMarkerEncoder(nn.Module):
    """
    Corresponds to PAMT's Pathway-Gene Encoder.
    Encodes spectral absorbance curve and genetic marker sets into high-level pathway tokens.
    """
    def __init__(self, spec_dim=60, marker_dim=5, embed_dim=64, num_heads=4):
        super().__init__()
        # Project raw spectral bands into spectral tokens
        self.spec_proj = nn.Sequential(
            nn.Linear(spec_dim, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )
        # Project 5 molecular markers into marker tokens
        self.marker_embed = nn.Linear(1, embed_dim)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, marker_dim + 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, spectrum, markers):
        # spectrum: [B, 60] -> token: [B, 1, embed_dim]
        spec_token = self.spec_proj(spectrum).unsqueeze(1)
        # markers: [B, 5] -> tokens: [B, 5, embed_dim]
        marker_tokens = self.marker_embed(markers.unsqueeze(-1))
        
        # Concat: [B, 6, embed_dim]
        tokens = torch.cat([spec_token, marker_tokens], dim=1)
        tokens = tokens + self.pos_embed
        out = self.transformer(tokens)
        return self.norm(out)

class GelImageEncoder(nn.Module):
    """
    Corresponds to PAMT's WSI Tile Encoder.
    Encodes gel electrophoresis / dipstick bands along spatial patches.
    """
    def __init__(self, num_patches=32, in_dim=8, embed_dim=64, num_heads=4):
        super().__init__()
        self.patch_proj = nn.Linear(in_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, 32, 8] -> [B, 32, embed_dim]
        tokens = self.patch_proj(x) + self.pos_embed
        out = self.transformer(tokens)
        return self.norm(out)

class SpectralGuidedFusion(nn.Module):
    """
    Corresponds to PAMT's Gene-Guided WSI Cross-Attention Fusion.
    Cross-attention: Spectral & Marker queries attend over Gel Image band keys/values.
    Includes contrastive alignment loss between spectral peaks and band signals.
    """
    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, gel_tokens, spectral_tokens):
        # Query = Spectral/Marker tokens [B, 6, 64]
        # Key, Value = Gel Image tokens [B, 32, 64]
        attn_out, attn_weights = self.cross_attn(
            query=spectral_tokens,
            key=gel_tokens,
            value=gel_tokens
        )
        x = self.norm1(spectral_tokens + attn_out)
        out = self.norm2(x + self.mlp(x))
        
        # Fine-grained Contrastive Alignment Loss (PAMT style)
        # Similarity matrix between spectral tokens and gel spatial patches: [B, 6, 32]
        sim_matrix = torch.bmm(spectral_tokens, gel_tokens.transpose(1, 2)) / (spectral_tokens.size(-1) ** 0.5)
        top2_vals, _ = torch.topk(sim_matrix, k=2, dim=-1)
        contrastive_loss = -torch.mean(F.logsigmoid(top2_vals))
        
        return out, contrastive_loss, attn_weights

class CottonPAMT(nn.Module):
    """
    End-to-End Cotton-PAMT Classifier.
    """
    def __init__(self, num_classes=4, embed_dim=64):
        super().__init__()
        self.gel_encoder = GelImageEncoder(num_patches=32, in_dim=8, embed_dim=embed_dim)
        self.spec_encoder = SpectralMarkerEncoder(spec_dim=60, marker_dim=5, embed_dim=embed_dim)
        self.fusion = SpectralGuidedFusion(embed_dim=embed_dim, num_heads=4)
        
        # Diagnostic Classification Head
        # Concatenates pooled spectral tokens and cross-attended fused tokens
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, gel_img, spectrum, markers):
        gel_feats = self.gel_encoder(gel_img)       # [B, 32, 64]
        spec_feats = self.spec_encoder(spectrum, markers) # [B, 6, 64]
        
        fused_feats, contrastive_loss, attn_weights = self.fusion(gel_feats, spec_feats)
        
        # Global pooling across tokens
        pool_spec = spec_feats.mean(dim=1)    # [B, 64]
        pool_fused = fused_feats.mean(dim=1)  # [B, 64]
        
        concat_feats = torch.cat([pool_spec, pool_fused], dim=-1) # [B, 128]
        logits = self.classifier(concat_feats)                    # [B, 4]
        
        return logits, contrastive_loss, attn_weights

# ==========================================
# 3. Training & Evaluation Pipeline
# ==========================================

def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Cotton-PAMT pipeline on: {device}")
    
    # 1. Prepare data
    print("Generating synthetic Bt cotton dataset (250 samples)...")
    dataset_samples = generate_synthetic_bt_cotton_dataset(num_samples=250, seed=42)
    train_samples = dataset_samples[:200]
    test_samples = dataset_samples[200:]
    
    train_loader = DataLoader(CottonBtDataset(train_samples), batch_size=16, shuffle=True)
    test_loader = DataLoader(CottonBtDataset(test_samples), batch_size=16, shuffle=False)
    
    # 2. Instantiate model
    model = CottonPAMT(num_classes=4, embed_dim=64).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Cotton-PAMT Total Model Parameters: {total_params:,} ({total_params/1e6:.3f} M)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce_loss_fn = nn.CrossEntropyLoss()
    
    # 3. Training loop
    epochs = 15
    print(f"\n--- Starting Training for {epochs} Epochs ---")
    start_train_time = time.perf_counter()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for gel_img, spectrum, markers, labels in train_loader:
            gel_img = gel_img.to(device)
            spectrum = spectrum.to(device)
            markers = markers.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits, cont_loss, _ = model(gel_img, spectrum, markers)
            
            cls_loss = ce_loss_fn(logits, labels)
            loss = cls_loss + 0.1 * cont_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        train_acc = correct / total * 100
        avg_loss = total_loss / total
        
        if epoch % 3 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs}] - Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")
            
    train_duration = time.perf_counter() - start_train_time
    print(f"Training completed in {train_duration:.2f} seconds.")
    
    # 4. Evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    class_names = ["Non-Bt", "Cry1Ac", "Cry2Ab", "Stacked (Cry1Ac+Cry2Ab)"]
    confusion_mat = np.zeros((4, 4), dtype=int)
    
    with torch.no_grad():
        for gel_img, spectrum, markers, labels in test_loader:
            gel_img = gel_img.to(device)
            spectrum = spectrum.to(device)
            markers = markers.to(device)
            labels = labels.to(device)
            
            logits, _, _ = model(gel_img, spectrum, markers)
            preds = torch.argmax(logits, dim=1)
            
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                confusion_mat[t, p] += 1
                
    test_acc = test_correct / test_total * 100
    print(f"\n--- Final Test Results ---")
    print(f"Test Accuracy on Hold-out Field Samples: {test_acc:.2f}%")
    print("\nConfusion Matrix (Rows=Actual, Cols=Predicted):")
    header = f"{'':<25}" + "".join([f"{name[:10]:>12}" for name in class_names])
    print(header)
    for i, row in enumerate(confusion_mat):
        row_str = f"{class_names[i]:<25}" + "".join([f"{val:>12}" for val in row])
        print(row_str)

if __name__ == "__main__":
    train_and_evaluate()
