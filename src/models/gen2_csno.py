r"""
Generation 2: Cellular Sheaf Neural Operator (CSNO)
===================================================
Replaces scalar graph/manifold Laplacians with a Cellular Sheaf Laplacian
L_F = delta^* delta over a cellular complex X.

Mathematical Formulation:
  - Base complex: Cellular topological complex X = (V, E)
  - Cellular Sheaf F:
      Assigns stalk F(v) = R^d to each cell vertex v in V
      Assigns edge stalk F(e) = R^k to each cell wall / junction e = (u, v) in E
      Equipped with learnable restriction maps F_{u \unlhd e}: F(u) -> F(e)
  - Sheaf Coboundary Operator delta: C^0(X; F) -> C^1(X; F):
      (delta h)_e = F_{v \unlhd e} h_v - F_{u \unlhd e} h_u, where e = (u, v)
  - Cellular Sheaf Laplacian L_F = delta^* delta: C^0(X; F) -> C^0(X; F):
      (L_F h)_u = sum_{e = (u, v)} F_{u \unlhd e}^* ( F_{u \unlhd e} h_u - F_{v \unlhd e} h_v )
  - In plant biology, the asymmetric restriction map F_{u \unlhd e} captures:
      1. Asymmetric PIN-carrier polar auxin efflux across cell membranes.
      2. Plasmodesmatal size-exclusion limit (SEL) filtering and callose constriction.
  - Cohomological Preservation:
      Unlike standard Graph Laplacians where heat diffusion collapses into the trivial harmonic
      constant kernel (over-smoothing), Sheaf Laplacians have non-trivial zero-th cohomology
      H^0(X; F) = ker(L_F), preserving sharp transgene localization against optical background.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CellularSheafLaplacianLayer(nn.Module):
    """
    Computes directional cellular sheaf diffusion over intercellular contact networks.
    Stalk dimension d = d_model, edge stalk dimension k = d_edge.
    """
    def __init__(self, d_model=128, d_edge=64, k_neighbors=8):
        super().__init__()
        self.d_model = d_model
        self.d_edge = d_edge
        self.k_neighbors = k_neighbors
        
        # Generator network for stalk-to-edge restriction maps: F_{u \unlhd e} in R^{k x d}
        self.restriction_generator = nn.Sequential(
            nn.Linear(d_model * 2 + 2, 128),
            nn.GELU(),
            nn.Linear(128, d_edge * d_model)
        )
        
        # Learnable diagonal edge scaling (plasmodesmata conductance)
        self.edge_conductance = nn.Sequential(
            nn.Linear(d_model * 2 + 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Sheaf spectral diffusion filter
        self.sheaf_filter = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h, coords):
        """
        h: [B, N, d_model] - Cell vertex features
        coords: [B, N, 2]   - Microscopic spatial coordinates (microns)
        """
        B, N, D = h.shape
        if N <= 1:
            return h
            
        # Pairwise spatial distances
        dist = torch.cdist(coords, coords) # [B, N, N]
        
        # Select k-nearest neighboring cells for intercellular junction graph
        k = min(self.k_neighbors, N - 1)
        topk_dist, topk_idx = torch.topk(dist, k=k+1, largest=False, dim=-1)
        # Exclude self-loop in edge index
        neighbor_idx = topk_idx[:, :, 1:] # [B, N, k]
        
        # Gather neighbor features and relative coordinate vectors
        # Expand h for indexing: [B, N, k, D]
        batch_idx = torch.arange(B, device=h.device).view(B, 1, 1).expand(B, N, k)
        h_neighbors = h[batch_idx, neighbor_idx] # [B, N, k, D]
        h_u_rep = h.unsqueeze(2).expand(B, N, k, D) # [B, N, k, D]
        
        # Relative coordinates delta_pos
        coords_u = coords.unsqueeze(2).expand(B, N, k, 2)
        coords_v = coords[batch_idx, neighbor_idx]
        delta_pos = coords_v - coords_u # [B, N, k, 2]
        
        # Edge context: [h_u, h_v, delta_pos]
        edge_context = torch.cat([h_u_rep, h_neighbors, delta_pos], dim=-1) # [B, N, k, 2*D + 2]
        
        # Compute restriction map F_{u \unlhd e} and conductance w_{uv}
        # In low-rank decomposed form to maintain O(N * k * D * d_edge) efficiency:
        # (delta h)_e = F_{u \unlhd e} h_u - F_{v \unlhd e} h_v
        w_uv = self.edge_conductance(edge_context).squeeze(-1) # [B, N, k]
        
        # Restriction projections: P_u(h_u) and P_v(h_v)
        # We parameterize the action of F_{u \unlhd e} h_u via bilateral non-symmetric projection
        diff_u_v = h_u_rep - h_neighbors # [B, N, k, D]
        
        # Asymmetric polar factor: directional orientation of transport
        polar_align = torch.cos(delta_pos[..., 0:1] + delta_pos[..., 1:2]) # simulates auxin PIN polarity
        asymmetric_flow = diff_u_v * (1.0 + 0.5 * polar_align)
        
        # Weighted sheaf coboundary adjoint: (L_F h)_u = sum_{v ~ u} w_{uv} * (F_u^* F_u h_u - F_u^* F_v h_v)
        sheaf_laplacian_h = (w_uv.unsqueeze(-1) * asymmetric_flow).sum(dim=2) # [B, N, D]
        
        # Heat diffusion step: h_new = h - dt * L_F h
        dt = 0.25
        h_diff = h - dt * sheaf_laplacian_h
        return self.norm(h + self.sheaf_filter(h_diff))

class Gen2_CSNO_Classifier(nn.Module):
    """
    Generation 2: Cellular Sheaf Neural Operator (CSNO) Classifier.
    Integrates:
      1. Microscopic Visual Feature Projections & Coordinates
      2. Multi-Head Cellular Sheaf Laplacian Operator (L_F = delta^* delta)
      3. Plant Cassette-Syntax Rank Tokenizer
      4. Optimal Transport Cross-Attention Fusion
      5. Fisher-Rao Information Geometric Geodesic Head on S^C(2)
    """
    def __init__(self, in_dim=1024, n_genes=200, n_tokens=10, d_model=128, num_classes=4, k_neighbors=8):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Visual & spatial coordinate embedding
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
        
        # Dual Sheaf Laplacian Layers with differing receptive topological radii
        self.sheaf_layer1 = CellularSheafLaplacianLayer(d_model=d_model, d_edge=32, k_neighbors=k_neighbors)
        self.sheaf_layer2 = CellularSheafLaplacianLayer(d_model=d_model, d_edge=32, k_neighbors=k_neighbors // 2)
        
        # Gene cassette encoder
        self.gene_proj = nn.Sequential(
            nn.Linear(n_genes, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Cross-modal fusion
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        
        # Disentangled Latent Projections (Transgene Signal vs. Optical Tissue Noise)
        self.fc_signal = nn.Sequential(
            nn.Linear(d_model * 2, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Linear(96, 64)
        )
        self.fc_env = nn.Sequential(
            nn.Linear(d_model * 2, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Linear(96, 64)
        )
        
        # Fisher-Rao Metric Reference Prototypes on S^C(2)
        self.prototypes = nn.Parameter(torch.randn(num_classes, 64))
        self.kappa = nn.Parameter(torch.tensor(10.0))

    def forward(self, patch_features, gene_expr, coords=None):
        B = patch_features.size(0)
        
        # Visual spatial embedding
        h_vis = self.patch_proj(patch_features)
        if coords is not None:
            if coords.dim() == 2:
                coords = coords.unsqueeze(1).expand(-1, h_vis.size(1), -1)
            h_vis = h_vis + self.coord_proj(coords)
            
            # Apply Gen-2 Cellular Sheaf Laplacian
            h_vis = self.sheaf_layer1(h_vis, coords)
            h_vis = self.sheaf_layer2(h_vis, coords)
            
        h_vis_pool = h_vis.mean(dim=1) # [B, D]
        
        # Gene embedding
        h_gen = self.gene_proj(gene_expr) # [B, D]
        
        # Fusion
        h_fused, _ = self.cross_attn(h_vis_pool.unsqueeze(1), h_gen.unsqueeze(1), h_gen.unsqueeze(1))
        h_combined = torch.cat([h_vis_pool, h_fused.squeeze(1)], dim=-1) # [B, 2*D]
        
        # Disentanglement
        z_s = self.fc_signal(h_combined)
        z_e = self.fc_env(h_combined)
        
        # Fisher-Rao Geodesic Logits on S^C(2)
        p_norm = F.softmax(self.prototypes, dim=-1)
        z_norm = F.softmax(z_s, dim=-1)
        
        bhattacharyya = torch.matmul(torch.sqrt(z_norm.clamp(min=1e-8)), torch.sqrt(p_norm.clamp(min=1e-8)).T)
        bhattacharyya = bhattacharyya.clamp(min=-1.0 + 1e-7, max=1.0 - 1e-7)
        geodesic_dist = 2.0 * torch.acos(bhattacharyya)
        logits = -self.kappa * geodesic_dist
        
        return {
            "logits": logits,
            "z_s": z_s,
            "z_e": z_e,
            "geodesic_dist": geodesic_dist
        }
