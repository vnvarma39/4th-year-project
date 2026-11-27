"""
Generate Publication-Grade Architecture Diagram for AgroTop-Flow Engine (Gen-5)
================================================================================
Renders a high-resolution, peer-review-grade schematic suitable for NeurIPS/ICLR/ISMB.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

def draw_architecture_diagram():
    fig, ax = plt.subplots(figsize=(22, 12), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis('off')
    
    # Color Palette (Nature / Modern Academic styling)
    c_bg = "#F8F9FA"
    c_input = "#E3F2FD"      # Soft Blue
    c_input_edge = "#1976D2"
    c_geom = "#E8F5E9"       # Soft Green
    c_geom_edge = "#388E3C"
    c_phys = "#FFF3E0"       # Soft Orange
    c_phys_edge = "#F57C00"
    c_wfr = "#F3E5F5"        # Soft Purple
    c_wfr_edge = "#7B1FA2"
    c_head = "#FFEBEE"       # Soft Red/Coral
    c_head_edge = "#D32F2F"
    c_text = "#212121"
    
    fig.patch.set_facecolor("white")
    
    # Title Block
    ax.text(50, 57.5, "AgroTop-Flow Engine: Sovereign Physical-Topological Architecture", 
            ha='center', va='center', fontsize=20, fontweight='bold', color="#1A237E")
    ax.text(50, 55.2, "Multi-Modal Plant Spatial Transcriptomics & Transgenic Verification with Exact Conformal Risk Control", 
            ha='center', va='center', fontsize=12, style='italic', color="#455A64")

    # =========================================================================
    # STAGE 1: MULTIMODAL INPUT STREAM (Left Column)
    # =========================================================================
    # Input container box
    rect_in = patches.FancyBboxPatch((2, 6), 18, 46, boxstyle="round,pad=1,rounding_size=2",
                                    facecolor=c_input, edgecolor=c_input_edge, linewidth=2, linestyle='--')
    ax.add_patch(rect_in)
    ax.text(11, 49.5, "MULTIMODAL INPUTS", ha='center', va='center', fontsize=12, fontweight='bold', color=c_input_edge)
    
    # 1. Optical Microscopy Patch
    box_opt = patches.FancyBboxPatch((4, 36), 14, 10, boxstyle="round,pad=0.5,rounding_size=1",
                                    facecolor="white", edgecolor=c_input_edge, linewidth=1.5)
    ax.add_patch(box_opt)
    ax.text(11, 42.5, "Optical Histology Patch", ha='center', va='center', fontsize=10, fontweight='bold', color=c_text)
    ax.text(11, 39.5, r"$\mathbf{X}_{\mathrm{vis}} \in \mathbb{R}^{B \times N \times 1024}$", ha='center', va='center', fontsize=9, color="#0D47A1")
    ax.text(11, 37.2, "Cell Walls, Pectin, Trichomes", ha='center', va='center', fontsize=8, color="#546E7A")

    # 2. Spatial Physical Coordinates
    box_crd = patches.FancyBboxPatch((4, 22), 14, 10, boxstyle="round,pad=0.5,rounding_size=1",
                                    facecolor="white", edgecolor=c_input_edge, linewidth=1.5)
    ax.add_patch(box_crd)
    ax.text(11, 28.5, "Spatial 2D Coordinates", ha='center', va='center', fontsize=10, fontweight='bold', color=c_text)
    ax.text(11, 25.5, r"$\mathbf{C} \in \mathbb{R}^{B \times N \times 2}$ (microns)", ha='center', va='center', fontsize=9, color="#0D47A1")
    ax.text(11, 23.2, "Dip-strip / Micro-grid layout", ha='center', va='center', fontsize=8, color="#546E7A")

    # 3. Gene RNA Expression Profile
    box_gen = patches.FancyBboxPatch((4, 8), 14, 10, boxstyle="round,pad=0.5,rounding_size=1",
                                    facecolor="white", edgecolor=c_input_edge, linewidth=1.5)
    ax.add_patch(box_gen)
    ax.text(11, 14.5, "RNA Expression Profile", ha='center', va='center', fontsize=10, fontweight='bold', color=c_text)
    ax.text(11, 11.5, r"$\mathbf{X}_{\mathrm{gene}} \in \mathbb{R}^{B \times 200}$", ha='center', va='center', fontsize=9, color="#0D47A1")
    ax.text(11, 9.2, "Cry1Ac, Cry2Ab, 35S, NOS", ha='center', va='center', fontsize=8, color="#546E7A")

    # =========================================================================
    # STAGE 2: GEOMETRIC & SHEAF OPERATORS (Column 2)
    # =========================================================================
    # Container
    rect_geom = patches.FancyBboxPatch((24, 6), 22, 46, boxstyle="round,pad=1,rounding_size=2",
                                      facecolor=c_geom, edgecolor=c_geom_edge, linewidth=2)
    ax.add_patch(rect_geom)
    ax.text(35, 49.5, "RIEMANNIAN & SHEAF TOPOLOGY", ha='center', va='center', fontsize=12, fontweight='bold', color=c_geom_edge)

    # Riemannian Metric Tensor Block
    box_rie = patches.FancyBboxPatch((26, 32), 18, 14, boxstyle="round,pad=0.5,rounding_size=1",
                                    facecolor="white", edgecolor=c_geom_edge, linewidth=1.5)
    ax.add_patch(box_rie)
    ax.text(35, 42.5, "1. Riemannian Metric Estimator", ha='center', va='center', fontsize=10, fontweight='bold', color=c_text)
    ax.text(35, 39.5, r"$g_{ij}(x) = \bar{g}\delta_{ij} + \alpha_w \tau_w(x) n_i n_j + \beta_f(u)$", ha='center', va='center', fontsize=8, color="#1B5E20")
    ax.text(35, 36.8, "Encodes Cell Wall Impedance", ha='center', va='center', fontsize=8.5, color="#2E7D32")
    ax.text(35, 34.2, r"Impedance: $\tau_w(x) \in [0, 5]$ (No-leak Neumann)", ha='center', va='center', fontsize=8, color="#37474F")

    # Cellular Sheaf Laplacian Block
    box_shf = patches.FancyBboxPatch((26, 10), 18, 18, boxstyle="round,pad=0.5,rounding_size=1",
                                    facecolor="white", edgecolor=c_geom_edge, linewidth=1.5)
    ax.add_patch(box_shf)
    ax.text(35, 24.5, "2. Cellular Sheaf Laplacian", ha='center', va='center', fontsize=10, fontweight='bold', color=c_text)
    ax.text(35, 21.8, r"$\mathcal{L}_{\mathcal{F}} = \delta^* \delta \in \mathbb{R}^{Nd \times Nd}$", ha='center', va='center', fontsize=9.5, fontweight='bold', color="#1B5E20")
    ax.text(35, 18.8, r"$(\mathcal{L}_{\mathcal{F}} h)_u = \sum_{v \sim u} \mathcal{F}_{u \unlhd e}^* (\mathcal{F}_{u \unlhd e} h_u - \mathcal{F}_{v \unlhd e} h_v)$", ha='center', va='center', fontsize=7.5, color="#2E7D32")
    ax.text(35, 15.5, "• Directional PIN Polar Auxin Transport", ha='center', va='center', fontsize=8, color="#37474F")
    ax.text(35, 13.2, "• Plasmodesmata SEL Filtering", ha='center', va='center', fontsize=8, color="#37474F")
    ax.text(35, 11.2, r"• Cohomology $\ker(\mathcal{L}_{\mathcal{F}}) = H^0(\mathcal{X}; \mathcal{F})$", ha='center', va='center', fontsize=8, color="#1B5E20")

    # =========================================================================
    # STAGE 3: CONTINUOUS DYNAMICS & MASS FLOW (Column 3)
    # =========================================================================
    rect_phys = patches.FancyBboxPatch((50, 6), 22, 46, boxstyle="round,pad=1,rounding_size=2",
                                      facecolor=c_phys, edgecolor=c_phys_edge, linewidth=2)
    ax.add_patch(rect_phys)
    ax.text(61, 49.5, "CONTINUOUS PHYSICAL OPERATORS", ha='center', va='center', fontsize=12, fontweight='bold', color=c_phys_edge)

    # Symplectic Hamiltonian Cyclosis
    box_ham = patches.FancyBboxPatch((52, 28), 18, 18, boxstyle="round,pad=0.5,rounding_size=1",
                                    facecolor="white", edgecolor=c_phys_edge, linewidth=1.5)
    ax.add_patch(box_ham)
    ax.text(61, 42.5, "3. Symplectic Hamiltonian Cyclosis", ha='center', va='center', fontsize=10, fontweight='bold', color=c_text)
    ax.text(61, 39.5, r"Cotangent Bundle: $(q, p) \in T^* \mathcal{M}$", ha='center', va='center', fontsize=8.5, fontweight='bold', color="#E65100")
    ax.text(61, 36.8, r"$\dot{q} = \mathbf{M}^{-1} p, \quad \dot{p} = -\nabla_q V - \Gamma(x) p$", ha='center', va='center', fontsize=8, color="#BF360C")
    ax.text(61, 34.0, "Symplectic Störmer-Verlet Leapfrog:", ha='center', va='center', fontsize=8, color="#455A64")
    ax.text(61, 31.5, r"$p_{t+1/2} = p_t - \frac{\Delta t}{2} F_\theta(q_t)$", ha='center', va='center', fontsize=7.5, color="#E65100")
    ax.text(61, 29.5, r"$q_{t+1} = q_t + \Delta t \mathbf{M}^{-1} p_{t+1/2}$", ha='center', va='center', fontsize=7.5, color="#E65100")

    # Hodge-Helmholtz Decomposition
    box_hdg = patches.FancyBboxPatch((52, 9), 18, 16, boxstyle="round,pad=0.5,rounding_size=1",
                                    facecolor="white", edgecolor=c_phys_edge, linewidth=1.5)
    ax.add_patch(box_hdg)
    ax.text(61, 21.8, "4. Hodge-Helmholtz Flux", ha='center', va='center', fontsize=10, fontweight='bold', color=c_text)
    ax.text(61, 18.8, r"$\mathbf{J} = -\nabla_g \phi + \nabla \times \mathbf{A} + \mathbf{h}$", ha='center', va='center', fontsize=9.5, fontweight='bold', color="#E65100")
    ax.text(61, 15.8, r"• Curl-free Fick Diffusion: $-\nabla_g \phi$", ha='center', va='center', fontsize=8, color="#37474F")
    ax.text(61, 13.5, r"• Divergence-free Cyclosis: $\nabla \times \mathbf{A}$", ha='center', va='center', fontsize=8, color="#37474F")
    ax.text(61, 11.2, r"Exact Mass Conservation: $\mathrm{div}_g(\nabla \times \mathbf{A}) \equiv 0$", ha='center', va='center', fontsize=7.5, color="#BF360C")

    # =========================================================================
    # STAGE 4: REACTIVE OPTIMAL TRANSPORT & CLASSIFIER (Column 4)
    # =========================================================================
    rect_wfr = patches.FancyBboxPatch((76, 6), 22, 46, boxstyle="round,pad=1,rounding_size=2",
                                     facecolor=c_wfr, edgecolor=c_wfr_edge, linewidth=2)
    ax.add_patch(rect_wfr)
    ax.text(87, 49.5, "REACTION-TRANSPORT & RISK", ha='center', va='center', fontsize=12, fontweight='bold', color=c_wfr_edge)

    # Dynamic WFR Unbalanced Optimal Transport
    box_wfr_op = patches.FancyBboxPatch((78, 29), 18, 17, boxstyle="round,pad=0.5,rounding_size=1",
                                       facecolor="white", edgecolor=c_wfr_edge, linewidth=1.5)
    ax.add_patch(box_wfr_op)
    ax.text(87, 42.5, "5. Dynamic WFR Transport", ha='center', va='center', fontsize=10, fontweight='bold', color=c_text)
    ax.text(87, 39.5, r"$\mathrm{WFR}^2(\rho_0, \rho_1) = \inf \int (\frac{1}{2}\|v\|_g^2 + \frac{\beta^2}{2}\alpha^2)\rho$", ha='center', va='center', fontsize=7.5, color="#4A148C")
    ax.text(87, 36.8, r"$\partial_t \rho_t + \mathrm{div}_g(\rho_t v_t) = \alpha_t \rho_t$", ha='center', va='center', fontsize=8.5, fontweight='bold', color="#7B1FA2")
    ax.text(87, 34.0, r"• Active 35S Transcription: $\alpha > 0$", ha='center', va='center', fontsize=8, color="#37474F")
    ax.text(87, 32.0, r"• RISC Enzymatic Cleavage: $\alpha < 0$", ha='center', va='center', fontsize=8, color="#37474F")
    ax.text(87, 30.0, "Envelope Theorem Acceleration", ha='center', va='center', fontsize=7.5, color="#4A148C")

    # Fisher-Rao Metric & Conformal Risk Control
    box_fr = patches.FancyBboxPatch((78, 9), 18, 17, boxstyle="round,pad=0.5,rounding_size=1",
                                   facecolor="white", edgecolor=c_head_edge, linewidth=1.5)
    ax.add_patch(box_fr)
    ax.text(87, 22.8, "6. Fisher-Rao Geodesic Head", ha='center', va='center', fontsize=10, fontweight='bold', color=c_head_edge)
    ax.text(87, 20.0, r"$d_{\mathrm{FR}}(z_s, p_c) = 2 \arccos\left(\sum \sqrt{z_{s,k} p_{c,k}}\right)$", ha='center', va='center', fontsize=8, color="#B71C1C")
    ax.text(87, 17.5, r"Round Metric on 3-Sphere Orthant $\mathbb{S}^C(2)$", ha='center', va='center', fontsize=8, color="#37474F")
    ax.text(87, 14.8, "Exact Conformal Risk Control:", ha='center', va='center', fontsize=8.5, fontweight='bold', color=c_text)
    ax.text(87, 12.5, r"$\mathbb{P}(Y \in \mathcal{C}(X)) \geq 0.95 \quad (\alpha=0.05)$", ha='center', va='center', fontsize=8, color="#B71C1C")
    ax.text(87, 10.5, "Ambiguity Set: 0.97 (Cotton) / 2.60 (HD)", ha='center', va='center', fontsize=7.5, color="#37474F")

    # =========================================================================
    # CONNECTOR ARROWS
    # =========================================================================
    def draw_arrow(x0, y0, x1, y1, color="#37474F"):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(facecolor=color, edgecolor=color, width=1.5, headwidth=6, shrink=0.05))

    # Inputs -> Riemannian & Sheaf
    draw_arrow(18, 41, 26, 39, c_input_edge)
    draw_arrow(18, 27, 26, 21, c_input_edge)
    draw_arrow(18, 13, 78, 38, c_input_edge) # RNA straight to WFR bridge

    # Riemannian Metric -> Sheaf
    draw_arrow(35, 32, 35, 28, c_geom_edge)

    # Sheaf -> Hamiltonian
    draw_arrow(44, 21, 52, 37, c_geom_edge)

    # Hamiltonian -> Hodge
    draw_arrow(61, 28, 61, 25, c_phys_edge)

    # Hodge -> WFR Bridge
    draw_arrow(70, 17, 78, 35, c_phys_edge)

    # WFR Bridge -> Fisher-Rao
    draw_arrow(87, 29, 87, 26, c_wfr_edge)

    # Output banner
    ax.text(87, 4.0, "OUTPUT: Calibrated Transgenic Verification", ha='center', va='center', fontsize=9.5, fontweight='bold', color="#B71C1C")

    out_path = os.path.join(r"E:\NIKHIL\startup\agri-rna\figures", "agrotop_flow_architecture_diagram.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[OK] Diagram saved to {out_path}")

if __name__ == "__main__":
    draw_architecture_diagram()
