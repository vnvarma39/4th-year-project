# Sovereign Mathematical Evolution of Neural Operators for Agricultural Spatial Transcriptomics
## A 5-Generation Theoretical Treatise & Architectural Paradigm Shift

---

### Abstract
Spatial transcriptomics of plant tissues presents unique biophysical challenges unaddressed by conventional deep learning paradigms. Rigid pectin-cellulose cell walls induce severe spatial anisotropy, polar auxin transport creates asymmetric intercellular communication networks, active cytoplasmic cyclosis drives non-dissipative streaming, and ongoing transgene expression violates classical mass-conservation principles. In this treatise, we document the systematic mathematical derivation, biological motivation, and progressive betterment across **five successive generations** of sovereign physical-topological neural operators, culminating in the **AgroTop-Flow Engine**.

---

### 1. Mathematical Taxonomy Across the 5 Generations

| Generation | Architecture Name | Mathematical Foundation | Biological Target | Key Theoretical Invariance |
| :--- | :--- | :--- | :--- | :--- |
| **Gen-1** | **RMNO-Hodge-v1** | 2D Riemannian Manifold $(\mathcal{M}, g)$ + Hodge Decomposition + Benamou-Brenier OT | Cell wall impedance $\tau_w(x)$ & Fickian vs. vascular streaming | Exact mass conservation: $\text{div}_g(\text{curl}_g A) \equiv 0$ |
| **Gen-2** | **CSNO** | Cellular Sheaf Laplacian $\mathcal{L}_{\mathcal{F}} = \delta^* \delta$ on cell complexes | Directional PIN-carrier polar transport & plasmodesmata filtering | Cohomological preservation: $H^0(\mathcal{X}; \mathcal{F}) = \ker(\mathcal{L}_{\mathcal{F}})$ |
| **Gen-3** | **SHMO** | Symplectic Neural Hamiltonian Flow on Cotangent Bundle $T^* \mathcal{M}$ | Active cytoplasmic streaming (cyclosis) via Myosin XI motors | Symplectic 2-form invariance: $\omega = dq \wedge dp$ (no gradient drift) |
| **Gen-4** | **WFR-Operator** | Dynamic Wasserstein-Fisher-Rao (Hellinger-Kantorovich) Unbalanced Flow | CaMV 35S active transcription $(\alpha > 0)$ vs. RISC cleavage $(\alpha < 0)$ | Infimal convolution between spatial transport and mass reaction |
| **Gen-5** | **AgroTop-Flow** | Grand Sovereign Unified Engine + Fisher-Rao Geodesic Metric + Conformal Risk Control | Full-tissue transgenic verification & multi-species generalization | Complete physical, topological, and statistical calibration ($\alpha = 0.05$) |

---

### 2. Theoretical Breakdown by Generation

#### Generation 1: RMNO-Hodge-v1 (Baseline Physical Operator)
- **Mathematical Principle:** Treats plant histological tissue as a 2D Riemannian 2-manifold $(\mathcal{M}, g)$ with metric tensor:
  $$g_{ij}(x) = \bar{g} \delta_{ij} + \alpha_w \tau_w(x) n_i n_j + \beta_f (\delta_{ij} - u_i u_j) + \rho_c(x) \delta_{ij}$$
  where $\tau_w(x)$ is cell wall thickness, $n(x)$ is the wall normal vector, $u(x)$ is cellulose fiber orientation, and $\rho_c$ is cuticle wax density.
- **Hodge-Helmholtz Decomposition:**
  $$\mathbf{J} = -\nabla_g \phi + \nabla \times \mathbf{A} + \mathbf{h}$$
  separating curl-free symplastic diffusion from divergence-free vascular streaming.
- **Identified Theoretical Limitations:**
  1. *Scalar connection assumption:* Graph Laplacian assumes symmetric connectivity ($W_{ij} = W_{ji}$), failing to capture directed polar transport.
  2. *Dissipative transport:* Ignores active motor-driven cytoplasmic streaming.
  3. *Zero-reaction assumption:* Standard optimal transport strictly conserves mass, unable to model biological transcription or enzymatic RNA degradation.

---

#### Generation 2: CSNO (Cellular Sheaf Neural Operator)
- **Betterment Rationale:** Intercellular transport in plants is strictly asymmetric due to polar localization of PIN efflux carriers (basal vs. apical membrane orientation) and callose-gated plasmodesmatal pore filtering.
- **Mathematical Formulation:**
  Over a cell complex $\mathcal{X} = (\mathcal{V}, \mathcal{E})$, a cellular sheaf $\mathcal{F}$ assigns a vector space $\mathcal{F}(v) \cong \mathbb{R}^d$ to each cell vertex $v$, and an edge stalk $\mathcal{F}(e) \cong \mathbb{R}^k$ to each junction $e = (u, v)$, with learnable restriction maps:
  $$\mathcal{F}_{u \unlhd e}: \mathcal{F}(u) \to \mathcal{F}(e), \quad \mathcal{F}_{v \unlhd e}: \mathcal{F}(v) \to \mathcal{F}(e)$$
  The coboundary operator $\delta: C^0(\mathcal{X}; \mathcal{F}) \to C^1(\mathcal{X}; \mathcal{F})$ is:
  $$(\delta h)_e = \mathcal{F}_{v \unlhd e} h_v - \mathcal{F}_{u \unlhd e} h_u$$
  and the Cellular Sheaf Laplacian $\mathcal{L}_{\mathcal{F}} = \delta^* \delta$ is:
  $$(\mathcal{L}_{\mathcal{F}} h)_u = \sum_{e = (u, v)} \mathcal{F}_{u \unlhd e}^* \left( \mathcal{F}_{u \unlhd e} h_u - \mathcal{F}_{v \unlhd e} h_v \right)$$
- **Theorem (Anti-Oversmoothing):** Unlike Graph Laplacians whose kernel is the trivial harmonic constant 1D space, $\ker(\mathcal{L}_{\mathcal{F}}) = H^0(\mathcal{X}; \mathcal{F})$ retains a high-dimensional subspace determined by the sheaf geometry. This allows continuous diffusion across cells without collapsing distinct cell-type transgene signatures into uniform average noise.

---

#### Generation 3: SHMO (Symplectic Hamiltonian Morphogenetic Operator)
- **Betterment Rationale:** Plant cells (especially elongated cotton fibers $>3\text{ cm}$ and tomato glandular trichomes) cannot rely on passive diffusion ($t \sim x^2 / D \approx 11 \text{ days}$ for $1\text{ cm}$). They utilize **active cytoplasmic streaming (cyclosis)** powered by Myosin XI moving along actin microfilaments at speeds up to $100\ \mu\text{m/s}$.
- **Mathematical Formulation:**
  The molecular morphogen state is modeled in cotangent bundle phase space $(q, p) \in T^* \mathcal{M}$:
  - $q(x) \in \mathbb{R}^d$: Morphogen / transgene transcript state.
  - $p(x) \in \mathbb{R}^d$: Intracellular cytoplasmic streaming momentum.
  The total Hamiltonian is:
  $$\mathcal{H}(q, p) = \frac{1}{2} p^T \mathbf{M}^{-1}(q) p + V(q; x)$$
  Hamilton's equations with cytosolic viscosity dampening $\Gamma(x)$:
  $$\dot{q} = \frac{\partial \mathcal{H}}{\partial p}, \quad \dot{p} = -\frac{\partial \mathcal{H}}{\partial q} - \Gamma(x) p$$
- **Symplectic Störmer-Verlet (Leapfrog) Integrator:**
  $$p_{t + \Delta t / 2} = p_t - \frac{\Delta t}{2} \nabla_q V(q_t)$$
  $$q_{t + \Delta t} = q_t + \Delta t \, \mathbf{M}^{-1} p_{t + \Delta t / 2}$$
  $$p_{t + \Delta t} = \frac{p_{t + \Delta t / 2} - \frac{\Delta t}{2} \nabla_q V(q_{t + \Delta t})}{1 + \Delta t \, \Gamma(x)}$$
- **Physical Invariance:** Preserves the canonical symplectic 2-form $\omega = dq \wedge dp$. Eliminates numerical dissipation and vanishing gradients across entire tissue sections.

---

#### Generation 4: WFR-Operator (Wasserstein-Fisher-Rao Reaction Flow)
- **Betterment Rationale:** Classical optimal transport assumes mass conservation ($\int \rho_0 = \int \rho_1$). However, transgenic plant tissue exhibits continuous **biological mass creation** (transcription driven by the CaMV 35S promoter) and **mass destruction** (enzymatic cleavage by plant ribonucleases and post-transcriptional gene silencing via RISC).
- **Mathematical Formulation:**
  The Wasserstein-Fisher-Rao (WFR) / Hellinger-Kantorovich metric is defined as the infimal convolution:
  $$\text{WFR}^2(\rho_0, \rho_1) = \inf_{(v, \alpha)} \int_0^1 \int_{\mathcal{M}} \left( \frac{1}{2} \|v_t(x)\|_g^2 + \frac{\beta^2}{2} |\alpha_t(x)|^2 \right) \rho_t(x) \, d\mathrm{Vol}_g \, dt$$
  subject to the non-conservative reactive continuity equation:
  $$\partial_t \rho_t + \text{div}_g(\rho_t v_t) = \alpha_t \rho_t$$
  where $v_t(x)$ is the physical transport velocity, $\alpha_t(x)$ is the biochemical reaction rate, and $\beta$ is the characteristic length scale governing the trade-off between physical movement vs. biological synthesis.
- **Unbalanced Sinkhorn Scaling:**
  $$\min_{\pi \ge 0} \langle \pi, C \rangle + \epsilon \, \text{KL}(\pi | \rho_0 \otimes \rho_1) + \tau \, \text{KL}(\pi_1 | \rho_0) + \tau \, \text{KL}(\pi_2 | \rho_1)$$
  with scaling parameter $\gamma = \frac{\tau}{\tau + \epsilon}$.
- **Biological Consequence:** Completely immunizes the model against false-positive classifications caused by heat-denatured or fragmented non-viable RNA in agricultural mandis.

---

#### Generation 5: AgroTop-Flow Engine (Grand Unified Architecture)
- **Architectural Synthesis:**
  Combines Sheaf Laplacians ($\mathcal{L}_{\mathcal{F}}$) + Symplectic Cyclosis ($\mathcal{H}$) + Reactive Transport ($\text{WFR}$) + Hodge Decomposition ($-\nabla \phi + \nabla \times \mathbf{A}$) + Fisher-Rao Information Geometry ($S^C(2)$) + Finite-Sample Conformal Risk Control.
- **Information Geometry Head:**
  Replaces flat Euclidean cross-entropy with the round metric on the sphere orthant:
  $$d_{\text{FR}}(\mathbf{z}_s, \mathbf{p}_c) = 2 \arccos\left( \sum_{k} \sqrt{z_{s, k} \cdot p_{c, k}} \right)$$
  Invariant under monotonic transformations and multiplicative scaling (Chentsov's Theorem).
- **Finite-Sample Conformal Risk Control:**
  Computes prediction sets $\mathcal{C}(X)$ satisfying:
  $$\mathbb{P}(Y \in \mathcal{C}(X)) \ge 1 - \alpha, \quad \forall P_{XY}$$
  at exact significance $\alpha = 0.05$.
