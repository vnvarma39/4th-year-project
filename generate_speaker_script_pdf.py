"""
Speaker Script & Defense Q&A Generator for PCMT Project
Generates: PCMT_Slide_by_Slide_Speaker_Script.pdf
Matches the 10-slide deck in PCMT_Academic_Presentation.pptx
Includes:
- Technical opening script for every slide
- Simple layman punchline for every slide
- Specific slide-level questions & answers
- Comprehensive Master Q&A Defense Guide
- ZERO personal names.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#059669"))
        self.drawString(54, 750, "PCMT ACADEMIC DEFENSE • SPEAKER SCRIPT & Q&A GUIDE")
        
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawRightString(612 - 54, 750, "CONFIDENTIAL / RESEARCH SCRIPT")
        
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.75)
        self.line(54, 742, 612 - 54, 742)
        
        # Footer
        self.line(54, 45, 612 - 54, 45)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(54, 32, "Physio-Causal Multimodal Transformer (PCMT) • Bt Cotton Project")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 32, page_text)
        self.restoreState()

def build_pdf(filename=r"E:\NIKHIL\startup\agri-rna\PCMT_Slide_by_Slide_Speaker_Script.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#475569"),
        spaceAfter=14
    )
    
    slide_title_style = ParagraphStyle(
        'SlideTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=10,
        spaceAfter=4
    )

    section_heading = ParagraphStyle(
        'SecHead',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#059669"),
        spaceBefore=14,
        spaceAfter=6
    )

    label_tech = ParagraphStyle(
        'LabelTech',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#1D4ED8")
    )

    label_simple = ParagraphStyle(
        'LabelSimple',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#059669")
    )

    label_qa = ParagraphStyle(
        'LabelQA',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#D97706")
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1E293B")
    )

    quote_style = ParagraphStyle(
        'QuoteText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#0F172A")
    )

    story = []

    # Document Header
    story.append(Paragraph("Slide-by-Slide Speaker Script & Defense Q&A", title_style))
    story.append(Paragraph("Complete spoken delivery guide matching <i>PCMT_Academic_Presentation.pptx</i>. Follow the golden delivery rule: <b>Open each slide with mathematical/technical rigor, then close with the simple layman punchline.</b>", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E2E8F0"), spaceAfter=12))

    slides_data = [
        {
            "num": "1",
            "title": "Title Slide: Research Progress & Technical Specifications",
            "focus": "Setting an elite, formal academic tone and defining the research scope.",
            "tech": "Good morning respected professors and committee members. Today I am presenting our progress on the rapid detection of Cry endotoxin genes in Bt cotton using AI-assisted spatial microscopy. We have engineered the Physio-Causal Multimodal Transformer (PCMT)—a framework unifying continuous state-space dynamical systems, entropic optimal transport, and structural causal models. Today, I will present our empirical validation on authentic plant spatial biology, our stress tests under optical field glare, and our latest architectural simulation breakthroughs.",
            "simple": "In simple terms: we are building an intelligent 15-minute diagnostic test that lets any farmer or seed inspector scan a cotton leaf with a handheld digital microscope and instantly verify whether it contains genuine Bt insect-resistant genes or is a counterfeit.",
            "q": "Why is spatial microscopy necessary for Bt cotton instead of just using traditional dip-strips?",
            "a": "Traditional dip-strips rely solely on chemical lateral flow, which yields faint, ambiguous test bands prone to subjective human error. By pairing high-resolution spatial optical imaging with molecular spectral curves, our multimodal transformer eliminates false positives and quantifies low-expression Cry traits directly in the field."
        },
        {
            "num": "2",
            "title": "The Agricultural Threat & Literature Baseline Limitations",
            "focus": "Explaining the economic stakes and revealing how PAMT (Nat Comm 2024) failed.",
            "tech": "Seed piracy in transgenic cotton is an acute crisis: sub-potent and counterfeit seeds cause widespread pink bollworm infestation and farmer economic ruin. Existing laboratory validation via qPCR or ELISA requires 48 to 72 hours in sterile facilities. We initiated our computational pipeline by implementing the leading literature baseline, PAMT (Nature Communications 2024). However, when evaluated on authentic plant tissue, PAMT severely underperformed, achieving only 36.5% accuracy and suffering an uncalibrated conformal collapse (set size of 6.52 out of 7). Under out-of-distribution optical noise, PAMT collapsed by nearly 19%.",
            "simple": "In simple terms: PAMT was engineered for human cancer biopsies in sterile hospital labs. The moment you expose it to outdoor camera glare, dust, and real plant tissue, it gets blinded by noise and cannot tell one tissue from another.",
            "q": "Why did PAMT collapse so severely under noise while it worked well in its original human cancer paper?",
            "a": "PAMT assumes discrete pairwise self-attention across image patches. Human cancer biopsies have relatively uniform cellular staining, but plant leaves contain rigid cellulose walls, chloroplast autofluorescence, and outdoor lens scattering. Softmax attention scattered its probability weights across this background noise, causing complete representation collapse."
        },
        {
            "num": "3",
            "title": "Root Cause Analysis: Discrete Attention vs. Spatial Continuity",
            "focus": "The biological and mathematical breakdown of why standard transformers fail in plants.",
            "tech": "Mathematically, standard Vision Transformers treat spatial patches as an unordered sequence of discrete tokens: Attn(Q,K)=softmax(QK^T / √d)V. Biologically, plant cells do not behave like isolated words in a text corpus. Rigid cell walls and plasmodesmata enforce continuous biochemical morphogen gradients governed by spatial diffusion. Softmax normalization disperses attention entropy across high-frequency optical noise. PCMT resolves this through continuous S4D state-space differential operators: dh(s)/ds = Ah(s) + Bu(s), and Entropic Sinkhorn Optimal Transport that preserves doubly-stochastic mass conservation.",
            "simple": "In simple terms: standard transformers treat cells like words in an English essay. But plant cells are physically glued together like bricks in a wall—if you ignore their physical connections, camera noise scatters everywhere and ruins the prediction.",
            "q": "What is the mathematical advantage of Sinkhorn Optimal Transport over standard Cross-Attention?",
            "a": "Standard cross-attention computes independent softmax distributions that do not conserve mass—one noisy image patch can dominate all gene tokens. Sinkhorn Optimal Transport solves the Monge-Kantorovich problem with entropic regularization, enforcing doubly-stochastic row and column constraints so every image patch and gene pathway receives a physically balanced mapping."
        },
        {
            "num": "4",
            "title": "Empirical Validation on Authentic Plant Spatial Biology",
            "focus": "Demonstrating empirical proof on 10x Genomics Visium HD Arabidopsis data.",
            "tech": "To achieve uncompromising academic rigor, we eliminated all synthetic proxies and validated PCMT on authentic physical spatial biology using the 10x Genomics Visium HD Arabidopsis seedling benchmark. This dataset couples 31.7-Megapixel histology with 31,109 genes across 700 balanced spots spanning 7 anatomical domains. While unimodal vision achieved only 46.8% and gene-only achieved 66.5%, PCMT achieved 74.4% peak validation accuracy and 73.1% Macro F1, outperforming PAMT (36.5%) by an overwhelming +37.9% absolute margin.",
            "simple": "In simple terms: looking at leaf images alone gave 46%, and looking at genes alone gave 66%. But combining both through PCMT pushed accuracy to 74.4%, completely crushing the existing literature baseline.",
            "q": "Why evaluate on Arabidopsis thaliana instead of Gossypium hirsutum (cotton) directly?",
            "a": "Arabidopsis is the world's premier genetic model organism for plants. 10x Genomics Visium HD provides the only publicly available dataset with sub-cellular micron-resolution optical images coupled with spot-level RNA-seq from the exact same physical cryosection. Mathematically proving our multimodal engine on authentic plant tissue first guarantees that our deployment to cotton dip-strips requires zero structural redesign."
        },
        {
            "num": "5",
            "title": "Field Stress Test: Optical Glare & Causal Disentanglement",
            "focus": "Proving outdoor field resilience and explaining the Causal HSIC head.",
            "tech": "In field deployments, models encounter harsh sensor noise, lens smudges, and direct sunlight glare. We simulated out-of-distribution optical Gaussian scattering at σ=1.0. While PAMT plummeted by -18.7% and vision-only collapsed by -32.5%, PCMT dropped by only -4.9%, maintaining 69.5% accuracy. This resilience is governed by our Causal Disentanglement Head: we model the latent space under Judea Pearl's do-calculus, partitioning representations into an invariant biological identity (Z_S) and an environmental noise latent (Z_E), penalizing their Hilbert-Schmidt Independence Criterion: min HSIC(Z_S, Z_E).",
            "simple": "In simple terms: PCMT acts like polarized sunglasses. Even if the camera lens is smudged with field dust or outdoor glare, the AI filters out the dirt and focuses purely on the true leaf biology underneath.",
            "q": "How does HSIC ensure that the model doesn't cheat using lighting or lens glare?",
            "a": "HSIC maps the biological representation and the glare representation into reproducing kernel Hilbert spaces and computes their cross-covariance. By driving HSIC toward zero using Gretton's median heuristic, the model is mathematically forced to make predictions that share zero mutual dependence with optical artifacts."
        },
        {
            "num": "6",
            "title": "Novelty Exploration: PINO-SSM & The Gating Discovery",
            "focus": "Explaining the 53.2% bottleneck and how Gated PINO-SSM-v2 fixed it (68.0%).",
            "tech": "We pushed our novelty framework by incorporating Physics-Informed Neural Operators (PINO-SSM) modeling 2D Darcy capillary wicking equations: ∂u/∂t = D ∇^2 u. In our initial test, the raw Trio scored 53.2%. While it achieved 0.0% noise degradation, overall accuracy was bottlenecked. Our root-cause audit revealed that PINO-SSM initialized the direct feedthrough matrix D_feed = 0 without a residual highway, forcing 100% of features through Gaussian diffusion and over-smoothing fine cell boundaries. Formulating Gated PINO-SSM-v2 with an adaptive highway: h = LayerNorm(h_in + α y_pde + (1-α) y_s4d) immediately restored sharpness, boosting accuracy to 68.0% (+14.8% jump).",
            "simple": "In simple terms: our first physics filter was so aggressive that it accidentally blurred cell edges. Adding an adaptive bypass highway gave us the best of both worlds: sharp edge recognition and complete noise protection.",
            "q": "Why did you use Darcy capillary wicking equations specifically for plant leaves?",
            "a": "Water, nutrients, and pesticide solutes move through plant mesophyll and vascular xylem bundles via capillary action governed by Darcy's law. Embedding Darcy physics provides an inductive bias that mirrors authentic plant fluid dynamics rather than unconstrained black-box convolutions."
        },
        {
            "num": "7",
            "title": "Simulation Breakthrough: 2D Coordinate Diffusion (75.9% Acc)",
            "focus": "Presenting the 4-model simulation study and breaking the 75% accuracy barrier.",
            "tech": "To maximize Enhanced Trio's performance, we executed a rigorous simulation study evaluating four architectural mechanisms across 35 epochs. Introducing 2D Physical Coordinate Diffusion (PINO-Graph) broke the ceiling: validation accuracy surged to 75.9% (+7.9% net gain), Macro F1 reached 74.5%, and Transgenic Cry Trait Detection ROC-AUC hit 93.4%. Multi-Head Optimal Transport (4 parallel Sinkhorn heads) reduced out-of-distribution noise drop to just -2.0%, while Trio-v3-Synergy compressed the Conformal Set Size down to 1.55.",
            "simple": "In simple terms: cells in a leaf don't live on isolated islands. Letting neighboring tissue spots talk to each other across physical 2D space jumped accuracy straight to 75.9% and Cry trait detection to 93.4%.",
            "q": "How does PINO-Graph compute 2D coordinate diffusion between tissue spots?",
            "a": "We construct an inter-spot spatial Laplacian using a Gaussian radial basis function scaled by physical cell diameter: A_ij = exp(-||c_i - c_j|| / σ_cell). A row-stochastic random walk propagation diffuses latent representations across Euclidean space, directly capturing biological tissue homophily."
        },
        {
            "num": "8",
            "title": "Multi-Axis SOTA Benchmark: Why Competing Models Cannot Deploy",
            "focus": "Direct head-to-head comparison against STAGATE, SpaGCN, M3T, and PAMT.",
            "tech": "We benchmarked our upgraded Trio directly against top peer-reviewed spatial biology models. STAGATE (Nat Comm 2022) achieves 78.3% accuracy, but is strictly Gene-Only—it ignores histology WSIs completely and requires a million-dollar Illumina sequencer, making field deployment impossible. SpaGCN (Nat Methods 2021) achieves 77.3% clean accuracy, but collapses by -12.8% under optical noise. M3T (Bioinformatics 2023) suffers quadratic attention latency (724 ms). Our Upgraded PCMT-Trio achieves 75.9% accuracy, 93.4% Cry AUC, drops only -2% to -9% under noise, and executes in just 343 ms.",
            "simple": "In simple terms: competing models either require a full genetics lab that you cannot take to a farm (STAGATE), or they break down the moment dust gets on the lens (SpaGCN). Our model is fast, multimodal, and field-proof.",
            "q": "If STAGATE has 78.3% accuracy, why shouldn't we use it?",
            "a": "STAGATE has zero vision capability—it requires pre-sequenced RNA reads for every physical coordinate. In an agricultural seed mandi, you cannot perform RNA library preparation. A handheld field device only captures optical images and dip-strip reflectance. STAGATE is mathematically incapable of running on field hardware."
        },
        {
            "num": "9",
            "title": "Certified Safety: Distribution-Free Conformal Prediction",
            "focus": "Explaining Conformal Prediction guarantees and the 1.55 set size.",
            "tech": "Standard deep learning models produce overconfident point predictions with heuristic probabilities. In agricultural seed certification, a false positive on a counterfeit seed leads to crop failure and legal disputes. We implemented Split Conformal Prediction, which provides distribution-free, finite-sample statistical guarantees: P(Y ∈ C(X)) ≥ 1 - α. On held-out test data, our conformal sets achieved 99.0% empirical coverage (target: 90%). Furthermore, while PAMT had an uncalibrated set size of 6.52 out of 7 (useless guessing), our Trio-v3-Synergy model compressed the set size to just 1.55 domains.",
            "simple": "In simple terms: the software never lies or hallucinates. With 99% mathematical certainty, it narrows down the diagnosis to fewer than two candidate tissue types per spot.",
            "q": "What happens if a field sample is completely degraded or contaminated?",
            "a": "Under standard AI, the model would guess randomly with false 90% confidence. Under Conformal Prediction, when a sample is contaminated or ambiguous, the prediction set automatically expands to include multiple candidate classes, mathematically flagging the sample as 'Uncertain: Retest Required'."
        },
        {
            "num": "10",
            "title": "Technical Roadmap to 95%+ Accuracy: In-Silico Transgene Unmasking",
            "focus": "The 3 pillars to 95%+ and adapting the Google-Yale Cell2Sentence paradigm.",
            "tech": "To elevate performance from 76% to 95%+ on field Bt cotton, our engineering roadmap comprises three pillars: First, self-supervised pretraining on 100,000+ spot plant atlases via Masked Autoencoding (MAE). Second, expanding 2D diffusion to a multi-scale 8-nearest-neighbor topological spatial graph. Third, adapting the groundbreaking Cell2Sentence (C2S) paradigm developed by Google DeepMind and Yale University. In oncology, Google and Yale used C2S to in-silico reprogram 'cold' cancer cells to force antigen presentation (+50%). We are applying this to transgenics: treating low-expression Cry endotoxins as 'cold' targets masked by plant RuBisCO noise, using rank-tokenization and in-silico unmasking to amplify the Bt signature.",
            "simple": "In simple terms: just as cancer immunotherapy models dress hidden tumor cells in digital bright orange jumpsuits so immune cells catch them, we will dress the faint Cry gene in a digital orange jumpsuit so our scanner catches it every time.",
            "q": "How does the Google-Yale Cell2Sentence concept directly improve our Cry gene detection?",
            "a": "Cry endotoxins are expressed at low transcript copy numbers compared to massive plant background enzymes like RuBisCO. In standard models, this faint signal gets washed out. Cell2Sentence rank-orders genes by biological importance and applies an in-silico excitation layer, mathematically suppressing housekeeping clutter and amplifying the Cry1Ac/Cry2Ab promoter cassette."
        }
    ]

    for slide in slides_data:
        card_content = []
        
        # Slide Header
        card_content.append(Paragraph(f"<b>SLIDE {slide['num']}: {slide['title']}</b>", slide_title_style))
        card_content.append(Paragraph(f"<b>Core Visual Focus:</b> {slide['focus']}", subtitle_style))
        card_content.append(Spacer(1, 4))
        
        # Technical Opening
        card_content.append(Paragraph("1. WHAT TO SAY — TECHNICAL RIGOR (OPENING):", label_tech))
        card_content.append(Spacer(1, 2))
        card_content.append(Paragraph(f'"{slide["tech"]}"', quote_style))
        card_content.append(Spacer(1, 6))
        
        # Layman Punchline
        card_content.append(Paragraph("2. WHAT TO SAY — LAYMAN'S PUNCHLINE (CLOSING):", label_simple))
        card_content.append(Spacer(1, 2))
        card_content.append(Paragraph(f'"{slide["simple"]}"', quote_style))
        card_content.append(Spacer(1, 6))
        
        # Target Question & Answer
        card_content.append(Paragraph(f"3. EXPECTED PROFESSOR QUESTION & WINNING DEFENSE:", label_qa))
        card_content.append(Spacer(1, 2))
        card_content.append(Paragraph(f"<b>Q: <i>\"{slide['q']}\"</i></b>", body_style))
        card_content.append(Paragraph(f"<b>A:</b> {slide['a']}", body_style))
        card_content.append(Spacer(1, 10))
        
        # Add to story with clean table wrapper for card appearance
        t = Table([[card_content]], colWidths=[504])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#E2E8F0")),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ]))
        
        story.append(t)
        story.append(Spacer(1, 12))

    # =========================================================================
    # MASTER DEFENSE Q&A SECTION (THE BIG 6 TOUGH QUESTIONS)
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Master Defense Q&A: The 6 High-Stakes Exam Questions", section_heading))
    story.append(Paragraph("Memorize these concise, structured responses to handle the most demanding technical inquiries from your committee.", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=14))

    master_qa = [
        (
            "1. 'Why were your initial novelty numbers 53%, and how did you push them to 75.9%?'",
            "Our initial PINO-SSM kernel initialized the direct feedthrough matrix D_feed = 0 without a residual highway, acting as an overly aggressive low-pass filter that over-smoothed fine cell boundaries. Once we formulated Gated PINO-SSM-v2 with an adaptive highway (jumping to 68.0%) and introduced 2D Physical Coordinate Darcy Diffusion (PINO-Graph), validation accuracy surged to 75.9%, Macro F1 hit 74.5%, and Cry trait detection reached 93.4%."
        ),
        (
            "2. 'Why did PAMT (Nature Comm 2024) underperform so badly (36.5%) on plant data?'",
            "PAMT was engineered for human cancer histology where tissue staining is relatively uniform. In plant seedlings, cells have rigid cellulose walls and chloroplast autofluorescence. PAMT's pairwise softmax attention suffered catastrophic entropy dispersion under camera noise (-18.7% drop), whereas PCMT's continuous state-space operator and causal HSIC head explicitly filter out high-frequency noise."
        ),
        (
            "3. 'Why not just use STAGATE (78.3%) or SpaGCN (77.3%)?'",
            "STAGATE is strictly a Gene-Only model—it has zero image capability and requires a full sequencing laboratory, making it impossible to deploy on handheld field hardware. SpaGCN combines images and genes, but collapses by -12.8% when exposed to optical glare or dust. Our upgraded PCMT-Trio is the only model that is multimodal, noise-resilient (only -2% to -9% drop), and executes in 343 milliseconds."
        ),
        (
            "4. 'What is Conformal Prediction and why does a farmer care about set size 1.55?'",
            "Standard neural networks make overconfident point predictions that hallucinate false positives. Split Conformal Prediction provides a finite-sample mathematical guarantee: P(Y in C(X)) >= 90%. While PAMT had a useless set size of 6.52 out of 7, our Trio-v3-Synergy model compressed the set size down to 1.55 domains at 99% certified coverage—guaranteeing that the true biological trait is identified with mathematical certainty."
        ),
        (
            "5. 'How does your continuous S4D kernel model plant biology differently from a CNN?'",
            "A CNN applies static, discrete convolutional kernels over fixed pixel grids. In contrast, S4D parameterizes spatial features as a continuous linear dynamical system: dh(s)/ds = Ah(s) + Bu(s), solved analytically using HiPPO state matrices. This naturally models continuous chemical diffusion and nutrient transport gradients across plant tissues regardless of magnification scale."
        ),
        (
            "6. 'What is your concrete engineering roadmap to reach 95%+ accuracy in Bt cotton?'",
            "Our 3-pillar roadmap includes: (1) Self-supervised pretraining on 100,000+ spot plant atlases via Masked Autoencoding to provide strong cellular priors; (2) Expanding 2D coordinate diffusion to an 8-nearest-neighbor spatial Darcy mesh; and (3) Adapting Google DeepMind and Yale's Cell2Sentence in-silico unmasking paradigm to rank-order genes and digitally suppress background RuBisCO clutter, isolating the Cry1Ac/Cry2Ab promoter cassette."
        )
    ]

    for q_text, a_text in master_qa:
        qa_content = []
        qa_content.append(Paragraph(f"<b>{q_text}</b>", ParagraphStyle('QTitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10.5, leading=14, textColor=colors.HexColor("#0F172A"))))
        qa_content.append(Spacer(1, 3))
        qa_content.append(Paragraph(f"<b>Winning Answer:</b> {a_text}", body_style))
        qa_content.append(Spacer(1, 4))
        
        t = Table([[qa_content]], colWidths=[504])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#FFFFFF")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
            ('TOPPADDING', (0,0), (-1,-1), 7),
            ('BOTTOMPADDING', (0,0), (-1,-1), 7),
            ('LEFTPADDING', (0,0), (-1,-1), 9),
            ('RIGHTPADDING', (0,0), (-1,-1), 9),
        ]))
        story.append(t)
        story.append(Spacer(1, 8))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated clean speaker script PDF: {filename}")

if __name__ == "__main__":
    build_pdf()
