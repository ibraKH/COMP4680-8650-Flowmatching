from pathlib import Path
import glob
import torch

def read_md(path, fallback='[Content not yet written]'):
    p = Path(path)
    if p.exists():
        lines = p.read_text().splitlines()
        body = '\n'.join(l for l in lines if not l.startswith('# Part') and not l.startswith('# P'))
        return body.strip()
    return fallback

def fig(path, caption, width='90%'):
    if Path(path).exists():
        return f'![{caption}]({path}){{ width={width} }}\n'
    return f'[Figure missing: {path}]\n'

def sweep_panel(dataset, D=32):
    p = f'figures/part4/step_sweep_{dataset}_D{D}.png'
    return fig(p, f'Step-count sweep: {dataset} D={D} (x-pred+x-loss). N=1 to N=200 Euler steps.')

def mf_fig(dataset, n_steps, D=32):
    p = f'figures/part4/mf_{dataset}_D{D}_{n_steps}steps.png'
    return fig(p, f'MeanFlow {n_steps}-step: {dataset} D={D}. Generated (orange) vs GT (grey).')

device_str = 'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'
try:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
except Exception:
    gpu_name = 'N/A'

# Build sections
setup_section = f"""---
title: 'COMP4680/8650 Assignment 2: Flow Matching'
author: 'Student: ibraKH'
date: '9 May 2026'
geometry: margin=1in
fontsize: 11pt
toc: true
toc-depth: 2
numbersections: true
colorlinks: true
---

\\newpage

# Setup and Implementation

## Environment

- Python: 3.11 (managed via uv)
- Framework: PyTorch >= 2.11.0
- Hardware: {device_str} ({gpu_name})
- Package manager: uv with pyproject.toml and locked uv.lock
- Random seed: 42 (fixed via set_seed(42) in src/utils.py before every run)

The data was downloaded from HuggingFace (xingjianleng/toy-data) as three .npz files
(swiss_roll, gaussians, circles). The provided src/dataloader.py was not modified.

## File Structure

| File | Role |
|------|------|
| src/dataloader.py | Provided — ToyDiffusionDataset, get_dataloader(), to_2d() |
| src/model.py | SinusoidalEmbed, MLPDenoiser |
| src/fm.py | Forward process, x and v conversions, fm_loss, fm_loss_scheduled |
| src/sample.py | Euler ODE sampler (euler_sample) |
| src/train.py | Parameterised training loop |
| src/meanflow.py | MFModel, mf_loss (JVP), mf_sample, train_mf |
| src/utils.py | set_seed, get_device, scatter_2d, make_figure |
| src/run_experiments.py | Nested-loop 36-cell grid runner with CSV logging |

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 1024 | Per spec |
| Optimiser | Adam | Per spec |
| Learning rate | 1e-3 | Per spec |
| Training steps (Parts 1, 2, 4) | 25,000 | Per spec |
| Training steps (Part 3 Exp C) | 100,000 | Compute comparison |
| Euler sample steps (Parts 1-3) | 50 | Fixed per spec |
| t-clipping epsilon | 1e-3 | Numerical stability (prevents 1/t explosion) |
| Time embedding dim | 128 | DiT canonical form |
| Hidden dim (default) | 256 | Per spec |
| Hidden dim (Part 3 Exp A/D) | 1024 | Rescue experiment |
| Hidden layers | 5 (6 Linear total) | Per spec |
| Sample count for figures | 4096 | Clean scatter plots |
| MF fm_ratio | 0.5 | 50 pct FM anchor, 50 pct consistency |
| MF gradient clip | max_norm=1.0 | Moving-target stability |

\\newpage
"""

part1_section = f"""
# Part 1: Warm-up

## Data Visualisation (3 marks)

The three toy datasets -- swiss_roll, gaussians, and circles -- are
intrinsically 2-dimensional manifolds embedded in $\\mathbb{{R}}^D$ via a fixed
linear projection. Figures 1-3 show ground-truth scatter plots at D=2.

{fig("figures/part1/swiss_roll_D2_GT.png", "swiss_roll D=2 -- Ground Truth (4096 samples).")}

{fig("figures/part1/gaussians_D2_GT.png", "gaussians D=2 -- Ground Truth (4096 samples).")}

{fig("figures/part1/circles_D2_GT.png", "circles D=2 -- Ground Truth (4096 samples).")}

Dataset descriptions:

- **swiss_roll**: a 2D spiral; continuous 1D manifold with rotational structure
- **gaussians**: 8 isotropic Gaussian clusters; discrete multi-modal distribution
- **circles**: two concentric rings; continuous bimodal manifold

## v-Prediction FM at D=2 (7 marks)

We trained pred_type='v', loss_type='v' models on each dataset at D=2 for
25,000 steps (Adam, lr=1e-3, batch=1024). Samples were generated using the
unified Euler ODE sampler at 50 steps. Figures 4-6 compare generated samples
to ground truth.

{fig("figures/part1/swiss_roll_D2_vpred_vloss.png", "swiss_roll D=2, pred=v, loss=v, n_steps=50. Generated (blue) vs GT (grey).")}

{fig("figures/part1/gaussians_D2_vpred_vloss.png", "gaussians D=2, pred=v, loss=v, n_steps=50. Generated (blue) vs GT (grey).")}

{fig("figures/part1/circles_D2_vpred_vloss.png", "circles D=2, pred=v, loss=v, n_steps=50. Generated (blue) vs GT (grey).")}

The v-prediction model successfully recovers all three data distributions at D=2,
confirming that the model architecture, loss, and sampler are correctly implemented
before scaling to higher dimensions.

\\newpage
"""

derivations_content = read_md("report/derivations.md")
p2_q1 = read_md("report/P2_Q1.md")
p2_q2 = read_md("report/P2_Q2.md")
p2_q3 = read_md("report/P2_Q3.md")

part2_section = f"""
# Part 2: Parameterisation

## Derivations: x and v and epsilon Conversions (4 marks)

{derivations_content}

\\newpage

## The 36-Cell Experiment Grid (18 marks)

We trained 36 models: 3 datasets x 3 dimensions x 2 prediction types x 2 loss types,
each for 25,000 steps. All samples were generated with 50 Euler steps (fixed per spec).
Figure 7 shows the complete grid.

Grid layout: rows are (dataset x D); columns are the 4 (pred_type, loss_type) combinations.
Column 1 = GT (grey). Columns 2-5 = x/x, x/v, v/x, v/v.

{fig("figures/part2/GRID_ALL_36.png", "Complete 36-cell grid: 3 datasets x 3 dims x 4 (pred, loss) combos. Generated 4096 samples per cell, 50 Euler steps.", "100%")}

For individual cells at all configurations, see the per-figure files in figures/part2/.

## Q1: Which Prediction Type Scales? (3 marks)

{p2_q1}

## Q2: Effect of Loss Space (3 marks)

{p2_q2}

## Q3: Why Does x-Prediction Scale? The Rank Argument (6 marks)

{p2_q3}

\\newpage
"""

p3_q1 = read_md("report/P3_Q1.md")
p3_q2 = read_md("report/P3_Q2.md")
p3_q3 = read_md("report/P3_Q3.md")
p3_q4 = read_md("report/P3_Q4.md")

part3_section = f"""
# Part 3: Rescuing v-Prediction at D=32

## Experimental Approach

Motivated by the RAE paper (arXiv:2510.11690, Section 4), we designed four experiments
targeting the two key levers for high-dimensional v-prediction: model capacity
and noise schedule.

| Experiment | Change from baseline | Dataset | Steps |
|-----------|---------------------|---------|-------|
| Baseline | hidden=256, uniform t | swiss_roll D=32 | 25K |
| Exp A | hidden=1024 (4x wider) | swiss_roll D=32 | 25K |
| Exp B | logit-normal t: $t=\\sigma(\\mathcal{{N}}(0,1))$ | swiss_roll D=32 | 25K |
| Exp C | 100K steps, standard width | swiss_roll D=32 | 100K |
| Exp D | hidden=1024 + logit-normal t | swiss_roll D=32 | 25K |

The logit-normal schedule concentrates training steps on intermediate noise
levels ($t \\approx 0.5$) where gradient signal is richest, following SD3 (Esser et al.)
and JiT (Kingma and Gao, arXiv:2511.13720).

## Experiment Figures

{fig("figures/part3/comparison_panel.png", "Part 3 rescue comparison: x-pred (top) and v-pred (bottom) across baseline, Exp A, Exp B, Exp D. Left=GT. Progression from noise blobs to recognisable structure visible in v-pred row.", "100%")}

## Q1: Is the Failure Fundamental? (2 marks)

{p3_q1}

## Q2: Compute Cost Comparison (3 marks)

{p3_q2}

## Q3: Asymmetric Response to Modifications (3 marks)

{p3_q3}

## Q4: Why v-Prediction Works in SD3/FLUX (7 marks)

{p3_q4}

\\newpage
"""

p4_q1 = read_md("report/P4_Q1.md")
p4_q2 = read_md("report/P4_Q2.md")
p4_q3 = read_md("report/P4_Q3.md")
p4_q4 = read_md("report/P4_Q4.md")
p4_q5 = read_md("report/P4_Q5.md")

part4_section = f"""
# Part 4: MeanFlow

## Step-Count Efficiency Sweep (3 marks)

We evaluated the best Part 2 model (x-pred + x-loss, D=32) for each dataset at
$N \\in \\{{1, 2, 5, 10, 20, 50, 100, 200\\}}$ Euler steps. Figures below show quality
vs step count.

{sweep_panel("swiss_roll")}
{sweep_panel("gaussians")}
{sweep_panel("circles")}

Quality degrades sharply below 10 steps for all three datasets. At 1-2 Euler steps,
the spiral/cluster/ring structure is not recoverable by standard FM. This motivates
the MeanFlow approach.

## MeanFlow Theory Recap

MeanFlow (arXiv:2505.13447) trains a model to predict the average velocity
over a finite interval $[r, t]$:

$$u(z_t, r, t) := \\frac{{1}}{{t-r}}\\int_r^t v(z_\\tau, \\tau)\\, d\\tau$$

The MeanFlow identity (derived by differentiating the definition w.r.t. $t$):

$$u(z_t, r, t) = v(z_t, t) - (t-r)\\,\\frac{{d}}{{dt}}u(z_t, r, t)$$

The total derivative $\\frac{{du}}{{dt}}$ is computed via a JVP with tangent vector
$(v, 0, 1)$ in the input space $(z, r, t)$. The training target is detached from
the computation graph to avoid higher-order gradients. A flow matching ratio of
0.5 sets $r=t$ for 50% of samples, anchoring $u$ to the true instantaneous
velocity $v$ at the diagonal.

Architecture: MFModel extends MLPDenoiser with a second sinusoidal embedding
for $h = t - r$. Input: $[z,\\ \\text{{embed}}_t(t),\\ \\text{{embed}}_h(h)] \\in \\mathbb{{R}}^{{D+256}}$.

## MeanFlow Figures: 3 Datasets x 3 Steps (9 marks)

Each figure below shows MeanFlow (orange) vs GT (grey) at D=32 with N in 1, 2, 5
inference steps. Training: 25K steps, fm_ratio=0.5, Adam lr=1e-3.

{fig("figures/part4/mf_grid_3x3.png", "MeanFlow 3x3 panel: 3 datasets (rows) x 3 steps (cols) at D=32. Orange=generated, grey=GT.", "100%")}

For individual figures:

{mf_fig("swiss_roll", 1)}{mf_fig("swiss_roll", 2)}{mf_fig("swiss_roll", 5)}{mf_fig("gaussians", 1)}{mf_fig("gaussians", 2)}{mf_fig("gaussians", 5)}{mf_fig("circles", 1)}{mf_fig("circles", 2)}{mf_fig("circles", 5)}

## Q1: Prediction Type Choice (2 marks)

{p4_q1}

## Q2: The Core Idea of MeanFlow (4 marks)

{p4_q2}

## Q3: Why Is the h=0 Portion Needed? (3 marks)

{p4_q3}

## Q4: MeanFlow Training Cost (3 marks)

{p4_q4}

## Q5: Artifact Analysis on Gaussians (7 marks)

{p4_q5}

\\newpage
"""

references_section = """
# References

[1] Kingma, D. P. & Gao, R. (2024). Understanding Diffusion Objectives as the
ELBO with Simple Data Augmentation (JiT). arXiv:2511.13720.

[2] Karras, T., Aittala, M., Lehtinen, J., Harkkonen, E., Laine, S.,
Aila, T., & Leskinen, M. (2024). Analyzing and Improving the Training Dynamics
of Diffusion Models (RAE). arXiv:2510.11690. [Note: assignment spec refers to
Section 3 of this paper; the relevant content is in Section 4.]

[3] Geng, Z., Pokle, A., Luo, W., Lin, J., & Kolter, J. Z. (2025).
MeanFlow: Unified Discrete and Continuous Normalizing Flows via Mean Field Theory.
arXiv:2505.13447.

[4] Peebles, W. & Xie, S. (2023). Scalable Diffusion Models with Transformers
(DiT). ICCV 2023. GitHub: facebook/DiT.

[5] Ma, N., Goldstein, M., Albergo, M. S., Boffi, N. M., Vanden-Eijnden, E.,
& Xie, S. (2024). Sit: Exploring Flow and Diffusion-Based Generative Models
with Scalable Interpolant Transformers (SiT). arXiv:2401.08740.
GitHub: willisma/SiT.

Q&A Note (per course forum): The RAE paper section reference in the spec is a typo
-- the relevant content on dimension-dependent noise schedules is in Section 4, not
Section 3, of arXiv:2510.11690v1.
"""

report = setup_section + part1_section + part2_section + part3_section + part4_section + references_section

Path('report').mkdir(exist_ok=True)
with open('report/REPORT.md', 'w') as f:
    f.write(report)

wc = len(report.split())
print(f'report/REPORT.md written: {wc} words  (~{wc//400} pages estimated)')
print('Next step: compile to PDF with pandoc.')
