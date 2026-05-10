"""
Rebuild figures/part3/comparison_panel.png as a 2 × 6 grid:
  Rows: x-pred (blue), v-pred (red)
  Cols: GT | Baseline | Exp A | Exp B | Exp C | Exp D

Cells sourced from:
  - checkpoints (inference)  → xpred-baseline, xpred-B, xpred-C
                                vpred-baseline, vpred-B, vpred-C
  - crop right-half of existing 2-panel PNG  → expA and expD for both rows
"""
import sys, re
sys.path.insert(0, 'src')

from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from dataloader import ToyDiffusionDataset
from model import MLPDenoiser
from sample import euler_sample
from utils import set_seed

set_seed(42)
CKPT   = Path('checkpoints')
FIG3   = Path('figures/part3')
DS, D  = 'swiss_roll', 32
NS     = 4096    # samples
NSTEPS = 50      # Euler steps
device = torch.device('cpu')


# ── helpers ────────────────────────────────────────────────────────────────────

def load_model(path, hidden=256):
    m = MLPDenoiser(D, hidden=hidden).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m

def infer(model, pred_type):
    with torch.no_grad():
        z = euler_sample(model, (NS, D), NSTEPS, pred_type, device=device)
    ds = ToyDiffusionDataset(DS, dim=D)
    return ds.to_2d(z.cpu().numpy())

def gt_2d():
    ds = ToyDiffusionDataset(DS, dim=D)
    data = ds.data.numpy()[:NS]
    return ds.to_2d(data)


def crop_generated_from_png(png_path: Path) -> np.ndarray:
    """Extract the right-half (Generated) panel from a saved 2-panel figure."""
    img = np.array(Image.open(png_path).convert('RGB'))
    h, w = img.shape[:2]
    return img[:, w // 2:, :]   # right half


# ── assemble scatter data for each cell ───────────────────────────────────────

gt = gt_2d()
margin = 0.4
xlim = (gt[:, 0].min() - margin, gt[:, 0].max() + margin)
ylim = (gt[:, 1].min() - margin, gt[:, 1].max() + margin)

def make_scatter_img(pts2d, color, size_inches=2.8):
    """Return a square numpy image array for embedding in the panel."""
    fig, ax = plt.subplots(figsize=(size_inches, size_inches))
    ax.scatter(pts2d[:, 0], pts2d[:, 1], s=2, alpha=0.4, color=color,
               rasterized=True, linewidths=0)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('equal'); ax.axis('off')
    plt.tight_layout(pad=0)
    tmp = '/tmp/_cell_tmp.png'
    plt.savefig(tmp, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return np.array(Image.open(tmp).convert('RGB'))

def make_gt_img(size_inches=2.8):
    fig, ax = plt.subplots(figsize=(size_inches, size_inches))
    ax.scatter(gt[:, 0], gt[:, 1], s=2, alpha=0.35, color='dimgrey',
               rasterized=True, linewidths=0)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('equal'); ax.axis('off')
    plt.tight_layout(pad=0)
    tmp = '/tmp/_gt_tmp.png'
    plt.savefig(tmp, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return np.array(Image.open(tmp).convert('RGB'))


print('Building cell images...')

# x-pred row (blue)
xbase = make_scatter_img(infer(load_model(CKPT/'swiss_roll_D32_xpred_xloss.pt'),    'x'), 'steelblue')
xpA   = crop_generated_from_png(FIG3/'swiss_roll_D32_xpred_xloss_expA.png')
xpB   = make_scatter_img(infer(load_model(CKPT/'swiss_roll_D32_xpred_xloss_expB.pt'), 'x'), 'steelblue')
xpC   = make_scatter_img(infer(load_model(CKPT/'swiss_roll_D32_xpred_xloss_expC.pt'), 'x'), 'steelblue')
xpD   = crop_generated_from_png(FIG3/'swiss_roll_D32_xpred_xloss_expD.png')

# v-pred row (tomato/red)
vpB_ckpt  = CKPT/'swiss_roll_D32_vpred_vloss_expB.pt'
vpC_ckpt  = CKPT/'swiss_roll_D32_vpred_vloss_expC.pt'

vbase = make_scatter_img(infer(load_model(CKPT/'swiss_roll_D32_vpred_vloss.pt'),   'v'), 'tomato')
vpA   = crop_generated_from_png(FIG3/'swiss_roll_D32_vpred_vloss_expA.png')
vpB   = make_scatter_img(infer(load_model(vpB_ckpt), 'v'), 'tomato')
vpC   = make_scatter_img(infer(load_model(vpC_ckpt), 'v'), 'tomato')
vpD   = crop_generated_from_png(FIG3/'swiss_roll_D32_vpred_vloss_expD.png')

gt_img = make_gt_img()


def pad_to_shape(img, h, w):
    """White-pad or centre-crop image to (h, w, 3)."""
    src_h, src_w = img.shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    oh = min(src_h, h); ow = min(src_w, w)
    y0 = (h - oh) // 2; x0 = (w - ow) // 2
    sy0 = (src_h - oh) // 2; sx0 = (src_w - ow) // 2
    out[y0:y0+oh, x0:x0+ow] = img[sy0:sy0+oh, sx0:sx0+ow]
    return out


# ── layout ────────────────────────────────────────────────────────────────────

col_labels = ['Ground\nTruth', 'Baseline\n(Part 2)', 'Exp A\nwide (1024)',
              'Exp B\nlogit-normal', 'Exp C\n4× longer', 'Exp D\nwide+logit']
row_labels  = ['pred = x  (blue)', 'pred = v  (red)']

x_row = [gt_img, xbase, xpA, xpB, xpC, xpD]
v_row = [gt_img, vbase, vpA, vpB, vpC, vpD]

# Normalise all cells to the same pixel size
CELL_H = 250; CELL_W = 250
all_cells = x_row + v_row
cells = [pad_to_shape(c, CELL_H, CELL_W) for c in all_cells]
x_cells = cells[:6]
v_cells = cells[6:]

LABEL_H = 40   # pixels for column-label strip
ROW_LABEL_W = 22  # narrow left strip for row label
PAD = 6        # padding between cells

n_cols = 6
n_rows = 2

fig_w = ROW_LABEL_W + n_cols * CELL_W + (n_cols - 1) * PAD
fig_h = LABEL_H + n_rows * CELL_H + (n_rows - 1) * PAD + 20   # +20 title

canvas = np.full((fig_h, fig_w, 3), 255, dtype=np.uint8)

# Place column labels
fig_tmp, ax_tmp = plt.subplots(1, 1, figsize=(fig_w / 110, LABEL_H / 110))
ax_tmp.axis('off')
plt.close(fig_tmp)

# Build final figure with matplotlib for clean label rendering
ncols = 6; nrows = 2
fig, axes = plt.subplots(
    nrows + 1, ncols + 1,
    figsize=(ncols * 2.5 + 0.9, nrows * 2.5 + 0.9),
    gridspec_kw={'width_ratios': [0.18] + [1]*ncols,
                 'height_ratios': [0.18] + [1]*nrows,
                 'hspace': 0.06, 'wspace': 0.06},
)

# Title cell (top-left corner)
axes[0, 0].axis('off')

# Column headers
for ci, lbl in enumerate(col_labels):
    axes[0, ci + 1].axis('off')
    axes[0, ci + 1].text(0.5, 0.5, lbl, ha='center', va='center',
                          fontsize=8.5, fontweight='bold',
                          transform=axes[0, ci + 1].transAxes,
                          multialignment='center')

# Row labels
for ri, lbl in enumerate(row_labels):
    axes[ri + 1, 0].axis('off')
    axes[ri + 1, 0].text(0.5, 0.5, lbl, ha='center', va='center',
                          fontsize=8, fontweight='bold', rotation=90,
                          transform=axes[ri + 1, 0].transAxes)

# Cell images
rows_data = [x_cells, v_cells]
for ri, row_cells in enumerate(rows_data):
    for ci, cell in enumerate(row_cells):
        ax = axes[ri + 1, ci + 1]
        ax.imshow(cell, aspect='equal', interpolation='lanczos')
        ax.axis('off')
        # Highlight GT column with a subtle box
        if ci == 0:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('#999999')
                spine.set_linewidth(0.8)

fig.suptitle('Part 3: Rescuing v-prediction at D=32 (swiss_roll)', fontsize=11, y=0.99)

out = FIG3 / 'comparison_panel.png'
plt.savefig(str(out), dpi=130, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')
