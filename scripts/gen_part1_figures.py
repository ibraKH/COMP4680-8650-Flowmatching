"""
Fix Issues 3 & 4: regenerate Part 1 ground-truth figures.

Issue 4: existing swiss_roll_D2_GT.png etc. contain a Generated panel.
         Replace them with GT-only single-panel figures.
Issue 3: add six figures total (D=2 GT + D=32 back-projected GT) per dataset.
         New files: figures/part1/{dataset}_D32_backproj.png
"""
import sys
sys.path.insert(0, 'src')

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from dataloader import ToyDiffusionDataset

DATASETS = ['swiss_roll', 'gaussians', 'circles']
OUT = Path('figures/part1')
OUT.mkdir(parents=True, exist_ok=True)


def scatter_gt_only(data2d: np.ndarray, title: str, save_path: str):
    """Single-panel ground-truth scatter, no Generated panel."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(data2d[:, 0], data2d[:, 1], s=4, alpha=0.35, color='grey')
    margin = 0.3
    ax.set_xlim(data2d[:, 0].min() - margin, data2d[:, 0].max() + margin)
    ax.set_ylim(data2d[:, 1].min() - margin, data2d[:, 1].max() + margin)
    ax.set_aspect('equal')
    ax.set_title('Ground Truth', fontsize=10)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {save_path}')


def scatter_d2_vs_d32(data2d: np.ndarray, backproj: np.ndarray,
                      dataset: str, save_path: str):
    """Side-by-side: original D=2 (grey) | back-projected D=32→2D (blue)."""
    margin = 0.3
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].scatter(data2d[:, 0], data2d[:, 1], s=4, alpha=0.35, color='grey')
    axes[0].set_title('Original D=2', fontsize=10)
    axes[0].set_xlim(data2d[:, 0].min() - margin, data2d[:, 0].max() + margin)
    axes[0].set_ylim(data2d[:, 1].min() - margin, data2d[:, 1].max() + margin)
    axes[0].set_aspect('equal')

    axes[1].scatter(backproj[:, 0], backproj[:, 1], s=4, alpha=0.35, color='steelblue')
    axes[1].set_title('Back-projected (D=32 → 2D)', fontsize=10)
    axes[1].set_xlim(data2d[:, 0].min() - margin, data2d[:, 0].max() + margin)
    axes[1].set_ylim(data2d[:, 1].min() - margin, data2d[:, 1].max() + margin)
    axes[1].set_aspect('equal')

    fig.suptitle(f'{dataset}: original 2D vs back-projected from D=32', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {save_path}')


print('=== Generating Part 1 figures (Issues 3 & 4) ===')
for ds in DATASETS:
    # Load D=2 ground truth
    ds2 = ToyDiffusionDataset(ds, dim=2)
    data2d = ds2.data.numpy()[:4096]

    # Issue 4: replace GT PNGs with ground-truth-only single panel
    scatter_gt_only(
        data2d,
        title=f'{ds} D=2 — Ground Truth',
        save_path=str(OUT / f'{ds}_D2_GT.png'),
    )

    # Issue 3: generate D=32 back-projected GT figures
    ds32 = ToyDiffusionDataset(ds, dim=32)
    data32 = ds32.data.numpy()[:4096]
    backproj = ds32.to_2d(data32)   # (N, 2) via samples @ P.T

    scatter_d2_vs_d32(
        data2d, backproj, ds,
        save_path=str(OUT / f'{ds}_D32_backproj.png'),
    )

print('Done.')
