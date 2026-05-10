import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scatter_2d(generated: np.ndarray, gt: np.ndarray, title: str, save_path: str):
    """Side-by-side scatter: GT (grey) vs generated (blue)."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].scatter(gt[:, 0], gt[:, 1], s=4, alpha=0.3, color="grey")
    axes[0].set_title("Ground Truth")
    axes[0].set_aspect("equal")
    axes[1].scatter(generated[:, 0], generated[:, 1], s=4, alpha=0.5, color="steelblue")
    axes[1].set_title("Generated")
    axes[1].set_aspect("equal")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def make_figure(
    generated: np.ndarray,
    gt: np.ndarray,
    dataset: str,
    D: int,
    pred_type: str,
    loss_type: str,
    n_steps: int,
    save_dir: str,
) -> str:
    """
    Emit a consistently-styled GT vs generated scatter for one experiment config.
    Returns the path to the saved PNG.
    """
    title = f"{dataset} D={D}  pred={pred_type}  loss={loss_type}  steps={n_steps}"
    filename = f"{dataset}_D{D}_{pred_type}pred_{loss_type}loss.png"
    save_path = str(Path(save_dir) / filename)

    margin = 0.5
    x_min, x_max = gt[:, 0].min() - margin, gt[:, 0].max() + margin
    y_min, y_max = gt[:, 1].min() - margin, gt[:, 1].max() + margin

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].scatter(gt[:, 0], gt[:, 1], s=4, alpha=0.3, color="grey")
    axes[0].set_title("Ground Truth")
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)
    axes[0].set_aspect("equal")

    axes[1].scatter(generated[:, 0], generated[:, 1], s=4, alpha=0.5, color="steelblue")
    axes[1].set_title("Generated")
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_min, y_max)
    axes[1].set_aspect("equal")

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path
