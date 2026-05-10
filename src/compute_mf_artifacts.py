"""Compute the MeanFlow gaussians artifact statistics reported in the
report (per-mode counts, heaviest:lightest ratio, near-mode coverage).

Loads the trained MF checkpoint at D=32, generates 4096 samples for
N in {1, 2, 5}, projects back to 2D using the dataset's projection
matrix, assigns each sample to its nearest ground-truth mode centre,
and prints the statistics.

Usage:
    uv run python src/compute_mf_artifacts.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dataloader import ToyDiffusionDataset  # noqa: E402
from meanflow import MFModel, mf_sample  # noqa: E402
from utils import set_seed, get_device  # noqa: E402


def _kmeans_once(data: np.ndarray, k: int, seed: int, n_iter: int = 100) -> tuple:
    rng = np.random.default_rng(seed)
    # k-means++ initialisation for stable convergence
    idx = [rng.integers(len(data))]
    for _ in range(k - 1):
        d2 = np.min(((data[:, None] - data[idx][None]) ** 2).sum(-1), axis=1)
        prob = d2 / d2.sum()
        idx.append(int(rng.choice(len(data), p=prob)))
    centres = data[idx].copy()
    for _ in range(n_iter):
        d = ((data[:, None] - centres[None]) ** 2).sum(-1)
        a = d.argmin(1)
        new = centres.copy()
        for j in range(k):
            mask = a == j
            if mask.any():
                new[j] = data[mask].mean(0)
        if np.allclose(new, centres):
            break
        centres = new
    wcss = float(((data - centres[a]) ** 2).sum())
    return centres, wcss


def estimate_mode_centres(data_2d: np.ndarray, k: int = 8, n_restarts: int = 20) -> np.ndarray:
    """k-means++ with multiple restarts; returns the centres of the best run."""
    best_c, best_w = None, float("inf")
    for s in range(n_restarts):
        c, w = _kmeans_once(data_2d, k, seed=s)
        if w < best_w:
            best_c, best_w = c, w
    return best_c


def main() -> None:
    set_seed(42)
    device = get_device()

    ds = ToyDiffusionDataset(name="gaussians", dim=32)
    data_2d = np.load(Path(__file__).resolve().parent.parent
                      / "data" / "gaussians.npz")["2d"]
    centres = estimate_mode_centres(data_2d, k=8)

    model = MFModel(D=32, hidden=256, n_layers=6).to(device)
    ckpt = Path(__file__).resolve().parent.parent / "checkpoints" / "mf_gaussians_D32.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    print(f"{'N':>3}  {'per-mode counts (sorted desc)':<45}  "
          f"{'ratio':>6}  {'coverage':>9}")
    for n_steps in (1, 2, 5):
        torch.manual_seed(42)
        z = mf_sample(model, (4096, 32), n_steps=n_steps, device=device)
        samples_2d = ds.to_2d(z.cpu().numpy())
        d = ((samples_2d[:, None] - centres[None]) ** 2).sum(-1)
        nearest = d.argmin(1)
        nearest_dist = np.sqrt(d.min(1))
        counts = np.bincount(nearest, minlength=8).tolist()
        counts.sort(reverse=True)
        ratio = counts[0] / max(counts[-1], 1)
        coverage = float((nearest_dist < 0.5).mean())
        print(f"{n_steps:>3}  {str(counts):<45}  {ratio:>6.2f}  {coverage:>9.1%}")


if __name__ == "__main__":
    main()
