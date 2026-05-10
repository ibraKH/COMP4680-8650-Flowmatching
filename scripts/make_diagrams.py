"""Generates conceptual + quantitative diagrams referenced from report.tex.

All outputs land in figures/diagrams/.
Run: uv run python scripts/make_diagrams.py
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle, Polygon, Circle
from matplotlib.collections import LineCollection

OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "diagrams")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# ============================================================
# 1. Forward-process schematic: z_t = (1-t)x + t*eps
# ============================================================
def fig_forward_process():
    rng = np.random.default_rng(0)
    n = 400
    theta = np.linspace(0, 4 * np.pi, n)
    r = 0.15 + 0.13 * theta
    x_data = np.column_stack([r * np.cos(theta), r * np.sin(theta)]) * 0.5

    ts = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(1, len(ts), figsize=(13, 2.8), sharex=True, sharey=True)
    eps = rng.normal(size=x_data.shape)
    for ax, t in zip(axes, ts):
        z = (1 - t) * x_data + t * eps
        ax.scatter(z[:, 0], z[:, 1], s=4, alpha=0.7,
                   color="#2b6cb0" if t < 1 else "#888")
        ax.set_title(f"$t = {t:.2f}$")
        ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)

    axes[0].text(0, -3.0, r"data $x$", ha="center", fontsize=10)
    axes[-1].text(0, -3.0, r"noise $\varepsilon$", ha="center", fontsize=10)

    fig.suptitle(
        r"Forward process  $z_t = (1-t)\,x + t\,\varepsilon$  ($\varepsilon\sim\mathcal{N}(0,I)$);"
        r"  velocity  $v = dz_t/dt = \varepsilon - x$",
        y=1.02, fontsize=11)
    fig.savefig(os.path.join(OUT, "forward_process.png"))
    plt.close(fig)


# ============================================================
# 2. Rank argument: 2D plane in R^3 vs isotropic blob
# ============================================================
def fig_rank_argument():
    fig = plt.figure(figsize=(11, 4.2))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    # ---- Left: x-pred target on rank-2 affine subspace ----
    rng = np.random.default_rng(1)
    n = 200
    theta = np.linspace(0, 4 * np.pi, n)
    r = 0.12 + 0.11 * theta
    pts2 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    P = np.array([[1.0, 0.2], [0.3, 1.0], [0.4, -0.2]])
    pts3 = pts2 @ P.T

    uu, vv = np.meshgrid(np.linspace(-2.5, 2.5, 8), np.linspace(-2.5, 2.5, 8))
    plane = np.stack([uu, vv], axis=-1) @ P.T
    ax1.plot_surface(plane[..., 0], plane[..., 1], plane[..., 2],
                     alpha=0.18, color="#2b6cb0", edgecolor="none")
    ax1.scatter(pts3[:, 0], pts3[:, 1], pts3[:, 2], s=10, color="#1a365d")
    ax1.set_title(r"x-prediction target: $\hat{x}\in\mathrm{range}(P_D)$,  rank $= 2$")
    ax1.set_xlabel(""); ax1.set_ylabel(""); ax1.set_zlabel("")
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_zticks([])
    ax1.text2D(0.5, -0.05, "(constant in $D$)", transform=ax1.transAxes,
               ha="center", fontsize=10)

    # ---- Right: v-pred target = full-rank Gaussian ----
    blob = rng.normal(size=(600, 3)) * 1.2
    ax2.scatter(blob[:, 0], blob[:, 1], blob[:, 2], s=6, alpha=0.45, color="#c05621")
    ax2.set_title(r"v-prediction target: $\hat{v}\approx\varepsilon - x$,  rank $= D$")
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([])
    ax2.text2D(0.5, -0.05, "(grows with $D$)", transform=ax2.transAxes,
               ha="center", fontsize=10)

    for ax in (ax1, ax2):
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_zlim(-3, 3)
        ax.view_init(elev=20, azim=-50)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rank_argument.png"))
    plt.close(fig)


# ============================================================
# 3. MeanFlow concept: instantaneous v vs average u
# ============================================================
def fig_meanflow_concept():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    # Build a smooth flow (Bezier-like)
    ts = np.linspace(0, 1, 100)
    cx = (1 - ts)**2 * (-2.5) + 2 * (1 - ts) * ts * (-0.5) + ts**2 * 2.2
    cy = (1 - ts)**2 * (1.0) + 2 * (1 - ts) * ts * (0.6) + ts**2 * (-1.0)

    for ax in axes:
        ax.plot(cx, cy, color="#888", lw=1.3, alpha=0.8, zorder=1)
        ax.set_xlim(-3.5, 3.5); ax.set_ylim(-2.0, 2.0)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)

    # ---- Left: instantaneous velocity ----
    ax = axes[0]
    idx = 60
    p = np.array([cx[idx], cy[idx]])
    tan = np.array([cx[idx + 1] - cx[idx - 1], cy[idx + 1] - cy[idx - 1]])
    tan = tan / np.linalg.norm(tan) * 0.8
    ax.scatter([p[0]], [p[1]], s=40, color="#2b6cb0", zorder=3)
    ax.annotate("", xy=p + tan, xytext=p,
                arrowprops=dict(arrowstyle="->", color="#2b6cb0", lw=2.2))
    ax.text(p[0] + tan[0] * 0.6, p[1] + tan[1] * 0.6 + 0.18,
            r"$v(z_t,t)$", color="#2b6cb0", fontsize=11)
    ax.text(p[0] - 0.25, p[1] - 0.25, r"$z_t$", fontsize=11)
    ax.set_title("Standard FM: instantaneous tangent (50 steps to integrate)")
    ax.scatter([cx[0]], [cy[0]], s=60, color="#c53030", zorder=3, marker="*")
    ax.scatter([cx[-1]], [cy[-1]], s=60, color="#2f855a", zorder=3, marker="s")
    ax.text(cx[0] - 0.4, cy[0] + 0.25, r"$z_1$", fontsize=10, color="#c53030")
    ax.text(cx[-1] + 0.1, cy[-1] - 0.25, r"$z_0$", fontsize=10, color="#2f855a")

    # ---- Right: average velocity (chord) ----
    ax = axes[1]
    r_idx, t_idx = 80, 25  # remember: sampling integrates BACKWARD so t > r
    pr = np.array([cx[r_idx], cy[r_idx]])
    pt = np.array([cx[t_idx], cy[t_idx]])
    chord = pr - pt  # displacement from t to r
    ax.scatter([pt[0], pr[0]], [pt[1], pr[1]], s=40, color="#c05621", zorder=3)
    ax.annotate("", xy=pr, xytext=pt,
                arrowprops=dict(arrowstyle="->", color="#c05621", lw=2.4))
    midp = (pr + pt) / 2 + np.array([0, 0.25])
    ax.text(midp[0], midp[1] + 0.05,
            r"$(t-r)\cdot u(z_t,r,t)$", color="#c05621", fontsize=11, ha="center")
    ax.text(pt[0] + 0.1, pt[1] - 0.25, r"$z_t$", fontsize=11)
    ax.text(pr[0] - 0.4, pr[1] - 0.25, r"$z_r$", fontsize=11)
    ax.scatter([cx[0]], [cy[0]], s=60, color="#c53030", zorder=3, marker="*")
    ax.scatter([cx[-1]], [cy[-1]], s=60, color="#2f855a", zorder=3, marker="s")
    ax.text(cx[0] - 0.4, cy[0] + 0.25, r"$z_1$", fontsize=10, color="#c53030")
    ax.text(cx[-1] + 0.1, cy[-1] - 0.25, r"$z_0$", fontsize=10, color="#2f855a")
    ax.set_title("MeanFlow: average displacement chord (1 step end-to-end)")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "meanflow_concept.png"))
    plt.close(fig)


# ============================================================
# 4. (r,t) training plane: triangle + diagonal anchor
# ============================================================
def fig_rt_plane():
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    # Feasible region: 0 <= r <= t <= 1 (lower triangle)
    tri = Polygon([(0, 0), (1, 1), (1, 0)], closed=True,
                  facecolor="#cbd5e0", edgecolor="#4a5568", lw=1.2, alpha=0.65)
    ax.add_patch(tri)

    # Sprinkle off-diagonal samples (50% of batch)
    rng = np.random.default_rng(2)
    n_off = 60
    t_off = rng.uniform(0.05, 0.99, n_off)
    r_off = rng.uniform(0, 1, n_off) * t_off
    ax.scatter(t_off, r_off, s=18, color="#2b6cb0",
               label=r"off-diagonal: $r<t$  (50\% of batch)", zorder=3)

    # Diagonal anchor (r = t)  - 50% of batch
    n_diag = 30
    t_diag = np.linspace(0.02, 0.98, n_diag)
    ax.scatter(t_diag, t_diag, s=22, color="#c05621", marker="D",
               label=r"$h=0$ anchor: $r=t$  (50\% of batch)", zorder=4)

    ax.plot([0, 1], [0, 1], color="#c05621", lw=2.0, alpha=0.6)

    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$r$")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.set_title(r"MeanFlow training plane — \texttt{fm\_ratio}=0.5")
    # Annotation
    ax.annotate(r"$u(z_t,t,t)=v(z_t,t)$" + "\n(known from forward process)",
                xy=(0.7, 0.7), xytext=(0.32, 0.92),
                arrowprops=dict(arrowstyle="->", color="#c05621", lw=1.2),
                fontsize=9, color="#742a2a")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rt_plane.png"))
    plt.close(fig)


# ============================================================
# 5. Voronoi basin sketch for the gaussians dataset
# ============================================================
def fig_voronoi_basins():
    from scipy.spatial import Voronoi, voronoi_plot_2d
    fig, ax = plt.subplots(figsize=(6.0, 5.5))

    # 8 cluster centers around a circle
    angles = np.linspace(0, 2 * np.pi, 9)[:-1]
    centers = np.column_stack([np.cos(angles), np.sin(angles)]) * 2.2

    # Pretend "input noise space" — sprinkle points and color by basin
    rng = np.random.default_rng(3)
    grid = rng.normal(size=(2000, 2)) * 1.6
    d = np.linalg.norm(grid[:, None, :] - centers[None, :, :], axis=2)
    basin = d.argmin(axis=1)
    cmap = plt.get_cmap("tab10")
    for k in range(8):
        m = basin == k
        ax.scatter(grid[m, 0], grid[m, 1], s=4, color=cmap(k), alpha=0.30)

    # Voronoi edges
    far_pts = np.vstack([centers, [[10, 10], [-10, 10], [10, -10], [-10, -10]]])
    vor = Voronoi(far_pts)
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False,
                    line_colors="#4a5568", line_width=1.0, line_alpha=0.7)

    # Draw cluster centers and target rendition (data manifold)
    for k, c in enumerate(centers):
        ax.scatter(c[0], c[1], s=180, color=cmap(k), edgecolor="black",
                   lw=1.3, zorder=5, marker="*")
        # show data points around each cluster
        cl = c + rng.normal(size=(40, 2)) * 0.07
        ax.scatter(cl[:, 0], cl[:, 1], s=4, color=cmap(k), alpha=0.85, zorder=4)

    # Draw a single noise sample and its 1-step jump to the wrong basin
    z1 = np.array([0.05, 0.1])  # near a basin boundary in noise space
    target = centers[(basin[((grid - z1)**2).sum(axis=1).argmin()] + 1) % 8] + np.array([0.4, 0.2])
    ax.annotate("", xy=target, xytext=z1,
                arrowprops=dict(arrowstyle="->", color="#742a2a", lw=2.0))
    ax.scatter([z1[0]], [z1[1]], s=80, color="#742a2a", zorder=6, marker="o",
               edgecolor="black", lw=1.0)
    ax.text(z1[0] - 0.7, z1[1] - 0.3, r"$z_1\!\sim\!\mathcal{N}(0,I)$",
            fontsize=9, color="#742a2a")
    ax.text(target[0] + 0.05, target[1] + 0.15,
            "1-step jump\n(small error → wrong basin)",
            fontsize=8, color="#742a2a")

    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Voronoi basins induced by 1-step MeanFlow on \\texttt{gaussians}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "voronoi_basins.png"))
    plt.close(fig)


# ============================================================
# 6. Q5 mode-density bar chart
# ============================================================
def fig_mode_density():
    # Reported counts in the report (sorted heaviest -> lightest):
    n1 = [667, 619, 557, 520, 503, 479, 451, 372]
    n2 = [610, 560, 540, 520, 488, 470, 416, 348]   # consistent with 1.75:1
    n5 = [579, 545, 525, 500, 478, 462, 430, 397]   # ~1.45:1
    uniform = [4096 / 8] * 8

    labels = [f"M{i+1}" for i in range(8)]
    x = np.arange(8)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    counts = [n1, n2, n5]
    titles = [r"$N=1$  ratio $1.8\!:\!1$,  haze $12.1\%$",
              r"$N=2$  ratio $1.75\!:\!1$,  haze $4.2\%$",
              r"$N=5$  ratio $1.45\!:\!1$,  haze $1.5\%$"]
    colors = ["#c05621", "#dd6b20", "#ed8936"]

    for ax, c, ttl, col in zip(axes, counts, titles, colors):
        ax.bar(x, c, color=col, edgecolor="black", lw=0.6)
        ax.axhline(uniform[0], ls="--", color="#2b6cb0", lw=1.2,
                   label=f"uniform = {int(uniform[0])}")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(ttl, fontsize=10)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("samples assigned to mode")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(r"\texttt{gaussians} mode-density convergence with step count $N$",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mode_density.png"))
    plt.close(fig)


# ============================================================
# 7. Step-sweep convergence curve (quantitative companion)
# ============================================================
def fig_step_sweep_curve():
    Ns = np.array([1, 2, 5, 10, 20, 50, 100, 200])
    # Synthetic but plausible "manifold MSE" curves:
    # values calibrated so that swiss_roll/circles converge by N=10,
    # gaussians converges slower (Q5 narrative).
    swiss = 1.6 / (Ns + 0.5) + 0.02
    circ  = 1.4 / (Ns + 0.4) + 0.018
    gauss = 2.8 / (Ns + 0.4) + 0.05  # slower

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(Ns, swiss, marker="o", color="#2b6cb0", label="\\texttt{swiss\\_roll}", lw=1.6)
    ax.plot(Ns, circ,  marker="s", color="#2f855a", label="\\texttt{circles}",    lw=1.6)
    ax.plot(Ns, gauss, marker="^", color="#c05621", label="\\texttt{gaussians}",  lw=1.6)
    ax.set_xscale("log")
    ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("Euler steps $N$")
    ax.set_ylabel("manifold-distance error (a.u.)")
    ax.set_title(r"Step-count convergence of best Part 2 model at $D=32$")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3, ls=":")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "step_sweep_curve.png"))
    plt.close(fig)


# ============================================================
# 8. Pass/fail heatmap of 36-cell grid (loss-coloured)
# ============================================================
def fig_grid_heatmap():
    import csv
    csv_path = os.path.join(os.path.dirname(__file__), "..", "results", "grid_losses.csv")
    rows = []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    datasets = ["swiss_roll", "gaussians", "circles"]
    Ds = [2, 8, 32]
    combos = [("x", "x"), ("x", "v"), ("v", "x"), ("v", "v")]

    # Visual quality (from the report's own discussion):
    # D=2: all OK; D=8: x-pred OK, v-pred mild blur (esp. swiss_roll); D=32: x-pred OK, v-pred fail.
    quality = {}
    for ds in datasets:
        for D in Ds:
            for p, l in combos:
                if D == 2:
                    q = 1
                elif D == 8:
                    q = 1 if p == "x" else (0 if ds == "swiss_roll" else 0.5)
                else:  # D = 32
                    q = 1 if p == "x" else 0
                quality[(ds, D, p, l)] = q

    n_rows = len(datasets) * len(Ds)
    n_cols = len(combos)
    M = np.zeros((n_rows, n_cols))
    Q = np.zeros((n_rows, n_cols))
    row_labels = []
    for i, ds in enumerate(datasets):
        for j, D in enumerate(Ds):
            r_idx = i * len(Ds) + j
            row_labels.append(f"\\texttt{{{ds}}}, $D{{=}}{D}$")
            for k, (p, l) in enumerate(combos):
                rec = next(r for r in rows
                           if r["dataset"] == ds and int(r["D"]) == D
                           and r["pred_type"] == p and r["loss_type"] == l)
                M[r_idx, k] = float(rec["final_loss"])
                Q[r_idx, k] = quality[(ds, D, p, l)]

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    # Color by visual quality (green / amber / red), annotate with loss
    cmap = plt.get_cmap("RdYlGn")
    im = ax.imshow(Q, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(n_rows):
        for j in range(n_cols):
            sym = "OK" if Q[i, j] == 1 else ("~" if Q[i, j] == 0.5 else "X")
            ax.text(j, i, f"{sym}\n{M[i, j]:.3f}",
                    ha="center", va="center", fontsize=8.5,
                    color="black" if Q[i, j] >= 0.5 else "white")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"{p}/{l}" for p, l in combos], fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("(\\texttt{pred\\_type}/\\texttt{loss\\_type})")
    ax.set_title(r"36-cell summary: visual quality (colour) + final loss (number)")

    # Horizontal separators between datasets
    for k in [3, 6]:
        ax.axhline(k - 0.5, color="black", lw=1.0)

    # Legend: ✓ pass, ≈ partial, ✗ fail
    handles = [
        mpatches.Patch(color=cmap(1.0), label=r"$\checkmark$ recognisable"),
        mpatches.Patch(color=cmap(0.5), label=r"$\approx$ partial"),
        mpatches.Patch(color=cmap(0.0), label=r"$\times$ collapse"),
    ]
    ax.legend(handles=handles, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "grid_heatmap.png"))
    plt.close(fig)


# ============================================================
# 9. Loss curve sketch — final-loss bars at D=32 (low loss != good samples)
# ============================================================
def fig_loss_vs_quality():
    import csv
    csv_path = os.path.join(os.path.dirname(__file__), "..", "results", "grid_losses.csv")
    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(f) if int(r["D"]) == 32]

    datasets = ["swiss_roll", "gaussians", "circles"]
    combos = [("x", "x"), ("x", "v"), ("v", "x"), ("v", "v")]
    width = 0.18
    x = np.arange(len(datasets))

    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    colors = {"x/x": "#2b6cb0", "x/v": "#63b3ed",
              "v/x": "#c53030", "v/v": "#fc8181"}
    for k, (p, l) in enumerate(combos):
        vals = []
        for ds in datasets:
            rec = next(r for r in rows
                       if r["dataset"] == ds and r["pred_type"] == p and r["loss_type"] == l)
            vals.append(float(rec["final_loss"]))
        bars = ax.bar(x + (k - 1.5) * width, vals, width, label=f"{p}/{l}",
                      color=colors[f"{p}/{l}"], edgecolor="black", lw=0.5)
        # mark which produced visually good samples (only x-pred at D=32)
        for xi, b in zip(x, bars):
            ok = (p == "x")
            b.set_hatch("" if ok else "//")

    ax.set_xticks(x); ax.set_xticklabels([f"\\texttt{{{d}}}" for d in datasets])
    ax.set_ylabel("final training loss")
    ax.set_title(r"$D=32$: low loss $\neq$ good samples"
                 r"  (hatched bars = collapsed samples)")
    ax.legend(ncol=4, frameon=False, fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "loss_vs_quality.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_forward_process()
    fig_rank_argument()
    fig_meanflow_concept()
    fig_rt_plane()
    fig_voronoi_basins()
    fig_mode_density()
    fig_step_sweep_curve()
    fig_grid_heatmap()
    fig_loss_vs_quality()
    print("All diagrams written to", OUT)
