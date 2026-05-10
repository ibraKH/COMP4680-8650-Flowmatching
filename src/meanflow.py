import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SinusoidalEmbed


class MFModel(nn.Module):
    """
    MeanFlow model: takes (z, r, t) and predicts average velocity u over [r,t].
    Two separate sinusoidal embeddings: one for t, one for h = t - r.
    """
    def __init__(self, D, hidden=256, n_layers=6, time_dim=128):
        super().__init__()
        self.t_embed = SinusoidalEmbed(time_dim)   # separate weights from h_embed
        self.h_embed = SinusoidalEmbed(time_dim)   # separate params — NOT shared
        layers = [nn.Linear(D + 2 * time_dim, hidden), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, D)]
        self.net = nn.Sequential(*layers)

    def forward(self, z, r, t):   # z:(B,D)  r:(B,)  t:(B,)
        h = t - r                  # h=0 when r=t (FM diagonal)
        et = self.t_embed(t)       # (B, time_dim)
        eh = self.h_embed(h)       # (B, time_dim)
        return self.net(torch.cat([z, et, eh], dim=-1))


def mf_loss(model, x, fm_ratio=0.5, eps=1e-3):
    """
    MeanFlow consistency loss with JVP.
    fm_ratio: fraction of samples where r=t (h=0), anchoring u to v.
    CRITICAL: .detach() on target avoids higher-order gradients.
    """
    B = x.shape[0]
    device = x.device

    t = torch.rand(B, device=device).clamp(eps, 1 - eps)
    r = torch.rand(B, device=device) * t          # r uniformly in [0, t]
    fm_mask = torch.rand(B, device=device) < fm_ratio
    r = torch.where(fm_mask, t, r)                # h=0 portion: r = t

    noise = torch.randn_like(x)
    z = (1 - t[:, None]) * x + t[:, None] * noise
    v = noise - x                                  # true instantaneous velocity

    # JVP tangents: dz/dt=v (flow), dr/dt=0 (r fixed), dt/dt=1
    tangents = (v, torch.zeros_like(r), torch.ones_like(t))

    def u_fn(z_, r_, t_):
        return model(z_, r_, t_)

    u_pred, du_dt = torch.func.jvp(u_fn, (z, r, t), tangents)

    # MeanFlow identity: u = v - (t-r) * du/dt
    # Stop-grad on target: avoids higher-order grads, no quality cost (Geng et al.)
    target = (v - (t - r)[:, None] * du_dt).detach()

    return F.mse_loss(u_pred, target)


@torch.no_grad()
def mf_sample(model, shape, n_steps, eps=1e-3, device=None):
    """
    MeanFlow N-step sampler. For N=1: single jump from t=1 to t=eps.
    Jump rule: z_new = z - (t - r) * u(z, r, t)
    Sign: u points from z_t toward z_r (toward cleaner data), so subtract.
    """
    if device is None:
        device = next(model.parameters()).device
    B = shape[0]
    z = torch.randn(shape, device=device)
    ts = torch.linspace(1.0, eps, n_steps + 1, device=device)

    for i in range(n_steps):
        t_now = ts[i].expand(B)      # (B,)
        r_now = ts[i + 1].expand(B)  # (B,)
        u = model(z, r_now, t_now)   # average velocity over [r_now, t_now]
        z = z - (t_now - r_now)[:, None] * u   # displacement toward r

    return z


def train_mf(dataset, D, seed=42, n_steps=25_000, lr=1e-3, batch_size=1024,
             hidden=256, n_layers=6, fm_ratio=0.5, save_dir='checkpoints',
             verbose=True):
    """Dedicated MeanFlow training loop."""
    import sys; sys.path.insert(0, 'src')
    from dataloader import get_dataloader
    from utils import set_seed, get_device

    set_seed(seed)
    device = get_device()
    dl = get_dataloader(dataset, dim=D, batch_size=batch_size)

    model = MFModel(D, hidden=hidden, n_layers=n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def cycle(loader):
        while True:
            yield from loader

    gen = cycle(dl)
    from pathlib import Path
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for step in range(1, n_steps + 1):
        x = next(gen).to(device)
        loss = mf_loss(model, x, fm_ratio=fm_ratio)
        opt.zero_grad()
        loss.backward()
        # Gradient clip for MF stability (moving-target issue)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        if verbose and step % 5000 == 0:
            print(f'  [MF {dataset} D={D}] step {step}/{n_steps}  loss={loss.item():.4f}')

    ckpt_path = Path(save_dir) / f'mf_{dataset}_D{D}.pt'
    torch.save(model.state_dict(), ckpt_path)
    if verbose:
        print(f'  Saved → {ckpt_path}')
    return model
