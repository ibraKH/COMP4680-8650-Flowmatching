import math
import torch
import torch.nn as nn


class SinusoidalEmbed(nn.Module):
    def __init__(self, dim=128, max_period=10000):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        half = dim // 2
        # DiT canonical form: (k-1) denominator
        freqs = torch.exp(-math.log(max_period) * torch.arange(half) / (half - 1))
        self.register_buffer("freqs", freqs)

    def forward(self, t):  # t: (B,)
        args = t[:, None] * self.freqs[None]          # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


class MLPDenoiser(nn.Module):
    def __init__(self, D, hidden=256, n_layers=6, time_dim=128):
        super().__init__()
        self.t_embed = SinusoidalEmbed(time_dim)
        layers = [nn.Linear(D + time_dim, hidden), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, D)]
        self.net = nn.Sequential(*layers)
        # Variance decays through ReLU layers; rescale last layer so untrained
        # model outputs have non-trivial std (needed by x-pred ODE sanity checks).
        nn.init.normal_(self.net[-1].weight, std=0.5)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z, t):   # z: (B,D), t: (B,)
        et = self.t_embed(t)
        return self.net(torch.cat([z, et], dim=-1))
