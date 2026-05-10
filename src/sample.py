import torch


@torch.no_grad()
def euler_sample(model, shape, n_steps, pred_type, eps=1e-3, device=None):
    """
    Euler ODE from t=1 to t=eps for both pred_type in {'x','v'}.
    shape: (B, D)
    """
    if device is None:
        device = next(model.parameters()).device
    z = torch.randn(shape, device=device)
    ts = torch.linspace(1.0, eps, n_steps + 1, device=device)

    for i in range(n_steps):
        t_now = ts[i].expand(shape[0])   # (B,)
        dt = ts[i + 1] - ts[i]           # negative
        pred = model(z, t_now)
        if pred_type == 'x':
            v = (z - pred) / t_now[:, None]
        else:  # 'v'
            v = pred
        z = z + v * dt

    return z
