import torch
import torch.nn.functional as F


def forward_process(x, t, eps):
    """z_t = (1-t)*x + t*eps.  x,eps: (B,D)  t: (B,)"""
    t_ = t[:, None]
    return (1 - t_) * x + t_ * eps


def x_to_v(x_hat, z, t, clip_eps=1e-3):
    """v_hat = (z - x_hat) / t.  Clips t away from 0."""
    t_ = t.clamp(clip_eps, 1 - clip_eps)[:, None]
    return (z - x_hat) / t_


def v_to_x(v_hat, z, t):
    """x_hat = z - t*v_hat."""
    return z - t[:, None] * v_hat


def x_to_eps(x_hat, z, t, clip_eps=1e-3):
    """eps_hat = (z - (1-t)*x_hat) / t."""
    t_ = t.clamp(clip_eps, 1 - clip_eps)[:, None]
    return (z - (1 - t[:, None]) * x_hat) / t_


def v_to_eps(v_hat, x_hat):
    """eps_hat = v_hat + x_hat  (since v = eps - x)."""
    return v_hat + x_hat


def fm_loss(model, x, pred_type, loss_type, eps=1e-3):
    """
    Unified flow matching loss for all 4 (pred_type, loss_type) combos.
    pred_type in {'x', 'v'}   loss_type in {'x', 'v'}
    """
    B = x.shape[0]
    t = torch.rand(B, device=x.device).clamp(eps, 1 - eps)
    noise = torch.randn_like(x)
    z = forward_process(x, t, noise)
    v_true = noise - x
    pred = model(z, t)

    if pred_type == 'x':
        x_hat = pred
        v_hat = x_to_v(pred, z, t, eps)
    else:  # 'v'
        x_hat = v_to_x(pred, z, t)
        v_hat = pred

    if loss_type == 'x':
        return F.mse_loss(x_hat, x)
    else:  # 'v'
        return F.mse_loss(v_hat, v_true)


def fm_loss_scheduled(model, x, pred_type, loss_type, t_schedule='uniform', eps=1e-3):
    """
    fm_loss with pluggable t-sampling schedule.
    t_schedule: 'uniform' (default, identical to fm_loss) or 'logit_normal'
    """
    B = x.shape[0]
    if t_schedule == 'logit_normal':
        # SD3 / JiT trick: concentrate training on intermediate noise levels
        t = torch.sigmoid(torch.randn(B, device=x.device) * 1.0 + 0.0).clamp(eps, 1 - eps)
    else:
        t = torch.rand(B, device=x.device).clamp(eps, 1 - eps)

    noise = torch.randn_like(x)
    z = forward_process(x, t, noise)
    v_true = noise - x
    pred = model(z, t)

    if pred_type == 'x':
        x_hat = pred
        v_hat = x_to_v(pred, z, t, eps)
    else:  # 'v'
        x_hat = v_to_x(pred, z, t)
        v_hat = pred

    if loss_type == 'x':
        return F.mse_loss(x_hat, x)
    else:  # 'v'
        return F.mse_loss(v_hat, v_true)
