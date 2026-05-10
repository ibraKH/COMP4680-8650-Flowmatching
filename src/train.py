import sys
sys.path.insert(0, 'src')

import torch
from pathlib import Path
from dataloader import get_dataloader
from model import MLPDenoiser
from fm import fm_loss_scheduled as fm_loss_fn
from utils import set_seed, get_device


def train(
    dataset,
    D,
    pred_type,
    loss_type,
    seed=42,
    n_steps=25_000,
    lr=1e-3,
    batch_size=1024,
    hidden=256,
    n_layers=6,
    t_schedule='uniform',
    exp_tag='',
    save_dir='checkpoints',
    verbose=True,
):
    set_seed(seed)
    device = get_device()
    dl = get_dataloader(dataset, dim=D, batch_size=batch_size)
    data_iter = _infinite(dl)

    model = MLPDenoiser(D, hidden=hidden, n_layers=n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(1, n_steps + 1):
        x = next(data_iter).to(device)
        loss = fm_loss_fn(model, x, pred_type, loss_type, t_schedule=t_schedule)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and step % 5000 == 0:
            print(f'  [{dataset} D={D} {pred_type}/{loss_type}] step {step}/{n_steps}  loss={loss.item():.4f}')

    tag_suffix = f'_{exp_tag}' if exp_tag else ''
    ckpt_path = Path(save_dir) / f'{dataset}_D{D}_{pred_type}pred_{loss_type}loss{tag_suffix}.pt'
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    if verbose:
        print(f'  Saved checkpoint → {ckpt_path}')

    return model


def _infinite(dl):
    while True:
        yield from dl


if __name__ == '__main__':
    # Quick smoke test
    model = train('swiss_roll', D=2, pred_type='v', loss_type='v', n_steps=200, verbose=True)
    print('train.py smoke test passed')
