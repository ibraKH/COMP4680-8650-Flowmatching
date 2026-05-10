import sys, csv, os, time
from pathlib import Path
sys.path.insert(0, 'src')

from train import train

DATASETS   = ['swiss_roll', 'gaussians', 'circles']
DIMS       = [2, 8, 32]
PRED_TYPES = ['x', 'v']
LOSS_TYPES = ['x', 'v']

CONFIGS = [
    (ds, D, pt, lt)
    for ds in DATASETS
    for D  in DIMS
    for pt in PRED_TYPES
    for lt in LOSS_TYPES
]   # 36 total

LOG_PATH = Path('results/grid_losses.csv')
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Load existing log so we know what's done
done = {}
if LOG_PATH.exists():
    with open(LOG_PATH) as f:
        for row in csv.DictReader(f):
            key = (row['dataset'], int(row['D']), row['pred_type'], row['loss_type'])
            done[key] = float(row['final_loss'])

# Open log in append mode
log_file = open(LOG_PATH, 'a', newline='')
writer = csv.DictWriter(log_file, fieldnames=['dataset','D','pred_type','loss_type','final_loss','elapsed_s'])
if LOG_PATH.stat().st_size == 0:
    writer.writeheader()

total = len(CONFIGS)
for idx, (ds, D, pt, lt) in enumerate(CONFIGS, 1):
    key = (ds, D, pt, lt)
    ckpt = Path('checkpoints') / f'{ds}_D{D}_{pt}pred_{lt}loss.pt'
    if ckpt.exists() and key in done:
        print(f'[{idx:2d}/{total}] SKIP  {ds} D={D} {pt}/{lt}  (checkpoint exists)')
        continue

    print(f'[{idx:2d}/{total}] RUN   {ds} D={D} {pt}/{lt} ...')
    t0 = time.time()
    model = train(ds, D, pt, lt, seed=42, n_steps=25_000, verbose=False)

    # Compute final loss estimate on one batch
    import torch
    from dataloader import get_dataloader
    from fm import fm_loss
    dl = get_dataloader(ds, dim=D, batch_size=1024)
    x = next(iter(dl))
    device = next(model.parameters()).device
    with torch.no_grad():
        loss_val = fm_loss(model, x.to(device), pt, lt).item()

    elapsed = time.time() - t0
    print(f'         final_loss={loss_val:.4f}  elapsed={elapsed:.0f}s')
    writer.writerow({'dataset': ds, 'D': D, 'pred_type': pt, 'loss_type': lt,
                     'final_loss': loss_val, 'elapsed_s': f'{elapsed:.0f}'})
    log_file.flush()

log_file.close()
print(f'\nGrid complete. Log: {LOG_PATH}')
print(f'Checkpoints: {len(list(Path("checkpoints").glob("*.pt")))} files')
