import json
from pathlib import Path

h = Path('runs/faster_rcnn/loss_history.json')
if h.exists():
    data = json.load(open(h))
    print(f'Epochs completed: {len(data)}')
    print('Last 5 epochs:')
    for row in data[-5:]:
        print(f'  Epoch {row["epoch"]:3d} | Loss: {row["loss"]:.4f}')
    print(f'Best loss so far: {min(d["loss"] for d in data):.4f}')
else:
    print('No history yet')

# Check checkpoints
ckpts = sorted(Path('runs/faster_rcnn').glob('epoch_*.pt'))
print(f'\nCheckpoints saved: {len(ckpts)}')
for c in ckpts:
    print(f'  {c.name}')
