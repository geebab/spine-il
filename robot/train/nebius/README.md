# GR00T N1.7 fine-tuning and serving on Nebius AI Cloud

Training and inference run on a Nebius GPU VM; the robot computer drives it over SSH with `vm.sh`. Creating, stopping and
paying for the VM is done in the Nebius console — none of these scripts does it. For a Serverless Endpoint image of the
policy server see `serverless/`.

What we used: RTX PRO 6000 (96 GB) — earlier a preemptible L40S (48 GB) — Ubuntu 24.04 with the NVIDIA driver
(CUDA 13.0), 200 GiB disk. A checkpoint is 24 GB (`pretrained_model` 12.6 GB + optimizer state 12 GB).

## You need

1. A GPU VM with a public IP and an SSH key for your user.
2. Hugging Face access to `nvidia/Cosmos-Reason2-2B` (accept the terms) and a read token.

## Steps

```bash
cd robot/train/nebius
bash vm.sh set <user>@<public IP>         # checks GPU, memory, disk, sudo
bash vm.sh push                           # kit + server files + dataset
bash vm.sh setup                          # uv, Python 3.12, lerobot[groot,training]==0.6.1
bash vm.sh hf && bash vm.sh hfcheck       # Hugging Face login on the VM (the token stays there)
bash vm.sh preflight                      # ../groot_preflight.py on the VM; must say "no problem found"
bash vm.sh smoke                          # 60-step trial: seconds per step, GPU memory
SAVE=2500 bash vm.sh train H90g 20000 32  # name, steps, batch (tmux; RESUME=1 continues from the last checkpoint)
bash vm.sh prune H90g                     # drops the optimizer state of older checkpoints
bash vm.sh status H90g                    # step, loss, speed, time left, disk
bash vm.sh serve H90g                     # policy_server.py on the VM's 127.0.0.1:8900
bash vm.sh tunnel H90g                    # SSH tunnel to the robot computer (127.0.0.1:18900)
bash vm.sh fetch H90g                     # copy the last pretrained_model back (parallel streams, sha256 checked)
bash vm.sh serve-stop H90g
```

`HELD="<session ids>"` leaves demonstrations out of training for evaluation; `HELD=""` trains on all of them.
`ckpt_sweep.sh <name> [steps…]` compares checkpoints without the robot (held-out demonstrations + observations replayed
from earlier robot runs, `../../inference/remote/replay_obs.py`).

The VM's public IP changes when it is started again: `bash vm.sh set <user>@<new IP>` and serve again. Shutting down
inside the VM does not stop billing; stop it in the console.

## Files

| file | runs on | does |
|---|---|---|
| `vm.sh` | robot computer | `set push setup hf hfcheck preflight smoke train status prune serve tunnel fetch restore stop serve-stop ssh` |
| `fetch_ranges.sh` | robot computer | downloads one large file over several SSH streams (used by `fetch`) |
| `ckpt_sweep.sh`, `ckpt_sweep.py` | robot computer | checkpoint comparison table |
| `vm_setup.sh` | VM | Python environment, GPU check |
| `vm_train.sh` | VM | one `lerobot-train` run with logs, GPU usage (`gpu.csv`) and settings (`run.json`) |
| `vm_prune.py` | VM | removes `training_state` of all but the newest checkpoint |
| `vm_serve.sh` | VM | `policy_server.py --backend lerobot` on 127.0.0.1 with a generated token |
| `serverless/` | — | policy server image for a Nebius Serverless Endpoint |
