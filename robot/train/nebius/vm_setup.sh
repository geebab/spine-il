#!/usr/bin/env bash
# ON THE GPU MACHINE (a Nebius VM: Ubuntu image with the NVIDIA driver, a user with sudo): the environment GR00T N1.7 is
# fine-tuned and served in. Safe to run again. Called by vm.sh setup from the robot server.
#   bash ~/il-robot/kit/vm_setup.sh
# LeRobot 0.6.1 with its groot extra (transformers, peft, diffusers, timm, decord, dm-tree — no flash-attn: LeRobot falls
# back to SDPA attention, numerically the same) and the training extra (accelerate, wandb — wandb stays off).
set -euo pipefail
WS=${IL_ROBOT_WS:-$HOME/il-robot}; V=${LEROBOT_VERSION:-0.6.1}
echo "== GPU"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || { echo "nvidia-smi failed — this image has no NVIDIA driver"; exit 1; }
need() { command -v "$1" > /dev/null 2>&1; }
if ! need ffmpeg || ! need tmux || ! need rsync || ! need git || ! need curl; then
  echo "== system packages"; sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg tmux rsync git curl
fi
if ! need uv && [ ! -x "$HOME/.local/bin/uv" ]; then echo "== uv"; curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh; fi
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$WS"/{datasets,out,kit,server} && cd "$WS"
echo "== python environment ($WS/.venv)"
uv python install 3.12 > /dev/null
[ -x .venv/bin/python ] || uv venv --managed-python --python 3.12 --seed .venv
uv pip install -q --python .venv/bin/python "lerobot[groot,training]==$V"
.venv/bin/python - <<'PY'
import importlib.metadata as md, shutil, torch
print("lerobot", md.version("lerobot"), "| torch", torch.__version__, "| transformers", md.version("transformers"))
print("cuda", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU SEEN BY TORCH",
      "|", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if torch.cuda.is_available() else 0, "GiB")
for m in ("torchcodec", "av", "decord"):
    try:
        print(m, md.version(m))
    except md.PackageNotFoundError:
        print(m, "not installed")
print("ffmpeg", bool(shutil.which("ffmpeg")))
PY
echo "== disk"; df -h "$WS" | tail -n 1
echo "setup done"
