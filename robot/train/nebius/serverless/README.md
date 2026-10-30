# Policy server on a Nebius Serverless Endpoint

`Dockerfile.policy` packs the reference policy server (`robot/inference/remote/policy_server.py`, protocol
`spine-policy/1`: `GET /health`, `GET /info`, `POST /act`) with LeRobot 0.6.1's GR00T policy. No weights, data or viewer
are inside the image; the fine-tuned checkpoint is mounted at `/ckpt` at run time.

```bash
B=/tmp/spine_policy_server && mkdir -p $B
cp robot/inference/remote/policy_server.py robot/inference/remote_policy.py robot/dataset/actions.py \
   robot/dataset/phases.py robot/dataset/instructions.py robot/train/nebius/serverless/policy_entry.sh $B/
docker build -f robot/train/nebius/serverless/Dockerfile.policy -t spine-policy:h90g $B

# local test on a GPU machine
docker run --rm --gpus all -p 8900:8900 -e POLICY_TOKEN=$(openssl rand -hex 32) -v /path/to/ckpt:/ckpt:ro spine-policy:h90g
```

`/ckpt/pretrained_model` = the checkpoint's `pretrained_model` folder (Hugging Face release), `/ckpt/export.json` = the
dataset's `meta/spine_il/export.json` (instruction set, optional). The base model's backbone (`nvidia/Cosmos-Reason2-2B`)
is gated on Hugging Face: set `HF_TOKEN`, or mount a hub cache at `/ckpt/hf-hub` (the server then runs offline).

On Nebius: push the image to Container Registry, then `nebius ai endpoint create` with the image, a GPU platform
(we use RTX PRO 6000, `gpu-rtx6000-a` / `1gpu-24vcpu-218gb`), `--container-port 8900`, the checkpoint bucket as
`--volume <bucket>:/ckpt:ro`, `--auth token`, and the same token as `POLICY_TOKEN`. Clients then use
`--policy-url https://<endpoint URL>` with that token (`robot/inference/remote_policy.py`).
