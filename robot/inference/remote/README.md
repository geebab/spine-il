# Remote policy server — protocol `spine-policy/1`

The policy runs on a GPU machine (a Nebius VM or Serverless Endpoint); the robot loop asks it over HTTP(S) with a bearer
token. JSON in, JSON out.

| request | answer |
|---|---|
| `GET /health` | `{"ok": true}` (no token) |
| `GET /info` | model description: `protocol`, `model`, `action_horizon`, `state_names`, `action_names`, `end_output` (`progress`), `image` (`size` 256, `optics` water), `tasks` |
| `POST /act` | request `{seq, run, image_png_b64, state[15], task}` → `{seq, action[T][16], infer_ms}` — absolute tool poses + trigger + progress |

## Server

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))" > policy.token && chmod 600 policy.token
python policy_server.py --backend lerobot --policy-path <checkpoint>/pretrained_model --device cuda --token-file policy.token --port 8900
curl -s http://127.0.0.1:8900/health
python ../remote_policy.py http://127.0.0.1:8900 --token-file policy.token --n 3     # /info + three /act calls: shapes, round trip
```

Needs the training environment (LeRobot 0.6.1 with GR00T) plus `../remote_policy.py` and `robot/dataset/`'s
`actions.py`, `phases.py`, `instructions.py` in the same folder. Backends: `lerobot` (LeRobot checkpoints, used for all results), `demo` (answers with the dataset's next
frames, for dry runs). The container image for a Nebius Serverless Endpoint is in `robot/train/nebius/serverless/`.

## Connecting

Do not expose plain HTTP with only a token. Either an SSH tunnel (`tunnel.sh up <user>@<host> [server port] [local port]`,
then `--policy-url http://127.0.0.1:18900`) or HTTPS (a Serverless Endpoint's managed URL, or TLS in front of the server;
a self-signed certificate can be pinned with `--policy-ca-file`).

## Robot side

`remote_policy.abs_to_chunk()` turns the absolute answer into actions relative to the observation; from there the loop is
the same as for a local model (step limits, pivot at the skin, safety gate, trigger off outside the view, end rules).
Only the endoscope image of the phantom scene, the two tool poses, the instruction sentence, a counter and the run name
leave the robot computer.

## Files

| file | does |
|---|---|
| `policy_server.py` | reference server (`lerobot`, `demo`; `groot` = Isaac-GR00T code, untested) |
| `../remote_policy.py` | protocol, client, conversions; run directly for a connection check |
| `connect_check.py`, `endpoint.py` | connection check and address classification before a robot run |
| `lang_check.py` | checks that the server knows the instruction sentences of a run |
| `replay_obs.py` | asks the server again with the observations a robot run recorded (checkpoint comparison without the robot) |
| `equiv_check.py` | the remote path and a local model give the same chunk |
| `intake_summary.py` | summary of a new model's connection check, open-loop scores and dry run |
| `tunnel.sh`, `make_bundle.sh` | SSH tunnel; flat server bundle |
| `groot_map.example.json` | dataset values ↔ GR00T modality keys |
