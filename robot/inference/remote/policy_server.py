#!/usr/bin/env python3
"""Reference policy server for the closed loop's remote policy (protocol "spine-policy/1", ../remote_policy.py).

    python policy_server.py --backend lerobot --policy-path CKPT/pretrained_model --device cuda --token-file policy.token
    python policy_server.py --backend groot   --model-path CKPT --groot-map groot_map.json --token-file policy.token
    python policy_server.py --backend demo    --dataset DS --episodes 2,18 [--progress-output] --token-file policy.token

The robot (Lenovo: tracker, arms, recorder page, safety gate) asks, the server answers with the next action chunk:

  GET  /health → {"ok": true}                          no token; "is it up?"
  GET  /info   → what the model is (see remote_policy.py)
  POST /act    ← {"seq", "run", "image_png_b64" (or "image_jpeg_b64"), "state": [15], "task": str}
               → {"seq", "action": [[15 or 16] × T], "infer_ms"}     ABSOLUTE rows: the tools' state 1 … T frames ahead
                                                                      [+ drill.progress | drill.done], the dataset's layout
                                                                      and units

Backends
  demo    answers with a dataset's own next states for the frame whose state is nearest to the request's — no model. For
          checking the connection, the token, the timing and the robot side's conversions (the loop then replays the demo).
  lerobot a LeRobot checkpoint (`lerobot-train` output, …/pretrained_model) — the way robot/dataset/README.md §6 trains
          GR00T (`--policy.type=groot`), and our ACT checkpoints. Needs lerobot (the version that trained it). The policy's
          own pre- and postprocessors are used, so the answer is in the dataset's action space: absolute for a GR00T trained
          on the shared dataset (its `use_relative_actions` is undone by the postprocessor over the whole chunk — LeRobot's
          GrootPolicy asks for exactly that), chunk-relative for our ACT (action_rep.json; turned into absolute rows here).
          Run on the robot server with ACT checkpoints (remote/equiv_check.py: the same chunk as a local run to 2e-6 mm);
          NOT run with a GR00T checkpoint by its author (no GPU there) — check it with remote_policy.py on the GPU machine.
  groot   a policy of NVIDIA's own Isaac-GR00T code (not LeRobot's). GrootBackend below is written from that API's
          documentation and was never run: change load()/predict() where your GR00T version differs.

One request at a time (a lock): the loop asks once per chunk and waits. Bind to 127.0.0.1 and reach it through an SSH
tunnel from the robot server, or put TLS in front — the token alone on plain HTTP over the internet is readable.
Needs: python ≥ 3.10, numpy, pillow (+ what the backend needs: lerobot / gr00t / pyarrow). Files: this, remote_policy.py,
actions.py, phases.py.
"""
from __future__ import annotations

import argparse
import hmac
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

for _d in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (_d / "remote_policy.py").exists() and str(_d) not in sys.path:
        sys.path.insert(0, str(_d))
import remote_policy as RP   # noqa: E402

MAX_BODY = 8 * 1024 * 1024   # a 256 × 256 PNG is ~150 kB


class Backend:
    """What a backend gives: info() once, predict() per request → absolute rows (T, 15 or 16)."""
    model = "?"; horizon = 16; image_size = 256; optics = "water"
    done_output = False              # the model has a 16th action value that ends the drilling …
    end_name = RP.PROGRESS_NAME      # … and this is what it is: drill.progress (removed fraction) or drill.done (0/1 flag)
    instructions = None              # the instruction set the model was trained with (dataset/instructions.py table()); None = phase1

    def read_instructions(self, path) -> None:
        """--instructions-file: the training dataset's meta/spine_il/export.json (its "instructions"), or such a table itself."""
        if not path:
            return
        d = json.loads(Path(path).expanduser().read_text()); d = d.get("instructions", d)
        if not isinstance(d, dict) or not d.get("name"):
            raise SystemExit(f"{path}: no instruction table (\"instructions\": {{\"name\": …}})")
        # a set this code knows is written out by this code (a dataset exported earlier may carry an older shape of the table)
        self.instructions = None if d["name"] == "phase1" else RP.INS.table(d["name"]) if d["name"] in RP.INS.SETS else d

    def info(self) -> dict:
        return {"protocol": RP.PROTOCOL, "model": self.model, "action_horizon": int(self.horizon), "fps": 15,
                "state_names": RP.STATE_NAMES, "action_names": RP.STATE_NAMES + ([self.end_name] if self.done_output else []),
                "done_output": bool(self.done_output), "end_output": RP.END_KINDS[self.end_name] if self.done_output else None,
                "image": {"view": "endo", "size": int(self.image_size), "optics": self.optics, "formats": list(RP.IMAGE_FORMATS)},
                "tasks": RP.INS.texts(self.instructions) if self.instructions else [RP.TASKS[3], RP.TASKS[4]] + ([] if self.done_output else [RP.TASKS[5]]),
                **({"instructions": self.instructions} if self.instructions else {}),
                **({"guidance": {"w": self.guidance, "negative": NEGATIVES[getattr(self, "guidance_negative", "other")]}} if getattr(self, "guidance", 0) else {})}

    def predict(self, image: np.ndarray, state: np.ndarray, task: str) -> np.ndarray:
        raise NotImplementedError


class DemoBackend(Backend):
    """The dataset's own continuation: the frame (of the given episodes) whose state is nearest to the request's, and the
    T states after it. With --progress-output the 16th value is those frames' cut.progress (bone removed / the session's
    total), with --done-output the flag "after the episode's last cut" (phase ≥ 5).
    The demonstration is continued IN ORDER: the nearest frame is looked for from where the last answer began to a few
    answers ahead. A demonstration passes the same places on its way in and on its way out — on 10-09 the first question
    of a dry run, asked at the entry, was answered with the episode's end ("progress 1.0") and the loop ended the drilling
    with nothing removed. A question that names the approach ("3 …") is answered from the frames before the episode's first
    cut; one that lies far from the stretch ahead starts anew (nearest frame of all). A question that IS a frame of the
    dataset — the evaluation tools ask those, in any order (remote/connect_check.py, end_level.py) — is answered from that
    frame: on 10-12 the connection check, jumping through an episode, got the continuation of a place 10–19° away."""
    AHEAD = 4        # answers: how far ahead of the last answer's frame the next one may begin
    FAR_MM = 25.0    # endoscope + drill tip distance beyond which the question is not on the stretch ahead

    def __init__(self, a):
        import glob
        import pyarrow.parquet as pq
        ds = Path(a.dataset).expanduser(); self.model = f"demo:{ds.name}"; self.horizon = a.horizon
        self.read_instructions(getattr(a, "instructions_file", None))
        self.relative = bool(getattr(a, "relative", False))
        if a.done_output and a.progress_output:
            raise SystemExit("--done-output and --progress-output are two forms of one value: give one")
        self.done_output = bool(a.done_output or a.progress_output); self.end_name = RP.DONE_NAME if a.done_output else RP.PROGRESS_NAME
        eps = [int(x) for x in a.episodes.split(",") if x != ""]
        cols = {"episode_index": [], "frame_index": [], "observation.state": [], "phase.index": [], **({"cut.progress": []} if a.progress_output else {})}
        for f in sorted(glob.glob(str(ds / "data/*/*.parquet"))):
            t = pq.read_table(f, columns=list(cols))
            for k in cols:
                cols[k] += t.column(k).to_pylist()
        e = np.asarray(cols["episode_index"]); fi = np.asarray(cols["frame_index"])
        st = np.asarray(cols["observation.state"], float)[:, :15]
        ph = np.asarray(cols["phase.index"]).reshape(-1)
        # the 16th value of every frame: its progress, or the flag "after the last cut"
        end = np.asarray(cols["cut.progress"], float).reshape(-1) if a.progress_output else (ph >= 5).astype(float)
        self.ep, self.first_cut, self.cur = {}, {}, None
        for k in eps:
            sel = np.nonzero(e == k)[0]; o = sel[np.argsort(fi[sel])]
            if not len(o):
                raise SystemExit(f"episode {k} is not in {ds}")
            self.ep[k] = (st[o], end[o])
            drilling = np.nonzero(ph[o] >= 4)[0]
            self.first_cut[k] = int(drilling[0]) if len(drilling) else len(o) - 1
        print(f"[server] demo backend: {ds.name}, episodes {eps}, {sum(len(v[0]) for v in self.ep.values())} frames", flush=True)

    @staticmethod
    def _dist(st, state):
        return np.linalg.norm(st[:, 7:10] - state[7:10], axis=1) + np.linalg.norm(st[:, 0:3] - state[0:3], axis=1)

    def _nearest(self, state, before_first_cut: bool):
        best = None
        for k, (st, _) in self.ep.items():
            n = self.first_cut[k] + 1 if before_first_cut else len(st)
            d = self._dist(st[:n], state); i = int(np.argmin(d))
            if best is None or d[i] < best[0]:
                best = (float(d[i]), k, i)
        return best[1], best[2]

    def _exact(self, state, before_first_cut: bool):
        hits = []
        for k, (st, _) in self.ep.items():
            n = self.first_cut[k] + 1 if before_first_cut else len(st)      # the approach is answered from the approach
            hits += [(k, int(j)) for j in np.nonzero(np.abs(st[:n, :14] - state[None, :14]).max(1) < 1e-4)[0][:8]]
        if hits and self.cur is not None:      # a place the demonstration rests at: the frame at or after the last answer's
            ahead = [h for h in hits if h[0] == self.cur[0] and h[1] >= self.cur[1]]
            return ahead[0] if ahead else hits[0]
        return hits[0] if hits else None

    def predict(self, image, state, task):
        approach = str(task).strip().startswith("3")
        pick = self._exact(np.asarray(state, float), approach)
        if pick is None and self.cur is not None and not approach:
            k, i0 = self.cur; st = self.ep[k][0]
            hi = min(len(st), i0 + self.AHEAD * self.horizon + 1)
            d = self._dist(st[i0:hi], state)
            if len(d) and float(d.min()) <= self.FAR_MM:
                pick = (k, i0 + int(np.argmin(d)))
        if pick is None:
            pick = self._nearest(state, before_first_cut=approach)
        self.cur = pick
        k, i = pick; st, end = self.ep[k]
        idx = np.minimum(np.arange(i + 1, i + 1 + self.horizon), len(st) - 1)   # past the episode's end: its last state, held
        rows = st[idx].copy()
        if self.relative:      # as a relative-action model answers: the state it was given + the demonstration's displacement
            rows = RP.chunk_to_abs(np.asarray(state, float), RP.abs_to_chunk(st[i], rows))[:, :15]
        return np.concatenate([rows, end[idx][:, None]], 1) if self.done_output else rows


# ---------------------------------------------------------------- guidance by the words (10-14)
# User 10-14: the extent words hardly steered the model (lang_check / why/vm_probe: progress ±0.04–0.07, the drill path at
# noise). GR00T draws a chunk in 4 Euler steps of a learned velocity field; with --guidance W the same noise is integrated
# with  v = v(the goal sentence) + W · (v(goal) − v(the other extent's sentence))  — classifier-free guidance with the other
# goal as the negative prompt. Offline on L30c2's decision frames (why/vm_guidance.py): W 1 → progress apart −0.08 / +0.17,
# W 3 → −0.17 / +0.40 and the drill path apart; chunks rougher by 8–12 % (W 1) … 35–43 % (W 3).
_GUIDE = {"w": 0.0}


NEGATIVES = {"other": "the other extent at the same level (first wording)",
             "neutral": "the sentence that names no extent, same level (first wording; lang2)"}


def negative_sentences(name: str = "lang1", negative: str = "other") -> dict:
    """drilling sentence of an extent (every wording) → its negative prompt: the other extent's first wording at the same
    level ("other"), or lang2's sentence that names no extent ("neutral": plain classifier-free guidance — a lang2 model
    learnt that answer with --cond-dropout). The neutral sentence itself is asked without guidance."""
    other = {"ipsilateral": "contralateral", "contralateral": "ipsilateral"}
    return {w: RP.INS.text(name, 4, lv, RP.INS.NEUTRAL if negative == "neutral" else other[ex])
            for lv in RP.INS.LEVELS for ex in RP.INS.EXTENTS for w in RP.INS.wordings(4, lv, ex, name)}


def _guide_groot_head(head_cls) -> None:
    """Replace GR00T N1.7's sampling loop for a batch of two (goal, negative): one noise, the guided velocity for both rows."""
    if getattr(head_cls, "_spine_guided", False):
        return
    orig = head_cls.get_action_with_features
    import torch

    def guided(self, backbone_features, state_features, embodiment_id, backbone_output, action_input, options=None):
        if backbone_features.shape[0] != 2 or "action" in action_input:
            return orig(self, backbone_features, state_features, embodiment_id, backbone_output, action_input, options)
        w, vl, dev = _GUIDE["w"], backbone_features, backbone_features.device
        a = torch.randn((1, self.config.action_horizon, self.action_dim), dtype=vl.dtype, device=dev).repeat(2, 1, 1)
        dt = 1.0 / self.num_inference_timesteps
        for t_step in range(self.num_inference_timesteps):
            ts = torch.full((2,), int(t_step / float(self.num_inference_timesteps) * self.num_timestep_buckets), device=dev)
            af = self.action_encoder(a, ts, embodiment_id)
            if self.config.add_pos_embed:
                af = af + self.position_embedding(torch.arange(af.shape[1], dtype=torch.long, device=dev)).unsqueeze(0)
            kw = dict(image_mask=backbone_output.image_mask, backbone_attention_mask=backbone_output.backbone_attention_mask) \
                if self.config.use_alternate_vl_dit else {}
            out = self.model(hidden_states=torch.cat((state_features, af), dim=1), encoder_hidden_states=vl, timestep=ts, **kw)
            v = self.action_decoder(out, embodiment_id)[:, -self.action_horizon:]
            a = a + dt * (v[0] + w * (v[0] - v[1]))[None].repeat(2, 1, 1)
        from transformers.feature_extraction_utils import BatchFeature
        return BatchFeature(data={"action_pred": a, "backbone_features": vl, "state_features": state_features})
    head_cls.get_action_with_features = guided
    head_cls._spine_guided = True


class LerobotBackend(Backend):
    """A LeRobot checkpoint (…/pretrained_model): GR00T trained with `lerobot-train --policy.type=groot`, or our ACT."""
    WARM = np.array([40, -20, 1030, 0.6, -0.6, -0.45, 0.28, 30, -25, 1020, 0.62, 0.64, 0.36, 0.3, 0], float)   # a pose inside the data

    def __init__(self, a):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        self.torch = torch; path = str(Path(a.policy_path).expanduser())
        if a.device == "cpu":
            torch.set_num_threads(a.torch_threads)
        cfg = PreTrainedConfig.from_pretrained(path); cfg.device = a.device
        self.pol = get_policy_class(cfg.type).from_pretrained(path, config=cfg); self.pol.eval()
        self.pre, self.post = make_pre_post_processors(cfg, pretrained_path=path,
                                                       preprocessor_overrides={"device_processor": {"device": a.device}})
        f = Path(path) / "action_rep.json"; rep = json.loads(f.read_text()) if f.exists() else {}
        self.read_instructions(getattr(a, "instructions_file", None))
        self.guidance = float(getattr(a, "guidance", 0.0) or 0.0); self.negative = {}
        self.guidance_negative = getattr(a, "guidance_negative", None) or "other"
        if self.guidance:
            name = (self.instructions or {}).get("name")
            if cfg.type != "groot" or name not in ("lang1", "lang2"):
                raise SystemExit("--guidance needs a GR00T model trained with the lang1 / lang2 instructions (--instructions-file)")
            if self.guidance_negative == "neutral" and name != "lang2":
                raise SystemExit("--guidance-negative neutral needs a lang2 model — only lang2 has a sentence that names no extent")
            head = next((m for n, m in self.pol.named_modules() if n.endswith("action_head") and hasattr(m, "get_action_with_features")), None)
            if head is None:
                raise SystemExit("--guidance: no GR00T action head with get_action_with_features in this policy")
            _guide_groot_head(type(head)); _GUIDE["w"] = self.guidance; self.negative = negative_sentences(name, self.guidance_negative)
        f = Path(path) / "step_norm.json"; sn = json.loads(f.read_text()) if f.exists() else None
        self.rep = rep.get("rep") or "absolute"                       # our ACT: "chunk_relative"; a dataset-space policy: absolute
        self.fold = bool(rep.get("done_output"))                      # our end-output ACT never saw the "withdraw" phase bit
        # what a 16th (ACT: 14th) value is: our ACT's action_rep.json says (end_output; the first such models: a flag),
        # another checkpoint is taken at --end-output's word
        kind = rep.get("end_output") or ("done" if rep.get("done_output") else a.end_output)
        self.end_name = RP.DONE_NAME if kind == "done" else RP.PROGRESS_NAME
        self.step_norm = {"mean": np.asarray(sn["mean"], float), "std": np.asarray(sn["std"], float)} if sn else None
        # the unnormaliser of ACT takes one step (B, A); GR00T's postprocessor decodes the whole chunk (B, T, A) — relative
        # actions back to absolute against the observed state
        self.per_step = (cfg.type == "act") if a.post == "auto" else (a.post == "step")
        self.n_state = int(cfg.input_features["observation.state"].shape[0])
        self.image_key = next(k for k in cfg.input_features if k.startswith("observation.images"))
        self.image_size = int(cfg.input_features[self.image_key].shape[-1])
        self.optics = rep.get("optics") or a.optics; self.model = f"{cfg.type}:" + "/".join(Path(path).parts[-4:])
        warm = self.WARM.copy(); warm[3:7] /= np.linalg.norm(warm[3:7]); warm[10:14] /= np.linalg.norm(warm[10:14])
        ch = self._chunk(np.zeros((self.image_size, self.image_size, 3), np.uint8), warm, RP.TASKS[4])   # loads lazily, shows T and A
        want = (13, 14) if self.rep == "chunk_relative" else (15, 16)
        if ch.ndim != 2 or ch.shape[1] not in want:
            raise SystemExit(f"the policy's chunk is {ch.shape}; a {self.rep} policy must give (T, {want[0]}) or (T, {want[1]})")
        self.horizon = len(ch); self.done_output = ch.shape[1] == want[1]
        print(f"[server] lerobot backend: {self.model} on {a.device}, {self.rep} actions, {self.horizon} rows, "
              f"end output {self.end_name if self.done_output else None}, postprocess per {'step' if self.per_step else 'chunk'}", flush=True)

    def _chunk(self, image, state, task) -> np.ndarray:
        """The policy's chunk in physical units, (T, A)."""
        torch = self.torch
        ph = next((k for k, v in RP.TASKS.items() if v == task), 4)   # the phase one-hot of a 21-value state (our ACT)
        if self.n_state == 15:
            sv = state
        elif self.n_state == 16:   # a --done-output dataset: the state's 16th value pad.zero, always 0 (export_lerobot.py)
            sv = np.r_[state, 0.0]
        else:                      # 21: the phase one-hot of our ACT datasets
            sv = np.concatenate([state, np.eye(6)[min(4 if self.fold else 5, max(3, ph)) - 1]])
        neg = self.negative.get(task)
        if neg is None:      # no guidance: the model as it was trained
            obs = {"observation.state": torch.tensor(sv, dtype=torch.float32)[None],
                   self.image_key: torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)[None] / 255.0, "task": [task]}
        else:                # a batch of two — the goal and its negative — sampled together by _guide_groot_head; row 0 is the answer
            obs = {"observation.state": torch.tensor(np.stack([sv, sv]), dtype=torch.float32),
                   self.image_key: torch.tensor(np.stack([image, image]), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0, "task": [task, neg]}
        with torch.no_grad():
            ch = self.pol.predict_action_chunk(self.pre(obs))
            ch = torch.stack([self.post(ch[:, j]) for j in range(ch.shape[1])], 1) if self.per_step else self.post(ch)
        ch = ch[0].float().cpu().numpy()
        if self.step_norm is not None:
            ch = ch * self.step_norm["std"][:len(ch)] + self.step_norm["mean"][:len(ch)]
        return ch

    def predict(self, image, state, task):
        ch = self._chunk(image, state, task)
        return RP.chunk_to_abs(state, ch) if self.rep == "chunk_relative" else ch


class GrootBackend(Backend):
    """A policy of NVIDIA's Isaac-GR00T code (for a LeRobot-trained GR00T use the lerobot backend). NEVER RUN by its
    author — written from that API's documentation; adjust on the GPU machine.

    --groot-map FILE says how the dataset's flat state/action (remote_policy.py STATE_NAMES [+ drill.done]) maps onto
    the modality keys your GR00T data config uses, e.g. (the keys are the ones of YOUR meta/modality.json):
        {"video": "video.endo", "language": "annotation.human.task_description",
         "state":  {"state.endo_pos": [0, 3], "state.endo_quat": [3, 7], "state.drill_pos": [7, 10],
                    "state.drill_quat": [10, 14], "state.trigger": [14, 15]},      (+ "state.pad": [15, 16] for a done dataset)
         "action": {"action.endo_pos": [0, 3], "action.endo_quat": [3, 7], "action.drill_pos": [7, 10],
                    "action.drill_quat": [10, 14], "action.trigger": [14, 15], "action.progress": [15, 16]}}
    "action" has the [15, 16] entry exactly when the model was trained with the progress (or done) output; --end-output
    says which of the two it is. The policy must return ABSOLUTE values in the dataset's units (GR00T undoes
    use_relative_actions and the normalisation itself).
    """

    def __init__(self, a):
        m = json.loads(Path(a.groot_map).read_text()); self.map = m
        self.read_instructions(getattr(a, "instructions_file", None))
        self.done_output = any(tuple(v) == (15, 16) for v in m["action"].values())
        self.end_name = RP.DONE_NAME if a.end_output == "done" else RP.PROGRESS_NAME
        self.model = "groot:" + Path(a.model_path).name; self.horizon = a.horizon
        self.policy, self.style = self.load(a)
        print(f"[server] groot backend ({self.style} API): {self.model}, end output {self.end_name if self.done_output else None}", flush=True)

    @staticmethod
    def load(a):
        """Build the policy. Two API generations are tried; keep the one your GR00T has and delete the other."""
        try:   # GR00T N1.6-style: gr00t.policy.gr00t_policy.Gr00tPolicy(embodiment_tag, model_path, device)
            from gr00t.policy.gr00t_policy import Gr00tPolicy
            from gr00t.data.embodiment_tags import EmbodimentTag
            return Gr00tPolicy(embodiment_tag=EmbodimentTag(a.embodiment_tag), model_path=a.model_path, device=a.device), "nested"
        except ImportError:
            pass
        # GR00T N1 / N1.5-style: gr00t.model.policy.Gr00tPolicy with a data config's modality config and transform
        from gr00t.experiment.data_config import DATA_CONFIG_MAP
        from gr00t.model.policy import Gr00tPolicy
        dc = DATA_CONFIG_MAP[a.data_config]
        return Gr00tPolicy(model_path=a.model_path, modality_config=dc.modality_config(), modality_transform=dc.transform(),
                           embodiment_tag=a.embodiment_tag, denoising_steps=a.denoising_steps, device=a.device), "flat"

    def predict(self, image, state, task):
        m = self.map; st = np.asarray(state, np.float32)
        st = np.r_[st, np.zeros(max(0, max(hi for _, hi in m["state"].values()) - len(st)), np.float32)]   # pad.zero of a done dataset
        if self.style == "flat":      # {"video.x": (1, H, W, 3) uint8, "state.y": (1, D), "annotation…": [str]} → {"action.z": (T, D)}
            obs = {m["video"]: image[None], m["language"]: [task], **{k: st[None, lo:hi] for k, (lo, hi) in m["state"].items()}}
            out = self.policy.get_action(obs)
        else:                         # {"video": {x: (1, 1, H, W, 3)}, "state": {y: (1, 1, D)}, "language": {z: [[str]]}} → ({key: (1, T, D)}, info)
            strip = lambda k: k.split(".", 1)[1] if "." in k else k
            obs = {"video": {strip(m["video"]): image[None, None]}, "language": {strip(m["language"]): [[task]]},
                   "state": {strip(k): st[None, None, lo:hi] for k, (lo, hi) in m["state"].items()}}
            out, _ = self.policy.get_action(obs)
            out = {"action." + k: np.asarray(v)[0] for k, v in out.items()}
        T = min(len(np.asarray(out[k])) for k in m["action"]); rows = np.zeros((T, 16 if self.done_output else 15))
        for k, (lo, hi) in m["action"].items():
            rows[:, lo:hi] = np.asarray(out[k], float).reshape(len(np.asarray(out[k])), -1)[:T]
        return rows


BACKENDS = {"demo": DemoBackend, "lerobot": LerobotBackend, "groot": GrootBackend}


def make_handler(backend: Backend, token: str | None, log_every: int = 1, delay_s: float = 0.0):
    lock = threading.Lock(); n = [0]

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):   # one line per /act below; nothing else
            pass

        def _send(self, code: int, obj: dict):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def _authorised(self) -> bool:
            if token is None:
                return True
            got = self.headers.get("Authorization", "")
            return got.startswith("Bearer ") and hmac.compare_digest(got[7:].strip(), token)

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"ok": True})
            if not self._authorised():
                return self._send(401, {"error": "token"})
            if self.path == "/info":
                return self._send(200, backend.info())
            self._send(404, {"error": "no such path"})

        def do_POST(self):
            if not self._authorised():
                return self._send(401, {"error": "token"})
            if self.path != "/act":
                return self._send(404, {"error": "no such path"})
            try:
                ln = int(self.headers.get("Content-Length") or 0)
                if not 0 < ln <= MAX_BODY:
                    return self._send(413, {"error": f"body of {ln} bytes"})
                d = json.loads(self.rfile.read(ln))
                if d.get("protocol") != RP.PROTOCOL:
                    return self._send(400, {"error": f"protocol {d.get('protocol')!r}, this server speaks {RP.PROTOCOL!r}"})
                state = np.asarray(d["state"], float)
                if state.shape != (15,) or not np.isfinite(state).all():
                    return self._send(400, {"error": f"state must be 15 finite numbers, got shape {state.shape}"})
                image = RP.decode_image(d.get("image_png_b64") or d["image_jpeg_b64"])
                if image.shape != (backend.image_size, backend.image_size, 3):
                    return self._send(400, {"error": f"image must be {backend.image_size} × {backend.image_size} RGB, got {image.shape}"})
                task = str(d.get("task") or "")
            except Exception as e:
                return self._send(400, {"error": f"{type(e).__name__}: {e}"})
            try:
                if delay_s > 0:      # --delay-ms: a stand-in for a far-away GPU (dry runs of the continuous loop)
                    time.sleep(delay_s)
                with lock:
                    t0 = time.monotonic()
                    rows = np.asarray(backend.predict(image, state, task), float)
                    ms = (time.monotonic() - t0) * 1000.0
                want = 16 if backend.done_output else 15
                if rows.ndim != 2 or rows.shape[1] != want or not np.isfinite(rows).all():
                    raise ValueError(f"backend returned {rows.shape}, expected (T, {want}) finite")
            except Exception as e:
                print(f"[server] /act seq {d.get('seq')} FAILED: {type(e).__name__}: {e}", flush=True)
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})
            n[0] += 1
            if n[0] % log_every == 0:
                print(f"[server] /act seq {d.get('seq')} run {str(d.get('run'))[:40]!r}: {len(rows)} rows in {ms:.0f} ms, "
                      f"trigger {rows[0, 14]:.2f}" + (f", {backend.end_name} up to {rows[:, 15].max():.2f}" if want == 16 else ""), flush=True)
            self._send(200, {"seq": d.get("seq"), "action": np.round(rows, 6).tolist(), "infer_ms": round(ms, 1)})

    return H


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = reachable through an SSH tunnel only (recommended)")
    ap.add_argument("--port", type=int, default=8900)
    ap.add_argument("--token-file", default=None, help="file with the bearer token (chmod 600); required unless --no-auth")
    ap.add_argument("--no-auth", action="store_true", help="no token — only with --host 127.0.0.1")
    ap.add_argument("--horizon", type=int, default=16, help="rows per answer (demo, groot)")
    ap.add_argument("--dataset", default=None); ap.add_argument("--episodes", default="2,18")
    ap.add_argument("--progress-output", action="store_true", help="demo: add drill.progress from the dataset's cut.progress column")
    ap.add_argument("--done-output", action="store_true", help="demo: add drill.done (the 0/1 flag) from the dataset's phase column")
    ap.add_argument("--end-output", choices=["progress", "done"], default="progress",
                    help="lerobot, groot: what the model's 16th action value is when it has one and the checkpoint does not say — "
                         "progress (the datasets from 10-09 on) or done (a 0/1 flag)")
    ap.add_argument("--policy-path", default=None, help="lerobot: the checkpoint's pretrained_model folder")
    ap.add_argument("--torch-threads", type=int, default=4, help="lerobot on cpu")
    ap.add_argument("--post", choices=["auto", "step", "chunk"], default="auto",
                    help="lerobot: the postprocessor takes one step (act) or the whole chunk (groot and others); auto = by policy type")
    ap.add_argument("--guidance", type=float, default=0.0,
                    help="lerobot GR00T with lang1 / lang2 instructions: guidance weight W of the goal sentence against the other extent's "
                         "(v = v(goal) + W·(v(goal) − v(other)); 0 = off). Offline 10-14: W 1–3 make the extent words count")
    ap.add_argument("--guidance-negative", choices=sorted(NEGATIVES), default="other",
                    help="--guidance: the negative prompt — the other extent's sentence, or (lang2) the sentence that names no extent")
    ap.add_argument("--instructions-file", default=None,
                    help="the instruction set the model was trained with: its dataset's meta/spine_il/export.json (or the table "
                         "alone). Told to the loop in /info, which then says the goal in that set's words; none = the legacy phase texts")
    ap.add_argument("--optics", default="water", help="the medium of the model's training images, when the checkpoint does not say")
    ap.add_argument("--model-path", default=None); ap.add_argument("--groot-map", default=None)
    ap.add_argument("--embodiment-tag", default="new_embodiment"); ap.add_argument("--data-config", default=None)
    ap.add_argument("--denoising-steps", type=int, default=4); ap.add_argument("--device", default="cuda")
    ap.add_argument("--delay-ms", type=float, default=0.0, help="answer /act this much later — a stand-in for network and model time in dry runs")
    ap.add_argument("--relative", action="store_true",
                    help="demo backend: answer as a relative-action model does — the request's state + the demonstration's "
                         "displacement from its matched frame — instead of the demonstration's own poses. The first continuous "
                         "robot run (10-12) swung because of exactly that difference: a fixed trajectory cannot echo what the "
                         "arms were measured to be off, a pose + displacement does (inference/stream_plan.py)")
    a = ap.parse_args()
    if a.no_auth and a.host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("--no-auth only on 127.0.0.1")
    if not a.no_auth and not a.token_file:
        raise SystemExit("--token-file is required (or --no-auth on 127.0.0.1)")
    token = None if a.no_auth else RP.read_token(a.token_file)
    backend = BACKENDS[a.backend](a)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(backend, token, delay_s=a.delay_ms / 1000.0))
    print(f"[server] {RP.PROTOCOL} on http://{a.host}:{a.port} — {json.dumps(backend.info(), ensure_ascii=False)}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
