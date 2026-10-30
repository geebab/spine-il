#!/usr/bin/env python3
"""A policy that lives on another machine: the closed loop asks a server for the next action chunk.

    live_loop.py --policy-url https://HOST:PORT --policy-token-file ~/.config/ube/policy.token [--optics water]

User, 10-09: "Nebius 서버로의 연결 해" — the GR00T models are trained and served on a Nebius GPU by a teammate; the robot
side (tracker, arms, recorder page, safety gate, phase rules) stays on the Lenovo and only the question "what next?" goes
out. This file is the robot side of that question and the wire format both sides keep to (protocol "spine-policy/1");
remote/policy_server.py is a reference server with the same format (backends: a dataset's own actions, a local ACT
checkpoint, GR00T).

Protocol (JSON over HTTP or HTTPS, every request with "Authorization: Bearer <token>"):

  GET  /info  →  {"protocol": "spine-policy/1", "model": str, "action_horizon": T, "fps": 15,
                  "state_names": [15], "action_names": [15 or 16], "done_output": bool,
                  "image": {"view": "endo", "size": 256, "optics": "water"}, "tasks": [str, …]}
  POST /act   ←  {"protocol": "spine-policy/1", "seq": int, "run": str,
                  "image_png_b64": str,      the endoscope view the policy was trained on, RGB, size × size, PNG, base64
                  "state": [15 floats],      endo tip xyz (scene mm), endo quat wxyz, drill tip xyz, drill quat wxyz, trigger
                  "task": str}               the phase instruction of the dataset (meta/tasks.parquet), see TASKS
              →  {"seq": int, "action": [[15 or 16 floats] × T], "infer_ms": float}

  action row j = the state the tools should have j+1 frames (1/15 s each) after the observation — ABSOLUTE, in the
  dataset's own layout and units: the 15 state values, and for a model that ends the drilling itself a 16th value —
  `drill.progress` (user 10-09: the bone removed so far / the demonstration's final removed volume, 0 … 1; the loop ends
  the drilling when it stays above 0.91 for 5 steps) or, the earlier form, `drill.done` (a 0/1 flag; above 0.5). The
  name in /info's action_names says which. That is what a GR00T policy returns for the shared dataset (action = next
  frame's state [+ that value]; `use_relative_actions` is undone by the policy itself before it answers).

The loop works in chunk-relative values (dataset/actions.py), so abs_to_chunk() turns the answer into exactly what
train/lerobot-train-rel.py's to_chunk_relative() makes of the same frames — everything after that (step limits, pivot,
safety gate, the done rule) is the code every ACT run used.

Nothing is sent but the endoscope image of the phantom scene, the two tool poses, the phase text, a counter and the
run's name. The token is read from a file (never a command line); a server that does not answer in time is a fault of
the run like any other: the loop stops asking and the harness withdraws the tools.
"""
from __future__ import annotations

import base64
import io
import json
import os
import stat
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

# dataset/actions.py and dataset/phases.py (numpy only): in the repository under robot/dataset, in the bundle handed to
# the server side (remote/make_bundle.sh) beside this file
for _d in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1] / "dataset"):
    if (_d / "actions.py").exists() and str(_d) not in sys.path:
        sys.path.insert(0, str(_d))
import actions as AC   # noqa: E402
import instructions as INS   # noqa: E402
import phases as PH    # noqa: E402

PROTOCOL = "spine-policy/1"
STATE_NAMES = [f"{t}.{c}" for t in ("endo", "drill") for c in ("x", "y", "z", "qw", "qx", "qy", "qz")] + ["drill.trigger"]
DONE_NAME = "drill.done"            # the 16th action value as a 0/1 flag: 1 = the drilling is over from this step on
PROGRESS_NAME = "drill.progress"    # … or as the progress: bone removed / the demonstration's final removed volume, 0 … 1
END_KINDS = {DONE_NAME: "done", PROGRESS_NAME: "progress"}
# the instruction of a frame's phase, as the dataset stores it (meta/tasks.parquet = dataset/phases.py PHASES): the trimmed
# data has phases 3 (skin → target), 4 (drilling) and 5 (withdraw). A model trained with the done output never saw the
# "withdraw" text — the frames after its last cut carry the drilling instruction (export_lerobot.py --done-output)
TASKS = {k: PH.PHASES[k] for k in (3, 4, 5)}


def task_for(phase: int, done_output: bool, instructions: dict | None = None, goal: dict | None = None) -> str:
    """The instruction the loop sends for the harness's phase: 3 before the first cut, 4 from it on; 5 only to a model
    without the done output (the harness has ended the drilling by then and does not ask any more).

    `instructions` = the server's /info "instructions" (the set the model was trained with, dataset/instructions.py table();
    none = the legacy phase texts above). A model of the set lang1 is told the goal in words — `goal` {"level", "extent"}
    (user, 10-13: language decides where the same scene calls for another action; default left L4, with contralateral decompression) —
    always in the set's first wording. The legacy models were never told a goal: one given for them is an error."""
    ph = min(4 if done_output else 5, max(3, int(phase)))
    name = (instructions or {}).get("name") or "phase1"
    if name == "phase1":
        if goal:
            raise ValueError("this model's instructions (phase1) name no goal — a goal can only be given to a model trained with lang1")
        return TASKS[ph]
    g = goal or {}
    if ph == 4 and g.get("drill_text"):     # a sentence of the caller's own (10-16: wordings the model never met)
        return str(g["drill_text"])
    return INS.text(name, ph, g.get("level") or "L4", g.get("extent") or INS.DEFAULT_EXTENT)


class RemoteError(RuntimeError):
    """The policy server did not give a usable answer (network, timeout, HTTP status, malformed reply)."""


def read_token(path: str | os.PathLike | None) -> str | None:
    """The bearer token from a file readable by its owner only. None when no file is given (a tunnel to a server without
    authentication)."""
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.is_file():
        raise RemoteError(f"token file {p} not found")
    if stat.S_IMODE(p.stat().st_mode) & 0o077:
        raise RemoteError(f"token file {p} is readable by others (chmod 600 {p})")
    tok = p.read_text().strip()
    if not tok:
        raise RemoteError(f"token file {p} is empty")
    return tok


IMAGE_FORMATS = ("png", "jpeg")


def image_format(spec: str | None) -> tuple[str, int]:
    """"png" | "jpeg" | "jpeg:Q" → (format, quality). PNG is the default: lossless, the server sees the pixels the loop saw.
    JPEG (4:4:4, quality 92 unless given) is a quarter of the bytes — 25 kB against 100 kB for an endoscope view — and the
    upload is what a round trip to a far server spends its time on (10-12: 1.3 s with the image, 0.75 s with a flat one).
    Whether a model answers the same from a JPEG has to be measured for that model (remote/replay_obs.py --image)."""
    fmt, _, q = (spec or "png").lower().partition(":")
    fmt = "jpeg" if fmt == "jpg" else fmt
    if fmt not in IMAGE_FORMATS:
        raise ValueError(f"image format {spec!r}: png or jpeg[:quality]")
    return fmt, int(q) if q else 92


def encode_image(img: np.ndarray, fmt: str = "png", quality: int = 92) -> str:
    """(H, W, 3) uint8 RGB → base64 PNG (lossless) or JPEG (4:4:4)."""
    from PIL import Image
    a = np.asarray(img)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"image must be (H, W, 3) uint8, got {a.dtype} {a.shape}")
    b = io.BytesIO()
    if fmt == "jpeg":
        Image.fromarray(a, "RGB").save(b, "JPEG", quality=int(quality), subsampling=0)
    else:
        Image.fromarray(a, "RGB").save(b, "PNG", compress_level=1)
    return base64.b64encode(b.getvalue()).decode("ascii")


def decode_image(b64: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


def abs_to_chunk(state: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Absolute action rows (T, 15 or 16: the tools' states 1 … T frames ahead [+ done]) → the loop's chunk-relative chunk
    (T, 13 or 14): per tool the position and the rotation vector relative to `state` (the observation the rows were
    predicted from), the trigger, [done]. Row by row this is dataset/actions.py to_delta(state, row) — the same numbers
    train/lerobot-train-rel.py to_chunk_relative() gives a chunk-relative policy as its target."""
    s = np.asarray(state, float)[:15]; r = np.asarray(rows, float)
    if r.ndim != 2 or r.shape[1] not in (15, 16):
        raise RemoteError(f"action must be (T, 15) or (T, 16), got {r.shape}")
    if not np.isfinite(r).all():
        raise RemoteError("action holds NaN or inf")
    qn = np.stack([np.linalg.norm(r[:, 3:7], axis=1), np.linalg.norm(r[:, 10:14], axis=1)])
    if (qn < 0.5).any() or (qn > 1.5).any():   # a regressed quaternion is a little off unit length; this is not a quaternion
        raise RemoteError(f"action quaternions are far from unit length (norms {qn.min():.2f} … {qn.max():.2f})")
    out = np.stack([AC.to_delta(s, row[:15]) for row in r]).astype(float)
    return np.concatenate([out, r[:, 15:16]], 1) if r.shape[1] == 16 else out


def chunk_to_abs(state: np.ndarray, chunk: np.ndarray) -> np.ndarray:
    """The inverse of abs_to_chunk: a chunk-relative chunk (T, 13 or 14) → absolute rows (T, 15 or 16). Used by the
    reference server to answer with a chunk-relative ACT checkpoint, and by the tests."""
    s = np.asarray(state, float)[:15]; c = np.asarray(chunk, float)
    rows = np.stack([AC.from_delta(s, row[:13]) for row in c])
    return np.concatenate([rows, c[:, 13:14]], 1) if c.shape[1] >= 14 else rows


class RemotePolicy:
    """The client. One instance per run; act() blocks for one request (the arms hold their last target meanwhile)."""

    def __init__(self, url: str, token: str | None = None, timeout_s: float = 10.0, retries: int = 1, run: str = "", ca_file: str | None = None,
                 image: str | None = None):
        self.url = url.rstrip("/"); self.token = token; self.timeout_s = float(timeout_s); self.retries = int(retries)
        self.run = run; self.seq = 0; self.info_: dict | None = None; self.log: list[dict] = []
        self._seq_lock = threading.Lock()
        self.image_fmt, self.image_q = image_format(image)
        # one connection a thread, kept between requests: through an SSH tunnel every new connection is a new channel —
        # a round trip to the far end before the request leaves (0.3 s to eu-north1)
        self._local = threading.local()
        if not self.url.startswith(("http://", "https://")):
            raise RemoteError(f"policy url must start with http:// or https://, got {url!r}")
        # https with a certificate no authority signed: the server's own certificate (PEM) is the one thing trusted —
        # given here or as SPINE_POLICY_CA_FILE (the run scripts set it from ~/.config/ube/policy.ca). Never "do not verify".
        ca = ca_file or os.environ.get("SPINE_POLICY_CA_FILE") or None
        self.ctx = None
        if ca and self.url.startswith("https://"):
            import ssl
            if not Path(ca).expanduser().is_file():
                raise RemoteError(f"certificate file {ca} not found")
            self.ctx = ssl.create_default_context(cafile=str(Path(ca).expanduser()))

    def _conn(self, timeout: float):
        """This thread's connection (made on first use, kept), and whether it has been used before."""
        import http.client
        c = getattr(self._local, "conn", None)
        if c is None:
            u = urllib.parse.urlsplit(self.url)
            if u.scheme == "https":
                c = http.client.HTTPSConnection(u.hostname, u.port or 443, timeout=timeout, context=self.ctx)
            else:
                c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
            self._local.conn = c; self._local.base = u.path.rstrip("/")
            return c, False
        c.timeout = timeout
        if c.sock is not None:
            try:
                c.sock.settimeout(timeout)
            except OSError:                       # the socket is gone: a fresh connection
                self._drop()
                return self._conn(timeout)
        return c, c.sock is not None

    def _drop(self) -> None:
        c = getattr(self._local, "conn", None)
        if c is not None:
            try:
                c.close()
            except Exception:
                pass
        self._local.conn = None

    def _call(self, path: str, body: dict | None = None, timeout: float | None = None) -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json", "Connection": "keep-alive"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        data = None if body is None else json.dumps(body).encode()
        last, attempt, again = None, 0, True
        while attempt <= self.retries:
            c, used = self._conn(timeout or self.timeout_s)
            try:
                c.request("GET" if body is None else "POST", self._local.base + path, body=data, headers=h)
                r = c.getresponse(); raw = r.read()
            except Exception as e:                # no answer (refused, reset, timed out)
                self._drop(); last = e
                if used and again and not isinstance(e, TimeoutError):
                    again = False                 # a kept connection the server had closed meanwhile: a fresh one, at once,
                    continue                      # and that is not yet "once more"
                attempt += 1
                if attempt <= self.retries:       # once more, then give up
                    time.sleep(0.5)
                continue
            if r.status >= 400:                   # the server answered: asking again does not change a 401 or a 400
                raise RemoteError(f"{path}: HTTP {r.status} {raw[:300].decode('utf-8', 'replace')}")
            try:
                return json.loads(raw)
            except ValueError as e:
                self._drop()
                raise RemoteError(f"{path}: the answer is not JSON ({e})") from None
        raise RemoteError(f"{path}: no answer from {self.url} after {self.retries + 1} tries ({type(last).__name__}: {last})")

    def info(self) -> dict:
        """The server's description of its model; checked once against what the loop will send and expects back."""
        if self.info_ is None:
            d = self._call("/info")
            if d.get("protocol") != PROTOCOL:
                raise RemoteError(f"server speaks {d.get('protocol')!r}, this loop {PROTOCOL!r}")
            names = list(d.get("action_names") or [])
            if names[:15] != STATE_NAMES or names[15:] not in ([], [DONE_NAME], [PROGRESS_NAME]):
                raise RemoteError(f"server's action_names are not the dataset's 15 state names [+ {DONE_NAME} | {PROGRESS_NAME}]: {names}")
            if list(d.get("state_names") or []) != STATE_NAMES:
                raise RemoteError(f"server's state_names are not the dataset's: {d.get('state_names')}")
            if bool(d.get("done_output")) != (len(names) == 16):
                raise RemoteError("server's done_output flag and its action_names disagree")
            if int(d.get("action_horizon") or 0) < 1:
                raise RemoteError(f"server's action_horizon is {d.get('action_horizon')!r}")
            if self.image_fmt != "png" and self.image_fmt not in ((d.get("image") or {}).get("formats") or ["png"]):
                raise RemoteError(f"the server does not list {self.image_fmt} images (image.formats) — send png")
            self.info_ = d
        return self.info_

    @property
    def end_kind(self) -> str | None:
        """What the model's 16th action value is: "done" (a 0/1 flag), "progress" (removed fraction) or None (no such value:
        the harness's bounds end the drilling)."""
        names = list(self.info().get("action_names") or [])
        return END_KINDS.get(names[15]) if len(names) == 16 else None

    goal: dict | None = None    # {"level", "extent"} for a model whose instructions name the goal (lang1); set by the caller

    def task(self, phase: int, goal: dict | None = None) -> str:
        """The instruction for the harness's phase, in the model's own instruction set (task_for)."""
        return task_for(phase, self.end_kind is not None, self.info().get("instructions"), goal if goal is not None else self.goal)

    def act(self, image: np.ndarray, state: np.ndarray, task: str) -> np.ndarray:
        """One observation → absolute action rows (T, 15 or 16). Raises RemoteError; never returns a partial answer."""
        inf = self.info()
        with self._seq_lock:      # the continuous loop asks again while an answer is still on its way: each request its own number
            self.seq += 1; seq = self.seq
        t0 = time.monotonic()
        d = self._call("/act", {"protocol": PROTOCOL, "seq": seq, "run": self.run,
                                f"image_{self.image_fmt}_b64": encode_image(image, self.image_fmt, self.image_q),
                                "state": [float(x) for x in np.asarray(state, float)[:15]], "task": task})
        dt = time.monotonic() - t0
        if d.get("seq") != seq:
            raise RemoteError(f"answer is for request {d.get('seq')!r}, asked {seq}")
        rows = np.asarray(d.get("action"), float)
        want = 16 if inf["done_output"] else 15
        if rows.ndim != 2 or rows.shape[1] != want or len(rows) < 1:
            raise RemoteError(f"action has shape {rows.shape}, expected (T, {want})")
        with self._seq_lock:
            self.log.append({"seq": seq, "round_trip_s": round(dt, 3), "infer_ms": d.get("infer_ms"), "rows": int(len(rows))})
        return rows

    def chunk(self, image: np.ndarray, state: np.ndarray, task: str) -> np.ndarray:
        """act() in the loop's chunk-relative form: (T, 13) or, with the done output, (T, 14)."""
        return abs_to_chunk(state, self.act(image, state, task))

    def summary(self) -> dict:
        rt = [x["round_trip_s"] for x in self.log]; im = [x["infer_ms"] for x in self.log if x.get("infer_ms") is not None]
        return {"url": self.url, "requests": len(self.log), "info": self.info_,
                "round_trip_s": {"median": round(float(np.median(rt)), 3), "max": round(float(np.max(rt)), 3)} if rt else None,
                "server_infer_ms_median": round(float(np.median(im)), 1) if im else None}


# a real state of a held-out demonstration (dataset episode 18 = session 92c5, frame 600): the connection check's default
CHECK_STATE = [38.7095, -33.011, 1027.2275, 0.6114, -0.626, -0.4173, 0.2452, 24.314, -51.087, 1024.7805, 0.652, 0.6371, 0.3885, 0.1346, 0.0]


def main() -> None:
    """A check of a policy server without the robot: /info, then /act a few times.

        python remote_policy.py URL [--token-file F] [--n 3] [--dataset DS --episode 18 --frame 600]

    Without --dataset the request carries a grey image and one real state (CHECK_STATE): the answer's shape, units and
    timing are checked, its content means little (the model has not seen a grey screen). With --dataset (needs lerobot)
    the image and the state are a frame of that dataset and the answer is compared with what the demonstration did next."""
    import argparse
    ap = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--token-file", default=None)
    ap.add_argument("--dataset", default=None, help="take the image and the state from this dataset frame")
    ap.add_argument("--episode", type=int, default=18)
    ap.add_argument("--frame", type=int, default=600)
    ap.add_argument("--task", default=TASKS[4])
    ap.add_argument("--n", type=int, default=3, help="requests (timing)")
    ap.add_argument("--timeout-s", type=float, default=30.0)
    a = ap.parse_args()
    rp = RemotePolicy(a.url, read_token(a.token_file), a.timeout_s, run="remote_policy.py check")
    inf = rp.info()
    print(json.dumps({"info": inf}, ensure_ascii=False))
    img = np.full((inf["image"]["size"], inf["image"]["size"], 3), 127, np.uint8)
    st = np.asarray(CHECK_STATE, float); st[3:7] /= np.linalg.norm(st[3:7]); st[10:14] /= np.linalg.norm(st[10:14])
    demo = None
    if a.dataset:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        d = LeRobotDataset("local/check", root=str(Path(a.dataset).expanduser()), episodes=[a.episode], video_backend="pyav")
        key = next(k for k in d.meta.features if k.startswith("observation.images"))
        img = (d[a.frame][key].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        states = np.asarray(d.hf_dataset.with_format("numpy")["observation.state"], float)[:, :15]   # action[k] = state[k + 1]
        st = states[a.frame]; demo = states[a.frame + 1:a.frame + 1 + inf["action_horizon"]]
    for _ in range(a.n):
        rows = rp.act(img, st, a.task)
    ch = abs_to_chunk(st, rows); k = min(15, len(ch) - 1)
    out = {"rows": list(rows.shape), "drill_move_mm_16": round(float(np.linalg.norm(ch[k, 6:9])), 3),
           "endo_move_mm_16": round(float(np.linalg.norm(ch[k, 0:3])), 3),
           "drill_turn_deg_16": round(float(np.degrees(np.linalg.norm(ch[k, 9:12]))), 2),
           "trigger_first_16": [round(float(x), 2) for x in rows[:16, 14]],
           (rp.end_kind or "end") + "_first_16": [round(float(x), 2) for x in rows[:16, 15]] if rows.shape[1] == 16 else None,
           "round_trip_s": rp.summary()["round_trip_s"], "server_infer_ms_median": rp.summary()["server_infer_ms_median"]}
    # a tool does not go 20 mm or turn 30° in one second in any demonstration: such an answer is not in the dataset's units
    out["plausible_units"] = bool(out["drill_move_mm_16"] < 20 and out["endo_move_mm_16"] < 20 and out["drill_turn_deg_16"] < 30)
    if demo is not None and len(demo):
        n = min(len(demo), len(rows)); dm = abs_to_chunk(st, demo[:n])
        out["demo_drill_move_mm_16"] = round(float(np.linalg.norm(dm[n - 1, 6:9])), 3)
        out["tip_distance_to_demo_mm"] = {"drill": round(float(np.linalg.norm(rows[n - 1, 7:10] - demo[n - 1, 7:10])), 3),
                                          "endo": round(float(np.linalg.norm(rows[n - 1, 0:3] - demo[n - 1, 0:3])), 3)}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
