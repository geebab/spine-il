#!/usr/bin/env python3
"""The first thing to run with a policy server's address: is it what the robot loop needs?

    python inference/remote/connect_check.py [URL | NAME] [--token-file F] [--dataset DS] [--episodes 2,18] [-o report.json]

URL or the name of a configured endpoint (remote/endpoint.py; nothing = the default one). No robot, no renderer: about a
dozen requests. What it settles, in this order — one line of the report each:

  address   what kind of address it is (this machine = an SSH tunnel, the tailnet, a private network, the internet) and
            whether the token and the images travel encrypted
  reach     the name resolves and the port answers (TCP connect time)
  auth      a request without the token and one with a wrong token are refused
  model     /info: the protocol, the 15 state names, 15 or 16 action values and what the 16th is, rows per answer, the
            image the model wants (256 px, the medium of its training renders), the phase texts
  answers   real frames of demonstrations the model was not trained on — one before the first cut, three while drilling,
            one after the last cut, per episode: the answer's shape, ABSOLUTE values in the dataset's units (row 1 lies
            next to the observed state; nothing jumps), trigger and progress in range. Beside these, reported and not
            judged: how far row 16 is from what the demonstration did, the progress error, how much two answers to the
            same question differ (GR00T samples its answer)
  timing    the first request, the median, the slowest — against the loop's limit (6 s a request, one more try, then the
            run is aborted and the tools are taken out)

Exit 0 = ready for the model evaluation and the dry run (live/remote_intake.sh runs all three) · 1 = something to fix, the
report says what and on which side · 2 = not reachable · 3 = something answers, but not this protocol (the report says
what it looks like, so the adapter can be written).
"""
from __future__ import annotations

import argparse
import json
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import endpoint as EP          # noqa: E402
import remote_policy as RP     # noqa: E402

LIMIT_S = 6.0        # live_loop.py --policy-timeout-s: a request slower than this is asked once more, then the run is aborted
N_EXEC = 16          # steps of an answer the loop executes
SIZE, OPTICS = 256, "water"       # what a robot run observes (live_run.sh: --size 256, LIVE_OPTICS=water)
STEP_MM, STEP_DEG = 10.0, 10.0    # row 1 = the state 1/15 s later: farther than this from the observed state is not "absolute"
MOVE_MM, TURN_DEG = 20.0, 30.0    # over 16 rows (1 s): no demonstration moves a tool that far (remote_policy.py's own check)
DS_NAME = "spine_hand24_replay_trim_water_progress"      # the caller's own copy, else the team share (a team account has none of its own)
DS_DEFAULT = next((d for d in (Path.home() / "il-robot/datasets" / DS_NAME, Path("/srv/il-robot/datasets") / DS_NAME) if (d / "meta/info.json").exists()),
                  Path.home() / "il-robot/datasets" / DS_NAME)
CACHE = Path.home() / "il-robot/runs/groot_prep/check_frames"


# ---------------------------------------------------------------- the wire, below the protocol
def reach(host: str, port: int, tries: int = 3, timeout: float = 5.0) -> dict:
    out: dict = {"host": host, "port": port}
    try:
        t0 = time.monotonic(); infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        out["resolve_ms"] = round((time.monotonic() - t0) * 1000, 1); out["addresses"] = sorted({i[4][0] for i in infos})
    except OSError as e:
        out["error"] = f"the name {host!r} does not resolve ({e})"; return out
    ms = []
    for _ in range(tries):
        t0 = time.monotonic()
        try:
            with socket.create_connection((host, port), timeout=timeout):
                ms.append((time.monotonic() - t0) * 1000)
        except ConnectionRefusedError:
            out["error"] = "connection refused — nothing listens on that port (the server is not started, or the tunnel is not up)"; break
        except (socket.timeout, TimeoutError):
            out["error"] = f"no answer in {timeout:g} s — a firewall drops the connection, or the machine is off"; break
        except OSError as e:
            out["error"] = f"{type(e).__name__}: {e}"; break
    if ms:
        out["connect_ms"] = {"median": round(float(np.median(ms)), 1), "max": round(float(np.max(ms)), 1)}
    return out


def http(url: str, path: str, token: str | None = None, timeout: float = 10.0, ctx=None) -> dict:
    """One GET that never raises: {"status", "text", "json"} or {"error"}."""
    h = {"Accept": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url + path, headers=h), timeout=timeout, context=ctx) as r:
            status, raw = r.status, r.read(4000)
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read(4000)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    text = raw.decode("utf-8", "replace")
    try:
        js = json.loads(text)
    except ValueError:
        js = None
    return {"status": status, "text": text[:300], "json": js}


def sniff(host: str, port: int, tls: bool) -> str | None:
    """What a port that does not speak HTTP/1.1 looks like: ZeroMQ greets first, an HTTP/2 (gRPC) server answers the preface."""
    try:
        if tls:
            c = ssl.create_default_context(); c.check_hostname = False; c.verify_mode = ssl.CERT_NONE   # only to read the ALPN choice
            c.set_alpn_protocols(["h2", "http/1.1"])
            with socket.create_connection((host, port), timeout=4) as s, c.wrap_socket(s, server_hostname=host) as t:
                return {"h2": "HTTP/2 over TLS (a gRPC server answers like this)", "http/1.1": "HTTP/1.1 over TLS"}.get(t.selected_alpn_protocol() or "", "TLS without ALPN")
        with socket.create_connection((host, port), timeout=4) as s:
            s.settimeout(1.5)
            try:
                first = s.recv(16)
            except (socket.timeout, TimeoutError):
                first = b""
            if len(first) >= 10 and first[0] == 0xFF and first[9] == 0x7F:
                return "ZeroMQ (ZMTP greeting) — e.g. Isaac-GR00T's PolicyServer"
            s.sendall(b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n" + b"\x00\x00\x00\x04\x00\x00\x00\x00\x00")
            try:
                ans = s.recv(16)
            except (socket.timeout, TimeoutError):
                ans = b""
            if len(ans) >= 4 and ans[3] == 0x04:
                return "HTTP/2 without TLS (a gRPC server answers like this — e.g. LeRobot's async policy_server)"
    except OSError as e:
        return f"could not look ({type(e).__name__}: {e})"
    return None


def foreign(url: str, token: str | None, ctx, addr: dict) -> dict:
    """Something answers, but /info is not ours: what is there? (for writing the adapter, not a verdict)"""
    seen = {}
    for path in ("/health", "/", "/openapi.json", "/docs", "/v1/models", "/act"):
        r = http(url, path, token, 8.0, ctx)
        seen[path] = r.get("error") or f"HTTP {r['status']} {r['text'][:120]!r}"
        if path == "/openapi.json" and isinstance(r.get("json"), dict) and "paths" in r["json"]:
            seen["routes"] = sorted(r["json"]["paths"])[:40]
    guess = None
    if "routes" in seen:
        guess = "an HTTP API with a published description (/openapi.json — FastAPI): its routes are listed"
    elif isinstance(http(url, "/v1/models", token, 8.0, ctx).get("json"), dict):
        guess = "an OpenAI-style API (/v1/models answers) — a language/vision model endpoint, not an action server"
    else:
        guess = sniff(addr["host"], addr["port"], addr["scheme"] == "https") or "an HTTP server that knows none of /info, /openapi.json, /v1/models"
    return {"looks_like": guess, "answers": seen}


# ---------------------------------------------------------------- real questions
def pick_frames(eps: dict, episodes: list[int]) -> list[tuple[int, int, str]]:
    """Per episode: one frame before the first cut, three while drilling, one after the last cut."""
    out = []
    for e in episodes:
        d = eps[e]; n = len(d["progress"]); fc, lc = int(d["first_cut"]), int(d["last_cut"])
        cand = [("approach", max(0, fc - 30)), ("drilling 25 %", fc + (lc - fc) // 4), ("drilling 50 %", fc + (lc - fc) // 2),
                ("drilling 75 %", fc + 3 * (lc - fc) // 4), ("after the last cut", min(n - 2, lc + 20))]
        seen = set()
        for label, k in cand:
            k = int(min(max(0, k), n - 2))
            if k not in seen:
                seen.add(k); out.append((e, k, label))
    return out


def load_frames(ds: Path, wanted: list[tuple[int, int]], cache: Path = CACHE) -> dict[tuple[int, int], np.ndarray]:
    """The dataset's images of the wanted (episode, frame) pairs; decoded once, kept as PNG for the next check."""
    from PIL import Image
    import end_level as EL
    cache.mkdir(parents=True, exist_ok=True)
    tag = ds.name; out = {}
    for e, k in wanted:
        f = cache / f"{tag}_e{e:02d}_k{k:05d}.png"
        if f.is_file():
            out[(e, k)] = np.asarray(Image.open(f).convert("RGB"))
    miss = [w for w in wanted if w not in out]
    if miss:
        need = set(miss)
        for e, k, img in EL.frames(ds, {e for e, _ in miss}):
            if (e, k) in need:
                out[(e, k)] = img; Image.fromarray(img, "RGB").save(cache / f"{tag}_e{e:02d}_k{k:05d}.png")
                need.discard((e, k))
                if not need:
                    break
    return out


def judge(state: np.ndarray, rows, horizon: int, end_kind: str | None, demo: np.ndarray | None = None, demo_progress: np.ndarray | None = None) -> dict:
    """One answer against what the loop needs (problems) and against the demonstration (measures, not judged)."""
    out: dict = {"problems": []}; P = out["problems"]
    r = np.asarray(rows, float)
    want = 16 if end_kind else 15
    if r.ndim != 2 or r.shape[1] != want:
        P.append(f"the answer has shape {list(r.shape)}, the model card promises (T, {want})"); return out
    out["rows"] = int(len(r))
    if len(r) != horizon:
        P.append(f"{len(r)} rows, the model card says action_horizon {horizon}")
    try:
        ch = RP.abs_to_chunk(state, r)
    except RP.RemoteError as e:
        P.append(f"not absolute tool states in the dataset's layout: {e}"); return out
    k = min(N_EXEC, len(ch)) - 1
    step = {"endo_mm": float(np.linalg.norm(ch[0, 0:3])), "drill_mm": float(np.linalg.norm(ch[0, 6:9])),
            "endo_deg": float(np.degrees(np.linalg.norm(ch[0, 3:6]))), "drill_deg": float(np.degrees(np.linalg.norm(ch[0, 9:12])))}
    out["row1_from_state"] = {a: round(b, 3) for a, b in step.items()}
    mm1, deg1 = max(step["endo_mm"], step["drill_mm"]), max(step["endo_deg"], step["drill_deg"])
    if mm1 > STEP_MM or deg1 > STEP_DEG:
        P.append(f"row 1 is {mm1:.1f} mm / {deg1:.1f}° from the observed state — "
                 + ("the answer is not the absolute next state (relative or normalised values? another coordinate frame?)" if mm1 > 100 or deg1 > 90 else
                    "a jump no demonstration makes in one frame (1/15 s): absolute values, but not a continuation of this observation"))
    mv = {"endo_mm": float(np.linalg.norm(ch[k, 0:3])), "drill_mm": float(np.linalg.norm(ch[k, 6:9])),
          "endo_deg": float(np.degrees(np.linalg.norm(ch[k, 3:6]))), "drill_deg": float(np.degrees(np.linalg.norm(ch[k, 9:12])))}
    out["row16_from_state"] = {a: round(b, 3) for a, b in mv.items()}
    if max(mv["endo_mm"], mv["drill_mm"]) > MOVE_MM or max(mv["endo_deg"], mv["drill_deg"]) > TURN_DEG:
        P.append(f"over 16 rows a tool goes {max(mv['endo_mm'], mv['drill_mm']):.1f} mm / turns {max(mv['endo_deg'], mv['drill_deg']):.1f}° — no demonstration does (units?)")
    jump = float(np.max(np.linalg.norm(np.diff(np.concatenate([np.asarray(state, float)[None, [7, 8, 9]], r[:, 7:10]]), axis=0), axis=1)))
    out["largest_drill_step_mm"] = round(jump, 3)
    trig = r[:, 14]
    out["trigger_first_16"] = [round(float(x), 2) for x in trig[:N_EXEC]]
    if trig.min() < -0.25 or trig.max() > 1.25:
        P.append(f"trigger values {trig.min():.2f} … {trig.max():.2f} are not a 0…1 switch")
    if end_kind:
        ev = r[:, 15]; out[f"{end_kind}_first_16"] = [round(float(x), 3) for x in ev[:N_EXEC]]
        if ev.min() < -0.25 or ev.max() > 1.25:
            P.append(f"{end_kind} values {ev.min():.2f} … {ev.max():.2f} are not in 0…1")
    if demo is not None and len(demo):
        n = min(len(demo), len(r), N_EXEC)
        dm = RP.abs_to_chunk(state, demo[:n])
        d_move, m_move = dm[n - 1, 6:9], ch[n - 1, 6:9]
        nd = float(np.linalg.norm(d_move))
        out["vs_demo"] = {"steps": n, "drill_tip_mm": round(float(np.linalg.norm(r[n - 1, 7:10] - demo[n - 1, 7:10])), 3),
                          "endo_tip_mm": round(float(np.linalg.norm(r[n - 1, 0:3] - demo[n - 1, 0:3])), 3),
                          "demo_drill_move_mm": round(nd, 3), "model_drill_move_mm": round(float(np.linalg.norm(m_move)), 3),
                          "drill_direction_cos": round(float(d_move @ m_move / max(nd * float(np.linalg.norm(m_move)), 1e-9)), 3) if nd > 1.0 else None,
                          "trigger_agree": round(float(((trig[:n] > 0.5) == (demo[:n, 14] > 0.5)).mean()), 3)}
        if end_kind == "progress" and demo_progress is not None and len(demo_progress) >= n:
            out["vs_demo"]["progress_error_mean"] = round(float((r[:n, 15] - demo_progress[:n]).mean()), 4)
            out["vs_demo"]["true_progress"] = round(float(demo_progress[n - 1]), 3)
    return out


def timing(log: list[dict], limit: float = LIMIT_S) -> dict:
    rt = [x["round_trip_s"] for x in log]; im = [x["infer_ms"] for x in log if x.get("infer_ms") is not None]
    if not rt:
        return {}
    warm = rt[1:] or rt
    out = {"requests": len(rt), "first_s": round(rt[0], 3), "median_s": round(float(np.median(warm)), 3), "slowest_s": round(float(np.max(warm)), 3),
           "server_infer_ms_median": round(float(np.median(im)), 1) if im else None, "limit_s": limit}
    if im:
        out["outside_the_model_s_median"] = round(float(np.median(warm)) - float(np.median(im)) / 1000.0, 3)   # network + encoding
    out["verdict"] = "too slow" if out["slowest_s"] >= limit else "close to the limit" if out["slowest_s"] >= limit / 2 else "ok"
    return out


# ---------------------------------------------------------------- the check
def run(url_or_name: str | None, token_file: str | None = None, dataset: Path | None = DS_DEFAULT, episodes=(2, 18), name: str | None = None,
        repeats: int = 2, timeout_s: float = 30.0, log=print) -> tuple[dict, int]:
    rep: dict = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": []}

    def note(what: str, ok, text: str):
        rep["checks"].append({"check": what, "ok": ok, "detail": text})
        log(f"[{'ok' if ok is True else '!!' if ok is False else '--'}] {what}: {text}")

    ep = EP.resolve(url_or_name, name)
    url = ep["url"]; tfile = token_file or ep["token_file"]; addr = EP.classify(url)
    rep.update(endpoint=ep["name"], url=url, address=addr, token_file=tfile, ca_file=ep["ca_file"])
    note("address", True if addr["encrypted"] else None, f"{url} — {addr['note']}")
    try:
        token = RP.read_token(tfile)
    except RP.RemoteError as e:
        note("token", False, str(e)); rep["verdict"] = "not ready"; return rep, 1
    ctx = None
    if ep["ca_file"] and addr["scheme"] == "https":
        ctx = ssl.create_default_context(cafile=ep["ca_file"])

    r = reach(addr["host"], addr["port"]); rep["reach"] = r
    if "error" in r:
        note("reach", False, r["error"]); rep["verdict"] = "unreachable"; return rep, 2
    note("reach", True, f"{', '.join(r['addresses'][:3])} port {addr['port']}: TCP connect {r['connect_ms']['median']:g} ms")

    # auth: what the server does without the token and with a wrong one
    bare = http(url, "/info", None, 10.0, ctx)
    if "error" in bare:
        fo = foreign(url, token, ctx, addr); rep["foreign"] = fo
        note("model", False, f"the port answers TCP but not HTTP ({bare['error']}) — it looks like: {fo['looks_like']}"); rep["verdict"] = "another protocol"; return rep, 3
    wrong = http(url, "/info", "not-the-token", 10.0, ctx)
    if bare["status"] in (401, 403):
        if token is None:
            note("auth", False, "the server wants a token and none is configured — python inference/remote/endpoint.py token"); rep["verdict"] = "not ready"; return rep, 1
        if http(url, "/info", token, 10.0, ctx).get("status") in (401, 403):
            note("auth", False, "the configured token is refused — is it this server's token? (python inference/remote/endpoint.py token)"); rep["verdict"] = "not ready"; return rep, 1
        if wrong.get("status") in (401, 403):
            note("auth", True, f"refused without the token (HTTP {bare['status']}) and with a wrong one (HTTP {wrong.get('status')}), accepted with the configured one")
        else:
            note("auth", False, f"a request with a WRONG token is answered (HTTP {wrong.get('status')}) — the server does not check the token's value")
    elif bare["status"] == 200:
        note("auth", None if addr["where"] == "this machine" else False,
             "the server answers WITHOUT a token" + (" (a tunnel end on this machine: only this machine's users reach it)" if addr["where"] == "this machine"
                                                     else " — anyone who finds the address can use the model and the GPU: start it with --token-file"))

    rp = RP.RemotePolicy(url, token, timeout_s, retries=0, run="connect_check", ca_file=ep["ca_file"])
    try:
        info = rp.info()
    except RP.RemoteError as e:
        msg = str(e)
        if "HTTP 401" in msg or "HTTP 403" in msg:
            note("auth", False, "the configured token is refused — is it this server's token? (python inference/remote/endpoint.py token)"); rep["verdict"] = "not ready"; return rep, 1
        if any(w in msg for w in ("action_names", "state_names", "done_output", "action_horizon", "server speaks 'spine-policy")):
            note("model", False, f"a spine-policy server, but its model card is not what the loop expects: {msg}"); rep["verdict"] = "not ready"; return rep, 1
        fo = foreign(url, token, ctx, addr); rep["foreign"] = fo
        note("model", False, f"{msg} — this is not a spine-policy/1 server; it looks like: {fo['looks_like']}"); rep["verdict"] = "another protocol"; return rep, 3
    rep["info"] = info
    end_kind = rp.end_kind; horizon = int(info["action_horizon"]); img_info = info.get("image") or {}
    card = [f"model {info.get('model')!r}", f"{horizon} rows an answer", f"{len(info['action_names'])} action values" + (f" (16th = {info['action_names'][15]})" if end_kind else " (no end output: the harness's bounds end the drilling — such a run is not \"ended by the policy\")"),
            f"image {img_info.get('size')} px, {img_info.get('optics')}"]
    bad = []
    if img_info.get("size") != SIZE or img_info.get("optics") != OPTICS or img_info.get("view", "endo") != "endo":
        bad.append(f"the model wants {img_info}, a robot run observes the endoscope view, {SIZE} px, {OPTICS}")
    if int(info.get("fps") or 15) != 15:
        bad.append(f"fps {info.get('fps')} — the loop's frames are 1/15 s apart")
    instr = info.get("instructions") or {"name": "phase1"}
    if instr.get("name") not in RP.INS.SETS:
        bad.append(f"instruction set {instr.get('name')!r} — the loop knows {', '.join(RP.INS.SETS)}")
    else:
        mine = set(RP.TASKS.values()) if instr["name"] == "phase1" else set(RP.INS.texts(RP.INS.table(instr["name"])))
        unknown = [t for t in (info.get("tasks") or []) if t not in mine]
        if unknown:
            bad.append(f"task texts the loop never sends: {unknown[:2]}")
        card.append(f"instructions {instr['name']}" + ("" if instr["name"] == "phase1" else " (the goal is told in words: --goal LEVEL:EXTENT)"))
    if horizon < N_EXEC:
        card.append(f"the loop will execute {horizon} steps an answer (not {N_EXEC})")
    note("model", not bad, "; ".join(card) + ("" if not bad else " — " + "; ".join(bad)))

    # real questions
    eps = frames = None; qs: list = []
    if dataset is not None and Path(dataset).is_dir():
        import end_level as EL
        eps = EL.load(Path(dataset)); episodes = [e for e in episodes if e in eps]
        qs = pick_frames(eps, list(episodes))
        try:
            frames = load_frames(Path(dataset), [(e, k) for e, k, _ in qs])
        except Exception as e:
            note("frames", None, f"the dataset's images could not be read ({type(e).__name__}: {e}) — a grey image is sent instead"); frames = None
    answers, problems, first_err = [], [], None
    if frames:
        rep["dataset"] = str(dataset)
        for e, k, label in qs:
            st = eps[e]["state"][k]; done_out = end_kind is not None
            task = rp.task(3 if k < eps[e]["first_cut"] else 4 if k <= eps[e]["last_cut"] else 5, eps[e].get("goal"))
            try:
                rows = rp.act(frames[(e, k)], st, task)
            except RP.RemoteError as ex:
                first_err = first_err or str(ex); answers.append({"episode": e, "frame": k, "where": label, "problems": [str(ex)]}); problems.append(f"episode {e} frame {k}: {ex}"); continue
            j = judge(st, rows, horizon, end_kind, eps[e]["state"][k + 1:k + 1 + len(rows)], eps[e]["progress"][k + 1:k + 1 + len(rows)])
            j.update(episode=e, frame=k, where=label, true_progress_now=round(float(eps[e]["progress"][k]), 3)); answers.append(j)
            problems += [f"episode {e} frame {k} ({label}): {p}" for p in j["problems"]]
        # the same question again: does the model answer the same?
        mid = next(((e, k) for e, k, label in qs if label == "drilling 50 %"), None)
        if mid and repeats > 0 and not first_err:
            e, k = mid; st = eps[e]["state"][k]; got = []
            for _ in range(repeats + 1):
                try:
                    got.append(np.asarray(rp.act(frames[(e, k)], st, rp.task(4, eps[e].get("goal"))), float))
                except RP.RemoteError as ex:
                    problems.append(f"repeat of episode {e} frame {k}: {ex}"); break
            if len(got) > 1:
                kk = min(N_EXEC, min(len(g) for g in got)) - 1
                tips = np.stack([g[kk, 7:10] for g in got])
                rep["same_question_again"] = {"answers": len(got), "drill_tip_spread_mm_at_row_16": round(float(np.max(np.linalg.norm(tips - tips.mean(0), axis=1))), 3),
                                              **({"progress_spread_at_row_16": round(float(np.ptp([g[kk, 15] for g in got])), 4)} if end_kind else {})}
    else:
        img = np.full((SIZE, SIZE, 3), 127, np.uint8); st = np.asarray(RP.CHECK_STATE, float)
        st[3:7] /= np.linalg.norm(st[3:7]); st[10:14] /= np.linalg.norm(st[10:14])
        for _ in range(3):
            try:
                rows = rp.act(img, st, rp.task(4))
            except RP.RemoteError as ex:
                problems.append(str(ex)); break
            j = judge(st, rows, horizon, end_kind); j.update(where="a grey image and one real state (no dataset here)"); answers.append(j)
            problems += j["problems"]
    rep["answers"] = answers
    good = [a for a in answers if not a["problems"]]
    if problems:
        note("answers", False, f"{len(answers) - len(good)} of {len(answers)} answers are not usable — " + " | ".join(sorted(set(problems))[:3]))
    else:
        vs = [a["vs_demo"] for a in good if "vs_demo" in a]
        extra = ""
        if vs:
            cs = [v["drill_direction_cos"] for v in vs if v.get("drill_direction_cos") is not None]
            extra = (f"; beside the demonstrations (not a verdict — {len(vs)} frames): drill tip {np.median([v['drill_tip_mm'] for v in vs]):.1f} mm from the demonstration's after 16 steps (median), "
                     f"model moves {np.median([v['model_drill_move_mm'] for v in vs]):.1f} mm where the demonstration moves {np.median([v['demo_drill_move_mm'] for v in vs]):.1f}"
                     + (f", direction {np.median(cs):.2f}" if cs else "") + f", trigger agreement {np.mean([v['trigger_agree'] for v in vs]):.2f}"
                     + (f", progress error {np.mean([v['progress_error_mean'] for v in vs if 'progress_error_mean' in v]):+.3f}" if any("progress_error_mean" in v for v in vs) else ""))
        note("answers", True, f"{len(good)} answers: {answers[0].get('rows')} rows of absolute tool states in the dataset's units, trigger" + (f" and {end_kind}" if end_kind else "") + " in range" + extra)
    if "same_question_again" in rep:
        s = rep["same_question_again"]
        note("same question again", None, f"{s['answers']} answers to one frame differ by up to {s['drill_tip_spread_mm_at_row_16']:g} mm at row 16"
             + (f", progress by {s['progress_spread_at_row_16']:g}" if "progress_spread_at_row_16" in s else "") + (" — the model samples its answer: one run is one draw" if s["drill_tip_spread_mm_at_row_16"] > 0.05 else " — the same answer every time"))
    tm = timing(rp.log, LIMIT_S); rep["timing"] = tm
    if tm:
        txt = (f"first request {tm['first_s']:g} s, then median {tm['median_s']:g} s, slowest {tm['slowest_s']:g} s of {tm['requests']}"
               + (f" (the model itself {tm['server_infer_ms_median']:g} ms; network and encoding {tm['outside_the_model_s_median']:g} s)" if tm.get("server_infer_ms_median") is not None else "")
               + f"; the loop's limit is {LIMIT_S:g} s")
        if tm["first_s"] >= LIMIT_S and tm["verdict"] == "ok":
            txt += " — the FIRST request is over the limit: the run scripts ask once before a run to warm the server up"
        note("timing", {"ok": True, "close to the limit": None, "too slow": False}[tm["verdict"]], txt)
    failed = [c for c in rep["checks"] if c["ok"] is False]
    rep["verdict"] = "ready" if not failed and answers else "not ready"
    rep["end_output"] = end_kind
    return rep, 0 if rep["verdict"] == "ready" else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", default=None, help="http(s)://host:port, or the name of a configured endpoint (default: the default endpoint)")
    ap.add_argument("--name", default=None, help="with a URL: the endpoint whose token/certificate files to use")
    ap.add_argument("--token-file", default=None)
    ap.add_argument("--dataset", default=str(DS_DEFAULT), help="demonstrations for the real questions (\"\" = a grey image)")
    ap.add_argument("--episodes", default="2,18", help="episodes the model was not trained on")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--timeout-s", type=float, default=30.0)
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args(argv)
    try:
        rep, rc = run(a.target, a.token_file, Path(a.dataset).expanduser() if a.dataset else None, [int(x) for x in a.episodes.split(",") if x], a.name, a.repeats, a.timeout_s)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr); return 2
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(f"verdict: {rep['verdict'].upper()}" + {0: " — next: bash inference/live/remote_intake.sh (model evaluation, then one dry run)", 1: " — fix the [!!] lines first",
                                                  2: " — the server is not reachable from here", 3: " — another API: the report says what it looks like"}[rc])
    return rc


if __name__ == "__main__":
    sys.exit(main())
