"""remote/connect_check.py, remote/endpoint.py, remote/intake_summary.py — the first contact with a policy server somebody
else runs: servers with one defect each (and one without), the check must name the defect and the side it is on."""
import json
import os
import socket
import stat
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "remote"))
import connect_check as CC      # noqa: E402
import end_level as EL          # noqa: E402
import endpoint as EP           # noqa: E402
import intake_summary as IS     # noqa: E402
import policy_server as PS      # noqa: E402
import remote_policy as RP      # noqa: E402

STATE = np.asarray(RP.CHECK_STATE, float); STATE[3:7] /= np.linalg.norm(STATE[3:7]); STATE[10:14] /= np.linalg.norm(STATE[10:14])


def moving(state, T=40, end=True, mm=0.4):
    """Absolute rows: the drill goes `mm` a frame along x from the observed state, trigger on, progress 0.50 → 0.60."""
    rows = np.repeat(np.asarray(state, float)[None, :15], T, 0).copy(); rows[:, 7] += mm * np.arange(1, T + 1); rows[:, 14] = 1.0
    return np.c_[rows, np.linspace(0.50, 0.60, T)] if end else rows


class Good(PS.Backend):
    model = "good"; horizon = 40; image_size = 256; done_output = True; end_name = RP.PROGRESS_NAME
    delay = 0.0; kind = "absolute"

    def predict(self, image, state, task):
        if self.delay:
            time.sleep(self.delay)
        rows = moving(state, self.horizon, self.done_output)
        if self.kind == "relative":            # what a policy returns when nobody undid use_relative_actions
            rows[:, :14] -= np.asarray(state, float)[None, :14]
        elif self.kind == "normalised":        # … or the min–max scaling
            rows[:, :14] = np.tanh(rows[:, :14] / 1000.0)
        elif self.kind == "short":
            rows = rows[:16]
        return rows


@pytest.fixture
def served():
    made = []

    def start(backend, token="s3cret"):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), PS.make_handler(backend, token))
        threading.Thread(target=srv.serve_forever, daemon=True).start(); made.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}"
    yield start
    for s in made:
        s.shutdown(); s.server_close()


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("UBE_CONFIG_DIR", str(tmp_path / "ube"))
    return tmp_path / "ube"


def check(url, token="s3cret", cfg_dir=None, **kw):
    tf = None
    if token is not None:
        tf = Path(cfg_dir or os.environ["UBE_CONFIG_DIR"]) / "t.token"; tf.parent.mkdir(parents=True, exist_ok=True)
        tf.write_text(token); tf.chmod(0o600)
    return CC.run(url, str(tf) if tf else None, dataset=None, log=lambda *_: None, **kw)


def line(rep, what):
    return next(c for c in rep["checks"] if c["check"] == what)


# ---------------------------------------------------------------- one answer
def test_an_absolute_answer_passes_and_is_compared_with_the_demonstration():
    rows = moving(STATE); demo = moving(STATE, mm=1.0)[:, :15]
    j = CC.judge(STATE, rows, 40, "progress", demo, np.linspace(0.52, 0.62, 40))
    assert j["problems"] == [] and j["rows"] == 40
    assert j["row1_from_state"]["drill_mm"] == pytest.approx(0.4) and j["row16_from_state"]["drill_mm"] == pytest.approx(6.4)
    v = j["vs_demo"]
    assert v["demo_drill_move_mm"] == pytest.approx(16.0) and v["model_drill_move_mm"] == pytest.approx(6.4) and v["drill_direction_cos"] == pytest.approx(1.0)
    assert v["drill_tip_mm"] == pytest.approx(9.6) and v["trigger_agree"] == 1.0 and v["progress_error_mean"] == pytest.approx(-0.02, abs=2e-3)


def test_answers_that_are_not_absolute_states_are_named():
    rel = moving(STATE); rel[:, :14] -= STATE[None, :14]
    assert any("not absolute tool states" in p for p in CC.judge(STATE, rel, 40, "progress")["problems"])       # quaternion of norm ~0
    far = moving(STATE); far[:, 7:10] += 300.0                                   # a valid pose in another coordinate frame
    assert any("row 1 is" in p and "not the absolute next state" in p for p in CC.judge(STATE, far, 40, "progress")["problems"])
    near = moving(STATE); near[:, 7:10] += 15.0                                  # absolute, but from another moment
    assert any("not a continuation of this observation" in p for p in CC.judge(STATE, near, 40, "progress")["problems"])
    fast = moving(STATE, mm=3.0)                                                 # 48 mm in a second
    assert any("no demonstration does" in p for p in CC.judge(STATE, fast, 40, "progress")["problems"])
    assert any("shape" in p for p in CC.judge(STATE, moving(STATE)[:, :15], 40, "progress")["problems"])
    assert any("action_horizon 40" in p for p in CC.judge(STATE, moving(STATE, 16), 40, "progress")["problems"])
    bad = moving(STATE); bad[:, 14] = 7.0; bad[:, 15] = -3.0
    p = CC.judge(STATE, bad, 40, "progress")["problems"]
    assert any("trigger" in x for x in p) and any("progress values" in x for x in p)


def test_timing_is_judged_against_the_loops_limit():
    log = [{"round_trip_s": 9.0, "infer_ms": 8000}] + [{"round_trip_s": x, "infer_ms": 300} for x in (0.8, 1.0, 0.9)]
    t = CC.timing(log, 6.0)
    assert t["first_s"] == 9.0 and t["median_s"] == 0.9 and t["slowest_s"] == 1.0 and t["verdict"] == "ok"      # the cold first request is not the verdict
    assert t["outside_the_model_s_median"] == pytest.approx(0.6)
    assert CC.timing(log + [{"round_trip_s": 3.5, "infer_ms": 300}], 6.0)["verdict"] == "close to the limit"
    assert CC.timing(log + [{"round_trip_s": 6.5, "infer_ms": 300}], 6.0)["verdict"] == "too slow"


def test_the_questions_cover_approach_drilling_and_the_end():
    eps = {2: {"progress": np.zeros(900), "first_cut": 200, "last_cut": 700}, 18: {"progress": np.zeros(400), "first_cut": 10, "last_cut": 395}}
    q = CC.pick_frames(eps, [2, 18])
    assert [(e, k) for e, k, _ in q if e == 2] == [(2, 170), (2, 325), (2, 450), (2, 575), (2, 720)]
    assert [k for e, k, _ in q if e == 18] == [0, 106, 202, 298, 398]            # clamped inside the episode
    assert [w for _, _, w in q][:5] == ["approach", "drilling 25 %", "drilling 50 %", "drilling 75 %", "after the last cut"]


# ---------------------------------------------------------------- a server
def test_a_sound_server_is_ready(served, cfg):
    rep, rc = check(served(Good()))
    assert rc == 0 and rep["verdict"] == "ready" and rep["end_output"] == "progress", rep["checks"]
    assert rep["address"]["where"] == "this machine" and line(rep, "auth")["ok"] is True and line(rep, "model")["ok"] is True
    assert line(rep, "answers")["ok"] is True and "40 rows" in line(rep, "answers")["detail"] and rep["timing"]["verdict"] == "ok"
    assert "s3cret" not in json.dumps(rep)                                       # the report can be shown to anybody


def test_a_model_without_the_end_output_is_ready_and_said_so(served, cfg):
    b = Good(); b.done_output = False
    rep, rc = check(served(b))
    assert rc == 0 and rep["end_output"] is None and "no end output" in line(rep, "model")["detail"]


def test_the_token_problems_are_told_apart(served, cfg):
    url = served(Good())
    rep, rc = check(url, token=None)
    assert rc == 1 and "none is configured" in line(rep, "auth")["detail"]
    rep, rc = check(url, token="another")
    assert rc == 1 and "configured token is refused" in line(rep, "auth")["detail"]
    rep, rc = check(served(Good(), token=None), token=None)                      # a server without a token, on this machine
    assert rc == 0 and line(rep, "auth")["ok"] is None and "WITHOUT a token" in line(rep, "auth")["detail"]


def test_relative_normalised_and_short_answers_are_not_ready(served, cfg):
    for kind, word in (("relative", "not absolute tool states"), ("normalised", "not absolute tool states"), ("short", "action_horizon 40")):
        b = Good(); b.kind = kind
        rep, rc = check(served(b))
        assert rc == 1 and rep["verdict"] == "not ready" and word in line(rep, "answers")["detail"], (kind, line(rep, "answers"))


def test_a_model_for_another_image_is_not_ready(served, cfg):
    b = Good(); b.optics = "air"
    rep, rc = check(served(b))
    assert rc == 1 and "a robot run observes" in line(rep, "model")["detail"]


def test_a_slow_server_is_too_slow(served, cfg, monkeypatch):
    monkeypatch.setattr(CC, "LIMIT_S", 0.2)
    b = Good(); b.delay = 0.25
    rep, rc = check(served(b))
    assert rc == 1 and rep["timing"]["verdict"] == "too slow" and line(rep, "timing")["ok"] is False


def test_nothing_there_is_unreachable(cfg):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    rep, rc = check(f"http://127.0.0.1:{port}")
    assert rc == 2 and rep["verdict"] == "unreachable" and "nothing listens" in line(rep, "reach")["detail"]


def test_another_http_api_is_described_not_guessed(cfg):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            body = json.dumps({"paths": {"/act": {}, "/health": {}}}).encode() if self.path == "/openapi.json" else b'{"detail":"Not Found"}'
            self.send_response(200 if self.path == "/openapi.json" else 404); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        rep, rc = check(f"http://127.0.0.1:{srv.server_address[1]}")
    finally:
        srv.shutdown(); srv.server_close()
    assert rc == 3 and rep["verdict"] == "another protocol" and "openapi" in rep["foreign"]["looks_like"] and rep["foreign"]["answers"]["routes"] == ["/act", "/health"]


def test_a_zeromq_port_is_recognised(cfg):
    lst = socket.socket(); lst.bind(("127.0.0.1", 0)); lst.listen(8); port = lst.getsockname()[1]; stop = threading.Event()

    def serve():
        lst.settimeout(0.2)
        while not stop.is_set():
            try:
                c, _ = lst.accept()
            except OSError:
                continue
            try:
                c.sendall(b"\xff\x00\x00\x00\x00\x00\x00\x00\x01\x7f\x03"); time.sleep(0.3)
            except OSError:
                pass
            c.close()
    th = threading.Thread(target=serve, daemon=True); th.start()
    try:
        assert "ZeroMQ" in CC.sniff("127.0.0.1", port, False)
        rep, rc = check(f"http://127.0.0.1:{port}")
    finally:
        stop.set(); th.join(1); lst.close()
    assert rc == 3 and "ZeroMQ" in rep["foreign"]["looks_like"]


# ---------------------------------------------------------------- where the server is
def test_addresses_are_told_apart():
    C = lambda u: (EP.classify(u)["where"], EP.classify(u)["encrypted"])          # noqa: E731
    assert C("http://127.0.0.1:18900") == ("this machine", True) and C("http://localhost:8900") == ("this machine", True)
    assert C("http://100.100.1.2:8900") == ("tailnet", True) and C("http://192.168.0.97:8900") == ("private network", False)
    assert C("http://203.0.113.7:8900") == ("internet", False) and C("https://203.0.113.7:8443") == ("internet", True)
    assert "readable on the way" in EP.classify("http://203.0.113.7:8900")["note"] and EP.classify("https://x.example.org")["port"] == 443
    for bad in ("203.0.113.7:8900", "ftp://x", "http://user:pw@host:1", "http://host:1/?token=abc"):
        with pytest.raises(ValueError):
            EP.check_url(bad)


def test_the_token_file_is_private_from_the_first_byte_and_names_fall_back(cfg):
    assert EP.load()["url"] is None and EP.names() == []
    EP.save_url("http://127.0.0.1:18900/"); p = EP.write_token("  abc123\n")
    assert stat.S_IMODE(p.stat().st_mode) == 0o600 and p.read_text() == "abc123\n" and stat.S_IMODE(cfg.stat().st_mode) == 0o700
    d = EP.load(); assert d["url"] == "http://127.0.0.1:18900" and d["token_file"] == str(p) and d["ca_file"] is None
    EP.save_url("https://203.0.113.7:8443", "groot_h")
    h = EP.load("groot_h"); assert h["token_file"] == str(p)                      # no token of its own: the default one
    t2 = EP.write_token("zzz", "groot_h"); assert EP.load("groot_h")["token_file"] == str(t2)
    assert EP.resolve("groot_h")["url"] == "https://203.0.113.7:8443" and EP.resolve("http://10.0.0.5:1")["token_file"] == str(p)
    assert EP.names() == [None, "groot_h"] and "abc123" not in json.dumps(EP.describe()) and "zzz" not in json.dumps(EP.describe("groot_h"))
    with pytest.raises(ValueError):
        EP.write_token("two words")
    with pytest.raises(ValueError):
        EP.resolve("nothing_here")
    with pytest.raises(ValueError):
        EP.paths("../etc")


# ---------------------------------------------------------------- after the evaluation
def sweep(best, lo, hi, missed_at=None):
    lv = [round(0.5 + 0.005 * i, 3) for i in range(100)]
    U = [1.0 - abs(x - best) for x in lv]
    return {"levels": lv, "U": U, "rows": [{"level": x, "U": u, "missed": (0.5 if missed_at and x >= missed_at else 0.0)} for x, u in zip(lv, U)],
            "best": {"level": best, "U": 1.0}, "indistinguishable": {"lowest": lo, "highest": hi}}


def test_the_level_stays_at_the_computed_one_while_the_model_cannot_tell_them_apart():
    a = IS.level_to_use(sweep(0.94, 0.88, 0.97), 0.91)
    assert a["level"] == 0.91 and "lies inside" in a["why"] and a["best_level"] == 0.94
    assert IS.level_to_use(sweep(0.80, 0.76, 0.84), 0.91)["level"] == 0.84         # the model ends lower: the nearest end of its range
    assert IS.level_to_use(sweep(0.97, 0.95, 0.99), 0.91)["level"] == 0.95
    assert IS.level_to_use(sweep(0.94, 0.88, 0.97, missed_at=0.90), 0.91)["missed"] == 0.5


def test_the_summary_prints_the_robot_commands_and_never_a_token(tmp_path):
    (tmp_path / "check.json").write_text(json.dumps({"verdict": "ready", "end_output": "progress", "info": {"model": "groot", "action_horizon": 40},
                                                     "timing": {"median_s": 0.7, "slowest_s": 1.1, "first_s": 4.0, "limit_s": 6.0}, "checks": [{"check": "address", "ok": None, "detail": "plain HTTP"}]}))
    sw = sweep(0.94, 0.88, 0.97); sw["predictor"] = {"mae": 0.031}
    (tmp_path / "model_eval.json").write_text(json.dumps({"model": "model groot", "predictors": {"model groot": sw}, "open_loop": {
        "drilling": {"trigger_agree_16": 0.93, "drill_H16": {"cos_median": 0.61, "gain_slope": 0.42, "demo_mm_median": 2.1, "pred_mm_median": 1.0}},
        "progress_error_where_true_progress_is_0.9_or_more": {"mean": -0.02, "sd": 0.03, "n": 80}}}))
    lines, out = IS.summarize(tmp_path, "GR00T_gen", "http://127.0.0.1:18900", "/home/x/.config/ube/policy.groot_h.token")
    text = "\n".join(lines)
    assert out["end_level"]["level"] == 0.91 and out["ready_for_robot_runs"] is False and "3 dry run: not done" in text       # no dry run yet
    assert "LIVE_END_LEVEL=0.91 LIVE_POLICY_TOKEN_FILE=/home/x/.config/ube/policy.groot_h.token bash run_remote_base.sh C1006 http://127.0.0.1:18900 GR00T_gen r0" in text
    assert "direction 0.61, gain 0.42" in text and "-- address: plain HTTP" in text and "not started" in text
    (tmp_path / "dry.json").write_text(json.dumps({"drill_end": "policy", "drill_s": 41.0, "removed_mm3": 380.0, "remote_policy": {"requests": 40}}))
    assert IS.summarize(tmp_path, "GR00T_gen", "http://127.0.0.1:18900")[1]["ready_for_robot_runs"] is True
    (tmp_path / "model_eval.json").write_text(json.dumps({"model": "model trial", "predictors": {}, "open_loop": {}}))
    lines, out = IS.summarize(tmp_path, "trial", "http://127.0.0.1:18900")
    assert "no progress output: no end level" in "\n".join(lines) and "LIVE_END_LEVEL" not in "\n".join(lines)


def test_open_loop_scores_a_model_without_the_progress_output():
    n = 200; st = np.repeat(STATE[None], n, 0).copy(); st[:, 7] += 0.5 * np.arange(n); st[:, 14] = 1.0
    eps = {0: {"state": st, "progress": np.linspace(0, 1, n), "first_cut": 10, "last_cut": 180}}
    ans = {0: {t: moving(st[t], 40, end=False, mm=0.25) for t in range(16, 160, 16)}}
    o = EL.open_loop(eps, ans)["drilling"]
    assert "progress_error" not in o and o["trigger_agree_16"] == 1.0
    assert o["drill_H16"]["gain_slope"] == pytest.approx(0.5, abs=0.01) and o["drill_H16"]["cos_median"] == pytest.approx(1.0)
