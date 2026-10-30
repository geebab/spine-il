"""render_gate.py — render work gives way to the robot. The watcher's decisions on made-up process tables (what counts as
robot use, what is stopped, what goes on and when), and the real thing on real child processes: stopped, continued, ended."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import render_gate as G     # noqa: E402

PY = "/usr/bin/python3"


@pytest.fixture
def gate(tmp_path, monkeypatch):
    for name, p in (("RUNTIME", tmp_path), ("PAUSE", tmp_path / "pause"), ("STATE", tmp_path / "gate.json"), ("HOLDS", tmp_path / "holds"),
                    ("WORK", tmp_path / "work"), ("LOG", tmp_path / "gate.log")):
        monkeypatch.setattr(G, name, p)
    (tmp_path / "holds").mkdir(); (tmp_path / "work").mkdir()
    return tmp_path


def proc(argv, ppid=1, state="S", pgid=None, start=100):
    return {"argv": argv, "ppid": ppid, "state": state, "pgid": pgid or ppid, "start": start}


VIDEO = {500: proc([PY, "-u", "inference/render_video.py", "s.jsonl", "-o", "x.mp4"], pgid=500, start=7),
         501: proc(["ffmpeg", "-i", "x"], ppid=500, pgid=500, start=8),
         600: proc(["bash", "-c", "cd w && python policy_eval.py rec"], pgid=600, start=9),        # the shell only mentions it
         601: proc(["/home/x/.venv/bin/python", "-u", "policy_eval.py", "rec"], ppid=600, pgid=600, start=10),
         700: proc(["bash", "ro_queue2.sh"], pgid=700, start=11)}


class Signals:
    def __init__(self):
        self.sent = []

    def kill(self, pid, sig):
        self.sent.append((pid, sig))

    def killpg(self, pgid, sig):
        self.sent.append((-pgid, sig))


def test_only_python_running_the_file_is_that_script():
    assert G.script_of([PY, "-u", "/a/b/endo_follow.py", "--arm", "left"]) == "endo_follow.py"
    assert G.script_of(["/x/.venv/bin/python3.12", "-X", "dev", "-u", "robot_origin_rec.py"]) == "robot_origin_rec.py"
    assert G.script_of([PY, "-m", "spine_render.cli", "serve"]) == "spine_render.cli"
    assert G.script_of([PY, "/x/.venv/bin/lerobot-train", "--policy.type=act"]) == "lerobot-train"
    assert G.script_of(["bash", "-c", "python endo_follow.py --arm left"]) is None
    assert G.script_of(["grep", "-n", "endo_follow.py"]) is None and G.script_of(["tail", "-f", "endo_follow.py.log"]) is None
    assert G.script_of([PY, "-c", "import endo_follow"]) is None and G.script_of([PY]) is None and G.script_of([]) is None


def test_robot_processes_are_signs_and_a_dry_run_is_not():
    t = {1: proc([PY, "-u", "endo_follow.py", "--arm", "right"]),
         2: proc([PY, "-u", "robot_origin_rec.py", "--recording", "r"]),
         3: proc([PY, "-u", "inference/live_loop.py", "--live-run", "RO_20261009_101500", "--speed", "0.33"]),
         4: proc([PY, "-u", "inference/live_loop.py", "--live-run", "FAKE_101500", "--cycle"]),         # dry_step.sh
         5: proc([PY, "-u", "inference/live_loop.py", "--prepare-out", "seed.json"]),                    # planning only
         6: proc(["bash", "-c", "pgrep -f endo_follow.py; tail robot_origin_rec.py"]),                   # a shell that names them
         7: proc([PY, "-u", "inference/fake_arms.py", "--run", "FAKE_101500"])}
    s = G.process_signs(t)
    assert len(s) == 3 and any("follower" in x and "pid 1" in x for x in s) and any("recorder" in x for x in s) and any("pid 3" in x for x in s)
    assert G.process_signs({k: t[k] for k in (4, 5, 6, 7)}) == []


def test_a_hold_counts_while_its_owner_lives(gate):
    me = os.getpid()
    G.write_json(G.HOLDS / "roq.json", {"pid": me, "start": G.start_of(me), "reason": "robot job pr_x"})
    G.write_json(G.HOLDS / "old.json", {"pid": me, "start": 1, "reason": "a pid that came round again"})
    G.write_json(G.HOLDS / "dead.json", {"pid": 2 ** 22 + 7, "start": 5, "reason": "gone"})
    assert G.hold_signs() == ["hold roq: robot job pr_x"]
    assert sorted(f.name for f in G.HOLDS.iterdir()) == ["roq.json"]          # the holds of dead owners are cleared away


def test_render_work_is_the_scripts_and_their_children(gate):
    assert G.render_pids(VIDEO) == {500, 501, 601}       # not the shell that mentions policy_eval.py, not the queue


def test_the_watcher_pauses_stops_lingers_and_resumes(gate):
    sig, arms = Signals(), []
    w = G.Watch(linger_s=10, arms=lambda: list(arms), sig=sig.kill, sigpg=sig.killpg)
    t = {k: dict(v) for k, v in VIDEO.items()}
    w.cycle(now=1000.0, table=t)
    assert G.pause_reason() is None and sig.sent == [] and G.read_json(G.STATE)["paused_since"] is None
    arms.append("the right arm is leased (member1)")
    w.cycle(now=1001.0, table=t)
    assert G.pause_reason() == "the right arm is leased (member1)"
    assert sorted(sig.sent) == [(500, signal.SIGSTOP), (501, signal.SIGSTOP), (601, signal.SIGSTOP)]
    for p in (500, 501, 601):
        t[p]["state"] = "T"
    t[602] = proc(["ffmpeg"], ppid=601, pgid=600, start=12)                    # a child that appeared just before the stop
    w.cycle(now=1002.0, table=t)
    assert sig.sent[3:] == [(602, signal.SIGSTOP)]                             # … is caught on the next look; nothing is stopped twice
    t[602]["state"] = "T"
    arms.clear()
    w.cycle(now=1003.0, table=t)
    w.cycle(now=1012.9, table=t)
    assert G.pause_reason() is not None and len(sig.sent) == 4                 # the sign is gone, the pause lingers
    arms.append("the left arm is MOVING")
    w.cycle(now=1013.0, table=t)                                               # the arm only blinked: the pause goes on
    arms.clear()
    w.cycle(now=1014.0, table=t)
    w.cycle(now=1023.5, table=t)
    assert G.pause_reason() is not None
    w.cycle(now=1024.0, table=t)
    assert G.pause_reason() is None and not G.PAUSE.exists()
    assert sorted(sig.sent[4:]) == [(p, signal.SIGCONT) for p in (500, 501, 601, 602)]
    assert G.read_json(G.STATE)["paused_since"] is None and "resume after 23 s (4 processes continued)" in G.LOG.read_text()


def test_what_somebody_else_stopped_stays_stopped_and_a_reused_pid_is_left_alone(gate):
    sig, arms = Signals(), ["the left arm is HOLD"]
    w = G.Watch(linger_s=0, arms=lambda: list(arms), sig=sig.kill, sigpg=sig.killpg)
    t = {k: dict(v) for k, v in VIDEO.items()}
    t[601]["state"] = "T"                                                      # stopped by hand before the pause
    w.cycle(now=10.0, table=t)
    assert sorted(sig.sent) == [(500, signal.SIGSTOP), (501, signal.SIGSTOP)]
    t[501] = proc([PY, "something_else.py"], start=999)                        # ffmpeg ended, its pid was given to another process
    arms.clear()
    w.cycle(now=11.0, table=t)
    assert sig.sent[2:] == [(500, signal.SIGCONT)]


def test_a_new_watcher_takes_over_what_a_dead_one_stopped(gate):
    G.write_json(G.STATE, {"pid": 2 ** 22 + 9, "t": 1.0, "paused_since": 5.0, "stopped": [[500, 7], [501, 8]]})
    sig = Signals()
    w = G.Watch(linger_s=0, arms=lambda: [], sig=sig.kill, sigpg=sig.killpg)
    w.cycle(now=20.0, table={k: dict(v, state="T") for k, v in VIDEO.items() if k in (500, 501)})
    assert sorted(sig.sent) == [(500, signal.SIGCONT), (501, signal.SIGCONT)]


def test_work_that_cannot_wait_is_ended_not_stopped(gate):
    sig, arms = Signals(), ["hold roq: robot job"]
    t = {800: proc(["bash", "dry_step.sh"], pgid=800, start=20),
         801: proc([PY, "-u", "inference/live_loop.py", "--live-run", "FAKE_1"], ppid=800, pgid=800, start=21),
         802: proc([PY, "-u", "inference/rollout.py"], ppid=800, pgid=800, start=22)}     # even a render script inside it
    G.write_json(G.WORK / "800.json", {"pid": 800, "pgid": 800, "start": 20, "on_pause": "kill", "name": "dry"})
    w = G.Watch(linger_s=0, arms=lambda: list(arms), sig=sig.kill, sigpg=sig.killpg)
    w.cycle(now=100.0, table=t)
    assert sig.sent == [(-800, signal.SIGTERM)] and (G.WORK / "800.killed").read_text() == "hold roq: robot job"
    w.cycle(now=103.0, table=t)
    assert len(sig.sent) == 1
    w.cycle(now=106.0, table=t)
    assert sig.sent[1:] == [(-800, signal.SIGKILL)]                            # it did not go by itself


def test_settled_needs_the_watcher_the_pause_and_everything_at_rest(gate):
    me = os.getpid()
    assert G.settled(lambda u: None) == (False, "the watcher is not running")
    G.write_json(G.STATE, {"pid": me, "start": G.start_of(me), "t": time.time(), "paused_since": None})
    assert G.settled(lambda u: None)[1] == "no pause in force"
    G.write_json(G.STATE, {"pid": me, "start": G.start_of(me), "t": time.time(), "paused_since": time.time()})
    G.write_json(G.PAUSE, {"reason": "robot"})
    assert G.settled(lambda u: {"busy": True, "paused": None})[0] is False      # a job's setup, or a step session
    assert G.settled(lambda u: {"busy": True, "paused": {"reason": "robot"}}) == (True, "render work is at rest")
    assert G.settled(lambda u: {"busy": False}) == (True, "render work is at rest")
    assert G.settled(lambda u: None)[0] is True                                 # no service: nothing renders


def test_gate_and_bridge_agree_on_the_switch(monkeypatch, tmp_path):
    pause = pytest.importorskip("spine_render.pause")
    monkeypatch.delenv("UBE_RENDER_PAUSE_FILE", raising=False)
    monkeypatch.delenv("UBE_RENDER_GATE_DIR", raising=False)
    assert pause.FRESH_S == G.FRESH_S
    src = (HERE.parent / "render_gate.py").read_text()
    assert 'RUNTIME / "pause"' in src and pause.pause_file().name == "pause" and pause.pause_file().parent.name == "ube-render"
    out = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); import render_gate as G; from spine_render import pause; "
                          "print(G.PAUSE == pause.pause_file())", str(HERE.parent)], capture_output=True, text=True,
                         env={**os.environ, "XDG_RUNTIME_DIR": str(tmp_path)})
    assert out.stdout.strip() == "True", out.stderr


# ---- the real thing: real processes, a real watcher loop
COUNTER = "import sys, time\nn = 0\nwhile True:\n    n += 1; open(sys.argv[1], 'w').write(str(n)); time.sleep(0.05)\n"


def gate_env(tmp_path):
    return {**os.environ, "UBE_RENDER_GATE_DIR": str(tmp_path), "UBE_RENDER_GATE_LOG": str(tmp_path / "gate.log"), "RENDER_API_URLS": "",
            "UBE_RENDER_PAUSE_FILE": str(tmp_path / "pause")}


def wait_for(cond, s=8.0):
    t0 = time.time()
    while time.time() - t0 < s:
        if cond():
            return True
        time.sleep(0.05)
    return False


def state_of(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]


@pytest.fixture
def live(tmp_path):
    """A watcher process on its own runtime directory (no arms: the arm package's servers are not asked — a hold is the sign)."""
    env = gate_env(tmp_path)
    env["PYTHONPATH"] = str(tmp_path / "noarms") + os.pathsep + env.get("PYTHONPATH", "")
    (tmp_path / "noarms" / "lerobot_robot_spinearm").mkdir(parents=True)
    (tmp_path / "noarms" / "lerobot_robot_spinearm" / "__init__.py").write_text("raise ImportError('no arms in this test')\n")
    gate = [sys.executable, str(HERE.parent / "render_gate.py")]
    w = subprocess.Popen(gate + ["watch", "--period", "0.1", "--linger", "0.3"], env=env, stdout=subprocess.DEVNULL)
    assert wait_for(lambda: (tmp_path / "gate.json").exists())
    made = []
    yield tmp_path, env, gate, made
    for p in made:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    w.terminate(); w.wait(timeout=5)


def test_registered_work_is_stopped_for_a_hold_and_goes_on_after(live):
    tmp, env, gate, made = live
    (tmp / "count.py").write_text(COUNTER)
    r = subprocess.Popen(gate + ["run", "--name", "counter", "--", sys.executable, str(tmp / "count.py"), str(tmp / "n")], env=env, start_new_session=True)
    made.append(r)
    assert wait_for(lambda: (tmp / "n").exists() and list((tmp / "work").glob("*.json")))
    child = json.loads(next((tmp / "work").glob("*.json")).read_text())["pid"]
    assert subprocess.run(gate + ["status"], env=env, capture_output=True).returncode == 0
    h = subprocess.run(gate + ["hold", "roq", "--pid", str(os.getpid()), "--reason", "robot job t1", "--wait", "5"], env=env, capture_output=True, text=True)
    assert h.returncode == 0, h.stdout + h.stderr                      # at rest within the wait
    assert state_of(child) == "T"
    n0 = (tmp / "n").read_text(); time.sleep(0.5)
    assert (tmp / "n").read_text() == n0                               # it really stands still
    assert subprocess.run(gate + ["status"], env=env, capture_output=True).returncode == 1
    assert subprocess.run(gate + ["settled"], env=env, capture_output=True).returncode == 0
    subprocess.run(gate + ["release", "roq"], env=env, check=True)
    assert wait_for(lambda: state_of(child) != "T" and (tmp / "n").read_text() != n0)
    assert not (tmp / "pause").exists()
    os.killpg(child, signal.SIGTERM)
    assert r.wait(timeout=5) == 128 + signal.SIGTERM                   # an ordinary end is passed on as it is
    assert "pause: hold roq: robot job t1" in (tmp / "gate.log").read_text()


def test_work_that_cannot_wait_ends_with_75_and_is_not_started_under_a_pause(live):
    tmp, env, gate, made = live
    (tmp / "count.py").write_text(COUNTER)
    r = subprocess.Popen(gate + ["run", "--on-pause", "kill", "--name", "dry", "--", sys.executable, str(tmp / "count.py"), str(tmp / "n")], env=env,
                         start_new_session=True, stderr=subprocess.PIPE, text=True)
    made.append(r)
    assert wait_for(lambda: (tmp / "n").exists())
    h = subprocess.run(gate + ["hold", "roq", "--pid", str(os.getpid()), "--wait", "5"], env=env, capture_output=True, text=True)
    assert h.returncode == 0, h.stdout
    assert r.wait(timeout=5) == 75 and "run it again later" in r.stderr.read()
    again = subprocess.run(gate + ["run", "--on-pause", "kill", "--", sys.executable, "-c", "print('ran')"], env=env, capture_output=True, text=True)
    assert again.returncode == 75 and "ran" not in again.stdout
    subprocess.run(gate + ["release", "roq"], env=env, check=True)
    assert wait_for(lambda: not (tmp / "pause").exists())
    again = subprocess.run(gate + ["run", "--on-pause", "kill", "--low", "--", sys.executable, "-c", "import os; print('ran', os.nice(0))"], env=env, capture_output=True, text=True)
    # to niceness 15 — or where the caller already is, if that is lower in priority still (the test suite run under nice)
    assert again.returncode == 0 and again.stdout.split()[0] == "ran" and int(again.stdout.split()[1]) == max(15, os.nice(0))


def test_low_work_has_a_memory_ceiling_of_its_own(tmp_path):
    """10-09 12:38: a script of mine took 4 GB of this 5.8 GB machine. Under --low the work is killed alone at its ceiling."""
    import shutil
    if not shutil.which("systemd-run") or subprocess.run(["systemctl", "--user", "is-system-running"], capture_output=True).returncode not in (0, 1):
        pytest.skip("no user service manager")
    env = gate_env(tmp_path)
    gate = [sys.executable, str(HERE.parent / "render_gate.py")]
    grow = "x = [bytearray(10 * 2**20) for _ in range(40)]; print('took', len(x) * 10, 'MB')"
    r = subprocess.run(gate + ["run", "--low", "--mem-gb", "0.15", "--", sys.executable, "-c", grow], env=env, capture_output=True, text=True)
    assert r.returncode == 128 + signal.SIGKILL and "took" not in r.stdout
    r = subprocess.run(gate + ["run", "--low", "--", sys.executable, "-c", grow], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "took 400 MB"


def test_a_watcher_that_ends_leaves_nothing_stopped(tmp_path):
    env = gate_env(tmp_path)
    gate = [sys.executable, str(HERE.parent / "render_gate.py")]
    (tmp_path / "count.py").write_text(COUNTER)
    w = subprocess.Popen(gate + ["watch", "--period", "0.1", "--linger", "60"], env=env, stdout=subprocess.DEVNULL)
    assert wait_for(lambda: (tmp_path / "gate.json").exists())
    r = subprocess.Popen(gate + ["run", "--", sys.executable, str(tmp_path / "count.py"), str(tmp_path / "n")], env=env, start_new_session=True)
    try:
        assert wait_for(lambda: list((tmp_path / "work").glob("*.json")))
        child = json.loads(next((tmp_path / "work").glob("*.json")).read_text())["pid"]
        subprocess.run(gate + ["hold", "x", "--pid", str(os.getpid())], env=env, capture_output=True)
        assert wait_for(lambda: state_of(child) == "T")
        w.terminate(); assert w.wait(timeout=5) == 0
        assert wait_for(lambda: state_of(child) != "T") and not (tmp_path / "pause").exists() and not (tmp_path / "gate.json").exists()
    finally:
        try:
            os.killpg(r.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if w.poll() is None:
            w.kill()
