"""The remote policy (remote_policy.py, remote/policy_server.py): the conversion between a server's absolute answer and
the loop's chunk-relative chunk, and the wire format between the two — client and reference server over localhost."""
import importlib.util
import json
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "remote"))
import remote_policy as RP     # noqa: E402
import policy_server as PS     # noqa: E402
import rollout as RO           # noqa: E402
from rollout import AC         # noqa: E402


def rand_state(rng) -> np.ndarray:
    s = np.zeros(15)
    for off in (0, 7):
        s[off:off + 3] = rng.uniform([10, -60, 1000], [70, 15, 1045])
        q = rng.normal(size=4); s[off + 3:off + 7] = q / np.linalg.norm(q)
    s[14] = float(rng.random() > 0.5)
    return s


def walk(rng, s0: np.ndarray, T: int) -> np.ndarray:
    """T future states: small steps from s0, as a demonstration moves (0.2–0.4 mm and a fraction of a degree a frame)."""
    rows, s = [], s0.copy()
    for _ in range(T):
        d = np.r_[rng.normal(0, 0.3, 3), rng.normal(0, 0.004, 3), rng.normal(0, 0.3, 3), rng.normal(0, 0.004, 3), float(rng.random() > 0.5)]
        s = AC.from_delta(s, d); rows.append(s.copy())
    return np.stack(rows)


def test_absolute_rows_become_the_chunk_the_training_code_makes():
    torch = pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("train_rel_r", HERE.parents[1] / "train" / "lerobot-train-rel.py")
    TR = importlib.util.module_from_spec(spec); sys.modules["train_rel_r"] = TR; spec.loader.exec_module(TR)
    rng = np.random.default_rng(0)
    for _ in range(5):
        s0 = rand_state(rng); rows = walk(rng, s0, 16)
        mine = RP.abs_to_chunk(s0, rows)
        theirs = TR.to_chunk_relative(torch.tensor(s0[None], dtype=torch.float64), torch.tensor(rows[None], dtype=torch.float64))[0].numpy()
        assert mine.shape == (16, 13)
        assert np.allclose(mine, theirs, atol=1e-6)


def test_chunk_to_absolute_and_back_is_the_same_chunk():
    rng = np.random.default_rng(1)
    s0 = rand_state(rng); rows = walk(rng, s0, 40)
    done = (np.arange(40) >= 25).astype(float)
    ch = RP.abs_to_chunk(s0, np.c_[rows, done])
    assert ch.shape == (40, 14) and np.array_equal(ch[:, 13], done) and np.array_equal(ch[:, 12], rows[:, 14])
    back = RP.chunk_to_abs(s0, ch)
    assert back.shape == (40, 16) and np.allclose(back[:, :3], rows[:, :3], atol=1e-9) and np.allclose(back[:, 7:10], rows[:, 7:10], atol=1e-9)
    for off in (3, 10):   # the same rotation (a quaternion and its negative are one)
        dots = np.abs((back[:, off:off + 4] * rows[:, off:off + 4]).sum(1))
        assert np.allclose(dots, 1.0, atol=1e-9)
    assert np.allclose(RP.abs_to_chunk(s0, back), ch, atol=1e-9)


def test_the_loop_applies_a_remote_chunk_to_where_the_server_pointed():
    """cum_to_steps + from_delta (what live_loop.py does with a chunk) lands on the absolute rows the server sent."""
    rng = np.random.default_rng(2)
    s0 = rand_state(rng); rows = walk(rng, s0, 16)
    v = s0.copy()
    for d, want in zip(AC.cum_to_steps(RP.abs_to_chunk(s0, rows)), rows):
        v = AC.from_delta(v, d)
        assert np.allclose(v[:3], want[:3], atol=1e-8) and np.allclose(v[7:10], want[7:10], atol=1e-8)
        assert abs(abs(np.dot(v[3:7], want[3:7])) - 1) < 1e-9 and abs(abs(np.dot(v[10:14], want[10:14])) - 1) < 1e-9
        assert v[14] == want[14]


def test_answers_that_are_not_actions_are_refused():
    s0 = rand_state(np.random.default_rng(3)); rows = walk(np.random.default_rng(3), s0, 4)
    bad = rows.copy(); bad[1, 2] = np.nan
    with pytest.raises(RP.RemoteError):
        RP.abs_to_chunk(s0, bad)
    bad = rows.copy(); bad[:, 3:7] = 0.01          # "quaternions" of a model that returns relative or normalised values
    with pytest.raises(RP.RemoteError):
        RP.abs_to_chunk(s0, bad)
    with pytest.raises(RP.RemoteError):
        RP.abs_to_chunk(s0, rows[:, :13])            # a chunk-relative answer is not this protocol


def test_the_task_text_never_tells_a_done_model_that_the_drilling_is_over():
    assert RP.task_for(3, True) == RP.TASKS[3] and RP.task_for(4, True) == RP.TASKS[4]
    assert RP.task_for(5, True) == RP.TASKS[4] and RP.task_for(6, True) == RP.TASKS[4]
    assert RP.task_for(5, False) == RP.TASKS[5] and RP.task_for(1, False) == RP.TASKS[3]


class Fixed(PS.Backend):
    """A backend with a known answer: the request's state moved 1 mm a row along x (drill), trigger on, and as the 16th
    value either the flag (done from row 8) or a progress that rises 0.80 → 0.99 and passes 0.91 at row 9."""
    model = "fixed"; horizon = 16; image_size = 32

    def __init__(self, done=True, end="done"):
        self.done_output = done; self.seen = []; self.end_name = RP.DONE_NAME if end == "done" else RP.PROGRESS_NAME

    def predict(self, image, state, task):
        self.seen.append((image.copy(), state.copy(), task))
        rows = np.repeat(state[None], 16, 0).copy(); rows[:, 7] += np.arange(1, 17); rows[:, 14] = 1.0
        end = (np.arange(16) >= 8).astype(float) if self.end_name == RP.DONE_NAME else np.r_[np.linspace(0.80, 0.94, 12), [0.96, 0.97, 0.98, 0.99]]
        return np.c_[rows, end] if self.done_output else rows


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


def test_client_and_server_agree_on_the_wire(served):
    be = Fixed(); url = served(be)
    rp = RP.RemotePolicy(url, "s3cret", timeout_s=5, run="unit test")
    inf = rp.info()
    assert inf["protocol"] == RP.PROTOCOL and inf["done_output"] and inf["action_names"][-1] == RP.DONE_NAME and inf["action_horizon"] == 16
    rng = np.random.default_rng(4); s0 = rand_state(rng)
    img = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    rows = rp.act(img, s0, RP.TASKS[4])
    got_img, got_state, got_task = be.seen[-1]
    assert np.array_equal(got_img, img)                         # PNG: the server sees the very pixels
    assert np.allclose(got_state, s0, atol=0) and got_task == RP.TASKS[4]
    assert rows.shape == (16, 16) and np.allclose(rows[:, 7] - s0[7], np.arange(1, 17), atol=1e-5)
    ch = rp.chunk(img, s0, RP.TASKS[4])
    assert ch.shape == (16, 14) and np.allclose(ch[:, 6], np.arange(1, 17), atol=1e-5) and np.allclose(ch[:, :6], 0, atol=1e-6)
    assert RO.done_step(ch[:, 13], 16) == 8                     # the loop's own rule reads the server's done
    assert RO.done_step(ch[:, 13], 8) is None                   # … and not before the executed steps reach it
    assert rp.summary()["requests"] == 2 and rp.seq == 2 and rp.end_kind == "done"


def test_a_progress_model_ends_the_drilling_at_its_own_level(served):
    """User 10-09: the end output becomes a progress value — the name in /info says which form the 16th value has, and the
    loop's rule takes that form's level."""
    rp = RP.RemotePolicy(served(Fixed(end="progress")), "s3cret", timeout_s=5)
    inf = rp.info()
    assert inf["action_names"][-1] == RP.PROGRESS_NAME and inf["done_output"] and inf["end_output"] == "progress" and rp.end_kind == "progress"
    ch = rp.chunk(np.zeros((32, 32, 3), np.uint8), rand_state(np.random.default_rng(8)), RP.TASKS[4])
    assert ch.shape == (16, 14) and ch[0, 13] == pytest.approx(0.80) and ch[15, 13] == pytest.approx(0.99)
    level = RO.END_LEVEL[rp.end_kind]
    first = int(np.nonzero(ch[:, 13] > level)[0][0])
    assert level == 0.91 and first == 9 and RO.done_step(ch[:, 13], 16, level=level) == 9      # above the level from row 9 on
    assert RO.done_step(ch[:, 13], 9, level=level) is None                      # … not inside the first 9 steps
    assert RO.done_step(ch[:, 13], 16) == 0                                     # the flag's level (0.5) would end it at once: the form matters
    assert RP.RemotePolicy(served(Fixed(done=False)), "s3cret", timeout_s=5).end_kind is None


def test_a_model_without_the_done_output_gives_13_values(served):
    rp = RP.RemotePolicy(served(Fixed(done=False)), "s3cret", timeout_s=5)
    assert rp.info()["done_output"] is False
    ch = rp.chunk(np.zeros((32, 32, 3), np.uint8), rand_state(np.random.default_rng(5)), RP.TASKS[4])
    assert ch.shape == (16, 13)


def test_no_token_no_answer(served):
    url = served(Fixed())
    with pytest.raises(RP.RemoteError, match="401"):
        RP.RemotePolicy(url, "wrong", timeout_s=5).info()
    with pytest.raises(RP.RemoteError, match="401"):
        RP.RemotePolicy(url, None, timeout_s=5).info()
    assert json.loads(urllib.request.urlopen(url + "/health", timeout=5).read()) == {"ok": True}   # "is it up?" needs none


def test_a_wrong_image_or_state_is_a_400_not_a_guess(served):
    url = served(Fixed()); rp = RP.RemotePolicy(url, "s3cret", timeout_s=5)
    with pytest.raises(RP.RemoteError, match="400"):
        rp.act(np.zeros((64, 64, 3), np.uint8), rand_state(np.random.default_rng(6)), RP.TASKS[4])   # not the model's image size
    bad = rand_state(np.random.default_rng(6)); bad[0] = np.nan
    with pytest.raises((RP.RemoteError, ValueError)):
        rp.act(np.zeros((32, 32, 3), np.uint8), bad, RP.TASKS[4])


def test_a_server_that_is_not_there_is_an_error_in_time():
    rp = RP.RemotePolicy("http://127.0.0.1:9", "x", timeout_s=1, retries=1)    # port 9: nothing listens
    with pytest.raises(RP.RemoteError, match="no answer"):
        rp.info()


def test_a_backend_that_breaks_is_a_500_and_the_client_says_so(served):
    class Broken(Fixed):
        def predict(self, image, state, task):
            raise RuntimeError("out of memory")
    rp = RP.RemotePolicy(served(Broken()), "s3cret", timeout_s=5)
    with pytest.raises(RP.RemoteError, match="500"):
        rp.act(np.zeros((32, 32, 3), np.uint8), rand_state(np.random.default_rng(7)), RP.TASKS[4])


def test_the_token_file_must_be_private(tmp_path):
    f = tmp_path / "t"; f.write_text("abc\n"); f.chmod(0o644)
    with pytest.raises(RP.RemoteError, match="readable by others"):
        RP.read_token(f)
    f.chmod(0o600)
    assert RP.read_token(f) == "abc" and RP.read_token(None) is None


def test_the_demo_backend_continues_a_demonstration_in_order():
    """10-09 dry run: the first question, asked at the entry, was answered with the episode's END (the tools pass the same
    place on their way out) and "progress 1.0" — the loop ended the drilling with nothing removed."""
    from types import SimpleNamespace
    ds = Path.home() / "il-robot/datasets/spine_hand24_replay_trim_water_progress"
    if not ds.exists():
        pytest.skip("dataset not on this machine")
    be = PS.DemoBackend(SimpleNamespace(dataset=str(ds), episodes="6", horizon=16, done_output=False, progress_output=True))
    st, prog = be.ep[6]; fc = be.first_cut[6]; img = np.zeros((256, 256, 3), np.uint8)
    assert 50 < fc < 200 and prog[fc - 3] == 0.0 and prog[fc - 1] < 0.05 and prog[-1] == pytest.approx(1.0)   # the render sees the first cut a frame early
    last = st[-1]                                                  # the pose at the episode's end: on the way out
    rows = be.predict(img, last, RP.TASKS[3])                      # … asked as the approach: answered from before the first cut
    assert rows.shape == (16, 16) and rows[:, 15].max() < 0.2 and be.cur[1] <= fc      # the approach or the very first cuts — not "1.0"
    seen = []
    for i in range(0, len(st) - 1, 16):                            # the loop following the demonstration, 16 frames an answer
        rows = be.predict(img, st[i], RP.TASKS[3] if i < fc else RP.TASKS[4])
        seen.append(be.cur[1])
        assert np.allclose(rows[0, :15], st[min(be.cur[1] + 1, len(st) - 1)])
    assert all(b >= a for a, b in zip(seen, seen[1:]))             # never backwards
    assert max(abs(a - i) for a, i in zip(seen, range(0, len(st) - 1, 16))) <= 16      # … and where the loop is
    assert rows[:, 15].max() == pytest.approx(1.0)                 # the end is reached at the end
    far = st[fc + 200].copy(); far[0:3] += 40.0; far[7:10] += 40.0 # a question from somewhere else: a new start, nearest of all
    be.predict(img, far, RP.TASKS[4])
    assert be.cur is not None



def test_two_requests_on_the_way_at_once_each_get_their_own_answer(served):
    """The continuous loop asks again while an answer is still on its way (10-12 dry run: a server 2.2 s slow, a look
    every 1.6 s — "answer is for request 1, asked 2" and the run was aborted)."""
    import threading
    import time as _t

    class Slow(Fixed):
        def predict(self, image, state, task):
            _t.sleep(0.3)
            return super().predict(image, state, task)
    rp = RP.RemotePolicy(served(Slow(done=False)), "s3cret", timeout_s=10)
    rp.info()
    img = np.zeros((32, 32, 3), np.uint8); rng = np.random.default_rng(3)
    states = [rand_state(rng) for _ in range(3)]; got: dict = {}

    def ask(k):
        _t.sleep(0.1 * k)                                   # the second and third leave while the first is being answered
        got[k] = rp.act(img, states[k], RP.TASKS[4])
    th = [threading.Thread(target=ask, args=(k,)) for k in range(3)]
    for x in th:
        x.start()
    for x in th:
        x.join()
    for k in range(3):                                      # each answer belongs to its own question: it starts from that state
        assert np.allclose(got[k][0, 7] - states[k][7], 1.0) and np.allclose(got[k][:, 0:3], states[k][0:3])
    assert sorted(x["seq"] for x in rp.log) == [1, 2, 3]


def test_one_connection_serves_the_requests_of_a_thread(served):
    """Through an SSH tunnel every new connection is a new channel — a round trip to the far end before the request
    leaves. The client keeps its connection; a server that closed it meanwhile costs one fresh connection, not an error."""
    rp = RP.RemotePolicy(served(Fixed()), "s3cret", timeout_s=5, retries=0, run="keep")
    img = np.zeros((32, 32, 3), np.uint8); st = rand_state(np.random.default_rng(7))
    rp.act(img, st, RP.TASKS[4]); c = rp._local.conn; sock = c.sock
    rp.act(img, st, RP.TASKS[4])
    assert rp._local.conn is c and c.sock is sock                       # the same connection, still open
    sock.close()                                                        # … cut under the client's feet
    rows = rp.act(img, st, RP.TASKS[4])                                 # retries=0: this must not count as the one failure
    assert rows.shape == (16, 16) and rp._local.conn.sock is not sock


def test_a_jpeg_observation_is_taken_when_the_server_lists_it(served):
    be = Fixed(); url = served(be)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    assert RP.image_format("jpg:80") == ("jpeg", 80) and RP.image_format(None) == ("png", 92)
    assert np.array_equal(RP.decode_image(RP.encode_image(img)), img)                                  # PNG: the very pixels
    smooth = np.repeat(np.repeat(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8), 8, 0), 8, 1)       # a picture, not noise
    jpg = RP.encode_image(smooth, "jpeg", 92)
    assert len(jpg) < len(RP.encode_image(smooth)) or len(jpg) < 20000
    assert np.abs(RP.decode_image(jpg).astype(int) - smooth).mean() < 4.0                              # close, not equal
    rp = RP.RemotePolicy(url, "s3cret", timeout_s=5, retries=0, run="jpeg", image="jpeg:92")
    assert "jpeg" in rp.info()["image"]["formats"]
    small = smooth[::8, ::8].copy()                                                                    # the fixed backend takes 32 × 32
    assert rp.act(small, rand_state(rng), RP.TASKS[4]).shape == (16, 16)
    assert be.seen[-1][0].shape == (32, 32, 3) and np.abs(be.seen[-1][0].astype(int) - small).mean() < 12
    with pytest.raises(ValueError):
        RP.image_format("webp")


def test_goal_in_words_only_for_a_model_that_was_taught_it(served, tmp_path):
    """lang1 (user, 10-13): the server says its model's instruction set, the loop says the goal in that set's first wording;
    the legacy models get their old phase texts and refuse a goal."""
    INS = RP.INS
    assert RP.task_for(3, True) == RP.TASKS[3] and RP.task_for(4, True, {"name": "phase1"}) == RP.TASKS[4]
    with pytest.raises(ValueError):
        RP.task_for(4, True, None, {"level": "L5", "extent": "contralateral"})
    tab = INS.table("lang1")
    assert RP.task_for(3, True, tab) == "Advance the endoscope and the drill to the left L4 lamina."
    assert RP.task_for(6, True, tab, INS.parse_goal("L5:ipsi")) == "Left L5 laminectomy, ipsilateral only."
    assert INS.parse_goal("ipsilateral") == {"level": "L4", "extent": "ipsilateral"} and INS.parse_goal("l3") == {"level": "L3", "extent": "contralateral"} and INS.parse_goal("") is None
    with pytest.raises(ValueError):
        INS.parse_goal("L6")
    f = tmp_path / "export.json"; f.write_text(json.dumps({"instructions": tab}))
    be = Fixed(end="progress"); be.read_instructions(f); url = served(be)
    rp = RP.RemotePolicy(url, "s3cret", timeout_s=5, run="unit test")
    inf = rp.info()
    assert inf["instructions"]["name"] == "lang1" and set(inf["tasks"]) == set(INS.texts(tab)) and len(inf["tasks"]) == 9 + 18 + 8
    rp.goal = INS.parse_goal("L3:contra")
    assert rp.task(3) in inf["tasks"] and "L3" in rp.task(3) and rp.task(4) == INS.wordings(4, "L3", "contralateral")[0]
    assert rp.task(4, {"level": "L4", "extent": "ipsilateral"}) == INS.wordings(4, "L4", "ipsilateral")[0]      # a held-out demonstration's own goal
    old = Fixed(end="progress"); f2 = tmp_path / "old.json"; f2.write_text(json.dumps({"instructions": INS.table("phase1", RP.PH.PHASES)}))
    old.read_instructions(f2); assert old.instructions is None and "instructions" not in old.info() and old.info()["tasks"] == [RP.TASKS[3], RP.TASKS[4]]


def test_guidance_negative_is_the_other_extent_at_the_same_level():
    neg = PS.negative_sentences()
    INS = RP.INS
    for lv in INS.LEVELS:
        for w in INS.wordings(4, lv, "ipsilateral"):
            assert neg[w] == INS.wordings(4, lv, "contralateral")[0]
        for w in INS.wordings(4, lv, "contralateral"):
            assert neg[w] == INS.wordings(4, lv, "ipsilateral")[0]
    assert RP.TASKS[4] not in neg and INS.text("lang1", 3, "L4") not in neg      # legacy texts and the approach: no guidance


def test_guidance_negative_can_be_lang2s_neutral_sentence():
    INS = RP.INS
    neg = PS.negative_sentences("lang2", "neutral")
    for lv in INS.LEVELS:
        for ex in INS.EXTENTS:
            for w in INS.wordings(4, lv, ex):
                assert neg[w] == INS.wordings(4, lv, INS.NEUTRAL)[0]      # plain classifier-free guidance
    assert not set(INS.wordings(4, "L4", INS.NEUTRAL)) & set(neg)         # the neutral goal itself: no guidance
    assert PS.negative_sentences("lang2", "other") == PS.negative_sentences()   # a lang2 model against the other extent: as lang1
    with pytest.raises(ValueError):
        PS.negative_sentences("lang1", "neutral")                           # lang1 never had it


def test_the_loop_says_a_lang2_goal_in_its_words():
    tab = RP.INS.table("lang2")
    assert RP.task_for(4, True, tab, {"level": "L4", "extent": RP.INS.NEUTRAL}) == "Left L4 laminectomy."
    assert RP.task_for(4, True, tab, {"level": "L5", "extent": "ipsilateral"}) == RP.INS.wordings(4, "L5", "ipsilateral")[0]
    with pytest.raises(ValueError):
        RP.task_for(4, True, RP.INS.table("lang1"), {"level": "L4", "extent": RP.INS.NEUTRAL})


def test_task_for_a_drilling_sentence_of_the_callers_own():
    """10-16 (live_loop --goal-text-file): the drilling sentence as given — a paraphrase the model never met; the approach
    sentence stays the set's for the goal's level."""
    ins = {"name": "lang2"}; own = "Take down the left L4 lamina and carry on over to the opposite side."
    g = {"level": "L4", "extent": "contralateral", "drill_text": own}
    assert RP.task_for(4, True, ins, g) == own
    assert RP.task_for(3, True, ins, g) == RP.INS.text("lang2", 3, "L4", "contralateral")
