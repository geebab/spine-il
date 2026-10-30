"""export_lerobot.py end to end on tiny synthetic sessions (no render API, no real files; ~10 s)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import export_lerobot as E  # noqa: E402
import render_cache as RC  # noqa: E402
import sources as S  # noqa: E402
import synth  # noqa: E402

HAND = "s_20260928T000000_altician_lumbar1_trk_aaaa"
ROBOT = "s_20260929T000000_altician_lumbar1_trk_bbbb"
RUN = "20260929T000000_replay_pr_aaaa01"


@pytest.fixture()
def world(tmp_path, monkeypatch):
    sessions, runs = tmp_path / "sessions", tmp_path / "runs"
    synth.write_hand(sessions / "2026-09-28" / f"{HAND}.jsonl", HAND)
    info = synth.write_robot(sessions, runs, ROBOT, HAND, RUN)
    # an old-format file (no hdr.origin) must be listed as skipped, not exported
    old = json.loads((sessions / "2026-09-28" / f"{HAND}.jsonl").read_text().splitlines()[0])
    old.pop("origin")
    old["sessionId"] = "s_20260927T000000_altician_lumbar1_trk_cccc"
    (sessions / "2026-09-27").mkdir()
    (sessions / "2026-09-27" / "old.jsonl").write_text(json.dumps(old) + "\n")
    monkeypatch.setattr(S, "RUNS_ROOT", runs)
    monkeypatch.setattr(S, "HAND_WINDOWS_FROM_S", {"aaaa": 3.0})
    monkeypatch.setattr(S, "WINDOW_LEN_S", 10.0)
    return {"sessions": sessions, "runs": runs, "tmp": tmp_path, **info}


def _load(out):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("local/test", root=out)
    hf = ds.hf_dataset.with_format("numpy")
    side = [json.loads(x) for x in (out / "meta/spine_il/episodes.jsonl").read_text().splitlines()]
    return ds, hf, side


def test_full_hand_and_robot(world):
    out = world["tmp"] / "ds"
    E.main(["--out", str(out), "--sessions", str(world["sessions"]), "--images", "none", "--repo-id", "local/test",
            "--fps", "15"])
    ds, hf, side = _load(out)
    assert ds.meta.total_episodes == 2 and ds.meta.fps == 15
    st, ac = np.asarray(hf["observation.state"]), np.asarray(hf["action"])
    ep = np.asarray(hf["episode_index"]).reshape(-1)
    hand, robot = side
    assert hand["source"] == "hand" and robot["source"] == "robot"
    # hand: the grid ends at the last shown endo frame (drawn.tExp = tExp − 30 ms → 19.97 s): 300 grid frames,
    # the last one only as an action
    assert hand["n_frames"] == 299
    h = ep == 0
    np.testing.assert_allclose(ac[h][:-1], st[h][1:], atol=1e-5)
    # positions: the synthetic endo tip at frame k (t = k/15)
    t = np.arange(299) / 15.0
    want = np.array([synth.hand_tip(x) for x in t])
    np.testing.assert_allclose(st[h][:, :3], want, atol=0.6)  # linear interpolation of a piecewise-linear path
    assert np.abs(np.linalg.norm(st[:, 3:7], axis=1) - 1).max() < 1e-5
    assert np.abs(np.linalg.norm(st[:, 10:14], axis=1) - 1).max() < 1e-5
    # trigger on 5–12 s → 105 of 299 frames
    assert st[h][:, -1].sum() == pytest.approx(105, abs=1)
    assert hand["trigger_duty_dataset"] == pytest.approx(hand["trigger_duty_source"], abs=0.01)
    # phases: all six, monotonic, tasks per frame
    ph = np.asarray(hf["phase.index"]).reshape(-1)
    assert set(ph[h].tolist()) == {1, 2, 3, 4, 5, 6} and np.all(np.diff(ph[h]) >= 0)
    tasks = set(ds.meta.tasks.index) if hasattr(ds.meta.tasks, "index") else set(ds.meta.tasks)
    assert "4 L4-5 후궁 드릴링 / drill the L4-5 lamina" in tasks
    # hand episodes carry no joints
    assert np.asarray(hf["robot.joints_valid"])[h].sum() == 0
    # robot: replay span 1–21 s robot time at speed 0.5 → 10 s hand time → frames k = 8..157 (step 133.3 ms)
    r = ep == 1
    assert robot["grid"]["k0"] == 8 and robot["grid"]["k1"] == 157 and robot["n_frames"] == 149
    assert robot["speed"] == 0.5 and robot["contaminated"] is True
    assert robot["cut_agreement"]["iou"] == 0.7 and robot["cut_agreement"]["volume_ratio"] == 1.1
    assert set(ph[r].tolist()) <= {3, 4, 5}
    # joints on the tExp axis: q_j(t) = j + 0.001 (t − BASE); left arm j0 = 0, right arm j0 = 100
    tq = synth.BASE + np.arange(8, 157) * 1000.0 / 7.5
    q = np.asarray(hf["robot.joints"])[r]
    np.testing.assert_allclose(q[:, 0], 0.001 * (tq - synth.BASE), atol=2e-3)
    np.testing.assert_allclose(q[:, 6], 100 + 0.001 * (tq - synth.BASE), atol=2e-3)
    # joint target = the goal held at the next frame (goal = q + 1, ZOH at 30 Hz)
    g = np.asarray(hf["robot.joints_target"])[r][:, 1]
    assert np.all(np.abs(g - (1 + 1 + 0.001 * (tq + 1000 / 7.5 - synth.BASE))) < 0.05)
    assert np.asarray(hf["robot.joints_valid"])[r].min() == 1
    # the old file is listed as skipped with its reason
    exp = json.loads((out / "meta/spine_il/export.json").read_text())
    assert any("origin" in s["why"] for s in exp["skipped"])
    splits = json.loads((out / "meta/spine_il/splits.json").read_text())
    assert splits["hand"] == [0] and splits["robot"] == [1] and splits["contaminated"] == [1]


def test_windows_mode(world):
    out = world["tmp"] / "dsw"
    E.main(["--out", str(out), "--sessions", str(world["sessions"]), "--images", "none", "--repo-id", "local/test",
            "--mode", "windows", "--include", "hand"])
    ds, hf, side = _load(out)
    assert ds.meta.total_episodes == 1
    m = side[0]
    assert m["window"]["from_s"] == 3.0 and m["kind"] == "window"
    assert m["n_frames"] == 150  # 10 s at 15 fps (151 grid frames, last one = action only)
    assert m["t_frame0_ms"] == pytest.approx(synth.BASE + 3000.0)


def test_images_from_cache(world):
    from PIL import Image
    cache = world["tmp"] / "cache"
    s = [x for x in S.discover(world["sessions"]) if x.sid == "aaaa"][0]
    plan = RC.plan_for(s, 15.0)
    d = RC.chunk_dir(cache, plan, ("endo",), 16, plan.k0, plan.k1)
    (d / "frames" / "endo").mkdir(parents=True)
    for i in range(plan.k1 - plan.k0 + 1):
        Image.new("RGB", (16, 16), (i % 256, 0, 0)).save(d / "frames" / "endo" / f"{i:06d}.png")
    (d / "DONE").write_text("{}")
    out = world["tmp"] / "dsi"
    E.main(["--out", str(out), "--sessions", str(world["sessions"]), "--cache", str(cache), "--size", "16",
            "--image-dtype", "image", "--repo-id", "local/test", "--include", "hand"])
    ds, hf, side = _load(out)
    item = ds[7]
    img = item["observation.images.endo"]
    assert tuple(img.shape) == (3, 16, 16)
    assert float(img[0].mean()) == pytest.approx(7 / 255, abs=1e-3)   # frame 7 ↔ grid frame 7
    assert item["task"].startswith("1 ")


def test_images_required_skips_uncached(world):
    out = world["tmp"] / "dsn"
    with pytest.raises(Exception):  # no episode has images → nothing to write / validate
        E.main(["--out", str(out), "--sessions", str(world["sessions"]), "--cache", str(world["tmp"] / "none"),
                "--repo-id", "local/test", "--include", "hand"])
    exp = json.loads((out / "meta/spine_il/export.json").read_text())
    assert any("렌더" in s["why"] for s in exp["skipped"])


def test_quat_hemisphere_and_nlerp():
    t = np.array([0.0, 100.0, 1000.0])
    q = np.array([[1, 0, 0, 0], [0.9950042, 0, 0, 0.0998334], [1, 0, 0, 0]], float)
    ts = E.ToolSamples(t, np.array([[0, 0, 0], [10, 0, 0], [0, 0, 0]], float), q)
    pos, qq, v = E.sample_tool(ts, np.array([50.0, 500.0, 2000.0]))
    assert pos[0, 0] == pytest.approx(5.0)
    assert pos[1, 0] == pytest.approx(10.0)          # 900 ms gap is held, not bridged
    assert v.tolist() == [1.0, 0.0, 0.0]
    np.testing.assert_allclose(np.linalg.norm(qq, axis=1), 1.0)


def test_exclude_splits_the_episode(world):
    # stamps 8.0–9.0 s of the hand session left out: grid frames 120..135 → frames 0..118 (119 = the last target) and 136..298
    ex, qc = world["tmp"] / "ex.json", world["tmp"] / "qc_trim.json"
    rng = [[synth.BASE + 7990.0, synth.BASE + 9010.0]]
    ex.write_text(json.dumps({"aaaa": rng}))
    qc.write_text(json.dumps({"schema": "altician.qc-trim/1", "sessions": {HAND: {"exclude_tExp_ms": rng, "reason": "test"}}}))
    out = world["tmp"] / "dsx"
    E.main(["--out", str(out), "--sessions", str(world["sessions"]), "--images", "none", "--repo-id", "local/test", "--include", "hand",
            "--exclude", str(ex)])
    ds, hf, side = _load(out)
    assert [m["n_frames"] for m in side] == [119, 163] and [m["part"] for m in side] == [[1, 2], [2, 2]]
    ac, ep = np.asarray(hf["action"]), np.asarray(hf["episode_index"]).reshape(-1)
    t = np.asarray(hf["time.src_s"]).reshape(-1)
    assert t[ep == 0].max() == pytest.approx(118 / 15, abs=1e-3) and t[ep == 1].min() == pytest.approx(136 / 15, abs=1e-3)
    np.testing.assert_allclose(ac[ep == 0][-1, :3], synth.hand_tip(119 / 15), atol=0.6)   # the last target = the frame before the gap
    assert side[1]["t_frame0_ms"] == pytest.approx(synth.BASE + 136 * 1000 / 15)
    # the viewer's qc_trim.json gives the same pieces; in the Open-H layout both stay in one split
    out2 = world["tmp"] / "dsq"
    E.main(["--out", str(out2), "--sessions", str(world["sessions"]), "--images", "none", "--repo-id", "local/test", "--include", "hand",
            "--exclude", str(qc), "--layout", "openh"])
    side2 = [json.loads(x) for x in (out2 / "meta/spine_il/episodes.jsonl").read_text().splitlines()]
    assert [m["n_frames"] for m in side2] == [119, 163] and {m["split"] for m in side2} == {"train"}


def test_done_output_is_a_16th_action_value_and_the_task_does_not_tell(world):
    """--done-output (user 10-07 / 10-09: the policy ends the drilling itself, in the GR00T data too)."""
    out = world["tmp"] / "ds_done"
    E.main(["--out", str(out), "--sessions", str(world["sessions"]), "--images", "none", "--repo-id", "local/test", "--include", "hand",
            "--fps", "15", "--done-output"])
    ds, hf, side = _load(out)
    assert ds.meta.features["action"]["names"] == E.STATE_NAMES + ["drill.done"] and ds.meta.features["action"]["shape"] == (16,)
    st, ac = np.asarray(hf["observation.state"]), np.asarray(hf["action"])
    ph = np.asarray(hf["phase.index"]).reshape(-1)
    assert st.shape[1] == 16 and ac.shape[1] == 16                              # the state is as long as the action …
    assert ds.meta.features["observation.state"]["names"] == E.STATE_NAMES + ["pad.zero"] and not st[:, 15].any()   # … by a value that is always 0
    np.testing.assert_allclose(ac[:-1, :15], st[1:, :15], atol=1e-5)            # the pose part is still the next state
    torch = pytest.importorskip("torch")                                        # LeRobot's relative actions work on it (15 under 16 does not)
    from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep, to_absolute_actions, to_relative_actions
    mask = RelativeActionsProcessorStep(enabled=True, exclude_joints=["trigger", "done"],
                                        action_names=ds.meta.features["action"]["names"])._build_mask(16)
    a_t, s_t = torch.tensor(ac[None, :8]), torch.tensor(st[:1])
    rel = to_relative_actions(a_t, s_t, mask)
    assert torch.equal(rel[..., 14:], a_t[..., 14:]) and torch.allclose(to_absolute_actions(rel, s_t, mask), a_t, atol=1e-5)
    with pytest.raises(RuntimeError):
        to_relative_actions(a_t, s_t[:, :15], mask)
    assert set(np.unique(ac[:, 15]).tolist()) <= {0.0, 1.0}
    np.testing.assert_array_equal(ac[:-1, 15], (ph[1:] >= 5).astype(np.float32))  # 1 = the frame it leads to is after the last cut
    assert 0 < ac[:, 15].sum() < len(ac)                                        # the synthetic session has both
    assert np.all(np.diff(ac[:, 15]) >= 0)                                      # a step: once over, over
    # the text of a frame after the last cut is the drilling instruction — and phase.index still says 5
    ti = np.asarray(hf["task_index"]).reshape(-1)
    tasks = {int(v["task_index"]): str(k) for k, v in ds.meta.tasks.iterrows()}
    texts = np.array([tasks[int(i)] for i in ti])
    assert (ph >= 5).any() and set(texts[ph >= 5]) == {E.PH.PHASES[4]}
    assert E.PH.PHASES[5] not in set(texts) and E.PH.PHASES[6] not in set(texts)
    assert set(texts[ph == 4]) == {E.PH.PHASES[4]}
    exp = json.loads((out / "meta/spine_il/export.json").read_text())
    assert exp["done_output"]["name"] == "drill.done" and exp["done_output"]["index"] == 15
    rep = E.validate(out, "local/test")
    assert rep["action_is_next_state"] and rep["done_is_target_frame_after_last_cut"] and rep["done_frames"] == int(ac[:, 15].sum())
    E.DONE_OUTPUT[0] = False


def test_without_the_option_nothing_changes(world):
    out = world["tmp"] / "ds_plain"
    E.main(["--out", str(out), "--sessions", str(world["sessions"]), "--images", "none", "--repo-id", "local/test", "--include", "hand",
            "--fps", "15"])
    ds, hf, _ = _load(out)
    assert ds.meta.features["action"]["shape"] == (15,) and np.asarray(hf["action"]).shape[1] == 15
    assert "done_output" not in json.loads((out / "meta/spine_il/export.json").read_text())
    ph = np.asarray(hf["phase.index"]).reshape(-1); ti = np.asarray(hf["task_index"]).reshape(-1)
    tasks = {int(v["task_index"]): str(k) for k, v in ds.meta.tasks.iterrows()}
    assert {tasks[int(i)] for i in ti[ph >= 5]} == {E.PH.PHASES[int(p)] for p in np.unique(ph[ph >= 5])}   # each phase its own text


def test_progress_is_a_column_of_every_dataset_and_can_be_the_16th_action_value(world):
    """cut.progress (bone removed so far / the session's total) and --progress-output (user 10-09: the end output becomes a
    progress value). Without cached frames it comes from the recording's cut rows."""
    out = world["tmp"] / "ds_prog"
    E.main(["--out", str(out), "--sessions", str(world["sessions"]), "--images", "none", "--repo-id", "local/test", "--include", "hand",
            "--fps", "15", "--progress-output"])
    ds, hf, side = _load(out)
    assert ds.meta.features["action"]["names"] == E.STATE_NAMES + ["drill.progress"]
    assert ds.meta.features["observation.state"]["names"] == E.STATE_NAMES + ["pad.zero"]
    st, ac = np.asarray(hf["observation.state"]), np.asarray(hf["action"])
    ph, pr = np.asarray(hf["phase.index"]).reshape(-1), np.asarray(hf["cut.progress"]).reshape(-1)
    assert pr.min() == 0.0 and pr.max() == 1.0 and np.all(np.diff(pr) >= 0)          # 0 → 1, never falling
    assert np.all(pr[ph < 4] == 0.0) and np.all(pr[ph >= 5] == 1.0)                  # nothing before the first cut, all after the last
    assert 0 < pr[ph == 4].min() <= pr[ph == 4].max() <= 1.0 and len(np.unique(pr)) > 3   # it rises while drilling
    np.testing.assert_allclose(ac[:-1, 15], pr[1:], atol=1e-6)                       # the 16th value = the progress of the frame it leads to
    np.testing.assert_allclose(ac[:-1, :15], st[1:, :15], atol=1e-5) ; assert not st[:, 15].any()
    ti = np.asarray(hf["task_index"]).reshape(-1)
    tasks = {int(v["task_index"]): str(k) for k, v in ds.meta.tasks.iterrows()}
    assert {tasks[int(i)] for i in ti[ph >= 5]} == {E.PH.PHASES[4]}                  # the text does not give the end away
    exp = json.loads((out / "meta/spine_il/export.json").read_text())
    assert exp["progress_output"]["name"] == "drill.progress" and "done_output" not in exp
    assert side[0]["progress_source"].startswith("cut rows")
    rep = E.validate(out, "local/test")
    assert rep["action_is_next_state"] and rep["progress_is_the_target_frames"] and rep["progress"]["never_falls"]
    E.PROGRESS_OUTPUT[0] = False
    with pytest.raises(SystemExit):
        E.main(["--out", str(world["tmp"] / "x"), "--sessions", str(world["sessions"]), "--images", "none", "--done-output", "--progress-output"])
    E.DONE_OUTPUT[0] = False; E.PROGRESS_OUTPUT[0] = False


def test_progress_from_the_render_cache_replaces_the_cut_rows(world):
    """With cached frames the fraction is the replay render's own removed volume per frame (what the evaluation measures);
    a cache without per-frame metrics leaves the cut rows' value."""
    from PIL import Image
    cache = world["tmp"] / "cache_p"
    s = [x for x in S.discover(world["sessions"]) if x.sid == "aaaa"][0]
    plan = RC.plan_for(s, 15.0)
    d = RC.chunk_dir(cache, plan, ("endo",), 16, plan.k0, plan.k1)
    (d / "frames" / "endo").mkdir(parents=True)
    n = plan.k1 - plan.k0 + 1
    for i in range(n):
        Image.new("RGB", (16, 16), (i % 256, 0, 0)).save(d / "frames" / "endo" / f"{i:06d}.png")
    (d / "DONE").write_text("{}")
    args = ["--sessions", str(world["sessions"]), "--cache", str(cache), "--size", "16", "--image-dtype", "image", "--repo-id", "local/test",
            "--include", "hand"]
    out0 = world["tmp"] / "dsp0"
    E.main(["--out", str(out0), *args])                                             # no metrics.jsonl in the cache
    _, hf0, side0 = _load(out0)
    p0 = np.asarray(hf0["cut.progress"]).reshape(-1)
    assert side0[0]["progress_source"].startswith("cut rows") and 0.0 < p0.max() <= 1.0
    # a render whose removed volume rises with the square of the frame number: unlike any cut-row count
    vol = [0.001 * i * i for i in range(n)]
    (d / "metrics.jsonl").write_text("\n".join(json.dumps({"i": i, "bridge": {"removedMm3": v}}) for i, v in enumerate(vol)) + "\n")
    out1 = world["tmp"] / "dsp1"
    E.main(["--out", str(out1), *args, "--progress-output"])
    ds, hf, side = _load(out1)
    pr, ac = np.asarray(hf["cut.progress"]).reshape(-1), np.asarray(hf["action"])
    assert side[0]["progress_source"] == "render removedMm3" and side[0]["removed_mm3_final"] == pytest.approx(vol[-1], abs=1e-3)
    np.testing.assert_allclose(pr, np.asarray(vol[:len(pr)]) / vol[-1], atol=1e-6)   # frame k of the dataset ↔ grid frame k
    np.testing.assert_allclose(ac[:-1, 15], pr[1:], atol=1e-6)
    assert not np.allclose(pr, p0[:len(pr)], atol=0.05)
    E.PROGRESS_OUTPUT[0] = False


def _with_cut_centres(world, x_mm: float) -> None:
    """The synthetic hand session's bone cuts at x = x_mm (scene), so that the lang1 / lang2 goal has an extent."""
    f = world["sessions"] / "2026-09-28" / f"{HAND}.jsonl"
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    for r in rows:
        if r.get("k") == "cut":
            r.update(c=[[x_mm, -10.0, 0.0]], mod=1.0)
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


@pytest.mark.parametrize("cut_x, extent, kind", [(40.0, "contralateral", "fork"), (60.0, "ipsilateral", "end"), (50.0, "unspecified", None)])
def test_lang2_dropout_and_decision_copies(world, monkeypatch, cut_x, extent, kind):
    """10-14: lang2 names only a clear extent; --cond-dropout gives a share of the drilling frames the neutral sentence;
    --decision-copies writes the frames where the two goals part again as episodes of their own (not recordings)."""
    _with_cut_centres(world, cut_x)
    monkeypatch.setattr(E.INS, "level_of", lambda cc, *a, **k: ("L4", {"share": {"L4": 1.0}, "surface_mm_median": 0.0}))
    monkeypatch.setitem(E.INS.MIDLINE_X_MM, "L4", 53.0)       # the drill works at x = 50: 3 mm past this midline
    monkeypatch.setitem(E.INS.EXTENT_X_MM, "L4", 51.0)
    out = world["tmp"] / f"ds_lang2_{extent}"
    E.main(["--out", str(out), "--sessions", str(world["sessions"]), "--images", "none", "--repo-id", "local/test", "--include", "hand",
            "--fps", "15", "--progress-output", "--instructions", "lang2", "--cond-dropout", "0.3", "--decision-copies", "2",
            "--decision-frames", "20"])
    ds, hf, side = _load(out)
    assert side[0]["goal"]["extent"] == extent and side[0]["goal"]["past_midline_mm"] == pytest.approx(53.0 - cut_x)
    ep = np.asarray(hf["episode_index"]).reshape(-1); st = np.asarray(hf["observation.state"])
    ph = np.asarray(hf["phase.index"]).reshape(-1); ti = np.asarray(hf["task_index"]).reshape(-1)
    tasks = {int(v["task_index"]): str(k) for k, v in ds.meta.tasks.iterrows()}
    texts = np.array([tasks[int(i)] for i in ti])
    neutral, named = set(E.INS.wordings(4, "L4", E.INS.NEUTRAL)), set(E.INS.wordings(4, "L4", extent))
    drill = (ep == 0) & (ph >= 4)
    assert set(texts[(ep == 0) & (ph == 3)]) <= set(E.INS.wordings(3, "L4"))          # the approach names no extent
    if kind is None:                                                                   # a neutral recording: no copies, no extent
        assert len(side) == 1 and set(texts[drill]) <= neutral
        assert json.loads((out / "meta/spine_il/export.json").read_text())["instructions"]["decision_copies"]["episodes"] == 0
        return
    share = np.isin(texts[drill], list(neutral)).mean()
    assert 0.15 < share < 0.45 and set(texts[drill]) <= neutral | named               # about 0.3 of them, drawn per frame
    assert E.frame_task(side[0], 4, 7) == E.frame_task(side[0], 4, 7)                  # fixed by the frame's id
    # two copies of the 40 frames around the point where the goals part
    cp = [m for m in side if m.get("decision_copy")]
    assert [m["decision_copy"]["copy"] for m in cp] == [1, 2] and all(m["decision_copy"]["of"] == 0 and m["decision_copy"]["kind"] == kind for m in cp)
    a, b = cp[0]["decision_copy"]["frames_of"]
    k = a + cp[0]["decision_copy"]["decision_frame"]
    p0 = ph[ep == 0]
    if kind == "fork":   # the first drilling frame with the trigger on (the drill is past midline − 2 mm from its first cut)
        assert k == int(np.flatnonzero((p0 == 4) & (st[ep == 0][:, 14] > 0.5))[0])
    else:                # the last drilling frame
        assert k == int(np.flatnonzero(p0 == 4)[-1])
    assert a == max(0, k - 20) and b == min(int((ep == 0).sum()), k + 20) and cp[0]["n_frames"] == b - a
    for m in cp:
        np.testing.assert_array_equal(st[ep == m["episode_index"]], st[ep == 0][a:b])   # the same frames again …
    t1, t2 = texts[ep == cp[0]["episode_index"]], texts[ep == cp[1]["episode_index"]]
    assert (t1 != t2).any()                                                            # … with their own wordings and dropouts
    exp = json.loads((out / "meta/spine_il/export.json").read_text())
    assert exp["instructions"]["name"] == "lang2" and exp["instructions"]["cond_dropout"] == 0.3
    assert exp["instructions"]["decision_copies"] == {"copies": 2, "frames": 20, "episodes": 2}
    assert json.loads((out / "meta/spine_il/splits.json").read_text())["decision_copies"] == [1, 2]
    rep = E.validate(out, "local/test")
    assert rep["action_is_next_state"] and rep["progress_is_the_target_frames"] and rep["progress"]["never_falls"]
    sys.path.insert(0, str(HERE.parents[1] / "inference"))
    import end_level as EL                                                             # the copies are not recordings
    assert sorted(EL.load(out)) == [0]


def test_lang2_takes_the_level_from_the_cuts(world, monkeypatch):
    """10-15: a lang2 export labelled every recording L4 (the level was looked up for lang1 only) — an L3 recording got L4's
    words and L4's midline. Under lang2 too the level comes from the cuts, and the extent from that level's midline."""
    _with_cut_centres(world, 50.0)
    monkeypatch.setattr(E.INS, "level_of", lambda cc, *a, **k: ("L5", {"share": {"L5": 1.0}, "surface_mm_median": 0.0}))
    monkeypatch.setitem(E.INS.MIDLINE_X_MM, "L5", 60.0)       # 10 mm past L5's midline: clearly contralateral (by L4's 21.9: ipsilateral)
    monkeypatch.setitem(E.INS.EXTENT_X_MM, "L5", 58.0)
    out = world["tmp"] / "ds_lang2_level"
    E.main(["--out", str(out), "--sessions", str(world["sessions"]), "--images", "none", "--repo-id", "local/test", "--include", "hand",
            "--fps", "15", "--instructions", "lang2"])
    ds, hf, side = _load(out)
    assert side[0]["goal"]["level"] == "L5" and side[0]["goal"]["extent"] == "contralateral"
    assert side[0]["goal"]["past_midline_mm"] == pytest.approx(10.0)
    ph, ti = np.asarray(hf["phase.index"]).reshape(-1), np.asarray(hf["task_index"]).reshape(-1)
    tasks = {int(v["task_index"]): str(k) for k, v in ds.meta.tasks.iterrows()}
    assert set(tasks[int(i)] for i in ti[ph == 3]) <= set(E.INS.wordings(3, "L5"))
    drill = set(tasks[int(i)] for i in ti[ph == 4])
    assert drill and drill <= set(E.INS.wordings(4, "L5", "contralateral"))


def test_lang2_options_are_checked(world):
    for bad in (["--instructions", "lang1", "--cond-dropout", "0.2"], ["--instructions", "lang2", "--cond-dropout", "1.5"],
                ["--decision-copies", "3"], ["--instructions", "lang2", "--decision-copies", "-1"]):
        with pytest.raises(SystemExit):
            E.main(["--out", str(world["tmp"] / "bad"), "--sessions", str(world["sessions"]), "--images", "none", *bad])
