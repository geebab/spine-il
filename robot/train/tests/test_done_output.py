"""train/lerobot-train-rel.py --done-output: the "drilling is over" target, the phase input without the answer, the
statistics with the 14th value and the start from a checkpoint that has 13."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("train_rel", HERE.parent / "lerobot-train-rel.py")
TR = importlib.util.module_from_spec(spec); sys.modules["train_rel"] = TR; spec.loader.exec_module(TR)


def test_done_is_everything_after_the_last_cut():
    ph = torch.tensor([[3, 3, 4, 4, 5, 5, 6]])
    assert TR.done_from_phase(ph).tolist() == [[0, 0, 0, 0, 1, 1, 1]]


def test_the_phase_input_no_longer_tells_drilling_from_withdrawing():
    st = torch.zeros(3, 21)
    st[0, 15 + 2] = 1; st[1, 15 + 3] = 1; st[2, 15 + 4] = 1          # phases 3, 4, 5
    st[:, :15] = torch.arange(15.0)
    out = TR.merge_phase_input(st)
    assert out[0, 15:].tolist() == [0, 0, 1, 0, 0, 0]               # the approach keeps its own bit
    assert out[1, 15:].tolist() == out[2, 15:].tolist() == [0, 0, 0, 1, 0, 0]
    assert torch.equal(out[:, :15], st[:, :15]) and st[2, 19] == 1  # the tools untouched, the input not changed in place
    plain = torch.arange(15.0)[None]
    assert TR.merge_phase_input(plain) is plain                     # a state without the one-hot


def _dataset(tmp_path, phases):
    import pyarrow as pa
    import pyarrow.parquet as pq
    n = len(phases)
    st = np.zeros((n, 21), np.float32); st[:, 0] = np.arange(n); st[:, 3] = 1; st[:, 10] = 1    # endo x moves 1 mm a frame
    st[:, 14] = (np.asarray(phases) == 4)                                                       # trigger on while drilling
    ac = np.r_[st[1:, :15], st[-1:, :15]]                                                       # action = the next state
    d = tmp_path / "data/chunk-000"; d.mkdir(parents=True)
    pq.write_table(pa.table({"observation.state": st.tolist(), "action": ac.tolist(), "phase.index": [[int(p)] for p in phases],
                             "episode_index": [0] * n, "frame_index": list(range(n))}), d / "file-000.parquet")
    return tmp_path


def test_chunk_stats_carry_the_done_column(tmp_path):
    root = _dataset(tmp_path, [3, 4, 4, 4, 5, 5])
    cs = TR.chunk_stats(root, 3, per_step=True, done=True)
    assert cs["mean"].shape == (14,) and cs["step_mean"].shape == (3, 14)
    # frame i, step j leads to frame i+1+j (clamped at 5): done where that frame's phase is 5
    #   i=0: frames 1,2,3 → 0,0,0   i=1: 2,3,4 → 0,0,1   i=2: 3,4,5 → 0,1,1   i=3: 4,5,5 → 1,1,1   i=4, 5: 1,1,1
    assert cs["step_mean"][:, 13].tolist() == pytest.approx([3 / 6, 4 / 6, 5 / 6])
    assert cs["mean"][13] == pytest.approx(12 / 18) and cs["max"][13] == 1.0 and cs["min"][13] == 0.0
    plain = TR.chunk_stats(root, 3, per_step=True)
    assert plain["mean"].shape == (13,) and np.allclose(plain["mean"], cs["mean"][:13])


def test_a_checkpoint_with_13_actions_starts_a_policy_with_14(tmp_path):
    from safetensors.torch import save_file

    class Net(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.body = torch.nn.Linear(4, 4); self.action_head = torch.nn.Linear(4, n); self.action_in = torch.nn.Linear(n, 4)

    old, new = Net(13), Net(14)
    save_file(old.state_dict(), str(tmp_path / "model.safetensors"))
    fresh_row = new.action_head.weight[13].clone(); fresh_col = new.action_in.weight[:, 13].clone()
    info = TR.load_partial(new, tmp_path)
    assert info["same"] == 3 and sorted(info["grown"]) == ["action_head.bias", "action_head.weight", "action_in.weight"] and not info["kept_fresh"]
    assert torch.equal(new.body.weight, old.body.weight) and torch.equal(new.action_in.bias, old.action_in.bias)
    assert torch.equal(new.action_head.weight[:13], old.action_head.weight) and torch.equal(new.action_head.weight[13], fresh_row)
    assert torch.equal(new.action_in.weight[:, :13], old.action_in.weight) and torch.equal(new.action_in.weight[:, 13], fresh_col)


def test_progress_is_the_14th_value_as_it_stands_in_the_dataset(tmp_path):
    """--progress-output (user 10-09: the end output becomes a progress value): cut.progress of the frames the chunk steps
    lead to, not a flag made from the phase."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    prog = [0.0, 0.0, 0.25, 0.5, 1.0, 1.0]
    n = len(prog)
    st = np.zeros((n, 21), np.float32); st[:, 0] = np.arange(n); st[:, 3] = 1; st[:, 10] = 1
    ac = np.r_[st[1:, :15], st[-1:, :15]]
    d = tmp_path / "data/chunk-000"; d.mkdir(parents=True)
    pq.write_table(pa.table({"observation.state": st.tolist(), "action": ac.tolist(), "cut.progress": [[float(p)] for p in prog],
                             "phase.index": [[3], [4], [4], [4], [5], [5]], "episode_index": [0] * n, "frame_index": list(range(n))}),
                   d / "file-000.parquet")
    cs = TR.chunk_stats(tmp_path, 3, per_step=True, end="progress")
    assert cs["mean"].shape == (14,) and cs["step_mean"].shape == (3, 14)
    # frame i, step j leads to frame i+1+j (clamped at 5): i=0 → 0, .25, .5 · i=1 → .25, .5, 1 · i=2 → .5, 1, 1 · i=3, 4, 5 → 1, 1, 1
    want = np.array([[0, .25, .5], [.25, .5, 1], [.5, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
    assert cs["step_mean"][:, 13].tolist() == pytest.approx(want.mean(0).tolist())
    assert cs["min"][13] == 0.0 and cs["max"][13] == 1.0 and cs["mean"][13] == pytest.approx(want.mean())
    done = TR.chunk_stats(tmp_path, 3, end="done")                               # the flag from the same file is another column
    assert done["mean"][13] == pytest.approx(12 / 18) and not np.isclose(done["mean"][13], cs["mean"][13])
    assert np.allclose(done["mean"][:13], cs["mean"][:13])
    assert TR.END_NAME == {"done": "drill.done", "progress": "drill.progress"} and TR.END_COLUMN["progress"] == "cut.progress"
    col = torch.tensor([[0.2, 0.9, 1.0]])
    assert torch.equal(TR.end_target("progress", col), col) and TR.end_target("done", torch.tensor([[4, 5, 6]])).tolist() == [[0, 1, 1]]
