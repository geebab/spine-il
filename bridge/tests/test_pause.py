"""The pause switch of batch jobs (pause.py): a fresh pause file makes a job wait between two frames, a stale one or none
does not. No browser — the job-level behaviour (a paused job renders the frames an unpaused one renders) is checked
against the running service, see the bridge README."""
from __future__ import annotations

import asyncio
import json
import os
import time

import pytest

from spine_render import pause


@pytest.fixture
def switch(tmp_path, monkeypatch):
    f = tmp_path / "ube-render" / "pause"
    monkeypatch.setenv("UBE_RENDER_PAUSE_FILE", str(f))
    f.parent.mkdir()
    return f


def test_no_file_no_pause(switch):
    assert pause.pause_file() == switch
    assert pause.pause_reason() is None
    assert asyncio.run(pause.wait_clear()) == 0.0


def test_the_default_file_is_in_the_runtime_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("UBE_RENDER_PAUSE_FILE", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    assert pause.pause_file() == tmp_path / "ube-render" / "pause"


def test_a_fresh_file_pauses_and_says_why(switch):
    switch.write_text(json.dumps({"reason": "the right arm is HOLD"}))
    assert pause.pause_reason() == "the right arm is HOLD"
    switch.write_text("not json")
    assert pause.pause_reason() == "pause file"
    switch.write_text("[1, 2]")
    assert pause.pause_reason() == "pause file"


def test_a_stale_file_is_a_dead_owner(switch):
    switch.write_text(json.dumps({"reason": "robot"}))
    old = time.time() - pause.FRESH_S - 5
    os.utime(switch, (old, old))
    assert pause.pause_reason() is None
    assert pause.pause_reason(now=old + 1) == "robot"       # … and was a pause while its owner still touched it


def test_wait_clear_waits_until_the_file_is_gone(switch):
    switch.write_text(json.dumps({"reason": "robot job r0"}))
    told = []

    async def go():
        async def lift():
            await asyncio.sleep(0.25)
            switch.write_text(json.dumps({"reason": "the left arm is MOVING"}))
            await asyncio.sleep(0.25)
            switch.unlink()
        t = asyncio.create_task(lift())
        waited = await pause.wait_clear(lambda reason, since: told.append((reason, since)), poll_s=0.05)
        await t
        return waited

    t0 = time.time()
    waited = asyncio.run(go())
    assert 0.45 <= waited < 2.0
    assert [r for r, _ in told] == ["robot job r0", "the left arm is MOVING", None]      # each reason once, then the end
    assert all(abs(since - t0) < 0.5 for _, since in told)


def test_wait_clear_ends_when_the_owner_stops_touching_the_file(switch, monkeypatch):
    monkeypatch.setattr(pause, "FRESH_S", 0.3)
    switch.write_text("{}")
    waited = asyncio.run(pause.wait_clear(poll_s=0.05))
    assert 0.25 <= waited < 2.0
    assert switch.exists()                                   # nobody removed it: it went stale
