"""Recording gate: the header facts a consumer must see before trusting a render (copied into job metadata).

Nothing here refuses a recording — the render still runs — but every result carries the recording's timing spec,
FE / tracker settings fingerprints, dataset-baseline check and operator overrides, plus warnings for the cases the
schema says must not be mixed or cannot be reproduced (sessionRecordSchemaV2.ts: timingSpec, fe.cfgHash,
tracker.cfgHash, baseline, baselineOverride, t0Override; replay.carvedBeforeStart).
"""

from __future__ import annotations


def recording_gate(h: dict) -> dict:
    fe = h.get("fe") or {}
    tracker = h.get("tracker") or {}
    replay = h.get("replay") or {}
    reg = h.get("registration") or {}
    gate = {
        "timingSpec": h.get("timingSpec"),
        "recorderVersion": h.get("recorderVersion"),
        "feCfgHash": fe.get("cfgHash"),
        "trackerCfgHash": tracker.get("cfgHash"),
        "baseline": h.get("baseline"),
        "baselineOverride": bool(h.get("baselineOverride")),
        "t0Override": bool(h.get("t0Override")),
        "registration": {"method": reg.get("method"), "dynamic": bool(reg.get("dynamic"))},
        "replay": replay or None,
        "meshSha256": h.get("meshSha256"),
        "build": h.get("build"),
    }
    w: list[str] = []
    spec = h.get("timingSpec")
    if spec == "v2-core":
        w.append("timingSpec v2-core: pilot recording without T0 stamps (no tExp/drawn) — keep it out of v2-t0 "
                 "datasets; it cannot be replayed (only simulated, as a diagnostic)")
    elif spec is None:
        w.append("no timingSpec: recorder older than rec-2.2 — it cannot be replayed (only simulated, as a diagnostic)")
    if gate["t0Override"]:
        w.append("t0Override: recording was started although the tracker sent no T0 stamps (tExp/cfgHash)")
    if gate["baselineOverride"]:
        w.append("baselineOverride: recording was started despite a dataset-baseline mismatch")
    base = h.get("baseline")
    if isinstance(base, dict) and base.get("match") is False:
        w.append("baseline.match is false: FE/tracker settings differed from the dataset baseline at start")
    carved = replay.get("carvedBeforeStart") or {}
    if carved.get("bone") or carved.get("soft"):
        which = "+".join(k for k in ("bone", "soft") if carved.get(k))
        w.append(f"carvedBeforeStart ({which}): the scene was already carved when recording started — the file's cut "
                 "rows cannot rebuild that starting state")
    # Tool identity (audit H5): the relay merges toolCalib from its own arguments, the tracker reports the model it
    # actually loaded — they must agree, and a second endoscope (robot rig block) must not be in the stream.
    tools = ((tracker.get("cfg") or {}).get("tools") or {})
    for tool, calib in (h.get("toolCalib") or {}).items():
        want = (calib or {}).get("sha256_12")
        have = (((tools.get(tool) or {}).get("calib") or {}).get("sha12"))
        if want and have and want != have:
            w.append(f"toolCalib.{tool} sha {want} ≠ tracker model {have}: the relay's calibration is not the one the "
                     "tracker tracked with")
    if "endo2" in tools:
        w.append("tracker cfg has endo2 (robot-rig block): the 'endo' rows may be the robot marker, not the hand tool")
    gate["warnings"] = w
    return gate
