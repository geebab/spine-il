"""Replay window of a hand recording: from = first bone cut − 2 s, on the recording's own axis (s after the first valid
pose row of either tool — recording.load_recording's t_start_s origin)."""
import json, math, sys
def window(path, lead_s=2.0):
    t0 = tcut = tbtn = None   # tbtn: first drill trigger — the fallback for policy recordings (no cut rows)
    with open(path) as fh:
        for line in fh:
            if '"k":"f"' in line[:12]:
                r = json.loads(line)
                if not r.get("lost") and r.get("tExp") is not None and r.get("pos") is not None:
                    t0 = r["tExp"] if t0 is None else min(t0, r["tExp"])
                    if r.get("tool") == "drill" and r.get("btn"):
                        tbtn = r["tExp"] if tbtn is None else min(tbtn, r["tExp"])
            elif '"k":"cut"' in line[:14]:
                r = json.loads(line)
                if r.get("target", "bone") == "bone" and r.get("tExp") is not None:
                    tcut = r["tExp"] if tcut is None else min(tcut, r["tExp"])
    tcut = tcut if tcut is not None else tbtn
    if tcut is None:
        return None
    return max(0, round((tcut - t0) / 1000.0 - lead_s))   # rounded, as the 09-28 windows
if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p[-9:-6], window(p))
