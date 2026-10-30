"""Thin client for the render API (stdlib HTTP for batch; `websockets` for stepping)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from .security import get_or_create_token

DEFAULT = "http://127.0.0.1:18780"


def _req(url: str, data: bytes | None = None, method: str = "GET", ctype: str = "application/json"):
    # Same env-var-or-file convention as the server (security.get_or_create_token): whichever of client/server
    # starts first creates ~/.config/spinearm-render-api/token, the other just reads it. No manual distribution.
    headers = {"content-type": ctype, "authorization": f"Bearer {get_or_create_token()}"}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.read()
    except urllib.error.HTTPError as e:  # keep the server's reason (FastAPI puts it in "detail")
        body = e.read().decode("utf-8", "replace")
        try:
            detail = json.loads(body).get("detail", body)
        except (ValueError, AttributeError):
            detail = body
        raise RuntimeError(f"{e.code}: {detail}") from None


def submit(src, options: dict | None = None, server: str = DEFAULT) -> dict:
    """src: path to a recording (.jsonl) / trajectory or delta (.json), or a dict."""
    if isinstance(src, (str, Path)) and Path(str(src)).exists():
        text = Path(src).read_text()
        try:
            payload = json.loads(text)
        except ValueError:
            payload = text  # JSONL recording
    else:
        payload = src
    body = json.dumps({"input": payload, "options": options or {}}).encode()
    return json.loads(_req(f"{server}/v1/jobs", body, "POST"))


def wait(job_id: str, server: str = DEFAULT, poll_s: float = 1.0, on_progress=None) -> dict:
    while True:
        job = json.loads(_req(f"{server}/v1/jobs/{job_id}"))
        if on_progress:
            on_progress(job)
        if job["status"] in ("done", "error", "cancelled"):
            return job
        time.sleep(poll_s)


def download(job_id: str, out: str | Path, server: str = DEFAULT) -> Path:
    out = Path(out)
    out.write_bytes(_req(f"{server}/v1/jobs/{job_id}/result.zip"))
    return out


def render(src, out: str | Path, options: dict | None = None, server: str = DEFAULT, on_progress=None) -> dict:
    job = submit(src, options, server)
    job = wait(job["id"], server, on_progress=on_progress)
    if job["status"] != "done":
        raise RuntimeError(job.get("error") or job["status"])
    download(job["id"], out, server)
    return job
