"""render API hardening (security audit finding B3): a file-path allowlist for path-shaped inputs, and a bearer
token for every /v1/* request. Stdlib only — this module is imported by `normalize`-adjacent code that must keep
working without the `[server]` extra (fastapi et al.), so the FastAPI-specific glue lives in `api/auth.py`.

Both guard the same host: the render API runs on a shared Lenovo box (other team users log into the same
machine); a port-level uid block is not a substitute for either check.
"""

from __future__ import annotations

import os
import secrets
import time
from pathlib import Path

# ── path allowlist (finding B2: arbitrary file read via a path-shaped `input`) ──

ALLOWED_ROOTS_ENV = "RENDER_API_ALLOWED_ROOTS"  # os.pathsep-separated list of directories
# The robot session's two inputs: recorded rig sessions, and run/trajectory outputs.
DEFAULT_ALLOWED_ROOTS = ("~/il-robot/viewer-data/sessions", "~/il-robot/runs")


class PathNotAllowedError(ValueError):
    """A path-shaped `input` resolved outside the configured allowed roots, or is not a regular file. A ValueError
    subclass so it flows through the API's existing `except ValueError` → HTTPException(400) handling unchanged."""


def _normalize_roots(paths) -> list[Path]:
    """`~`-expanded, resolved (symlinks followed, `..` collapsed) absolute paths. A root that does not exist yet is
    kept, lexically normalised, so it can still be named in a rejection message — nothing can resolve inside a
    root that is not there. Applied to every `roots` value, default or caller-supplied: an un-resolved root (e.g.
    a bare `tmp_path / "root"` in a test, or on macOS a `/var/...` path that is itself a symlink to
    `/private/var/...`) would otherwise never match a *resolved* target path even when it is the same directory."""
    out = []
    for p in paths:
        if not p:
            continue
        expanded = Path(p).expanduser()
        try:
            out.append(expanded.resolve(strict=False))
        except OSError:
            out.append(expanded)
    return out


def allowed_roots() -> list[Path]:
    """The configured allowed roots: `RENDER_API_ALLOWED_ROOTS` (os.pathsep-separated) if set, else
    DEFAULT_ALLOWED_ROOTS — expanded and resolved for the current (service) user."""
    raw = os.environ.get(ALLOWED_ROOTS_ENV)
    parts = raw.split(os.pathsep) if raw else list(DEFAULT_ALLOWED_ROOTS)
    return _normalize_roots(parts)


def resolve_allowed_path(src: str, roots: list[Path] | None = None) -> Path:
    """A path-shaped `input` resolved to a real, existing, regular file inside one of `roots` (default
    `allowed_roots()`; also normalized if given — see `_normalize_roots`).

    `Path.resolve(strict=True)` both follows symlinks and collapses `..` — so a symlink inside an allowed root
    that targets a file outside it, or a `..`-laden path that lexically looks like it stays inside, are both
    caught the same way: by checking the *final real path*'s ancestry, never the string. Anything else (missing,
    a directory, a fifo/socket, or simply outside every root) raises PathNotAllowedError naming the allowed roots.
    """
    roots = allowed_roots() if roots is None else _normalize_roots(roots)
    try:
        resolved = Path(src).expanduser().resolve(strict=True)
    except OSError as e:
        raise PathNotAllowedError(f"path not found: {src!r}") from e
    inside = any(resolved == root or root in resolved.parents for root in roots)
    if not inside:
        names = ", ".join(str(r) for r in roots) or "(none configured)"
        raise PathNotAllowedError(f"path {src!r} is outside the allowed roots ({names}); "
                                  f"set {ALLOWED_ROOTS_ENV} to override")
    if resolved.is_dir():
        raise PathNotAllowedError(f"path {src!r} is a directory, not a file")
    if not resolved.is_file():
        raise PathNotAllowedError(f"path {src!r} is not a regular file")
    return resolved


# ── bearer token auth (finding B1: no authentication at all) ──

TOKEN_ENV = "RENDER_API_TOKEN"
CONFIG_DIRNAME = "spinearm-render-api"  # ~/.config/spinearm-render-api/token


def token_file_path() -> Path:
    return Path.home() / ".config" / CONFIG_DIRNAME / "token"


def get_or_create_token() -> str:
    """RENDER_API_TOKEN if set, else the token at token_file_path() — created (0600, parents 0700) with a fresh
    secrets.token_urlsafe(32) on first call, read back on every call after. Server and client(s) call this the
    same way, so whichever runs first creates the shared secret and the other just reads it; no distribution step.
    O_CREAT|O_EXCL makes the create-or-read race safe across concurrent first starts."""
    env = os.environ.get(TOKEN_ENV)
    if env:
        return env
    path = token_file_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        # Another process may have created it and not written yet (concurrent first start) — give it a moment.
        # A file that stays empty means a create was interrupted: fail loudly instead of running with token "".
        for _ in range(20):
            text = path.read_text().strip()
            if text:
                return text
            time.sleep(0.05)
        raise RuntimeError(f"token file {path} is empty (interrupted create?) — delete it and restart the service")
    token = secrets.token_urlsafe(32)
    with os.fdopen(fd, "w") as f:
        f.write(token)
    return token


def check_bearer(authorization: str | None) -> bool:
    """True if `authorization` is `Bearer <the current token>` (constant-time compare). Missing header, wrong
    scheme, empty token and a wrong token are all just False — the caller (api/auth.py) turns that into a 401
    that does not distinguish why."""
    if not authorization:
        return False
    scheme, _, provided = authorization.partition(" ")
    if scheme.lower() != "bearer" or not provided:
        return False
    return secrets.compare_digest(provided, get_or_create_token())
