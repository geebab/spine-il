#!/usr/bin/env python3
"""Where the policy server is — kept outside the repository, one small file per fact.

    ~/.config/ube/policy.url      the address (http(s)://host:port) — not a secret
    ~/.config/ube/policy.token    the bearer token — chmod 600, never on a command line, in a document or in the repository
    ~/.config/ube/policy.ca       optional: the server's own certificate (PEM), for https with a certificate no authority signed

A second model on another address: policy.<NAME>.url / .token / .ca, NAME = the model's tag (e.g. groot_h). A name without
a token file of its own uses policy.token (two models behind one server process share the token).

    python endpoint.py set URL [--name NAME]     remember the address
    python endpoint.py token [--name NAME]       type or paste the token — it is not shown and not kept in the shell history;
                                                 run this in your own terminal (or pipe a file: endpoint.py token < file)
    python endpoint.py show                      what is configured (a token is never printed, only that it is there)
    python endpoint.py env [NAME]                EP_URL= EP_TOKEN_FILE= EP_CA_FILE= lines for a shell script's eval

The run scripts (live/remote_intake.sh, live/run_remote_base.sh, live/dry_remote.sh) read these files.
"""
from __future__ import annotations

import argparse
import getpass
import ipaddress
import os
import re
import stat
import sys
from pathlib import Path
from urllib.parse import urlsplit

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")


def cfg_dir() -> Path:
    return Path(os.environ.get("UBE_CONFIG_DIR") or Path.home() / ".config/ube")


def paths(name: str | None = None) -> dict[str, Path]:
    if name and not NAME_RE.match(name):
        raise ValueError(f"endpoint name {name!r}: letters, digits, _ and - only")
    stem = "policy" + (f".{name}" if name else "")
    return {k: cfg_dir() / f"{stem}.{k}" for k in ("url", "token", "ca")}


def check_url(url: str) -> str:
    u = urlsplit(url.strip())
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError(f"the address must be http(s)://host[:port], got {url!r}")
    if u.username or u.password or u.query or u.fragment:
        raise ValueError("the address must not carry a user, a password or a query — the token goes in its own file")
    return url.strip().rstrip("/")


def classify(url: str) -> dict:
    """What kind of address this is: where the server is seen from here and whether the wire is encrypted."""
    u = urlsplit(url); host = u.hostname or ""; port = u.port or (443 if u.scheme == "https" else 80)
    try:
        ip = ipaddress.ip_address(host)
        lan = any(ip in ipaddress.ip_network(n) for n in (("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16") if ip.version == 4 else ("fc00::/7", "fe80::/10")))
        where = ("this machine" if ip.is_loopback else
                 "tailnet" if ip.version == 4 and ip in ipaddress.ip_network("100.64.0.0/10") else
                 "private network" if lan else "internet")
    except ValueError:
        where = "this machine" if host == "localhost" else "tailnet" if host.endswith(".ts.net") else "name"
    enc = u.scheme == "https" or where in ("this machine", "tailnet")   # a tunnel's or the tailnet's own encryption
    return {"scheme": u.scheme, "host": host, "port": port, "where": where, "encrypted": enc,
            "note": {"this machine": "a port on this machine — an SSH tunnel (or a local test server)",
                     "tailnet": "a tailnet address — encrypted by Tailscale",
                     "private network": "a private network address",
                     "internet": "a public address", "name": "a host name"}[where]
                    + ("" if enc else "; plain HTTP: the token and the images are readable on the way — use an SSH tunnel or https")}


def load(name: str | None = None) -> dict:
    """{"name", "url" | None, "token_file" | None, "ca_file" | None} of the endpoint (None: the default one)."""
    p = paths(name); url = None
    if p["url"].is_file():
        lines = [ln.strip() for ln in p["url"].read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        url = check_url(lines[0]) if lines else None
    tok = p["token"] if p["token"].is_file() else paths(None)["token"] if name and paths(None)["token"].is_file() else None
    return {"name": name or "default", "url": url, "token_file": str(tok) if tok else None, "ca_file": str(p["ca"]) if p["ca"].is_file() else None}


def resolve(arg: str | None, name: str | None = None) -> dict:
    """An address or an endpoint's name (or nothing: the default endpoint) → load()'s dict with the address filled in."""
    if arg and arg.startswith(("http://", "https://")):
        d = load(name); d["url"] = check_url(arg); return d
    d = load(arg or name)
    if not d["url"]:
        raise ValueError(f"no address for endpoint {d['name']!r}: python inference/remote/endpoint.py set URL" + (f" --name {arg or name}" if (arg or name) else ""))
    return d


def save_url(url: str, name: str | None = None) -> Path:
    p = paths(name)["url"]; p.parent.mkdir(parents=True, exist_ok=True); os.chmod(p.parent, 0o700)
    tmp = p.with_suffix(p.suffix + ".tmp"); tmp.write_text(check_url(url) + "\n"); tmp.replace(p)
    return p


def write_token(token: str, name: str | None = None) -> Path:
    """The token into its file, private from the first byte (the file is created 600, never widened)."""
    tok = token.strip()
    if not tok or any(c.isspace() for c in tok):
        raise ValueError("the token must be one word without spaces")
    p = paths(name)["token"]; p.parent.mkdir(parents=True, exist_ok=True); os.chmod(p.parent, 0o700)
    tmp = p.with_suffix(p.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(tok + "\n")
    os.chmod(tmp, 0o600); tmp.replace(p)
    return p


def describe(name: str | None = None) -> dict:
    d = load(name); out = {"name": d["name"], "url": d["url"]}
    if d["url"]:
        out["address"] = classify(d["url"])["note"]
    if d["token_file"]:
        st = Path(d["token_file"]).stat()
        out["token"] = f"{d['token_file']} ({st.st_size} bytes, mode {stat.S_IMODE(st.st_mode):o})" + ("" if not stat.S_IMODE(st.st_mode) & 0o077 else " — READABLE BY OTHERS: chmod 600")
    else:
        out["token"] = "none"
    out["ca"] = d["ca_file"] or "none (the system's authorities)"
    return out


def names() -> list[str | None]:
    found = {None} if any(paths(None)[k].exists() for k in ("url", "token")) else set()
    for f in cfg_dir().glob("policy.*.url"):
        found.add(f.name[len("policy."):-len(".url")])
    return sorted(found, key=lambda x: (x is not None, x or ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("set"); s.add_argument("url"); s.add_argument("--name", default=None)
    t = sub.add_parser("token"); t.add_argument("--name", default=None)
    sub.add_parser("show")
    e = sub.add_parser("env"); e.add_argument("target", nargs="?", default=None); e.add_argument("--name", default=None)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "set":
            p = save_url(a.url, a.name); print(f"{p}: {classify(check_url(a.url))['note']}")
        elif a.cmd == "token":
            tok = getpass.getpass("token (not shown): ") if sys.stdin.isatty() else sys.stdin.readline()
            p = write_token(tok, a.name); print(f"{p}: written, mode 600 ({p.stat().st_size} bytes)")
        elif a.cmd == "env":
            import shlex
            d = resolve(a.target, a.name)
            print(f"EP_NAME={shlex.quote(d['name'])}; EP_URL={shlex.quote(d['url'])}; EP_TOKEN_FILE={shlex.quote(d['token_file'] or '')}; EP_CA_FILE={shlex.quote(d['ca_file'] or '')}")
        else:
            ns = names()
            if not ns:
                print(f"nothing configured in {cfg_dir()} — python endpoint.py set URL, then python endpoint.py token")
            for n in ns:
                d = describe(n); print(f"{d['name']}: {d['url'] or 'no address'}\n  address: {d.get('address', '—')}\n  token:   {d['token']}\n  ca:      {d['ca']}")
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
