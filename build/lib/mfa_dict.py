"""MFA Brazilian-Portuguese pronunciation dictionary (CC BY 4.0) — loader.

`data/_mfa_dict/portuguese_brazil_mfa.dict` (MFA "portuguese_brazil_mfa"
v2.0.0a, 34,978 lines: `word<TAB>space-separated phones`, no stress marks,
`x` for every non-tap r, palatal `c`/`ɟ` for fronted k/g). Re-downloaded from
the pinned release if missing; the sha256 is checked either way.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MFA_PATH = REPO_ROOT / "data" / "_mfa_dict" / "portuguese_brazil_mfa.dict"
MFA_URL = ("https://raw.githubusercontent.com/MontrealCorpusTools/mfa-models/main/"
           "dictionary/portuguese/brazil_mfa/portuguese_brazil_mfa.dict")
MFA_SHA256 = "6ec51b884952f1076ad4a3ae17ec7e0c274b8df0dbd681b32e31ee78118f1861"


class MfaDictError(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"ERROR (mfa_dict): {msg}")


def ensure_mfa_dict(path: Path = MFA_PATH) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = httpx.get(MFA_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60,
                      follow_redirects=True)
        r.raise_for_status()
        path.write_bytes(r.content)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MFA_SHA256:
        raise MfaDictError(f"{path} sha256 {digest[:12]}… ≠ pinned {MFA_SHA256[:12]}…")
    return path


def load_mfa_dict(path: Path = MFA_PATH) -> dict[str, list[list[str]]]:
    """{lower-case word: [phone list, …]} (all pronunciation variants)."""
    ensure_mfa_dict(path)
    out: dict[str, list[list[str]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) != 2 or parts[0].startswith(("<", "[")):
            continue
        out.setdefault(parts[0].lower(), []).append(parts[1].split())
    return out
