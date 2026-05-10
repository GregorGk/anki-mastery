"""Stage 8 / Step 4 — ElevenLabs alias-dictionary upload.

Reads `data/_pronunciation_aliases.tsv` (deduped by pt), verifies
conflict-free, optionally runs a 1-rule sandbox smoke test, then POSTs
to `/v1/pronunciation-dictionaries/add-from-rules` and persists
`dictionary_id` + `version_id` to `data/_audio_dictionary_meta.tsv`.

Usage:
    # Smoke test (1 rule sandbox dict to verify SDK accepts our schema):
    .venv/bin/python build/08_4_upload_dictionary.py --smoke-test

    # Real upload:
    .venv/bin/python build/08_4_upload_dictionary.py --confirm
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
ALIASES_PATH = DATA_DIR / "_pronunciation_aliases.tsv"
CONFLICTS_PATH = DATA_DIR / "_pronunciation_alias_conflicts.tsv"
META_PATH = DATA_DIR / "_audio_dictionary_meta.tsv"

NAME_PREFIX = "anki-bp-stage-8"


def _next_version_n() -> int:
    """Inspect existing meta file to compute next vN."""
    if not META_PATH.exists():
        return 1
    rows = read_tsv(META_PATH)
    n = 1
    for r in rows:
        name = r.get("dict_name", "")
        # Extract trailing -vN
        if "-v" in name:
            try:
                v = int(name.rsplit("-v", 1)[1])
                if v >= n:
                    n = v + 1
            except ValueError:
                pass
    return n


def _build_rules(alias_rows: list[dict]) -> list[dict]:
    """Convert alias rows to ElevenLabs alias-rule dicts."""
    return [
        {
            "string_to_replace": r["pt"],
            "type": "alias",
            "alias": r["pt_respelling"],
            "case_sensitive": True,
            "word_boundaries": True,
        }
        for r in alias_rows
        if r.get("pt") and r.get("pt_respelling") and r["pt"] != r["pt_respelling"]
    ]


def _md5(rows: list[dict]) -> str:
    """Stable hash of (pt, pt_respelling) pairs for traceability."""
    h = hashlib.md5()
    for r in sorted(rows, key=lambda x: x.get("pt", "")):
        h.update((r.get("pt", "") + "→" + r.get("pt_respelling", "")).encode("utf-8"))
    return h.hexdigest()[:12]


def _post_dictionary(name: str, rules: list[dict]) -> dict:
    """Call ElevenLabs SDK to create a pronunciation dictionary from rules.

    Returns dict with: id, version_id, version_rules_num.
    """
    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    resp = client.pronunciation_dictionaries.create_from_rules(
        rules=rules,
        name=name,
        description=f"Stage 8 alias dictionary for Anki BP deck. {len(rules)} rules.",
        workspace_access="admin",
    )
    return {
        "id": getattr(resp, "id", None),
        "version_id": getattr(resp, "version_id", None),
        "version_rules_num": getattr(resp, "version_rules_num", None),
    }


def cmd_smoke_test() -> int:
    """Upload a 1-rule sandbox dict to verify the SDK accepts our schema."""
    if not os.environ.get("ELEVENLABS_API_KEY"):
        print("ERROR: ELEVENLABS_API_KEY not set in environment.", file=sys.stderr)
        return 1
    rules = [{
        "string_to_replace": "smoketest_animal",
        "type": "alias",
        "alias": "smoketest_animau",
        "case_sensitive": True,
        "word_boundaries": True,
    }]
    name = f"{NAME_PREFIX}-smoke-{int(time.time())}"
    print(f"Smoke uploading 1-rule sandbox dict name={name}...")
    info = _post_dictionary(name, rules)
    print(f"  ✓ id={info['id']}")
    print(f"  ✓ version_id={info['version_id']}")
    print(f"  ✓ version_rules_num={info['version_rules_num']}")
    print()
    print("Schema verified: case_sensitive + word_boundaries fields accepted as documented.")
    return 0


def cmd_upload(confirm: bool) -> int:
    if not confirm:
        print("ERROR: --confirm required for real upload.", file=sys.stderr)
        return 1
    if not os.environ.get("ELEVENLABS_API_KEY"):
        print("ERROR: ELEVENLABS_API_KEY not set in environment.", file=sys.stderr)
        return 1
    if not ALIASES_PATH.exists():
        print(f"ERROR: aliases file not found at {ALIASES_PATH}.", file=sys.stderr)
        return 1

    aliases = read_tsv(ALIASES_PATH)
    if not aliases:
        print("ERROR: aliases file is empty. Nothing to upload.", file=sys.stderr)
        return 1

    # Verify no conflicts
    if CONFLICTS_PATH.exists():
        conflicts = read_tsv(CONFLICTS_PATH)
        if conflicts:
            print(f"ERROR: {len(conflicts)} conflict(s) at {CONFLICTS_PATH}. Resolve before upload.", file=sys.stderr)
            for c in conflicts[:10]:
                print(f"  {c.get('pt')}: {c.get('existing_respelling')} vs {c.get('new_respelling')}", file=sys.stderr)
            return 2

    rules = _build_rules(aliases)
    if not rules:
        print("ERROR: no usable rules built from aliases.tsv.", file=sys.stderr)
        return 1

    n = _next_version_n()
    name = f"{NAME_PREFIX}-v{n}"
    aliases_md5 = _md5(aliases)
    print(f"Uploading {len(rules)} alias rules as dict name={name} (md5={aliases_md5})...")

    info = _post_dictionary(name, rules)
    print(f"  ✓ id={info['id']}")
    print(f"  ✓ version_id={info['version_id']}")
    print(f"  ✓ version_rules_num={info['version_rules_num']}")

    # Persist
    new_row = {
        "dictionary_id": info["id"] or "",
        "version_id": info["version_id"] or "",
        "rule_count": str(info["version_rules_num"] or len(rules)),
        "dict_name": name,
        "source_aliases_md5": aliases_md5,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "notes": f"Stage 8 alias dictionary v{n}",
    }
    existing = read_tsv(META_PATH) if META_PATH.exists() else []
    existing.append(new_row)
    fns = ["dictionary_id", "version_id", "rule_count", "dict_name", "source_aliases_md5", "created_at", "notes"]
    write_tsv(META_PATH, existing, fieldnames=fns)
    print(f"  ✓ recorded in {META_PATH}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--smoke-test", action="store_true",
                        help="Upload 1-rule sandbox dict to verify SDK accepts schema.")
    parser.add_argument("--confirm", action="store_true",
                        help="Required for the real upload.")
    args = parser.parse_args()

    if args.smoke_test:
        return cmd_smoke_test()
    return cmd_upload(args.confirm)


if __name__ == "__main__":
    raise SystemExit(main())
