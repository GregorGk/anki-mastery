"""Stage 19 / Step 3 — IPA v2: validated São Paulo IPA for every word and sentence.

Reads (read-only):
    data/06-final.tsv            pt, pt_display, en_primary, pos, example_pt,
                                 target_word_used, ipa_word, ipa_example (Stage 05)
    data/_mfa_dict/…dict         MFA BP dictionary (soft cross-check)
    data/_manual_ipa.tsv         user overrides — NOT read here; derive_final
                                 applies them on top of this stage's output
Writes:
    data/_ipa_lexicon.tsv        one row per token type (+ heterophone occurrences)
    data/_ipa_v2.tsv             one row per sense: ipa_word, ipa_example, statuses
    reports/19_1_ipa_v2.html     status counts, Stage-05 → v2 diffs, conflicts,
                                 LLM decisions, Fable audit, conventions table
    audit/19_1_ipa_llm.jsonl     every LLM call (AnthropicClient audit)

Pipeline
  1. Tokenize every example (whitespace tokens, edge punctuation stripped —
     the Stage-05 convention) and every headword; align Stage-05 IPA tokens
     where the counts match.
  2. Per spelling: normalize each Stage-05 transcription to the São Paulo
     convention (build/lib/bp_ipa.py) and take the majority.
  3. Validate: symbol whitelist, one primary stress, stress vs the espeak-ng
     oracle, written-accent vowel quality, MFA soft check. Deterministic
     fixes (accent quality, oracle stress when unambiguous) → `fixed`.
  4. Everything else (`conflict`), heterophone occurrences (`homograph`, with
     sentence + gloss) and digit sentences → Claude Opus 5.5 (structured tool
     call, Batches API), re-validated; second failure → `unresolved`
     (falls back to the normalized Stage-05 form, flagged).
  5. Fable 5.1 independently transcribes a 200-token audit sample; the
     disagreement rate is reported.
  6. Assemble per-sense word + sentence IPA (clitic weak forms, cross-word
     coda-s voicing) and write the sidecars.

Usage:
    uv run python build/19_1_ipa_v2.py --analyze-only     # no LLM, prints stats
    uv run python build/19_1_ipa_v2.py --dry-run          # LLM cost estimate only
    uv run python build/19_1_ipa_v2.py --yes              # full run
"""
from __future__ import annotations

import argparse
import collections
import html
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env", override=True)

from build.lib import bp_ipa as B  # noqa: E402
from build.lib.mfa_dict import load_mfa_dict  # noqa: E402
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

DATA = REPO_ROOT / "data"
FINAL = DATA / "06-final.tsv"
LEXICON_OUT = DATA / "_ipa_lexicon.tsv"
IPA_V2_OUT = DATA / "_ipa_v2.tsv"
REPORT = REPO_ROOT / "reports" / "19_1_ipa_v2.html"
LLM_AUDIT = REPO_ROOT / "audit" / "19_1_ipa_llm.jsonl"
CACHE = REPO_ROOT / "build" / "cache" / "19_1_llm_decisions.jsonl"

ADJUDICATOR_MODEL = "claude-opus-5-5"
AUDITOR_MODEL = "claude-fable-5-1"
AUDIT_SAMPLE = 200
SEED = 19

EDGE_PUNCT = "\"'“”‘’«»()[]{}.,;:!?…—–-"
DIGIT_RE = re.compile(r"\d")

LEXICON_FIELDS = ["key", "context", "ipa", "status", "source", "stage5_votes", "n_occurrences",
                  "mfa_variants", "oracle", "issues", "llm_confidence", "notes"]
IPA_V2_FIELDS = ["sense_id", "pt", "ipa_word", "ipa_word_status", "ipa_example",
                 "ipa_example_status", "example_spoken", "changed_word", "changed_example",
                 "flags"]


class IpaV2Error(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"ERROR (19_1_ipa_v2): {msg}")


# ── tokenization ─────────────────────────────────────────────────────────────
def tokens_of(text: str) -> list[str]:
    """Whitespace tokens with edge punctuation stripped (hyphens/apostrophes
    inside a token are kept) — the Stage-05 token convention."""
    out = []
    for raw in text.split():
        tok = raw.strip(EDGE_PUNCT)
        if tok:
            out.append(tok)
    return out


def key_of(tok: str) -> str:
    return tok.lower()


# ── lexicon model ────────────────────────────────────────────────────────────
@dataclass
class Entry:
    key: str
    occurrences: list = field(default_factory=list)   # (sense_id, where, idx, raw_ipa)
    votes: collections.Counter = field(default_factory=collections.Counter)
    ipa: str = ""
    status: str = ""
    source: str = ""
    issues: list = field(default_factory=list)
    mfa: list = field(default_factory=list)
    confidence: str = ""
    notes: str = ""


def collect(rows: list[dict]) -> tuple[dict[str, Entry], dict]:
    lex: dict[str, Entry] = {}
    stats = collections.Counter()

    def entry(k: str) -> Entry:
        if k not in lex:
            lex[k] = Entry(k)
        return lex[k]

    for r in rows:
        sid = r["sense_id"]
        ex_toks = tokens_of(r["example_pt"])
        ipa_toks = r["ipa_example"].split()
        aligned = len(ex_toks) == len(ipa_toks)
        stats["sentences"] += 1
        stats["aligned" if aligned else "misaligned"] += 1
        for i, t in enumerate(ex_toks):
            e = entry(key_of(t))
            raw = ipa_toks[i] if aligned else ""
            e.occurrences.append((sid, "example", i, raw))
            if raw:
                e.votes[B.to_sp(raw)[0]] += 1
        head = tokens_of(r["pt"])
        if len(head) == 1:
            e = entry(key_of(head[0]))
            e.occurrences.append((sid, "headword", 0, r["ipa_word"]))
            if r["ipa_word"]:
                e.votes[B.to_sp(r["ipa_word"])[0]] += 1
        else:
            for i, t in enumerate(head):
                entry(key_of(t)).occurrences.append((sid, "headword_mwe", i, ""))
    return lex, stats


def _mfa_ok(e: Entry, ipa: str) -> tuple[bool | None, str]:
    """Soft MFA check: consonant skeleton + stressed-vowel quality."""
    if not e.mfa:
        return None, ""
    skel = B.consonant_skeleton(ipa)
    if not any(B.mfa_consonants(v) == skel for v in e.mfa):
        return False, "mfa:consonants"
    sv = B.stressed_vowel(ipa)
    if sv and B.base(sv) in ("e", "ɛ", "o", "ɔ") and not B.is_nasal(sv):
        qualities = set().union(*(B.mfa_stressed_vowels(v, ipa) for v in e.mfa))
        if qualities and B.base(sv) not in qualities:
            return False, f"mfa:stressed_vowel({B.base(sv)}∉{''.join(sorted(qualities))})"
    return True, ""


def classify(e: Entry, mfa: dict) -> None:
    """Assign ipa/status/issues deterministically (no LLM)."""
    k = e.key
    e.mfa = mfa.get(k, [])
    if DIGIT_RE.search(k):
        e.status, e.source = "digit", "llm_sentence"
        return
    if k in B.WEAK_FORMS:
        e.ipa, e.status, e.source = B.WEAK_FORMS[k], "ok", "weak_form_table"
        return
    if k in B.HETEROPHONES:
        e.status, e.source = "homograph", "llm_context"
        if e.votes:
            e.ipa = e.votes.most_common(1)[0][0]
        return
    if not e.votes:
        e.status, e.source = "conflict", "llm"
        e.issues = ["no_stage5_ipa"]
        return
    cand = e.votes.most_common(1)[0][0]
    issues = list(B.whitelist_issues(cand))
    _, raw_issues = B.to_sp(cand)
    issues += raw_issues
    fixed = False
    # orthographic final unstressed -e/-o/-a ⇒ i/u/ɐ (deterministic fix)
    fv = B.final_vowel_fix(k, cand)
    if fv:
        cand, fixed = fv, True
    # written accent ⇒ stressed vowel quality (deterministic fix)
    need = B.accent_issue(k, cand)
    if need:
        cand = B.fix_stressed_vowel(cand, need)
        fixed = True
    # stress vs espeak oracle (deterministic fix only when unambiguous)
    oracle = B.stress_matches_oracle(k, cand)
    if oracle is False or "stress:none" in issues:
        target = B.oracle_stress_target(k, cand)
        if target is not None:
            cand2, _ = B.to_sp(cand, stressed_override=target)
            if B.stress_matches_oracle(k, cand2):
                cand, fixed = cand2, True
                issues = [x for x in issues if not x.startswith("stress:")]
                oracle = True
        if oracle is False:
            issues.append("stress:oracle_mismatch")
    # written accent ⇒ stress must fall on that vowel (suicídio, evolui-type errors)
    if B.accent_stress_mismatch(k, cand):
        issues.append("stress:accent_mismatch")
    multi_variants = len(e.votes) > 1 and _stress_or_quality_disagree(e.votes)
    mfa_ok, mfa_issue = _mfa_ok(e, cand)
    if mfa_ok is False:
        issues.append(mfa_issue)
    if multi_variants:
        issues.append("stage5:variants_disagree")
    e.ipa = cand
    hard = [x for x in issues if not x.startswith("mfa:") and x != "stage5:variants_disagree"]
    soft_mfa = any(x.startswith("mfa:") for x in issues)
    if hard or (soft_mfa and multi_variants) or (soft_mfa and "stressed_vowel" in mfa_issue):
        e.status, e.source = "conflict", "llm"
    else:
        e.status = "fixed" if fixed else "ok"
        e.source = "stage5_majority"
    e.issues = issues


def _stress_or_quality_disagree(votes: collections.Counter) -> bool:
    """Do the normalized Stage-05 variants disagree on stress position or the
    stressed vowel's quality? (Pure notation differences don't count.)"""
    sigs = set()
    for ipa in votes:
        end, start, _ = B.stress_positions(ipa)
        sigs.add((end, B.base(B.stressed_vowel(ipa))))
    return len(sigs) > 1


def analyze(rows: list[dict]) -> dict[str, Entry]:
    mfa = load_mfa_dict()
    lex, stats = collect(rows)
    keys = list(lex)
    # warm the espeak cache in parallel (one subprocess per token type)
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(B.espeak_ipa, keys))
    for k in keys:
        classify(lex[k], mfa)
    by = collections.Counter(e.status for e in lex.values())
    iss = collections.Counter(i.split("(")[0] for e in lex.values() for i in e.issues)
    print(f"  sentences: {stats['sentences']:,} (aligned {stats['aligned']:,}, "
          f"misaligned {stats['misaligned']:,})")
    print(f"  token types: {len(lex):,}  status: {dict(by)}")
    print(f"  issues: {dict(iss.most_common())}")
    return lex


# ── LLM adjudication ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a Brazilian Portuguese phonetician writing broad IPA in one fixed São Paulo convention for an A1 learner's flashcards.

Convention — follow it exactly:
- Primary stress ˈ immediately before the stressed syllable (before its onset consonants). No secondary stress, no syllable dots, no slashes or brackets.
- Strong r (word-initial r, rr, r after n/l/s) = h. r closing a syllable (before a consonant or word-final) = tap ɾ. r in clusters (br, pr, tr, dr, cr, gr, fr, vr) and single r between vowels = ɾ.
- t and d before [i] = tʃ and dʒ, written without tie bars.
- Unstressed final e = i and final o = u (also before a final s); unstressed final a = ɐ.
- s closing a syllable = s (z before a voiced consonant); never ʃ.
- Final l = w. lh = ʎ, nh = ɲ.
- Nasal vowels ɐ̃ ẽ ĩ õ ũ; nasal diphthongs ɐ̃w̃ (ão), ẽj̃ (final -em/-ém), õj̃ (õe), ɐ̃j̃ (ãe), ũj̃ (muito = ˈmũj̃tu).
- Stressed open vs closed vowels matter: ɛ/e and ɔ/o. Use standard Brazilian (São Paulo) pronunciation; for verbs, use the vowel of the actual conjugated form.
- Allowed symbols only: p b t d k ɡ f v s z ʃ ʒ m n ɲ l ʎ ɾ h j w a e i o u ɐ ɛ ɔ, the combining tilde ̃ and ˈ.

You receive a Portuguese word (sometimes with the sentence it occurs in and its meaning), candidate transcriptions from other sources, and validator notes. The candidates may be wrong — decide the correct pronunciation yourself."""

DECIDE_TOOL = "decide_ipa"
DECIDE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["choice", "ipa", "stressed_syllable_from_end", "stressed_vowel",
                 "is_heterophone", "confidence", "rationale"],
    "properties": {
        "choice": {"type": "string", "enum": ["A", "B", "C", "D", "E", "F", "G", "corrected"],
                   "description": "letter of the candidate you adopt, or 'corrected'"},
        "ipa": {"type": "string", "description": "final IPA in the convention"},
        "stressed_syllable_from_end": {"type": "integer", "enum": [1, 2, 3, 4]},
        "stressed_vowel": {"type": "string",
                           "enum": ["a", "ɐ", "e", "ɛ", "i", "o", "ɔ", "u",
                                    "ɐ̃", "ẽ", "ĩ", "õ", "ũ"]},
        "is_heterophone": {"type": "boolean",
                           "description": "true if this spelling has 2+ pronunciations "
                                          "depending on meaning / part of speech"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "rationale": {"type": "string", "description": "one short sentence"},
    },
}
DIGIT_TOOL = "spell_out_sentence"
DIGIT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["example_spoken", "ipa_tokens", "confidence"],
    "properties": {
        "example_spoken": {"type": "string",
                           "description": "the sentence with every number written out in words, "
                                          "exactly as a Brazilian would read it aloud"},
        "ipa_tokens": {"type": "array", "items": {"type": "string"},
                       "description": "one IPA string per whitespace token of example_spoken "
                                      "(function words in their weak form)"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
}
AUDIT_TOOL = "transcribe_word"
AUDIT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["ipa", "confidence"],
    "properties": {"ipa": {"type": "string"},
                   "confidence": {"type": "string", "enum": ["high", "medium", "low"]}},
}
LLM_WORKERS = 8
EST_USD_PER_CALL = 0.03


def _prompt_hash(*parts: str) -> str:
    import hashlib
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def _load_cache() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["cache_key"]] = rec["decision"]
    return out


def run_llm(items: list[dict], *, model: str, system: str, tool: str, schema: dict,
            label: str) -> dict[str, dict]:
    """items: {id, user_message}. Concurrent sync tool calls, cached on disk
    (cache_key = id + prompt hash) so re-runs are free and resumable."""
    from build.lib.llm import AnthropicClient
    import threading

    cache = _load_cache()
    out: dict[str, dict] = {}
    todo = []
    for it in items:
        ck = f"{model}|{tool}|{it['id']}|{_prompt_hash(system, it['user_message'])}"
        it["cache_key"] = ck
        if ck in cache:
            out[it["id"]] = cache[ck]
        else:
            todo.append(it)
    if not todo:
        return out
    print(f"  {label}: {len(todo)} calls to {model} ({len(items) - len(todo)} cached)", flush=True)
    client = AnthropicClient(model=model, audit_path=LLM_AUDIT)
    lock = threading.Lock()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    done = 0

    def one(it):
        nonlocal done
        try:
            d = client.call_tool(system=system, user_message=it["user_message"], tool_name=tool,
                                 tool_input_schema=schema, stage="19_1_ipa_v2",
                                 provenance_key=it["id"], strict=True, effort="high")
        except Exception as exc:  # noqa: BLE001
            d = {"_error": f"{type(exc).__name__}: {exc}"[:300]}
        with lock:
            out[it["id"]] = d
            done += 1
            if "_error" not in d:
                with CACHE.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"cache_key": it["cache_key"], "decision": d},
                                       ensure_ascii=False) + "\n")
            if done % 25 == 0 or done == len(todo):
                print(f"    {label}: {done}/{len(todo)}", flush=True)

    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as ex:
        list(ex.map(one, todo))
    return out


def _examples_for(e: Entry, rows_by_sid: dict, k: int = 3) -> list[str]:
    seen, out = set(), []
    for sid, where, _, _ in e.occurrences:
        if where == "example" and sid not in seen:
            seen.add(sid)
            out.append(rows_by_sid[sid]["example_pt"])
        if len(out) >= k:
            break
    return out


def _candidates(e: Entry) -> list[tuple[str, str]]:
    cands = [(ipa, f"Stage-5 transcription ({n} occurrence{'s' if n > 1 else ''})")
             for ipa, n in e.votes.most_common(4)]
    for v in e.mfa[:2]:
        cands.append((" ".join(v), "MFA pronunciation dictionary (phones, no stress; "
                                   "x = strong/coda r, c/ɟ = k/ɡ before front vowels)"))
    return cands[:7]


def _oracle_note(word: str) -> str:
    o = B.espeak_ipa(word)
    end, _, _ = B.stress_positions(o) if o else (None, None, 0)
    return (f"A rule-based stress checker expects primary stress on syllable {end} "
            f"counting from the end." if end else "")


def conflict_message(e: Entry, rows_by_sid: dict, *, retry_note: str = "") -> str:
    letters = "ABCDEFG"
    cands = _candidates(e)
    lines = [f"Word: {e.key}"]
    exs = _examples_for(e, rows_by_sid)
    if exs:
        lines.append("Used in: " + " | ".join(exs))
    lines.append("Candidates:")
    lines += [f"  {letters[i]}) {ipa}   [{src}]" for i, (ipa, src) in enumerate(cands)]
    if e.issues:
        lines.append("Validator notes: " + ", ".join(e.issues))
    note = _oracle_note(e.key)
    if note:
        lines.append(note)
    if retry_note:
        lines.append(retry_note)
    return "\n".join(lines)


def homograph_message(e: Entry, occ: tuple, rows_by_sid: dict, *, retry_note: str = "") -> str:
    sid, where, idx, _ = occ
    r = rows_by_sid[sid]
    letters = "ABCDEFG"
    cands = _candidates(e)
    lines = [f"Word: {e.key}"]
    if where == "example":
        lines.append(f"Sentence: {r['example_pt']}  (English: {r['example_en']})")
        lines.append(f"The word is token #{idx + 1} of the sentence.")
    else:
        lines.append(f"Headword sense: {r['pt_display']} — {r['pos']} — meaning: {r['en_primary']}")
    lines.append("This spelling can be pronounced differently depending on meaning / part of "
                 "speech; give the pronunciation for THIS use.")
    lines.append("Candidates:")
    lines += [f"  {letters[i]}) {ipa}   [{src}]" for i, (ipa, src) in enumerate(cands)]
    if retry_note:
        lines.append(retry_note)
    return "\n".join(lines)


def validate_word_ipa(word: str, ipa: str) -> tuple[str, list[str]]:
    sp, iss = B.to_sp(ipa)
    errs = list(B.whitelist_issues(sp)) + iss
    sp = B.final_vowel_fix(word, sp) or sp
    need = B.accent_issue(word, sp)
    if need:
        sp = B.fix_stressed_vowel(sp, need)
    if B.accent_stress_mismatch(word, sp):
        errs.append("stress_vs_written_accent")
    if B.stress_matches_oracle(word, sp) is False:
        errs.append("stress_vs_oracle")
    return sp, errs


def adjudicate(lex: dict[str, Entry], rows: list[dict]) -> dict[str, dict]:
    """Run Opus on conflicts + heterophone occurrences; returns per-occurrence
    decisions {f'{key}|{sid}|{where}|{idx}': {...}} for heterophones."""
    rows_by_sid = {r["sense_id"]: r for r in rows}
    conflicts = [e for e in lex.values() if e.status == "conflict"]
    homos = [(e, o) for e in lex.values() if e.status == "homograph"
             for o in e.occurrences if o[1] in ("example", "headword")]
    items = [{"id": f"conflict|{e.key}", "user_message": conflict_message(e, rows_by_sid)}
             for e in conflicts]
    items += [{"id": f"homograph|{e.key}|{o[0]}|{o[1]}|{o[2]}",
               "user_message": homograph_message(e, o, rows_by_sid)} for e, o in homos]
    dec = run_llm(items, model=ADJUDICATOR_MODEL, system=SYSTEM_PROMPT, tool=DECIDE_TOOL,
                  schema=DECIDE_SCHEMA, label="adjudication")
    # validate; one retry with the validator errors
    retry = []
    results: dict[str, tuple[str, list[str], dict]] = {}
    for it in items:
        d = dec.get(it["id"], {"_error": "missing"})
        word = it["id"].split("|")[1]
        if "_error" in d:
            results[it["id"]] = ("", ["llm_error"], d)
            continue
        sp, errs = validate_word_ipa(word, d["ipa"])
        results[it["id"]] = (sp, errs, d)
        if errs:
            note = (f"Your previous answer {d['ipa']} failed validation: {', '.join(errs)}. "
                    "Fix it if you agree; if you are certain it is right, repeat it with "
                    "confidence high.")
            e = lex[word]
            if it["id"].startswith("conflict|"):
                msg = conflict_message(e, rows_by_sid, retry_note=note)
            else:
                _, k, sid, where, idx = it["id"].split("|")
                occ = next(o for o in e.occurrences
                           if o[0] == sid and o[1] == where and str(o[2]) == idx)
                msg = homograph_message(e, occ, rows_by_sid, retry_note=note)
            retry.append({"id": it["id"] + "|retry", "user_message": msg})
    if retry:
        dec2 = run_llm(retry, model=ADJUDICATOR_MODEL, system=SYSTEM_PROMPT, tool=DECIDE_TOOL,
                       schema=DECIDE_SCHEMA, label="adjudication retry")
        for it in retry:
            base_id = it["id"][: -len("|retry")]
            d = dec2.get(it["id"], {"_error": "missing"})
            word = base_id.split("|")[1]
            if "_error" in d:
                continue
            sp, errs = validate_word_ipa(word, d["ipa"])
            only_oracle = errs == ["stress_vs_oracle"]
            if not errs or (only_oracle and d.get("confidence") == "high"):
                results[base_id] = (sp, ["llm_overrides_oracle"] if only_oracle else [], d)
            else:
                results[base_id] = (sp, errs, d)
    occ_decisions: dict[str, dict] = {}
    for iid, (sp, errs, d) in results.items():
        hard = [x for x in errs if x != "llm_overrides_oracle"]
        kind, word = iid.split("|")[:2]
        if kind == "conflict":
            e = lex[word]
            if hard or not sp:
                e.status, e.source = "unresolved", "stage5_normalized"
                e.notes = f"llm rejected: {hard}"
            else:
                e.ipa, e.status, e.source = sp, "llm", ADJUDICATOR_MODEL
                e.confidence = d.get("confidence", "")
                e.notes = "; ".join(x for x in [d.get("rationale", ""), *errs] if x)[:300]
                if d.get("is_heterophone"):
                    e.notes += " [LLM: heterophone]"
        else:
            occ_decisions[iid.split("|", 1)[1]] = {
                "ipa": sp if not hard else "", "errors": hard, "confidence": d.get("confidence", ""),
                "rationale": d.get("rationale", ""), "flags": errs}
    return occ_decisions


def spell_out_digits(rows: list[dict]) -> dict[str, dict]:
    items = []
    for r in rows:
        if any(DIGIT_RE.search(t) for t in tokens_of(r["example_pt"])):
            items.append({"id": f"digits|{r['sense_id']}",
                          "user_message": (f"Sentence: {r['example_pt']}\n"
                                           f"English: {r['example_en']}\n"
                                           "Write the sentence with every number spelled out in "
                                           "words as read aloud in Brazil, then give the IPA of "
                                           "each whitespace token of that spelled-out sentence.")})
    dec = run_llm(items, model=ADJUDICATOR_MODEL, system=SYSTEM_PROMPT, tool=DIGIT_TOOL,
                  schema=DIGIT_SCHEMA, label="digit sentences")
    out = {}
    for it in items:
        d = dec.get(it["id"], {})
        sid = it["id"].split("|")[1]
        spoken = d.get("example_spoken", "")
        toks = tokens_of(spoken)
        ipas = [B.to_sp(x)[0] for x in d.get("ipa_tokens", [])]
        ok = bool(spoken) and len(toks) == len(ipas) and all(
            not B.whitelist_issues(x) for x in ipas)
        out[sid] = {"example_spoken": spoken, "ipa_tokens": ipas, "ok": ok,
                    "confidence": d.get("confidence", "")}
    return out


def audit_sample(lex: dict[str, Entry], rows: list[dict]) -> dict:
    """Fable 5.1 transcribes a sample independently (no candidates shown)."""
    rows_by_sid = {r["sense_id"]: r for r in rows}
    rng = random.Random(SEED)
    consensus = [e for e in lex.values() if e.status in ("ok", "fixed")
                 and e.source == "stage5_majority" and len(e.key) > 2]
    decided = [e for e in lex.values() if e.status == "llm"]
    sample = (rng.sample(consensus, min(AUDIT_SAMPLE // 2, len(consensus)))
              + rng.sample(decided, min(AUDIT_SAMPLE // 2, len(decided))))
    items = []
    for e in sample:
        exs = _examples_for(e, rows_by_sid, 1)
        items.append({"id": f"audit|{e.key}",
                      "user_message": f"Word: {e.key}" + (f"\nUsed in: {exs[0]}" if exs else "")
                      + "\nGive its broad IPA in the convention."})
    dec = run_llm(items, model=AUDITOR_MODEL, system=SYSTEM_PROMPT, tool=AUDIT_TOOL,
                  schema=AUDIT_SCHEMA, label="Fable audit")
    rows_out = []
    for e in sample:
        d = dec.get(f"audit|{e.key}", {})
        if "ipa" not in d:
            continue
        fa, _ = B.to_sp(d["ipa"])
        ours_end = B.stress_positions(e.ipa)[0]
        fa_end = B.stress_positions(fa)[0]
        sv_ours, sv_fa = B.base(B.stressed_vowel(e.ipa)), B.base(B.stressed_vowel(fa))
        agree = ours_end == fa_end and sv_ours == sv_fa
        rows_out.append({"key": e.key, "group": "consensus" if e.status != "llm" else "llm",
                         "ours": e.ipa, "fable": fa, "agree": agree})
    by = collections.defaultdict(lambda: [0, 0])
    for r in rows_out:
        by[r["group"]][0] += r["agree"]
        by[r["group"]][1] += 1
    return {"rows": rows_out,
            "rates": {g: f"{a}/{n} agree" for g, (a, n) in by.items()}}


# ── assembly ─────────────────────────────────────────────────────────────────
def _voice_coda_s(ipas: list[str]) -> list[str]:
    """Cross-word coda-s voicing: final s → z before a vowel / voiced consonant."""
    out = list(ipas)
    for i in range(len(out) - 1):
        cur, nxt = out[i], out[i + 1].lstrip(B.STRESS)
        if not cur.endswith("s") or not nxt:
            continue
        first = B.segments(nxt)[0]
        if B.is_vowel(first) or B.is_glide(first) or first in B.VOICED_CONSONANTS:
            out[i] = cur[:-1] + "z"
    return out


def token_ipa(lex: dict[str, Entry], occ_dec: dict, tok: str, sid: str, where: str,
              idx: int) -> tuple[str, str]:
    k = key_of(tok)
    e = lex.get(k)
    if e is None:
        return "", "missing"
    if e.status == "homograph":
        d = occ_dec.get(f"{k}|{sid}|{where}|{idx}")
        if d and d["ipa"]:
            return d["ipa"], "llm_context"
        return e.ipa, "homograph_unresolved"
    if e.status in ("digit",):
        return "", "digit"
    return e.ipa, e.status


def assemble(lex: dict[str, Entry], rows: list[dict], occ_dec: dict, digits: dict) -> list[dict]:
    out = []
    for r in rows:
        sid = r["sense_id"]
        flags = []
        # word (article excluded, as in Stage 05)
        head = tokens_of(r["pt"])
        if len(head) == 1:
            w_ipa, w_st = token_ipa(lex, occ_dec, head[0], sid, "headword", 0)
        else:
            parts = [token_ipa(lex, occ_dec, t, sid, "headword_mwe", i) for i, t in enumerate(head)]
            w_ipa = " ".join(p[0] for p in parts)
            w_st = "mwe:" + ",".join(sorted({p[1] for p in parts}))
        if not w_ipa:
            w_ipa, w_st = B.to_sp(r["ipa_word"])[0], "stage5_fallback"
            flags.append("word_fallback")
        # sentence
        spoken = ""
        if sid in digits:
            d = digits[sid]
            if d["ok"]:
                spoken = d["example_spoken"]
                ex_ipas, ex_states = d["ipa_tokens"], ["llm_digits"] * len(d["ipa_tokens"])
            else:
                ex_ipas, ex_states = [], ["digit_failed"]
        else:
            toks = tokens_of(r["example_pt"])
            pairs = [token_ipa(lex, occ_dec, t, sid, "example", i) for i, t in enumerate(toks)]
            ex_ipas = [p[0] for p in pairs]
            ex_states = [p[1] for p in pairs]
        if ex_ipas and all(ex_ipas):
            ex_ipa = " ".join(_voice_coda_s(ex_ipas))
            bad = {s for s in ex_states if s in ("unresolved", "homograph_unresolved")}
            ex_st = "partial" if bad else "ok"
            if bad:
                flags.append("example_partial:" + ",".join(sorted(bad)))
        else:
            ex_ipa, ex_st = " ".join(B.to_sp(x)[0] for x in r["ipa_example"].split()), "stage5_fallback"
            flags.append("example_fallback")
        # target-token consistency (uninflected, non-heterophone)
        tgt = r.get("target_word_used", "")
        if (len(head) == 1 and key_of(tgt) == key_of(head[0])
                and key_of(tgt) not in B.HETEROPHONES and not spoken):
            toks = tokens_of(r["example_pt"])
            idx = next((i for i, t in enumerate(toks) if key_of(t) == key_of(tgt)), None)
            if idx is not None and ex_st != "stage5_fallback":
                in_sent = ex_ipa.split()[idx].rstrip("sz")
                if in_sent != w_ipa.rstrip("sz"):
                    flags.append("target_mismatch")
        out.append({
            "sense_id": sid, "pt": r["pt"], "ipa_word": w_ipa, "ipa_word_status": w_st,
            "ipa_example": ex_ipa, "ipa_example_status": ex_st, "example_spoken": spoken,
            "changed_word": int(w_ipa != r["ipa_word"]),
            "changed_example": int(ex_ipa != r["ipa_example"]), "flags": "|".join(flags),
        })
    return out


# ── golden checks (also exercised by tests) ──────────────────────────────────
GOLDEN = {
    "bonito": "boˈnitu", "domingo": "doˈmĩɡu", "está": "esˈta", "quarto": "ˈkwaɾtu",
    "verde": "ˈveɾdʒi", "mulher": "muˈʎɛɾ", "animal": "aniˈmaw", "hospital": "ospiˈtaw",
    "café": "kaˈfɛ", "carro": "ˈkahu", "rua": "ˈhuɐ", "porta": "ˈpɔɾtɐ", "falar": "faˈlaɾ",
    "pessoa": "peˈsoɐ", "semana": "seˈmɐ̃nɐ", "muito": "ˈmũj̃tu", "pão": "ˈpɐ̃w̃",
}


def golden_report(lex: dict[str, Entry]) -> list[tuple[str, str, str, bool]]:
    out = []
    for w, want in GOLDEN.items():
        got = lex[w].ipa if w in lex else ""
        g_end, _, _ = B.stress_positions(got) if got else (None, None, 0)
        w_end, _, _ = B.stress_positions(want)
        ok = bool(got) and g_end == w_end and B.base(B.stressed_vowel(got)) == B.base(
            B.stressed_vowel(want))
        out.append((w, want, got, ok))
    return out


# ── outputs ──────────────────────────────────────────────────────────────────
def write_lexicon(lex: dict[str, Entry], occ_dec: dict) -> None:
    rows = []
    for e in sorted(lex.values(), key=lambda x: x.key):
        rows.append({"key": e.key, "context": "*", "ipa": e.ipa, "status": e.status,
                     "source": e.source,
                     "stage5_votes": "; ".join(f"{k}×{n}" for k, n in e.votes.most_common()),
                     "n_occurrences": len(e.occurrences),
                     "mfa_variants": " / ".join(" ".join(v) for v in e.mfa[:3]),
                     "oracle": B.espeak_ipa(e.key) if len(e.key) < 40 else "",
                     "issues": ", ".join(e.issues), "llm_confidence": e.confidence,
                     "notes": e.notes})
    for occ_id, d in sorted(occ_dec.items()):
        key, sid, where, idx = occ_id.split("|")
        rows.append({"key": key, "context": f"{sid}:{where}:{idx}", "ipa": d["ipa"],
                     "status": "llm_context" if d["ipa"] else "unresolved",
                     "source": ADJUDICATOR_MODEL, "issues": ", ".join(d["errors"]),
                     "llm_confidence": d["confidence"], "notes": d["rationale"][:300]})
    write_tsv(LEXICON_OUT, rows, fieldnames=LEXICON_FIELDS)


def write_report(lex, senses, rows, occ_dec, digits, audit, golden) -> None:
    css = """
    :root{--bg:#0f1115;--panel:#181b22;--border:#2c3140;--text:#e6e9ef;--muted:#9098a6;
      --ok:#4ade80;--warn:#fbbf24;--bad:#ff6b6b;--accent:#5aa9ff;}
    *{box-sizing:border-box}
    body{background:var(--bg);color:var(--text);margin:0;padding:24px 16px;
      font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
    .wrap{max-width:1150px;margin:0 auto} h1{font-size:21px;margin:0 0 4px}
    h2{font-size:16px;margin:24px 0 8px} .sub{color:var(--muted);margin:0 0 16px}
    table{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:8px}
    td,th{padding:5px 9px;text-align:left;border-bottom:1px solid var(--border);vertical-align:top}
    th{color:var(--muted);font-weight:500} .ipa{font-family:ui-monospace,Menlo,monospace}
    .ok{color:var(--ok)} .bad{color:var(--bad)} .muted{color:var(--muted)}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
    .card{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:10px 12px}
    .card b{font-size:20px;display:block}
    .scroll{overflow-x:auto}
    """
    esc = html.escape
    st = collections.Counter(e.status for e in lex.values())
    cards = "".join(f"<div class='card'><b>{n:,}</b>{esc(k)}</div>" for k, n in st.most_common())
    sense_st = collections.Counter(s["ipa_example_status"] for s in senses)
    changed_w = sum(s["changed_word"] for s in senses)
    changed_e = sum(s["changed_example"] for s in senses)
    conv = [
        ("Primary stress", "ˈ before the stressed syllable's onset; no secondary stress / dots"),
        ("Strong r (rua, carro, honra, Israel)", "h — ˈhuɐ, ˈkahu, ˈõhɐ, izhaˈɛw"),
        ("Coda r (verde, falar, porta)", "ɾ (São Paulo tap) — ˈveɾdʒi, faˈlaɾ, ˈpɔɾtɐ"),
        ("t, d + i", "tʃ, dʒ — ˈtʃiɐ, ˈdʒiɐ"),
        ("Final unstressed e / o / a", "i / u / ɐ — ˈveɾdʒi, ˈkwaɾtu, ˈkazɐ"),
        ("Coda s", "s, z before voiced; across words: uz aˈmiɡus"),
        ("Final l", "w — aniˈmaw"),
        ("Nasal diphthongs", "ɐ̃w̃ ẽj̃ õj̃ ɐ̃j̃ ũj̃ — ˈpɐ̃w̃, tɐ̃ˈbẽj̃, ˈmũj̃tu"),
        ("Clitics (o, de, que, em …)", "weak forms: u, dʒi, ki, ẽj̃ — unstressed"),
    ]
    conv_html = "".join(f"<tr><td>{esc(a)}</td><td class='ipa'>{esc(b)}</td></tr>" for a, b in conv)
    gold_html = "".join(
        f"<tr><td>{esc(w)}</td><td class='ipa'>{esc(want)}</td><td class='ipa'>{esc(got)}</td>"
        f"<td class='{'ok' if ok else 'bad'}'>{'✓' if ok else '✗'}</td></tr>"
        for w, want, got, ok in golden)
    rng = random.Random(SEED)
    rows_by_sid = {r["sense_id"]: r for r in rows}
    changed = [s for s in senses if s["changed_word"]]
    diff_html = "".join(
        f"<tr><td>{esc(s['sense_id'])}</td><td>{esc(s['pt'])}</td>"
        f"<td class='ipa muted'>{esc(rows_by_sid[s['sense_id']]['ipa_word'])}</td>"
        f"<td class='ipa'>{esc(s['ipa_word'])}</td><td>{esc(s['ipa_word_status'])}</td></tr>"
        for s in rng.sample(changed, min(60, len(changed))))
    llm = sorted((e for e in lex.values() if e.status in ("llm", "unresolved")), key=lambda e: e.key)
    llm_html = "".join(
        f"<tr><td>{esc(e.key)}</td><td class='ipa muted'>"
        f"{esc('; '.join(f'{k}×{n}' for k, n in e.votes.most_common(3)))}</td>"
        f"<td class='ipa'>{esc(e.ipa)}</td><td>{esc(e.status)}</td><td>{esc(e.confidence)}</td>"
        f"<td class='muted'>{esc(e.notes[:160])}</td></tr>" for e in llm)
    homo_html = "".join(
        f"<tr><td>{esc(k.split('|')[0])}</td><td>{esc(k.split('|', 1)[1])}</td>"
        f"<td class='ipa'>{esc(d['ipa'])}</td><td>{esc(d['confidence'])}</td>"
        f"<td class='muted'>{esc(d['rationale'][:140])}</td></tr>"
        for k, d in sorted(occ_dec.items()))
    dig_html = "".join(
        f"<tr><td>{esc(sid)}</td><td>{esc(rows_by_sid[sid]['example_pt'])}</td>"
        f"<td>{esc(d['example_spoken'])}</td><td class='ipa'>{esc(' '.join(d['ipa_tokens']))}</td>"
        f"<td class='{'ok' if d['ok'] else 'bad'}'>{'ok' if d['ok'] else 'fallback'}</td></tr>"
        for sid, d in sorted(digits.items()))
    aud_html = "".join(
        f"<tr><td>{esc(r['key'])}</td><td>{esc(r['group'])}</td><td class='ipa'>{esc(r['ours'])}</td>"
        f"<td class='ipa'>{esc(r['fable'])}</td><td class='{'ok' if r['agree'] else 'bad'}'>"
        f"{'agree' if r['agree'] else 'DISAGREE'}</td></tr>"
        for r in sorted(audit.get("rows", []), key=lambda r: (r["agree"], r["group"])))
    doc = f"""<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>IPA v2 report</title>
<style>{css}</style></head><body><div class='wrap'>
<h1>Stage 19 — IPA v2 (São Paulo)</h1>
<p class='sub'>{len(lex):,} token types · {len(senses):,} senses · word IPA changed on {changed_w:,},
sentence IPA changed on {changed_e:,} · sentence status {dict(sense_st)} ·
Fable audit: {esc(json.dumps(audit.get('rates', {})))}</p>
<div class='grid'>{cards}</div>
<h2>Conventions (please sign off)</h2><table>{conv_html}</table>
<h2>Golden checks</h2><table><tr><th>word</th><th>expected</th><th>v2</th><th></th></tr>{gold_html}</table>
<h2>Sample of changed word IPA (Stage 05 → v2)</h2><div class='scroll'><table>
<tr><th>sense</th><th>pt</th><th>Stage 05</th><th>v2</th><th>status</th></tr>{diff_html}</table></div>
<h2>Decided by {esc(ADJUDICATOR_MODEL)} / unresolved ({len(llm)})</h2><div class='scroll'><table>
<tr><th>word</th><th>Stage-05 votes</th><th>v2</th><th>status</th><th>conf.</th><th>note</th></tr>
{llm_html}</table></div>
<h2>Heterophones in context ({len(occ_dec)})</h2><div class='scroll'><table>
<tr><th>word</th><th>occurrence</th><th>IPA</th><th>conf.</th><th>rationale</th></tr>{homo_html}</table></div>
<h2>Sentences with numbers ({len(digits)})</h2><div class='scroll'><table>
<tr><th>sense</th><th>sentence</th><th>spoken</th><th>IPA</th><th></th></tr>{dig_html}</table></div>
<h2>Independent audit by {esc(AUDITOR_MODEL)}</h2><div class='scroll'><table>
<tr><th>word</th><th>group</th><th>v2</th><th>Fable</th><th></th></tr>{aud_html}</table></div>
</div></body></html>"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(doc, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--analyze-only", action="store_true", help="no LLM; print stats")
    ap.add_argument("--dry-run", action="store_true", help="print the LLM cost estimate")
    ap.add_argument("--yes", action="store_true", help="run the LLM phases")
    ap.add_argument("--skip-audit", action="store_true")
    ap.add_argument("--show", type=int, default=0, help="print N sample conflicts")
    args = ap.parse_args()
    rows = read_tsv(FINAL)
    if not rows:
        raise IpaV2Error(f"{FINAL} is empty")
    print("=== Stage 19.1 — IPA v2 ===")
    lex = analyze(rows)
    if args.show:
        rng = random.Random(SEED)
        conf = [e for e in lex.values() if e.status == "conflict"]
        for e in rng.sample(conf, min(args.show, len(conf))):
            print(f"    {e.key:<16} {e.ipa:<16} votes={dict(e.votes)} issues={e.issues} "
                  f"mfa={[' '.join(v) for v in e.mfa][:2]} espeak={B.espeak_ipa(e.key)}")
    if args.analyze_only:
        return 0
    n_conf = sum(1 for e in lex.values() if e.status == "conflict")
    n_homo = sum(1 for e in lex.values() if e.status == "homograph"
                 for o in e.occurrences if o[1] in ("example", "headword"))
    n_dig = sum(1 for r in rows if any(DIGIT_RE.search(t) for t in tokens_of(r["example_pt"])))
    n_calls = n_conf + n_homo + n_dig + (0 if args.skip_audit else AUDIT_SAMPLE)
    print(f"  LLM plan: {n_conf} conflicts + {n_homo} heterophone occurrences + {n_dig} digit "
          f"sentences ({ADJUDICATOR_MODEL}) + {0 if args.skip_audit else AUDIT_SAMPLE} audit "
          f"({AUDITOR_MODEL}) ≈ {n_calls} calls, est. ≤ ${n_calls * EST_USD_PER_CALL:.0f} "
          f"(+~20% retries; cached calls are free)")
    if args.dry_run:
        return 0
    if not args.yes:
        raise IpaV2Error("pass --yes to run the LLM phases (see the estimate above)")
    occ_dec = adjudicate(lex, rows)
    digits = spell_out_digits(rows)
    audit = {} if args.skip_audit else audit_sample(lex, rows)
    senses = assemble(lex, rows, occ_dec, digits)
    write_lexicon(lex, occ_dec)
    write_tsv(IPA_V2_OUT, senses, fieldnames=IPA_V2_FIELDS)
    golden = golden_report(lex)
    write_report(lex, senses, rows, occ_dec, digits, audit, golden)
    st = collections.Counter(e.status for e in lex.values())
    print(f"  lexicon status: {dict(st)}")
    print(f"  senses: word changed {sum(s['changed_word'] for s in senses):,}, "
          f"example changed {sum(s['changed_example'] for s in senses):,}, "
          f"flags {collections.Counter(f for s in senses for f in s['flags'].split('|') if f)}")
    print(f"  golden: {sum(g[3] for g in golden)}/{len(golden)}  audit: {audit.get('rates')}")
    print(f"  wrote {LEXICON_OUT.relative_to(REPO_ROOT)}, {IPA_V2_OUT.relative_to(REPO_ROOT)}, "
          f"{REPORT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
