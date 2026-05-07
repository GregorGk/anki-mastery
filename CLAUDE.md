# Anki-Ready BP Portuguese Dataset — Project Conventions

## What this project is

A data-engineering pipeline that turns the 4,985-entry frequency dictionary at
`data/source.txt` into a single authoritative TSV (`data/06-final.tsv`) plus
audio assets on Cloudflare R2, suitable for driving a Brazilian-Portuguese
Anki deck. The user is an A1 Brazilian-Portuguese learner who knows many
languages but no Romance languages.

## Spec is at `docs/plan.md` — read the relevant stage section before designing

The plan is the authoritative specification. Always read the section that
describes the stage you're working on **before** writing code. Do not
paraphrase the plan in chat; refer the user to the file when asked about
design decisions.

## Hard rules

- `data/source.txt` is **immutable**. Never modify it. Never re-tokenize it.
  Stage 1a parses it line by line and never writes back.
- `data/_source_ledger.tsv` and `data/_audio_manifest.tsv` are the two
  authoritative stateful artifacts. Other stage outputs (`01-normalized.tsv`,
  …) are regenerable from these plus the manual override files.
- Manual override files (`data/_manual_*.tsv`) **always win** over LLM output.
  Apply overrides before the LLM is invoked for a row, not after.
- `sense_id` format is `{RRRR.EE.SS}` — three zero-padded fields
  (rank.expansion_index.sense_index). Stable forever once assigned.
- Audio filenames bake the version in: `{sense_id}-{word|ex}-{m|f}-v{N}.mp3`.
  Anki strips URL query strings on download, so `?v=N` would silently fail.
- Never run `git commit` or `git push` unless the user explicitly asks.

## Code conventions

- Python 3.11+, managed by `uv`. Run scripts with `uv run python build/...`.
- TSV I/O via Python `csv` with `dialect='excel-tab'` and
  `quoting=QUOTE_MINIMAL`. Source line 262 has internal `"folk"` quotes —
  the round-trip must preserve them.
- LLM outputs use **Anthropic Tool Use** for structured fields (no free-text
  parsing). Schema enums are enforced server-side.
- Async I/O for any pipeline that fans out > 5 parallel calls. Bounded
  concurrency via `asyncio.Semaphore`. Full-jitter exponential backoff,
  6 attempts, 1s base, 60s cap. Honor `Retry-After` headers.
- Idempotency keys on retried writes: `sense_id + clip_type + voice_gender + version`
  for ElevenLabs/TTS uploads.
- All LLM calls log to `audit/<stage>.jsonl` with `model_id`, `prompt_hash`,
  `response_hash`, `decision`, `confidence`, `generated_at`, `manual_override`.

## Stage 1 specifics (current focus)

- Stage 1a (parse): no API calls. Pure Python on `data/source.txt`.
- Stage 1b (orthographic normalize): table-driven via
  `data/_ep_spelling_map.tsv` and `data/_hyphen_rules.tsv`. Anthropic Tool
  Use **only** for ambiguous accent cases the heuristic flags as borderline.
- Stage 1c (lexical replace + collision merge): table-driven via
  `data/_lexical_bp_replacements.tsv`. Anthropic only for ambiguous merges;
  otherwise rows go to `data/_ep_drop_or_replace_review.tsv` for human triage.
- 3 embedded-`=` cases on RHS: lines 31 (`eu`), 314 (`medida`), 3372 (`vós`).
  Always split on first ` = `.
- Idiom-candidate detection covers: `medida` (a m. que), `diante` (em d.),
  `cento` (por c.), `seguida` (em s.), `vigor` (em v.), `redor` (em / ao r.),
  `repente` (de r.), `invés` (ao i.), `contrapartida` (em c.),
  `obstante` (não o.), `mercê` (a m. de), `tona` (à t.).
- False-positive guards (must NOT be expanded as gender markers or idioms):
  `mina (M. Gerais: state in B)`, `OBJ = me`, `OBJ = vos`, `e.g.`, `i.e.`.

## Test before scaling

- Always run `pytest tests/` and pass before running a stage on the full
  corpus. The golden set covers every edge case the pipeline must handle.
- Stage outputs go to `data/`. Do not commit them without spot-checking.
- Soft diagnostics (row counts) are warnings, not failures. Hard
  invariants (uniqueness, ledger consistency, schema) are failures.

## Forbidden absent explicit user request

- `git commit`, `git push`, `git tag`, `git rebase`
- Modifying `.env`, `data/source.txt`, `docs/plan.md`
- Running scripts that incur API costs without showing the projected spend
- Auto-generating `_manual_*.tsv` content (manual files are the user's, not
  the LLM's)
