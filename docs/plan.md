# Plan: Anki-Ready BP Portuguese Dataset (Data Phase)

## Context

The user is an A1 Brazilian Portuguese learner who knows many languages but none Romance. The goal is to build a **complete, high-leverage dataset** from the 4,985-entry frequency dictionary at [data/source.txt](data/source.txt) that can later drive an Anki deck — but the current phase is strictly **data preparation**, not Anki card construction.

Completeness is non-negotiable: no words skipped, no meanings skipped. Leverage means every row carries enough structure to support audio-rich, example-anchored, IPA-marked, sense-disambiguated review. The source is EP-leaning with BP swap tags; the user wants pure BP output. Downstream tooling (Anki note type, card templates, scheduling) is out of scope for this plan.

The outcome of this plan is a single authoritative TSV, plus linked audio in Google Drive, that the user can open in a spreadsheet to review and from which an Anki deck can later be deterministically generated.

## Settled decisions (from Q&A)

| Decision | Value |
|---|---|
| Dialect | BP only; normalize EP pre-reform spellings in-place |
| EP-only entries | Auto-drop with audit log |
| BP-native cross-check | Skipped |
| Sense granularity | **Sense is the unit of study**; one TSV row per sense |
| Sense-split method | LLM (Claude), user spot-checks first 200 |
| Sense distinction rule | Senses are distinct iff they require different example sentences |
| Example sentences | 1 per sense, ≤15 words, A2-background grammar, target word in typical use for THAT sense; EN translation on card |
| Register | Neutral-everyday; tag if naturally travel/legal |
| IPA | BP neutral-paulistano, broad phonemic with stress, per-word for examples |
| Audio | 4 clips per sense (2 voices × {word, example}); fixed 1M+1F across whole deck; ElevenLabs |
| Audio stage | **Pilot first 500, then full batch** |
| Audio storage | Google Drive, URLs in TSV |
| Cognate flag | `#cognate-en` tag, no review-order change |
| Legal/travel expansion | Out of scope for now |
| Data format | TSV, one row per sense, stable `sense_id = rank.sense_index` |
| Study app | Mac desktop + iPhone |
| LLM providers | Anthropic + OpenAI OK; ElevenLabs for audio |

## Repo layout

```
anki-mastery/
├── data/
│   ├── source.txt                    # EXISTING — never mutated
│   ├── 01-normalized.tsv             # post EP-drop + BP spelling normalize
│   ├── 02-senses.tsv                 # post sense-split (one row per sense)
│   ├── 03-enriched.tsv               # post gender, PoS, family, tags, cognate
│   ├── 04-examples.tsv               # post example PT + EN
│   ├── 05-ipa.tsv                    # post IPA for word + example
│   ├── 06-final.tsv                  # FINAL DELIVERABLE (audio URLs attached)
│   ├── _ep_drops.tsv                 # audit log of auto-dropped EP entries
│   ├── _ep_spelling_map.tsv          # EP→BP spelling normalization rules
│   └── _pilot_500.tsv                # first 500 senses for audio QA gate
├── build/
│   ├── 01_normalize.py
│   ├── 02_split_senses.py            # Anthropic
│   ├── 03_enrich.py                  # Anthropic + dictionary APIs
│   ├── 04_examples.py                # Anthropic
│   ├── 05_ipa.py                     # Anthropic (+ spot-check gate)
│   ├── 06_audio_pilot.py             # ElevenLabs + Drive, first 500 only
│   ├── 07_audio_full.py              # ElevenLabs + Drive, remaining
│   └── lib/
│       ├── tsv.py                    # read/write with quoting (handles line 262's quotes)
│       ├── parse.py                  # split-on-first-` = `, extract parentheticals
│       ├── bp_rules.py               # EP→BP spelling table; EP-only drop list
│       ├── llm.py                    # Anthropic client wrapper, batching, caching
│       ├── drive.py                  # Google Drive upload, shareable URL retrieval
│       └── validate.py               # row-count, uniqueness, schema invariants
├── tests/
│   └── test_invariants.py            # sense_id uniqueness, no empty fields, etc.
├── .env.example                      # ANTHROPIC_API_KEY, ELEVENLABS_API_KEY, GOOGLE_*
├── pyproject.toml                    # or requirements.txt
└── README.md                         # how to run the pipeline
```

## Pipeline stages

Each stage reads the previous stage's TSV and writes the next. Each is idempotent (safe to re-run) and has a validation check.

### Stage 1 — Parse & BP-normalize ([build/01_normalize.py](build/01_normalize.py))

- Read [data/source.txt](data/source.txt) line by line.
- Split on **first** ` = ` only (handles [line 31](data/source.txt:31), [line 314](data/source.txt:314)).
- Extract parentheticals from RHS into a structured `annotation` JSON field (dialect, number, gender-note, reflexive `+se`, etc.) per the catalog in the earlier conversation.
- Apply `_ep_spelling_map.tsv` to transform headwords (`facto→fato`, `óptimo→ótimo`, etc.). Rules table is hand-curated once (~15-30 entries), committed to the repo.
- Identify EP-only headwords via two signals: (a) presence of an explicit BP equivalent already in the source (`comboio` vs `trem`), (b) a hand-curated `_ep_only.tsv` list. Write dropped rows to [data/_ep_drops.tsv](data/_ep_drops.tsv) with reason column. Drop from main flow.
- Output [data/01-normalized.tsv](data/01-normalized.tsv): `rank, pt, gender (blank), pos (blank), en_all, annotation, source_line`.
- Validate: ~4,950–4,970 rows; every row has `pt` and `en_all`; no `=` in `pt`.

### Stage 2 — Sense split ([build/02_split_senses.py](build/02_split_senses.py))

- For each row where `en_all` contains `/`: Claude classifies whether the slash-separated items are **distinct senses** (need different examples) or **English near-synonyms** (one sense, list of translations).
- Rule enforced in the prompt: "Two items are distinct senses if you cannot construct a single example sentence where both translations are natural." Few-shot with `melhor = better / best` (1 sense), `ponto = point / dot / period` (3 senses).
- Emit one row per sense. Assign `sense_id = zero-padded-rank.sense_index` (e.g., `0001.1` … `0001.5` for `o`). Sense IDs are stable forever.
- Carry `en_primary` (single gloss for this sense) alongside the preserved `en_all`.
- **QA gate**: pause after first 200 headwords are sense-split; user spot-checks [data/_pilot_senses.tsv](data/_pilot_senses.tsv). User approval required before processing the remainder.
- Output [data/02-senses.tsv](data/02-senses.tsv): ~8,000–10,000 rows.
- Validate: `sense_id` uniqueness; every row has `en_primary`; re-joining `en_primary` by sense_index reconstructs a superset of `en_all`.

### Stage 3 — Enrichment ([build/03_enrich.py](build/03_enrich.py))

Deterministic + LLM hybrid:

- **Gender** (nouns): look up in Wiktionary BP-PT API or dicionario.priberam.org (BP mode). Claude fallback with prompt "reply only with `o`, `a`, `o/a`, or `—`". Fill `gender` and compose `pt_display` (`a casa`, `o caminho`, or bare word for non-nouns).
- **PoS**: derive from gloss pattern — "to X" → `verb`; noun if gender known; adjective if gloss is bare adjective. Leave blank when ambiguous.
- **Tags**: `#top500`/`#top1000`/`#top2000`/`#top5000` from rank band; `#verb`/`#noun`/etc. from PoS; `#reflexive` if `annotation.reflexive`; `#gendered-meaning` for the 5 M/F-split homographs; `#interjection`, `#numeral` from explicit lists; `#cognate-en` via Claude classification (one pass, per-row binary); `#hyphenated` for compound headwords.
- **Family root**: Claude proposes Portuguese derivational root (e.g., `criar` for `criação`, `criador`, `criatura`). Empty string when standalone. One pass across all rows.
- Output [data/03-enriched.tsv](data/03-enriched.tsv).
- Validate: every noun has `gender` (or `—` if genuinely uncertain); tags are space-separated and well-formed.

### Stage 4 — Example sentences ([build/04_examples.py](build/04_examples.py))

- Claude generates one example sentence per sense. Prompt enforces: ≤15 words, A2-background grammar only (no compound tenses or embedded clauses unless the target word itself requires it), target word in its **typical** use for **this specific sense**, neutral register, BP vocabulary and spelling. Also produces `example_en`.
- Batch in groups of ~50 senses per request for throughput.
- **QA gate**: pause after first 500 senses; user spot-checks. Issues go into a `_example_fixes.tsv` for targeted re-generation.
- Output [data/04-examples.tsv](data/04-examples.tsv).
- Validate: every row has `example_pt` and `example_en`; target headword (lemma form) appears in `example_pt` (simple regex — imperfect for verbs but flags obvious misses).

### Stage 5 — IPA ([build/05_ipa.py](build/05_ipa.py))

- Claude generates broad phonemic BP neutral-paulistano IPA with stress, for (a) the headword single token, (b) each token of the example sentence (word-aligned, not connected speech).
- Output tokens separated by single space.
- **QA gate**: user reviews first 100 `ipa_word` values. Fix patterns of error if found, re-run.
- Output [data/05-ipa.tsv](data/05-ipa.tsv).
- Validate: `ipa_word` non-empty for every row; `ipa_example` token count matches `example_pt` token count.

### Stage 6 — Audio pilot ([build/06_audio_pilot.py](build/06_audio_pilot.py))

- Take first 500 senses from `05-ipa.tsv`.
- Generate 4 audio clips per sense via ElevenLabs (2 voices × {word, example}) = 2,000 clips.
- Voices: user-provided male voice ID; female voice ID placeholder until user supplies.
- Upload each clip to a Google Drive folder; capture shareable URL.
- Write [data/_pilot_500.tsv](data/_pilot_500.tsv) with audio URL columns populated.
- **QA gate**: user opens `_pilot_500.tsv` in Numbers/Sheets, plays a sample via URL, confirms voice quality + pronunciation accuracy on a random sample. Approval required.

### Stage 7 — Audio full ([build/07_audio_full.py](build/07_audio_full.py))

- Same pipeline for the remaining ~7,500–9,500 senses after pilot approval.
- Writes final [data/06-final.tsv](data/06-final.tsv).
- Validate: every row has 4 audio URLs; URLs resolve (HEAD request sample check).

## Final TSV schema (data/06-final.tsv)

Columns in order:

| # | Column | Type | Notes |
|---|---|---|---|
| 1 | `sense_id` | `0001.1` | Stable forever |
| 2 | `rank` | int | From source.txt |
| 3 | `pt` | string | BP-normalized headword |
| 4 | `gender` | `o`/`a`/`o/a`/`—` | Non-nouns get `—` |
| 5 | `pt_display` | string | With article for nouns |
| 6 | `pos` | string or blank | Only when certain |
| 7 | `sense_index` | int | 1-indexed within headword |
| 8 | `en_primary` | string | Single English gloss for THIS sense |
| 9 | `en_all` | string | Full original RHS; audit |
| 10 | `annotation` | JSON string | Parenthetical metadata |
| 11 | `ipa_word` | string | BP broad phonemic + stress |
| 12 | `example_pt` | string | ≤15 words, A2 background |
| 13 | `example_en` | string | English translation |
| 14 | `ipa_example` | string | Space-separated per-word IPA |
| 15 | `audio_word_v1` | URL | Drive link, speaker 1 |
| 16 | `audio_word_v2` | URL | Drive link, speaker 2 |
| 17 | `audio_example_v1` | URL | Drive link, speaker 1 |
| 18 | `audio_example_v2` | URL | Drive link, speaker 2 |
| 19 | `family_root` | string | Empty if standalone |
| 20 | `tags` | string | Space-separated |
| 21 | `source_line` | string | Raw original; audit |
| 22 | `notes` | string | Escape hatch |

TSV chosen over CSV to avoid quoting friction (one source line has internal `"folk"` quotes). Fields with tabs are disallowed; Python `csv` module with `dialect='excel-tab'` and `quoting=QUOTE_MINIMAL` handles the remaining edge cases (commas, internal quotes).

## Verification

**Per-stage**: each `build/NN_*.py` ends with assertions (row count, unique IDs, no empty required fields). CI-able via `pytest tests/test_invariants.py` on any stage output.

**End-to-end**:

1. Run `python build/01_normalize.py` → inspect `_ep_drops.tsv` and `01-normalized.tsv`; confirm drop count is plausible (~15-30).
2. Run `python build/02_split_senses.py --limit 200` → open `02-senses.tsv` in Numbers; spot-check sense splits; re-run without `--limit` after approval.
3. Run `python build/03_enrich.py` → confirm gender filled for common nouns, tags look sane.
4. Run `python build/04_examples.py --limit 500` → spot-check examples; re-run for remainder after approval.
5. Run `python build/05_ipa.py --limit 100` → spot-check IPA; re-run for remainder.
6. Run `python build/06_audio_pilot.py` → open `_pilot_500.tsv`, play sample audios via Drive URLs; approve.
7. Supply female voice ID (if still placeholder) and re-run pilot for affected rows, OR accept placeholder and continue.
8. Run `python build/07_audio_full.py` → `06-final.tsv` produced.
9. `pytest tests/test_invariants.py` passes.

**Cost budget**: ElevenLabs ~$200–330 one-time; Anthropic API ~$50–100; total ~$300–430.

**Timeline**: ~1 week of elapsed time assuming prompt user review at each QA gate.

## What this plan deliberately does NOT do

- No Anki note-type design, card templates, or `.apkg` generation (next phase).
- No EN→PT production cards or cloze cards (deferred).
- No BP cross-check against external corpora (user opted out).
- No legal/travel expansion vocabulary (out of scope for now).
- No conjugation sub-deck (deferred).
- No derivational-family **review order** — family_root is a field for future use only.
- No storage of the authoritative data as anything other than TSV files in this repo (simplicity over DB flexibility).
