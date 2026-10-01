# Frequency-Dictionary → Anki Deck Pipeline

A data-engineering pipeline that turns a plain-text frequency dictionary into a
single authoritative TSV plus a full set of synthesized, quality-gated audio
clips, and finally a learner-ordered Anki import file.

The current target language is **Brazilian Portuguese (BP)** built from a
4,985-line European/Brazilian Portuguese frequency list, producing **~5,725
senses** and **17,175 audio clips** (word + example + English-example per
sense). The architecture is deliberately language-agnostic; see
[§10 Porting to another language](#10-porting-to-another-language).

> **`docs/plan.md` is the authoritative specification.** This README is the
> navigational overview and the per-stage reference. Where the two disagree,
> the plan and the code win — this document is descriptive, not normative.

---

## Table of contents

1. [What this produces](#1-what-this-produces)
2. [Architecture & invariants](#2-architecture--invariants)
3. [Setup & reproduce](#3-setup--reproduce)
4. [Stage-by-stage reference](#4-stage-by-stage-reference)
5. [ASR verification — deep dive](#5-asr-verification--deep-dive)
6. [Audio judgment — deep dive](#6-audio-judgment--deep-dive)
7. [The final TSV schema](#7-the-final-tsv-schema)
8. [Repository layout](#8-repository-layout)
9. [Testing & verification](#9-testing--verification)
10. [Porting to another language](#10-porting-to-another-language)

---

## 1. What this produces

```
data/source.txt                         ← immutable input: 4,985 "headword = gloss" lines
        │
        ▼  Stages 1–5.5   (parse, normalize, sense-split, enrich, examples, IPA, audit)
data/06-final.tsv                        ← the canonical enriched master (40 columns, ~5,725 senses)
        │
        ▼  Stages 6–11    (TTS synthesis, pronunciation correction, model migrations)
data/_audio_manifest.tsv  +  Cloudflare R2 ← 17,175 audio clips, ASR-verified
        │
        ▼  Stages 12–15   (LLM metadata: usage hints, family roots, topic tags, risk register)
data/06-final.tsv (40 columns)            ← re-derived after each metadata stage
        │
        ▼  Stage 16       (SQLite mirror + multimodal audio-judge QA gate)
data/06-final.sqlite  +  data/_gemini_drift_watchlist.tsv
        │
        ▼  Stage 17       (deterministic SQL ordering + clean export)
data/07-anki-listening-general.tsv        ← THE DELIVERABLE: 5,725 learner-ordered cards
data/07-anki-media-manifest.tsv           ← 17,175 audio files to download into Anki's media folder
```

**Two-axis audio QA.** Every clip is checked on two independent axes:

- **ASR roundtrip** ([§5](#5-asr-verification--deep-dive)) — *did the TTS say the
  right **words**?* Synthesize → transcribe back → compare to the input text.
- **Audio judgment** ([§6](#6-audio-judgment--deep-dive)) — *does it **sound**
  like the target dialect?* A multimodal LLM listens and rates the accent.

ASR cannot hear an accent (an English-accented "animal" still transcribes as
"animal"); the audio judge cannot cheaply catch a dropped word. You need both.

---

## 2. Architecture & invariants

### Two authoritative stateful artifacts

Everything else is regenerable from these two files plus the manual-override
files:

| File | Role |
|---|---|
| `data/_source_ledger.tsv` | One row per source line. An append-only state machine: tracks `action` (`keep` / `drop_ep_only` / `merge_into_existing_bp_row` / `replace_with_bp_equivalent` / `manual_review` / …), `normalization_action`, merge targets, and the `output_sense_ids` each line produced. No silent drops — every drop carries a `drop_reason`. |
| `data/_audio_manifest.tsv` | One row per `(sense_id, clip_type)` — 23 columns covering TTS provider/model/voice, R2 `object_key` + `url`, `version`, `md5`, the ASR roundtrip result (`asr_transcript`, `asr_similarity`, `asr_decision`), loudness diagnostics, and `status`. |

`data/source.txt` is **immutable** — never modified, never re-tokenized.

### `sense_id` — the stable join key

Format `{RRRR.EE.SS}`, three zero-padded fields:

- `RRRR` — frequency **rank** from `source.txt` (0001–4985).
- `EE` — **expansion index** (00 = base headword; 01+ = idiom expansions, e.g.
  `medida` → `0314.01.01` = `à medida que`).
- `SS` — **sense index** (01 = primary sense; 02+ = polysemy splits).

Assigned once in Stage 2, **stable forever**. Zero-padding means lexical sort ==
numeric sort. Audio filenames bake it in:
`{sense_id}-{word|ex|en_ex}-{model_id}-v{N}.mp3`.

### Idempotency & audit model

- Every stage reads the prior stage's output; re-running a stage on unchanged
  input produces byte-identical output.
- Stages resume: a killed run re-reads its own output/audit and skips
  already-processed rows.
- Every LLM call is logged to `audit/<stage>.jsonl` with `model_id`,
  `prompt_hash`, `response_hash`, the structured `decision`, `confidence`,
  token counts (incl. cache hits), and `generated_at`.

### Manual overrides always win

Each LLM stage consults a `data/_manual_*.tsv` file **before** calling the
model — a manual decision short-circuits the LLM entirely. The manual files are
the human's, never auto-generated. Examples: `_manual_bp_status.tsv`,
`_manual_gender.tsv`, `_manual_examples.tsv`, `_manual_ipa.tsv`,
`_manual_audit.tsv`, `_manual_risk_register.tsv`, `_manual_ordering.tsv`.

### LLM usage conventions

- **Structured output via tool use / response schemas only** — no free-text
  parsing. Enums are enforced server-side.
- **Prompt caching** on every high-volume stage (the system prompt is identical
  across thousands of calls).
- **Bounded concurrency** with a sliding-window RPM limiter
  (`build/lib/rate_limit.py`); full-jitter exponential backoff, honoring
  `Retry-After`.
- **Cross-family validation** — the generator, validator, and auditor of a given
  artifact are deliberately different model families.

### Model registry (as run)

| Role | Provider · model | Notes |
|---|---|---|
| Text classification & generation (Stages 1–5.5) | Anthropic Claude Sonnet (4.5-class) | tool use + caching |
| Metadata classification (Stages 12–15) | Anthropic Claude Sonnet (4.6-class) | tool use + caching |
| Example *validator*, speaker-gender classifier | OpenAI `gpt-4o-mini` | cross-family check |
| **ASR roundtrip** | OpenAI `gpt-4o-transcribe` | locked by A/B — [§5](#5-asr-verification--deep-dive) |
| **Audio judge** (production) | Google `gemini-3.1-pro-preview` AND `gemini-3.8-flash`, expected-IPA prompt (Stage 19) | chosen on 239 user-labeled clips — [§6](#6-audio-judgment--deep-dive), Stage 19 |
| Audio judge (legacy / A/B losers) | OpenAI `gpt-4o-audio-preview`, `gpt-audio-1.5` | used Stages 8–9; lost the Stage-16 bake-off |
| TTS | ElevenLabs `eleven_v4` (Portuguese word + example), `eleven_v3` (English) | migrated `multilingual_v2` → `flash_v2_5` → `eleven_v3` → `eleven_v4` (Stage 19) |
| IPA (cards) | IPA v2: Stage-05 + validators + `claude-opus-5-5`, audited by `claude-fable-5-1` | São Paulo convention (Stage 19) |
| IPA baseline | eSpeak-NG (`pt-BR` locale) | deterministic, free; LLM corrects it |

The exact model snapshot for any run is recorded in that stage's `audit/*.jsonl`.

---

## 3. Setup & reproduce

### Prerequisites

- **Python ≥ 3.11**, managed by [`uv`](https://docs.astral.sh/uv/).
- **ffmpeg + ffprobe** — loudness normalization and silence padding.
- **eSpeak-NG** — deterministic IPA baseline.
- API keys in `.env` (gitignored): `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
  `GEMINI_API_KEY`, `ELEVENLABS_API_KEY`, and Cloudflare R2 credentials
  (`R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`,
  `R2_PUBLIC_URL_BASE`). See `build/lib/r2_client.py::R2Config.from_env()` for
  the exact R2 variable names.

```bash
uv sync                       # install dependencies
cp .env.example .env          # then fill in keys
```

### Run order

Scripts are run with `uv run python build/<script>.py`. Most accept
`--dry-run` (cost estimate, no API calls), `--limit N` (smoke test), and
`--yes` (skip the GO prompt). Stages that incur API spend print a projected
cost and require confirmation.

```
# Text pipeline
01a_parse → 01b_orthographic_normalize → 01c_lexical_replace → 015_bp_status
  → 018_dedupe → 02_split_senses → sensitive_screen → 03_enrich → 04_examples
  → 045_speaker_gender → 05_ipa → 055_audit

# Audio pipeline
06_audio_pilot (→ full corpus) → 08_* (pronunciation correction)
  → 09_* (Flash migration) → 10_0_render_en_audio → 11_* (eleven_v3 migration)

# Derive the master TSV (run after any stage that adds columns)
derive_final.py  →  verify_all.py --http-sample 50

# Metadata classification (each followed by derive_final.py)
12_0_usage_hint_classifier → 13_0_family_root_classifier
  → 14_0_topic_classifier → 15_0_risk_register_classifier

# QA gate + export
16_0_build_sqlite → 16_8_prefetch_audio → 16_8_gemini_qa_gate
  → 16_9_surgical_rerender → 17_0_build_ordering → 17_1_export_anki
  → 17_2_ordering_html
```

`build/verify_all.py` is the cross-stage invariant checker — run it after every
milestone. `pytest tests/` runs the per-stage unit + parity tests.

---

## 4. Stage-by-stage reference

Each stage: **purpose**, **in → out**, **LLM** used (if any), and **notes**
(key locked decisions, manual-override file, what is language-specific).

### Stage 1a — Parse & ledger init · `build/01a_parse.py`
**Purpose.** Parse `source.txt` line-by-line, detect idiom-expansion candidates,
initialize the source ledger and provenance map, extract `(BP)`/`(EP)`
annotations.
**In → Out.** `data/source.txt` → `data/_source_ledger.tsv`,
`data/_sense_source_map.tsv`, `data/_flags.tsv`, `data/_idioms_expanded.tsv`.
**LLM.** Anthropic Claude — tool use — for ambiguous idiom expansions only;
pattern heuristics otherwise.
**Notes.** Split on the *first* ` = ` (3 source lines contain embedded `=`).
False-positive guards prevent `M. Gerais`, `OBJ = me`, `e.g.` being misread.
*Language-specific:* the idiom-candidate list.

### Stage 1b — Orthographic normalization · `build/01b_orthographic_normalize.py`
**Purpose.** Apply European→Brazilian orthographic rules (accent shifts, `ct`/`pt`
cluster drops, 1990-accord hyphenation).
**In → Out.** ledger → `data/01-normalized.tsv`.
**LLM.** None — fully table-driven.
**Notes.** Tables: `data/_ep_spelling_map.tsv` (~157 rules),
`data/_hyphen_rules.tsv`. Rows with dialect markers are flagged for later review.
*Language-specific:* the entire rule set — this is the "dialect normalization"
slot.

### Stage 1c — Lexical EP→BP replacement & collision merge · `build/01c_lexical_replace.py`
**Purpose.** Substitute lexically different words (`comboio → trem`), then detect
and merge post-substitution collisions.
**In → Out.** `01-normalized.tsv` → `data/012-lexical_replaced.tsv`.
**LLM.** None — table-driven (`data/_lexical_bp_replacements.tsv`); ambiguous
merges go to `data/_ep_drop_or_replace_review.tsv` for human triage.
**Notes.** Assigns the ledger `action` state. Every drop/merge/replace carries a
reason. *Language-specific:* the replacement map.

### Stage 1.5 — BP-status classification · `build/015_bp_status.py`
**Purpose.** Classify each row as `standard` / `uncommon` / `false_friend` /
`nsfw` / `ep_only`; filter `ep_only` rows out.
**In → Out.** `012-lexical_replaced.tsv` → `data/015-bp_status.tsv`.
**LLM.** Anthropic Claude — tool use + caching.
**Notes.** Manual override: `data/_manual_bp_status.tsv`. *Language-specific:*
the category definitions and the slang/false-friend examples in the prompt.

### Stage 1.8 — Post-normalization duplicate detection · `build/018_dedupe.py`
**Purpose.** After all normalization, detect remaining `pt` collisions; merge
exact duplicates, route ambiguous ones to human review.
**In → Out.** `015-bp_status.tsv` → `data/018-deduped.tsv`.
**LLM.** None — deterministic set operations.

### Stage 2 — Sense split · `build/02_split_senses.py`
**Purpose.** Decide how many distinct senses each row carries; assign the
permanent `sense_id` to every output row.
**In → Out.** `018-deduped.tsv` → `data/02-senses.tsv` (~5,725 senses).
**LLM.** Anthropic Claude — tool use — only for genuine-polysemy rows; a
deterministic pre-classifier handles synonyms, inflections, gendered splits,
and idioms with no API call.
**Notes.** Split rule: *two items are distinct senses iff you cannot write one
example sentence where both translations are natural.* Forced splits for
`(M … / F …)` source markers. Manual override: `data/_manual_sense_splits.tsv`.
*Language-specific:* the pronoun policy.

### Stage 2.5 — Sensitive-term screen · `build/sensitive_screen.py`
**Purpose.** Flag terms (sexual, violence, slurs, medical, substances, …) that
need a careful example-sentence policy. **Not a drop list.**
**In → Out.** `02-senses.tsv` → `data/_sensitive_terms.tsv` (a policy sidecar
consumed by Stage 4).
**LLM.** Deterministic keyword screen + Anthropic Claude tool use for the rest.

### Stage 3 — Enrichment · `build/03_enrich.py`
**Purpose.** Assign gender (nouns), part of speech, English-cognate flag;
compute `pt_display`; emit frequency/morphology tags.
**In → Out.** `02-senses.tsv` → `data/03-enriched.tsv`.
**LLM.** Anthropic Claude — tool use — for rows not resolved by deterministic
shortcuts. Manual override: `data/_manual_gender.tsv`.

### Stage 4 — Example sentences · `build/04_examples.py`
**Purpose.** Generate a 3-field example per sense (`example_pt`, `example_en`,
`target_word_used`), then cross-validate it.
**In → Out.** `03-enriched.tsv` → `data/04-examples.tsv`,
`data/_example_fixes.tsv`.
**LLM.** **Generator:** Anthropic Claude (tool use + caching). **Validator:**
OpenAI `gpt-4o-mini` (different family) — scores intended-sense use, BP purity,
naturalness, translation match, sensitive-policy compliance.
**Notes.** Examples are A2-level, ≤15 words. Deterministic token-match validation
handles hyphens and enclitic pronouns. Manual override: `data/_manual_examples.tsv`.

### Stage 4.5 — Speaker gender + voice assignment · `build/045_speaker_gender.py`
**Purpose.** Infer speaker gender from in-sentence cues (adjective/participle
endings, possessives, cultural markers), then deterministically assign one
voice from the pool.
**In → Out.** `04-examples.tsv` + `config/voices.tsv` →
`data/045-speaker_gender.tsv`.
**LLM.** OpenAI `gpt-4o-mini` — tool use — gender classification only. Voice
assignment is pure Python: neutral senses are split 50/50 with a seeded
shuffle, then round-robin within each gender pool (seeds 44/45/46) so per-voice
usage is balanced to ±1. Manual override: `data/_manual_speaker_gender.tsv`.
*Language-specific:* the gender-cue rules.

### Stage 5 — IPA · `build/05_ipa.py`
**Purpose.** Transcribe headword and example to IPA.
**In → Out.** `04-examples.tsv` → `data/05-ipa.tsv`.
**LLM.** Two-phase: **(1)** eSpeak-NG `pt-BR` deterministic baseline (free);
**(2)** Anthropic Claude tool-use correction for BP-specific phonetics (closed
vowels, `/ʁ/` vs `/ɾ/`, `/t/`-`/d/` palatalization, nasals). Manual override:
`data/_manual_ipa.tsv`. *Language-specific:* the eSpeak locale + the correction
prompt's phoneme rules.

### Stage 5.5 — Single-pass auditor · `build/055_audit.py`
**Purpose.** Adversarially audit every sense (find defects across 10 axes), emit
a verdict (`pass` / `regenerate` / `human_review`), and auto-regenerate fixable
failures through Stage 4.
**In → Out.** `05-ipa.tsv` → `data/055-audit.tsv`,
`data/_jury_disagreements.tsv`.
**LLM.** Anthropic Claude — tool use — one call produces both the defect list
and the verdict (a cost-driven consolidation of an originally two-pass design).
Manual override: `data/_manual_audit.tsv`.

### Stage 6 — Audio pilot · `build/06_audio_pilot.py` (and full-corpus run)
**Purpose.** Synthesize word + example audio, loudness-normalize, ASR-verify,
upload to R2, register in the manifest. Stage 6 is the 500-sense pilot; the same
reusable runner does the full corpus ("Stage 7" in the plan — no separate
script).
**In → Out.** `05-ipa.tsv` + `config/voices.tsv` → `data/_audio_manifest.tsv` +
R2 objects.
**LLM.** None for synthesis. ASR verification uses `gpt-4o-transcribe` —
[§5](#5-asr-verification--deep-dive).
**Notes.** Pipeline per clip: ElevenLabs PCM → ffmpeg loudnorm (closed-loop,
−16 LUFS ±1 LU) → MP3 → ASR roundtrip → R2 upload → manifest row. Audio
filenames bake `model_id` + version (`?v=` query strings are stripped by Anki on
mobile sync).

### Stage 8 — Pronunciation correction · `build/08_0…08_9*.py`
**Purpose.** Detect and repair clips where the TTS used non-BP phonetic priors
(English-leaning cognates, EP/Spanish/French flavoring) **even though the
transcription is correct** — failures ASR cannot see.
**Pipeline.** `08_0` deterministic risk classifier (P0–P3) → `08_1` audio judge
on the risky bucket → `08_2` human calibration with adaptive sample expansion →
`08_3` alias-respelling generation (sentinel smoke test + Claude per-word) →
`08_4` upload an ElevenLabs pronunciation dictionary → `08_5` re-render affected
clips with the dictionary attached + verify → `08_6` before/after listening gate
+ TSV rebuild.
**LLM.** Audio judge: OpenAI `gpt-4o-audio-preview`
([§6](#6-audio-judgment--deep-dive)). Alias generation: Anthropic Claude tool
use. *Language-specific:* the risk patterns and the respelling templates.

### Stage 9 — Migration to ElevenLabs Flash v2.5 · `build/09_0…09_4.py`
**Purpose.** A pilot showed `eleven_flash_v2_5` handled BP phonotactics natively
(45/46 comparison pairs favored Flash) at half the per-character cost — so the
whole deck (11,450 clips) was re-rendered and the 25-rule alias dictionary
collapsed to 1 rule.
**Steps.** Full re-render → an ASR-flagged human-review HTML queue
(`LEAVE` / `REGEN_VOICE_SWAP` / `TAG_ALIAS`) → an audio-judge audit of the
`LEAVE` decisions → `derive_final.py` rebuilds `06-final.tsv` → R2 orphan
cleanup. **Result: 99.95% audio-verified-BP rate.**
**LLM.** `gpt-4o-audio-preview` for the LEAVE audit only.

### Stage 10 — English back-of-card audio · `build/10_0_render_en_audio.py`
**Purpose.** Render the English translation of each example (`en_ex` clip type)
as a back-of-card pronunciation reference, gender-matched to the BP voice.
**In → Out.** adds `en_ex` rows to `data/_audio_manifest.tsv` + R2 objects; adds
3 columns to `06-final.tsv`.
**LLM.** None.

### Stage 11 — Migration to `eleven_v3` · `build/11_0…11_2.py`
**Purpose.** Full-deck migration to ElevenLabs `eleven_v3` with the v6
pronunciation dictionary. `11_0` pilots 100 senses; `11_1` re-renders the entire
deck (17,174 of 17,175 clips on v3 — one `2801.00.01 word` "ó" interjection
holdout stays on Flash for a loudnorm edge case); `11_2` auditions tricky cases.
**LLM.** None for synthesis (ASR verification as in Stage 6).
**Notes.** This is the current production TTS model. Earlier in this stage a BP
male voice ("Lair") was retired for EP drift on common verbs and its 476 senses
surgically swapped to another voice — see `config/voices.tsv` notes.

### Stages 12–15 — LLM metadata classification
Each adds learner-facing structured metadata to `06-final.tsv` via a sidecar
TSV + `derive_final.py`. All use **Anthropic Claude Sonnet (4.6-class), tool use
+ caching**, gated by deterministic pre-rules that skip the LLM where possible.

| Stage | Script | Adds | Coverage |
|---|---|---|---|
| **12** Usage hints | `12_0_usage_hint_classifier.py` | `usage_hint`, `usage_hint_priority`, `risk_note` — learner trap warnings on verb/idiom rows | ~1% ship rate (essential/useful only) |
| **13** Family roots | `13_0_family_root_classifier.py` | `family_root` — derivational-family head (`decidir`/`decisão`/`decisivo`) | 31.9% of content words |
| **14** Topic tags | `14_0_topic_classifier.py` | exactly one `#topic-*` tag (50-tag frozen taxonomy in `build/lib/topic_tag_rules.py`), merged into `tags` | 100% of senses |
| **15** Risk register | `15_0_risk_register_classifier.py` | `bp_validity` (7 enum), `register` (10 enum), `risk_flags` (20-flag pipe list) — splits three overloaded safety concepts apart | 100% of senses (~320 via LLM, rest deterministic default) |

### Stage 16 — SQLite mirror + multimodal audio-judge QA gate
**16.0 — SQLite mirror** (`16_0_build_sqlite.py`): rebuilds `data/06-final.sqlite`
from the canonical TSVs — `senses` (mirrors all 40 columns) plus exploded
`sense_tags`, `sense_risk_flags`, `sense_en_glosses`, joined `audio_clips`,
`voices`, and an FTS5 index. Idempotent (wipes + recreates). 26 parity tests
enforce TSV↔SQLite faithfulness.
**16.1–16.10 — the audio-judge suite.** Documented in full in
[§6 Audio judgment](#6-audio-judgment--deep-dive): an A/B bake-off of three
multimodal judges, a deck-wide Gemini QA gate that found 222 drifting clips, a
surgical re-render loop that resolved 221 of them, and an escape-hatch voice for
the last one. **End state: 0 production drifts across 11,449 BP `eleven_v3`
clips.**

### Stage 17 — Deterministic SQL ordering + clean Anki export
**Purpose.** Replace the deck's frequency-rank row order with a pedagogical
teaching order, and emit the learner-facing Anki import.
**`17_0_build_ordering.py`** — builds `topic_order` + `anki_ordering` tables
inside the SQLite mirror via SQL window functions. Order =
`topic_index → topic_bucket → topic_subrank → sense_id`, then a polysemy-spacing
pass (the N senses of a polysemous lemma are pushed onto successive "rounds" so
they don't bunch together). Topic order comes from the frozen Stage-14 taxonomy;
five high-impact topics (`grammar`, `numbers`, `time-calendar`, `measurement`,
`movement-position`) get hand-curated semantic sub-buckets in
`build/lib/order_rules.py` (100% lemma coverage). Also dumps `data/_ordering.tsv`.
Manual override: `data/_manual_ordering.tsv` (strict-validated header).
**`17_1_export_anki.py`** — writes `data/07-anki-listening-general.tsv` (25
learner/card/filter columns, audio as `[sound:…]` tags, generated `anki_tags`,
`#separator:tab` import directives) and `data/07-anki-media-manifest.tsv` (the
17,175 raw audio URLs + md5s to fetch).
**`17_2_ordering_html.py`** — `reports/17_ordering.html`, the ordering review
report.
**LLM.** None — pure deterministic SQL + Python.

### Stage 19 — ElevenLabs `eleven_v4` re-render + IPA v2 · `build/19_0…19_5*.py`
**Purpose.** Move the 11,450 Portuguese word + example clips from `eleven_v3` to
`eleven_v4` (released 2026-09-28) at the best quality the pipeline can *verify*,
and replace the Stage-05 IPA with one validated São Paulo convention for the
cards. English `en_ex` clips stay on v3. Guardrail: a v3 clip is never replaced
by a v4 take that fails QA.
**`19_0_preflight.py`** — free + paid smoke test of every key and constraint
(subscription tier/credits, voices on v4, model lists, R2 write, local tools),
IPA-adherence probes, direction-tag probes, one call per judge/ASR candidate.
Finding: inline IPA (`"/…/"`, bare `/…/`, in context, dictionary phoneme rules)
does **not** steer v4 for Portuguese (stress pairs 3–5/20 vs 16/20 for plain
spelling) → TTS input stays plain text. v4 bills ~0.13 credits/char.
**`19_1_ipa_v2.py`** + `build/lib/bp_ipa.py` — IPA v2 for every word and
sentence: Stage-05 majority vote per spelling, notation normalization (stress
before the onset, strong r → `h`, coda r → `ɾ`, `tʃ/dʒ`, final `i/u/ɐ`, coda-s
voicing across words, clitic weak forms), validators (eSpeak stress oracle by
position + vowel class, written-accent rules, final-vowel rule, MFA soft
check); 255 conflicts + 310 heterophone occurrences + 32 number sentences
adjudicated by `claude-opus-5-5`; independent `claude-fable-5-1` audit
(95/100 consensus, 92/100 LLM-decided). `derive_final.py` applies
`_manual_ipa` > `_ipa_v2` > `05-ipa` and the EP→BP spelling map.
**`19_2_model_selection.py`** — judge + ASR bake-off on 239 clips the user had
labeled. Winner: `gemini-3.1-pro-preview` AND `gemini-3.8-flash`, both with the
expected-IPA prompt (J2): 94 % recall on mispronounced clips (production J1:
77 %). OpenAI `gpt-audio` rejects 70 %+ of good clips. ASR stays
`gpt-4o-transcribe` (`gpt-transcribe` silently "fixes" dropped final -r).
**`19_3_pilot.py`** — 685-sense pilot through the production engine
(`build/lib/v4_tts.py`, `build/lib/audio_qa.py`): arms, v3 baseline under the
same gate, a blind listening page, pre-registered decisions →
`config/stage19_policy.tsv`.
**`19_4_render_v4.py`** — full run: per-clip ladder (best-of-2 words, re-takes,
same-gender voice swaps), acceptance = PCM sanity ∧ loudness ∧ relaxed ASR ∧
both judges `bp_ok`; winners only uploaded as `…-eleven_v4-v{N}.mp3`, manifest
rows rewritten only on accept, unresolved clips keep v3 (`_v4_unresolved.tsv`).
**`19_5_qa_report.py`** — coverage / failure-reason report + residue page.
**LLM.** Claude Opus 5.5 + Fable 5.1 (IPA), Gemini 3.1 Pro + 3.8 Flash (judges),
`gpt-4o-transcribe` (ASR).

---

## 5. ASR verification — deep dive

**Module: `build/lib/asr.py`.** Used by every audio stage (6, 8.5, 9, 11, 16.9).

### What it checks, and what it can't

ASR roundtrip answers exactly one question: **did the TTS engine pronounce the
correct *words*?** It catches dropped syllables, substituted words, mangled
morphology, and outright garbage. It is **content** verification.

It **cannot** catch a *correctly-worded* clip spoken in the wrong accent — an
English-accented "animal" transcribes perfectly as "animal". That failure mode
is the entire reason the [audio judge](#6-audio-judgment--deep-dive) exists.
The two are complementary; neither replaces the other.

### The model — locked by A/B

`gpt-4o-transcribe` is the locked default, chosen by an A/B on 200 cached clips:

| Model | Pass rate | Notes |
|---|---|---|
| **`gpt-4o-transcribe`** | **99.0%** | winner; ~7× fewer false regenerations on short PT function words |
| `whisper-1` | 95.5% | |
| `gpt-4o-mini-transcribe` | 95.0% | occasional cross-language hallucinations on isolated phonemes |
| `elevenlabs/scribe_v2` | 90.0% | |

Same per-minute price as `whisper-1` ($0.006/min audio). `temperature=0.0`.
Retries: 5 attempts, exponential backoff, retryable on `{408, 409, 429, 500,
502, 503, 504}`.

### The roundtrip flow

```
synthesized MP3
   │
   ├─ clip_type == "word":
   │     pad 0.5 s silence at both ends   (ffmpeg; clips < 1 s make ASR hallucinate)
   │     transcribe with a BIASED prompt: "Palavra em português brasileiro: {input}"
   │     │
   │     ├─ input ≤ 3 normalized chars  → PHONETIC-ONLY decision path  (see below)
   │     │
   │     └─ longer input → text-similarity path:
   │           if biased transcript == input verbatim (sim ≥ 0.999):
   │              run a SECOND, UNBIASED pass and take the WORSE of the two
   │              (guards against the prompt feeding the model the answer)
   │
   └─ clip_type == "example":
         transcribe directly — no padding, no bias (long enough to be robust)

→ normalize both texts (lowercase, strip diacritics, drop punctuation, collapse whitespace)
→ rapidfuzz Levenshtein normalized similarity ∈ [0,1]
→ decision
```

### Length-aware decision policy

This is the heart of the module. Whisper-family models hallucinate badly on
isolated 1–3-character function words (`o`, `de`, `em` → "OU", "G", "PING"), so
text similarity is useless there. Two branches:

| Input length (after normalization) | Signal | Pass threshold |
|---|---|---|
| **≤ 3 chars** (function words) | **Phonetic distance** — eSpeak-NG IPA of input vs IPA of the transcript, Levenshtein distance | distance **≤ 0.40** |
| **> 3 chars**, top-1000 rank | Text Levenshtein similarity | **≥ 0.95** |
| **> 3 chars**, long-tail | Text Levenshtein similarity | **≥ 0.92** |

For long inputs, phonetic distance is computed too but is only a **secondary
diagnostic** (warning at 0.30) — it is *not* a hard gate, because a
mispronunciation the ASR consistently mishears the same way is rare and is
caught later by the audio judge or at study time.

If eSpeak-NG cannot produce IPA for a short input, the path falls back to a
strict text-similarity check (`≥ 0.99`).

### Decisions and what they trigger

`AsrResult.decision` is one of:

- **`pass`** — similarity/distance within threshold; clip accepted.
- **`regen`** — below threshold; queued for re-render (≤ 2 retries).
- **`human`** — still failing after retries; routed to a human-review HTML
  queue (this is how Stage 9 produced its 99-row review queue).

The result also carries `transcript`, `biased_transcript`, `text_similarity`,
`phonetic_distance`, `threshold_used`, `attempts`, `cost_usd`, and a free-text
`notes` explaining the path taken. The manifest persists `asr_transcript`,
`asr_similarity`, `asr_decision`.

### Language-specific vs language-agnostic

| Language-specific | Language-agnostic |
|---|---|
| ASR `language` code (`pt`) | the whole roundtrip algorithm |
| the biased word prompt wording | silence padding, biased+unbiased cross-check |
| eSpeak-NG locale for phonetic fallback | the length-aware two-branch policy |
| threshold tuning (if a language needs it) | `pass`/`regen`/`human` decision model |

To re-target: change the `language` code, translate the one-line biased prompt,
point eSpeak-NG at the new locale. The thresholds are reasonable defaults for
any language.

---

## 6. Audio judgment — deep dive

**Modules: `build/lib/audio_judge.py` (OpenAI) and
`build/lib/gemini_audio_judge.py` (Gemini).** Used by Stages 8, 9, and 16.

### What it checks, and why it's separate from ASR

The audio judge answers: **does the clip *sound* like the target dialect?** A
multimodal LLM listens to the audio and rates the *accent* — independent of
whether the words are correct. It catches the failure ASR is blind to: a clip
whose transcription is perfect but whose pronunciation drifted toward English,
European Portuguese, Spanish, or French.

### The verdict schema (identical across both backends)

```
pronunciation_verdict : "bp_ok" | "non_bp" | "unclear"   ← primary verdict
drift                 : "EN" | "EP" | "ES" | "FR" | "other" | "none"  ← drift source if non_bp
severity              : "low" | "medium" | "high"        ← how acoustically strong
confidence            : "low" | "medium" | "high"        ← the model's own certainty
evidence              : str                              ← ONE concrete acoustic cue
```

### The system prompt — phonetic features, not transcription

The judge is told to **focus on phonetics only** (not transcription, not audio
fidelity, not voice quality) and is given an explicit BP feature checklist:

- Final `-l` → `[w]` (vocalized), **not** consonantal `[l]`.
- Final `-de` / `-te` → `[dʒi]` / `[tʃi]` (palatalized), **not** silent `[d]`/`[t]` (= EP).
- Initial `r` / `rr` → `[h]` or `[χ]` (back fricative), **not** French uvular `[ʁ]`.
- Coda `s` → `[s]`/`[z]`, **not** `[ʃ]` (= EP).
- Final unstressed `-e` → `[i]`, **not** `[ɨ]`/silent (= EP).
- Cognate stress (`animal`, `hospital`, `social`) → final syllable, **not** initial (= EN).
- Silent initial `h` → no aspiration, **not** English aspirated `[h]`.

…plus the acoustic signature of each of the four `drift` directions (EN, EP, ES,
FR). `evidence` must cite one concrete cue (e.g. *"final /l/ vocalized to [w]
confirms BP"*).

### Two interchangeable backends

| | `AudioJudgeClient` (OpenAI) | `GeminiAudioJudgeClient` (Google) |
|---|---|---|
| Model | `gpt-4o-audio-preview` | `gemini-3.1-pro-preview` |
| Audio in | base64, chat-completions `input_audio` | `types.Part.from_bytes` |
| Structured output | tool use (`judge_pronunciation`) | `response_schema` + JSON mime |
| Cost / ~3 s clip | ~$0.005 | **~$0.0002** (~25× cheaper) |
| Quirks | — | sometimes leaks a JSON preamble despite the schema → a lenient brace-counting parser recovers the first `{…}` block; `thinking_budget=1024`, `max_output_tokens=5000` |

Both share the verdict schema, both retry with full-jitter backoff (6 attempts,
60 s cap), both append a per-call `audit/*.jsonl` record (verdict, drift,
severity, confidence, evidence, latency, cost, tokens). They are drop-in
interchangeable.

### History — how the production judge was chosen

**Stage 8.1 / 9.3.** `gpt-4o-audio-preview` was the first audio judge — used to
triage the risky-bucket pronunciation failures and to audit Stage 9's ASR
`LEAVE` decisions.

**Stage 16.1–16.7 — the A/B bake-off.** Three multimodal judges
(`gpt-4o-audio-preview`, `gpt-audio-1.5`, `gemini-3.1-pro-preview`) were run over
an **800-clip pool** (200 BP-voiced senses × {word, example} + 200 EP-voiced
control clips × {word, example}). Forty clips were hand-labelled by ear as the
gold standard. Result:

| Judge | Agreement with human ear | non-BP recall | non-BP precision |
|---|---|---|---|
| **`gemini-3.1-pro-preview`** | **92.5%** | **100%** | **96.7%** |
| `gpt-4o-audio-preview` | ~40% (≈ coin-flip) | — | — |
| `gpt-audio-1.5` | ~40% (≈ coin-flip) | — | — |

Gemini was locked as the production judge. The OpenAI audio judges were retired
from this role.

**Stage 16.8 — the deck-wide QA gate** (`16_8_gemini_qa_gate.py`). Gemini judged
**all 11,449 BP `eleven_v3` word + example clips**. To avoid a macOS DNS storm at
high concurrency, `16_8_prefetch_audio.py` first bulk-downloads every clip to a
local cache; the judge then reads from disk. Output: `data/_gemini_drift_
watchlist.tsv` — **222 clips** Gemini flagged as `non_bp`.

**Stage 16.9 — surgical re-render** (`16_9_surgical_rerender.py`). A per-drift
state machine:

- **Phase A — same-voice retry** (≤ 2 attempts): re-render with the *same* voice.
  ElevenLabs synthesis is stochastic, so a re-roll often lands a clean take.
- **Phase B — voice swap**: if retries fail, swap to another active voice of the
  same gender, tried in ascending deck-wide drift-rate order (cleanest voice
  first).
- **Phase C — hard case**: if every option fails, revert to the original and log
  it to `data/_audio_known_drifts.tsv`.

Each re-render is judged by Gemini before being accepted. **Result: 222 → 1**
(111 fixed by same-voice retry, 110 by voice swap).

**Stage 16.10 — escape-hatch voice** (`16_10_try_voice_on_sense.py`). The single
hard case ("a fortuna", where both active female voices produced EP vowel
reduction) was fixed by adding a fresh ElevenLabs voice to `config/voices.tsv`
with `status = active_escape_hatch` (documented, but excluded from round-robin
assignment) and rendering that one clip with it. **Final state: 0 drifts.**

### Language-specific vs language-agnostic

| Language-specific | Language-agnostic |
|---|---|
| the BP phonetic-feature checklist in the system prompt | the verdict schema (`verdict`/`drift`/`severity`/`confidence`/`evidence`) |
| the `drift` enum (which neighboring languages/varieties to name) | both client wrappers, retry, audit logging |
| which clips count as "control" in an A/B pool | the A/B bake-off methodology + human-label scoring |
| — | the QA-gate → drift-watchlist → surgical-rerender → escape-hatch loop |

To re-target: rewrite the system prompt's feature checklist for the new target
variety and its plausible drift directions; everything else — the schema, the
two backends, the A/B harness, the QA gate, the re-render state machine — is
reused unchanged.

---

## 7. The final TSV schema

`data/06-final.tsv` — the canonical enriched master, **40 columns**, ~5,725 rows,
one per sense. Rebuilt by `build/derive_final.py` from the stage outputs +
sidecars. **Never reordered or mutated in place** — Stage 17 reads it, never
writes it.

| # | Column | Source stage | Description |
|---|---|---|---|
| 1 | `sense_id` | 2 | stable `RRRR.EE.SS` key |
| 2 | `rank` | 1a | frequency rank |
| 3 | `source_pt` | 1a | original headword (audit) |
| 4 | `pt` | 1b/1c | BP-normalized headword |
| 5 | `pt_type` | 1b | single_word / hyphenated / space_compound / idiom / abbreviation_expansion |
| 6 | `gender` | 3 | `o` / `a` / `o/a` / — |
| 7 | `pt_display` | 3 | display form with article |
| 8 | `pos` | 3 | part of speech |
| 9 | `sense_index` | 2 | sense number within the row |
| 10 | `en_primary` | 2 | single English gloss for this sense |
| 11 | `en_all` | 1a | full original RHS (audit) |
| 12 | `annotation` | 3 | dialect/number/gender/reflexive metadata |
| 13 | `bp_status` | 1.5 | standard / uncommon / false_friend / nsfw (superseded by Stage 15; kept for provenance) |
| 14 | `normalization_action` | 1b/1c | none / spelling / lexical / hyphen / idiom_expansion |
| 15 | `ipa_word` | 5 | IPA of the headword |
| 16 | `example_pt` | 4 | BP example sentence (≤ 15 words) |
| 17 | `example_en` | 4 | English translation |
| 18 | `target_word_used` | 4 | exact surface form in `example_pt` |
| 19 | `ipa_example` | 5 | per-word IPA of the example |
| 20 | `audio_word` | 6–11 | R2 URL — headword clip |
| 21 | `audio_example` | 6–11 | R2 URL — example clip |
| 22 | `audio_word_md5` | 6–11 | md5 of the headword clip |
| 23 | `audio_example_md5` | 6–11 | md5 of the example clip |
| 24 | `audio_en_example` | 10 | R2 URL — English example clip |
| 25 | `audio_en_example_md5` | 10 | md5 of the English example clip |
| 26 | `voice_id` | 4.5 | ElevenLabs BP voice |
| 27 | `voice_id_en` | 10 | ElevenLabs English voice |
| 28 | `voice_gender` | 4.5 | `m` / `f` |
| 29 | `family_root` | 13 | derivational-family head |
| 30 | `tags` | 3 + 14 | space-separated tags incl. exactly one `#topic-*` |
| 31 | `source_line` | 1a | raw source line (audit) |
| 32 | `source_line_number` | 1a | ledger join key |
| 33 | `notes` | various | escape-hatch annotations |
| 34 | `pt_display_safe` | 10.5 | POS-disambiguated display form |
| 35 | `usage_hint` | 12 | learner trap hint |
| 36 | `usage_hint_priority` | 12 | essential / useful / "" |
| 37 | `risk_note` | 12 + 15 | learner-facing warning |
| 38 | `bp_validity` | 15 | standard / rare_in_bp / ep_leaning / ep_only / regional_br / nonstandard / uncertain |
| 39 | `register` | 15 | neutral / informal / formal / technical / literary / archaic / slang / vulgar / taboo / uncertain |
| 40 | `risk_flags` | 15 | `none` or a pipe-list from a 20-flag allowlist |

The Anki export (`07-anki-listening-general.tsv`) is a **clean 25-column subset**
of this — learner/card/filter fields only, no md5s, no provenance, no raw URLs
(audio becomes `[sound:…]` tags).

---

## 8. Repository layout

```
build/
  NN_*.py            stage entry-point scripts (01a … 17_2)
  derive_final.py    rebuilds data/06-final.tsv from stage outputs + sidecars
  verify_all.py      cross-stage invariant checker (hard checks + soft warnings)
  lib/               shared modules — see below
  prompts/           LLM prompt templates (one per stage)
  policies/          locked policy docs (pronoun strategy, function words)
config/
  voices.tsv         the voice pool: voice_id, gender, pool_index, bp_name,
                     en_voice_id, en_name, status, notes
data/
  source.txt         immutable input
  _source_ledger.tsv, _audio_manifest.tsv   the two authoritative artifacts
  _manual_*.tsv      manual overrides (always win over the LLM)
  _ep_*, _hyphen_rules.tsv, _lexical_bp_replacements.tsv   language rule tables
  NN-*.tsv           per-stage outputs (01-normalized … 06-final)
  06-final.tsv       the canonical enriched master (40 cols)
  06-final.sqlite    the SQL mirror (gitignored; rebuilt by 16_0)
  07-anki-*.tsv      the deliverables
audit/               per-stage JSONL logs + review HTML (audit/*.jsonl gitignored)
reports/             review HTML (topic ordering, classifier pilots)
tests/               pytest per-stage unit + parity tests
docs/plan.md         the authoritative specification
```

### `build/lib/` modules

| Module | Role |
|---|---|
| `llm.py` | Anthropic client wrapper — caching, batch API, tool use, retries |
| `rate_limit.py` | sliding-window RPM limiter |
| `tsv.py` | TSV read/write (`excel-tab`, `QUOTE_MINIMAL`) |
| `ledger.py` | source-ledger helpers |
| `parse.py`, `lexical.py`, `bp_rules.py`, `sense_split.py`, `enrich.py` | Stage 1–3 logic |
| `example_gen.py`, `validate.py`, `gloss_correction.py` | Stage 4 / 5.5 logic |
| `ipa.py` | eSpeak-NG IPA transcription |
| `elevenlabs_client.py` | TTS client — locked voice settings, retry, seed determinism, pronunciation-dictionary locators |
| `loudness.py` | closed-loop ffmpeg loudnorm (−16 LUFS ±1 LU, verified) |
| `r2_client.py` | Cloudflare R2 upload/download/delete |
| `audio_manifest.py` | manifest schema, `bump_version`, status lifecycle |
| `voices.py` | voice-pool loader + round-robin assignment |
| **`asr.py`** | **ASR roundtrip — [§5](#5-asr-verification--deep-dive)** |
| **`audio_judge.py`**, **`gemini_audio_judge.py`** | **audio judges — [§6](#6-audio-judgment--deep-dive)** |
| `audio_logger.py`, `progress.py` | live progress / monitoring |
| `topic_tag_rules.py`, `family_root_rules.py`, `risk_register_rules.py`, `usage_hint_rules.py` | Stage 12–15 taxonomies + deterministic pre-rules |
| `order_rules.py` | Stage 17 topic order + custom bucket curation |

---

## 9. Testing & verification

- **`pytest tests/`** — per-stage unit and parity tests
  (`test_stage_1a/1b/1c/2/3/4/45/5/55/6/15`, `test_stage_16_sqlite` with 26
  TSV↔SQLite parity checks, `test_stage_17_ordering` with 42 ordering/export
  checks, `test_stage_18` for the 1.8 dedupe logic), plus LLM-infra tests
  (`test_llm_batch`, `test_llm_tier`) and an end-to-end `test_smoke_sample_1000`.
- **`build/verify_all.py`** — the cross-stage invariant checker. Hard checks
  (row count, schema, `sense_id` uniqueness/pattern, audio URLs reachable and on
  the expected model, voice resolution, every Stage 12–15 enum/format
  invariant) exit non-zero on failure; soft checks print warnings. Currently:
  **0 hard failures, 5 benign soft warnings** (notably the 1 intentional
  legacy-Flash holdout clip). `--http-sample N` controls how many audio URLs
  are HEAD-checked.
- **Golden rule:** run `pytest tests/` and `verify_all.py` before running any
  stage on the full corpus.

---

## 10. Porting to another language

The pipeline architecture — the ledger/manifest state model, `sense_id`,
idempotency, the audit trail, the stage DAG, the ASR roundtrip algorithm, the
audio-judge harness, the synth→loudnorm→upload→manifest loop, the SQLite
mirror, the ordering/export — is **language-agnostic**. Re-targeting is a matter
of swapping a bounded set of language-specific assets.

### What you swap

| Asset | Where | Notes |
|---|---|---|
| Frequency dictionary | `data/source.txt` | the immutable input; any "headword = gloss" list |
| Dialect/variety normalization | `data/_ep_spelling_map.tsv`, `_hyphen_rules.tsv`, `_lexical_bp_replacements.tsv` | encodes EP→BP; for a language with no variety split, these can be near-empty |
| Validity & register taxonomies | Stage 1.5 prompt, `build/lib/risk_register_rules.py` | the `bp_status` / `bp_validity` / `register` category sets |
| Pronoun & sensitive-term policy | `build/policies/`, Stage 2.5 keyword list | language-specific |
| IPA | eSpeak-NG locale + Stage 5 correction prompt | swap the locale; rewrite the phoneme-correction rules |
| Voice pool | `config/voices.tsv` | target-language native voices |
| TTS pronunciation dictionary | Stage 8 alias seeds + `_pronunciation_aliases.tsv` | language-specific respellings; may be empty if the TTS model handles the language natively |
| **ASR language code + biased prompt** | `build/lib/asr.py` | one constant + one one-line prompt — [§5](#5-asr-verification--deep-dive) |
| **Audio-judge feature checklist** | the `JUDGE_SYSTEM_PROMPT` in both audio-judge modules | the target-variety phonetic features and plausible `drift` directions — [§6](#6-audio-judgment--deep-dive) |
| Topic taxonomy | `build/lib/topic_tag_rules.py` | the 50-tag set is fairly universal (family, body, food, …); the Stage-17 custom-order buckets in `order_rules.py` reference specific lemmas and need re-curation |
| Prompt templates | `build/prompts/*.md` | translate/adapt each |

### What you keep unchanged

The ledger and manifest schemas; `sense_id`; the idempotency + audit-JSONL
pattern; the manual-override hierarchy; the ASR roundtrip *algorithm*
(silence padding, biased/unbiased cross-check, length-aware two-branch
decision, phonetic fallback); the audio-judge *schema and harness* (verdict
fields, both client wrappers, the A/B bake-off methodology, the QA-gate →
drift-watchlist → surgical-rerender → escape-hatch loop); loudness
normalization; the R2 storage + versioned-filename model; the SQLite mirror;
the Stage-17 ordering/spacing/export machinery; `verify_all.py` and the test
harness.

### Porting checklist

1. Drop in `data/source.txt`; adjust the Stage 1a parser if the line format differs.
2. Author the variety-normalization rule tables (or stub them).
3. Re-curate the Stage 1.5 / 12–15 taxonomies and prompts.
4. Point eSpeak-NG at the new locale; rewrite the Stage 5 correction prompt.
5. Build `config/voices.tsv` with native voices; run a pilot.
6. Rewrite the **ASR** language code + biased prompt and the **audio-judge**
   feature checklist — the two QA axes.
7. Re-run the audio-judge A/B bake-off on a small labelled pool to pick the
   production judge for the new language (do not assume the BP winner transfers).
8. Re-curate the Stage-17 custom-order buckets in `build/lib/order_rules.py`.
9. Run `pytest tests/` + `build/verify_all.py` at every milestone.
