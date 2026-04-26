# Plan: Anki-Ready BP Portuguese Dataset (Data Phase) — v2

## Context

The user is an A1 Brazilian Portuguese learner who knows many languages but none Romance. The goal is to build a **complete, high-leverage, ledger-driven dataset** from the 4,985-entry frequency dictionary at [data/source.txt](data/source.txt) that can later drive an Anki deck. The current phase is strictly **data preparation**, not Anki card construction.

The source is EP-leaning with BP swap tags; the user wants pure BP output. Downstream tooling (Anki note type, card templates, scheduling) is out of scope for this plan.

The outcome is a single authoritative TSV plus immutable audio assets on **Cloudflare R2** (with stable HTTPS URLs in the TSV) that the user can open in a spreadsheet to review and from which an Anki deck can later be deterministically generated.

### Project promise (revised)

The original v1 promise — *"completeness is non-negotiable: no words skipped, no meanings skipped"* — over-promised while simultaneously authorizing EP auto-drops. The two are incompatible. The revised, enforceable promise is:

> **Every source line is accounted for.** The final learning dataset is BP-only. EP-only forms are mapped to BP equivalents, merged into existing BP rows, manually reviewed, or dropped with audited rationale. No source line disappears silently. No meaning is intentionally discarded without an explicit ledger entry.

Enforced by `_source_ledger.tsv` (§ Source ledger) — the backbone of the project.

## Settled decisions

| Decision | Value |
|---|---|
| Dialect | BP only; normalize EP pre-reform spellings; apply post-1990 hyphen rules |
| EP-only entries | **Action-state model**, not binary drop (see § Stage 1c) |
| BP-native cross-check | Skipped |
| Sense granularity | Sense is the unit of study; one TSV row per sense |
| Sense-split method | Claude with deterministic pre-classifier; user spot-checks multiple QA queues |
| Sense distinction rule | Two items are distinct senses iff they require different example sentences |
| **Forced sense splits** | M/F gender homographs (`capital`, `corte`, `cura`, `grama`, `banana`, `cabra`, `polícia`, `rádio`); parenthetical idioms expanded to their own rows |
| Function words / pronouns | Manual sense templates; exempt from generic slash-splitting |
| Example sentences | 1 per sense, ≤15 words, A2-background grammar; target word in typical use for THAT sense; EN translation on card |
| Register | Neutral-everyday; tag if naturally travel/legal |
| IPA | BP neutral-paulistano, broad phonemic with stress; **deterministic baseline + LLM correction** |
| **IPA sandhi gap** | Documented: per-word IPA does not reflect connected-speech word-linking present in audio. Not patched. |
| Audio | 4 clips per sense (2 voices × {word, example}); fixed 1M+1F across whole deck; **Google Cloud TTS Neural2 BR voices** (default — ~$30–60 vs. ElevenLabs' ~$260–430). ElevenLabs swap-in available if user later wants premium naturalness. Open-source local TTS (Kokoro / XTTS / MeloTTS) is the zero-cost option. |
| **Audio column naming** | `audio_word_m`, `audio_word_f`, `audio_example_m`, `audio_example_f` — never `_v1/_v2`. Eliminates voice-mix-up risk on script restart. |
| Audio stage | Pilot first 500, then full batch |
| **Audio storage** | **Cloudflare R2**, exposed via R2 public bucket custom domain (not per-object public-read ACL). Stable HTTPS URLs with **filename-baked versioning** (`-v{N}.mp3`) — Anki strips query strings on download, so `?v=N` would silently fail to propagate regenerated clips to mobile devices. |
| Audio manifest | `data/_audio_manifest.tsv` is source of truth; final TSV URL columns derived from it |
| Cognate flag | `#cognate-en` tag, no review-order change |
| Legal/travel expansion | Out of scope for now |
| Data format | TSV, one row per sense, stable `sense_id = rank.sense_index` |
| Manual overrides | First-class: every LLM stage has a corresponding `_manual_*.tsv`; overrides always win |
| LLM provenance | Recorded in stage audit JSONL files (model, prompt-hash, response-hash, confidence) |
| Model config | Centralized in `config/models.yaml`; never hard-coded in scripts |
| **Review strategy** | **LLM 3-model jury + adversarial auditor + verdict auditor + N-best on top-1000 + ASR roundtrip; human review is study-time flagging, not upfront QA queues** |
| **Audio QA** | **Whisper-class ASR roundtrip + phonetic-distance check; human listens only to ASR-flagged clips and the pilot voice-quality sample (~15 min)** |
| Study-time loop | Anki flag-for-regeneration field → weekly batch regeneration of flagged cards |
| Study app | macOS desktop + iPhone |
| LLM providers | Anthropic + OpenAI + Google (three families for jury diversity); Google Cloud TTS for audio; OpenAI Whisper-class ASR for roundtrip |
| **Batch APIs** | All offline LLM stages (1.5, 2, 4, 5, both auditors) use Anthropic Message Batches and OpenAI Batch API — **50% discount** for ≤24h turnaround. Realtime tiebreaker calls stay on synchronous APIs. Saves ~$140–240 with zero quality loss. |
| **Juror tier** | Generator and verdict-auditor use top-tier models. Jurors vote on small structured outputs (enums + short text), so they use **smaller, cheaper models** (Haiku-class + GPT-5-mini-class + Gemini-Flash-class). Adversarial auditor uses a mid-tier model (defect listing benefits from capability but doesn't need top-tier reasoning). Saves ~$80–120. |
| **Model registry** | `config/models.yaml` is the **single source of truth** for model IDs. Plan exemplars below name 2026-current frontier models; the registry is updated as models age out (e.g., GPT-4o → GPT-5 family; Claude 3.5 → Claude 4.x; Gemini 1.5 → Gemini 2.5). Stage scripts read role-keys (`generator`, `juror_a`, `juror_b`, `juror_c`, `tiebreaker`, `validator`, `auditor_verdict`, `auditor_adversarial`), never hard-coded IDs. |

## Review strategy: LLM jury, auditor, ASR — human review is post-hoc, not upfront

The v2 plan demanded ~30–40h of upfront human review across many QA queues. That is unstable: too much attention required, too easy to skim and rubber-stamp. The v3 reframe replaces almost all upfront human review with **LLM-driven review** at three tiers, plus a study-time correction loop.

### Tier 1 — Three-model jury on every LLM decision

Every LLM stage that produces a structured output (`bp_status`, sense-split classification, gender fallback, cognate, family root, sensitive-term classification, IPA correction, example generation) runs the same prompt through **three different models from three different families** — Anthropic Claude Sonnet 4.x, OpenAI GPT-5 family, Google Gemini 2.5 Pro (or whatever the registry currently names). All three responses are recorded in the audit JSONL.

Resolution rule:

- **All three agree** (same enum value; or, for free-text outputs, the validator scores them mutually semantically equivalent): accept silently. This is the common case.
- **Two-of-three agree**: accept the majority answer. Disagreement logged but does not block.
- **All three disagree**: route to a **fourth tiebreaker model** (typically the strongest available — Claude Opus 4.x or GPT-5 Pro). If the tiebreaker matches one of the three, accept. If the tiebreaker proposes a fourth answer or expresses low confidence, route to `_jury_disagreements.tsv` for human review.

Why three not two: a two-model jury fails open on shared-blind-spots — both models confidently agree on the same wrong answer. A three-model cross-family jury makes that failure mode require *three* simultaneous shared blind spots, which is empirically rare. Cost is ~3× the single-model line item (still small relative to audio).

Disagreement rate is itself a quality metric: if Stage 4 examples have >15% three-way disagreement, the prompt is broken and needs revision before further generation.

### Tier 1.5 — N-best on top-1000 headwords

The top 1000 most-frequent headwords drive ~80% of actual study time. Errors there are encountered constantly; errors at rank 7,800 are encountered rarely. The plan applies extra scrutiny **specifically** to the top 1000:

- **N=2 generation for example sentences.** Each top-1000 sense gets two candidate `(example_pt, example_en, target_word_used)` tuples generated by two different models. The verdict auditor picks the better one or rejects both (which triggers a regen with both prior candidates shown as anti-examples).
- **Stricter ASR threshold.** Top-1000 audio clips must clear Levenshtein similarity ≥ 0.95 (vs. 0.92 for the long tail). Anything lower regenerates twice before falling through.
- **Adversarial auditor required.** Below.

Cost addition: ~$15–25 (1000 senses × 1 extra generation × 1 selection call).

### Tier 2 — Adversarial auditor + verdict auditor (Stage 5.5)

Stage 5.5 runs **two auditor passes**, by two different models:

**Adversarial auditor** uses "find every fault" framing — **not** "is this OK?". The prompt is explicit: *"Your job is to find faults. List every defect you can identify in this row, even minor ones. If you find no defects, list 'none' and explain why each axis below is clean."* This adversarial framing empirically catches issues that verdict prompting waves through.

```json
{
  "type": "object",
  "properties": {
    "defects": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "axis": {"type": "string", "enum": [
            "sense_consistency", "example_uses_intended_sense", "translation_match",
            "bp_purity", "sensitive_policy", "ipa_plausibility", "target_word_token_match",
            "naturalness", "level_appropriateness", "other"
          ]},
          "severity": {"type": "string", "enum": ["low", "medium", "high"]},
          "description": {"type": "string"}
        },
        "required": ["axis", "severity", "description"]
      }
    }
  },
  "required": ["defects"]
}
```

**Verdict auditor** (different model again) reviews the row + the adversarial auditor's defect list, and decides:

```json
{
  "type": "object",
  "properties": {
    "agrees_with_defects": {"type": "array", "items": {"type": "string"}},
    "disagrees_with_defects": {"type": "array", "items": {"type": "string"}},
    "verdict": {"type": "string", "enum": ["pass", "regenerate", "human_review"]},
    "reason": {"type": "string"}
  },
  "required": ["verdict"]
}
```

Decision flow:

- `verdict = pass`: row accepted, even if low-severity defects exist (acceptable noise floor).
- `verdict = regenerate`: row jumps back to the relevant generation stage with the defect list in the prompt; ≤2 regen attempts.
- `verdict = human_review`: row written to `_jury_disagreements.tsv`.

Three different models touch each row across these two passes (adversarial + verdict + the original generator). Shared blind spots have to be 3-way coincident to slip through. Combined with Tier 1's 3-model jury, that's effectively 5-model coverage on every row.

### Tier 3 — Audio ASR roundtrip (replaces human listening QA)

After every audio clip is generated, [build/lib/asr_check.py](build/lib/asr_check.py):

1. **Pre-process the clip for Whisper.** This is essential for word-level audio: Whisper hallucinates badly on isolated short clips (< 1 sec). A 0.6-second clip of `o` or `e` will routinely transcribe as `Obrigado por assistir!`, `Amor.`, `[Música]`, or other YouTube-caption garbage. Mitigations applied **always** for `clip_type = word`, optionally for `example`:
   - **Pad with 0.5s of silence** at start and end before sending to Whisper. Brings clip length above the hallucination threshold.
   - **Set `initial_prompt = "Palavra em português brasileiro: {target_word_used}"`** to bias decoding toward the expected word and away from English/YouTube prior. This is a soft bias, not a constraint, so the ASR can still disagree if the audio is genuinely wrong.
   - **Set `temperature = 0.0`** and `condition_on_previous_text = false` to suppress free-form drift.
   - **Set `no_speech_threshold = 0.6`** (looser than default) so silent-prefix clips don't classify as no-speech.
   - **Cross-check with a second ASR call** without `initial_prompt` for any clip whose transcript exactly matches the prompt-biased target — if the unbiased pass also returns the target (or a near match), accept; if it returns garbage, the prompt was lying for us, regenerate.
2. Transcribes the (padded, prompted) clip with the registry-configured ASR model (current frontier multilingual: OpenAI `whisper-large-v3` or successor; the registry tracks the active ID), `language=pt`.
3. Computes normalized **Levenshtein similarity** between transcript and the input text.
4. Computes **phonetic distance** between the transcript's IPA (via eSpeak-NG roundtrip) and the expected `ipa_word_final` / `ipa_example_final`.

Decision rule per clip:

- **Long tail** (rank > 1000): Levenshtein ≥ 0.92 AND phonetic distance ≤ threshold → pass.
- **Top 1000** (rank ≤ 1000): Levenshtein ≥ **0.95** AND phonetic distance ≤ stricter threshold → pass. Top-1000 is the dominant share of study time, so cleanliness there matters disproportionately.
- Levenshtein in the warning band → **regenerate once with bumped version (new filename `-v{N+1}.mp3`)** (often fixes bad TTS prosody).
- Levenshtein well below threshold → **regenerate twice**; if still failing, route to `_audio_human_review.tsv` (the only audio rows a human ever listens to).
- Phonetic-distance outlier with Levenshtein pass → flag but accept (likely TTS mispronunciation that the ASR also mishears consistently — rare; weekly study-time loop catches them).

This catches the catastrophic failure mode (audio says something clearly wrong) without requiring human listening. Cost: ~$10–30 for ASR on ~38k clips. Time saved: ~10–20 hours of listening.

### Tier 4 — Study-time correction loop

Anki note type includes a `flag_for_regeneration` field exposed as a one-tap action during review. When the user encounters a card that is wrong (bad audio, wrong gender, awkward example, wrong sense), they tap the flag. Once a week:

1. Anki export → `_study_flags.tsv`.
2. `python build/regenerate_flagged.py` reads the flagged `sense_id`s, jumps to the relevant stage, regenerates, bumps audio `version` (which produces a new filename `-v{N+1}.mp3`), uploads new R2 object, updates manifest, re-derives final TSV.
3. Anki sees a net-new filename in the note and pulls it cleanly across desktop and mobile on next sync.

This distributes correction across the study lifetime instead of front-loading it, and only attacks rows that actually matter to the user.

### Human gates that remain (~2 hours total, lifetime)

- **Voice ID selection** (~10 min): pick male and female ElevenLabs voices.
- **Audio pilot voice quality sample** (~15 min): listen to 10 random pilot clips (post-ASR-roundtrip) to confirm the chosen voices sound right. Sanity check, not pronunciation review.
- **`_jury_disagreements.tsv` review** (~30–60 min): scan the LLM disagreement queue at the end. Most rows are easy 5-second decisions.
- **`_audio_human_review.tsv` review** (~10–20 min): listen to the clips that failed ASR roundtrip twice. Probably <50 clips.
- **Final 50-row sanity scroll on `06-final.tsv`** (~30 min): open in Numbers, eyeball 50 random rows end-to-end. Catches systemic errors the jury and auditor missed.

That is the complete human attention budget. Everything else runs unattended.

## Source ledger (the backbone)

`data/_source_ledger.tsv` is generated in Stage 1 and updated at every subsequent stage. It is the **single source of truth** for what happened to each input line.

### Schema

| Column | Type | Notes |
|---|---|---|
| `source_line_number` | int | 1-indexed, matches `source.txt` line number |
| `rank` | int | Parsed rank field from the line |
| `source_raw` | string | Verbatim line content |
| `source_pt` | string | Headword as written in source (pre-normalization) |
| `source_en_all` | string | RHS as written in source |
| `action` | enum | `keep` / `normalize_spelling` / `replace_with_bp_equivalent` / `merge_into_existing_bp_row` / `keep_tag_rare` / `manual_review` / `drop_ep_only` / `expand_idiom_added` |
| `normalization_action` | enum | `none` / `spelling` / `lexical` / `hyphen` / `idiom_expansion` (composable as comma-separated when multiple apply) |
| `bp_replacement` | string | Target BP form when `action = replace_with_bp_equivalent` |
| `merge_target_rank` | int | When `action = merge_into_existing_bp_row` |
| `merge_target_pt` | string | Resolved BP target for the merge |
| `output_sense_ids` | string | Comma-separated `sense_id`s produced from this line (filled by Stage 2) |
| `drop_reason` | string | When `action = drop_ep_only`; required |
| `manual_review_status` | enum | `not_required` / `pending` / `approved` / `rejected` |
| `stage_decided` | string | Stage that set the current `action` (`1a`, `1b`, `1c`, `1.5`, `manual`) |
| `notes` | string | Free text |

### Required invariants (`build/verify_all.py::verify_ledger`)

- Every line in `source.txt` appears exactly once in the ledger (`source_line_number` is a primary key).
- Every final sense row in `06-final.tsv` traces back to ≥1 ledger row via `output_sense_ids`.
- Every ledger row with `action = drop_ep_only` has a non-empty `drop_reason`.
- Every ledger row with `action = replace_with_bp_equivalent` has a non-empty `bp_replacement`.
- Every ledger row with `action = merge_into_existing_bp_row` has a non-empty `merge_target_rank` and `merge_target_pt`.
- Every ledger row with `manual_review_status = pending` blocks Stage 4 (example generation) for any `output_sense_ids` it produces.
- Ledger math (replaces fixed row-count expectations):
  ```
  source_line_count = kept + dropped + merged + replaced + manual_review_pending
  normalized_row_count = kept_or_replaced + idiom_expansion_rows − merged_duplicate_rows
  ```

## Repo layout

```
anki-mastery/
├── data/
│   ├── source.txt                          # EXISTING — never mutated
│   ├── _source_ledger.tsv                  # backbone; one row per source line
│   ├── 01-normalized.tsv                   # post orthographic normalization + idiom expansion
│   ├── 012-lexical_replaced.tsv            # post lexical EP→BP replacement + collision merge
│   ├── 015-bp_status.tsv                   # post cheap LLM BP-vocab classification
│   ├── 018-deduped.tsv                     # post duplicate detection + collision resolution
│   ├── 02-senses.tsv                       # post sense-split (one row per sense)
│   ├── 03-enriched.tsv                     # post gender, PoS, family, tags, cognate
│   ├── 04-examples.tsv                     # post example PT + EN + target_word_used
│   ├── 05-ipa.tsv                          # post IPA (baseline + correction) for word + example
│   ├── 06-final.tsv                        # FINAL DELIVERABLE (R2 audio URLs + md5)
│   ├── _ep_spelling_map.tsv                # orthographic EP→BP rules (accents, ct/pt clusters)
│   ├── _lexical_bp_replacements.tsv        # lexical EP→BP swaps (comboio→trem, equipa→equipe)
│   ├── _hyphen_rules.tsv                   # post-1990 hyphen normalization rules
│   ├── _idioms_expanded.tsv                # audit: parenthetical idioms expanded into new rows
│   ├── _ep_drops.tsv                       # audit: drop_ep_only entries (with reasons)
│   ├── _ep_drop_or_replace_review.tsv      # manual queue for ambiguous EP/BP cases
│   ├── _normalized_duplicates.tsv          # collisions after normalization (merge or split decisions)
│   ├── _flags.tsv                          # structured (BP)/(EP)/false-friend/NSFW review queue
│   ├── _sensitive_terms.tsv                # screened terms requiring neutral-example policy
│   ├── _audio_manifest.tsv                 # source of truth for every audio clip
│   ├── _pilot_500.tsv                      # first 500 senses for audio QA gate
│   ├── _example_fixes.tsv                  # Stage 4 failures routed for re-generation
│   ├── _manual_bp_status.tsv               # overrides: bp_status
│   ├── _manual_sense_splits.tsv            # overrides: sense decisions
│   ├── _manual_gender.tsv                  # overrides: gender
│   ├── _manual_examples.tsv                # overrides: example_pt / example_en / target_word_used
│   ├── _manual_ipa.tsv                     # overrides: ipa_word / ipa_example
│   ├── _manual_audio.tsv                   # overrides: audio re-generation directives
│   ├── _sense_review_top1000.tsv           # QA queue: top 1000 headwords
│   ├── _sense_review_low_confidence.tsv    # QA queue: low-confidence LLM splits
│   ├── _sense_review_polysemous.tsv        # QA queue: ≥3-sense rows
│   ├── _sense_review_idioms.tsv            # QA queue: all idiom rows
│   ├── _sense_review_function_words.tsv    # QA queue: function words & pronouns
│   └── _sense_review_sensitive.tsv         # QA queue: sensitive-term review
├── build/
│   ├── 01a_parse.py                        # parse + ledger init + idiom detect
│   ├── 01b_orthographic_normalize.py       # spelling normalization (accents, ct/pt)
│   ├── 01c_lexical_replace.py              # lexical EP→BP + collision merge
│   ├── 015_bp_status.py                    # cheap LLM classification (Tool Use)
│   ├── 018_dedupe.py                       # post-normalization duplicate detection
│   ├── 02_split_senses.py                  # sense-split, with forced M/F splits + function-word policy
│   ├── 03_enrich.py                        # gender + PoS via API with canonicalized lookup
│   ├── 04_examples.py                      # examples + token-list validation + semantic validator
│   ├── 05_ipa.py                           # eSpeak-NG baseline + LLM correction
│   ├── 06_audio_pilot.py                   # ElevenLabs + R2, first 500 only
│   ├── 07_audio_full.py                    # ElevenLabs + R2, remaining
│   ├── verify_all.py                       # cross-stage validator (incl. verify_ledger)
│   ├── prompts/
│   │   ├── sense_split.md
│   │   ├── bp_status.md
│   │   ├── example_generate.md
│   │   ├── example_validate.md             # second-model semantic check
│   │   ├── ipa_correct.md
│   │   ├── cognate_classify.md
│   │   ├── idiom_expand.md
│   │   └── sensitive_classify.md
│   ├── policies/
│   │   ├── function_word_strategy.md       # how to handle `o`, `de`, `que`, `se`, …
│   │   ├── pronoun_policy.md               # você / tu / vós / vosso / lhe / se
│   │   ├── sensitive_terms_policy.md       # neutral-example rules
│   │   └── bp_replacement_policy.md        # when to merge vs replace vs split
│   └── lib/
│       ├── tsv.py                          # read/write with quoting (handles line 262's quotes)
│       ├── parse.py                        # split-on-first-` = `, parenthetical extractor (with false-positive guards)
│       ├── ledger.py                       # ledger read/write/append; verify_ledger
│       ├── overrides.py                    # manual override loader (applied before LLM calls)
│       ├── bp_rules.py                     # spelling map + hyphen rules + lexical replacements
│       ├── lookup.py                       # Wiktionary BR / Priberam BR with canonicalization
│       ├── llm.py                          # Anthropic + OpenAI Tool-Use wrappers + provenance hashing
│       ├── elevenlabs_client.py            # batch generation, retry, rate-limit handling
│       ├── r2_client.py                    # R2 upload via custom-domain public bucket
│       ├── audio_manifest.py               # manifest read/write, version bumping, hash dedupe
│       └── validate.py                     # token_in_sentence, ledger math, schema invariants
├── config/
│   └── models.yaml                         # model IDs (bp_status, sense_split, example, ipa, validator)
├── audit/                                  # JSONL provenance per stage (gitignored beyond samples)
│   ├── 015_bp_status.jsonl
│   ├── 02_sense_split.jsonl
│   ├── 04_examples.jsonl
│   └── 05_ipa.jsonl
├── tests/
│   └── test_invariants.py
├── .env.example                            # ANTHROPIC_API_KEY, OPENAI_API_KEY, ELEVENLABS_API_KEY, R2_*
├── pyproject.toml
└── README.md
```

## Pipeline stages

Each stage reads the previous stage's TSV, the manual override file (if any), and the ledger; writes the next stage's TSV; appends to ledger; appends to provenance JSONL. Each is idempotent. Each ends with assertions that fail loudly. **Manual overrides always win** over regenerated model output.

### Stage 1a — Parse & ledger init ([build/01a_parse.py](build/01a_parse.py))

**In**: `data/source.txt` → **Out**: `_source_ledger.tsv` (initial), raw parsed rows in memory

1. Read `source.txt` line by line. Split on **first** ` = ` only. The source has **3 embedded-`=` cases on the RHS** (not 2 as in v1):
   - [line 31 `eu = I (OBJ = me)`](data/source.txt:31)
   - [line 314 `medida = measure (a m. que = to the degree / extent that)`](data/source.txt:314)
   - [line 3372 `vós = you (PL) (OBJ = vos)`](data/source.txt:3372)
2. Initialize the ledger: one row per `source.txt` line with `action = keep` and `source_pt`, `source_en_all`, `source_raw`, `source_line_number`, `rank` populated. All later stages mutate this ledger by changing `action` and adding columns.
3. Extract parentheticals from RHS into a structured `annotation` JSON field. **False-positive guards** (must NOT be treated as gender/idiom markers):
   - `M. Gerais` → `Minas Gerais` proper-noun reference (e.g., [line for `mina`](data/source.txt) — `mina = mine (M. Gerais: state in B)`). Specifically: `mina` is **not** added to forced M/F split list.
   - `e.g.`, `i.e.` → English meta-markers.
   - `OBJ = me` / `OBJ = vos` → object-pronoun cross-references (lines 31, 3372).
4. **Idiom-candidate detection** — pattern-based, not hard-coded list. Trigger when a parenthetical contains an abbreviated form of the headword (single-letter-with-period that matches the headword's initial). Confirmed corpus list, expanded from v1:
   | Source headword | Source pattern | Expansion |
   |---|---|---|
   | `medida` | `(a m. que)` | `à medida que` |
   | `diante` | `(em d.)` | `em diante` |
   | `cento` | `(por c.)` | `por cento` |
   | `seguida` | `(em s.)` | `em seguida` |
   | `vigor` | `(em v.)` | `em vigor` |
   | `redor` | `(em / ao r.)` | `em redor` / `ao redor` |
   | `repente` | `(de r.)` | `de repente` |
   | `invés` | `(ao i.)` | `ao invés` |
   | `contrapartida` | `(em c.)` | `em contrapartida` |
   | `obstante` | `(não o.)` | `não obstante` |
   | `mercê` | `(a m. de)` | `à mercê de` |
   | `tona` | `(à t.)` | `à tona` |
   Detector confirms each candidate against false-positive guards above. Few-shot Tool-Use prompt at [build/prompts/idiom_expand.md](build/prompts/idiom_expand.md). Each expansion creates an additional row with `action = expand_idiom_added` and shared `rank` plus a unique `expansion_index ≥ 1`. Original row keeps `expansion_index = 0`. Logged to `_idioms_expanded.tsv`.

**Validation (Stage 1a)**:

- Ledger has exactly `len(source.txt)` rows.
- `pt` does not contain `=` on any kept row.
- Ledger row count by `source_line_number` is 1.

### Stage 1b — Orthographic normalization ([build/01b_orthographic_normalize.py](build/01b_orthographic_normalize.py))

**In**: ledger + parsed rows → **Out**: `01-normalized.tsv` (orthography-normalized but not yet lexically replaced)

Two **separate** kinds of operation, both keyed off `_ep_spelling_map.tsv` and `_hyphen_rules.tsv` respectively. Lexical replacement (`comboio → trem`) does NOT happen here.

1. **Apply `_ep_spelling_map.tsv`** — orthographic only. Schema:
   ```tsv
   source_form	bp_form	rule_type	confidence	note
   facto	fato	ct_drop	high	
   óptimo	ótimo	pt_drop	high	
   contacto	contato	ct_drop	high	
   excepção	exceção	pç_to_ç	high	
   económico	econômico	accent	high	
   género	gênero	accent	high	
   patrimônio	patrimônio	accent	high	
   crónica	crônica	accent	high	
   académico	acadêmico	accent	high	
   colónia	colônia	accent	high	
   autónomo	autônomo	accent	high	
   polémico	polêmico	accent	high	
   bebé	bebê	accent	high	
   anónimo	anônimo	accent	high	
   húmido	úmido	accent_and_h	high	
   gémeo	gêmeo	accent	high	
   irónico	irônico	accent	high	
   cómico	cômico	accent	high	
   oxigénio	oxigênio	accent	high	
   humidade	umidade	accent_and_h	high	
   fenómeno	fenômeno	accent	high	
   quilómetro	quilômetro	accent	high	
   convénio	convênio	accent	high	
   cerimónia	cerimônia	accent	high	
   ```
   Plus auto-detect: any source headword matching `(.+)[óé](\w{2,})` where swapping to `[ôê]` produces a more frequent BP form is routed to manual review. `rule_type` ∈ `{accent, ct_drop, pt_drop, pç_to_ç, accent_and_h, other}`.
2. **Apply `_hyphen_rules.tsv`** (post-1990 Acordo Ortográfico): `mão-de-obra→mão de obra`, `dia-a-dia→dia a dia`. **Preserve hyphens for**: weekdays (`segunda-feira`…), `vice-*`, `ex-*`, `bem-*`, `meio-*`, `meia-*`, `*-presidente`, `secretário-geral`, `matéria-prima`, `porta-voz`, geographic adjectives (`norte-americano`, `sul-americano`, `sul-africano`).
3. Update ledger: append `spelling` and/or `hyphen` to `normalization_action`; preserve `source_pt` unchanged.
4. **Programmatic flagging** — every entry containing `(BP)`, `(EP)`, or `mainly EP` in source RHS is logged to `_flags.tsv` with structured `flag_type`. Schema:
   ```tsv
   rank	source_pt	pt	source_line	flag_type	flag_reason	manual_review_required	manual_status	approved_by	approved_at	notes
   ```
   `flag_type` ∈ `{bp_marker, ep_marker, false_friend, nsfw_or_sensitive, regional_brazil, possible_drop, possible_replacement, pronoun_policy, manual_review_required}`. Examples: [`rapariga`](data/source.txt:2124) (BP NSFW), [`camisola`](data/source.txt:3842) (false friend), [`bala`](data/source.txt:2940) (BP), [`trem`](data/source.txt:2604), [`sítio`](data/source.txt:913), [`policial`](data/source.txt:1125), [`vosso`](data/source.txt) (mainly EP), [`troço`](data/source.txt) (EP).

**Output schema** (`01-normalized.tsv`): `source_line_number, rank, expansion_index, source_pt, pt, pt_type, gender (blank), pos (blank), en_all, annotation, normalization_action, source_line`.

`pt_type` ∈ `{single_word, hyphenated_compound, space_compound, idiom, abbreviation_expansion}` — assigned in Stage 1a/1b based on whitespace and hyphen content.

**Validation**:

- No row has `=` in `pt`.
- Every row's `source_line_number` exists exactly once in the ledger.
- `(rank, expansion_index)` is unique within `01-normalized.tsv`.
- Ranks are **non-decreasing** (gaps allowed because Stage 1c may later merge/replace; idiom expansions share rank with `expansion_index ≥ 1`).

### Stage 1c — Lexical EP→BP replacement & collision merge ([build/01c_lexical_replace.py](build/01c_lexical_replace.py))

**In**: `01-normalized.tsv` + `_lexical_bp_replacements.tsv` → **Out**: `012-lexical_replaced.tsv`, updated ledger

This is **separate** from orthographic normalization. Lexical replacement substitutes a different word, not a different spelling.

1. **Apply `_lexical_bp_replacements.tsv`**:
   ```tsv
   source_pt	bp_replacement	action	confidence	note
   comboio	trem	replace_or_merge	high	
   equipa	equipe	replace_or_merge	high	
   desporto	esporte	replace_or_merge	high	
   paragem	parada	replace_or_merge	high	
   utilizador	usuário	replace_or_merge	high	
   registar	registrar	replace_or_merge	high	
   registo	registro	replace_or_merge	high	
   planeamento	planejamento	replace_or_merge	high	
   controlo	controle	replace_or_merge	high	
   autocarro	ônibus	replace_or_merge	high	
   telemóvel	celular	replace_or_merge	high	
   frigorífico	geladeira	replace_or_merge	high	
   chouriço	linguiça	replace_or_review	medium	may differ regionally
   pequeno-almoço	café da manhã	replace_or_merge	high	
   ```
   For each match: set `pt = bp_replacement` and add `lexical` to `normalization_action`.
2. **Collision detection** — after replacement, if the new `pt` collides with an existing kept row (e.g., source has both `equipa` and `equipe`):
   - **Equivalent gloss** (high English-overlap): action becomes `merge_into_existing_bp_row`. Provenance from both source lines is preserved in `merged_from_source_lines` and `source_variants` columns. Glosses are union'd into `en_all`. Only one row survives in `012-lexical_replaced.tsv`.
   - **Inequivalent gloss**: action becomes `manual_review`; routed to `_ep_drop_or_replace_review.tsv`. Both rows persist until human decision.
   - **Ambiguous**: action becomes `manual_review`.
3. **Action-state assignment** for all lines (replaces v1's binary auto-drop):

   | Case | `action` |
   |---|---|
   | EP-only with clear BP equivalent already present in source | `merge_into_existing_bp_row` |
   | EP-only with clear BP equivalent absent | `replace_with_bp_equivalent` (and we synthesize the BP form if needed) |
   | Valid BP but non-primary | `keep_tag_rare` (tag `#bp-rare` or `#not-primary-bp`) |
   | False friend | `keep` + flag for manual review before Stage 4 |
   | Offensive / sensitive | `keep` + flag for manual review before Stage 4 |
   | Truly EP-only with no useful BP equivalent | `drop_ep_only` (must have `drop_reason`) |
   | Unclear | `manual_review` |

4. Ledger updates: every modified row has `bp_replacement`, `merge_target_*`, or `drop_reason` populated as appropriate. Approval files: `_ep_drop_or_replace_review.tsv` for the manual queue.

**Validation**:

- For every ledger row with `action = drop_ep_only`, `drop_reason` is non-empty.
- For every `action = replace_with_bp_equivalent`, `bp_replacement` is non-empty.
- For every `action = merge_into_existing_bp_row`, `merge_target_rank` and `merge_target_pt` are non-empty.
- No silent drops: `len(_ep_drops.tsv)` rows all have `drop_reason`.

### Stage 1.5 — BP-status classification ([build/015_bp_status.py](build/015_bp_status.py))

**In**: `012-lexical_replaced.tsv` + `_manual_bp_status.tsv` → **Out**: `015-bp_status.tsv`, ledger updates, `audit/015_bp_status.jsonl`

Cheap Claude Haiku 3.5 pass with **mandatory Tool Use**. For each row not covered by manual override, classify:

- `standard` — everyday BP vocabulary
- `uncommon` — BP-acceptable but rare; tag `#bp-rare`
- `false_friend` — different/offensive meaning in BP than EP; flag for manual review, tag `#false-friend`
- `nsfw` — offensive/sexual in BP (e.g., `puto`, `bicha`, `rapariga` BP-sense); flag, tag `#nsfw`
- `ep_only` — drop (catches lexical EP-isms missed by Stage 1c — words this pass classifies as truly EP)

**Output enforcement (critical)**: do **not** rely on free-text "reply with one of: standard, uncommon, …". Haiku will eventually emit prose like `"I classify this as: standard"` and break the parser. Force structured output via Anthropic **Tool Use** with `tool_choice = {"type": "tool", "name": "classify_bp_status"}`. Schema:

```json
{
  "type": "object",
  "properties": {
    "bp_status": {"type": "string", "enum": ["standard", "uncommon", "false_friend", "nsfw", "ep_only"]},
    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    "reason": {"type": "string"}
  },
  "required": ["bp_status", "confidence"]
}
```

Out-of-enum values are rejected by the API. Same Tool-Use pattern reused for cognate-classify, idiom-expand, sense-split, and sensitive-term classification.

**Provenance**: every call writes to `audit/015_bp_status.jsonl` with `source_line_number, model_id, prompt_hash, response_hash, decision, confidence, generated_at`.

**Cost**: ~$0.20 (Haiku, ~5k entries × ~50 tokens).

**Validation**: every row has a non-empty `bp_status` from the enum. Rows classified `ep_only` get ledger `action = drop_ep_only` with `drop_reason = "Stage 1.5 LLM bp_status=ep_only"`. Manual overrides in `_manual_bp_status.tsv` always win.

### Stage 1.8 — Post-normalization duplicate detection ([build/018_dedupe.py](build/018_dedupe.py))

**In**: `015-bp_status.tsv` → **Out**: `018-deduped.tsv`, `_normalized_duplicates.tsv`

After all normalization (orthographic + lexical + bp_status drops), detect any remaining `pt` collisions. Schema for `_normalized_duplicates.tsv`:

```tsv
pt	ranks	source_pts	source_lines	en_all_values	suggested_action	manual_status	notes
```

Duplicate types: `exact_duplicate`, `spelling_collision`, `lexical_replacement_collision`, `same_headword_different_senses`, `ambiguous_collision`. Truly equivalent duplicates are merged (provenance preserved). Same-headword-different-senses cases pass through as multiple rows with identical `pt` but distinct ranks; Stage 2 will assign distinct `sense_id`s. Ambiguous cases go to manual review.

**Validation**: No two rows in `018-deduped.tsv` share `(rank, expansion_index)`.

### Stage 2 — Sense split ([build/02_split_senses.py](build/02_split_senses.py))

**In**: `018-deduped.tsv` + `_manual_sense_splits.tsv` + `build/policies/function_word_strategy.md` + `build/policies/pronoun_policy.md` → **Out**: `02-senses.tsv`, ledger `output_sense_ids` populated, `audit/02_sense_split.jsonl`

#### Deterministic pre-classification (before LLM)

For each row where `en_all` contains `/`, classify the split into one of:

- `single_sense_synonyms` — e.g., `begin / start`, `house / home`, `to seek / look for` (1 sense)
- `inflection_or_degree` — e.g., `better / best` (1 sense, comparative-superlative)
- `grammar_function_split` — function words; route to manual sense template
- `true_polysemy` — e.g., `point / dot / period` (multiple senses)
- `gendered_polysemy` — forced split (M/F)
- `regional_or_register_difference` — handled by `bp_status` + manual review
- `idiom_or_phrase` — already expanded in Stage 1a
- `manual_review` — low confidence

Pre-classifier uses a small synonym dictionary, comparative/superlative detector, and known-function-word list. Only `true_polysemy` and high-confidence cases go to the LLM splitter. Rest are handled deterministically or routed to manual queues.

Fields added per row: `sense_split_strategy, split_confidence, split_reason, manual_review_required`.

#### LLM splitter (Tool Use)

For rows reaching the LLM, prompt with the rule: *"Two items are distinct senses iff you cannot construct a single example sentence where both translations are natural."* Few-shot examples: `melhor = better / best` (1 sense), `ponto = point / dot / period` (3 senses).

#### Forced splits

- Source RHS contains `(M ... / F ...)` or `(F ... / M ...)`. Confirmed corpus list: [408 `capital`](data/source.txt:408), [591 `polícia`](data/source.txt:591), [650 `rádio`](data/source.txt:650), [921 `corte`](data/source.txt:921), [3007 `cabra`](data/source.txt:3007), [3160 `cura`](data/source.txt:3160), [3333 `grama`](data/source.txt:3333), [4167 `banana`](data/source.txt:4167). Tag `#gendered-meaning`. Note: `mina (M. Gerais)` is **excluded** by Stage 1a's false-positive guard.
- Idiom rows from Stage 1a (already separate, preserved as-is).

#### Function words and pronouns

Function words (`o`, `de`, `em`, `que`, `se`, `a`, `lhe`, `por`, `para`, …) and pronouns (`você`, `tu`, `vós`, `vosso`, …) are exempt from generic slash-splitting. They follow [`build/policies/function_word_strategy.md`](build/policies/function_word_strategy.md) and [`build/policies/pronoun_policy.md`](build/policies/pronoun_policy.md). Special handling:

| Item | Treatment |
|---|---|
| `você` | Keep; central BP |
| `tu` | Keep but tag `#regional` and grammar note |
| `vós` | Likely drop or tag `#archaic` (manual review) |
| `vosso` | Manual review; mainly EP/formal/religious |
| `lhe` | Keep but explain BP usage carefully |
| `se` | Special handling: reflexive, impersonal, passive-like, conditional senses each forced |

Tags: `#function-word`, `#pronoun`, `#grammar`, `#manual-sense`.

#### sense_id assignment

`sense_id = {rank:04d}.{expansion_index}{sense_index}` where `expansion_index` is from Stage 1a (default `0`, `1+` for idiom expansions) and `sense_index` is 1-indexed within `(rank, expansion_index)`. Example: `0001.01` for `o` first sense; `0314.11` for the idiom-expanded `à medida que` first sense. Sense IDs are stable forever.

#### Review

Stage 2 runs the **two-model jury** (§ Review strategy). Disagreements go to `_jury_disagreements.tsv` and are auto-resolved by the tiebreaker model. The auditor pass at Stage 5.5 catches anything that slipped through both. **No upfront human review queues** for Stage 2 — the v2 plan's six queues (`_sense_review_top1000.tsv`, `_sense_review_low_confidence.tsv`, `_sense_review_polysemous.tsv`, `_sense_review_idioms.tsv`, `_sense_review_function_words.tsv`, `_sense_review_sensitive.tsv`) are removed.

The 8 forced-gender-split entries and the ~12 idiom-expansion entries are short enough to **inline as manual seed senses** in `_manual_sense_splits.tsv` once at the start of the project (~5 min), bypassing both LLM splitting and review.

**Validation**:

- `sense_id` uniqueness.
- Every row has `en_primary`.
- Re-joining `en_primary` by `sense_index` reconstructs a superset of `en_all`.
- Ledger `output_sense_ids` populated for every input row that reached this stage.
- ~8,000–10,000 rows (soft diagnostic, not asserted).

### Stage 2.5 — Sensitive-term screen ([build/sensitive_screen.py](build/sensitive_screen.py))

**In**: `02-senses.tsv` → **Out**: `_sensitive_terms.tsv`

Screen for terms that need careful example-generation policy even when not flagged `(BP)`/`(EP)`. Two-pass: deterministic keyword screen (e.g., `gozar`, `rapariga`, `mulato`, `índio`, `raça`, `cigano`, `homossexual`, `aborto`, `suicídio`, `violar`, `droga`, `bala`, `arma`, `faca`, `espingarda`, `revólver`) followed by LLM Tool-Use classification. Schema:

```tsv
sense_id	rank	source_pt	pt	sensitive_category	risk_level	example_policy	manual_status	notes
```

`sensitive_category` ∈ `{sexual, violence, slur_or_identity, race_ethnicity, religion, politics, medical, self_harm, crime, weapons, substance, offensive_possible}`. `risk_level` ∈ `{low, medium, high}`. `example_policy` is a short directive that flows into the Stage 4 prompt (e.g., "use neutral medical context, no graphic detail"). High-risk and medium-risk rows tagged `#sensitive-reviewed`. The auditor at Stage 5.5 verifies that generated examples actually obey their `example_policy`; failures auto-regenerate. **No upfront manual approval gate** — the auditor's `violates_sensitive_policy` boolean is the gate.

This is **not a drop list**. Most words remain. The purpose is to ensure example sentences are neutral, safe, and learner-appropriate.

### Stage 3 — Enrichment ([build/03_enrich.py](build/03_enrich.py))

**In**: `02-senses.tsv` + `_manual_gender.tsv` → **Out**: `03-enriched.tsv`

Deterministic + LLM hybrid:

1. **Canonicalize headword for API lookup** ([build/lib/lookup.py](build/lib/lookup.py)):
   - Strip `-se` from reflexive verbs (preserve flag in `annotation.reflexive`).
   - Try alternate hyphenations if first attempt 404s.
   - Try without diacritics as last resort.
   - For compound nouns, look up the **whole compound** (e.g., `mão de obra`, not `mão` + `obra`).
   - **Aggressive local cache** — Wiktionary and Priberam are not stable APIs. `lookup.py` scrapes HTML; structure changes break the pipeline, and unthrottled re-runs invite IP bans. Cache layer: SQLite at `build/cache/lookup.sqlite` keyed by `(source, canonical_form)` with columns `source, canonical_form, raw_html, fetched_at, http_status, etag, parser_version`. Re-runs hit the cache; only forced refresh (`--refresh`) or rows older than `LOOKUP_CACHE_TTL` (default 90 days) re-fetch. Parsing happens off the cached HTML so a Wiktionary layout change requires only a parser update, not a re-scrape. Politeness: per-host rate limit (default 1 req/sec) + jittered backoff (see § API client conventions).
   - **Graceful parser failure (do not crash the pipeline)** — Wiktionary and Priberam DOM structures change with no warning. The parser MUST handle "field absent" as a valid outcome, not an exception:
     - Parser returns `LookupResult(gender=None, pos=None, confidence='none', parser_status='dom_miss')` when it cannot find the expected DOM nodes.
     - The stage logs the miss (with `source`, `canonical_form`, `parser_version`, snippet of the cached HTML) to `_lookup_misses.tsv` for later review.
     - Lookup-cascade (manual override → Wiktionary → Priberam → LLM Tool-Use) treats `dom_miss` as "this source had no answer," moves to the next source, and ultimately falls through to the LLM jury. The pipeline never crashes on a scraper failure.
     - When `_lookup_misses.tsv` accumulates >5% of rows from a single source, it's signal that the parser needs an update — fix the parser, bump `parser_version` in the cache schema, re-parse from cached HTML (no re-scrape needed). The cost is engineering time, not API quota.
2. **Gender** (nouns): manual override → Wiktionary BR → Priberam BR → Claude Tool-Use classification (`{"gender": "o" | "a" | "o/a" | "—"}`). Fill `gender` and compose `pt_display` (`a casa`, `o caminho`, or bare word for non-nouns / idioms with their natural form).
3. **PoS**: derive from gloss pattern — `"to X"` → `verb`; gender known → `noun`; bare adjective → `adj`. **Leave blank when ambiguous**.
4. **Tags**:
   - Frequency tier: `#top500`/`#top1000`/`#top2000`/`#top3000`/`#top5000`.
   - PoS: `#verb`/`#noun`/`#adj`/etc. when known.
   - Morphology: `#reflexive`, `#gendered-meaning`, `#hyphenated`, `#idiom`.
   - `pt_type`: `#single-word`, `#hyphenated-compound`, `#space-compound`, `#idiom`, `#abbreviation-expansion`.
   - Regional: `#bp-rare` from Stage 1.5.
   - Special: `#interjection`, `#numeral`, `#nsfw`, `#false-friend`, `#cognate-en`, `#sensitive-reviewed`, `#function-word`, `#pronoun`, `#manual-sense`.
5. **Family root**: Claude proposes Portuguese derivational root. Empty when standalone. Schema-reserved.
6. M/F-split senses from Stage 2: gender already assigned, just confirm.

**Validation**: every noun has `gender ∈ {o, a, o/a, —}`; every reflexive verb has flag; tags space-separated and well-formed.

### Stage 4 — Example sentences ([build/04_examples.py](build/04_examples.py))

**In**: `03-enriched.tsv` + `_manual_examples.tsv` + `_sensitive_terms.tsv` (for `example_policy`) → **Out**: `04-examples.tsv`, `_example_fixes.tsv`, `audit/04_examples.jsonl`

Per sense, prompt Claude Sonnet 4.6 (or successor; configured in `models.yaml`) with full sense context + sensitive-term policy if applicable. Generate three fields:

- `example_pt`: ≤15 words; A2 background grammar. Target word used in typical way **for THIS specific sense**. BP vocabulary and spelling.
- `example_en`: faithful English translation.
- `target_word_used`: the **exact surface form** of the target word as it appears in `example_pt` (e.g., `vou` for headword `ir`; `pelo` for `por` if contracted).

Output enforced via Tool Use:

```json
{
  "type": "object",
  "properties": {
    "example_pt": {"type": "string"},
    "example_en": {"type": "string"},
    "target_word_used": {"type": "string"}
  },
  "required": ["example_pt", "example_en", "target_word_used"]
}
```

#### Per-row validation

- **Token-list comparison** (not regex `\b`). Python's `\b` treats hyphens as non-word characters, breaking `primeiro-ministro`, `segunda-feira`, and enclitic-pronoun forms (`dizer-lhe`, `dá-me`). Algorithm in [build/lib/validate.py](build/lib/validate.py) as `token_in_sentence(target, sentence)`:
  ```python
  PUNCT = '.,;:!?¿¡«»"\'()[]{}…—–'
  def token_in_sentence(target, sentence):
      tr = str.maketrans('', '', PUNCT)  # keep hyphens and apostrophes
      tokens = sentence.lower().translate(tr).split()
      return target.lower().translate(tr) in tokens
  ```
  Avoids both the `por`-inside-`porque` substring trap and the hyphen-boundary regex trap.
- Word count of `example_pt` ≤ 15.
- `example_en` non-empty.
- Rows in `_flags.tsv` (NSFW / false-friend / (BP)-tagged) and rows in `_sensitive_terms.tsv` with `risk_level ∈ {medium, high}` are processed only after `manual_status = approved`.

#### Semantic validator (second pass)

After generation, a **separate model** (e.g., GPT-4o-mini, configured in `models.yaml`) validates each row against [`build/prompts/example_validate.md`](build/prompts/example_validate.md). Tool-Use schema:

```json
{
  "type": "object",
  "properties": {
    "uses_intended_sense": {"type": "boolean"},
    "is_bp": {"type": "boolean"},
    "is_natural": {"type": "boolean"},
    "translation_matches": {"type": "boolean"},
    "fails_neutral_example_policy": {"type": "boolean"},
    "validation_status": {"type": "string", "enum": ["pass", "fail", "borderline"]},
    "validation_reason": {"type": "string"}
  },
  "required": ["validation_status"]
}
```

Failures (`fail`) routed to `_example_fixes.tsv` for re-generation. `borderline` routed to manual review queue. Do **not** rely on the same model validating itself.

#### Review

Stage 4 examples flow into Stage 5.5's adversarial + verdict auditor pair (§ Review strategy). Top-1000 senses get N=2 generation with auditor selection. Borderline rows route to `_jury_disagreements.tsv`. Failures auto-regenerate up to twice with the defect list in the prompt. **No "first 500 senses, user-approved" gate** — that v2 step is dropped.

**Cost**: ~$30–60 (Sonnet generator full corpus) + ~$15–25 (top-1000 N=2 second generation) + ~$15–25 (validator pass).

### Stage 5 — IPA ([build/05_ipa.py](build/05_ipa.py))

**In**: `04-examples.tsv` + `_manual_ipa.tsv` → **Out**: `05-ipa.tsv`, `audit/05_ipa.jsonl`

**Strategy: deterministic baseline + LLM correction.** LLM-only IPA is risky (hallucinated stress, invented phonemes). Baseline first.

Fields per row:

- `ipa_word_machine` — eSpeak-NG `--ipa pt-BR` output for the headword (or full idiom for idiom rows).
- `ipa_word_final` — after LLM correction pass for known BP/paulistano issues (closed-vowel realizations, /ʁ/ vs /r/, palatalization of /t/, /d/ before /i/, nasal handling).
- `ipa_example_machine` — eSpeak-NG per-token IPA for `example_pt`, **isolated form**, space-separated.
- `ipa_example_final` — after LLM correction.
- `ipa_source` — `machine`, `corrected`, or `manual_override`.
- `ipa_confidence` — `high` / `medium` / `low`.

Final TSV exposes `ipa_word` and `ipa_example` (= `*_final`).

**Documented limitation (not a defect)**: per-word isolated-form IPA does not reflect connected-speech sandhi — vowel reductions, /s/-linking across word boundaries, contractions like `para o → pro`. The audio in Stage 6 will exhibit natural sandhi. Both serve different pedagogical purposes. README spells this out.

**Review**: the auditor at Stage 5.5 runs the `ipa_plausible` check across all rows; outliers regenerate. The eSpeak-NG baseline gives a strong machine-readable lower bound, so LLM "correction" is heavily constrained. Manual override remains available per-row via `_manual_ipa.tsv` for any user-noticed issue at study time.

**Validation**: `ipa_word_final` non-empty for every row; `ipa_example_final` token count == `example_pt` token count.

**Cost**: ~$15–25 (LLM correction only; eSpeak-NG is free).

### Stage 5.5 — Adversarial + verdict auditor ([build/055_audit.py](build/055_audit.py))

**In**: `05-ipa.tsv` → **Out**: `_auditor_flags.tsv`, regeneration directives, `_jury_disagreements.tsv` borderline queue

Runs the **two-auditor pipeline** described in § Review strategy: adversarial auditor produces a defect list; verdict auditor (different model) reviews the row + defect list and decides `pass` / `regenerate` / `human_review`. Tool Use enforces both schemas. Failures route to auto-regeneration (≤2 attempts, with the defect list in the regen prompt as anti-examples). Borderlines route to the small human disagreement queue.

The two auditor models are different from each other AND different from the generator AND different from any of the three jurors. In practice this means rotating across families: e.g., generator = Anthropic, jurors = {Anthropic-small, OpenAI, Google}, adversarial auditor = OpenAI top-tier, verdict auditor = Google top-tier. The registry handles the assignment.

**Cost**: ~$30–60 (two auditor calls per row × ~9k rows).

**Validation**: every row in `05-ipa.tsv` has both an adversarial defect list and a verdict. Rows with `verdict = regenerate` after 2 failed regen attempts are surfaced loudly to `_auditor_flags.tsv` — should be rare (<1%).

### TTS provider (default: Google Cloud TTS)

The audio line is the largest in v2/early-v3. Switching from ElevenLabs to **Google Cloud TTS Neural2 BR voices** is the single biggest cost win. Naturalness gap is small at A1 listening level; both produce intelligible Brazilian Portuguese with correct stress and prosody. Pricing comparison (rough, as of 2026):

| Provider | Quality (A1 use) | Pricing | Total at ~970k chars × 2 voices |
|---|---|---|---|
| ElevenLabs (Multilingual v2) | premium | ~$0.15–0.22/1k chars | ~$260–430 |
| **Google Cloud TTS Neural2 (default)** | excellent | ~$0.016/1k chars | **~$30–60** |
| Azure Neural TTS | excellent | ~$0.016/1k chars | ~$30–60 |
| OpenAI TTS (`tts-1` / `tts-1-hd`) | very good | ~$0.015–0.030/1k chars | ~$30–60 |
| Open-source local (Kokoro / XTTS / MeloTTS) | adequate | $0 (local compute) | $0 |

`config/models.yaml` carries the active TTS provider as a role-key. Switching providers later is a config change + regenerate-with-version-bump (Stage 6 idempotency handles this).

Voice IDs are pinned per gender. For Google Cloud TTS BR voices, defaults: male = `pt-BR-Neural2-B`, female = `pt-BR-Neural2-A`. User picks final voices at step 13 of runbook. ASR roundtrip and filename versioning (`-v{N}.mp3`) are unchanged across providers.

### Stage 6 — Audio pilot ([build/06_audio_pilot.py](build/06_audio_pilot.py))

**In**: `05-ipa.tsv` (first 500 rows) + `_manual_audio.tsv` → **Out**: `_pilot_500.tsv`, `_audio_manifest.tsv` (initial), audio assets on R2

#### Audio manifest is source of truth

`_audio_manifest.tsv` schema:

```tsv
sense_id	clip_type	voice_gender	voice_id	text_input	text_hash	object_key	url	version	md5	generated_at	elevenlabs_model	status	notes
```

`clip_type` ∈ `{word, example}`. Each sense has 4 manifest rows. Final TSV's URL columns are **derived** from the manifest, not hand-maintained. Regeneration: bump `version`, re-upload, update manifest, re-derive TSV.

#### Generation

Generate 4 mp3 clips per sense via the configured TTS provider (default: Google Cloud TTS Neural2):

- `audio_word_m`: male voice, headword
- `audio_word_f`: female voice, headword
- `audio_example_m`: male voice, example sentence
- `audio_example_f`: female voice, example sentence

**File naming**: `{sense_id}-{word|ex}-{m|f}-v{version}.mp3` (e.g., `0001.03-word-m-v1.mp3`). Gender AND version are baked into the filename. **The version goes in the filename, not in a `?v=` query string** — this is the critical Anki-compatibility fix. Reasons:

- **Anki strips URL query parameters** when downloading media into `collection.media/`. A URL `...0001.03-word-m.mp3?v=2` lands locally as `0001.03-word-m.mp3` (no version), so Anki sees an existing file with the same name and skips the download. Versioned filenames like `0001.03-word-m-v2.mp3` are net-new filenames; Anki always fetches them.
- **Anki's media sync compares filenames, not file hashes.** If a regenerated clip keeps the same filename, mobile devices won't re-download it during the next AnkiWeb sync. Versioning the filename guarantees clean propagation across desktop + iPhone.
- Cloudflare CDN caching becomes irrelevant (different filename = different object key = cache miss = new fetch).
- The `?v=N` query-string strategy from prior iterations is **abandoned** for audio URLs that are referenced by Anki notes. It would still work for browser-only previews, but Anki is the consumer that matters.

When a clip is regenerated: bump the manifest `version`, write to a new R2 object key with the new filename, update the manifest URL, and the final TSV's audio columns now point at the new filename. The old object can be deleted from R2 after a grace period (or kept; storage is ~$0.02/mo for the lot). Voice mix-up after a mid-batch crash remains structurally impossible because gender stays baked in.

#### Workflow

1. Voices: user-provided male voice ID; female voice ID placeholder until user supplies.
2. Compute md5 per file; store in manifest and `_md5` columns.
3. Upload objects to R2.
4. **Public access**: expose audio through an **R2 public bucket custom domain**. Do **not** rely on per-object public-read ACL semantics — that wording was incorrect in v1. Public base URL stored in `.env`.
5. Stable URL pattern: `https://<R2-public-domain>/audio/{sense_id}-{word|ex}-{m|f}-v{version}.mp3`. Version is **in the filename**, not the query string (see § File naming above for the Anki-compatibility rationale).
6. Local `build/audio_cache/` retained until final `.apkg` bundling.

#### TTS error handling (explicit policy)

TTS providers return transient `500`/`502`/`503`/`504` and `429` under load. The script must not turn a transient 5xx into a permanent manifest failure. [build/lib/tts_client.py](build/lib/tts_client.py) policy (provider-agnostic; Google/Azure/ElevenLabs share the same retry envelope):

- **Retryable** (HTTP `429`, `500`, `502`, `503`, `504`, connection errors, read timeouts): exponential backoff with full jitter, base `1s`, cap `60s`, up to **6 attempts**. Honor `Retry-After` header when present.
- **Non-retryable** (HTTP `400`, `401`, `403`, `422` invalid voice / unsupported text): record `status = failed_permanent` in manifest with the response body in `notes`; do not retry on script restart.
- **Final failure** (retries exhausted): record `status = failed_transient` so a later retry-only pass picks the row up. Resume mode skips `status = uploaded` and re-attempts `status = failed_transient`.
- Manifest `status` enum: `pending` / `uploading` / `uploaded` / `failed_transient` / `failed_permanent`.

#### Cloudflare cache-busting (now via filename, not query string)

Filename versioning makes cache-busting trivial: a new version is a new filename, which is a new object key, which is a fresh CDN miss with a fresh fetch. No `?v=` games, no manual purge needed, no Anki query-strip trap. Old object keys can be left in R2 (effectively immutable) so existing cards keep working until their notes are updated.

Belt-and-suspenders: upload with `Cache-Control: public, max-age=31536000, immutable` so the CDN treats every versioned filename as cacheable forever. Safe because each filename's content never changes.

The `version` integer for each clip is owned by the manifest. Never silently regenerate without bumping it AND emitting a new file.

#### Cost estimate refinement

Pre-flight cost report runs before audio: count actual characters from `04-examples.tsv` (headwords + examples), multiply by 2 voices, add **20–40% buffer** for retries, regenerations, voice swaps, version bumps, failed requests. User confirms before full run.

#### Review (ASR roundtrip + minimal human gate)

Every clip is validated automatically via Whisper-based ASR roundtrip (§ Review strategy, Tier 3). The pipeline regenerates any clip below the Levenshtein threshold; only clips that fail twice land in `_audio_human_review.tsv`.

The single human gate before Stage 7 is **voice quality**, not pronunciation accuracy: the user listens to ~10 random pilot clips that **already passed ASR roundtrip** to confirm the chosen male and female voices sound right. ~15 minutes. If voices are wrong, swap voice IDs and regenerate (config change + ~30 min compute). Pronunciation accuracy is ASR's job, not the human's.

### Stage 7 — Audio full ([build/07_audio_full.py](build/07_audio_full.py))

**In**: `05-ipa.tsv` (rows 501+) + manifest → **Out**: `06-final.tsv`, manifest fully populated

Same pipeline for the remaining ~7,500–9,500 senses after pilot approval.

**Idempotency**: resume = skip senses whose 4 expected manifest rows already exist with `status = uploaded` and HEAD-request 200 OK on URL.

**Validation**: every row has 4 audio URLs (derived from manifest); HEAD requests 200 OK on a random 5% sample; md5 matches between local cache, manifest, and R2-downloaded.

## Final TSV schema (`data/06-final.tsv`)

| # | Column | Type | Notes |
|---|---|---|---|
| 1 | `sense_id` | `RRRR.EES` | Stable forever (rank.expansion_index||sense_index) |
| 2 | `rank` | int | From source.txt |
| 3 | `source_pt` | string | Original headword as in source.txt (audit) |
| 4 | `pt` | string | BP-normalized headword |
| 5 | `pt_type` | enum | `single_word`/`hyphenated_compound`/`space_compound`/`idiom`/`abbreviation_expansion` |
| 6 | `gender` | `o`/`a`/`o/a`/`—` | Non-nouns get `—` |
| 7 | `pt_display` | string | With article for nouns; idiom form for idiom rows |
| 8 | `pos` | string or blank | Only when certain |
| 9 | `sense_index` | int | 1-indexed within (rank, expansion_index) |
| 10 | `en_primary` | string | Single English gloss for THIS sense |
| 11 | `en_all` | string | Full original RHS; audit |
| 12 | `annotation` | JSON string | `{dialect, number, gender_note, reflexive, idiom}` |
| 13 | `bp_status` | string | `standard`/`uncommon`/`false_friend`/`nsfw` |
| 14 | `normalization_action` | string | `none`/`spelling`/`lexical`/`hyphen`/`idiom_expansion` (composable) |
| 15 | `ipa_word` | string | `ipa_word_final` from Stage 5 |
| 16 | `example_pt` | string | ≤15 words, A2 background |
| 17 | `example_en` | string | English translation |
| 18 | `target_word_used` | string | Exact surface form of target word in `example_pt` |
| 19 | `ipa_example` | string | `ipa_example_final`; per-word IPA, space-separated, isolated form |
| 20 | `audio_word_m` | URL | R2 link, filename ends `-v{N}.mp3`, male voice (derived from manifest) |
| 21 | `audio_word_f` | URL | R2 link, filename ends `-v{N}.mp3`, female voice |
| 22 | `audio_example_m` | URL | R2 link, filename ends `-v{N}.mp3`, male voice |
| 23 | `audio_example_f` | URL | R2 link, filename ends `-v{N}.mp3`, female voice |
| 24 | `audio_word_m_md5` | hex | Corruption detection |
| 25 | `audio_word_f_md5` | hex | |
| 26 | `audio_example_m_md5` | hex | |
| 27 | `audio_example_f_md5` | hex | |
| 28 | `family_root` | string | Empty if standalone |
| 29 | `tags` | string | Space-separated |
| 30 | `source_line` | string | Raw original; never mutated |
| 31 | `source_line_number` | int | Ledger join key |
| 32 | `notes` | string | Escape hatch |

Audit-only columns kept in stage TSVs and ledger but **not** in final TSV: `expansion_index`, `bp_replacement`, `merge_target_*`, `decision_model`, `decision_prompt_hash`, `decision_confidence`, `manual_override`, `ipa_word_machine`, `ipa_example_machine`, `ipa_source`, `ipa_confidence`.

TSV chosen over CSV; Python `csv` with `dialect='excel-tab'` and `quoting=QUOTE_MINIMAL`.

### Tag taxonomy

- **PoS**: `#noun #verb #adj #adv #prep #conj #pron #art #num #interj`
- **Frequency tier**: `#top500 #top1000 #top2000 #top3000 #top5000`
- **Morphology / pt_type**: `#single-word #hyphenated-compound #space-compound #idiom #abbreviation-expansion #reflexive #gendered-meaning #compound`
- **Regional**: `#bp-rare #regional`
- **Special**: `#nsfw #false-friend #cognate-en #numeral #interjection #sensitive-reviewed #function-word #pronoun #grammar #manual-sense #archaic`
- **Semantic** (optional, deferred): `#travel #legal #family #body #time #politics #religion #food #nature #numbers #colors #emotions`

## Manual override system

Every LLM stage has a corresponding `_manual_*.tsv`. Loader at [build/lib/overrides.py](build/lib/overrides.py) applies overrides **before** the LLM is called for a row. Overrides:

- Always win over regenerated model output.
- Carry `override_reason` and `override_set_at`.
- Validated to refer to existing rows by `sense_id` (or `source_line_number` for Stage 1.x).
- Documented in `README.md` so the user can edit them in Numbers/Sheets.

Files: `_manual_bp_status.tsv`, `_manual_sense_splits.tsv`, `_manual_gender.tsv`, `_manual_examples.tsv`, `_manual_ipa.tsv`, `_manual_audio.tsv`.

## LLM provenance

For every model-derived decision, audit JSONL files in `audit/` record:

```json
{"source_line_number": 42, "sense_id": "0042.01", "stage": "015_bp_status",
 "model": "claude-haiku-3-5-20250101", "prompt_hash": "sha256:...",
 "response_hash": "sha256:...", "decision": {...}, "confidence": "high",
 "manual_override": false, "generated_at": "2026-04-26T10:23:11Z"}
```

Applies to Stages 1.5, 2 (sense split + sensitive screen), 3 (gender fallback, cognate, family root), 4 (generation + validation), 5 (IPA correction). Re-runs are reproducible because model IDs and prompt hashes are recorded.

## API client conventions (rate limits, concurrency, retries)

Sequential calls across ~10,000 senses take days; naive `asyncio.gather` immediately trips Anthropic, OpenAI, and ElevenLabs rate limits. All three API wrappers ([build/lib/llm.py](build/lib/llm.py) for Anthropic+OpenAI, [build/lib/elevenlabs_client.py](build/lib/elevenlabs_client.py) for audio, [build/lib/lookup.py](build/lib/lookup.py) for HTML scrapers) follow one shared policy.

### Concurrency model

- **Bounded concurrency** via `asyncio.Semaphore`, not unbounded `gather`. Per-provider concurrency caps configured in `config/models.yaml`:
  ```yaml
  concurrency:
    anthropic: 8        # parallel inflight requests
    openai: 6
    elevenlabs: 4       # ElevenLabs is more sensitive
    wiktionary: 1       # politeness on third-party HTML
    priberam: 1
  ```
- **Token-bucket throttling** in addition to the semaphore, sized below the documented TPM/RPM ceilings (e.g., Anthropic Tier 2: ~200k input TPM → throttle to ~150k TPM to leave headroom). Reads `Anthropic-RateLimit-*` response headers and slows down adaptively when remaining quota drops below 20%.

### Retry policy (shared)

- **Retryable**: HTTP `429`, `500`, `502`, `503`, `504`, `408`, connection errors, read/write timeouts.
- **Backoff**: exponential with **full jitter** (`sleep = random(0, min(cap, base * 2**attempt))`); base `1s`, cap `60s`. Up to **6 attempts** before surfacing the error.
- **Honor `Retry-After`** when the server provides it (overrides computed backoff for that one wait).
- **Non-retryable**: HTTP `400`/`401`/`403`/`404`/`422`. Surface immediately with full response body in the audit JSONL or manifest `notes`.
- **Idempotency keys** on writes (where supported): include `sense_id + clip_type + voice_gender + version` for ElevenLabs uploads so a retry after a network drop is harmless.

### Implementation surface

- `build/lib/llm.py::call_with_retry(client, payload, *, provider)` is the only public entry point for Anthropic and OpenAI. Stage scripts never call SDKs directly.
- `build/lib/elevenlabs_client.py::generate_with_retry(text, voice_id, sense_id, clip_type)` likewise. Internal logic checks the manifest first to skip already-uploaded clips.
- `build/lib/lookup.py::fetch_with_retry(source, canonical_form)` likewise, plus the SQLite cache layer (§ Stage 3).

### Cost/time impact

With concurrency at the caps above, full-corpus runs:

- Stage 1.5 + 2 + 4 + 5 + auditors via **Batch API**: ~12–24 hours wall clock (the batches themselves complete asynchronously, with tighter typical turnaround).
- Realtime tiebreaker calls (small fraction of total): ~30 min wall clock.
- Stage 6+7 audio (Google Cloud TTS, ~38k clips at 4 concurrent): ~2–4 hours wall clock (faster than ElevenLabs).
- Stage 3 dictionary lookups (~8k unique heads at 1 req/s per source): ~2–3 hours; cached on subsequent runs.

### Batch API specifics

- **Anthropic Message Batches**: submit ≤10k requests per batch, results within 24h, 50% discount. Used for jury, validator, auditor, and example generation (any stage where rows are independent).
- **OpenAI Batch API**: similar; same 50% discount; same 24h SLA. Used for OpenAI juror and ASR roundtrip (Whisper accepts batch).
- **Google**: Gemini's Batch Predictions equivalent — 50% discount on most tiers.
- Batch results are pulled by [build/lib/llm.py](build/lib/llm.py) into the same audit JSONLs as synchronous calls, with `batch_id` field added for provenance. Stage scripts treat batch and sync identically downstream.

## Repo hygiene: TSV + Git interaction

Version-controlling 10,000-row TSVs that mutate at every stage produces enormous, noisy diffs and frequent merge conflicts when scripts and data are iterated together. Mitigations:

- **`.gitattributes`**:
  ```
  data/*.tsv          merge=union  diff=tsv  -text
  data/_audio_manifest.tsv  merge=union  diff=tsv  -text
  data/06-final.tsv   filter=lfs   diff=lfs   merge=lfs  -text
  audit/*.jsonl       merge=union  -text
  ```
  `merge=union` makes Git keep both sides of a TSV merge conflict (lines, not characters), which usually produces a recoverable result for append-only ledger and manifest files. The final TSV is large and changes infrequently — Git LFS is appropriate for it.
- **`.gitignore`**:
  ```
  build/audio_cache/
  build/cache/
  audit/*.jsonl
  !audit/.gitkeep
  ```
  Audio cache and lookup cache are reproducible from the manifest and SQLite. Provenance JSONLs are kept locally for debugging but not committed (a `.gitkeep` preserves the directory). If long-term provenance is needed, periodically snapshot a sampled subset to `audit/snapshots/`.
- **Stage outputs**: `01-normalized.tsv` through `05-ipa.tsv` are intermediate. They CAN be committed for reproducibility audits but are regenerated from `_source_ledger.tsv` + `data/source.txt` + manual override files. The committed authoritative artifacts are: `data/source.txt` (immutable), `_source_ledger.tsv`, all `_manual_*.tsv`, `_audio_manifest.tsv`, and `06-final.tsv`. Everything else is a build artifact.
- **Pre-commit hook** (optional): refuse commits where `06-final.tsv` and `_audio_manifest.tsv` row counts disagree on derived URL columns.

## Verification

**Per-stage**: each `build/NN_*.py` ends with assertions (uniqueness, no empty required fields, ledger consistency). Cross-stage validator at [build/verify_all.py](build/verify_all.py).

### Rank invariants (revised)

Idiom expansion (Stage 1a) and lexical merging (Stage 1c) make per-row rank uniqueness incorrect. Replace v1's "strictly increasing, unique" with:

- Rows are ordered by `(rank, expansion_index)`.
- Rank is **non-decreasing** (gaps allowed).
- Within each rank, `expansion_index` is **unique**.
- `(rank, expansion_index)` is unique across `01-normalized.tsv`, `012-lexical_replaced.tsv`, `015-bp_status.tsv`, `018-deduped.tsv`.
- From Stage 2 onward, **`sense_id` is the canonical primary key**; rank/expansion_index are diagnostic.

### Row-count invariants (revised)

**Replace fixed expectations** (`~4,950–4,970 rows`, `~10 idiom rows added`) **with ledger math** (`build/lib/validate.py::verify_ledger_math`):

```
source_line_count
  = kept_source_lines
  + dropped_source_lines
  + merged_source_lines
  + replaced_source_lines
  + manual_review_pending_source_lines

normalized_row_count
  = kept_or_replaced_rows
  + idiom_expansion_rows
  − merged_duplicate_rows
```

Soft diagnostics (warning, not failure): row-count by `action` reported. No fixed range asserted.

### What validators MUST do

- Assert `sense_id` uniqueness from Stage 2 onward.
- Assert ledger math equality.
- Assert every `drop_ep_only` ledger row has a `drop_reason`.
- Assert every `replace_with_bp_equivalent` row has a `bp_replacement`.
- Assert every `merge_into_existing_bp_row` row has both `merge_target_rank` and `merge_target_pt`.
- Assert `(rank, expansion_index)` uniqueness in Stage 1.x outputs.
- Assert ranks are non-decreasing.
- Assert no `=` in any `pt`.
- Assert audio manifest md5 matches local cache md5 for sampled clips.
- Assert HEAD 200 OK on a random 5% of audio URLs.
- Assert every final `sense_id` traces back to ≥1 ledger row via `output_sense_ids`.

### What validators MUST NOT do

- Do not assert exactly 4,985 rows in any file.
- Do not assert ranks form `range(1, N+1)`.
- Do not assert "no missing ranks".
- Do not assert per-row rank uniqueness (idiom expansions share rank).

### End-to-end runbook (v3 lean)

Most of this runs unattended. Human touchpoints are explicitly tagged ⚑.

0. ⚑ **Seed `_manual_sense_splits.tsv`** with the 8 forced gender splits and ~12 idiom expansions (~5 min).
1. `python build/01a_parse.py` → ledger initialized; 3 embedded-`=` cases parsed; idiom candidates flagged.
2. `python build/01b_orthographic_normalize.py` → `01-normalized.tsv`; `_flags.tsv` populated with structured `flag_type`.
3. `python build/01c_lexical_replace.py` → `012-lexical_replaced.tsv`. Replacement collisions auto-merged where gloss overlap is high; ambiguous ones written to `_ep_drop_or_replace_review.tsv` (small queue, ~20 rows max — handled later in jury-disagreement review).
4. `python build/015_bp_status.py` → Tool-Use jury (Sonnet + GPT-4o); tiebreaker on disagreements; `015-bp_status.tsv`.
5. `python build/018_dedupe.py` → `018-deduped.tsv`; auto-merges trivial duplicates.
6. `python build/02_split_senses.py` → jury + tiebreaker; `02-senses.tsv`. **No `--limit`, no human gate.**
7. `python build/sensitive_screen.py` → jury-classified `_sensitive_terms.tsv`; `example_policy` populated for Stage 4.
8. `python build/03_enrich.py` → cached Wiktionary/Priberam + jury fallback; `03-enriched.tsv`.
9. `python build/04_examples.py` → generator + separate-model semantic validator; failures auto-regenerate; `04-examples.tsv`. **No human gate.**
10. `python build/05_ipa.py` → eSpeak baseline + LLM correction; `05-ipa.tsv`.
11. `python build/055_audit.py` → auditor pass; failures auto-regenerate; borderlines into `_jury_disagreements.tsv`.
12. ⚑ **Pre-audio cost report**: review character count and confirm spend (~5 min).
13. ⚑ **Supply male and female voice IDs** (~10 min); configure R2 public custom domain.
14. `python build/06_audio_pilot.py` → ASR-validated; `_pilot_500.tsv`.
15. ⚑ **Voice quality check**: listen to ~10 random ASR-passed pilot clips (~15 min). If voices are wrong, swap and rerun pilot.
16. `python build/07_audio_full.py` → ASR-validated; `06-final.tsv` produced; `_audio_human_review.tsv` for the rare twice-failed clips.
17. ⚑ **Disagreement queue review** (~30–60 min): walk through `_jury_disagreements.tsv` and `_audio_human_review.tsv`. Most rows are 5-second decisions.
18. ⚑ **Final 50-row sanity scroll on `06-final.tsv`** (~30 min): catches systemic errors.
19. `python build/verify_all.py` passes; `pytest tests/test_invariants.py` passes.
20. **Ongoing**: study with the deck. Use Anki's flag-for-regeneration field as you encounter issues. Run `python build/regenerate_flagged.py` weekly.

## Cost & resource budget

Three cost-reduction levers built in: (a) Google Cloud TTS instead of ElevenLabs, (b) Batch APIs everywhere offline (50% off), (c) smaller juror models. Numbers below assume registry-current model pricing and the three levers active.

| Item | Cost (default lean) | Cost (premium opt-in) |
|---|---|---|
| Anthropic (generator + small juror + auditor share, batch-discounted) | ~$50–90 | ~$140–230 (sync, top-tier juror) |
| OpenAI (mini juror + validator + ASR + auditor share, batch-discounted) | ~$25–50 | ~$80–140 |
| Google (Flash juror + tiebreaker share + auditor share, batch-discounted) | ~$15–35 | ~$50–90 |
| Top-1000 N-best second generation + selection | ~$10–18 | ~$15–25 |
| ASR roundtrip (~38k clips, batch-discounted) | ~$5–15 | ~$10–30 |
| **TTS** (Google Cloud TTS Neural2 BR, ~970k chars × 2 voices, +30% buffer) | **~$30–60** | ~$260–430 (ElevenLabs swap) |
| Cloudflare R2 storage (~1.4 GB) | ~$0.02/month | ~$0.02/month |
| Cloudflare R2 egress | $0 | $0 |
| **One-time total** | **~$135–270** | ~$555–945 |
| Ongoing | ~$0/month + occasional weekly regeneration ($0.50–2/week) | same |

The premium column is the prior-pass numbers, kept for reference if the user later decides ElevenLabs naturalness is worth the ~$420–675 premium. The default lean column is the recommendation: ~$135–270 total, with quality differences invisible at A1.

If the user wants to push cost even lower:

- Switch TTS to local open-source (Kokoro / XTTS / MeloTTS): saves another $30–60. Requires a GPU-capable Mac and ~1 day of generation time. Quality is "adequate, not great" — fine for early study, may want re-generation later.
- Drop the third juror and adversarial auditor; rely on jury-of-2 + verdict auditor only: saves ~$30–60. Residual error rate climbs back toward ~2%, study-time flags toward ~3–5/week.

**Active human time**: **~2 hours total upfront** (seed manual splits + cost confirmation + voice IDs + pilot voice quality + small disagreement queue + final sanity scroll), plus **1–3 flags per week** of study-time flag-and-regenerate. Lean cost path does not change human time.

**Wall clock**: ~1–2 days of mostly unattended pipeline runs. Batch APIs add ~12–24h of async wait, but that wait costs zero attention.

## What this plan deliberately does NOT do

- No Anki note-type design, card templates, or `.apkg` generation (next phase). One exception: the note type must include a `flag_for_regeneration` field exposed as a tappable flag during review — required to close the study-time correction loop.
- No EN→PT production cards or cloze cards (deferred).
- No BP cross-check against external corpora (user opted out).
- No legal/travel expansion vocabulary (out of scope for now).
- No conjugation sub-deck (deferred).
- No derivational-family **review order** — `family_root` is a schema-reserved field for future use.
- No connected-speech IPA — gap documented, not patched.
- No storage of authoritative data as anything other than TSV files in this repo (simplicity over DB flexibility).
- No silent drops, silent replacements, silent merges, or silent regenerations. Everything is in the ledger or the manifest.
- **No upfront human QA queues across thousands of rows.** v2's six sense-review queues, the 200-headword spot-check, the 500-example spot-check, and the per-stage human approval gates are all dropped in favor of the LLM jury + auditor + ASR roundtrip + study-time flagging. Residual risk is shared-blind-spots between models; the mitigation is the study-time loop, not heroic upfront review.

## Residual risks of the lean approach

Honest about what the v3 reframe trades away:

- **Shared LLM blind spots.** Three frontier models from three families can still all confidently produce the same wrong answer on a subtle case (e.g., a low-frequency idiom translated in the same drift direction by all three). The two-auditor pass (adversarial + verdict, two further models) catches most of these. The N-best on top-1000 catches more on the rows that matter most. The study-time loop catches the rest. Residual: probably 0.3–1% of ~9,000 rows have undetected errors at deck-launch — call it ~30–90 rows. The user encounters and fixes ~1–3 per week of study, which is sustainable and far below the daily card-review count.
- **Audio rare failure modes.** ASR can mis-transcribe in the same direction TTS mis-pronounces (both treat a foreign loanword the same wrong way). Phonetic-distance check against eSpeak helps. Stricter top-1000 thresholds reduce frequency-weighted impact. Final defense is study-time flagging.
- **Sensitive-term policy drift.** The verdict auditor's check is the gate. If auditor policy understanding differs from the user's, some examples pass that the user would have rejected. Mitigation: keep [build/policies/sensitive_terms_policy.md](build/policies/sensitive_terms_policy.md) explicit and short (1 page, examples-driven), and feed it to **every** auditor and juror so all five models read identical constraints.
- **Cost overrun.** Five-model coverage triples or quadruples LLM line items vs. a single-model pipeline. Pre-flight cost report at step 12 is the gate before the audio line — the only line that's large in absolute terms.
- **Model deprecation mid-run.** A model named in `config/models.yaml` may be retired between dev and full-corpus runs. Mitigation: stage scripts read role-keys, not IDs; the registry records both `current_id` and `last_run_id` so re-runs are reproducible against the same model when possible, or fall forward to the new one with a recorded provenance note.
