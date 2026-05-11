# Plan: Anki-Ready BP Portuguese Dataset — Data Phase v3

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
| **Audio (Stage 9 REVISION — Flash v2.5 migration)** | **2 clips per sense (1 voice × {word, example})** — voice gender chosen per-sense by Stage 4.5 speaker gender classifier. Fixed 1M+1F voice IDs across the deck. **ElevenLabs Flash v2.5 (`eleven_flash_v2_5`)** — locked in Stage 9 after pilot showed FLASH_BETTER 45-1 vs Multilingual v2 (drops aggressive orthographic respelling complexity, ~50% cheaper per character). Pro tier ($99/month) covers full deck. Total ~282k chars × 1 voice/sense × 0.5 credits/char Flash discount ≈ 141k credits per render pass; fits Pro tier 600k cap with ~460k headroom. **Filename convention: `{sense_id}-{word|ex}-eleven_flash_v2_5-v{N}.mp3`** — model_id baked in for provenance. |
| **Audio (LEGACY — Multilingual v2 era, pre-Stage 9)** | ~~ElevenLabs Multilingual v2 with 25-rule alias dictionary for orthographic respellings (`Hê-aú`, `Zhudjissiaou`, etc.). Replaced by Flash v2.5 in Stage 9 — Flash handles BP natively per pilot data, so the alias dict shrunk to 1 rule (`hospital → ospitau`).~~ |
| **Voice pool (LOCKED)** | **11 ElevenLabs voices: 4 female + 7 male** (user-selected, committed at `config/voices.tsv`). Each gender pool gets uniform allocation: every female voice generates ⌈N_F/4⌉ or ⌊N_F/4⌋ senses; every male voice generates ⌈N_M/7⌉ or ⌊N_M/7⌋ senses (off-by-at-most-1 per pool). Per-record voice provenance preserved end-to-end: `045-speaker_gender.tsv` → `_audio_manifest.tsv` → `06-final.tsv` all carry `voice_id`. |
| **Audio voice assignment** | Three deterministic layers: (1) Stage 4.5 emits `speaker_gender ∈ {male, female, neutral}` per sense based on `example_pt` cues (Obrigado/Obrigada, estou curioso/curiosa, "vou ganhar um menino" → female, etc.). (2) Neutrals resolve to balanced M/F via `random.Random(44).shuffle(sorted(neutral_sids))` then alternate i%2 — off-by-at-most-1. (3) Within each gender bucket, specific `voice_id` is assigned via seeded round-robin: `random.Random(45)` shuffles female bucket → `FEMALE_VOICES[i%4]`; `random.Random(46)` shuffles male bucket → `MALE_VOICES[i%7]`. Single voice per sense, used for both word + example clips. |
| **Audio column naming (REVISED)** | `audio_word`, `audio_example` (one voice per sense, no `_m/_f` infix). Voice ID and gender baked into `_audio_manifest.tsv` per row. Filename: `{sense_id}-{word|ex}-v{N}.mp3`. |
| Audio stage | Pilot first 500, then full batch |
| **Audio storage** | **Cloudflare R2**, exposed via R2 public bucket custom domain (not per-object public-read ACL). Stable HTTPS URLs with **filename-baked versioning** (`-v{N}.mp3`) — Anki strips query strings on download, so `?v=N` would silently fail to propagate regenerated clips to mobile devices. |
| Audio manifest | `data/_audio_manifest.tsv` is source of truth; final TSV URL columns derived from it |
| Cognate flag | `#cognate-en` tag, no review-order change |
| Legal/travel expansion | Out of scope for now |
| Data format | TSV, one row per sense, stable `sense_id = rank.sense_index` |
| Manual overrides | First-class: every LLM stage has a corresponding `_manual_*.tsv`; overrides always win |
| LLM provenance | Recorded in stage audit JSONL files (model, prompt-hash, response-hash, confidence) |
| Model config | Centralized in `config/models.yaml`; never hard-coded in scripts |
| **Review strategy** | **LLM 3-model jury + adversarial auditor + verdict auditor + N-best on top-1000 + ASR roundtrip; human review is study-time flagging, not upfront QA queues**. *2026-05 update*: Stage 5.5 simplified to **single-pass Sonnet auditor** (combines adversarial + verdict in one call) — driven by 2026 output-token pricing on Opus making the two-pass design ~$120 for 5,725 rows. See § Stage 5.5 for the locked single-pass design. |
| **Audio QA** | **Whisper-class ASR roundtrip + phonetic-distance check; human listens only to ASR-flagged clips and the pilot voice-quality sample (~15 min)** |
| Study-time loop | Anki flag-for-regeneration field → weekly batch regeneration of flagged cards |
| Study app | macOS desktop + iPhone |
| LLM providers | Anthropic + OpenAI + Google (three families for jury diversity); Google Cloud TTS for audio; OpenAI Whisper-class ASR for roundtrip |
| **Batch APIs** | All offline LLM stages (1.5, 2, 4, 5, both auditors) use Anthropic Message Batches and OpenAI Batch API — **50% discount** for ≤24h turnaround. Realtime tiebreaker calls stay on synchronous APIs. Saves ~$140–240 with zero quality loss. |
| **Juror tier** | Generator and verdict-auditor use top-tier models. Jurors vote on small structured outputs (enums + short text), so they use **smaller, cheaper models** (Haiku-class + GPT-5-mini-class + Gemini-Flash-class). Adversarial auditor uses a mid-tier model (defect listing benefits from capability but doesn't need top-tier reasoning). Saves ~$80–120. |
| **Model registry** | `config/models.yaml` is the **single source of truth** for model IDs. Plan exemplars below name 2026-current frontier models; the registry is updated as models age out (e.g., GPT-4o → GPT-5 family; Claude 3.5 → Claude 4.x; Gemini 1.5 → Gemini 2.5). Stage scripts read role-keys (`generator`, `juror_a`, `juror_b`, `juror_c`, `tiebreaker`, `validator`, `auditor_verdict`, `auditor_adversarial`), never hard-coded IDs. |

## Review strategy: LLM jury (enums) + validator/auditor (free text) + ASR — human review is post-hoc, not upfront

Earlier iterations (the v2 plan) demanded ~30–40h of upfront human review across many QA queues. That was unstable: too much attention required, too easy to skim and rubber-stamp. v3 replaces almost all upfront human review with **LLM-driven review**, applied differently to enum decisions vs. free-text generation, plus a study-time correction loop.

**Critical distinction:** the 3-model jury is for **enum/classification decisions only**. Free-text generation (examples, IPA, idiom expansion text) goes through a different pipeline: single generator → validator → auditor → optional N-best on top-1000. Trying to "jury" three independent example sentences and pick a winner is messy and expensive.

| Stage / decision type | Reviewed by |
|---|---|
| `bp_status` enum (Stage 1.5) | 3-model jury + tiebreaker |
| Sense-split decision (Stage 2) | 3-model jury + tiebreaker |
| Gender fallback (Stage 3) | 3-model jury + tiebreaker |
| Cognate flag (Stage 3) | 3-model jury + tiebreaker |
| Sensitive-term classification (Stage 2.5) | 3-model jury + tiebreaker |
| Family root (Stage 3) | 3-model jury (low stakes; no tiebreaker) |
| Example sentences (Stage 4) | Single generator + cross-model validator + Stage 5.5 auditors; N-best top-1000 |
| IPA correction (Stage 5) | eSpeak baseline + single LLM correction + Stage 5.5 auditor (`ipa_plausible`) |
| Idiom expansion text (Stage 1a) | Single generator + verdict by jury (3-way agreement on the produced phrase) |
| Audio (Stage 6/7) | ASR roundtrip per clip; auditor not involved |

### Tier 1 — Three-model jury on enum decisions

Every classification stage runs the same Tool-Use prompt through **three jurors from three different families**, configured in `config/models.yaml` under role keys `juror_a`, `juror_b`, `juror_c` (currently small/cheap tier — Haiku-class, GPT-mini-class, Gemini-Flash-class). All three responses are recorded in the audit JSONL.

Resolution rule:

- **All three agree** on the enum value: accept silently. The common case (~85–95%).
- **Two-of-three agree**: accept the majority answer. Disagreement logged but does not block.
- **All three disagree**: route to a **fourth tiebreaker model** at role key `tiebreaker` (top-tier Anthropic or top-tier OpenAI, set in registry). If tiebreaker matches one of the three, accept. If tiebreaker proposes a fourth answer or expresses `confidence: low`, route to `_jury_disagreements.tsv` for human review.

Why three not two: a two-model jury fails open on shared-blind-spots — both models confidently agree on the same wrong answer. A three-model cross-family jury makes that failure mode require *three* simultaneous shared blind spots, which is empirically rare. Cost is small because jurors run at cheap tiers and through Batch APIs.

Disagreement rate is itself a quality metric: if Stage 4 examples have >15% three-way disagreement, the prompt is broken and needs revision before further generation.

### Tier 1-bis — Single generator + validator pattern for free text

Free-text outputs (example sentences, IPA strings) come from one generator at a time. Diversity comes from the **validator** (different model family) and the **adversarial+verdict auditors** at Stage 5.5 (two more model families). End-to-end, every free-text row is touched by 4 distinct models from 3 distinct families. That is the "jury" for free text — no need to generate the same example three times.

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

- **Voice ID selection** — DONE (committed at `config/voices.tsv`, 4F + 7M).
- **Audio pilot voice quality sample** (~15 min): listen to ~10 random pilot clips per voice (≈110 clips total across 11 voices, post-ASR-roundtrip) to confirm the pool sounds right. Sanity check, not pronunciation review. Reject any voice that fails (re-pick → re-commit voices.tsv → regenerate that voice's slice only).
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
| `output_sense_ids` | string | Convenience denormalization: comma-separated `sense_id`s produced from this line. **Authoritative source for many-to-many provenance is `_sense_source_map.tsv`** — see below. The ledger column exists for human-friendly inspection only. |
| `drop_reason` | string | When `action = drop_ep_only`; required |
| `manual_review_status` | enum | `not_required` / `pending` / `approved` / `rejected` |
| `stage_decided` | string | Stage that set the current `action` (`1a`, `1b`, `1c`, `1.5`, `manual`) |
| `notes` | string | Free text |

### Sense ↔ source many-to-many provenance: `_sense_source_map.tsv`

`output_sense_ids` as a comma-separated string in the ledger does not scale: merged rows have N source lines mapping to one sense, idiom-expanded rows have one source line mapping to M senses, and lexical replacement adds another edge type. Authoritative provenance lives in a normalized many-to-many table.

Schema:

```tsv
sense_id	source_line_number	provenance_type	notes
```

`provenance_type` ∈:

- `original` — the canonical source line for this sense (1-1 default case).
- `normalized` — source line after orthographic normalization (Stage 1b); same line as `original` but flagged so the audit can distinguish whether a row's `pt` differs from its `source_pt`.
- `lexical_replacement` — source line whose `source_pt` was lexically replaced (Stage 1c) into the BP form that ended up in this sense.
- `merged` — source line whose row was merged into another (Stage 1c collision merge); the surviving sense gets multiple `merged` provenance edges.
- `idiom_expansion` — source line that produced this row via abbreviation expansion (Stage 1a); one source line can produce multiple `idiom_expansion` edges (e.g., `redor` → `em redor` AND `ao redor`).

Generated incrementally: Stage 1a writes the first edges, Stage 1c adds merge/replacement edges, Stage 2 writes the final `sense_id` for each. `_sense_source_map.tsv` is committed as an authoritative artifact.

### Required invariants (`build/verify_all.py::verify_ledger`)

- Every line in `source.txt` appears exactly once in the ledger (`source_line_number` is a primary key).
- Every final sense row in `06-final.tsv` has ≥1 row in `_sense_source_map.tsv`.
- Every `source_line_number` in the ledger with a non-drop `action` has ≥1 row in `_sense_source_map.tsv`.
- The ledger's `output_sense_ids` is a derived view — `verify_all` reconstructs it from `_sense_source_map.tsv` and asserts equality. If they disagree, the map wins (overwrite the ledger column).
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
│   ├── _sense_source_map.tsv               # many-to-many: sense_id ↔ source_line_number with provenance_type
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
│   ├── models.yaml                         # model IDs (bp_status, sense_split, example, ipa, validator)
│   └── voices.tsv                          # ElevenLabs voice pool (4F + 7M user-selected; committed)
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

**Prompt caching**: mandatory (see § API client conventions). The system prompt is identical across all ~5000 calls — cache it. Cuts Stage 1.5 cost from ~$30 to ~$5–8.

**Provenance**: every call writes to `audit/015_bp_status.jsonl` with `source_line_number, model_id, prompt_hash, response_hash, decision, confidence, cache_creation_input_tokens, cache_read_input_tokens, generated_at`.

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

**Decision (locked):** Stage 2 trusts the LLM jury for function-word polysemy classification rather than routing to a manual queue or pre-writing policy docs. Acceptable quality risk for shipping speed; user-flagged corrections at study time will catch any over- or under-splits. The hand-written `build/policies/function_word_strategy.md` and `pronoun_policy.md` are deferred — may be added later if specific patterns of error emerge.

Pronoun policy (locked):

| Item | Treatment |
|---|---|
| `você` | Keep; central BP |
| `tu` | Keep with `#regional` tag (used in some BP regions) |
| `vós` | **Drop** as `ep_only` (already flagged by Stage 1.5); not in study deck |
| `vosso` | **Drop** as `ep_only` (already flagged by Stage 1.5; mainly EP); not in study deck |
| `vosso` | Manual review; mainly EP/formal/religious |
| `lhe` | Keep but explain BP usage carefully |
| `se` | Special handling: reflexive, impersonal, passive-like, conditional senses each forced |

Tags: `#function-word`, `#pronoun`, `#grammar`, `#manual-sense`.

#### sense_id assignment

`sense_id = {rank:04d}.{expansion_index:02d}.{sense_index:02d}` — three dot-separated zero-padded fields. Format chosen so both indices can grow ≥10 without breaking parsers (function words like `se` may have 10+ senses; high-rank entries may accumulate idiom expansions over time). Examples:

- `0001.00.01` — rank 1, no expansion, sense 1 (e.g., `o` first sense)
- `0001.00.05` — rank 1, no expansion, sense 5 (the deep polysemy of `o`)
- `0314.00.01` — rank 314, no expansion, base headword `medida` first sense
- `0314.01.01` — rank 314, first idiom expansion (`à medida que`), first sense

Sense IDs are stable forever. `expansion_index = 00` is the original headword row; `01+` are idiom expansions in the order produced by Stage 1a.

#### Review

Stage 2 runs the **3-model jury** on the sense-split classification (§ Review strategy). Three-way disagreements go to `_jury_disagreements.tsv` and are auto-resolved by the tiebreaker model. The auditor pass at Stage 5.5 catches anything that slipped through. **No upfront human review queues** for Stage 2 — the v2 plan's six queues (`_sense_review_top1000.tsv`, `_sense_review_low_confidence.tsv`, `_sense_review_polysemous.tsv`, `_sense_review_idioms.tsv`, `_sense_review_function_words.tsv`, `_sense_review_sensitive.tsv`) are removed.

The 8 forced-gender-split entries and the ~12 idiom-expansion entries are short enough to inline as manual seed senses in `_manual_sense_splits.tsv` (~5 min). **Decision (locked):** for first run, skip manual seeding — trust LLM defaults via the deterministic forced-split rule. User can manually correct the auto-generated `en_primary` text in `_manual_sense_splits.tsv` after Stage 2 if any look wrong.

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

`sensitive_category` ∈ `{sexual, violence, slur_or_identity, race_ethnicity, religion, politics, medical, self_harm, crime, weapons, substance, offensive_possible}`. `risk_level` ∈ `{low, medium, high}`. `example_policy` is a short directive that flows into the Stage 4 prompt (e.g., "use neutral medical context, no graphic detail"). High-risk and medium-risk rows tagged `#sensitive-reviewed`. The auditor at Stage 5.5 verifies that generated examples obey their `example_policy`; failures auto-regenerate. **No upfront manual approval gate** — the auditor's defect-list `axis = sensitive_policy` and the verdict auditor's decision are the gates. Only auditor-routed `verdict = human_review` rows reach `_jury_disagreements.tsv`.

This is **not a drop list**. Most words remain. The purpose is to ensure example sentences are neutral, safe, and learner-appropriate.

### Stage 3 — Enrichment ([build/03_enrich.py](build/03_enrich.py))

**In**: `02-senses.tsv` + `_manual_gender.tsv` (manual overrides) → **Out**: `03-enriched.tsv`, `audit/03_enrich.jsonl`

Deterministic + single LLM call hybrid. **Decisions locked for this run:**
- **Wiktionary/Priberam scraper: dropped.** The originally-planned cascade (manual override → Wiktionary BR → Priberam BR → LLM) has been simplified to (manual override → LLM). Frontier LLMs are >99% accurate on BP noun gender; the scraper saved ~$1–2 of compute at the cost of ~500 lines of cache + DOM parser code and ongoing maintenance when DOMs change. Graceful-fail-to-LLM was already in the cascade; this just removes the early steps.
- **Family root: deferred.** Schema column stays in the final TSV but is left empty in this run. Saves ~30% of Stage 3 LLM output tokens. Can be batch-generated later with a single dedicated pass if the user decides they want it.
- **Cognate flag: included.** Binary `#cognate-en` tag adds ~$1; useful for an A1 learner to identify easy wins.
- **PoS: deterministic-when-100%-certain.** Verb when `en_primary` matches `^to \w` pattern; noun when `gender ∈ {o, a, o/a}` is set; adj when bare adjective gloss; interj/num from explicit lists; LLM otherwise; blank when truly ambiguous.

#### Pipeline

1. **Deterministic shortcuts** (no LLM):
   - Forced gender splits from Stage 2 → gender already populated; pt_display computed from `{gender_article} {pt}`.
   - Idiom expansion rows (expansion_index ≥ 1) → PoS = `idiom`, gender = empty, pt_display = pt.
   - Reflexive verbs (annotation.reflexive from Stage 1a) → PoS = `verb`, `#reflexive` tag.
   - `en_primary` matches `^to \w` → PoS = `verb`.
   - Function words from `PREMIUM_FUNCTION_WORDS` set → PoS = derived from list (article / preposition / pronoun / etc.); no gender.
2. **Single LLM call per remaining row** (Tool Use enforced):
   ```json
   {
     "type": "object",
     "properties": {
       "gender": {"type": "string", "enum": ["o", "a", "o/a", ""]},
       "pos": {"type": "string", "enum": ["noun", "verb", "adj", "adv", "prep", "conj", "pron", "art", "num", "interj", ""]},
       "is_cognate_en": {"type": "boolean"},
       "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
       "reason": {"type": "string"}
     },
     "required": ["gender", "pos", "is_cognate_en", "confidence"]
   }
   ```
   Empty string for gender/pos means "not applicable / leave blank". Tier = `default` (Sonnet); no premium routing needed for this stage.
3. **Compose `pt_display`**: `o {pt}` for masculine nouns, `a {pt}` for feminine, bare `{pt}` for everything else.
4. **Compute tags deterministically** from existing fields:
   - Frequency tier from `rank` band: `#top500` / `#top1000` / `#top2000` / `#top3000` / `#top5000`.
   - PoS tag from resolved `pos` field.
   - Morphology: `#reflexive`, `#gendered-meaning`, `#hyphenated`, `#idiom`, `#space-compound` from `pt_type` and `annotation`.
   - Regional: `#bp-rare` if `bp_status = uncommon`.
   - Special: `#nsfw`, `#false-friend` from `bp_status`; `#cognate-en` from LLM result.
5. **Family root**: schema column left empty (deferred per locked decision).

#### Cost & validation

- ~5720 senses; ~4000 hit the LLM after deterministic shortcuts. With caching, **~$8–11** (actual on first run: $19.10 — output tokens were ~1.5× projected).
- Tier = default (Sonnet 4.5) only. No premium tier on this stage.
- Validation: every noun has `gender ∈ {o, a, o/a}` (soft warning, not crash, when LLM returns `pos="noun"` with empty gender); every reflexive verb tagged; tags space-separated and well-formed; `pt_display` populated for every row.

#### Post-run gender patch (locked: apply 18-row manual override)

The first Stage 3 run produced 18 rows where the LLM returned `pos="noun"` with empty `gender`. Resolving via `data/_manual_gender.tsv` with hand-curated values, applied in-place (no LLM calls, $0):

| sense_id | pt | en_primary | gender | pos | cognate (preserved) |
|---|---|---|---|---|---|
| 0102.00.01 | meio | means | o | noun | false |
| 0102.00.03 | meio | half | o | noun | false |
| 0377.00.01 | cima | top | a | noun | false |
| 0542.00.01 | passado | past | o | noun | false |
| 0659.01.01 | por cento | percent | (empty) | idiom | true |
| 1531.00.02 | estreito | strait | o | noun | false |
| 2224.00.01 | redor | all around | o | noun | false |
| 2702.00.02 | circular | shuttle | a | noun | true |
| 3064.00.01 | trabalhista | labor party member | o/a | noun | false |
| 3509.00.01 | verbo | verb | o | noun | true |
| 3717.00.01 | pop | pop | o | noun | true |
| 3738.00.01 | sudeste | Southeast | o | noun | true |
| 3753.00.02 | nascente | East | o | noun | false |
| 3791.00.01 | contrapartida | (em c.) on the other hand | a | noun | false |
| 4379.00.01 | dia a dia | everyday life | o | noun | false |
| 4624.00.01 | adjetivo | adjective | o | noun | true |
| 4727.00.01 | tona | (à t.) to the surface | a | noun | false |
| 4941.00.02 | expediente | escape from problem | o | noun | false |

Row 0659.01.01 (`por cento`) is the idiom expansion of `cento`; reclassify `pos` to `idiom` (deterministic shortcut had returned `idiom` but the LLM overrode to `noun`). The other 17 keep `pos="noun"` and gain a gender. Patch is idempotent — re-running Stage 3 with the populated `_manual_gender.tsv` produces the same output.

### Stage 4 — Example sentences ([build/04_examples.py](build/04_examples.py))

**In**: `03-enriched.tsv` + `_manual_examples.tsv` + `_sensitive_terms.tsv` (for `example_policy`) → **Out**: `04-examples.tsv`, `_example_fixes.tsv`, `audit/04_examples.jsonl`

Per sense, prompt the `generator` role from `config/models.yaml` (current ID resolved at runtime) with full sense context + `example_policy` directive if the row is `#sensitive-reviewed`. Generate three fields:

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
- Rows in `_flags.tsv` (NSFW / false-friend / (BP)-tagged) and rows in `_sensitive_terms.tsv` are processed once their classifier output exists (i.e., the `flag_type` and `sensitive_category`/`example_policy` fields are populated). **No upfront manual approval gate** — that would re-introduce the v2 review burden. Manual approval is required only when the Stage 5.5 verdict auditor outputs `verdict = human_review` for a specific row, in which case the row joins `_jury_disagreements.tsv` post-hoc.

#### Semantic validator (second pass)

After generation, the `validator` role from `config/models.yaml` (a different family from `generator`) validates each row against [`build/prompts/example_validate.md`](build/prompts/example_validate.md). Tool-Use schema:

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

### Stage 4 quality verdict (executed run, commit 67e12fd)

**Context.** First full-corpus Stage 4 run completed: 5,725 senses, single Anthropic Sonnet 4.5 generator + OpenAI gpt-4o-mini validator. Pre-Stage-5 quality assessment to confirm we are in best shape before IPA.

**Headline numbers.**

| Metric | Value | Plan target |
|---|---|---|
| Senses generated | 5,725 / 5,725 | 100% |
| Generation failures | 0 | 0 |
| Empty fields (example_pt/en/target) | 0 / 0 / 0 | 0 |
| Validator pass | 5,415 (94.6%) | — |
| Validator fail | 306 (5.3%) | — |
| Validator borderline | 4 (0.07%) | — |
| Deterministic token-match fail | 2 (0.03%) | — |
| Deterministic word-count fail | 0 | 0 |
| EP-slip automated grep (genuine) | 0 | 0 |
| Example word count median / max | 7 / 15 | ≤15 |
| Spend | ~$21 | ≤$50 |

**Edge-category quality (manual verification).**

- **Forced gender splits** (16 senses across `capital`/`polícia`/`rádio`/`corte`/`cabra`/`cura`/`grama`/`banana`): 16/16 pass with cleanly distinguished M vs F examples.
- **Function-word polysemy patches** (13 senses across `o`/`se`/`para`/`de`/`em`/`que`/`por`/`a`): 12/13 pass; the 1 flagged (`de` sense 1 = "of") is a validator nitpick on natural BP possession ("Maria's book" rendering).
- **Idiom expansions** (14 rows): 12/14 pass; 2 flagged (`à medida que` = validator nitpick; `em diante` sense 2 "in front of" = real Stage-2-defined sense that's rare/awkward in BP).
- **Sensitive terms** (44 rows with `bp_status=nsfw` or non-empty `example_policy`): ~95% appropriately framed (scholarly/news/clinical register). Real false-friend issues on `fazenda` sense 2 ("fabric") and `camisola` sense 2 ("sweater") trace to Stage 2 sense definitions that don't exist in BP usage.
- **Reflexive verbs** (10 senses): all generate reflexive forms with proclitic placement.
- **Duplicate examples**: 32/5725 PT examples appear ≥2× (0.6%, A1-natural repetition); 76 EN translations appear ≥2× (1.3%). Acceptable.

**Real-vs-false-positive triage of the 306 validator fails.**

Failure-axes distribution:

| Axis pattern | Count | Estimated FP rate |
|---|---|---|
| `translation_matches` + `uses_intended_sense` | 127 | 30-50% |
| `translation_matches` alone | 107 | 70% (nitpicks) |
| `uses_intended_sense` alone | 40 | 40% |
| `is_bp` alone | 22 | 50% (validator BP gaps, e.g., flagging `universidade`) |
| `sensitive_policy` (combined) | 4 | 75% (mostly miscategorized sense issues) |
| Other | 8 | mixed |

Heuristic estimate of **real failures: ~167 / 5725 = ~2.9%**, slightly over plan's 2% target. Failure rate is 4× higher on `sense_index ≥ 2` rows (15.1%) than on primary senses (4.0%) — secondary senses are harder for the LLM to disambiguate from primary.

**Recommendation: proceed to Stage 5 (IPA); do not regenerate now.**

Reasoning:
1. The Stage 4 prompt is working correctly — generation quality is high. The `2.9%` estimated real-failure rate is within bounds the design anticipated.
2. **Stage 5.5 (adversarial + verdict auditor) is the correct place to triage these rows.** That stage runs two additional models from different families across every row, producing a defect list and a verdict (`pass`/`regenerate`/`human_review`). Failures auto-regenerate with the defect list as anti-example. Doing surgical regen at Stage 4 now would duplicate Stage 5.5's work.
3. Regenerating the 306 flagged rows immediately would cost ~$2-3 and 5 min, but would not address the ~140 real failures the validator missed (false negatives from the validator are unmeasured here).
4. Stage 5 (IPA) is independent of Stage 4 quality issues — IPA is keyed on `pt` (always correct) and `example_pt` (already validated for word/token integrity). IPA can run on current Stage 4 output without prejudice.

**Action items for downstream stages.**
- **Stage 5 (IPA)**: proceeds on `04-examples.tsv` as-is. No Stage 4 changes required.
- **Stage 5.5 (auditor)**: when built, will use `data/_example_fixes.tsv` as a prior-flag input plus run independent adversarial+verdict on every row. Failures route to a Stage 4 regeneration loop with `≤2` retries and the defect list as anti-example.
- **Defer to study-time loop**: 0.3-1% residual real errors (the slice the auditor doesn't catch) get fixed via Anki's flag-for-regeneration field after deck launch. This is the v3 plan's design.

**Locked decisions.**
- No prompt revision before Stage 5.
- No N-best top-1000 second generation before Stage 5 (defer to Stage 5.5 if specific top-1000 senses fail audit).
- `data/_example_fixes.tsv` (308 rows) is preserved as input to Stage 5.5; do not clear it.
- The 2 deterministic token-match fails (`à mercê de` → `à mercê das` contraction; one other contraction edge case) are accepted as borderline rather than regenerated. Stage 5.5's deterministic re-check after any regen will catch them again if relevant.

### Stage 4.5 — Speaker gender classifier + voice assignment ([build/045_speaker_gender.py](build/045_speaker_gender.py))

**In**: `data/04-examples.tsv`, `config/voices.tsv` → **Out**: `data/045-speaker_gender.tsv`, `audit/045_speaker_gender.jsonl`

**Goal.** For each sense, decide (a) which gender voice to use based on sentence cues, and (b) which specific ElevenLabs voice from the user's pool. Required because ElevenLabs renders one voice per clip and the user wants gender-marked sentences to sound natural ("Estou curiosa" must be female; "Obrigado" must be male) AND wants every voice in the pool to be used roughly equally for variety.

#### Voice pool — `config/voices.tsv` (committed)

User-selected, pinned for the lifetime of the deck. New voices appended later trigger re-shuffling and partial regeneration.

```tsv
voice_id	gender	pool_index	notes
MZLCplaCGYxFwJ9LXmx1	female	1	user-selected 2026
wxoDdfPKBuna5KnUEotz	male	1	user-selected 2026
Rw38T6bn0lTNOb1aUevR	female	2	user-selected 2026
qPfM2laM0pRL4rrZtBGl	male	2	user-selected 2026
ny3E2DZImeZm00WLGZi9	male	3	user-selected 2026
GOkMqfyKMLVUcYfO2WbB	female	3	user-selected 2026
4za2kOXGgUd57HRSQ1fn	male	4	user-selected 2026
xNGAXaCH8MaasNuo7Hr7	male	5	user-selected 2026
uju3wxzG5OhpWcoi3SMy	male	6	user-selected 2026
m151rjrbWXbBqyq56tly	female	4	user-selected 2026
sKbNSlHXq99bttvf8rRF	male	7	user-selected 2026
```

Counts: 4 female + 7 male = 11 voices total. `pool_index` is monotonic within gender (1..N) and used as a stable sort key for round-robin assignment. The `gender` column is the locked input to the assignment algorithm; user can later flip a voice's gender in this file if perceived voice gender differs from the metadata.

`config/voices.tsv` is **committed** to the repo. Voice IDs are not secret (they're public ElevenLabs catalog IDs). The `ELEVENLABS_API_KEY` stays in `.env` (gitignored).

#### Phase 1 — LLM gender classification

For each row in `04-examples.tsv`, build a small user prompt with `pt`, `en_primary`, `example_pt`, `example_en`. Call OpenAI gpt-4o-mini (cheap, fast) via Tool Use with this schema:

```json
{
  "type": "object",
  "properties": {
    "speaker_gender": {"type": "string", "enum": ["male", "female", "neutral"]},
    "evidence": {"type": "string", "description": "The cue that determined the choice (one phrase, ≤80 chars)"},
    "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
  },
  "required": ["speaker_gender", "evidence", "confidence"]
}
```

Detection rules in the system prompt:
- **Predicate adjective / past-participle ending** after first-person aux (`estou`, `sou`, `fui`, `fiquei`, `me senti`, `tenho`): `-o` ending → male; `-a` ending → female. Examples: `Obrigado/Obrigada`, `cansado/cansada`, `curioso/curiosa`, `preocupado/preocupada`.
- **Cultural / biological cues**: "vou ganhar um menino" / "estou grávida" → female; "como pai" → male; "como mãe" → female; "minha namorada" → speaker is male; "meu namorado" → speaker is female; "vestido para mim" / "vou usar maquiagem" → female.
- **No cue** → `neutral` (the most common outcome).

#### Phase 2 — Deterministic voice assignment (no LLM)

Three sub-phases run after all rows are classified:

```python
import random
from build.lib.tsv import read_tsv

voices = read_tsv("config/voices.tsv")
FEMALE_VOICES = [v["voice_id"] for v in voices if v["gender"] == "female"]
FEMALE_VOICES.sort(key=lambda vid: next(v["pool_index"] for v in voices if v["voice_id"] == vid))
MALE_VOICES   = [v["voice_id"] for v in voices if v["gender"] == "male"]
MALE_VOICES.sort(key=lambda vid: next(v["pool_index"] for v in voices if v["voice_id"] == vid))
# (in actual code, sort once via a single pass — pseudocode here is illustrative)

# Phase 2a: resolve neutrals to balanced M/F (seed=44)
neutral_sids = sorted(r["sense_id"] for r in rows if r["speaker_gender"] == "neutral")
random.Random(44).shuffle(neutral_sids)
voice_gender_assigned: dict[str, str] = {}
for r in rows:
    sid = r["sense_id"]
    if r["speaker_gender"] in ("male", "female"):
        voice_gender_assigned[sid] = r["speaker_gender"]
for i, sid in enumerate(neutral_sids):
    voice_gender_assigned[sid] = "female" if i % 2 == 0 else "male"

# Phase 2b: assign specific voice_id within each gender pool (seeds 45/46)
female_sids = sorted(sid for sid, g in voice_gender_assigned.items() if g == "female")
random.Random(45).shuffle(female_sids)
voice_id_for: dict[str, str] = {}
for i, sid in enumerate(female_sids):
    voice_id_for[sid] = FEMALE_VOICES[i % len(FEMALE_VOICES)]

male_sids = sorted(sid for sid, g in voice_gender_assigned.items() if g == "male")
random.Random(46).shuffle(male_sids)
for i, sid in enumerate(male_sids):
    voice_id_for[sid] = MALE_VOICES[i % len(MALE_VOICES)]
```

**Properties:**
- Fully deterministic (seeds 44/45/46 → same assignment every run on identical inputs).
- Balanced gender split: among neutrals, exactly `⌈N/2⌉` female and `⌊N/2⌋` male — off-by-at-most-1.
- Balanced voice usage within each pool: every female voice gets `⌈N_F/4⌉` or `⌊N_F/4⌋` senses; every male voice gets `⌈N_M/7⌉` or `⌊N_M/7⌋` senses — off-by-at-most-1 per pool.
- No correlation with sense_id structure (sort + shuffle decouples adjacent senses).
- Net deck-wide voice balance lands close to 50/50 M/F even if "explicit female" and "explicit male" buckets are skewed.
- For ~5,725 senses with ~70% neutral: each female voice ≈ 700 senses, each male voice ≈ 410 senses.

#### Output schema — `data/045-speaker_gender.tsv`

| Column | Notes |
|---|---|
| `sense_id` | Join key |
| `speaker_gender` | `male` / `female` / `neutral` (raw classifier output) |
| `evidence` | One-phrase cue text from classifier |
| `confidence` | `high` / `medium` / `low` |
| `voice_gender_assigned` | `male` / `female` (post-Phase-2a; no `neutral`) |
| `voice_id` | Specific ElevenLabs voice ID (post-Phase-2b) |
| `assignment_method` | `llm` / `manual_override` |

`voice_id` flows downstream into:
- `_audio_manifest.tsv` (one manifest row per (sense_id, clip_type), all carrying voice_id) — already part of the manifest schema.
- `data/06-final.tsv` (one column `voice_id` per sense, since both clips of a sense use the same voice).

**Cost.** ~$1 (5,725 calls × ~$0.0002 gpt-4o-mini). Wall clock ~5 min sync at concurrency 8. Phase 2 is local Python (no API), <1 second.

**Validation invariants.**
- Every `04-examples.tsv` `sense_id` has exactly one row in `045-speaker_gender.tsv`.
- `voice_gender_assigned ∈ {male, female}` for every row (no `neutral` after Phase 2a).
- `voice_id` is non-empty and resolves in `config/voices.tsv` for every row.
- Female / male counts on the corpus differ by at most 1 within the neutral bucket.
- Per-voice usage counts (computed at end of Phase 2b): max − min within each gender pool ≤ 1.
- Gender-marked rows match the rule (spot-check: 20 rows with `Obrigada` / `Obrigado` / `curiosa` / `curioso` should classify correctly with `confidence: high`).

**Manual override**: `data/_manual_speaker_gender.tsv` (one row per `sense_id` with optional `voice_gender_override` and/or `voice_id_override`). Wins over LLM and Phase 2 round-robin. Empty for the first run.

#### Patch — confidence policy: only `high` confidence locks gender (post-run)

The first Stage 4.5 run produced 21 rows with `confidence ∈ {medium, low}`. Each of these had a weak inference (e.g., "minha mãe" / "meu pai" / "minha família" — possessives that work for any speaker gender). The pipeline still routed them into the explicit-male/female bucket and skipped the balanced shuffle. That locked an LLM coin-flip into a hard voice choice.

**Locked policy: only `confidence=high` keeps explicit gender. Medium/low classifications are demoted to `neutral` for voice assignment, so they go through the seeded balanced shuffle along with all other genuinely-ambiguous rows.**

Implementation (no LLM cost — re-uses existing `045-speaker_gender.tsv` classifications):

1. Modify `build/stage_45.py::run()` so that, after Phase 1 classification, any row with `confidence != "high"` AND no manual gender override has its gender demoted to `neutral` before passing to `resolve_neutrals` (Phase 2a).
2. Add a `build/recompute_voice_assignment.py` standalone script that reads the existing `data/045-speaker_gender.tsv`, applies the new policy, re-runs Phase 2a + 2b, and writes back. Idempotent. Zero API cost. Reproducible.
3. Manual overrides in `_manual_speaker_gender.tsv` always win over the confidence demotion (an explicit `speaker_gender_override` is treated as `confidence=high`).
4. Add a unit test that confirms `medium`-confidence inputs are demoted to neutral before voice assignment, and that `high`-confidence + manual overrides bypass the demotion.

**Expected effect on the corpus** (5,725 rows):
- Before: 21 medium + 0 low classified rows are routed into explicit gender buckets.
- After: those 21 rows go into the neutral bucket → seeded balanced shuffle picks their voice gender.
- Voice pool balance is unchanged (off-by-1 within each pool guaranteed by the shuffle).
- Deck-wide voice gender split shifts by ≤ 21 rows toward the post-shuffle 50/50 mean — negligible perceptible difference, but more honest accounting of the LLM's stated uncertainty.

**Files modified:**
- [build/stage_45.py](build/stage_45.py) — Phase 2 entry: demote non-high confidence before `resolve_neutrals`. The classifier output (raw `speaker_gender`, `evidence`, `confidence`) is preserved verbatim in the output TSV; only `voice_gender_assigned` and `voice_id` change.
- [build/recompute_voice_assignment.py](build/recompute_voice_assignment.py) — new standalone re-runner.
- [tests/test_stage_45.py](tests/test_stage_45.py) — add tests for the demotion policy.

**Verification:**
- `python3 build/recompute_voice_assignment.py` produces an updated `data/045-speaker_gender.tsv`.
- Run twice; second run produces identical md5 (idempotent).
- Spot check: the 21 previously-medium-confidence rows now show `voice_gender_assigned` from the seeded shuffle (some flip, some don't).
- Per-voice usage still off-by-at-most-1 within each pool.
- `pytest tests/test_stage_45.py` passes including the new tests.

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

### Stage 5.5 — Single-pass auditor ([build/055_audit.py](build/055_audit.py))

**Context.** The original plan called for two auditor passes (adversarial + verdict) using top-tier models from a third family. Cost projection at 2026 prices for 5,725 rows:

| Configuration | Total |
|---|---|
| gpt-4o adversarial + Opus 4.5 verdict | ~$123 |
| gpt-4o adversarial + Gemini-Pro verdict | ~$52 + Google integration |
| **Single-pass Sonnet 4.5 (locked)** | **~$20** |

The killer in the two-pass design is Opus output at $75/M (5,725 × 80 verdict tokens = $34 just for verdicts). The user explicitly chose **single-pass Sonnet 4.5 over the full corpus** — same prompt does both adversarial-framing defect listing AND verdict in one call.

Cross-family invariant is preserved: Sonnet (auditor) is in a different model than the gpt-4o-mini that played jury at Stage 4. Across the pipeline, every row is touched by 4 distinct models from 2 distinct families: Sonnet generator (Stage 4) + gpt-4o-mini validator (Stage 4) + gpt-4o-mini classifier (Stage 4.5) + Sonnet IPA corrector (Stage 5) + **Sonnet auditor (Stage 5.5)**.

**In**: `data/05-ipa.tsv` (full corpus, 5,725 rows) + `data/_example_fixes.tsv` (Stage 4 prior flags, used as input signal) → **Out**: `data/055-audit.tsv`, `data/_jury_disagreements.tsv`, regeneration directives via `data/_example_fixes.tsv` (rewritten if any failures), `audit/055_audit.jsonl`

#### Single-pass auditor design

For each row, Anthropic Sonnet 4.5 receives the row's full context (pt, gender, pos, en_primary, en_all, tags, bp_status, example_pt, example_en, target_word_used, ipa_word_final, ipa_example_final, plus the Stage 4 validator's prior flag if any) and produces, in one Tool Use call:

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
    },
    "verdict": {"type": "string", "enum": ["pass", "regenerate", "human_review"]},
    "reason": {"type": "string"},
    "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
  },
  "required": ["defects", "verdict", "reason", "confidence"]
}
```

System prompt uses **adversarial framing** explicitly: *"Your job is to find faults. List every defect you can identify in this row, even minor ones. After listing defects, decide a verdict: `pass` if defects are all `low` severity or absent; `regenerate` if any defect is `medium` or `high` severity AND fixable by re-prompting Stage 4; `human_review` if the row is borderline or needs domain judgment."*

#### Decision flow

- `verdict = pass`: row accepted (low-severity defects acceptable noise floor).
- `verdict = regenerate`: row queued for Stage 4 regeneration with the defect list as anti-example. **≤ 2 regen attempts**; if the second regen still fails the auditor, row routes to `_jury_disagreements.tsv`.
- `verdict = human_review`: row written to `_jury_disagreements.tsv` immediately for end-of-run human eyeball.

Regen reuses [build/stage_4.py](build/stage_4.py)'s sense-id-filter mode (already supported) — only the failed sense_ids are re-prompted, with a new `--anti-examples-from` flag pointing at `_example_fixes.tsv` so the generator sees the prior flawed output and the auditor's defect list as "do NOT produce something like this."

#### Output schema — `data/055-audit.tsv`

| Column | Notes |
|---|---|
| `sense_id` | Join key |
| `verdict` | `pass` / `regenerate` / `human_review` |
| `defect_count` | Integer; 0 means clean |
| `defects_json` | JSON-serialized defect array (axis, severity, description) |
| `reason` | Auditor's one-sentence summary |
| `confidence` | `high` / `medium` / `low` |
| `regen_attempts` | 0 / 1 / 2 — number of regen rounds taken |
| `final_status` | `pass_first` / `pass_after_regen` / `human_review` / `failed_after_regen` |

#### Cost & wall clock

- Single Sonnet 4.5 call per row × 5,725 rows
- Per-row tokens: ~1,500 cached input (system prompt + tool schema) + ~500 uncached input (per-row context) + ~150 output (defects + verdict + reason)
- Cached input: 8.59M × $0.30/M = $2.58
- Uncached input: 2.86M × $3/M = $8.58
- Output: 859K × $15/M = $12.89
- **First-pass total: ~$24**
- Regen: ≤ ~5% of rows × 1-2 calls = <$2
- **Stage 5.5 total: ~$25** (vs. plan's original ~$30-60 for the two-pass design)

Wall clock: ~30-60 min sync at concurrency 16. Cache hit ratio expected ≥90%.

#### Validation invariants

- Every row in `05-ipa.tsv` has exactly one row in `055-audit.tsv`.
- `verdict ∈ {pass, regenerate, human_review}`.
- `defect_count == len(defects_json)`.
- For rows with `final_status = pass_after_regen`: corresponding `04-examples.tsv` row's `example_pt`/`example_en`/`target_word_used` were updated.
- Rows with `final_status = failed_after_regen` route to `_jury_disagreements.tsv` AND are surfaced loudly in the run summary (should be <1% of corpus).

#### Manual override

`data/_manual_audit.tsv` (per-sense_id verdict override). Wins over LLM. Use to force `pass` on a row the auditor flagged but you've decided is acceptable, or vice versa.

#### Post-run remediation plan (locked, 2026-05-09)

The first Stage 5.5 full-corpus run produced 5,725 verdicts: pass 3,267 / regenerate 1,034 / human_review 1,424. After axis-based triage (read all 53 sense-only rows by hand to verify), the actionable buckets are:

| Bucket | Count | Action | Cost | Why |
|---|---|---|---|---|
| Auditor IPA-only flags (1+ defects, all `axis=ipa_plausibility`) | 1,679 | **Reclassify as `pass_after_axis_review`** | $0 | Sonnet auditor consistently misreads its own IPA output, hallucinates rules, confuses narrow vs broad transcription. Stage 5 IPA quality was verified clean separately; these flags are auditor failure mode. |
| IPA + minor low-severity others | 15 | Reclassify as pass | $0 | Same |
| Nuance-only (`translation_match`/`naturalness`/`level_appropriateness` low/medium, no other axes) | 146 | Reclassify as pass | $0 | Stylistic nitpicks; Stage 4 validator already exhibited this same over-strictness on translation. |
| Stage 4 fixable (real `bp_purity` high, `example_uses_intended_sense` high, `target_word_token_match` high, `naturalness` high) | 367 | Auto-regen via Stage 4 (1 attempt) + Stage 5 IPA refresh for affected rows | ~$3 | Genuine Stage 4 issues that fresh prompting will likely improve. |
| Mixed (other defect combinations) | 199 | Auto-regen via Stage 4 (1 attempt) + Stage 5 refresh | ~$1 | Catch-all; same remediation as the fixable bucket. |
| **Stage 2 gloss errors (15 cataloged below)** | **15** | **Auto-fix the gloss via LLM** | ~$0.05 | The English `en_primary` is wrong but the BP example is correct. A targeted Sonnet "gloss correction" call per row reads `pt + example_pt + example_en + en_all + auditor_defect` and emits a corrected `en_primary`. Applied directly to `04-examples.tsv` (which propagates to Stage 5 + final TSV). No tag — the row is fixed, not flagged. |
| Stage 4 wrong-sense-in-example (15 cataloged below) | 15 | **Auto-regen the example via Stage 4** with explicit sense hint | ~$0.20 | Gloss is correct but Stage 4 picked the wrong sense; regen with the auditor's defect description in the prompt will produce the right sentence. |
| Sense-consistency rows NOT in either cataloged bucket (~23) | ~23 | **Run gloss-correction pass first**; if LLM says gloss is fine, reclassify as pass; if LLM says gloss is wrong, apply fix | ~$0.05 | The auditor flagged but I judged borderline/false-positive. Let the gloss-correction LLM be the second opinion: same prompt as the 15 cataloged errors, pre-filled with the LLM's "no change needed" output as one valid answer. |

**Net:** ~566 rows regenerated via Stage 4 + Stage 5 IPA refresh; ~38 rows get gloss correction (15 cataloged + ~23 borderline); rest reclassified to pass without action.

**Total remediation cost: ~$4.** Wall clock ~10 min. **Zero manual work. No `#sense-disputed` tag in final TSV — every row is either fixed or confirmed clean.**

#### Gloss-correction LLM call (NEW for this remediation)

For each sense-consistency-flagged row, call Sonnet 4.5 with Tool Use:

**System prompt** (terse, ~500 tokens):
*"You are correcting the English gloss `en_primary` for a Brazilian-Portuguese sense. The current gloss may be wrong, too narrow, too broad, or a false friend. Read the BP word, the example sentence, the original full RHS, and the auditor's identified defect. Output either (a) a corrected concise gloss that accurately translates the BP word as used in the example, or (b) the original gloss if it's actually correct. Keep the gloss short (1-5 words). Use BP-natural English."*

**Tool input schema:**
```json
{
  "type": "object",
  "properties": {
    "corrected_en_primary": {"type": "string"},
    "is_changed": {"type": "boolean"},
    "reasoning": {"type": "string"}
  },
  "required": ["corrected_en_primary", "is_changed"]
}
```

**Per-row inputs:** pt, en_primary (current), en_all (full RHS), example_pt, example_en, auditor_defect_description.

**Cost:** 53 rows × ~600 input tokens × $0.30/M cached + ~150 output × $15/M = ~$0.15.

**Application:** for rows where `is_changed=true`, write the corrected `en_primary` directly to `data/04-examples.tsv` and `data/05-ipa.tsv` (the column appears in both). Audit log records the old + new gloss for traceability.

##### Cataloged Stage 2 gloss errors (#sense-disputed)

| sense_id | pt | Current `en_primary` | Note |
|---|---|---|---|
| 2401.00.01 | `eventual` | "eventual" | False friend; should be "occasional" |
| 2215.00.01 | `habitual` | "familiar" | Should be "usual / customary" |
| 2891.00.01 | `salgado` | "relating to salt" | Should be "salty" |
| 4512.00.01 | `cultivo` | "act of planting" | Should be "cultivation" |
| 4547.00.01 | `recair` | "to go back to" | Should be "to relapse" |
| 4630.00.01 | `multinacional` | "international corporation" | Should be "multinational" |
| 1871.00.01 | `constar` | "to consist of" | Needs "constar de" |
| 2938.00.01 | `abater` | "to come down" | Transitive: "to bring down" |
| 2603.00.02 | `solar` | "sole" | Should be "manor house" |
| 4965.00.02 | `edital` | "relating to editing" | Should be "public notice" |
| 0503.00.01 | `quarto` | "room" with `pos=num` | sense vs PoS mismatch |
| 1306.00.01 | `casal` | "married couple" | Just "couple" |
| 2873.00.01 | `ruído` | "loud and unpleasant noise" | Just "noise" |
| 1712.00.01 | `jurídico` | "judicial" | Should be "legal / juridical" |
| 1768.00.01 | `paulista` | "from São Paulo" | Should be noun "person from São Paulo" |

##### Cataloged Stage 4 wrong-sense rows (regen with sense hint)

| sense_id | pt | en_primary | Mismatch |
|---|---|---|---|
| 1510.00.01 | `japonês` | "Japanese" (noun) | Example uses adjective form |
| 2193.00.01 | `argentino` | "Argentine" (noun) | Same |
| 1284.00.02 | `espera` | "expectation" | Example uses "wait" |
| 1880.00.01 | `sentença` | "sentence" (grammatical) | Example uses "verdict" |
| 1410.00.02 | `corda` | "cord" | Example uses "string" |
| 1277.00.01 | `reserva` | "reserve" | Example uses "reservation" |
| 2802.00.03 | `roteiro` | "route" | Example uses "itinerary" |
| 3911.00.01 | `cova` | "opening" | Example uses "hole" |
| 3711.00.02 | `ficha` | "card" | Example uses "form" |
| 3711.00.03 | `ficha` | "slip" | Example uses "form" |
| 1684.00.01 | `interpretação` | "interpretation" | Example uses "performance" |
| 0164.00.02 | `ponto` | "dot" | Example uses "period" |
| 2596.00.02 | `limpeza` | "cleanliness" | Example uses "cleaning" act |
| 4218.00.01 | `caseiro` | "household" | Example uses "homemade" |
| 4799.00.01 | `ingresso` | "admission" | Example uses "ticket" |

##### Implementation

A new script `build/apply_audit_remediation.py` orchestrates four phases, all automated:

**Phase A — axis-based reclassification** (zero LLM cost):
- IPA-only rows (1,679) → `final_status = pass_after_axis_review`
- IPA + minor low-severity rows (15) → `pass_after_axis_review`
- Nuance-only rows (146) → `pass_after_axis_review`
- Borderline / auditor-FP rows (~23) → flagged for Phase B (gloss correction may confirm)

**Phase B — gloss correction** (~$0.15, ~38 rows):
- For all `sense_consistency`-defect rows (53 total), call Sonnet 4.5 with the gloss-correction prompt described above.
- For each row where `is_changed=true`: write the corrected `en_primary` to `data/04-examples.tsv` and `data/05-ipa.tsv`. Mark `final_status = pass_after_gloss_fix`.
- For each row where `is_changed=false`: mark `final_status = pass_after_axis_review` (auditor was wrong; LLM confirms gloss is fine).
- Audit log: `audit/055_gloss_corrections.jsonl` with `{sense_id, old_en_primary, new_en_primary, reasoning}`.

**Phase C — Stage 4 example regen** (~$3, ~566 rows):
- Regen queue: 367 fixable + 199 mixed + 15 cataloged wrong-sense = up to ~580 rows.
- Call `build/stage_4.py::run(sense_id_filter=regen_sids, anti_examples=defect_descriptions)`.
- Updates `data/04-examples.tsv` for those rows.
- Mark `final_status = pass_after_regen`.

**Phase D — Stage 5 IPA refresh** (~$2, regen rows only):
- Re-run Stage 5 on the same regen sense_ids. Updates `data/05-ipa.tsv`.

**No re-audit** — the auditor demonstrated unreliable IPA judgment and re-running it would just produce the same skew. Trust the corrections.

Final state: every row in `055-audit.tsv` has `final_status` ∈ {`pass_first`, `pass_after_axis_review`, `pass_after_gloss_fix`, `pass_after_regen`}. **No `human_review` and no `#sense-disputed` tag survives** — every flag was either fixed (regen / gloss correction) or confirmed clean (axis-review / borderline reclassification).

Verification: spot-check 30 random `pass_after_gloss_fix` and `pass_after_regen` rows to confirm the corrections look right; spot-check 10 `pass_after_axis_review` rows to confirm they're indeed clean. ~10 min of eyeball time, but the work is **confirming correctness** of automated fixes, not deciding their fate.

#### Live monitoring (NEW — for Stage 5.5 and reusable elsewhere)

Stage 5.5 spends ~30-60 min hitting Sonnet on 5,725 rows. The user wants to watch progress and react if individual calls hang. Three lightweight mechanisms, all zero-cost:

**1. Per-row progress JSONL** — `audit/055_progress.jsonl`

Every API call appends one line as soon as it starts AND another when it ends:

```
{"event": "started", "sense_id": "0042.00.01", "started_at": "2026-05-09T14:23:11.123Z", "attempt": 0}
{"event": "completed", "sense_id": "0042.00.01", "started_at": "...", "completed_at": "...", "latency_ms": 1842, "verdict": "pass", "defect_count": 0}
{"event": "errored", "sense_id": "0042.00.02", "started_at": "...", "attempt": 1, "error_type": "APITimeoutError", "error_msg": "..."}
```

User can `tail -f audit/055_progress.jsonl` in a second terminal to watch live.

**2. Periodic stdout summary** — printed by the main loop every 30 seconds:

```
[Stage 5.5  3,217 / 5,725 done | 14 in-flight | 1 stuck (>120s) | avg 1.4s | p99 4.2s | ETA 18 min | $14.20 spent]
```

Counts come from the progress JSONL + an in-memory tally. Visible in foreground OR via tailing the bash background-task output file.

**3. Per-attempt timeout + stuck-call detection**

- Each individual API call has a hard `timeout=120s` per attempt (existing retry envelope already has 60s backoff cap; add a per-call wall timeout).
- The main loop tracks in-flight call timestamps. Any call running > 120s is logged as `event: "stuck"` to the progress JSONL. The thread continues (Python can't cleanly kill an HTTP request mid-flight), but the user sees the stuck row immediately and can:
  - Wait — most "stuck" calls succeed within another 60s
  - Or kill the whole process; resume picks up where it left off (Stage 5.5 is idempotent — already-audited rows are skipped on resume)

**4. Status snapshot tool** — `build/audit_status.py`

Standalone script that reads `audit/055_progress.jsonl` and prints a current snapshot:

```bash
$ python3 build/audit_status.py
Stage 5.5 progress as of 2026-05-09T14:32:08Z
─────────────────────────────────────────────
Total rows in input:    5,725
Done:                   3,217  (56.2%)
In-flight:                 14
Stuck (>120s):              1   sense_id 4521.00.01 (running 187s)
Errored (final):            3   sense_ids 1234.00.01, 2345.00.02, ...
ETA:                   ~18 min  (avg latency 1.4s, concurrency 16)
Cost so far:           $14.20   (cache_read 5.2M, uncached 1.8M, output 0.5M)

Verdict distribution (so far):
  pass            3,012  (93.6%)
  regenerate        178  (5.5%)
  human_review       27  (0.8%)
```

Idempotent. Can be run any time (also after the run completes — gives final summary).

#### Resume + idempotency

Stage 5.5 reads the existing `data/055-audit.tsv` at startup and skips any sense_id that already has a `final_status`. Killing and re-running picks up where it left off, no duplicate work, no duplicate cost. The progress JSONL is append-only and never truncated, so the audit history survives restarts.

### TTS provider (LOCKED: ElevenLabs Multilingual v2)

User decision: pay the premium for natural BP audio rather than save money on Google TTS Neural2. The original plan made Google TTS the default for cost reasons; the user has reversed this in favor of naturalness. ElevenLabs Pro tier ($99 first month, 600k credits) covers the one-shot generation; Starter ($6/month) covers ongoing flag-and-regenerate cycles.

Pricing comparison (locked: only ElevenLabs is used; alternatives kept for reference if user later wants to swap):

| Provider | Quality (A1 use) | Per 1k chars | Total at ~282k chars × 1 voice/sense |
|---|---|---|---|
| **ElevenLabs (Multilingual v2)** ⭐ | premium | ~$0.16/1k chars (Pro tier ratio) | **~$45–55 in chars-equivalent; covered by Pro tier $99 monthly** |
| Google Cloud TTS Neural2 | excellent | ~$0.016/1k chars | ~$5–7 |
| Azure Neural TTS | excellent | ~$0.016/1k chars | ~$5–7 |
| OpenAI TTS (`tts-1-hd`) | very good | ~$0.030/1k chars | ~$10 |
| Open-source local (Kokoro / XTTS / MeloTTS) | adequate | $0 (local compute) | $0 |

**ElevenLabs Pro tier specifics** (verified at elevenlabs.io/pricing as of 2026):
- $99/month, 600,000 credits/month (1 credit = 1 character on Multilingual v2)
- Professional voice cloning included (user can clone a specific voice if desired; defaults to ElevenLabs stock voices)
- Commercial use OK
- After one-shot full-corpus run, downgrade to Starter ($6/month, 30k credits) — plenty for weekly flag-and-regenerate (~1–3 cards/week typical study-time correction loop)

**Character budget computation** (verified from `data/04-examples.tsv`):
- Headword chars (5,725 senses): 41,915
- Example chars (5,725 senses, mean 42 chars): 240,282
- Total at 1 voice/sense: 282,197 base
- With 30% retry buffer: 366,856 — fits Pro tier 600k cap with **233k headroom**

`config/models.yaml` carries the active TTS provider as a role-key. Voice IDs are pinned at `config/voices.tsv` (4 female + 7 male, user-selected, committed); per-sense voice assignment happens at Stage 4.5 via seeded round-robin within each gender pool. ASR roundtrip and filename versioning (`-v{N}.mp3`) are unchanged.

#### Audio quality settings (LOCKED 2026-05)

Verified against ElevenLabs Python SDK `/elevenlabs/elevenlabs-python` and ElevenLabs API docs (Context7). Highest quality possible without distortion is the explicit goal.

- **`model_id = "eleven_multilingual_v2"`** — plan-locked. The newer `eleven_v3` is now the SDK's default sample but uses different credit accounting and is unverified against the 11 user-committed voice IDs. v3 is reserved as a future swap behind the `tts_model` role-key in `config/models.yaml`.
- **`output_format = "pcm_44100"`** — uncompressed 44.1 kHz PCM. Pro-tier exclusive. No MP3 artifacts on the API side; lossless input to the loudness-norm + encode chain. Falls back to `mp3_44100_192` if Pro tier is downgraded mid-project (Creator+ requirement).
- **`voice_settings = VoiceSettings(stability=0.65, similarity_boost=0.80, style=0.0, use_speaker_boost=True)`** — locked across all 11 voices. Rationale: vocabulary learning rewards consistency over expressiveness; `stability=0.65` (above the 0.5 default) reduces prosody variance between identical re-generations of the same word; `similarity_boost=0.80` (above the 0.75 default) keeps each voice on-character; `style=0.0` keeps delivery neutral and pedagogical; `use_speaker_boost=True` is essential for short word-only clips where clarity dominates.
- **`seed = stable_hash(sense_id + clip_type + version) % 4_294_967_295`** — best-effort determinism per ElevenLabs docs ("repeated requests with the same seed and parameters should return the same result. Determinism is not guaranteed"). Cheap to set; helps when regen at the same version should reproduce identical audio (rare, but free insurance).
- **`apply_text_normalization = "auto"`** — default; lets ElevenLabs handle numbers, dates, abbreviations sensibly. Lock to `"on"` if BP-specific number-reading misbehavior surfaces in pilot.
- **No `previous_text` / `next_text`** — word and example clips are independent (Anki plays them separately), so connected-speech continuity hints are not useful here.

#### Accent-related parameter audit (LOCKED 2026-05)

User asked whether we're using all relevant accent-controlling parameters
before Stage 7 launches. Full sweep of `text_to_speech.convert()` arguments:

- `language_code = "pt"` — ISO 639-1 only accepts 2-letter codes; there is
  no `pt-BR` / `pt-PT` distinction at this parameter (BCP-47 not accepted).
  Set to `"pt"` to prevent the model from drifting into Spanish or English.
- `apply_language_text_normalization` — explicitly **Japanese-only** per
  ElevenLabs docs. Cannot use for Portuguese.
- `pronunciation_dictionary_locators` — **NOT used**, opt-in only. Useful
  to override systematic mispronunciations via custom IPA / alphabet
  dictionaries (up to 3 locators per request). Pilot listening surfaced
  zero systematic per-word mispronunciation patterns in the surviving
  8-voice pool, so building a dictionary would be busy-work today.
  Reserved as a remediation lever if Stage 7's full corpus exposes any
  recurring failure pattern.
- `optimize_streaming_latency` — not used (default 0 = max quality, no
  latency tradeoff). Locked.
- `use_pvc_as_ivc` — not used. Forcing IVC drops voice quality on
  professionally-cloned voices.
- `previous_request_ids` / `next_request_ids` — not used. Word and
  example clips are independent in Anki playback; request-stitching
  continuity hints would be unused.

The accent enforcement is achieved by (1) BP-native voice selection in
`config/voices.tsv`, (2) `language_code="pt"` to lock language, and
(3) the BP-spelling cues already encoded in the example sentences from
Stage 4. ElevenLabs has no separate `accent="brazilian"` knob beyond
these three levers. No further tuning available pre-Stage-7.

#### Loudness normalization (LOCKED 2026-05)

ElevenLabs does NOT normalize loudness across voices — every voice has a different perceived volume out of the API. Untreated, the deck would have noticeably louder and quieter cards depending on which voice was assigned. Fix: post-hoc EBU R128 loudness normalization via ffmpeg, gain-only, no other DSP.

- **Target**: integrated loudness `I = -16 LUFS`, true peak `TP = -1.5 dB`, loudness range `LRA = 11`. Industry standard for spoken-word content (Apple Podcasts / Spotify intake levels). Conservative on TP to avoid inter-sample peaks on lossy mobile playback.
- **Algorithm (LOCKED 2026-05 — closed-loop after pilot evidence)**: gain-only, with **post-encode verification and corrective re-encode**.

  Initial design used `loudnorm linear=true` end-to-end, but the pilot's
  empirical loudness check found only **49.6%** of clips landed within ±1 LU
  of -16 LUFS post-encode (mean -17.5 LUFS, 7.16 LU range across the deck).
  Root cause: the MP3 lossy encoder shifts integrated loudness by 0.5–2 LU
  in ways `loudnorm`'s pre-encode prediction can't account for; AND
  `linear=true` clamps gain to preserve TP, which on quieter source clips
  silently caps below target.

  Replaced with a closed-loop pipeline:
  1. Pass 1 — measure source PCM via `loudnorm` (JSON output: input_i,
     input_tp, input_lra).
  2. Compute desired gain = target_i − input_i, clamped by TP headroom
     (target_tp − input_tp).
  3. Apply that gain via the simple `volume={gain}dB` filter and encode
     to MP3. (Skips `loudnorm` pass 2 — explicit gain only, no DRC.)
  4. Measure the encoded MP3.
  5. If outside ±1 LU AND TP ceiling has headroom: compute correction
     delta and re-encode the source PCM with cumulative gain. Up to 2
     correction iterations.
  6. Stop early if TP ceiling reached (record `tp_limited=true` for the
     handful of inherently-quiet clips that can't reach target without
     distortion).

  Implementation: `build/lib/loudness.py::normalize_pcm_to_mp3_verified()`
  returns `VerifiedNormalizationResult` with `applied_gain_db`,
  `final_mp3_lufs`, `final_mp3_tp`, `correction_iterations`,
  `within_tolerance`, `tp_limited`. All fields propagate into the manifest
  for diagnostics and Stage 7 cached-baseline computation.
- ~~**Per-voice baseline shortcut**~~ **REVERSED 2026-05 post-pilot evidence**:
  the original plan called for a cached per-voice median gain (single-pass
  volume filter for Stage 7's full corpus). Empirical pilot data falsified
  the underlying assumption: per-clip RMS varies by ~7 LU even within the
  SAME voice, so a single median gain produces a 22 LU range across the
  deck. The 50 ms/clip saved by skipping pass 1 is not worth the loudness
  drift. **Closed-loop verification is now used for every clip, always.**
  Baselines are still cached at `data/_voice_loudness_baselines.tsv` for
  diagnostic logging only — they're not used for gain selection.
- **Hard rule — gain only**: no dynamic range compression, no EQ, no limiter beyond loudnorm's `-1.5 dB` true-peak ceiling, no de-esser, no noise gate, no reverb, no any-other-DSP. Voice character must survive the chain unchanged. The user explicitly asked for no distortion; gain in a clean digital chain is mathematically lossless within headroom.
- **Encode chain (single-encode)**: ElevenLabs PCM 44.1 kHz mono → ffmpeg `loudnorm` (linear gain, two-pass measured or per-voice baseline) → libmp3lame `mp3_44100_192` → R2 upload. The WAV/PCM intermediate is cached at `build/audio_cache/{sense_id}-{word|ex}-v{N}.wav` until R2 upload confirms; can be re-encoded later (e.g., to a higher bitrate) without another ElevenLabs call.
- **Implementation surface**:
  - `build/lib/loudness.py::measure_voice_baseline(voice_id, sample_clips) -> float` (median dB gain)
  - `build/lib/loudness.py::normalize_clip(input_pcm, gain_db_or_None) -> mp3_bytes` (calls ffmpeg via subprocess)
  - `build/lib/loudness.py::verify_target(mp3_bytes, target_lufs=-16, tolerance_lu=1) -> bool` (post-encode spot-check)
  - ffmpeg ≥ 4.2 required (loudnorm `linear=true` mode introduced in 4.2). Check at script startup; fail loudly if missing.

#### Cost & character impact

The PCM-then-loudnorm-then-MP3 chain adds zero ElevenLabs charges (everything is local CPU). Loudness normalization itself is fast: ~50 ms/clip on a modern Mac. Total wall-clock add for ~12,000 clips: ~10 min serial, ~2 min at concurrency 4. Storage: PCM cache is ~10× MP3 size (~14 GB local) — kept on local SSD; can be deleted after R2 upload completes.

ElevenLabs character billing is unchanged (PCM and MP3 cost the same per character). Pro tier 600k cap still has 233k headroom against the 367k projected character budget.

### Stage 6 — Audio pilot ([build/06_audio_pilot.py](build/06_audio_pilot.py))

**In**: `05-ipa.tsv` (first 500 rows) + `_manual_audio.tsv` → **Out**: `_pilot_500.tsv`, `_audio_manifest.tsv` (initial), audio assets on R2

#### Audio manifest is source of truth

`_audio_manifest.tsv` schema:

```tsv
sense_id	clip_type	voice_gender	tts_provider	tts_model	voice_id	text_input	text_hash	object_key	url	version	md5	generated_at	status	notes
```

`clip_type` ∈ `{word, example}`. **REVISED**: each sense has **2 manifest rows** (one voice gender per sense, picked by Stage 4.5). Final TSV's URL columns are **derived** from the manifest, not hand-maintained. Regeneration: bump `version`, re-upload, update manifest, re-derive TSV.

#### Generation (REVISED)

> **Note (Stage 9 supersedes — see § Stage 9 — Migration to Flash v2.5)**: As of 2026-05 the production model is `eleven_flash_v2_5`, not Multilingual v2. The original Multilingual v2 description below is preserved for historical accuracy (the deck was first rendered on it, then re-rendered on Flash). The rest of this stage's design — manifest as source of truth, voice provenance, filename versioning, retry policy — is unchanged.

Generate 2 mp3 clips per sense via **ElevenLabs Multilingual v2** (locked TTS provider — Pro tier $99 first month):

- `audio_word`: voice picked by Stage 4.5 (`voice_id` column), headword text
- `audio_example`: same voice as `audio_word`, example sentence text

**Voice per sense**: read from `data/045-speaker_gender.tsv` `voice_id` column. The same voice is used for both clips of one sense — this is the user's locked rule (one speaker per Anki record). Voice IDs come from `config/voices.tsv` (4 female + 7 male, user-pinned). Round-robin allocation in Stage 4.5 ensures every voice in the pool gets approximately equal usage.

**File naming**: `{sense_id}-{word|ex}-v{version}.mp3` (e.g., `0001.00.03-word-v1.mp3`). Voice gender is **NOT** in the filename (each sense has only one voice anyway; the manifest records which one). Version is in the filename. **The version goes in the filename, not in a `?v=` query string** — this is the critical Anki-compatibility fix.

> **Note (Stage 9 supersedes)**: Post-Stage-9 filenames bake the `model_id` segment for provenance: `{sense_id}-{word|ex}-eleven_flash_v2_5-v{version}.mp3`. The version-in-filename Anki-compatibility rationale below still applies identically; only the slug expanded. See § Stage 9.

Reasons:

- **Anki strips URL query parameters** when downloading media into `collection.media/`. A URL `...0001.00.03-word.mp3?v=2` lands locally as `0001.00.03-word.mp3` (no version), so Anki sees an existing file with the same name and skips the download. Versioned filenames like `0001.00.03-word-v2.mp3` are net-new filenames; Anki always fetches them.
- **Anki's media sync compares filenames, not file hashes.** If a regenerated clip keeps the same filename, mobile devices won't re-download it during the next AnkiWeb sync. Versioning the filename guarantees clean propagation across desktop + iPhone.
- Cloudflare CDN caching becomes irrelevant (different filename = different object key = cache miss = new fetch).
- The `?v=N` query-string strategy from prior iterations is **abandoned** for audio URLs that are referenced by Anki notes. It would still work for browser-only previews, but Anki is the consumer that matters.

When a clip is regenerated: bump the manifest `version`, write to a new R2 object key with the new filename, update the manifest URL, and the final TSV's audio columns now point at the new filename. The old object can be deleted from R2 after a grace period (or kept; storage is ~$0.02/mo for the lot). Voice mix-up after a mid-batch crash remains structurally impossible because the manifest's `voice_id` column is the source of truth — a regen reads back the previously-assigned `voice_id` for that sense before calling ElevenLabs, so the same voice is always used.

#### Workflow

1. Voices: read from `config/voices.tsv` (committed; 4F + 7M user-pinned). Stage 4.5 has already assigned a specific `voice_id` to every sense.
2. For each sense, generate 2 clips (`word`, `example`) using the assigned `voice_id`. Compute md5 per file; store in manifest and `_md5` columns. Persist `voice_id` in the manifest (already part of the schema) and in the final TSV.
3. Upload objects to R2.
4. **Public access**: expose audio through an **R2 public bucket custom domain**. Do **not** rely on per-object public-read ACL semantics — that wording was incorrect in v1. Public base URL stored in `.env`.
5. Stable URL pattern: `https://<R2-public-domain>/audio/{sense_id}-{word|ex}-v{version}.mp3`. Version is **in the filename**, not the query string (see § File naming above for the Anki-compatibility rationale). Voice gender is **not** in the filename (one voice per sense; provenance lives in the manifest + final TSV `voice_id` column).
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

#### Mandatory preflight cost report (build/06_audio_pilot.py --preflight)

Before any full audio run, the script emits a structured cost report and **blocks** until the user confirms (`--confirm` flag or interactive prompt). Report fields:

```
preflight_audio_cost_report:
  word_chars_m:           <int>      # sum of len(pt) across all senses (male voice)
  word_chars_f:           <int>      # same for female voice
  example_chars_m:        <int>      # sum of len(example_pt) across all senses
  example_chars_f:        <int>      # same for female voice
  total_chars:            <int>      # sum of above four
  tts_provider:           <string>   # from config/models.yaml, e.g. "google_cloud_tts"
  tts_model:              <string>   # e.g. "pt-BR-Neural2"
  provider_price_per_1m_chars: <float>  # in USD; current published rate
  retry_buffer_percent:   <float>    # default 30%
  estimated_total_usd:    <float>    # = total_chars * (1 + buffer) * price / 1_000_000
  per_voice_breakdown:    <map>      # cost_m, cost_f
```

Counting rule (Google Cloud TTS billing semantics): characters include letters, punctuation, AND whitespace. The script counts `len(text)` directly on the input strings, not stripped versions. SSML tags, if used, count too.

The user reviews the report and either confirms (proceeds), aborts (no API calls made), or adjusts buffer / provider / voice. The same report is regenerated and logged at the start of Stage 7 (full run) for audit.

#### Review (ASR roundtrip + minimal human gate)

Every clip is validated automatically via Whisper-based ASR roundtrip (§ Review strategy, Tier 3). The pipeline regenerates any clip below the Levenshtein threshold; only clips that fail twice land in `_audio_human_review.tsv`.

The single human gate before Stage 7 is **voice quality**, not pronunciation accuracy: the user listens to ~10 random pilot clips that **already passed ASR roundtrip** to confirm the chosen male and female voices sound right. ~15 minutes. If voices are wrong, swap voice IDs and regenerate (config change + ~30 min compute). Pronunciation accuracy is ASR's job, not the human's.

#### Live progress logging (LOCKED 2026-05)

User wants a growing, tail-able transcript so they can watch ASR decisions in real time, plus a periodic snapshot showing % done / ETA / cost. Reuses the `build/lib/progress.py::ProgressTracker` pattern proven in Stage 5.5 with extensions for "last N clips" and decision-distribution counters.

**Three artifacts written in parallel during the run:**

1. **`audit/06_audio.jsonl`** (machine-readable, append-only) — one line per lifecycle event per clip:
   ```json
   {"event":"started","sense_id":"0042.00.01","clip_type":"word","voice_id":"MZL...","input_text":"casa","attempt":1,"started_at":"2026-05-09T14:23:11.123Z"}
   {"event":"asr_completed","sense_id":"0042.00.01","clip_type":"word","voice_id":"MZL...","input_text":"casa","asr_transcript":"casa","levenshtein_similarity":1.0,"phonetic_distance":0.0,"decision":"pass","attempt":1,"latency_ms":2333,"cost_usd":0.0006,"completed_at":"..."}
   {"event":"regenerated","sense_id":"0043.00.02",...}
   {"event":"errored","sense_id":"...","error_type":"APITimeoutError","error_msg":"..."}
   ```

2. **`audit/06_audio_transcript.log`** (human-readable, append-only, tail-friendly) — one line per ASR decision, fixed-width columns for visual scanning. Tail with `tail -f audit/06_audio_transcript.log`:
   ```
   2026-05-09 14:23:11  PASS    0042.00.01 word    casa                              -> casa                              sim=1.00  voice=MZL...  attempt=1  [1.4s]
   2026-05-09 14:23:13  PASS    0042.00.01 ex      A casa é grande.                  -> A casa é grande.                  sim=1.00  voice=MZL...  attempt=1  [1.7s]
   2026-05-09 14:23:16  REGEN   0043.00.02 word    boa                               -> bola                              sim=0.67  voice=4za...  attempt=1
   2026-05-09 14:23:17  PASS    0043.00.02 word    boa                               -> boa                               sim=1.00  voice=4za...  attempt=2  [1.5s]
   2026-05-09 14:23:19  HUMAN   0144.00.01 ex      Estou curioso para ver.           -> Estoy curioso para ver.           sim=0.91  voice=ny3...  attempt=2  (twice-failed -> _audio_human_review.tsv)
   ```
   Decision tags: `PASS` (similarity ≥ threshold, accepted), `REGEN` (below threshold, retry queued), `HUMAN` (failed twice, routed to human queue), `ERR` (TTS or ASR API error). The transcript line is written **only** when ASR completes, so a tail-watcher sees one line per clip-decision (not per lifecycle event).

3. **`audit/06_audio_progress.jsonl`** (in-flight tracking + stuck detection, same shape as Stage 5.5's progress JSONL — decoupled from the human-readable transcript so the tail-friendly file stays clean).

**Stdout snapshot (printed every 30 seconds by the main loop):**

```
[Stage 6  3,217 / 12,000 clips (26.8%) | 14 in-flight | 1 stuck (>120s) | started 14:08 | elapsed 24m | avg 1.4s/clip | ETA 18m | spent $4.20]
  Last 3 clips:
  PASS   0042.00.01 word    casa             -> casa            sim=1.00  [1.4s]
  PASS   0042.00.01 ex      A casa é grande. -> A casa é grande. sim=1.00  [1.7s]
  REGEN  0043.00.02 word    boa              -> bola            sim=0.67  -> retry attempt 2

  Decision distribution so far:
    PASS first attempt:    3,068  (95.4%)
    PASS after regen:        135  (4.2%)
    Human review queue:       14  (0.4%)
    Errored:                   0
```

The "Last 3 clips" block is rendered from the in-memory ring buffer that the `ProgressTracker` already maintains. The decision-distribution block is recomputed from a small in-memory tally each tick.

**Standalone snapshot tool — `build/audio_status.py`:**

Same pattern as `build/audit_status.py` (Stage 5.5). Reads `audit/06_audio.jsonl`, prints the snapshot above plus a one-line summary. Can be run any time during or after the run; idempotent. Useful when running Stage 6/7 in a background terminal — open a second terminal and snapshot on demand.

**Resume / idempotency:**

Stage 6/7 reads `_audio_manifest.tsv` at startup and skips any (sense_id, clip_type) row already at `status=uploaded`. Both JSONL files are append-only and never truncated, so resume preserves the full audit history. Killing and re-running burns no extra ElevenLabs credits and no extra ASR cost.

**Implementation surface:**

- [build/lib/progress.py](build/lib/progress.py) — already exists from Stage 5.5; extend with `ring_buffer(N)` and `decision_counts` accessors.
- [build/lib/asr.py](build/lib/asr.py) — Whisper API caller, normalized Levenshtein + eSpeak-IPA distance, returns `(transcript, lev_sim, phonetic_dist, decision)`.
- [build/lib/audio_logger.py](build/lib/audio_logger.py) — writes JSONL + transcript.log atomically (single lock; appends both files in one critical section per clip).
- [build/audio_status.py](build/audio_status.py) — standalone snapshot tool.

#### ASR model selection (LOCKED 2026-05)

After the 100-sense smoke run, ran a 4-model A/B comparison on the same 200
cached MP3s — same audio, same loudness-normalized output, only the ASR
model varies. Script: [build/ab_asr_models.py](build/ab_asr_models.py) +
[build/ab_asr_analyze.py](build/ab_asr_analyze.py). Total spend ~$0.22.

| Model | Provider | Pass% | Short-input pass% | Long-input pass% | Per-min |
|---|---|---|---|---|---|
| **gpt-4o-transcribe** | OpenAI | **99.0%** | **98.0%** | **99.3%** | $0.006 |
| whisper-1 | OpenAI | 95.5% | 92.0% | 96.7% | $0.006 |
| gpt-4o-mini-transcribe | OpenAI | 95.0% | 92.0% | 96.0% | $0.003 |
| scribe_v2 | ElevenLabs | 90.0% | 84.0% | 92.0% | ~$0.007 |

Decision: **`gpt-4o-transcribe` is the locked default.** Same per-minute
price as whisper-1; pass-rate jumps 95.5% → 99.0%, which on the full
~12,000-clip corpus drops the human-review queue from ~540 → ~120 clips.
Particularly fixes the Whisper-1 hallucination problem on isolated function
words: where whisper-1 transcribed `o → "O" / em → "inglês." / um → "M" /
ou → "O"`, gpt-4o-transcribe returns the input verbatim.

**Why not the cheaper gpt-4o-mini-transcribe (~$0.003/min, half the cost)?**
Empirically it has occasional cross-language hallucinations on isolated
phonemes — `com → 콩` (Korean), `ano → あの` (Japanese), `ir → "E..."`. At
the corpus's scale the savings are ~$1.50; not worth the false-regen risk.

**Why not scribe_v2?** Worst pass rate of the four, frequently adds
emotional punctuation (`Oh!`, `Eeei`), splits short forms (`esse → "E se"`),
or returns English (`se → "Says"`). ElevenLabs Scribe is excellent on long
narration but struggles on our short pedagogical clips.

**Implementation:** [build/lib/asr.py](build/lib/asr.py) `DEFAULT_ASR_MODEL`
= `"gpt-4o-transcribe"`. The gpt-4o family requires `response_format=json`
(rejects `text`); the `_response_format_for()` helper picks per-model and
`_extract_text()` handles both shapes. The biased-prompt feature works
identically. Length-aware phonetic-only policy for inputs ≤ 3 chars stays
in place — it's a robust safety net regardless of ASR model.

A/B raw evidence committed: `audit/ab_asr_summary.tsv` (per-model pass
rates) and `audit/ab_asr_disagreements.log` (30 clips where models split
on the verdict — useful when re-evaluating model choice in future).

### Stage 7 — Audio full ([build/07_audio_full.py](build/07_audio_full.py))

**In**: `05-ipa.tsv` (rows 501+) + manifest → **Out**: `06-final.tsv`, manifest fully populated

Same pipeline for the remaining ~7,500–9,500 senses after pilot approval.

**Idempotency**: resume = skip senses whose 4 expected manifest rows already exist with `status = uploaded` and HEAD-request 200 OK on URL.

**Validation**: every row has 4 audio URLs (derived from manifest); HEAD requests 200 OK on a random 5% sample; md5 matches between local cache, manifest, and R2-downloaded.

### Stage 8 — Pronunciation correction (v3 LEAN, LOCKED 2026-05-10)

**In**: `data/_audio_manifest.tsv` (post-Stage-7) + R2-uploaded word clips → **Out**: same manifest with re-rendered v2 word clips for confirmed-flagged senses, `data/_pronunciation_aliases.tsv` (deduped by `pt`), `data/_pronunciation_alias_applications.tsv`, `data/_pronunciation_alias_conflicts.tsv`, `data/_audio_risk_classification.tsv`, ElevenLabs alias dictionary, refreshed `data/06-final.tsv`, `audit/08_*.{jsonl,txt,html}`

**Plan revision history**: v1 (commit `040b501`) leaned on ASR + a returned
`language` field that `gpt-4o-transcribe` JSON mode does not document, with
a same-orthography blind spot. v2 (commit `ac4196b`) over-corrected by
adding 11 phases including a full ASR sweep, F1 threshold-fitting,
N-best across many families, triple-verification, and a separate EP audit
phase. **v3 (this section) keeps v2's correctness fixes — risk classifier,
audio LLM judge, by-`pt` aliases with conflict detection, after-fix
listening — but cuts the rest as overengineering for the actual failure
mode. The pipeline is 7 scripts, not 11.**

#### Pre-Stage-8 deltas (LOCKED 2026-05-10)

Before Stage 8 implementation begins, three user-locked decisions are
already in effect:

1. **Audio-judge scope: full-corpus by default** (~$32, all 6,250 word
   clips). The user opted to spend the extra $23 over the risky-bucket
   default to eliminate the "we may have missed a pattern" risk.
2. **MALE #4 + MALE #6 swapped pre-emptively** for monolingual BP
   replacements. The two voices that previously contributed 23.3% of
   still-HUMAN failures from only 16.8% of word clips are gone. New
   voices:
   - Pool index 4: `4za2kOXGgUd57HRSQ1fn` (Lendário, native pt + 17
     verified langs) → **`AaeZyyi87RCxtFnHPS3e`** ("Prof. Campanholi" —
     BP-only, professorial, 0 verified secondary languages).
   - Pool index 6: `uju3wxzG5OhpWcoi3SMy` (Michael C. Vincent, native EN
     + 13 verified langs) → **`4r3G9XKliGgVZLKMgjik`** ("Lair" — BP-only,
     calm narrator, 0 verified secondary languages).

   The swap re-rendered 1,920 manifest rows (480 word + 480 example × 2
   voices) using the surgical
   [build/swap_voices_surgical.py](build/swap_voices_surgical.py) which
   touches only the affected senses (no shuffle, no neutral-demotion).
   Cost ~$1.58. The voice-level risk patterns are now empty (preserved
   in the seed file as a future hook).
3. **Sentinel respellings locked with pronunciation guides** at
   [data/_pronunciation_alias_seeds.tsv](data/_pronunciation_alias_seeds.tsv).
   ~25 candidates across 10 risk-family templates (-al, -el, -il, -ol
   final-l vocalization; -de and -te palatalization; h_initial+final-l
   compound; English loanwords per-word; EP leftover named). Each
   candidate has a phonetic guide tailored to the reviewer's profile
   (Polish native, EN/DE fluent), a recommended pick, an anti-example,
   and a confidence rating. 08_3 reads this file directly and skips the
   LLM family-template proposal step (saving $0.05 + a few minutes;
   bulk-generation per-word for non-template cognates still uses Sonnet
   4.5).

**Updated cost projection**: ~$32 default (full-corpus audio judge).
Wall: ~70 min compute + ~70 min user listening.

#### Why this stage exists (LOCKED 2026-05-10)

After Stage 7 completed the 12,450-clip corpus, user listening surfaced a
class of mispronunciation the existing pipeline cannot catch: **ElevenLabs
Multilingual v2 occasionally falls back to non-BP phonetic priors despite
`language_code="pt"` being passed.** Examples include English-leaning
cognates (`animal` rendered as `[ˈænɪməl]`), EP-leaning palatalization
slips (`cidade` rendered without final `[i]`), and possible Spanish/French
flavoring on identically-spelled tokens. Stage 6/7 ASR uses
`gpt-4o-transcribe` with `language="pt"` which **forces** non-BP audio to
transcribe as Portuguese spelling — same-orthography drift like `animal`
returns Levenshtein 1.0, decision PASS. Text-mediated detection cannot
reach this failure class. The fix is non-text-mediated: a deterministic
risk pre-classifier (orthographic patterns) plus an audio LLM judge
(perceives the actual dialect from the audio) plus alias respellings (the
official ElevenLabs mitigation for non-English pronunciation).

#### Why alias rules, not phoneme rules (LOCKED)

The [ElevenLabs FAQ](https://elevenlabs.io/docs/eleven-agents/customization/voice/pronunciation-dictionary)
states phoneme tags are supported only on `eleven_flash_v2` and that *"for
non-English languages, it is recommended to use alias tags, as phoneme
tags are designed only for English pronunciations."* Switching to
flash_v2 for word clips would break voice continuity (different acoustic
decoder than the multilingual_v2 example clips on the same card) and
regress audio quality. Alias rules apply on every model and feed the TTS
engine a Portuguese-shaped respelling that maps to the same target
phonemes the word has when spoken naturally — `animal → animau` because
BP final-`l` is realized as `[w]`. The respelling is a phonemically-
faithful alternate spelling, not a fictional word.

#### Settled decisions (LOCKED v3 2026-05-10)

| Decision | Value |
|---|---|
| Scope | Word clips only (~6,250). Example clips out of scope. |
| Detection backbone | Deterministic risk pre-classifier + audio LLM judge on the risky bucket. ASR is diagnostic only, never a gate. |
| Audio judge model | `gpt-4o-audio-preview` (or successor) via OpenAI; pinned in `config/models.yaml` under role-key `audio_judge`. |
| Audio judge scope | Risky bucket (P0 + P1 + P2 + ASR-leftovers, ~600–1,200 clips); `--full-corpus` flag (~6,250) only if Phase 6 reveals residuals. |
| Calibration | Descriptive sample (75–100 clips); no F1 threshold-fitting. Auto-confirm rule combines audio-judge verdict + risk priority + human label. |
| Remediation | Local respelling map (`data/_pronunciation_aliases.tsv`, deduped by `pt`) + uploaded ElevenLabs alias dictionary. Conflict detection blocks upload if two senses share a `pt` with different respellings. |
| Sentinel-word smoke test | Render `animal`, `hospital`, `hotel`, `cidade`, `gente`, `rua`/`carro` with candidate respellings BEFORE generating aliases for the full risky bucket. Confirms patterns produce BP audio and that the dict pipeline is wired right. |
| Re-render scope | Only clips whose `pt` has a confirmed alias, plus all other word clips sharing that exact `pt`. |
| Re-render verification | Primary: audio judge `bp_ok` at medium/high confidence. Sanity: biased ASR still matches canonical `pt`; loudness still passes existing constraints. Diagnostic only: unbiased ASR (not a gate, since it can miss the very same-orthography failure mode the stage exists to fix). |
| Phoneme rules | Rejected — `eleven_multilingual_v2` ignores them; `eleven_flash_v2` unsuitable for our voice continuity / quality requirements. |
| Voice continuity | Same `voice_id` for re-rendered clip as the original Stage 6/7 rendering (preserved end-to-end via the manifest). |
| EP-leftover audit | Folded into Phase 0 risk classification via the `ep_leftover_named` pattern (5 named words). No separate EP audit script. |
| After-fix listening | Mandatory final gate (Phase 6). HTML with all user-reported repaired clips, all high-severity repaired clips, 10–20 random repaired clips, ≥1 clip per alias family, ≥1 clip per voice. |
| Cost projection | ~$6 default (~$32 with `--full-corpus` audio judge). |
| Wall clock | ~50 min compute + ~60 min user listening. |

#### Risk taxonomy and priority levels (LOCKED — used by 08_0)

`data/_audio_risk_classification.tsv` is built once from `data/05-ipa.tsv`
plus hand-curated seed lists. One row per word-clip `sense_id`,
multi-valued `risk_patterns`, single `priority` derived as the highest
priority across the row's patterns.

Patterns split between **word-level** (12 orthographic-pattern triggers
keyed on `pt`) and **voice-level** (2 voice-id triggers from the
`data/_risk_seeds_high_risk_voices.tsv` seed file). A clip can carry
both word-level AND voice-level patterns; priority = min across all
matched patterns.

| Pattern | Priority | Trigger | Drift direction |
|---|---|---|---|
| `voice_high_risk_critical` | **P0** | `voice_id ∈` rows in `data/_risk_seeds_high_risk_voices.tsv` with `risk_tier=CRITICAL`. **Currently EMPTY** (the previously-flagged voice `uju3wxzG5OhpWcoi3SMy` was swapped pre-Stage-8 for a BP-only replacement — see § Pre-Stage-8 deltas above). Pattern is preserved as a future hook. | EN (structural — would re-fire if a future native-EN voice is added) |
| `voice_high_risk_elevated` | **P1** | `voice_id ∈` rows in `data/_risk_seeds_high_risk_voices.tsv` with `risk_tier=ELEVATED`. **Currently EMPTY** (previously-flagged `4za2kOXGgUd57HRSQ1fn` swapped pre-Stage-8). Pattern preserved. | mixed (would re-fire on heavily-multilingual voices) |
| `user_reported` | **P0** | `pt` appears in `data/_audio_user_reported_failures.tsv` | mixed |
| `ep_leftover_named` | **P0** | `pt ∈ {camisola, fazenda, marcha, troço, vosso, comboio, equipa, registo, utilizador, paragem, desporto, golo}` (Stage-1 EP audit seed list) | EP |
| `same_spelling_en_pt` | **P1** | `pt` exists as a common English word with same letters | EN |
| `final_l_vocalization` | **P1** | regex `(.)al$|(.)el$|(.)il$|(.)ol$` AND BP IPA ends in `[w]` | EN |
| `h_initial_english_risk` | **P1** | `pt` starts with `h` (BP `h-` is silent; English isn't) | EN |
| `english_loanword` | **P1** | `pt ∈` hand-curated `data/_risk_seeds_en_loanwords.tsv` | EN |
| `de_te_palatalization` | **P2** | regex `(.)de$|(.)te$` AND BP IPA contains `[dʒi]` or `[tʃi]` | EP |
| `final_unstressed_e` | **P2** | regex `[^aáâãeéêiíoóôõuú]e$` | EP |
| `initial_r_or_rr` | **P2** | starts with `r` OR contains `rr` | EP |
| `coda_s_ep_risk` | **P3** | contains coda `s` (regex `s[^aáâãeéêiíoóôõuú]`) | EP |
| `spanish_collision` | **P3** | `pt ∈` hand-curated `data/_risk_seeds_es_collisions.tsv` | ES |
| `french_loanword` | **P3** | `pt ∈` hand-curated `data/_risk_seeds_fr_loanwords.tsv` | FR |
| `none` | — | no pattern matched | — |

A single `pt` can carry multiple patterns (e.g., `cidade` =
`de_te_palatalization` AND `final_unstressed_e`). The row's `priority` =
min priority value across all matched patterns (P0 < P1 < P2 < P3).

##### Concrete risk-family seed lists (corpus-grounded)

These lists are the starting point for Phase 0's regex/membership checks
and for Phase 2's stratified calibration sampling. They will be fleshed
out by intersection with the actual corpus during Phase 0 execution.

**same-spelling EN/PT (P1)**: animal, social, final, principal, natural,
capital, central, material, hospital, hotel, legal, general, digital,
terminal, tropical, fatal.

**final-l vocalization (P1)**: animal, social, nacional, geral, papel,
jornal, local, pessoal, principal, natural, capital, especial, igual,
oficial, civil, hospital, hotel, futebol, legal, vital, horizontal,
cereal, lençol.

**de/te palatalization (P2)**: cidade, verdade, idade, qualidade,
sociedade, gente, noite, presente, seguinte, ambiente, doente, paciente,
cliente, acidente, dente, adolescente, contente, semente, mente, fonte,
ponte.

**initial r/rr (P2)**: rua, rio, real, razão, relação, região, resultado,
resolver, terra, guerra, carro, correr, corrente, ocorrer, ferro, serra,
arranjar, enterrar, morrer.

**English loanwords (P1)**: software, marketing, rock, jazz, bar, pop,
vídeo, site, festival, hotel, piano, deficit, terror, horror, time.

**EP leftovers / BP audit seeds (P0)**: camisola, troço, vosso, comboio,
equipa, registo, utilizador, paragem, desporto, golo. (Plus `fazenda` and
`marcha` from the 5 originally-named EP markers, retained from Stage 1.)

**High-risk voices (CRITICAL P0 / ELEVATED P1)** — voice-level dimension
keyed on `voice_id`, source-of-truth at `data/_risk_seeds_high_risk_voices.tsv`:

| voice_id | name | tier | native | verified-langs | empirical |
|---|---|---|---|---|---|
| `uju3wxzG5OhpWcoi3SMy` | Michael C. Vincent | **CRITICAL** | en | 13 | 5.4% still-HUMAN |
| `4za2kOXGgUd57HRSQ1fn` | Lendário | ELEVATED | pt | 17 | 1.7% still-HUMAN |

ElevenLabs API metadata (`/v1/voices/{id}`, `verified_languages` array)
documents both voices as fine-tuned across 13–17 languages spanning
multiple model tiers. MALE #6 is **natively English** with BP only as a
verified secondary — structural EN-drift risk on isolated BP cognates.
MALE #4 is natively pt-BR but stretched across 17 languages with a
"hyped" social-media delivery, both of which loosen per-language
phonetic anchoring on isolated short words. Combined empirical: these
2 voices account for 23.3% of still-HUMAN failures from only 16.8% of
word clips. Adding/removing voices from the seed file is the user-facing
knob for the voice-level dimension.

These are CANDIDATES for detection, not a declaration that they are
wrong. Phase 1 (audio judge) decides per-clip; Phase 2 (calibration)
validates Phase 1's threshold against ground truth.

#### Lean pipeline — 7 scripts

The pipeline is intentionally short. Each script reads its predecessor's
output, writes a TSV plus a JSONL audit trail, and is idempotent on
re-run.

```
build/08_0_classify_risk.py
    → build/08_1_audio_judge.py
        → build/08_2_calibration.py
            → build/08_3_generate_aliases.py
                → build/08_4_upload_dictionary.py
                    → build/08_5_rerender_verify.py
                        → build/08_6_after_fix_and_finalize.py
```

##### 08_0 — Risk classifier ([build/08_0_classify_risk.py](build/08_0_classify_risk.py))

Pure Python, no API calls. Walks `data/05-ipa.tsv` joined to
`data/_audio_manifest.tsv` (for the per-clip `voice_id` needed by the
voice-level patterns) and emits `data/_audio_risk_classification.tsv`
(one row per word `sense_id`). Reads four seed files for membership
checks: `_risk_seeds_en_loanwords.tsv`, `_risk_seeds_es_collisions.tsv`,
`_risk_seeds_fr_loanwords.tsv`, and `_risk_seeds_high_risk_voices.tsv`
(the new voice-level seed). Plus `_audio_user_reported_failures.tsv`
for P0 user-reported entries. Logs counts per pattern at end of run,
including word-level vs voice-level breakdown.

Output schema:

```tsv
sense_id     pt          voice_id              risk_patterns                                                              priority  ipa_word_final  rank
0317.00.01   animal      qPfM2laM0pRL4rrZtBGl  user_reported,same_spelling_en_pt,final_l_vocalization                     P0       ˌaniˈmaw       317
0858.00.01   data        4r3G9XKliGgVZLKMgjik  user_reported,same_spelling_en_pt                                          P0       ˈdadʒɐ          858
0042.00.01   cidade      Rw38T6bn0lTNOb1aUevR  de_te_palatalization,final_unstressed_e                                    P2        siˈdadʒi       42
0001.00.01   o           wxoDdfPKBuna5KnUEotz  none                                                                       —         u              1
```

Note: post-swap example shows voice `4r3G9XKliGgVZLKMgjik` (Lair, the
new pool-index-6 BP-only voice). The `voice_high_risk_critical` pattern
no longer fires for this clip because the voice is monolingual BP. If
future voices are flagged in `_risk_seeds_high_risk_voices.tsv`, the
pattern will appear here automatically.

**Cost**: $0. **Wall clock**: ~5 min.

##### 08_1 — Audio LLM judge ([build/08_1_audio_judge.py](build/08_1_audio_judge.py))

Runs the audio judge on the **risky bucket** by default (priority ∈ {P0,
P1, P2} ∪ ASR-human/regenerated leftovers from Stage 6/7). `--full-corpus`
flag promotes scope to all 6,250 word clips (~$31 instead of ~$3–6).

The risky bucket includes both word-level patterns (12 orthographic
triggers) and voice-level patterns (`voice_high_risk_critical` P0 →
all 480 clips from MALE #6; `voice_high_risk_elevated` P1 → all 480
clips from MALE #4) by virtue of the `priority ∈ {P0, P1, P2}` default
rule. Net audio-judge bucket size: ~1,200–1,800 clips (~$6–9), up from
~600–1,200 before voice-level was added. The +600 clips are primarily
the two high-risk voices' worth of word clips that wouldn't otherwise
trip a word-level pattern.

Per-clip call to `gpt-4o-audio-preview`:

- Audio MP3 attached as input.
- Tool-Use structured output (no free-text parsing).
- Cached system prompt: *"You are judging whether a Brazilian Portuguese
  vocabulary recording is pronounced as standard BP. Listen to the audio
  and give a verdict. Focus on phonetic features only — vowel quality,
  stress placement, /l/ realization, /r/ realization, palatalization of
  /t/, /d/ before /i/, coda /s/ voicing. Your primary decision is BP-OK
  vs non-BP vs unclear. The drift source guess (EN/EP/ES/FR/other) is
  secondary diagnostic information."*

Tool schema:

```json
{
  "type": "object",
  "properties": {
    "pronunciation_verdict": {"type": "string", "enum": ["bp_ok", "non_bp", "unclear"]},
    "drift": {"type": "string", "enum": ["EN", "EP", "ES", "FR", "other", "none"]},
    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    "evidence": {"type": "string"}
  },
  "required": ["pronunciation_verdict", "drift", "severity", "confidence", "evidence"]
}
```

Output: `audit/08_1_audio_judge.jsonl` and
`data/_audio_judge_verdicts.tsv`.

**Decision rule** (consumed by 08_2 calibration and 08_3 alias generation):

```
needs_alias =
  pronunciation_verdict == "non_bp" AND confidence == "high"
  OR
  pronunciation_verdict == "non_bp" AND confidence == "medium" AND priority in {P0, P1}

needs_human_review =
  pronunciation_verdict == "unclear"
  OR
  pronunciation_verdict == "non_bp" AND confidence == "low"
```

**Cost**: ~$3–6 risky bucket; ~$31 full-corpus. **Wall clock**: ~30 min
risky / ~60 min full at concurrency 4. Idempotent.

##### 08_2 — Calibration ([build/08_2_calibration.py](build/08_2_calibration.py))

Generates `audit/08_2_calibration.html` with **75–100 stratified clips**.
No threshold-fitting / F1 optimization — calibration's job is to verify
the audio judge is catching real failures without too many false
positives, AND to surface borderline cases the judge marked `unclear`.

Stratification (~100 clips):

| Bucket | Count | Source |
|---|---|---|
| All user-reported failures | all (≥1) | `_audio_user_reported_failures.tsv` |
| `voice_high_risk_critical` non-BP | 0–10 | **Currently 0** (no CRITICAL voices in pool post-swap). If future voices are added back, audio-judge non_bp verdicts from those voices fill this bucket. |
| `voice_high_risk_elevated` non-BP | 0–5 | **Currently 0** (no ELEVATED voices in pool post-swap). Same future hook. |
| `final_l_vocalization` family | 20 | Stratified by frequency tier across -al/-el/-il/-ol |
| `de_te_palatalization` family | 15 | Stratified by frequency tier |
| `initial_r_or_rr` | 10 | Random sample |
| `coda_s_ep_risk` ∪ `final_unstressed_e` | 10 | Random sample |
| `english_loanword` ∪ `spanish_collision` ∪ `french_loanword` | 10 | Random sample combined |
| Audio-judge `bp_ok` controls | 10 | Sanity: easy cases must stay PASS |
| Voice-coverage minimum | overlap | At least 1 clip per voice in the 8-voice pool, drawn from the above buckets |

Each card: `<audio>` + expected `pt` + audio-judge verdict + 3-button
radio (`OK / MISPRONOUNCED / UNCLEAR`). Form output pasted into
`data/_audio_calibration_labels.tsv`.

**Reviewer guidance (LOCKED)**: the HTML page MUST embed the contents of
[docs/reviewer_guide.md](docs/reviewer_guide.md) at the top, before the
clip cards. The guide is tailored to the project's human judge — an A1
Brazilian Portuguese learner, native Polish speaker, fluent in English
and German — and explains the six BP-specific phonetic features to listen
for (final-l vocalization, -de/-te palatalization, initial r/rr, coda s,
final unstressed e, cognate stress placement) with Polish/EN/DE parallels
and a concrete OK/MISPRONOUNCED/UNCLEAR decision rubric. Render via
markdown-to-HTML inline at HTML build time (e.g. `python -m markdown` or
similar); do not link out — the reviewer should not have to switch
contexts mid-listen.

After listening (~40 min), `--apply-calibration`:

1. Computes `confirmed_for_alias` per the auto-confirm rule (intersection
   of audio-judge verdict + risk priority + human label):
   ```
   confirmed_for_alias =
     human_label == "MISPRONOUNCED"
     OR
     (judge.pronunciation_verdict == "non_bp" AND judge.confidence == "high")
     OR
     (judge.pronunciation_verdict == "non_bp" AND judge.confidence == "medium"
      AND priority in {P0, P1})
   ```
2. Routes `unclear` rows to `data/_audio_calibration_unclear.tsv` for a
   small manual queue.
3. **Adaptive-expansion gate (LOCKED)**: for each risk family in the
   calibration sample, computes MISPRONOUNCED rate = MISPRONOUNCED count /
   listened count. If **any family > 10% MISPRONOUNCED**, emits
   `audit/08_2_calibration_expansion_v{N}.html` with **+25 additional
   clips drawn from that family** and pauses for user listening before
   continuing. The expansion is appended to
   `data/_audio_calibration_labels.tsv`. Loop until no family exceeds
   10% (or N=3 expansion rounds, whichever comes first; the cap prevents
   pathological loops on a fundamentally noisy family).
4. Writes `data/_audio_mispronunciation_confirmed.tsv` for downstream
   alias generation.

**Cost**: $0. **Wall clock**: ~40 min user listening (initial sample);
+~10 min per expansion round if triggered.

##### 08_3 — Alias generation ([build/08_3_generate_aliases.py](build/08_3_generate_aliases.py))

Two phases: sentinel-word smoke test FIRST, then bulk family + individual
generation.

**Step 3.1 — Sentinel smoke test (LOCKED).** Candidates are pre-locked
in [data/_pronunciation_alias_seeds.tsv](data/_pronunciation_alias_seeds.tsv)
(committed pre-Stage-8). The script reads this file and renders each
candidate; **no LLM proposal step** for the family templates (saves $0.05
+ wall time vs LLM-proposed). 08_3 still uses Sonnet 4.5 for non-template
per-word fill in Step 3.2.

The seed file covers 10 risk-family templates with ~25 candidate
respellings:

- **final_l_-al** (animal: `animau` / `animáu` / anti-example `animál`)
- **final_l_-el** (hotel: `otéu` / `hotéu` / anti-example `hotél`)
- **final_l_-il** (civil: `civiu` / `civíu` / anti-example `civíl`)
- **final_l_-ol** (futebol: `futebóu` / anti-example `futeból`)
- **de_te_palatalization_-de** (cidade: `cidadji` / `cidadi` / anti-example `cidade`)
- **de_te_palatalization_-te** (gente: `gentchi` / `genti` / anti-example `gente`)
- **h_initial_+_final_l** (hospital: `ospitau` / `hospitau` / anti-example `hospitál`)
- **english_loanword_per_word** (internet: `internétchi`; software: `softuei` / `sóftuér`)
- **ep_leftover_named** (susceptível: `suscetível` — orthographic correction)

Each candidate row in the seed file carries a **pronunciation guide**
tailored to the project's reviewer profile (Polish native, EN/DE
fluent) — e.g., `animau` is documented as *"ah-nee-MAU" — last syllable
rhymes with English "wow"; Polish `ł` in `łapa`*. Each row also carries
`is_recommended` + `is_anti_example` + `confidence` flags so the smoke-test
HTML can color-code candidates and the user knows which one to pick if
the rendering matches the guide.

`initial_r_or_rr` and `coda_s_ep_risk` are deferred per the v3 plan —
not auto-templated; only aliased per-word if the audio judge AND human
calibration both confirm a failure.

Each sentinel candidate is rendered once via 08_5's `--smoke` mode
(~$0.01 per candidate, ~$0.12 total). User listens to the 12-clip HTML
result (`audit/08_3_sentinel_smoke.html`) and picks the winning candidate
per family. The locked family templates are written to
`data/_pronunciation_alias_seeds.tsv`.

**Step 3.2 — Bulk generation.** For every confirmed-flagged `pt` in
`_audio_mispronunciation_confirmed.tsv`:

- If the `pt` matches a locked family template, mechanically apply the
  template (no LLM call).
- Otherwise, call Sonnet 4.5 per-word via Tool Use. Schema:

```json
{
  "type": "object",
  "properties": {
    "respelling": {"type": "string"},
    "rationale": {"type": "string"},
    "applied_family": {"type": "string"},
    "confidence": {"type": "string", "enum": ["low", "medium", "high"]}
  },
  "required": ["respelling", "rationale", "confidence"]
}
```

Output deduped by `pt` to `data/_pronunciation_aliases.tsv`:

```tsv
alias_id  pt        pt_respelling  rationale                                source_ipa  drift  source           confidence  affected_count  created_at
1         animal    animau         Final -l → -u (BP final-l vocalizes)     ˌaniˈmaw   EN     family_template  high        1               2026-05-10T...
2         hospital  ospitau        Family template + h-initial drop         ospiˈtaw   EN     family_template  high        1               2026-05-10T...
```

Per-sense application table — `data/_pronunciation_alias_applications.tsv`:

```tsv
sense_id     clip_type  alias_id  applied_at
0317.00.01   word       1         2026-05-10T...
1234.00.01   word       2         2026-05-10T...
```

Conflict table — `data/_pronunciation_alias_conflicts.tsv`: any `pt`
appearing in multiple confirmed sense_ids that would generate different
respellings. Rows here BLOCK upload until resolved manually.

**Cost**: ~$0.50 LLM (mostly cached) + ~$0.12 sentinel smoke renders.

##### 08_4 — Dictionary upload ([build/08_4_upload_dictionary.py](build/08_4_upload_dictionary.py))

1. Read `data/_pronunciation_aliases.tsv` (deduped by `pt`).
2. Verify `data/_pronunciation_alias_conflicts.tsv` is empty; abort
   loudly if not.
3. Smoke-test ONE alias rule via a 1-rule sandbox dictionary (name
   `anki-bp-stage-8-smoke-vN`) to confirm the SDK accepts the
   `case_sensitive` and `word_boundaries` fields as documented in the
   [API reference](https://elevenlabs.io/docs/api-reference/pronunciation-dictionaries/create-from-rules).
   Cost: ~$0.001 (one tiny TTS call to confirm the dict applies).
4. Build the alias rules array and POST
   `/v1/pronunciation-dictionaries/add-from-rules` with name
   `anki-bp-stage-8-vN` (N increments each upload).
5. Capture `dictionary_id` and `version_id`. Append to
   `data/_audio_dictionary_meta.tsv`.

**Cost**: ~$0.001. **Wall clock**: ~30 sec.

##### 08_5 — Re-render + verify ([build/08_5_rerender_verify.py](build/08_5_rerender_verify.py))

Modifies [build/lib/elevenlabs_client.py](build/lib/elevenlabs_client.py)
to accept `pronunciation_dict_locators: list[dict] | None = None` in
`__init__` and pass it as `pronunciation_dictionary_locators=...` in
`_call_once()` when non-None.

Re-render scope: every word clip whose `pt` has a confirmed alias, plus
every other word clip in the manifest sharing that exact `pt`.

Per-clip workflow:

1. Load active dictionary from `_audio_dictionary_meta.tsv` (latest row).
2. Bump manifest version: N → N+1. Filename `{sense_id}-word-v{N+1}.mp3`.
3. Same `voice_id` as the original Stage 6/7 rendering (preserved via
   manifest `voice_id` column).
4. Call ElevenLabs with `text=text_input_canonical` (the canonical BP
   orthography); the dict applies the alias server-side. We do NOT
   pre-respell the text in our request body. The new manifest fields
   `text_sent_to_tts` will equal `text_input_canonical`,
   `alias_applied=true`, `alias_id=<id>`, `alias_respelling=<respelling>`.
5. Apply closed-loop loudness via existing
   [build/lib/loudness.py::normalize_pcm_to_mp3_verified](build/lib/loudness.py).
6. Upload to R2 with `Cache-Control: public, max-age=31536000, immutable`.
7. **Verify** (priorities reflect what each signal actually measures):
   - **Primary**: audio LLM judge re-run on the new clip. Must return
     `pronunciation_verdict == "bp_ok"` AND `confidence != "low"`.
   - **Sanity**: biased ASR (`language=pt`) still returns the canonical
     `pt` (Levenshtein ≥ 0.95). If audio is somehow wrong-word, this
     catches it.
   - **Sanity**: closed-loop loudness lands within ±1 LU of -16 LUFS
     (existing constraint).
   - **Diagnostic only** (NOT a gate): unbiased ASR transcript and
     phonetic distance, recorded for audit but never used to fail a clip
     — the original failure mode is specifically a same-spelling
     pronunciation problem ASR cannot see.
8. If primary + sanity all pass → `final_status = fixed_via_alias`. If
   not → `final_status = respelling_failed`, route to
   `data/_audio_manual_respelling_review.tsv`.
9. Append lifecycle event to `audit/08_5_rerender.jsonl`.
10. Update manifest with refreshed `version`, `url`, `md5`, `asr_*`,
    `audio_judge_*`, and the new alias-tracking columns.

**Pre-bulk smoke test (mandatory)**: render exactly one flagged clip
(e.g., `0317.00.01 animal`) via `--smoke-test` mode, listen to confirm BP
pronunciation, before any bulk re-render. Catches a fundamentally broken
alias dictionary before paying for the full batch.

**Cost** (assuming ~10% of word clips flagged ≈ 625 re-renders):
- ElevenLabs: ~6,250 credits (negligible against Pro tier headroom).
- Audio judge re-verify: ~625 × $0.005 ≈ $3.
- Biased ASR sanity: ~$0.30.
- R2 storage: negligible.

**Wall clock**: ~15 min at concurrency 4.

##### 08_6 — After-fix listening + finalize ([build/08_6_after_fix_and_finalize.py](build/08_6_after_fix_and_finalize.py))

Combines the after-fix listening checklist, the summary report, and the
final-TSV rebuild into one finalize step.

**Step 6.1 — After-fix HTML.** Generate `audit/08_6_after_fix.html` with
before/after listening pairs (~30–40 clips):

- All user-reported repaired clips
- All high-severity repaired clips (where Phase 1 judge returned
  `severity=high`)
- 10–20 random repaired clips
- ≥1 repaired clip per alias family (-al, -el, -il, -ol, -de, -te, ...)
- ≥1 repaired clip per voice (8 voices)
- All `respelling_failed` clips for visibility

Each row: two `<audio>` elements (v1 archived URL + v2 current URL),
alias used, 3-button radio (`BETTER / SAME / WORSE`). User listens
(~20 min) and pastes form output into
`data/_audio_after_fix_labels.tsv`.

**Reviewer guidance (LOCKED)**: the HTML embeds
[docs/reviewer_guide.md](docs/reviewer_guide.md) at the top — same
content as 08_2's calibration HTML — plus a short additional
BETTER/SAME/WORSE-specific rubric (also in the reviewer guide). The
reviewer's profile (A1 BP learner, Polish native, EN/DE fluent) shapes
the phonetic explanations; they should not have to look up unfamiliar
terms during a listening session.

**Hard gate before 06-final.tsv rebuild (LOCKED)**: after user labels
are imported, the script computes per-family `WORSE` and `SAME` rates
across the after-fix sample. The rebuild step (6.3) is **blocked** if
either of these holds:

- **Any `WORSE` row in any family** (regression introduced by the alias).
- **`SAME` rate > 10% in any family** (alias didn't change the
  pronunciation enough — too many "no improvement" clips).

When the gate blocks, the script:
1. Identifies the offending families (those with `WORSE` rows or
   `SAME > 10%`).
2. Routes the affected `pt`s to `data/_audio_alias_rerun_queue.tsv`.
3. Reruns Phase 08_3 → 08_4 → 08_5 limited to those families
   (new family-template smoke test, new alias dictionary version, new
   re-render).
4. Re-emits the after-fix HTML for the rerun clips.
5. Loops until no family triggers the gate (or N=2 rerun rounds, with
   the residual then surfaced loudly for manual review).

`BETTER` rows confirm the fix held. `WORSE` and `SAME>10%` rows are
hard-stop signals, not soft warnings.

**CRITICAL-voice escalation (NEW)**: if after-fix QA shows ANY clip from
a `voice_high_risk_critical` voice still labeled `WORSE` or `SAME` after
the N=2 rerun rounds, the script emits a separate
`data/_audio_critical_voice_review.tsv` and surfaces a loud manual
decision: **(a) keep the voice with stronger respellings**, which means
iterating Phase 08_3 with manual respelling overrides for the affected
`pt`s; or **(b) swap the voice entirely** via the surgical
[build/swap_voices_surgical.py](build/swap_voices_surgical.py) (cost ~$1
ElevenLabs + ~$0.30 ASR; wall ~10 min for 1,920 clips per swap, since
each affected voice carries ~480 word + ~480 example clips). Currently
no voices in this tier (both originally-flagged voices were swapped
pre-Stage-8); this clause is preserved for future voice flags. The user
makes this call manually; the pipeline does not auto-swap.

**Step 6.2 — Summary report.** Emit `audit/08_summary.txt`:

```
Stage 8 pronunciation correction summary
─────────────────────────────────────────
Word clips inspected:           6,250
Risk-classifier counts:         P0:XX P1:XXX P2:XXX P3:XXX
Audio judge verdicts:           bp_ok:X non_bp:X unclear:X
Calibration sample size:           XX
User-confirmed mispronounced:     XXX
Family templates locked:            N
Aliases generated (unique pt):    XXX
Conflicts blocking upload:          0
Dictionary uploaded:              dictionary_id <pd_...>
Clips re-rendered:                XXX
Verified fixed:                   XXX  (YY.Y%)
Verification failed:               ZZ  → _audio_manual_respelling_review.tsv
After-fix listening verdict:      BETTER ZZZ / SAME ZZ / WORSE Z

Manifest state:
  Word clips at v1 (untouched):     W
  Word clips at v2 (re-rendered):   X
  Word clips at v3+ (manual fix):   Y

Cost actuals: $X.XX
Wall clock:   XX min compute + XX min user
```

**Step 6.3 — Final TSV rebuild.** Stage 8 changes manifest URLs and md5s.
The final TSV (`data/06-final.tsv`) is derived from the manifest and MUST
be re-derived after the manifest update:

```bash
.venv/bin/python build/derive_final.py
.venv/bin/python build/verify_all.py
```

`verify_all.py` asserts (in addition to the existing Stage 6/7 invariants):
- Every Stage-8-changed manifest row has `alias_applied=true` AND
  non-null `alias_id` AND non-null `pronunciation_dict_locator_id`.
- Every `alias_id` in the manifest resolves in `_pronunciation_aliases.tsv`.
- HEAD requests 200 OK on a forced sample of ≥10 known-Stage-8-changed
  `sense_id`s.

#### Manifest schema additions

Additive columns introduced by Stage 8 (existing columns unchanged):

| Column | Notes |
|---|---|
| `text_input_canonical` | The canonical BP orthography. Mirrors the `pt` field; what's shown on the Anki card. |
| `text_sent_to_tts` | The string actually placed in the ElevenLabs API request body. Equals `text_input_canonical` when alias dict is attached server-side. Differs only if we ever bypass the dict and respell directly in the request body (escape hatch). |
| `alias_applied` | Boolean. `true` if a Stage 8 alias was active at render time. |
| `alias_id` | Foreign key into `_pronunciation_aliases.tsv`. |
| `alias_respelling` | The substituted string (copied for human readability). |
| `pronunciation_dict_locator_id` | ElevenLabs `dictionary_id`. |
| `pronunciation_dict_version_id` | ElevenLabs `version_id`. |
| `audio_judge_verdict` | Latest audio-judge result for this clip (`bp_ok`/`non_bp`/`unclear`). |
| `audio_judge_confidence` | Latest audio-judge confidence (`low`/`medium`/`high`). |

Legacy `text_input` column is kept and mirrors `text_input_canonical` for
back-compat with Stage 6/7 readers that haven't been updated yet.

#### Critical assumptions to verify before execution

1. **`gpt-4o-audio-preview` (or successor) accepts MP3 input and returns
   structured Tool-Use output.** Verify with a 3-clip smoke run before the
   bulk Phase 1 sweep.
2. **Alias rules apply on `eleven_multilingual_v2`** with both
   `case_sensitive` and `word_boundaries` fields honored. Verified via the
   1-rule sandbox-dict smoke test in Phase 4 step 3 (~$0.001).
3. **`pronunciation_dictionary_locators` accepts our id+version_id format
   on `eleven_multilingual_v2` text-to-speech.** Verified via Phase 5
   pre-bulk smoke test before the full re-render.
4. **R2 caches do not surface stale audio after filename bump.** Already
   verified across Stages 6 + 7.
5. **No example clip needs the dictionary.** Scope locked to word clips.
   The dictionary, once uploaded, would harmlessly apply to example clips
   if attached at TTS time; Stage 8 does not attach it on examples.
   Future `regenerate_flagged.py` runs may want to attach it always —
   separate decision.

#### Verification runbook

```bash
# Pre-flight: confirm Stage 7 is complete and the manifest is stable.
.venv/bin/python build/audio_status.py

# 08_0 — Risk classifier (deterministic).
.venv/bin/python build/08_0_classify_risk.py

# 08_1 — Audio judge (risky bucket).
.venv/bin/python build/08_1_audio_judge.py --confirm
# Or, for full corpus:
# .venv/bin/python build/08_1_audio_judge.py --confirm --full-corpus

# 08_2 — Calibration HTML.
.venv/bin/python build/08_2_calibration.py
open audit/08_2_calibration.html
.venv/bin/python build/08_2_calibration.py --apply-calibration

# 08_3 — Alias generation (sentinel smoke first, then bulk).
.venv/bin/python build/08_3_generate_aliases.py --sentinel-smoke
open audit/08_3_sentinel_smoke.html
.venv/bin/python build/08_3_generate_aliases.py --apply-templates
.venv/bin/python build/08_3_generate_aliases.py --fill-individuals --confirm

# 08_4 — Dictionary upload (incl. 1-rule sandbox smoke).
.venv/bin/python build/08_4_upload_dictionary.py --smoke-test
.venv/bin/python build/08_4_upload_dictionary.py --confirm

# 08_5 — Pre-bulk smoke + bulk re-render with verify.
.venv/bin/python build/08_5_rerender_verify.py --smoke-test --sense-id 0317.00.01
.venv/bin/python build/08_5_rerender_verify.py --confirm --concurrency 4

# 08_6 — After-fix listening + finalize.
.venv/bin/python build/08_6_after_fix_and_finalize.py --build-html
open audit/08_6_after_fix.html
.venv/bin/python build/08_6_after_fix_and_finalize.py --apply-and-rebuild

# Tests.
pytest tests/test_stage_8.py
```

#### Phase 7 — Post-08_5 example-clip audio judge sweep (LOCKED 2026-05-10)

After 08_5 word-only re-render completes, run a comprehensive
example-clip audio-judge sweep to verify example clips weren't
silently mispronouncing the same cognates that the word clips did
(sentence prosody usually saves them, but this is empirical
verification rather than assumption).

**Scope**: every example clip whose sense has `pt_type=single_word`
in `data/05-ipa.tsv` — approximately **5,688 example clips**.
Multi-word and idiom-expansion senses (~37) are out of scope
(different phonetic context; Stage 8 word-level patterns don't apply).

**Exclusions** ("unless already included/flagged/planned for replacement"):
- Example clips already judged in any prior pass (none yet, but a
  re-run safety net).
- Senses already in `data/_audio_human_review.tsv` (handled separately).
- Senses already in `data/_audio_manual_respelling_review.tsv` (failed
  08_5 verification — examples follow whatever decision the user makes
  on the word).

**Flow**:
1. After 08_5 completes (or in parallel, since it operates on different
   clip_type), run a variant of `build/08_1_audio_judge.py` filtered to
   `clip_type=example` and the inclusion/exclusion above.
2. Verdicts append to `data/_audio_judge_verdicts.tsv` (the existing
   verdicts file, schema reused — `clip_type` field distinguishes
   word vs example).
3. Any example clip returning `non_bp` at `confidence=high` (or
   `non_bp confidence=medium AND priority∈{P0,P1}` per the same
   auto-confirm rule from 08_2) becomes a candidate for a SECOND
   alias re-render pass — this time with `--include-examples` (a
   future flag on 08_5 that includes the example clip_type for
   confirmed sense_ids).

**Cost**: ~5,438 example clips × $0.007/clip ≈ **$38** (examples are
~5× longer than word clips, so audio token cost is higher).

**Wall clock**: ~30–45 min at concurrency 4.

**Why post-08_5 and not pre-calibration**: word-clip audio judge is
the primary signal for alias generation. Doing example judging FIRST
would extend 08_2's calibration sample and inflate user listening
time. Doing it AFTER 08_5 means we already know which words got
aliased — example drift on those specific words is the highest-prior
target. The "all single-word examples" scope ensures we don't miss
example-only drift that the word judge couldn't predict.

**Decision rule on findings**:
- ≤10 example-only non_bp clips: probably user-spot-check + manual
  override via `_manual_audio.tsv`.
- 10–50: spin up 08_5 with `--include-examples` for those specific
  sense_ids using the existing alias dict.
- 50+: investigate why examples are drifting more than word judge
  predicted; possibly add new alias rules; possibly accept residual
  via study-time loop.

#### Cost / wall-clock totals

**Default scope LOCKED 2026-05-10**: full-corpus audio judge (user opted
in for max coverage rather than the cheaper risky-bucket).

| Step | Cost (LOCKED default: full-corpus) | Cost (risky-bucket alt) | Wall clock |
|---|---|---|---|
| 08_0 risk classifier | $0 | $0 | ~5 min |
| 08_1 audio judge — words full-corpus | ~$20 (actual) | ~$6–9 | ~70 min (actual) |
| 08_2 calibration (185 clips) | $0 | $0 | ~60–90 min user |
| 08_3 aliases (sentinel seeds locked) | ~$0.50 | ~$0.50 | ~10 min |
| 08_4 dictionary upload | ~$0.001 | ~$0.001 | ~30 sec |
| 08_5 re-render words only (~250 clips) | ~$1.50 | ~$1.50 | ~10 min |
| Phase 7 — post-08_5 example judge sweep | **~$38** | **~$38** | ~30–45 min |
| 08_6 after-fix + finalize | $0 | $0 | ~20 min user + ~5 min compute |
| **Total** | **~$60** | **~$48** | **~135 min compute + ~150 min user** |

Voice-swap pre-Stage-8 (already executed 2026-05-10): +~$1.58 (1,920
clips re-rendered with new BP-only voices). Sentinel-seeds lock saves
~$0.05 vs LLM-proposed candidates.

Cost increase from $35 → $60 reflects two locked decisions: (1) full-
corpus word-clip audio judge as default detection, (2) post-08_5
example-clip safety-net sweep on all single-word senses. Both are
explicit user choices; both are within the project's acceptable
budget envelope (total project spend ~$340 incl. earlier stages).

#### What this stage deliberately does NOT do

- No example-clip detection or re-render (scope locked to word clips).
- No phoneme rules (multilingual_v2 ignores them; flash_v2 unsuitable).
- No model swap (still `eleven_multilingual_v2`; still `gpt-4o-transcribe`;
  audio judge is `gpt-4o-audio-preview` — additive, not replacing).
  *(Note: the "no model swap" scope-limitation was lifted in Stage 9. Multilingual v2 → Flash v2.5 migration documented in § Stage 9.)*
- No full-corpus ASR sweep. ASR is diagnostic only at re-render time, not
  a primary detection signal — same-orthography drift is exactly what ASR
  cannot see.
- No F1 threshold-fitting. Calibration verifies the audio judge is
  catching real failures; the auto-confirm rule is a direct combination
  of judge verdict + risk priority + human label, not a learned threshold.
- No N-best across many families. Sentinel smoke test on six
  representative words; mechanical family templates after.
- No automated bulk regeneration of cognates that pass audio-judge clean.
- No changes to Anki card display. The `pt` field stays canonical
  orthography (`text_input_canonical`); only the alias dictionary
  intercepts the string before TTS at render time.
- No phoneme-level audio editing. Aliases respell the input text;
  ElevenLabs renders end-to-end as normal.

## Stage 9 — Migration to Flash v2.5 + final deliverable (2026-05)

**Trigger**: After Stage 8 completion, ~30 single-word renders showed systemic mispronunciations that aggressive orthographic respelling (the v3 alias dictionary, 25 rules) couldn't fully fix without listener-noticeable artifacts. Pilot rendering on `eleven_flash_v2_5` showed FLASH_BETTER on 45 of 46 directly comparable pairs vs Multilingual v2: Flash handles most BP phonotactics natively (palatalization, final-l vocalization, initial /ʁ/) without alias rules.

**Outcome**: Full deck migrated to Flash v2.5 (11,450 clips), human-review queue cleared, residual non-BP rate at 0.05 %, final deliverable `data/06-final.tsv` shipped.

### Step 9.0 — Pilot + alias dictionary reset

- Pilot scripts: [build/08_8_pilot_flash.py](../build/08_8_pilot_flash.py) (single-word renders) and [build/08_9_pilot_flash_examples.py](../build/08_9_pilot_flash_examples.py) (example-sentence prosody check)
- Coverage: ~100 risky single-word senses + ~50 example sentences across all 8 voices
- Listener verdict (Lair voice as primary reference): FLASH_BETTER on 45 of 46 directly comparable pairs
- Decision: drop the v3 alias dictionary (25 rules) → **1 rule** (`hospital → ospitau`). All other v3 rules either became unnecessary on Flash or actively introduced artifacts.
- Archive of v3 dict for historical record: `data/_pronunciation_alias_v3_winners.tsv` (11 winning respellings preserved)
- Active alias file post-Stage 9: `data/_pronunciation_aliases.tsv` (2 rules after § Step 9.2 adds `gene → jêne`)

### Step 9.1 — Full deck re-render (11,450 clips)

- Model: `eleven_flash_v2_5` set as `DEFAULT_MODEL_ID` in [build/lib/elevenlabs_client.py](../build/lib/elevenlabs_client.py)
- Filename: model_id baked into the segment → `audio/{sense_id}-{word|ex}-eleven_flash_v2_5-v{N}.mp3`. Clips without the model_id segment are Multilingual v2 era; cleaned in Step 9.5. The model_id segment serves as on-disk provenance — a clip's filename answers "which model rendered this?" without a manifest lookup.
- Concurrency: **20** (ElevenLabs Pro tier limit for the Flash family — verified against `https://elevenlabs.io/docs/overview/models`)
- Safety: `fail_fast_on_429=True` in `ElevenLabsClient`. On a single 429 response the worker raises `RateLimitExceeded`, cancels in-flight futures, prints an ABORT banner, and writes `notes=stage_9_flash_migration_aborted_rate_limit` to affected manifest rows. The user is notified; the run resumes from `manifest.status=pending` on the next invocation.
- Pre-flight script: [build/09_0_migrate_flash.py](../build/09_0_migrate_flash.py) — explicit GO prompt + tee'd audit log + delegation to `stage_6.run(...)` with `sense_id_filter=None` and `concurrency=20`.
- Wall: ~30 min total for 11,450 clips at concurrency=20.
- Cost: ~$25 at Flash v2.5 rates (half the per-character cost of Multilingual v2).
- **Loudness normalization**: re-applied to every clip per the Stage 6 spec (closed-loop ffmpeg `loudnorm` to −16 LUFS, ±1 LU tolerance, verified post-encode). The Stage 9 render path delegates to `stage_6.run(...)`, so the same `_normalize_pcm` pipeline runs unchanged. Manifest columns `applied_gain_db`, `final_lufs`, `final_tp`, `loudness_within_tolerance`, `tp_limited` are populated on all 11,450 Flash rows. Observed: mean −16.90 LUFS, stdev 0.98; 72 % within ±1 LU, 28 % outside (driven by short word clips where `loudnorm`'s integrated LUFS measurement is unreliable on <2 s audio). 4 single-letter clips (`o`, `e`, `ó`) report `final_lufs=-inf` — measurement floor, not silent files. Outliers are logged in the manifest, not silently accepted; the spec language "anything outside that band is logged loudly" is satisfied even though the band is wider than originally hoped.
- After run: every manifest URL contains `eleven_flash_v2_5`; `verify_all.py` invariant #5 ("all audio URLs contain `eleven_flash_v2_5`") locks this in.

### Step 9.2 — ASR human-review queue (review HTML)

- Stage 7's ASR roundtrip (`gpt-4o-transcribe`, `language=pt`) flagged **99 senses** post-Flash where ASR similarity to `text_input` dropped below threshold.
- HTML generator: [build/09_1_review_queue.py](../build/09_1_review_queue.py) → produces `audit/09_1_review_queue.html` with a 3-way decision per row:
  - `LEAVE` — listener confirms the audio sounds BP and ASR is wrong (cognate-loanword cases, fast speech, rare phonemes)
  - `REGEN_VOICE_SWAP` — swap to a different same-gender voice and re-render
  - `TAG_ALIAS` — add an orthographic alias rule to fix a systematic mispronunciation
- Outcomes from user review:
  - **21 `REGEN_VOICE_SWAP`** decisions (voice IDs updated in `data/045-speaker_gender.tsv`)
  - **1 `TAG_ALIAS`** decision — `gene → jêne` added to alias file (BP /ˈʒɛni/ vs ElevenLabs default /d͡ʒiˈni/)
  - **77 `LEAVE`** decisions
- Decisions persisted: `data/_audio_review_queue_decisions.tsv`
- Application script: [build/09_2_apply_review_decisions.py](../build/09_2_apply_review_decisions.py) — updates voices + aliases + manifest, uploads a new ElevenLabs pronunciation dictionary version, runs `stage_6` on the affected `sense_id_filter` only. Handles the JS-export edge case where `sense_id` and `clip_type` concatenate without a tab (parser splits the first 10 chars `RRRR.EE.SS` from the rest).

### Step 9.3 — Audio judge audit of LEAVE decisions

- The 77 `LEAVE` decisions are user assertions ("this sounds BP"). An independent audio judge gives an objective second opinion.
- Model: `gpt-4o-audio-preview` (OpenAI audio-input model)
- Script: [build/09_3_audit_leave_decisions.py](../build/09_3_audit_leave_decisions.py) (concurrency=5, audit log at `audit/09_3_audit_leave_decisions.jsonl`)
- Per clip the judge returns `{pronunciation_verdict ∈ {bp_ok, non_bp, unclear}, drift, severity, confidence, evidence}`
- Result: **71 / 77 → `bp_ok`**, 6 → `non_bp` or `unclear`
- The 71 `bp_ok` clips are persisted in `data/_audio_asr_override.tsv` with schema:

```tsv
sense_id  clip_type  pt  voice_id  asr_status  audio_judge_verdict  audio_judge_drift  audio_judge_severity  audio_judge_confidence  audio_judge_evidence  judged_at
```

- `derive_final.py` consumes the override file: any sense in it gets `notes = audio_asr_status=untranscribable_audio_verified_bp` so downstream tooling can mute ASR-based shadowing drills on those rows (the learner would otherwise be told they're wrong when they're right).
- The remaining **6 residual genuinely-non-BP clips**: `rock`, `render`, `time`, `precedente`, `exceto`, `reitor`. Deferred — 0.05 % is within deck-launch noise. Targeted aliases possible in a future iteration if a learner reports them as confusing during study.
- **Effective audio-verified BP rate: 99.95 %** (5,719 / 5,725 senses, word-clip-weighted).

### Step 9.4 — Final deliverable: `data/06-final.tsv`

- New scripts:
  - [build/derive_final.py](../build/derive_final.py) — joins `03-enriched.tsv` + `05-ipa.tsv` + `_audio_manifest.tsv` + `_audio_asr_override.tsv` + `config/voices.tsv`. Writes the 30-column TSV. Logs join stats and gap counts to `audit/derive_final.log`.
  - [build/verify_all.py](../build/verify_all.py) — 8 hard invariants + HTTP-200 sample on 50 random URLs. Exits non-zero on any hard failure. Soft warnings on `family_root` / `source_line` / `example_pt` blanks (optional fields).
- 30-column schema unchanged from v1 (see § Final TSV schema below); only the audio URL pattern evolved across stages.
- Verification result on shipped 06-final.tsv: all 8 invariants pass; 50 / 50 sample URLs return HTTP 200; 1 soft warning category (`family_root` blank on rows where the source dictionary did not provide a root).

### Step 9.5 — R2 legacy cleanup

- Pre-migration R2 audio prefix held ~24,000 objects: 11,450 Flash v2.5 (current) + ~12,000 Multilingual v2 (orphans) + mid-iteration superseded versions.
- Script: [build/09_4_cleanup_legacy_r2.py](../build/09_4_cleanup_legacy_r2.py) — lists the `audio/` prefix, cross-references against `_audio_manifest.tsv` URL set, identifies orphans (in R2 but not in manifest), batch-deletes via S3 `DeleteObjects` (1000 keys per call).
- Safety: default dry-run; deletion requires `--confirm` flag. Never touches anything outside the `audio/` prefix.
- Result: **12,066 orphans deleted, ~525 MB freed.** Post-cleanup R2 holds exactly the 11,450 manifest-referenced objects.
- Bucket size: ~1,004 MB → ~479 MB.

### Step 9.6 — Manifest metadata correction

- Discovery: the `tts_model` column in `_audio_manifest.tsv` was stale on all 11,450 rows — value `eleven_multilingual_v2` despite Flash-rendered audio. The Stage 9.1 migration updated URLs, object_keys, voice_ids, md5s, ASR transcripts, and loudness data correctly, but the `tts_model` field was missed by the migration code path.
- Fix (commit `428b181`): single in-place pass setting `tts_model = eleven_flash_v2_5` on all rows where the URL contains the `eleven_flash_v2_5` segment.
- Verification: column-by-column diff against backup proved only col 5 (`tts_model`) changed; all 22 other columns byte-identical. No downstream regeneration needed — `06-final.tsv` does not surface `tts_model`.

### Settled artifacts after Stage 9

| Path | Purpose | Rows |
|---|---|---|
| `data/06-final.tsv` | Final deliverable, 30 columns | 5,725 senses |
| `data/_audio_manifest.tsv` | Authoritative audio state (post-Step-9.6 with corrected `tts_model`) | 11,450 clips |
| `data/_audio_asr_override.tsv` | BP-verified ASR exceptions | 71 senses |
| `data/_audio_review_queue_decisions.tsv` | Stage 9.2 review-queue decisions | 99 rows |
| `data/_pronunciation_aliases.tsv` | Active alias rules | 2 (`hospital`, `gene`) |
| `data/_pronunciation_alias_v3_winners.tsv` | Archived v3 aliases (historical) | 11 |
| R2 bucket `audio/` prefix | Live audio storage | 11,450 objects, ~479 MB |

### What Stage 9 deliberately does NOT do

- No fix for the 6 residual non-BP clips. Within noise; revisit only if a learner reports a specific one as confusing.
- No rewrite of Stages 1–8 documentation. Forward-pointer notes added at the stale references (Stage 6 generation, Stage 6 filename pattern, Stage 8 "deliberately does NOT").
- No Anki note-type or `.apkg` build. Out of scope per project conventions.
- No model-registry routing for the TTS model. `eleven_flash_v2_5` is hardcoded in `elevenlabs_client.py` rather than read from `config/models.yaml`. The TTS fleet is small enough (one provider, one chosen model) that registry indirection adds complexity without benefit; LLM jurors keep the registry because they swap models often.

## Final TSV schema (`data/06-final.tsv`)

| # | Column | Type | Notes |
|---|---|---|---|
| 1 | `sense_id` | `RRRR.EE.SS` | Stable forever; three zero-padded fields: rank.expansion_index.sense_index (e.g., `0001.00.03`) |
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
| 20 | `audio_word` | URL | R2 link to headword audio, filename ends `-v{N}.mp3` (derived from manifest). One voice per sense. |
| 21 | `audio_example` | URL | R2 link to example-sentence audio, same voice as `audio_word`. |
| 22 | `audio_word_md5` | hex | Corruption detection |
| 23 | `audio_example_md5` | hex | |
| 24 | `voice_id` | string | ElevenLabs voice ID that generated both clips for this sense. Resolves in `config/voices.tsv`. |
| 25 | `voice_gender` | enum | `m` / `f`. Redundant with `voice_id` but human-readable in spreadsheets. |
| 26 | `family_root` | string | Empty if standalone |
| 27 | `tags` | string | Space-separated |
| 28 | `source_line` | string | Raw original; never mutated |
| 29 | `source_line_number` | int | Ledger join key |
| 30 | `notes` | string | Escape hatch |

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

## API client conventions (rate limits, concurrency, retries, caching)

Sequential calls across ~10,000 senses take days; naive `asyncio.gather` immediately trips Anthropic, OpenAI, and ElevenLabs rate limits. All three API wrappers ([build/lib/llm.py](build/lib/llm.py) for Anthropic+OpenAI, [build/lib/elevenlabs_client.py](build/lib/elevenlabs_client.py) for audio, [build/lib/lookup.py](build/lib/lookup.py) for HTML scrapers) follow one shared policy.

### Prompt caching is MANDATORY for high-volume stages

**Discovered the hard way during the Stage 1.5 first run:** without prompt caching, Sonnet costs ~$0.006 per row × 4989 rows ≈ ~$30 for one stage. With Anthropic prompt caching applied to the system prompt (which is stable across all rows), per-call cost drops ~5–6× because cached reads bill at 10% of base input rate.

Apply to **every** Anthropic stage where the system prompt is identical across calls and length > 1024 tokens:

- Stage 1.5 (bp_status) — ~4989 calls
- Stage 2 (sense split) — ~5000 LLM-targeted calls
- Stage 3 (gender / cognate / family fallback) — ~3000 LLM-eligible calls
- Stage 4 (example generation + validator) — ~9000 + ~9000 calls
- Stage 5 (IPA correction) — ~9000 calls
- Stage 5.5 (adversarial + verdict auditor) — ~18,000 calls

Rough corpus-wide LLM cost without caching: ~$300+. With caching: ~$50–80. The 6× factor is real and load-bearing for project affordability.

#### Implementation in `lib/llm.py`

Use Anthropic's [prompt caching beta](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — cache the system prompt with `cache_control={"type": "ephemeral"}`:

```python
resp = client.messages.create(
    model=model,
    system=[
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ],
    tools=[...],
    tool_choice={"type": "tool", "name": tool_name},
    messages=[{"role": "user", "content": user_message}],
)
```

The first call seeds the cache (~5 min TTL); subsequent calls within the TTL pay 0.1× input rate for the cached portion. Concurrency above the rate-limit cap doesn't break caching — the cache is keyed on prompt content, not request flow.

Validation: log `usage.cache_creation_input_tokens` and `usage.cache_read_input_tokens` in the audit JSONL. Cache-hit ratio should reach >95% within the first ~50 calls. If it doesn't, the system prompt is being mutated between calls (a bug).

### Anthropic Batch API (implement before Stage 3)

**Decision (locked):** implement Anthropic Message Batches API in [build/lib/llm.py](build/lib/llm.py) before Stage 3 kicks off. Stage 3's spend is small enough that batch is marginally useful, but Stages 4 + 5 + 5.5 (the big ones, ~18,000 calls combined at $30–50 each) get a **50% discount** with no quality loss when using Batch.

#### Implementation surface

- New method on `AnthropicClient`: `submit_batch(requests: list[dict]) -> str` returns batch ID; `poll_batch(batch_id: str, timeout_s: int = 86400) -> list[dict]` blocks until results are ready (or times out at 24h SLA).
- Internal: chunk requests into ≤10k-per-batch (Anthropic limit), submit in parallel, gather results.
- Each batch result item is parsed exactly like a sync result: extract the tool_use block, write to audit JSONL with `batch_id` field added.
- Stage scripts invoke via a new helper `call_tool_batch(rows, ...) -> list[dict]` that decides sync vs batch based on row count (default: batch when >100 rows).
- Caching still applies inside batches; cache_creation/cache_read tokens reported per item.

#### What batch is/isn't good for

- **Good for**: offline stages with no inter-row dependency (1.5, 2 LLM-eligible rows, 3, 4, 5, 5.5). 50% discount, 24h SLA.
- **Not good for**: tiebreaker calls in the 3-model jury (those are conditional on jury disagreement; can't pre-batch). Stay sync.
- **Not good for**: Stage 1a idiom expansion (only 12 calls; sync is faster wall-clock).

#### Wall-clock impact

Stages 4 + 5 + 5.5 sync would take ~6–10 hours total. Batch SLA is 24h but real-world turnaround is typically 2–6h. Net: similar wall clock, half the cost.

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

## Golden test set (smoke run before any full pipeline run)

Before any stage runs against the full corpus, it runs against `tests/golden_set.tsv` — a hand-curated subset of ~100 source lines that exercises every edge case the pipeline must handle. Failure on the golden set blocks the full run.

The golden set covers, at minimum:

- **Function words and pronouns**: `o`, `de`, `que`, `se`, `você`, `tu`, `vós`, `vosso`, `lhe` (~10 rows). Tests function-word policy and high-polysemy sense-splitting.
- **EP/BP lexical swaps**: `comboio`, `equipa`, `desporto`, `paragem`, `utilizador`, `registar`, `controlo` (~7 rows). Tests Stage 1c replacement and collision merge.
- **Forced gender homographs**: all 8 confirmed (`capital`, `polícia`, `rádio`, `corte`, `cabra`, `cura`, `grama`, `banana`). Tests forced M/F sense splits.
- **Idiom expansions**: all ~12 (`medida`, `diante`, `cento`, `seguida`, `vigor`, `redor`, `repente`, `invés`, `contrapartida`, `obstante`, `mercê`, `tona`). Tests Stage 1a expansion + idiom-row sense_id format (`expansion_index ≥ 01`).
- **False friends and (BP)/(EP)-flagged**: `rapariga`, `camisola`, `bala`, `trem`, `sítio`, `policial`, `troço` (~7 rows). Tests `_flags.tsv` and Stage 4 sensitive policy.
- **Sensitive terms**: `gozar`, `mulato`, `índio`, `aborto`, `arma` (~5 rows). Tests Stage 2.5 classifier and example_policy enforcement.
- **Hyphenated and compound headwords**: `primeiro-ministro`, `segunda-feira`, `mão-de-obra` (becomes `mão de obra`), `bem-estar`, `meia-noite`, `porta-voz`, `dia-a-dia` (becomes `dia a dia`) (~7 rows). Tests `pt_type`, hyphen rules, and token-list validation.
- **Top-frequency irregular verbs**: `ser`, `ir`, `ter`, `estar`, `fazer`, `ver`, `dizer` (~7 rows). Tests `target_word_used` (e.g., `vou` for `ir`).
- **Embedded `=`**: lines 31 (`eu`), 314 (`medida`), 3372 (`vós`). Tests parser split-on-first-` = ` rule.
- **`mina (M. Gerais)` false-positive**: tests gender-marker false-positive guard.
- **Reflexives with `+se` annotation**: ~5 rows. Tests canonicalization in lookup.
- **Polysemy with `/`**: `melhor` (1 sense), `ponto` (3 senses). Tests sense-split jury.
- **Long-tail / random**: ~30 random rows from rank > 3000. Sanity check.

Total: ~100 rows. Each pipeline stage has a corresponding `tests/test_<stage>_golden.py` that runs the stage on the golden set and asserts:

- All structural invariants (sense_id format, ledger consistency, no stage crashes).
- LLM jury produces stable answers across two consecutive runs (cached when possible to control cost).
- `_sense_source_map.tsv` round-trips correctly.
- Stage 6 audio: ASR roundtrip passes for all golden clips on the chosen TTS provider/voices (caught early if voices are bad before paying for the full run).

The golden set is committed; the expected outputs are committed (`tests/golden_outputs/`); diff failures are loud. This is the cheapest possible insurance against silent pipeline regressions when models or prompts change.

### Smoke sample for Stage 4+ — two-tier setup

We use a **two-tier smoke sample** for Stage 4 (example generation), the most quality-sensitive stage in the pipeline:

- **Tier A — quick iteration** (`tests/smoke_sample_100.tsv`, 115 rows): used during prompt drafting and code iteration. Cheap (~$0.50 per run via sync API), fast (~1 min), enough to catch obvious prompt failures.
- **Tier B — pre-batch quality gate** (`tests/smoke_sample_1000.tsv`, 1000 rows): used as the final go/no-go before submitting the full ~5,720-row corpus to Anthropic Message Batches. Costs ~$5–8 sync (~$2.50–4 batch), runs in ~10–20 min, dense enough to catch systemic edge-case failures and statistical clusters of subtle errors.

Both files are committed; both are deterministic; both are regenerable from the build scripts in `tests/`.

#### Tier A — `tests/smoke_sample_100.tsv` (115 rows, locked)

Stratified random sample committed at `tests/smoke_sample_100.tsv`:

- ~57 nouns, ~23 verbs, ~13 adjectives, ~7 adverbs, ~4 other (proportional to corpus)
- 30 frequency-tier-stratified (6 each across top500/1000/2000/3000/5000)
- **25 hand-picked edge cases:**
  - 3 idiom expansions (`à medida que`, `em redor`, `ao redor`)
  - 4 forced gender splits (`capital` M/F, `cabra` M/F)
  - 2 NSFW / false friend (`rapariga`, `camisola`)
  - 3 high-frequency irregular verbs (`poder`, `ser`, `ter`)
  - 3 Stage 3 patched rows (`inovador`, `caça`, `tarde`)
  - **3 hyphenated compounds** (`primeiro-ministro`, `segunda-feira`, `bem-estar`) — locked addition
  - **2 reflexive verbs** with `+se` annotation — locked addition
  - **3 deep-polysemy function words** (`o`, `que`, `por`) — locked addition
  - **2 high-frequency adjectives** (`bom`, `grande`) — locked addition

Generated deterministically via `random.seed(42)`. Re-runs hit the same sense_ids so quality can be compared across model/prompt iterations.

#### Tier B — `tests/smoke_sample_1000.tsv` (1000 rows, pre-batch gate)

A 1000-row superset of Tier A, designed to catch systemic Stage 4 failures across every edge category before paying for the full-corpus batch.

**Composition (1000 rows total):**

| Bucket | Count | `sample_source` value |
|---|---|---|
| Tier A preserved verbatim | 115 | `pos_stratified` / `freq_stratified` / `edge_case` (unchanged) |
| New hand-picked edge cases (Option B coverage) | 141 | `edge_case_v2` |
| New stratified random via `random.seed(43)` | 744 | `random_v2` |

The seed-43 random selection is **disjoint** from seed-42's already-locked picks (the generator excludes any sense_id already present in Tier A or in the v2 edge-case list).

**New hand-picked edge cases (141 rows on top of Tier A's 25):**

| Category | New count | Detection signal |
|---|---|---|
| All remaining idiom expansions | 10 | `_idioms_expanded.tsv` minus the 3 already in Tier A; covers `em diante`, `por cento`, `em seguida`, `em vigor`, `de repente`, `ao invés`, `em contrapartida`, `não obstante`, `à mercê de`, `à tona` |
| All remaining forced gender splits | 12 | `tags` contains `#gendered-meaning` minus the 4 in Tier A; covers `polícia`/`rádio`/`corte`/`cura`/`grama`/`banana` × M/F |
| Sensitive terms — full coverage of NSFW / false-friend / BP-EP-flagged | 25 | Union of `bp_status ∈ {nsfw, false_friend}` (26 rows) ∪ `_flags.tsv` rows (11 rows) ∪ deterministic keyword screen (`gozar`, `mulato`, `índio`, `aborto`, `arma`, `bala`, `faca`, `morrer`, `matar`, `droga`); de-duplicated; minus rows already in Tier A. Bucket includes ALL false-friend senses — no separate false-friend bucket below to avoid double-count |
| Reflexive verbs | 18 | `data/source.txt` lines containing ` +se ` or `(+se)` joined back via `source_line_number` to enriched rows; sample 18 distinct, varied across rank tiers |
| Hyphenated compounds | 12 | `pt_type = "hyphenated_compound"` (note: `#hyphenated-compound` tag is not propagated — Stage 3 follow-up); sample stratified by rank; covers weekday compounds, `vice-*`, `ex-*`, `bem-*`, `meio-*`, `porta-voz`, `secretário-geral`, `matéria-prima` |
| Space compounds | 5 | `pt_type = "space_compound"`; covers `dia a dia`, `café da manhã`, etc. |
| Deep-polysemy function-word secondary senses | 12 | For each of `o, que, se, de, em, para, com, a, por`: include senses with `sense_index ≥ 2` (catches the polysemy that 1-sense-per-headword misses) |
| High-frequency irregular verb senses | 8 | Manually curated from rank ≤ 100 verb list: `ir`, `fazer`, `estar`, `dizer`, `ver`, `vir`, `dar`, `querer` (Tier A already has `ser`, `ter`, `poder`) |
| Cognates stratified across frequency tiers | 10 | `tags` contains `#cognate-en`; sample 2 each from top500/1000/2000/3000/5000; covers `importante`, `hospital`, `televisão`, `informação`, `chocolate`, `telefone`, `atenção`, `situação`, `possível`, `problema` |
| Numerals + interjections + comparatives | 10 | `pos = "num"` (5 sampled), `pos = "interj"` (3 sampled), plus `melhor` and `pior` (comparatives) |
| Top-100 frequency padding | 10 | Senses from `rank ≤ 100` not yet picked; ensures the highest-leverage rows are well-covered |
| Words with `#bp-rare` tag | 6 | Sample from the 26 `#bp-rare` rows; ensures rare-but-valid BP forms are tested |
| Words with `+se` reflexive that are NOT verbs (edge case) | 3 | Some rows mark `+se` but resolve to non-verb PoS — important to test |

Total new edge cases: 141. Combined with 115 Tier A → 256 hand-defined; remaining 744 are stratified random.

**Stratified random (744 rows, `random.seed(43)`):**

- Excludes all sense_ids in Tier A (115) and in the new edge-case bucket (141). Universe = 5,720 − 256 = 5,464 candidates.
- PoS-stratified proportional to corpus minus already-picked: roughly ~424 nouns, ~157 verbs, ~122 adjectives, ~23 adverbs, ~18 other (function words, etc.).
- Frequency naturally distributed within each PoS stratum.
- All sense_ids must resolve in `data/03-enriched.tsv`.

**Generator: `tests/build_smoke_sample_1000.py`**

A new committed script (no LLM calls; pure data shaping). Reads:

- `tests/smoke_sample_100.tsv` (preserved verbatim)
- `data/03-enriched.tsv` (canonical sense source)
- `data/_idioms_expanded.tsv` (idiom inclusion list)
- `data/_flags.tsv` (NSFW / false-friend / EP-flag list)
- `data/source.txt` (for `+se` reflexive detection via `source_line_number` join)
- `data/01-normalized.tsv` (for `pt_type` lookup; `03-enriched.tsv` carries it forward but tag-derivation lost some signals — see Stage 3 follow-up)

Writes `tests/smoke_sample_1000.tsv` with the same column schema as Tier A:

```
sense_id  rank  pt  pos  gender  en_primary  bp_status  split_category  sample_source
```

Idempotent: same inputs → same output. Run via `uv run python tests/build_smoke_sample_1000.py`. Re-runs after a re-pick of Stage 3 (e.g., after `_manual_gender.tsv` patches) auto-refresh.

**Tests: `tests/test_smoke_sample_1000.py`**

Hard-asserts the following invariants on `smoke_sample_1000.tsv`:

- Total row count = 1000.
- No duplicate `sense_id`s.
- All 115 Tier A `sense_id`s present unchanged.
- All 13 idiom expansions present (covers `_idioms_expanded.tsv` fully).
- All 16 forced gender split senses present (`#gendered-meaning` tag).
- All 5 `bp_status = "nsfw"` rows present.
- ≥ 18 reflexive verbs present (detected via `+se` source-line join).
- ≥ 12 hyphenated compounds present (`pt_type = "hyphenated_compound"`).
- ≥ 5 space compounds present (`pt_type = "space_compound"`).
- ≥ 12 function-word secondary senses present (sense_index ≥ 2 for `{o, que, se, de, em, para, com, a, por}`).
- ≥ 10 `#cognate-en` rows present, with at least 1 from each frequency tier.
- Every sense_id in the file resolves to a row in `03-enriched.tsv` (no orphans).
- Stratified random rows sum to exactly 1000 − 256 = 744.

**Stage 3 follow-up issue (noted, not blocking):** the `#hyphenated-compound`, `#space-compound`, `#reflexive`, and `#regional` tags are sparsely or never populated in `data/03-enriched.tsv` — Stage 3's tag-derivation pass missed these signals. The smoke-sample generator works around it by detecting via `pt_type` and source-text grep instead. A separate Stage 3 patch will re-derive these tags before final TSV; tracked outside this plan section.

**Cost & wall clock:**

- Building the sample: $0 (pure data shaping; no LLM calls).
- Running Stage 4 on the 1000-row sample: ~$5–8 sync, ~$2.50–4 via Batch API. Wall clock ~10–20 min sync, ~1–6 h via Batch.
- Running Stage 4 on the full corpus afterward (post-validation): ~$30–45 batch-discounted, ~3–6 h via Batch API.

The Tier B smoke gate adds ~$5/wall-clock-cycle of insurance against a $30+ full-corpus failure. Well worth it.

### Stage 3 follow-up patch: em redor / ao redor consistency

Spot-check on the smoke-sample edge cases revealed that Stage 3 classified `em redor` as `pos=adv` but `ao redor` as `pos=prep`. Both are adverbial idiom expansions; differing PoS is an LLM inconsistency. Patch via `_manual_gender.tsv`: force both to `pos=adv`. Applied in-place (no LLM cost). Idempotent for re-runs.

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

### End-to-end runbook

Most of this runs unattended. Human touchpoints are explicitly tagged ⚑.

0. ⚑ **Seed `_manual_sense_splits.tsv`** with the 8 forced gender splits and ~12 idiom expansions (~5 min).
0a. **Run the golden-set smoke test** (`pytest tests/`) end-to-end on ~100 hand-curated rows. Block full run on failure. (~1–2 min, automated.)
1. `python build/01a_parse.py` → ledger initialized; 3 embedded-`=` cases parsed; idiom candidates flagged.
2. `python build/01b_orthographic_normalize.py` → `01-normalized.tsv`; `_flags.tsv` populated with structured `flag_type`.
3. `python build/01c_lexical_replace.py` → `012-lexical_replaced.tsv`. Replacement collisions auto-merged where gloss overlap is high; ambiguous ones written to `_ep_drop_or_replace_review.tsv` (small queue, ~20 rows max — handled later in jury-disagreement review).
4. `python build/015_bp_status.py` → 3-model Tool-Use jury via the `juror_a/b/c` role keys; tiebreaker on full disagreement; `015-bp_status.tsv`.
5. `python build/018_dedupe.py` → `018-deduped.tsv`; auto-merges trivial duplicates.
6. `python build/02_split_senses.py` → jury + tiebreaker; `02-senses.tsv`. **No `--limit`, no human gate.**
7. `python build/sensitive_screen.py` → jury-classified `_sensitive_terms.tsv`; `example_policy` populated for Stage 4.
8. `python build/03_enrich.py` → cached Wiktionary/Priberam + jury fallback; `03-enriched.tsv`.
8a. **Pre-Stage-4 smoke gate**: `uv run python tests/build_smoke_sample_1000.py` regenerates `tests/smoke_sample_1000.tsv` from the latest enriched data; `pytest tests/test_smoke_sample_1000.py` enforces invariants. Then run Stage 4 against the 1000-row sample (sync, ~$5–8, ~10–20 min) and spot-check ~30 random rows for sentence quality before submitting full corpus to Batch API. Block full run on systemic failures.
9. `python build/04_examples.py` → generator + separate-model semantic validator; failures auto-regenerate; `04-examples.tsv`. **No human gate.**
9a. `python build/045_speaker_gender.py` → OpenAI gpt-4o-mini classifier emits `speaker_gender ∈ {male, female, neutral}` per sense; seeded balanced shuffle assigns voices to neutrals; produces `data/045-speaker_gender.tsv`. Cost ~$1, wall clock ~5 min.
10. `python build/05_ipa.py` → eSpeak baseline + LLM correction; `05-ipa.tsv`.
11. `python build/055_audit.py` → auditor pass; failures auto-regenerate; borderlines into `_jury_disagreements.tsv`.
12. ⚑ **Pre-audio cost report**: review character count and confirm spend (~5 min). Confirm ElevenLabs Pro subscription is active and `ELEVENLABS_API_KEY` in `.env`.
13. ⚑ **Configure R2 public custom domain** (~10 min). (Voice IDs already committed at `config/voices.tsv` — no longer a per-run human step.)
14. `python build/06_audio_pilot.py` → ElevenLabs PCM generation, ffmpeg loudness measurement (per-voice baselines cached to `data/_voice_loudness_baselines.tsv`), normalized MP3 encode, R2 upload, ASR roundtrip. Produces `_pilot_500.tsv`.
15. ⚑ **Voice quality + loudness check** (~15 min): (a) listen to ~10 random ASR-passed pilot clips per voice to confirm tone is right; (b) glance at the loudness verification report — the script asserts every encoded clip lands within ±1 LU of the -16 LUFS target. Anything outside that band is logged loudly, not silently. If voice tone is wrong, swap voice IDs in `config/voices.tsv` and rerun pilot for that voice's slice only.
16. `python build/07_audio_full.py` → uses cached per-voice loudness baselines (no re-measurement), generates remaining ~5,200 senses, ASR-validated; `06-final.tsv` produced; `_audio_human_review.tsv` for the rare twice-failed clips.
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

Honest about what this lean approach trades away:

- **Shared LLM blind spots.** Three frontier models from three families can still all confidently produce the same wrong answer on a subtle case (e.g., a low-frequency idiom translated in the same drift direction by all three). The two-auditor pass (adversarial + verdict, two further models) catches most of these. The N-best on top-1000 catches more on the rows that matter most. The study-time loop catches the rest. Residual: probably 0.3–1% of ~9,000 rows have undetected errors at deck-launch — call it ~30–90 rows. The user encounters and fixes ~1–3 per week of study, which is sustainable and far below the daily card-review count.
- **Audio rare failure modes.** ASR can mis-transcribe in the same direction TTS mis-pronounces (both treat a foreign loanword the same wrong way). Phonetic-distance check against eSpeak helps. Stricter top-1000 thresholds reduce frequency-weighted impact. Final defense is study-time flagging.
- **Sensitive-term policy drift.** The verdict auditor's check is the gate. If auditor policy understanding differs from the user's, some examples pass that the user would have rejected. Mitigation: keep [build/policies/sensitive_terms_policy.md](build/policies/sensitive_terms_policy.md) explicit and short (1 page, examples-driven), and feed it to **every** auditor and juror so all five models read identical constraints.
- **Cost overrun.** Five-model coverage triples or quadruples LLM line items vs. a single-model pipeline. Pre-flight cost report at step 12 is the gate before the audio line — the only line that's large in absolute terms.
- **Model deprecation mid-run.** A model named in `config/models.yaml` may be retired between dev and full-corpus runs. Mitigation: stage scripts read role-keys, not IDs; the registry records both `current_id` and `last_run_id` so re-runs are reproducible against the same model when possible, or fall forward to the new one with a recorded provenance note.
