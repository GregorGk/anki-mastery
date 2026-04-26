# Plan: Anki-Ready BP Portuguese Dataset (Data Phase)

## Context

The user is an A1 Brazilian Portuguese learner who knows many languages but none Romance. The goal is to build a **complete, high-leverage dataset** from the 4,985-entry frequency dictionary at [data/source.txt](data/source.txt) that can later drive an Anki deck — but the current phase is strictly **data preparation**, not Anki card construction.

Completeness is non-negotiable: no words skipped, no meanings skipped. Leverage means every row carries enough structure to support audio-rich, example-anchored, IPA-marked, sense-disambiguated review. The source is EP-leaning with BP swap tags; the user wants pure BP output. Downstream tooling (Anki note type, card templates, scheduling) is out of scope for this plan.

The outcome of this plan is a single authoritative TSV plus immutable audio assets on **Cloudflare R2** (with stable HTTPS URLs in the TSV) that the user can open in a spreadsheet to review and from which an Anki deck can later be deterministically generated.

## Settled decisions

| Decision | Value |
|---|---|
| Dialect | BP only; normalize EP pre-reform spellings; apply post-1990 hyphen rules |
| EP-only entries | Auto-drop with audit log (`data/_ep_drops.tsv`) |
| BP-native cross-check | Skipped |
| Sense granularity | Sense is the unit of study; one TSV row per sense |
| Sense-split method | Claude, user spot-checks first 200 |
| Sense distinction rule | Two items are distinct senses iff they require different example sentences |
| **Forced sense splits** | M/F gender homographs (`capital`, `corte`, `cura`, `grama`, `banana`, `cabra`, `polícia`, `rádio`); parenthetical idioms expanded to their own rows |
| Example sentences | 1 per sense, ≤15 words, A2-background grammar; target word in typical use for THAT sense; EN translation on card |
| Register | Neutral-everyday; tag if naturally travel/legal |
| IPA | BP neutral-paulistano, broad phonemic with stress, per-word for examples (isolated form) |
| **IPA sandhi gap** | Documented: per-word IPA does not reflect connected-speech word-linking present in audio. Not patched. |
| Audio | 4 clips per sense (2 voices × {word, example}); fixed 1M+1F across whole deck; ElevenLabs |
| **Audio column naming** | `audio_word_m`, `audio_word_f`, `audio_example_m`, `audio_example_f` — never `_v1/_v2`. Eliminates voice-mix-up risk on script restart. |
| Audio stage | Pilot first 500, then full batch |
| **Audio storage** | **Cloudflare R2** (S3-compatible, free egress, ~$0.02/mo). Stable HTTPS URLs in TSV. |
| Cognate flag | `#cognate-en` tag, no review-order change |
| Legal/travel expansion | Out of scope for now |
| Data format | TSV, one row per sense, stable `sense_id = rank.sense_index` |
| Study app | macOS desktop + iPhone |
| LLM providers | Anthropic + OpenAI; ElevenLabs for audio |

## Repo layout

```
anki-mastery/
├── data/
│   ├── source.txt                   # EXISTING — never mutated
│   ├── 01-normalized.tsv            # post EP-drop + BP spelling + hyphen + idiom expansion
│   ├── 015-bp_status.tsv            # post cheap LLM BP-vocab classification
│   ├── 02-senses.tsv                # post sense-split (one row per sense)
│   ├── 03-enriched.tsv              # post gender, PoS, family, tags, cognate
│   ├── 04-examples.tsv              # post example PT + EN + target_word_used
│   ├── 05-ipa.tsv                   # post IPA for word + example
│   ├── 06-final.tsv                 # FINAL DELIVERABLE (R2 audio URLs + md5)
│   ├── _ep_drops.tsv                # audit: EP-only entries removed
│   ├── _ep_spelling_map.tsv         # EP→BP spelling normalization rules
│   ├── _hyphen_rules.tsv            # post-1990 hyphen normalization rules
│   ├── _idioms_expanded.tsv         # audit: parenthetical idioms expanded into new rows
│   ├── _flags.tsv                   # NSFW / false-friend / (BP)-(EP)-tagged manual review queue
│   └── _pilot_500.tsv               # first 500 senses for audio QA gate
├── build/
│   ├── 01_normalize.py              # parse + BP-normalize + idiom expand + flag (BP)/(EP)
│   ├── 015_bp_status.py             # cheap LLM classification (catches lexical EP-isms)
│   ├── 02_split_senses.py           # sense-split, with forced M/F splits
│   ├── 03_enrich.py                 # gender + PoS via API with canonicalized lookup
│   ├── 04_examples.py               # examples + target_word_used validation
│   ├── 05_ipa.py                    # IPA, isolated-form
│   ├── 06_audio_pilot.py            # ElevenLabs + R2, first 500 only
│   ├── 07_audio_full.py             # ElevenLabs + R2, remaining
│   ├── verify_all.py                # cross-stage validator
│   ├── prompts/
│   │   ├── sense_split.md
│   │   ├── bp_status.md
│   │   ├── example_generate.md
│   │   ├── ipa_generate.md
│   │   ├── cognate_classify.md
│   │   └── idiom_expand.md
│   └── lib/
│       ├── tsv.py                   # read/write with quoting (handles line 262's quotes)
│       ├── parse.py                 # split-on-first-` = `, extract parentheticals
│       ├── bp_rules.py              # EP→BP spelling table; EP-only drop list; hyphen rules
│       ├── lookup.py                # Wiktionary BR / Priberam BR with canonicalization (-se strip, hyphen variants)
│       ├── llm.py                   # Anthropic + OpenAI wrappers with batching
│       ├── elevenlabs_client.py     # batch generation, retry, rate-limit handling
│       ├── r2_client.py             # Cloudflare R2 upload with public-read ACL
│       └── validate.py              # row-count, uniqueness, schema invariants
├── tests/
│   └── test_invariants.py
├── .env.example                     # ANTHROPIC_API_KEY, OPENAI_API_KEY, ELEVENLABS_API_KEY, R2_*
├── pyproject.toml
└── README.md
```

## Pipeline stages

Each stage reads the previous stage's TSV and writes the next. Each is idempotent (safe to re-run; rows with valid outputs already present are skipped). Each ends with assertions that fail loudly on schema/count/constraint violations.

### Stage 1 — Parse & BP-normalize ([build/01_normalize.py](build/01_normalize.py))

**In**: `data/source.txt` → **Out**: `01-normalized.tsv`, `_ep_drops.tsv`, `_idioms_expanded.tsv`, `_flags.tsv`

1. Read source.txt line by line. Split on **first** ` = ` only — handles the 2 lines with embedded `=` ([line 31 `eu`](data/source.txt:31), [line 314 `medida`](data/source.txt:314)).
2. Extract parentheticals from RHS into a structured `annotation` JSON field (dialect, number, gender-note, reflexive `+se`, etc.).
3. Apply `_ep_spelling_map.tsv` to transform headwords: `facto→fato`, `óptimo→ótimo`, `óptico→ótico`, `contacto→contato`, `contactar→contatar`, `afecto→afeto`, `adopção→adoção`, `excepção→exceção`, plus auto-detect `pt`/`ct` clusters and confirm with LLM.
4. Apply `_hyphen_rules.tsv` (post-1990 Acordo Ortográfico): `mão-de-obra→mão de obra`, `dia-a-dia→dia a dia`. **Preserve hyphens for**: weekdays (`segunda-feira`…), `vice-*`, `ex-*`, `bem-*`, `meio-*`, `meia-*`, `*-presidente`, `secretário-geral`, `matéria-prima`, `porta-voz`, geographic adjectives (`norte-americano`, `sul-americano`, `sul-africano`).
5. **EP-only headword detection**: curated drop list + heuristic (BP equivalent already exists in source AND headword matches known EP-only pattern). Examples: `comboio`/`trem`, `paragem`/`parada`, `equipa`/`equipe`, `planeamento`/`planejamento`, `registar`/`registrar`, `concelho`, `telemóvel`, `autocarro`. Drop with reason logged to `_ep_drops.tsv`.
6. **Parenthetical idiom expansion** — confirmed pattern (9 entries from grep): [575 `diante (em d.)`](data/source.txt:575), [659 `cento (por c.)`](data/source.txt:659), [1674 `seguida (em s.)`](data/source.txt:1674), [2042 `vigor (em v.)`](data/source.txt:2042), [2329 `repente (de r.)`](data/source.txt:2329), [3378 `invés (ao i.)`](data/source.txt:3378), [3791 `contrapartida (em c.)`](data/source.txt:3791), [4060 `obstante (não o.)`](data/source.txt:4060), [4727 `tona (à t.)`](data/source.txt:4727); plus the embedded-`=` case at [314 `medida (a m. que)`](data/source.txt:314). For each: resolve abbreviation against the headword (`(em s.)` + `seguida` → `em seguida`); spawn an additional row with `pt_display` = full idiom and `annotation.idiom` set; the bare-headword row is kept only if it has standalone use. Few-shot prompt template in [build/prompts/idiom_expand.md](build/prompts/idiom_expand.md). Logged to `_idioms_expanded.tsv`.
7. **Programmatic flagging**: every entry containing `(BP)` or `(EP)` in source RHS is logged to `_flags.tsv` for manual review **before** Stage 4 (example generation). Examples: [`rapariga`](data/source.txt:2124) (BP NSFW: prostitute), [`camisola`](data/source.txt:3842) (false friend: BP nightgown vs EP sweater), [`bala`](data/source.txt:2940) (BP: candy), [`trem`](data/source.txt:2604), [`sítio`](data/source.txt:913), [`policial`](data/source.txt:1125).

**Output schema**: `rank, pt, gender (blank), pos (blank), en_all, annotation, source_line`

**Validation**: ~4,950–4,970 rows post-drops; every row has `pt` and `en_all`; no `=` in `pt`; ~10 idiom rows added.

### Stage 1.5 — BP-status classification ([build/015_bp_status.py](build/015_bp_status.py))

**In**: `01-normalized.tsv` → **Out**: `015-bp_status.tsv`

Cheap Claude Haiku 3.5 pass. For each row, classify the headword as:

- `standard` — everyday BP vocabulary
- `uncommon` — BP-acceptable but rare; tag `#bp-rare`
- `false_friend` — different/offensive meaning in BP than EP; flag for manual review, tag `#false-friend`
- `nsfw` — offensive/sexual in BP (e.g., `puto`, `bicha`, `rapariga` BP-sense); flag, tag `#nsfw`
- `ep_only` — drop (catches lexical EP-isms missed by Stage 1's curated list — words like `frigorífico`, `chouriço`, `pequeno-almoço` patterns)

**Cost**: ~$0.20 (Haiku, ~5k entries × ~50 tokens). Catches what hand-curated maps miss.

**Validation**: every row has a non-empty `bp_status`. Rows classified `ep_only` appended to `_ep_drops.tsv` and removed from output.

### Stage 2 — Sense split ([build/02_split_senses.py](build/02_split_senses.py))

**In**: `015-bp_status.tsv` → **Out**: `02-senses.tsv`

For each row where `en_all` contains `/`: Claude classifies whether the slash-separated items are distinct senses or English near-synonyms. Few-shot examples: `melhor = better / best` (1 sense), `ponto = point / dot / period` (3 senses).

Rule enforced in prompt: *"Two items are distinct senses iff you cannot construct a single example sentence where both translations are natural."*

**Forced splits**:

- Source RHS contains `(M ... / F ...)` or `(F ... / M ...)` — gender-determined polysemy. Each gender-meaning becomes its own row; `gender` field carried correctly per sense (so Stage 3 doesn't re-look-up). Confirmed list: [408 `capital`](data/source.txt:408), [591 `polícia`](data/source.txt:591), [650 `rádio`](data/source.txt:650), [921 `corte`](data/source.txt:921), [3007 `cabra`](data/source.txt:3007), [3160 `cura`](data/source.txt:3160), [3333 `grama`](data/source.txt:3333), [4167 `banana`](data/source.txt:4167). Tag `#gendered-meaning`.
- Idiom rows from Stage 1 (already separate rows; preserved as-is).

Emit one row per sense. Assign `sense_id = {rank:04d}.{sense_index}` (e.g., `0001.1` … `0001.5` for `o`). Sense IDs are stable forever.

**QA gate**: pause after first 200 headwords are sense-split; user spot-checks. Approval required before processing remainder.

**Validation**: `sense_id` uniqueness; every row has `en_primary`; re-joining `en_primary` by sense_index reconstructs a superset of `en_all`; ~8,000–10,000 rows total.

### Stage 3 — Enrichment ([build/03_enrich.py](build/03_enrich.py))

**In**: `02-senses.tsv` → **Out**: `03-enriched.tsv`

Deterministic + LLM hybrid:

1. **Canonicalize headword for API lookup** ([build/lib/lookup.py](build/lib/lookup.py)):
   - Strip `-se` from reflexive verbs (preserve flag in `annotation.reflexive`).
   - Try alternate hyphenations if first attempt 404s (insert/remove hyphens per common patterns).
   - Try without diacritics as last resort.
   - For compound nouns, look up the **whole compound** (e.g., `mão de obra`, not `mão` + `obra`).
2. **Gender** (nouns): Wiktionary BR (primary) → Priberam BR (fallback) → Claude classification (last resort, prompt: *"reply only with `o`, `a`, `o/a`, or `—`"*). Fill `gender` and compose `pt_display` (`a casa`, `o caminho`, or bare word for non-nouns / idioms with their natural form).
3. **PoS**: derive from gloss pattern — "to X" → `verb`; gender known → `noun`; bare adjective → `adj`. **Leave blank when ambiguous** (per user requirement: PoS only when 100% certain).
4. **Tags**: 
   - Frequency tier: `#top500`/`#top1000`/`#top2000`/`#top3000`/`#top5000` from rank band.
   - PoS: `#verb`/`#noun`/`#adj`/etc. when known.
   - Morphology: `#reflexive` if `annotation.reflexive`; `#gendered-meaning` for the M/F splits; `#hyphenated` for preserved-hyphen compounds; `#idiom` for expanded idiom rows.
   - Regional: `#bp-rare` from Stage 1.5 `uncommon`.
   - Special: `#interjection`, `#numeral` from explicit lists; `#nsfw`, `#false-friend` from Stage 1.5; `#cognate-en` via Claude classification (one binary pass).
5. **Family root**: Claude proposes Portuguese derivational root (e.g., `criar` for `criação`, `criador`, `criatura`). Empty when standalone. Schema-reserved; population may be deferred without breaking downstream stages.
6. For M/F-split senses from Stage 2: gender already assigned, just confirm.

**Validation**: every noun has `gender ∈ {o, a, o/a}` (or `—` if genuinely uncertain); every reflexive verb has flag; tags space-separated and well-formed.

### Stage 4 — Example sentences ([build/04_examples.py](build/04_examples.py))

**In**: `03-enriched.tsv` → **Out**: `04-examples.tsv`

Per sense, prompt Claude Sonnet 4.6 with full sense context. Generate three fields:

- `example_pt`: ≤15 words; A2 background grammar (no compound tenses except where target verb requires it; subordinate clauses limited to `que`+indicative; pronoun clitics only when natural). Target word used in typical way **for THIS specific sense**. BP vocabulary and spelling.
- `example_en`: faithful English translation.
- `target_word_used`: the **exact surface form** of the target word as it appears in `example_pt` (e.g., `vou` for headword `ir` sense "to go"; `casas` for `casa` if pluralized; `pelo` for `por` if contracted). Solves the lemma-regex trap for irregular verbs.

**Validation per row**:

- `target_word_used` appears in `example_pt` as a **token** (whitespace + punctuation boundaries, **not** raw substring — avoids `por` falsely matching inside `porque`).
- Word count of `example_pt` ≤ 15.
- `example_en` non-empty.
- Rows in `_flags.tsv` (NSFW / false friend / (BP)-tagged) are processed only after manual approval is recorded in `_flags.tsv`.

**QA gate**: pause after first 500 senses; user spot-checks. Issues go into `_example_fixes.tsv` for targeted re-generation.

**Cost**: ~$30–60 (Sonnet, ~150 tokens/row × ~8,500 rows).

### Stage 5 — IPA ([build/05_ipa.py](build/05_ipa.py))

**In**: `04-examples.tsv` → **Out**: `05-ipa.tsv`

Claude generates broad phonemic BP neutral-paulistano IPA with primary stress, for:

- `ipa_word`: the headword as a single token (or the full idiom for idiom rows).
- `ipa_example`: each token of `example_pt`, **isolated form**, space-separated, aligned 1:1 with example tokens.

**Documented limitation (not a defect)**: per-word isolated-form IPA does not reflect connected-speech sandhi — vowel reductions, /s/-linking across word boundaries, contractions like `para o → pro`. The audio in Stage 6 will exhibit natural sandhi. Both serve different pedagogical purposes: IPA teaches the underlying form for production/decoding; audio trains the ear for connected speech. README spells this out for the eventual learner.

**QA gate**: user reviews first 100 `ipa_word` values. Spot-check vs `eSpeak-NG --ipa pt-BR` for sanity. Fix patterns of error if found, re-run. Manual override possible per-row.

**Validation**: `ipa_word` non-empty for every row; `ipa_example` token count == `example_pt` token count.

**Cost**: ~$15–25.

### Stage 6 — Audio pilot ([build/06_audio_pilot.py](build/06_audio_pilot.py))

**In**: `05-ipa.tsv` (first 500 rows) → **Out**: `_pilot_500.tsv`, audio assets on R2

Generate 4 mp3 clips per sense via ElevenLabs:

- `audio_word_m`: male voice, headword
- `audio_word_f`: female voice, headword
- `audio_example_m`: male voice, example sentence
- `audio_example_f`: female voice, example sentence

**File naming**: `{sense_id}-{word|ex}-{m|f}.mp3` (e.g., `0001.3-word-m.mp3`). Gender baked into filename → voice mix-up after a mid-batch crash is structurally impossible.

**Workflow**:
1. Voices: user-provided male voice ID; female voice ID placeholder until user supplies (swap-and-regenerate is a config change + ~30 min compute).
2. Compute md5 per file; store in `_md5` columns.
3. Upload to R2 with public-read ACL.
4. Stable URL pattern: `https://<R2-public-domain>/audio/{filename}`.
5. Local `build/audio_cache/` retained until final `.apkg` bundling.

**QA gate**: user opens `_pilot_500.tsv` in Numbers/Sheets, plays a sample via R2 URLs, confirms voice quality + pronunciation accuracy. Approval required before Stage 7.

### Stage 7 — Audio full ([build/07_audio_full.py](build/07_audio_full.py))

**In**: `05-ipa.tsv` (rows 501+) → **Out**: `06-final.tsv`

Same pipeline for the remaining ~7,500–9,500 senses after pilot approval.

**Idempotency**: resume = skip senses whose 4 expected R2 URLs are already in TSV and HEAD-request 200 OK.

**Validation**: every row has 4 audio URLs; HEAD requests 200 OK on a random 5% sample; md5 matches between local cache and R2-downloaded.

## Final TSV schema (`data/06-final.tsv`)

| # | Column | Type | Notes |
|---|---|---|---|
| 1 | `sense_id` | `RRRR.S` | Stable forever |
| 2 | `rank` | int | From source.txt |
| 3 | `pt` | string | BP-normalized headword |
| 4 | `gender` | `o`/`a`/`o/a`/`—` | Non-nouns get `—` |
| 5 | `pt_display` | string | With article for nouns; idiom form for idiom rows |
| 6 | `pos` | string or blank | Only when certain |
| 7 | `sense_index` | int | 1-indexed within headword |
| 8 | `en_primary` | string | Single English gloss for THIS sense |
| 9 | `en_all` | string | Full original RHS; audit |
| 10 | `annotation` | JSON string | `{dialect, number, gender_note, reflexive, idiom}` |
| 11 | `bp_status` | string | `standard`/`uncommon`/`false_friend`/`nsfw` |
| 12 | `ipa_word` | string | BP broad phonemic + stress |
| 13 | `example_pt` | string | ≤15 words, A2 background |
| 14 | `example_en` | string | English translation |
| 15 | `target_word_used` | string | Exact surface form of target word in `example_pt` |
| 16 | `ipa_example` | string | Per-word IPA, space-separated, isolated form |
| 17 | `audio_word_m` | URL | R2 link, male voice |
| 18 | `audio_word_f` | URL | R2 link, female voice |
| 19 | `audio_example_m` | URL | R2 link, male voice |
| 20 | `audio_example_f` | URL | R2 link, female voice |
| 21 | `audio_word_m_md5` | hex | Corruption detection |
| 22 | `audio_word_f_md5` | hex | |
| 23 | `audio_example_m_md5` | hex | |
| 24 | `audio_example_f_md5` | hex | |
| 25 | `family_root` | string | Empty if standalone |
| 26 | `tags` | string | Space-separated |
| 27 | `source_line` | string | Raw original; never mutated |
| 28 | `notes` | string | Escape hatch |

TSV chosen over CSV to avoid quoting friction (one source line has internal `"folk"` quotes). Python `csv` with `dialect='excel-tab'` and `quoting=QUOTE_MINIMAL` handles edge cases.

### Tag taxonomy

- **PoS**: `#noun #verb #adj #adv #prep #conj #pron #art #num #interj`
- **Frequency tier**: `#top500 #top1000 #top2000 #top3000 #top5000`
- **Morphology**: `#reflexive #gendered-meaning #hyphenated #compound #idiom`
- **Regional**: `#bp-rare`
- **Special**: `#nsfw #false-friend #cognate-en #numeral #interjection`
- **Semantic** (optional, deferred): `#travel #legal #family #body #time #politics #religion #food #nature #numbers #colors #emotions`

## Verification

**Per-stage**: each `build/NN_*.py` ends with assertions (row count, unique IDs, no empty required fields). Cross-stage validator at [build/verify_all.py](build/verify_all.py).

**End-to-end**:

1. `python build/01_normalize.py` → inspect `_ep_drops.tsv` (~15-30), `_idioms_expanded.tsv` (~10), `_flags.tsv` (~9 BP/EP-tagged), `01-normalized.tsv`.
2. `python build/015_bp_status.py` → inspect `015-bp_status.tsv`; review `_ep_drops.tsv` additions from this pass.
3. `python build/02_split_senses.py --limit 200` → spot-check sense splits in Numbers; re-run without `--limit` after approval.
4. `python build/03_enrich.py` → confirm gender filled for common nouns; M/F splits have correct `gender` per sense.
5. `python build/04_examples.py --limit 500` → spot-check examples, validate `target_word_used` token-boundary match.
6. `python build/05_ipa.py --limit 100` → spot-check IPA; re-run for remainder.
7. Supply female voice ID (or accept placeholder).
8. `python build/06_audio_pilot.py` → open `_pilot_500.tsv`, play sample audios via R2 URLs, approve.
9. `python build/07_audio_full.py` → `06-final.tsv` produced.
10. `python build/verify_all.py` passes; `pytest tests/test_invariants.py` passes.

## Cost & resource budget

| Item | Cost |
|---|---|
| Claude (sense split + examples + IPA + bp_status + cognate + idiom expand) | ~$60–110 |
| ElevenLabs (~970k chars × 2 voices) | ~$200–330 |
| Cloudflare R2 storage (~1.4GB) | ~$0.02/month |
| Cloudflare R2 egress | $0 (free) |
| **One-time total** | **~$260–440** |
| Ongoing | ~$0/month |

**Active human time**: ~25–30h, dominated by spot-check passes (sense-split sample, pilot 500 audio review, NSFW/false-friend manual review).

**Wall clock**: ~1 week assuming prompt user review at each QA gate.

## What this plan deliberately does NOT do

- No Anki note-type design, card templates, or `.apkg` generation (next phase).
- No EN→PT production cards or cloze cards (deferred).
- No BP cross-check against external corpora (user opted out).
- No legal/travel expansion vocabulary (out of scope for now).
- No conjugation sub-deck (deferred).
- No derivational-family **review order** — `family_root` is a schema-reserved field for future use only.
- No connected-speech IPA — gap documented, not patched.
- No storage of authoritative data as anything other than TSV files in this repo (simplicity over DB flexibility).
