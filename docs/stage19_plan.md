# Stage 19 — Portuguese audio on ElevenLabs v4 + validated São Paulo IPA (word & sentence), with the best current models

## Context

ElevenLabs released `eleven_v4` on 2026-09-28. Its documented strengths:
- the voice "no longer drifts back toward its source accent";
- inline IPA in the text (`"/ˈveɾdʒi/"`);
- 90+ languages, where the Portuguese variant is **Brazilian**;
- better instruction-following.

Today's deck is `eleven_v3`. You still hear mispronunciations (verde, proveniente, cara, quarto…), and some of them pass the Gemini judge (verde v2). Re-scoring the stored v3 transcripts also turns up defects the judge missed: pensar→"Pensa", "a honra"→"A uva", "o tapete"→"Utapit".

The Stage-05 IPA, shown on cards and needed to steer TTS, is inconsistent:
- stress mark placed before the vowel in ~85% of entries;
- ~6% of headwords have wrong stress (pessoa `pˈesoɐ`, política, semana, modelo, bonito, domingo);
- "está" has 10 different spellings across sentences;
- French-style `ʁ`;
- eSpeak artefacts.

**Goal:**
1. A validated São Paulo IPA for words and sentences, used on the cards and to steer TTS.
2. All **11,450 Portuguese clips** (word + example) re-rendered on v4 at the best quality we can *verify*.

Guardrails:
- never replace a v3 clip with a v4 clip that fails QA;
- every model is the best *available* one, chosen by measurement, not by assumption.

### Decisions so far (2026-10-01)
- English `en_ex` stays v3.
- You listen to **≤ ~100** tricky / previously failed recordings; everything else is gated by automated judges.
- A bigger automated test set runs first.
- **ElevenLabs Pro is active:** 925,303 credits, resets 2026-11-01; 160 voice slots; PVC/IVC enabled; no overage.
- Accent target **São Paulo** (coda r = tap `ɾ`). "Don't force it" is the fallback if IPA hurts.
- **Fix word + sentence IPA on the cards** in this stage.

### Keys and constraints (checked today with free read-only calls)

| Service | Result | Action |
|---|---|---|
| ElevenLabs | ✅ `pro`, 925,303 credits. `eleven_v4`: 1 credit/char, `language_code="pt"` = Brazilian Portuguese, `pcm_44100` allowed on Pro. All 17 voice ids reachable. | — |
| Anthropic | ✅ `claude-opus-5-5`, `claude-fable-5-1` available | — |
| OpenAI | ✅ `gpt-4o-transcribe` (current ASR) + newer `gpt-transcribe`, `gpt-audio` | — |
| Gemini | ✅ `gemini-3.1-pro-preview` (current judge) + `gemini-3.8/3.7/3.5-flash`, `gemini-3.5-transcribe` | — |
| Azure Speech | ❌ 401 invalid key; only an unmerged spike uses it | No new key needed |
| R2 | ✅ 35,409 objects, 1.88 GB of 10 GB free (~0.86 GB is unreferenced old Flash audio); v4 adds ~0.7 GB | Nothing to remove first |
| Local disk | 46 GiB free; v4 takes ~2 GB | Nothing to remove first |
| Env hygiene | `google-genai` installed but undeclared (`uv sync` would remove it); `.env.example` stale | Fixed in Step 0 |

**No new keys are needed, and nothing has to be deleted before starting.**

## Models — newest candidates, chosen by measurement

| Role | Today | Candidates | Chosen by |
|---|---|---|---|
| TTS | eleven_v3 | **`eleven_v4`** (not v4 Turbo, the lower-latency/lower-cost variant), `pcm_44100`, `language_code="pt"`, inline IPA, stability and audio-tag probes | Step 1 probes + Step 4 pilot |
| Audio judge | gemini-3.1-pro-preview, generic prompt | 3.1-pro, **3.8-flash**, 3.5-flash, OpenAI `gpt-audio`; prompts J1 (today), J1p (+ São Paulo coda-r line), J2 (+ expected IPA, stress and vowel checks) | Step 2 offline calibration on 176 clips you labeled |
| ASR (content check) | gpt-4o-transcribe | `gpt-transcribe`, `gemini-3.5-transcribe`, ElevenLabs `scribe_v2` | Step 2 A/B on the cached Stage-6 set + known v3 content defects |
| IPA adjudication | claude-sonnet-4-5 (Stage 05) | **`claude-opus-5-5`**, with `claude-fable-5-1` as independent auditor | Audit-sample disagreement rate |

## Budget

| Step | ElevenLabs credits | Other |
|---|---|---|
| 1 paid probes | ≤ 3K | < $0.20 |
| 2 model selection (offline) | 0 | ~$10 |
| 3 IPA v2 | 0 | Opus 5.5 Batches ~$25–50, Fable audit ~$5 |
| 4 pilot (~650 senses, winners reused) | ~60K | judges ~$25, ASR ~$2 |
| 5 full run (best-of-2 words, 1 take + ladder for examples) | ~400K (range 350–480K) | judge ~$80, ASR ~$10 |
| 6 residue | ~5K | — |
| **Total** | **~470K of 925K** | **≈ $150–190** (plus Pro) |

## Your touchpoints
1. Approve this plan, including the EP-spelling fix in Step 0.
2. Listen to ~10 probe clips in the preflight report.
3. Optionally skim the IPA v2 report and sign off its conventions table.
4. Label ≤ 90 blind recordings in the pilot.
5. Approve the full-run cost table (`--yes`).
6. Optionally review ≤ 10 residue clips.
7. Import the deck into Anki and confirm.
8. Approve R2 cleanup; cancel Pro.

Git: a restore-point commit + push comes first (Step −1). After that there are exactly three pre-authorized checkpoint commits + pushes (after Step 3, before Step 5, after Step 7).

## Steps

### Step −1 — Restore point: commit & push first (your request; done before anything else)
Tracked files are unchanged and `main` is in sync with `origin/main`, so this commit only adds today's untracked work.

**Commit (≈ 5 MB):**
- spike and review code: `build/azure_pa_spike.py`, `build/azure_pa_demo.html`, `build/borderline_review.py`, `build/phoneme_asr_spike*.py` (4 files);
- inputs and your labels:
  - `data/_mfa_dict/portuguese_brazil_mfa.dict` (MFA, CC BY 4.0, used by Step 3);
  - `data/_borderline_review_seed.tsv`;
  - `data/_audio_judge_ab_human_labels{,.raw}.tsv`;
  - `phoneme_asr_spike_review.tsv`;
- reports: `reports/18_*.html`, `reports/borderline_review.html`, `reports/phoneme_asr_spike*.html`, the 7 untracked `audit/*.html`.

**Left untracked, not committed:**
- `data/*.bak*` and `audit/*.bak*` (16.7 MB of backups of git-tracked files, redundant with history);
- `tmp/`.

Commit message: "Pre-Stage-19 restore point: phoneme/Azure spikes, borderline review, MFA dict, human labels, reports", plus the Co-Authored-By trailer. Push to `origin main`; report the hash.

**Later checkpoints:** approving this plan also authorizes commit + push at exactly three checkpoints, so the auto-run doesn't stall:
- after Step 3 (IPA v2);
- just before Step 5 (full render, which also covers the manifest backup);
- after Step 7 (rebuilt deck).

Nothing else gets committed without asking.

### Step 0 — Prep (no API spend)
- **EP spellings still in the deck:** 3359.00.01 génio, 3755.00.01 ingénuo, 3887.00.01 polémica, 4052.00.01/02 cómodo, plus 3 examples.
  - Add génio, ingénuo, polémica, cómodo and incómodo to `data/_ep_spelling_map.tsv`.
  - `build/derive_final.py` applies that map whole-token and case-preserving to `pt`, `pt_display`, `example_pt` and `target_word_used`, logging each change.
  - Re-derive before Step 3. These 5 senses are re-rendered in the pilot.
- **`build/lib/llm.py`:** make it work with Claude 5.x. Per the planning agent, Opus/Sonnet 5.5 reject forced `tool_choice` and require thinking, which Step 1 verifies.
  - For these models: `tool_choice=auto` + `strict` tool schema + a "call the tool" instruction; `max_tokens` ~16K; effort setting.
  - The same in `submit_batch`.
  - Audit records gain `generated_at`, `confidence`, `manual_override`.
- **`build/lib/elevenlabs_client.py`:**
  - model-aware voice settings: v4 gets stability + similarity only (v4 reports `can_use_style` / `speaker_boost` = False);
  - `seed` override;
  - `generate_pcm_meta()` via `with_raw_response`, returning `character-cost`, `request-id` and concurrency headers;
  - case-insensitive `Retry-After`;
  - typed `QuotaExceeded`;
  - fix the stale docstring.
- **`build/lib/asr.py`:**
  - catch `subprocess.TimeoutExpired` / `OSError` in `phonetic_distance`, plus an espeak semaphore;
  - a **relaxed comparison** that ignores spaces and articles and accepts `example_spoken` for digits and `_manual_audio.tsv` overrides;
  - a leaked-token check (barra / aspas / slash / IPA symbol names).
- **`build/lib/loudness.py`:** `-f mp3` when measuring MP3 input; guard against `-inf`.
- **`build/lib/tsv.py`:** atomic `write_tsv` (tmp + `os.replace`).
- **`build/lib/gemini_audio_judge.py`:**
  - `prompt_version` J1 / J1p / J2 (J1 stays byte-identical, so it stays comparable with the v3 verdicts);
  - model-specific thinking config;
  - prompt hash in the audit record;
  - fix the cost docstring (≈ $0.0027/clip, not $0.0002).
- **`build/verify_all.py:87`:** accept `eleven_v4`; a v3/v4 mix is a soft warning.
- **`pyproject.toml`:** declare `google-genai`.
- **`.env.example`:** real variable names (no values).
- **New `19_*` scripts** load `.env` with python-dotenv.

### Step 1 — Preflight smoke test · `build/19_0_preflight.py` (new)
Re-runnable. Modes `--free` and `--paid`. Writes `reports/19_0_preflight.html` + `audit/19_0_preflight.jsonl`; exits nonzero on FAIL.

**Free checks:**
- ElevenLabs: tier = pro, credits, `/v1/models` (v4, pt).
- Per voice `GET /v1/voices/{id}`: category, `free_users_allowed`, fine-tuning state, v4 high-quality list (as in `build/11_0_pilot_v3.py:98`).
- Anthropic, OpenAI and Gemini model lists contain every candidate in the models table.
- R2: list + size, and put/get/delete of `_smoke/preflight.txt`.
- Local: ffmpeg ≥ 4.2, espeak-ng, `google-genai`, disk free; every v3 BP clip is present locally (`data/anki_media/`, 11,450/11,450 today).

**Paid checks (≤ 3K credits):**
- One v4 `pcm_44100` render per production voice. This proves voice access, and records credits/char (header + subscription delta), the concurrency limit (sets the semaphore to limit − 1) and latency.
- Seed determinism: two same-seed renders → same md5?
- Accepted parameters: `language_code`, voice settings, dictionary locator, `apply_text_normalization` auto vs off with IPA.
- **IPA adherence** (2 voices), scored by ASR:
  - stress minimal pairs: sábia / sabia / sabiá, fábrica / fabrica, secretária / secretaria, público / publico / publicou;
  - leakage: `"/ˈɡatu/"` alone must come back as "gato";
  - nasal diphthongs (pão, mãe);
  - strong r `"/ˈhatu/"`, coda tap `"/ˈveɾdʒi/"`;
  - article placement `o "/doˈmĩɡu/"` vs `"/u doˈmĩɡu/"` (pause check with silencedetect).
- **Delivery probes:** a leading direction tag (e.g. `[sotaque paulistano, pronúncia clara]`) on 20 words, checking that ASR shows no tag leak.
- One call per judge/ASR candidate.
- A Claude 5.x forced vs auto tool-call check.

**Gate:**
- IPA is used in the pilot only if ≥ 80% of the stress pairs are honored, there is zero leakage, and the nasals are correct. Otherwise the pilot runs plain text ("don't force it").
- Direction tags go into the pilot only if they don't leak.

### Step 2 — Model selection, offline (no ElevenLabs credits) · `build/19_2_model_selection.py`
**Judge sets:**
- **L** = `data/_audio_calibration_labels.tsv`: your 185 labels on legacy word clips (53 MISPRONOUNCED, 83 OK, 48 UNCLEAR, which are reported only). The MP3s are in `build/audio_cache/` with verified md5s. The set is rich in subtle errors (série, roda, poço, cela, sede) that the old judge called "high confidence BP".
- **B** = `data/_audio_judge_ab_human_labels.tsv` (40, including EP controls).
- Plus the 21 labels in `phoneme_asr_spike_review.tsv` and verde v2.

**Judge configs:** {3.1-pro, 3.8-flash, 3.5-flash, gpt-audio} × {J1, J1p, J2}. Each is run twice to measure verdict flips. Confidence is ignored (99.95% of v3 verdicts were "high").

**Pre-registered judge rule:**
- Pick the config, or a 2-model AND gate, with the best recall on MISPRONOUNCED.
- Its false-reject rate on OK must be ≤ 10 points above today's.
- Its flip rate must be ≤ 10%.
- J1p replaces J1 only if its agreement on B is no lower.
- The runner-up is documented as a fallback in case a preview model disappears mid-run.

**ASR A/B:** extend `build/ab_asr_models.py` / `build/lib/asr_alt.py` with `gpt-transcribe` and `gemini-3.5-transcribe`, and run them on the cached Stage-6 A/B set + the v3 relaxed-ASR fails.
- Switch from `gpt-4o-transcribe` only if the challenger is no worse on good clips **and** no worse at catching the known content defects. An ASR that "fixes" pensa→pensar is worse for us.

Output: `data/_v4_model_selection.tsv` + `reports/19_2_model_selection.html` + `config/stage19_models.tsv` (pinned choices).

### Step 3 — IPA v2 (word + sentence; cards + TTS) · `build/lib/bp_ipa.py`, `build/lib/mfa_dict.py`, `build/19_1_ipa_v2.py`
**Sources:**
- G1 = Stage-05 IPA converted to the São Paulo convention, with a majority vote across all occurrences of each spelling.
- G1b = `bp_ipa` in `data/_ep_cues.tsv`.
- G2 = MFA dictionary, converted: `x` → `h`/`ɾ` by position; `c`/`ɟ` are fronted velars, so `k`/`ɡ`. It is a **soft** signal: only ~48% exact agreement, because of legitimate pretonic raising and similar regional variation.
- G3 = spelling rules for stress position and accent-mark vowel quality.

`mfa_dict.py` re-downloads the pinned dict if it is missing (sha256 `6ec51b88…`).

**Conventions** (shown in the report for your one-time sign-off):
- primary `ˈ` before the syllable onset; no secondary stress or syllable dots;
- strong r → `h`; coda and cluster r → `ɾ`;
- `tʃ`/`dʒ` before i, no tie bars;
- final unstressed e/o → `i`/`u`; final a → `ɐ`;
- coda s → `s`/`z`, voiced across words before a vowel or voiced consonant in sentences;
- final l → `w`;
- nasal diphthongs `ɐ̃w̃ ẽj̃ õj̃ ɐ̃j̃`;
- artefacts removed (ə, ʊ, ɪ, ŋ after a nasal, ASCII g);
- a ~40-entry clitic weak-form table (o `u`, de `dʒi`, e `i`, que `ki`, em `ẽj̃`, por `poɾ`, com `kõ`…).

**One pronunciation per spelling by default.** Stage-05's per-sense differences are mostly noise: vez `vɛs`/`ves`, letra, modelo, corrente…
- A curated **heterophone list** (gosto, jogo, olho, colher, almoço, começo, acordo, esforço, forma, molho, sede, seca, governo…) is resolved per occurrence, with sentence and gloss.

**Accept without the LLM** only when:
- G1 stress = G3;
- accent-mark vowel constraints hold;
- G1 ≈ some G2 variant under an allophony-insensitive key (stress and stressed-vowel quality stay strict);
- or, if the word isn't in MFA, G1 has no unaccented stressed e/o.

Everything else goes to **Claude Opus 5.5** (Batches; strict enum schema; candidates + spelling + sense gloss/sentence), is re-validated, and gets one retry with the validator errors. Unresolved → keep converted Stage-05 IPA, flagged `partial`. Logged to `audit/19_1_ipa_llm.jsonl`.
- **Pivot:** if > 35% of token types conflict, Opus transcribes every token, with the same validators.
- **Audit:** 200 tokens (100 accepted by consensus, 100 decided by Opus) are independently transcribed by **Fable 5.1**. The disagreement rate is reported.

**Validators:**
- symbol whitelist;
- exactly one `ˈ` per content word;
- stress = spelling rule unless on the adjudicated exception list;
- sentence token count = example tokens (whitespace tokens, as in Stage 05);
- the `target_word_used` token equals its word-level IPA when uninflected and not a heterophone;
- 32 digit sentences use `example_spoken`.

**Outputs:**
- `data/_ipa_lexicon.tsv`;
- `data/_ipa_v2.tsv` (sense_id, ipa_word, ipa_example, statuses, `example_spoken`, `tts_word_text`, `tts_ipa_ok`, flags);
- `reports/19_1_ipa_v2.html` (status counts, Stage-05 → v2 diffs by category, homograph decisions, audit rate, golden-test results).

**`build/derive_final.py`:** per-field precedence `_manual_ipa.tsv` (yours, read only) > `_ipa_v2.tsv` > `05-ipa.tsv`. The schema stays 40 columns.

This step can ship to your cards on its own, before any audio work.

### Step 4 — Pilot · `build/lib/v4_tts.py` (engine), `build/lib/audio_qa.py` (gate), `build/19_3_pilot.py`
The pilot is the production engine in pilot mode: winning takes are cached and reused by Step 5.

**Sample (seed 19, `data/_v4_pilot_sample.tsv`):**

| Stratum | Size | Contents |
|---|---|---|
| A — known weak | ≤ 300 | Calibration MISPRONOUNCED/UNCLEAR, confirmed mispronunciations, your 85 reported headwords, v3 judge-flagged and 16.9 re-renders, v3 relaxed-ASR fails, spike-bads, 11 alias winners |
| B — risky | 100 | Further P0/P1 senses |
| C — controls | 150 | Random, at least 12 per voice; every voice ≥ 45 senses overall |
| D — special cases | ~110 | Heterophones, the ~40 one- and two-letter headword senses (o, e, a, ó, de, em…), 10 o/a headwords, 10 multiword headwords, the 37 "hospital" senses, 10 digit sentences, the 5 EP-spelling fixes |

**Arms:**
- Words: plain ×2 takes and IPA ×2 takes.
- Examples: plain + ladder.
- Probes:
  - example target-word IPA (40);
  - alias dictionary on/off on the hospital senses;
  - stability 0.5 / 0.65 / 0.8 on 100 risky words;
  - direction tag on/off on 100 words, if Step 1 passed it.
- v3 baseline: every pilot sense's v3 clips are re-judged with the Step-2 gate.

**Takes:** in `build/audio_cache/v4_takes/` (already gitignored) plus `data/_audio_render_provenance.tsv`. That file holds the exact TTS text, seed, settings, verdicts and selection reason. The manifest gets no new columns, because `16_0` has a fixed DDL.

**Your page `reports/19_3_listen.html`:** **30 of your reported words × 3 blind versions (v3 / v4-plain / v4-IPA) = 90 recordings.**
- Shuffled per row; the answer key lives in `data/_v4_listen_key.tsv`, not in the HTML.
- Gloss shown; IPA hidden until revealed.
- G/B/U + notes, localStorage autosave, base64 audio (built from `build/borderline_review.py`).
- Export → `data/_v4_human_labels.tsv`.

**Pre-registered decisions** (`19_3_pilot.py --decide` writes `config/stage19_policy.tsv`, which you confirm):

| Rule | Decides | Rule as pre-registered |
|---|---|---|
| D1 | v4 go | On controls, v4's first-take pass rate is ≥ v3's − 1 pt under the same gate; on stratum A, v4 is ≥ 10 pts better or fixes ≥ 30% of v3 fails; your Bad-rate for v4 is ≤ v3's. Otherwise STOP. |
| D2 | Word IPA | IPA-first if ≥ 5 pts better than plain on A∪B, ≤ 1 pt worse on controls, ≤ 0.5% artefacts, and no more of your Bad labels than plain. Else plain-first with IPA as a rescue rung. Else plain only ("don't force it"). |
| D3 | Gate | Confirm it rejects ≤ 10% of the recordings you rated Good. |
| D4 | Voices | A voice failing at > 2× the median moves to the end of the swap order; above 25%, you decide. |
| D5 | Short words | ≤ 2-letter headwords (~40 senses) keep v3 unless both v4 takes pass and the v4 pass rate on D-short isn't worse. The gate is weakest there. |
| D6–D8 | Dictionary, stability, direction tag | Each adopted only if it improves the gate pass rate on its probe with no ASR regressions. |
| D9 | Example IPA rescue | Enabled only if it rescues ≥ 30% of failing examples with no pause artefacts. |

### Step 5 — Full render · `build/19_4_render_v4.py` (generalizes `build/16_9_surgical_rerender.py`)
**Scope and order:**
- every word/example row without an accepted pilot take;
- `en_ex` untouched (asserted byte-identical at exit);
- processed in `spaced_topic_order` (`data/_ordering.tsv`), earliest-studied first;
- each clip starts with its manifest voice (this keeps the 16.9 swaps).

**Words — best-of-2:**
1. Policy variant ×2 takes. Risky words also get the alternate variant ×1.
2. If none passes: 2 more takes.
3. Then up to 2 voice swaps, same gender, live fail-rate order; Dani is the last female option.
4. Cap 8 renders.

Among passing takes, pick by: more judge passes (an extra vote for risky words) → raw ASR pass → policy variant → higher ASR similarity → closer to −16 LUFS.

**Examples:** take 1 → take 2 → (example IPA if D9) → 2 swaps; cap 5.

**Acceptance:**
- TTS ok;
- audio sanity:
  - non-silent, even byte count;
  - word 0.25–3 s and 0.5–2× the v3 duration; example 0.6–1.7× v3 duration and ≤ 12 s;
  - ≤ 0.7 s edge silence;
- loudnorm ok (finite LUFS, true peak ≤ −1 dBTP; the tolerance flag is recorded, not gated);
- **relaxed ASR pass** on the plain display text, with no leaked tokens;
- the Step-2 gate passes.

ASR and the judge **never** see the IPA-bearing TTS string; a single `RenderSpec` carries both strings.

**Unresolved:** the v3 row stays untouched, and the clip is logged in `data/_v4_unresolved.tsv`.

**Seeds:** `sha256(sid|clip|eleven_v4|variant|voice|t{take})`; takes continue from the ledger, so a seed is never reused.

**Upload:** winners only, as `audio/{sid}-{word|ex}-eleven_v4-v{N}.mp3`, with N monotonic from an R2 listing + an in-run registry. HEAD before PUT; never overwrite.

**Manifest:**
- `.bak_pre_stage19` made once, refusing to overwrite an existing backup; atomic writes every 25 accepts;
- rows are written **only on accept**: all fields, including the 5 loudness fields; `notes=stage_19_v4:{variant}`; `text_input` stays orthographic.

**Throughput and budget:**
- ElevenLabs semaphore = measured limit − 1 (~9) and halves after repeated 429s;
- Gemini `SlidingWindowRateLimiter` (`build/lib/rate_limit.py`) ≤ 900 RPM; ASR and espeak semaphores.
- Budget guard:
  - sums `character-cost` and polls the subscription every 200 calls;
  - stops cleanly below 30K credits or on `QuotaExceeded` (exit 3, resumable);
  - `--dry-run` prints the projected spend; `--yes` is required.
- A live smoke test with `--limit 20 --no-upload` runs first.

### Step 6 — QA report + residue · `build/19_5_qa_report.py`
- `reports/19_5_v4_qa.html`: coverage, pass rates per ladder step, variant mix, swaps, v3 vs v4 gate rates, credits.
- Optional residue page (≤ 10 recordings), only for clips where neither v3 nor any v4 take passes. You choose "keep v3" or "take N".

### Step 7 — Rebuild and import
1. `derive_final`
2. `verify_all`
3. `16_0_build_sqlite`
4. **`17_0_build_ordering`** (16_0 wipes the DB)
5. `17_1_export_anki`
6. `17_2_ordering_html`
7. `18_0_fetch_anki_media` (fetches only the new files)
8. `18_1_build_apkg` (bundles only referenced media)
9. `pytest tests/`

Then **you** import `dist/bp-listening-general.apkg` and choose "update existing notes". GUIDs are stable, so the new audio and IPA replace the old in place and scheduling is kept. Run Tools → Check Media → Delete Unused. Add a Stage 19 section to `README.md` (not `docs/plan.md`).

### Step 8 — Cleanup (after your sign-off; each step needs your "yes")
- `build/09_4_cleanup_legacy_r2.py` gains `--keep-manifest .bak_pre_stage19`. It deletes the old Flash objects (~0.86 GB) and superseded v3, and deletes v3 BP objects only after a grace period.
- Prune losing takes and the old-model local caches.
- Cancel Pro.

## Tests (new; existing suites stay green)
- **`tests/test_stage_19_ipa.py`** — golden São Paulo cases:
  - stress: bonito `boˈnitu`, domingo `doˈmĩɡu`, está `esˈta`, pessoa `peˈsoɐ`, política `poˈlitʃikɐ`, criação `kɾjaˈsɐ̃w̃`, café `kaˈfɛ`, órfã `ˈɔɾfɐ̃`, rainha `haˈĩɲɐ`, juiz `ʒuˈis`, ruim `huˈĩ`, táxi `ˈtaksi`, lápis `ˈlapis`, ônibus `ˈonibus`, país vs pais;
  - r: verde `ˈveɾdʒi`, carro `ˈkahu`, rua `ˈhuɐ`, porta `ˈpɔɾtɐ`, falar `faˈlaɾ`, honra `ˈõhɐ`, Israel `izhaˈɛw`, mulher `muˈʎɛɾ`;
  - quarto `ˈkwaɾtu`, animal `aniˈmaw`, hospital `ospiˈtaw`, pão, mãe;
  - heterophones in context: "Eu gosto…" `ˈɡɔstu` vs "O gosto…" `ˈɡostu`; sede; colher;
  - cross-word coda-s voicing;
  - converter cases (`vˈeʁdʒi`→`ˈveɾdʒi`, `kwaɾtʊ`→`ˈkwaɾtu`, `eɾədˈejɾu`→`eɾˈdejɾu`);
  - a property test: every non-unresolved row validates.
- **`tests/test_stage_19_qa.py`** — relaxed ASR on real v3 transcripts:
  - "a corda"→"Acorda" passes; pensar→"Pensa" fails; "a honra"→"A uva" fails;
  - `_manual_audio` and digit overrides;
  - leak detection; sanity bounds; gate truth table.
- **`tests/test_stage_19_render.py`** — fake TTS/ASR/judge/R2 clients:
  - unique seeds, monotonic versions, no overwrite;
  - manifest changes only on accept, atomically, after a backup;
  - `en_ex` and short words untouched;
  - resume without re-rendering;
  - budget guard / `QuotaExceeded` leave the manifest consistent;
  - ladder caps.
- **`tests/test_stage_19_downstream.py`:**
  - IPA precedence;
  - EP-spelling map application;
  - `verify_all` accepts v4;
  - the `llm.py` Claude 5.x path.

## Verification
1. `19_0_preflight.py --paid` is all green, and you've heard the probe clips.
2. `pytest tests/` is green.
3. `reports/19_2_model_selection.html` shows the pinned judge and ASR beating or equalling today's.
4. `reports/19_1_ipa_v2.html` is reviewed: spot-check bonito, está, verde, quarto, sede, gosto and 20 random sentences; Fable audit disagreement reported.
5. The pilot decision box is confirmed with your 90 labels.
6. `19_4 --dry-run` cost table → `--yes`.
7. After the run:
   - every BP row is `eleven_v4`, or listed in `_v4_unresolved.tsv` / short-word v3;
   - the `verify_all` HEAD sample passes;
   - `18_1` builds;
   - the full suite is green;
   - the Anki import shows the new IPA and audio.

## Risks → mitigations

| Risk | Mitigation |
|---|---|
| v4 ignores or speaks the IPA | Step 1 gate; "don't force it" fallback |
| The judge misses subtle drift | Model and prompt chosen on your 185 labels; J2 is IPA-aware; relaxed ASR in the gate; blind 3-way check |
| IPA errors hurt the cards and TTS | Consensus + Opus + Fable audit; abstain → plain; your `_manual_ipa` wins |
| Library voices not tuned for v4 | Per-voice probes and fail rates; D4 |
| A preview model disappears mid-run | Pinned choice + runner-up fallback; resumable |
| 429s or credit exhaustion | Adaptive semaphore; Retry-After; budget guard |
| Manifest or object corruption | Accept-only writes; atomic; HEAD-before-PUT; monotonic versions |

## Rollback
- **Manifest:** `data/_audio_manifest.tsv` is git-tracked (checkpoint commit before Step 5), and `.bak_pre_stage19` is kept. A planned `build/19_9_rollback.py --sense-ids|--voice|--all` restores rows.
- **R2:** v3 objects stay until Step 8.
- **Display IPA:** remove `_ipa_v2.tsv` to fall back to Stage-05.
