# Stage 19 — hand-off (Mac → VPS, 2026-10-01)

Read this first, then `docs/stage19_plan.md` (the approved plan) and the Stage 19
section of `README.md`. Work moved from the Mac to the OVH VPS
(`ubuntu@vps-9b2263c0.vps.ovh.net`, repo at `~/anki-mastery`) so long runs don't
depend on the laptop. Continue **only** on the VPS from now on.

## Where we are

| Step | State |
|---|---|
| −1 restore point | `4e49d23` |
| 0 prep, 1 preflight | done — `reports/19_0_preflight.html`, gate in `config/stage19_preflight.json` |
| 2 model selection | done, pinned in `config/stage19_models.tsv` |
| 3 IPA v2 | done, shipped to `06-final.tsv` / `07-anki-listening-general.tsv` (`2fe6984`) |
| 4 pilot | **in progress** — 685 senses rendered (`data/_v4_pilot_sample.tsv`, ledger `data/_v4_takes.tsv`, MP3s in `build/audio_cache/v4_takes/`); judging partly done; baseline, listening page, decide still to do |
| 5–8 | not started |

Last commit `cca0a9a`. All processes were stopped cleanly; the ledger is consistent.

## Decisions the user made (do not re-litigate)

- English `en_ex` stays on `eleven_v3`. Portuguese word + example → `eleven_v4`.
- Quality first: best/latest models, chosen by measurement.
- Accent target São Paulo (coda r = tap ɾ); fallback "don't force it".
- Fix word + sentence IPA on cards (done).
- The user listens to ≤ ~100 tricky/previously-failed recordings; everything else is gated by the automated judges.
- `o/a X` headwords are spoken as both forms: "o X, a X" (`spoken_headword`).
- ElevenLabs Pro for one month (resets 2026-11-01), then cancel.

## Findings that shape the engine

- v4 TTS input is **plain text**: inline IPA in any form does not steer v4 for Portuguese (stress pairs 3–5/20 vs 16/20 plain). Direction tag `[sotaque paulistano]` doesn't leak; it is a pilot arm.
- v4 accepts only stability + similarity; `language_code="pt"` = Brazilian Portuguese; ~0.133 credits/char; concurrency limit 10 (use 9); seeds are not deterministic.
- Judge = `gemini-3.1-pro-preview` AND `gemini-3.8-flash`, prompt J2 (expected IPA). Fallback pair with J1p. `gpt-audio` unusable.
- ASR stays `gpt-4o-transcribe` (`gpt-transcribe` "fixes" dropped final -r).
- Function words ("que", "o") fail J2 because the expected IPA is the weak form → short-word policy D5 (keep v3 unless v4 clearly passes).

## Gotchas

- Claude 5.x (`claude-opus-5-5`, `claude-fable-5-1`): forced `tool_choice` → 400; use `build/lib/llm.py` (`tool_request_params`), which handles it.
- google-genai 2.26 closes an unreferenced `Client` mid-request — keep a reference.
- Run long jobs with `python -u` under `nohup`/tmux, or output buffers.
- Gemini 402 = prepaid credits exhausted → tell the user to top up; don't retry-loop.
- Scripts load `.env` via python-dotenv. **Never `source .env` into the shell that runs `claude`**: it contains `ANTHROPIC_API_KEY`, which switches Claude Code to API-key auth and breaks Remote Control.
- Don't print `.env` values.

## Next steps (exact commands)

```bash
uv run pytest tests/ -q                                         # environment check first
uv run python -u build/19_3_pilot.py --render --yes             # finish judging (cached takes reused)
uv run python -u build/19_3_pilot.py --baseline --yes           # v3 clips under the same gate
uv run python build/19_3_pilot.py --listen-page                 # reports/19_3_listen.html (30 words × 3 blind)
```

1. The user labels the listening page (offer a phone-ready Artifact; the user is on an iPhone 17 Pro Max). Export → `data/_v4_human_labels.tsv`.
2. `uv run python build/19_3_pilot.py --decide` → `config/stage19_policy.tsv`; the user confirms.
3. Checkpoint commit + push (pre-authorized by the plan) before Step 5.
4. `uv run python build/19_4_render_v4.py --dry-run` → show the cost table → `--limit 20 --no-upload` smoke → `--yes` (in tmux).
5. `uv run python build/19_5_qa_report.py` (+ `--residue` page, ≤ 10 clips for the user).
6. Step 7 rebuild: derive_final → verify_all → 16_0 → 17_0 → 17_1 → 17_2 → 18_0 → 18_1 → full pytest; README final numbers; checkpoint commit + push.
7. The user imports into Anki; Step 8 cleanup only with the user's yes; remind the user to cancel ElevenLabs Pro.

## Rules recap

Commit/push only at the plan's three checkpoints or when the user asks. Show projected spend before any paid run. Never touch `data/source.txt`, `docs/plan.md`, `.env`, or `_manual_*.tsv` content.

## Running on the VPS (added after the 2026-10-01 OOM crash)

- Claude runs in tmux `anki` via `~/bin/claude-anki` (auto-restart, `claude --continue`,
  own scope with OOMPolicy=continue + MemoryMax=8G). Started by the user service
  `claude-anki` (also at boot). Restart log: `~/.cache/claude-anki-restarts.log`.
- `/tmp` is RAM (tmpfs) and there is no swap. Never put large scratch data there;
  scratch copies/rebuilds go under the repo's `tmp/` (on disk).
- Ubuntu 26.04's `tail` is uutils: `tail -1 f >> f` loops forever. Never read and
  append the same file in one command; use `l=$(tail -n1 f); printf '%s\n' "$l" >> f`.
- Long paid jobs (19_3, 19_4) run as transient user services, not as Claude
  background tasks, so they survive a Claude restart and vice versa; Claude
  only watches the log:
  `systemd-run --user --unit=anki-19-3 --collect --same-dir -p MemoryMax=4G -p StandardOutput=append:$PWD/tmp/logs/19_3.log -p StandardError=append:$PWD/tmp/logs/19_3.log $HOME/.local/bin/uv run python -u build/19_3_pilot.py --render --yes`
  (`mkdir -p tmp/logs` first; `systemctl --user status anki-19-3`; never auto-restart a paid job).
- Background tasks, monitors and workflows die with the Claude process and are not
  restored by `--continue`: after a restart, check `systemctl --user list-units 'anki-*'`
  and the ledgers before re-launching anything.
