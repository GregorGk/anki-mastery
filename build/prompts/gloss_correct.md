# Gloss correction (Stage 5.5 remediation)

## Role

You're correcting English glosses for Brazilian-Portuguese senses. Stage 2
produced an `en_primary` for each sense; Stage 5.5's auditor flagged some
rows as having `sense_consistency` issues (gloss doesn't match BP usage).

Your job: read the BP word, the correct BP example sentence, the current
gloss, and the auditor's complaint — then decide if the gloss needs
fixing, and if so, produce a clean replacement.

## Inputs

- `pt` — the Brazilian-Portuguese headword
- `en_primary` — current English gloss (may be wrong)
- `en_all` — full original RHS from the source (for context)
- `example_pt` — example sentence using `pt` correctly
- `example_en` — current English translation
- `auditor_defect` — what Stage 5.5's auditor said is wrong with this row

## Decision rule

Output `is_changed=true` ONLY when the current `en_primary` is genuinely
inaccurate as a gloss for the sense the example illustrates. Specifically:

- **False friend**: `eventual` glossed as "eventual" but BP `eventual`
  means "occasional". Fix.
- **Wrong English word**: `salgado` glossed as "relating to salt" but
  the sense is "salty". Fix.
- **Too narrow**: `cultivo` glossed as "act of planting" but `cultivo`
  means "cultivation" (broader). Fix.
- **Too broad / vague**: `recair` glossed as "to go back to" but
  `recair` specifically means "to relapse". Fix.
- **Wrong PoS form**: `paulista` glossed as "from São Paulo" (prep
  phrase) but `paulista` is a noun/adjective meaning "person from SP" /
  "São Paulo native". Fix.
- **Missing required preposition**: `constar` glossed as "to consist of"
  but the construction is `constar de` — the gloss should reflect that.
- **PoS mismatch**: row has `pos=num` but gloss/example show noun. Fix
  the gloss to match the actual sense; PoS is not your concern (a
  different stage handles that).

Output `is_changed=false` (i.e., return the same `en_primary` verbatim) when:

- The gloss is already correct and the example matches it.
- The gloss is correct but the example illustrates a different sense
  (this is a Stage 4 example issue, not a gloss issue — leave it alone,
  Stage 4 regen will fix). Set `is_changed=false`.
- The auditor was over-strict on a closely-related sense variant where
  the gloss is acceptable.
- You're unsure — preserve the original.

## Style for corrected glosses

- Short: 1-5 English words. No articles unless required.
- Use the most natural English for an A1 learner. Prefer the common word
  over the technical one.
- Match the BP word's actual register. If `recair` means "relapse" in
  medical/behavioral context and "fall back" in legal context, pick the
  one matching `example_pt`.
- For multi-sense headwords where this row teaches one specific sense,
  give a gloss specific to THAT sense (don't include alternatives).

## Output (Tool Use, mandatory)

```json
{
  "type": "object",
  "properties": {
    "corrected_en_primary": {"type": "string"},
    "is_changed": {"type": "boolean"},
    "reasoning": {"type": "string"}
  },
  "required": ["corrected_en_primary", "is_changed", "reasoning"]
}
```

If `is_changed=true`, `corrected_en_primary` is the new gloss.
If `is_changed=false`, `corrected_en_primary` should equal the input
`en_primary` verbatim. `reasoning` is ≤80 chars.

## Worked examples

| pt | en_primary (in) | example_pt | corrected | is_changed | reasoning |
|---|---|---|---|---|---|
| `eventual` | "eventual" | "Aceito trabalhos eventuais aos fins de semana." | "occasional" | true | False friend; PT eventual=occasional |
| `habitual` | "familiar" | "Ele sentou no seu lugar habitual." | "usual" | true | habitual=usual/customary, not familiar |
| `salgado` | "relating to salt" | "A água do mar é muito salgada." | "salty" | true | Sense is taste of salt |
| `multinacional` | "international corporation" | "...trabalha em uma multinacional americana." | "multinational corporation" | true | "international" was wrong word |
| `caça` | "hunting" | "A caça é proibida nesta região." | "hunting" | false | Gloss correct |
| `corda` | "cord" | "A corda do violão quebrou..." | "cord" | false | Gloss is correct; example issue is sep. (Stage 4 regen) |
