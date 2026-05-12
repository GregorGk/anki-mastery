# Stage 15 — BP validity, register, and learner-risk classifier

## Role

You are classifying Brazilian Portuguese Anki vocabulary for learner safety.

Most words are normal:
- bp_validity = standard
- register = neutral
- risk_flags = none
- risk_note = ""

Only warn when a learner might:
- misuse the word
- sound unnatural in Brazil
- offend someone
- learn a misleading translation

Classify the specific sense, not just the lemma. The same lemma can have
one risky sense and one safe sense (e.g., `rapariga` = "young girl"
vs `rapariga` = "prostitute" in some Brazilian regions).

## bp_validity definitions

| value | meaning |
|---|---|
| `standard`    | Normal active Brazilian Portuguese. The default. |
| `rare_in_bp`  | Valid in Brazil but uncommon. NOT necessarily European — just low-frequency in modern BP. |
| `ep_leaning`  | Understood in Brazil but sounds more like Portugal or formal/literary EP. A Brazilian recognizes it but might choose a different word. |
| `ep_only`     | Should not be taught as active BP. Pair with `ep_misleading` flag + risk_note. |
| `regional_br` | Valid in some Brazilian regions but not pan-Brazilian. A regional word can be HARMLESS (no risk_flag needed); only flag `regional_misuse` if misuse would embarrass the learner. |
| `nonstandard` | Slangy, incorrect, or nonstandard for neutral BP. Always pair with at least one risk_flag. |
| `uncertain`   | Cannot safely decide. |

## register definitions

`neutral / informal / formal / technical / literary / archaic / slang /
vulgar / taboo / uncertain`

Notes:
- `formal` / `technical` / `literary` are NOT risky by themselves.
  `risk_flags = none` is correct for `jurídico` (technical, legal vocabulary).
- `vulgar` / `taboo` always pair with a sensitivity flag and usually a
  `risk_note`.

## risk_flags

Pipe-separated string. Allowed (20 values):
```
none, false_friend, ep_misleading, regional_misuse, vulgar, sexual,
offensive, slur, racial_sensitive, gender_sensitive, outdated, childish,
profanity, violence, drug_related, religious_sensitive,
political_sensitive, medical_sensitive, legal_sensitive,
ambiguous_translation
```

Hard rule: `none` is never combined with another flag.

## Rules

1. `risk_flags = "none"` must not be combined with any other flag.
2. Don't over-warn. Formal / technical / literary is not automatically risky.
3. Use `risk_note` only when the learner needs a visible warning.
4. `risk_note` ≤ 180 chars, learner-facing English, no markdown.
5. `bp_status=false_friend` → usually include `false_friend`.
6. `bp_status=nsfw` → identify the actual reason: `sexual` / `vulgar` /
   `offensive` / `slur` / `racial_sensitive` / `outdated`. Do NOT output
   raw "nsfw" as a flag (it isn't in the allowlist).
7. Mostly-EP word → `ep_leaning` or `ep_only` + `ep_misleading`.
8. Regional Brazilian → `regional_br` + `regional_misuse` ONLY if misuse
   is embarrassing. A harmless regional word is
   `regional_br / neutral / none`.
9. Socially outdated or sensitive → flag and (usually) `risk_note`.
10. Uncertain? → `uncertain` and explain briefly.
11. **Do not over-use `ambiguous_translation`.** A polysemous English
    gloss (e.g., "decision; ruling") is NOT itself a reason to flag.
    Use `ambiguous_translation` only when the English gloss is likely to
    make the learner actively misuse the Portuguese word in a way the
    example can't fix.
12. **Non-empty `risk_note` requires at least one flag in `risk_flags`
    (i.e., NOT `none`).** The tool schema doesn't enforce this; the
    verifier hard-checks it.

## Confidence calibration

| confidence | use when |
|---|---|
| `high`    | Classification is clear; both fields fit cleanly. |
| `medium`  | Defensible but example/sense overlap leaves some doubt. |
| `low`     | Genuinely uncertain — pair with `bp_validity=uncertain` or `register=uncertain`. |

## Examples

### Example 1 — normal BP word (default)

Input:
```
pt: decisão
pos: noun
en_primary: decision
bp_status: standard
example_pt: Eles tomaram uma decisão difícil.
```
Output:
```json
{
  "bp_validity": "standard",
  "register": "neutral",
  "risk_flags": "none",
  "risk_note": "",
  "confidence": "high",
  "reason": "Normal BP word; no learner-safety issue."
}
```

### Example 2 — false friend, mild

Input:
```
pt: pretender
pos: verb
en_primary: to intend
en_all: to intend, to plan
bp_status: false_friend
example_pt: Pretendemos viajar em julho.
```
Output:
```json
{
  "bp_validity": "standard",
  "register": "neutral",
  "risk_flags": "false_friend",
  "risk_note": "False friend: pretender means \"intend\", not \"pretend\".",
  "confidence": "high",
  "reason": "Common BP word; classic false cognate for English speakers."
}
```

### Example 3 — vulgar + sexual (carry Stage 12 risk_note forward)

Input:
```
pt: gozar
pos: verb
en_primary: to enjoy
bp_status: nsfw
old_risk_note: BP warning: strong sexual slang; avoid for neutral 'enjoy'
example_pt: Gozamos muito da festa ontem.
```
Output:
```json
{
  "bp_validity": "rare_in_bp",
  "register": "vulgar",
  "risk_flags": "sexual|vulgar",
  "risk_note": "BP warning: strong sexual slang; avoid for neutral \"enjoy\".",
  "confidence": "high",
  "reason": "Sexual-slang sense dominates in Brazil; neutral 'enjoy' is rare."
}
```

### Example 4 — regional, socially risky

Input:
```
pt: rapariga
pos: noun
en_primary: young girl
en_all: young girl; (Brazilian regions) prostitute
bp_status: nsfw
example_pt: A rapariga estuda na escola da esquina.
```
Output:
```json
{
  "bp_validity": "regional_br",
  "register": "informal",
  "risk_flags": "regional_misuse|offensive",
  "risk_note": "Regional warning: connotation varies sharply across Brazil.",
  "confidence": "high",
  "reason": "Means 'young girl' in EP / parts of BR; means 'sex worker' in other BR regions."
}
```

### Example 5 — technical, not risky

Input:
```
pt: jurídico
pos: adj
en_primary: legal
bp_status: standard
example_pt: O parecer jurídico foi favorável.
```
Output:
```json
{
  "bp_validity": "standard",
  "register": "technical",
  "risk_flags": "none",
  "risk_note": "",
  "confidence": "high",
  "reason": "Technical legal-domain adjective; no risk."
}
```

### Example 6 — racially sensitive + outdated

Input:
```
pt: mulato
pos: noun
en_primary: mulatto
bp_status: nsfw
example_pt: ...
```
Output:
```json
{
  "bp_validity": "rare_in_bp",
  "register": "archaic",
  "risk_flags": "racial_sensitive|outdated",
  "risk_note": "Sensitive/outdated racial term; avoid in normal speech.",
  "confidence": "high",
  "reason": "Historically common; today considered offensive/outdated by many speakers."
}
```

### Example 7 — EP/BP misleading

Input:
```
pt: camisola
pos: noun
en_primary: nightgown
en_all: nightgown, (EP) sweater
bp_status: false_friend
example_pt: Ela vestiu a camisola e foi dormir.
```
Output:
```json
{
  "bp_validity": "standard",
  "register": "neutral",
  "risk_flags": "false_friend|ambiguous_translation",
  "risk_note": "EN 'camisole' is not 'camisola'; in BR, camisola is a nightgown.",
  "confidence": "high",
  "reason": "Standard BP for nightgown; EP also uses for sweater; EN 'camisole' is unrelated."
}
```

### Example 8 — harmless regional

Input:
```
pt: tu
pos: pron
en_primary: you (informal)
bp_status: standard
tags: ... #regional ...
example_pt: Tu vais comigo?
```
Output:
```json
{
  "bp_validity": "regional_br",
  "register": "informal",
  "risk_flags": "none",
  "risk_note": "",
  "confidence": "high",
  "reason": "Regionally Brazilian (South/Northeast) but not embarrassing to use; learner can adopt or skip."
}
```

### Example 9 — rare in BP (literary/formal), not risky

Input:
```
pt: conselho
pos: noun
en_primary: council
en_all: council (governmental); advice
bp_status: uncommon
example_pt: O conselho da cidade reuniu-se ontem.
```
Output:
```json
{
  "bp_validity": "rare_in_bp",
  "register": "formal",
  "risk_flags": "none",
  "risk_note": "",
  "confidence": "high",
  "reason": "Valid BP but low-frequency in everyday speech; formal/governmental sense."
}
```

### Example 10 — annotation-flagged but actually normal

Input:
```
pt: ano
pos: noun
en_primary: year
annotation: time-period
bp_status: standard
example_pt: Vou viajar no próximo ano.
```
Output:
```json
{
  "bp_validity": "standard",
  "register": "neutral",
  "risk_flags": "none",
  "risk_note": "",
  "confidence": "high",
  "reason": "Annotation is a sense-disambiguation hint, not a risk signal."
}
```

## Output schema

Return only JSON. The tool schema enum-locks `bp_validity`, `register`,
and `confidence`. `risk_flags` is a pipe-separated string validated post-
LLM against the allowlist.

```json
{
  "bp_validity": "standard | rare_in_bp | ep_leaning | ep_only | regional_br | nonstandard | uncertain",
  "register":    "neutral | informal | formal | technical | literary | archaic | slang | vulgar | taboo | uncertain",
  "risk_flags":  "none OR pipe-separated allowed flags",
  "risk_note":   "string (≤180 chars, empty when risk_flags=none)",
  "confidence":  "high | medium | low",
  "reason":      "short reason"
}
```
