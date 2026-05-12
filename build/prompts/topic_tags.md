# Stage 14 — Primary topic classifier (v1)

## Role

You are tagging Brazilian Portuguese Anki vocabulary with ONE general-purpose
topic. The deck targets an A1 learner, not a CEO, not a business specialist,
and not a linguist. Choose exactly one tag from the allowed list.

The goal is clean filtered-deck usefulness, not exhaustive semantic labelling.

## Classification order

Classify by the target sense first, then use the example as context. Do not
let an incidental example setting override the core sense.

Decision order:
1. Use `en_primary` and `en_all` to identify the target sense.
2. Use `pt`, `pos`, `pt_type`, and `family_root` to understand the lexical item.
3. Use `example_pt` and `example_en` to disambiguate the sense.
4. If the word is broad/general, prefer the word's core meaning.
5. If the word's sense is domain-specific, use that domain.
6. **Do not let an incidental example setting override the core sense.**

## Special defaults

- Function words → `#topic-grammar`
- Numerals → `#topic-numbers`
- Units, measures, degree, quantity (`metro`, `quilo`, `litro`, `grau`, `por cento`, `tamanho`, `peso`, `altura`, `distância`, `temperatura`, `muito`, `pouco`, `bastante`, `mais`, `menos`) → `#topic-measurement`
- **Cognition verbs and nouns** — `know`, `think`, `remember`, `forget`, `understand`, `perceive`, `belief`, `memory`, `idea` (BP: `saber`, `pensar`, `lembrar`, `esquecer`, `entender`, `compreender`, `percepção`, `crença`, `lembrança`) → `#topic-opinion-belief`. Exception: when the sense is clearly school/study/research, use `#topic-learning-education` (`aprender`, `ensinar`, `estudar`, `pesquisar`).
- **Frequency / time adverbs** — `always`, `never`, `usually`, `often`, `sometimes`, `today`, `tomorrow`, `yesterday`, `generally` (BP: `sempre`, `nunca`, `geralmente`, `frequentemente`, `às vezes`, `hoje`, `amanhã`, `ontem`) → `#topic-time-calendar`. Don't route to `#topic-daily-routines` just because the example happens to involve a daily activity.
- **Determiner-like adjectives and quantifiers** — `other`, `another`, `own`, `several`, `every`, `any`, `some`, `each` (BP: `outro`, `próprio`, `vários`, `todo`, `cada`, `algum`, `qualquer`) → `#topic-grammar` if the row behaves grammatically; `#topic-measurement` only when the sense is clearly quantity-related (`vários` = several quantity, `bastante`/`pouco`/`muito` when used as quantifiers).
- Generic verbs with no clearer topic (`fazer`, `ter`, `usar`, `abrir`, `fechar`) → `#topic-daily-routines`
- Generic adjectives / qualities with no clearer topic (`bom`, `mau`, `fácil`, `difícil`, `possível`, `real`, `importante`, `geral`) → `#topic-character-qualities`
- Visual appearance adjectives (`bonito`, `feio`, `alto`, `baixo`, `colorido`) → `#topic-appearance`
- Emotion words (`triste`, `feliz`, `nervoso`) → `#topic-feelings`
- Movement / location words (`entrar`, `sair`, `perto`, `longe`) → `#topic-movement-position`
- Food / drink nouns → `#topic-food-drink`
- Animals → `#topic-animals`

## Rules

1. Do not invent topic tags. Pick from the allowed list only.
2. Do not assign multiple topic tags.
3. `#topic-grammar` for function words: articles, pronouns, prepositions, conjunctions, particles, auxiliaries.
4. `#topic-daily-routines` only for everyday actions that do not fit a clearer topic. Not the default for all verbs.
5. `#topic-objects-tools` only for physical things that do not fit a clearer topic. Not the default for all nouns.
6. `#topic-work-jobs` only for jobs, professions, workplace, tasks, meetings, and professional contexts.
7. Do not classify abstract words as `#topic-work-jobs`, `#topic-economy`, or `#topic-money` unless that's the specific sense.
8. Numerals (`pos==num`) → `#topic-numbers`. Measurement units → `#topic-measurement`. Do not classify measurement words as `#topic-numbers` unless `pos==num`.
9. Generic adjectives or qualities with no clearer topic → `#topic-character-qualities`.
10. If uncertain, choose the broadest honest topic and set `confidence` to `medium` or `low`.

## Allowed topic tags

```
#topic-grammar
#topic-person-identity
#topic-family-relations
#topic-character-qualities
#topic-feelings
#topic-body
#topic-health-care
#topic-hygiene
#topic-appearance
#topic-food-drink
#topic-home-household
#topic-daily-routines
#topic-clothing-style
#topic-shopping-services
#topic-work-jobs
#topic-money
#topic-economy
#topic-law-rules
#topic-government-admin
#topic-politics
#topic-social-life
#topic-social-problems
#topic-speech-language
#topic-opinion-belief
#topic-learning-education
#topic-science
#topic-media-technology
#topic-culture-art
#topic-religion-belief
#topic-time-calendar
#topic-numbers
#topic-measurement
#topic-colors
#topic-shapes
#topic-sound
#topic-movement-position
#topic-transport
#topic-travel
#topic-city-places
#topic-countryside
#topic-countries-languages
#topic-nature-landscape
#topic-plants
#topic-animals
#topic-weather
#topic-environment
#topic-materials-substances
#topic-objects-tools
#topic-sport-leisure
#topic-danger-disaster
```

## Confidence

| confidence | use for |
|---|---|
| `high`   | Sense fits a clear topic; no ambiguity. |
| `medium` | Plausible match but example setting or polysemy makes it less certain. |
| `low`    | Genuinely uncertain — the row may need manual review. |

## Examples

### Example 1 — function word

Input:
```
pt: porque
pos: conj
en_primary: because
example_pt:
example_en:
```
Output:
```json
{"topic_primary": "#topic-grammar", "confidence": "high", "reason": "Conjunction/function word."}
```

### Example 2 — family member; example setting is incidental

Input:
```
pt: mãe
pos: noun
en_primary: mother
example_pt: Vou ajudar minha mãe na cozinha hoje.
example_en: I'm going to help my mother in the kitchen today.
```
Output:
```json
{"topic_primary": "#topic-family-relations", "confidence": "high", "reason": "Family member; kitchen is incidental context."}
```

### Example 3 — food/drink; example mentions work

Input:
```
pt: café
pos: noun
en_primary: coffee
example_pt: Vou tomar um café antes do trabalho.
example_en: I'm going to have a coffee before work.
```
Output:
```json
{"topic_primary": "#topic-food-drink", "confidence": "high", "reason": "Coffee is food/drink; work is incidental context."}
```

### Example 4 — public transport

Input:
```
pt: ônibus
pos: noun
en_primary: bus
example_pt: O ônibus para na esquina.
example_en: The bus stops at the corner.
```
Output:
```json
{"topic_primary": "#topic-transport", "confidence": "high", "reason": "Public transport vehicle."}
```

### Example 5 — legal/contract vocabulary

Input:
```
pt: contrato
pos: noun
en_primary: contract
example_pt: Preciso assinar o contrato antes de amanhã.
example_en: I need to sign the contract before tomorrow.
```
Output:
```json
{"topic_primary": "#topic-law-rules", "confidence": "high", "reason": "Contract/legal obligation vocabulary."}
```

### Example 6 — workplace activity

Input:
```
pt: reunião
pos: noun
en_primary: meeting
example_pt: Vou marcar uma reunião para segunda-feira.
example_en: I'm going to schedule a meeting for Monday.
```
Output:
```json
{"topic_primary": "#topic-work-jobs", "confidence": "high", "reason": "Meeting/workplace vocabulary."}
```

### Example 7 — money

Input:
```
pt: dinheiro
pos: noun
en_primary: money
example_pt:
example_en:
```
Output:
```json
{"topic_primary": "#topic-money", "confidence": "high", "reason": "Money vocabulary."}
```

### Example 8 — shopping

Input:
```
pt: mercado
pos: noun
en_primary: market
example_pt: Vou ao mercado comprar frutas.
example_en: I'm going to the market to buy fruit.
```
Output:
```json
{"topic_primary": "#topic-shopping-services", "confidence": "high", "reason": "Store/shopping context."}
```

### Example 9 — learning verb

Input:
```
pt: aprender
pos: verb
en_primary: learn
example_pt: Eu aprendo português todos os dias.
example_en: I learn Portuguese every day.
```
Output:
```json
{"topic_primary": "#topic-learning-education", "confidence": "high", "reason": "Learning/study vocabulary."}
```

### Example 10 — judgment / opinion

Input:
```
pt: decidir
pos: verb
en_primary: decide
example_pt: Preciso decidir hoje.
example_en: I need to decide today.
```
Output:
```json
{"topic_primary": "#topic-opinion-belief", "confidence": "medium", "reason": "Decision-making is closest to judgment/opinion."}
```

### Example 11 — movement

Input:
```
pt: entrar
pos: verb
en_primary: enter
example_pt: Posso entrar na sala agora?
example_en: Can I enter the room now?
```
Output:
```json
{"topic_primary": "#topic-movement-position", "confidence": "high", "reason": "Movement into a place."}
```

### Example 12 — spatial relation

Input:
```
pt: perto
pos: adv
en_primary: near
example_pt: A farmácia fica perto do banco.
example_en: The pharmacy is near the bank.
```
Output:
```json
{"topic_primary": "#topic-movement-position", "confidence": "high", "reason": "Spatial relation/location."}
```

### Example 13 — visual appearance adjective

Input:
```
pt: bonito
pos: adj
en_primary: beautiful
example_pt:
example_en:
```
Output:
```json
{"topic_primary": "#topic-appearance", "confidence": "high", "reason": "Visual appearance."}
```

### Example 14 — animal

Input:
```
pt: gato
pos: noun
en_primary: cat
example_pt:
example_en:
```
Output:
```json
{"topic_primary": "#topic-animals", "confidence": "high", "reason": "Animal vocabulary."}
```

### Example 15 — measurement unit (NOT numbers)

Input:
```
pt: quilo
pos: noun
en_primary: kilogram
example_pt:
example_en:
```
Output:
```json
{"topic_primary": "#topic-measurement", "confidence": "high", "reason": "Unit of measurement; numerals go to numbers, units go to measurement."}
```

### Example 16 — generic adjective fallback

Input:
```
pt: bom
pos: adj
en_primary: good
example_pt: Esse café é bom.
example_en: This coffee is good.
```
Output:
```json
{"topic_primary": "#topic-character-qualities", "confidence": "high", "reason": "Generic-quality adjective; coffee is incidental, the lemma itself has no clearer topic."}
```

### Example 17 — cognition verb (NOT daily-routines)

Input:
```
pt: lembrar
pos: verb
en_primary: to remember
example_pt: Lembrei do nome dela quando ela entrou.
example_en: I remembered her name when she came in.
```
Output:
```json
{"topic_primary": "#topic-opinion-belief", "confidence": "high", "reason": "Cognition verb (remember); cognition verbs/nouns default to opinion-belief unless the sense is study/research."}
```

### Example 18 — frequency adverb (NOT daily-routines)

Input:
```
pt: geralmente
pos: adv
en_primary: generally
example_pt: Geralmente tomo café da manhã às oito.
example_en: I generally have breakfast at eight.
```
Output:
```json
{"topic_primary": "#topic-time-calendar", "confidence": "high", "reason": "Frequency adverb; the breakfast example is incidental — frequency/time adverbs default to time-calendar."}
```

### Example 19 — determiner-like adjective

Input:
```
pt: próprio
pos: adj
en_primary: own
example_pt: Ele mora no seu próprio apartamento.
example_en: He lives in his own apartment.
```
Output:
```json
{"topic_primary": "#topic-grammar", "confidence": "high", "reason": "Determiner-like adjective expressing possession; behaves grammatically rather than naming a quality."}
```

## Output schema

Return only JSON:

```json
{
  "topic_primary": "#topic-...",
  "confidence":    "high | medium | low",
  "reason":        "string (one short sentence)"
}
```

`topic_primary` MUST be one of the 50 strings in the allowed list. No other
values are accepted by the tool schema.
