"""Stage 17 — deterministic ordering rules: topic order + topic-internal buckets.

Single-sources the 50-topic pedagogical order from
`topic_tag_rules.ALLOWED_TOPIC_TAGS`. Defines hand-curated semantic
buckets for the 5 custom-ordered topics (grammar, numbers,
time-calendar, measurement, movement-position). Everything here is pure
data + pure functions — no I/O, no sqlite. `build/17_0_build_ordering.py`
is the only caller.

Bucket model
------------
Each custom topic maps to an ordered tuple of buckets. A bucket is
`(bucket_id: int, label: str, lemmas: tuple[str, ...])`:
  - `bucket_id` is curator-assigned, stable, unique within a topic, < 999.
  - The tuple ORDER of buckets == teaching order.
  - `lemmas` are matched against `senses.pt` (the bare lemma). Their
    position in the tuple is the within-bucket `topic_subrank`.
  - An empty `lemmas` tuple means the bucket is filled by a POS rule
    instead (only `#topic-grammar` uses this — art/pron/prep/conj/adv/
    verb/noun are POS-derivable).

A sense whose lemma matches no bucket (and no POS rule) falls to
`FALLBACK_BUCKET_ID` (999) with `topic_subrank = sense_id-as-int`, i.e.
it keeps original sense_id order. Non-custom topics are entirely 999.
"""
from __future__ import annotations

import re

from build.lib.topic_tag_rules import ALLOWED_TOPIC_TAGS, ALLOWED_TOPIC_TAG_SET

# --------------------------------------------------------------------------- #
# Topic order — single-sourced from the frozen Stage 14 taxonomy
# --------------------------------------------------------------------------- #
TOPIC_INDEX: dict[str, int] = {tag: i for i, tag in enumerate(ALLOWED_TOPIC_TAGS)}

FALLBACK_BUCKET_ID = 999

# --------------------------------------------------------------------------- #
# Custom-topic semantic buckets
# --------------------------------------------------------------------------- #
Bucket = tuple[int, str, tuple[str, ...]]

CUSTOM_TOPIC_BUCKETS: dict[str, tuple[Bucket, ...]] = {
    # ---- #topic-grammar -------------------------------------------------- #
    # Buckets 0/1/3/4/5 are POS-derived (empty lemma tuple) — see
    # GRAMMAR_POS_BUCKET. Buckets 2 and 6 are curated lemma lists.
    "#topic-grammar": (
        (0, "articles", ()),                       # pos == art
        (1, "pronouns", ()),                       # pos == pron
        (2, "determiner-like adjectives", (
            "outro", "todo", "mesmo", "próprio", "cada", "qualquer",
            "tal", "único", "ambos", "diverso", "restante", "dado",
        )),
        (3, "prepositions", ()),                   # pos == prep
        (4, "conjunctions", ()),                   # pos == conj
        (5, "question / negation / discourse adverbs", ()),  # pos == adv
        (6, "grammar terms", (
            "verbo", "adjetivo", "oração", "complemento", "singular",
            "masculino", "irregular", "porquê", "sujeito", "referente",
            "referido", "respectivo", "pertencente", "prestes",
        )),
    ),
    # ---- #topic-numbers -------------------------------------------------- #
    "#topic-numbers": (
        (0, "small cardinals (0-10)", (
            "zero", "dois", "três", "quatro", "cinco", "seis", "sete",
            "oito", "nove", "dez",
        )),
        (1, "teens, tens & hundreds", (
            "onze", "doze", "treze", "catorze", "quinze", "dezoito",
            "vinte", "trinta", "quarenta", "cinquenta", "sessenta",
            "setenta", "oitenta", "noventa", "cem", "duzentos",
            "trezentos", "quinhentos",
        )),
        (2, "thousands, millions & up", (
            "mil", "milhão", "bilhão",
        )),
        (3, "ordinals", (
            "primeiro", "segundo", "terceiro", "quarto", "quinto",
            "sexto", "sétimo", "oitavo", "décimo",
        )),
        (4, "fractions", (
            "terço", "fração",
        )),
        (5, "numeric concepts & operations", (
            "número", "contar", "contagem", "soma", "somar", "divisão",
            "múltiplo", "primo", "sequência", "dobrar", "minoria",
            "dúzia", "dezena", "centena", "milhar",
        )),
    ),
    # ---- #topic-time-calendar ------------------------------------------- #
    "#topic-time-calendar": (
        (0, "now / today / yesterday / tomorrow", (
            "agora", "hoje", "ontem", "amanhã", "já", "ainda", "logo",
            "cedo", "ora", "atualmente",
        )),
        (1, "parts of the day", (
            "manhã", "tarde", "noite", "madrugada", "meio-dia",
            "meia-noite",
        )),
        (2, "core time units", (
            "minuto", "hora", "dia", "semana", "mês", "ano", "década",
            "século", "data",
        )),
        (3, "calendar words", (
            "domingo", "segunda-feira", "terça-feira", "quarta-feira",
            "quinta-feira", "sexta-feira", "sábado", "verão", "outono",
            "estação", "calendário", "horário", "relógio",
        )),
        (4, "frequency", (
            "sempre", "nunca", "jamais", "geralmente", "normalmente",
            "frequentemente", "raramente", "constantemente",
            "habitualmente", "regularmente", "diariamente", "anualmente",
            "dificilmente", "ultimamente", "costumar", "regular",
            "diário", "anual", "mensal", "semanal", "periódico",
        )),
        (5, "sequence", (
            "antes", "depois", "anterior", "posterior", "seguinte",
            "próximo", "último", "anteriormente", "posteriormente",
            "previamente", "recentemente", "novamente", "finalmente",
            "afinal", "enfim", "atrás", "seguida", "em seguida",
            "suceder", "anteceder", "preceder", "decorrer",
            "consecutivo", "sucessivo", "sucessão", "simultaneamente",
            "simultâneo", "paralelamente", "coincidir", "inicialmente",
            "imediatamente", "subitamente", "repente", "de repente",
            "precisamente", "progressivamente", "eventualmente",
        )),
        (6, "beginning / end / deadline", (
            "início", "começo", "princípio", "fim", "final", "prazo",
            "vencimento", "durar", "demorar", "romper", "findar",
            "antecipar", "espera", "intervalo",
        )),
        (7, "age / era / past & future", (
            "antigo", "atual", "futuro", "passado", "presente",
            "recente", "época", "altura", "período", "momento",
            "instante", "vez", "tempo", "diante", "em diante",
            "antigamente", "outrora", "atualidade", "meado", "temporal",
            "oportuno", "atrasado", "marcado", "véspera", "modificação",
            "alteração", "resolução", "passar",
        )),
    ),
    # ---- #topic-measurement --------------------------------------------- #
    "#topic-measurement": (
        (0, "quantity & degree", (
            "mais", "muito", "pouco", "menos", "tanto", "quanto",
            "quase", "bastante", "cerca", "demais", "demasiado",
            "suficiente", "completamente", "exatamente", "praticamente",
            "relativamente", "ligeiramente", "aproximadamente",
            "parcialmente", "altamente", "suficientemente", "vários",
            "inúmero", "numeroso", "escasso", "considerável",
        )),
        (1, "size & extent", (
            "maior", "menor", "longo", "altura", "espaço", "área",
            "tamanho", "distância", "dimensão", "extensão",
            "profundidade", "comprimento", "largura", "altitude",
            "alcance", "porte", "grandeza", "nível", "elevado",
            "pesado", "traço",
        )),
        (2, "units", (
            "metro", "quilômetro", "centímetro", "milímetro", "quilo",
            "grama", "tonelada", "litro", "hectare", "libra", "grau",
            "unidade",
        )),
        (3, "percent, fraction & proportion", (
            "cento", "por cento", "percentagem", "metade", "meio",
            "média", "proporção", "proporcional", "fração", "dobro",
            "duplo", "par", "totalidade",
        )),
        (4, "increase / decrease / variation", (
            "aumentar", "diminuir", "diminuição", "variar", "variação",
            "oscilar", "rondar", "acrescido", "acréscimo", "descida",
        )),
        (5, "scale, limit, capacity & measuring", (
            "máximo", "limite", "capacidade", "escala", "padrão",
            "peso", "pesar", "medir", "bastar", "precisão", "estimativa",
            "balança", "quantidade", "maioria", "resto", "frequência",
            "excesso", "excessivo", "intensidade", "velocidade",
            "rapidez", "temperatura", "duração", "equivalente", "teor",
            "amplitude", "monte",
        )),
    ),
    # ---- #topic-movement-position --------------------------------------- #
    "#topic-movement-position": (
        (0, "static location", (
            "aqui", "aí", "ali", "lá", "cá", "perto", "longe",
            "distante", "próximo", "lugar", "local", "lado", "frente",
            "fundo", "cima", "trás", "beira", "topo", "meio", "posição",
            "direção", "caminho", "passo", "esquerda", "esquerdo",
            "sul", "leste", "sudoeste", "cardeal", "proximidade",
            "redor", "em redor", "ao redor", "adiante", "longínquo",
            "afastado", "situado", "localizado", "tona", "à tona",
        )),
        (1, "spatial relations", (
            "dentro", "acima", "abaixo", "embaixo", "atrás", "inferior",
            "lateral", "oposto", "subterrâneo",
        )),
        (2, "core movement verbs", (
            "ir", "vir", "chegar", "partir", "sair", "entrar", "voltar",
            "regressar", "retornar", "passar", "andar", "caminhar",
            "marchar", "seguir", "avançar", "recuar", "subir", "descer",
            "ascender", "cair", "baixar", "levantar", "fugir", "escapar",
            "atravessar", "cruzar", "percorrer", "penetrar", "virar",
            "girar", "rodar", "rolar", "saltar", "pular", "deslizar",
            "flutuar", "pairar", "mover", "movimentar", "deslocar",
            "mexer", "sacudir", "pisar", "marcha", "deixar", "parar",
            "arrancar", "desviar", "apressar",
        )),
        (3, "placement & manipulation verbs", (
            "pôr", "colocar", "botar", "meter", "enfiar", "inserir",
            "encaixar", "tirar", "retirar", "remover", "levar",
            "buscar", "carregar", "puxar", "empurrar", "arrastar",
            "largar", "agarrar", "segurar", "juntar", "sentar",
            "erguer", "estender", "alongar", "alargar", "inclinar",
            "debruçar", "curvar", "encostar", "alinhar", "esconder",
            "bloquear", "travar", "paralisar",
        )),
        (4, "appearance, reach & abstract movement", (
            "aparecer", "surgir", "desaparecer", "sumir", "emergir",
            "originar", "provir", "proveniente", "atingir", "alcançar",
            "ocupar", "caber", "afastar", "aproximar", "aproximação",
            "distanciar", "conduzir", "guiar", "atrair", "perseguir",
            "rondar", "cercar", "rodear", "contornar", "ultrapassar",
            "prosseguir", "proceder", "adiantar", "assentar", "residir",
            "situar", "localizar", "centrar", "pousar", "afundar",
            "dispersar", "deparar", "avistar", "espreitar", "olhar",
            "ver", "esperar", "mudar", "ficar", "tornar-se",
            "permanecer", "restar", "lançar", "queda", "subida",
            "vinda", "retorno", "movimento", "movimentação",
            "deslocamento", "obstáculo", "faixa", "rapidamente",
            "lentamente", "depressa", "devagar", "livremente",
            "imóvel", "móvel", "disposto", "perdido", "liberado",
            "sentado", "parado", "virado", "voltado", "escondido",
            "encostado", "carregado", "colocado", "pendurado",
            "acessível", "inverter", "desdobrar", "suportar",
        )),
    ),
}

CUSTOM_TOPICS: frozenset[str] = frozenset(CUSTOM_TOPIC_BUCKETS)

# #topic-grammar POS rule — applied when a grammar lemma matches no curated
# bucket list. Keyed on senses.pos.
GRAMMAR_POS_BUCKET: dict[str, int] = {
    "art": 0, "pron": 1, "prep": 3, "conj": 4, "adv": 5,
    "verb": 6, "noun": 6, "num": 6,
}

# lemma -> (bucket_id, index_within_bucket), built once per custom topic.
_LEMMA_INDEX: dict[str, dict[str, tuple[int, int]]] = {}
for _topic, _buckets in CUSTOM_TOPIC_BUCKETS.items():
    _m: dict[str, tuple[int, int]] = {}
    for _bid, _label, _lemmas in _buckets:
        for _i, _lemma in enumerate(_lemmas):
            _m[_lemma] = (_bid, _i)
    _LEMMA_INDEX[_topic] = _m

# bucket_id -> label, per topic (includes the synthetic 999 row).
_BUCKET_LABELS: dict[str, dict[int, str]] = {}
for _topic, _buckets in CUSTOM_TOPIC_BUCKETS.items():
    _BUCKET_LABELS[_topic] = {_bid: _label for _bid, _label, _ in _buckets}
    _BUCKET_LABELS[_topic][FALLBACK_BUCKET_ID] = "(unbucketed — sense_id order)"


# --------------------------------------------------------------------------- #
# Public functions
# --------------------------------------------------------------------------- #
def topic_index_for(topic_primary: str) -> int:
    """0-based pedagogical index of a topic. Raises KeyError if not one of the 50."""
    return TOPIC_INDEX[topic_primary]


def is_custom_topic(topic_primary: str) -> bool:
    return topic_primary in CUSTOM_TOPICS


def subrank_from_sense_id(sense_id: str) -> int:
    """Fallback subrank: int value of RRRREESS so order == sense_id order.
    int('0001.00.01'.replace('.','')) == 10001 — monotonic in sense_id."""
    return int(sense_id.replace(".", ""))


def bucket_for(
    topic_primary: str, pt: str, pos: str, sense_id: str
) -> tuple[int, float, str]:
    """Return (topic_bucket, topic_subrank, order_reason) for one sense.

    Non-custom topic            -> (999, sense_id-as-int, "topic_default")
    Custom topic, lemma matched -> (bucket_id, idx_in_bucket, "bucket:<label>")
    #topic-grammar, pos matched -> (pos_bucket_id, sense_id-as-int, "grammar_pos:<pos>")
    Custom topic, no match      -> (999, sense_id-as-int, "custom_unmatched")

    `topic_subrank` is returned as a float because force_after overrides
    (resolved in 17_0) need sub-integer epsilon offsets.
    """
    if topic_primary not in CUSTOM_TOPICS:
        return (FALLBACK_BUCKET_ID, float(subrank_from_sense_id(sense_id)),
                "topic_default")

    hit = _LEMMA_INDEX[topic_primary].get(pt)
    if hit is not None:
        bid, idx = hit
        label = _BUCKET_LABELS[topic_primary][bid]
        return (bid, float(idx), f"bucket:{label}")

    if topic_primary == "#topic-grammar":
        gb = GRAMMAR_POS_BUCKET.get(pos)
        if gb is not None:
            return (gb, float(subrank_from_sense_id(sense_id)),
                    f"grammar_pos:{pos}")

    return (FALLBACK_BUCKET_ID, float(subrank_from_sense_id(sense_id)),
            "custom_unmatched")


def all_bucket_labels(topic_primary: str) -> list[tuple[int, str]]:
    """Ordered (bucket_id, label) for a custom topic, incl. the synthetic 999.
    Empty list for non-custom topics."""
    if topic_primary not in CUSTOM_TOPICS:
        return []
    base = [(bid, lbl) for bid, lbl, _ in CUSTOM_TOPIC_BUCKETS[topic_primary]]
    return base + [(FALLBACK_BUCKET_ID,
                    _BUCKET_LABELS[topic_primary][FALLBACK_BUCKET_ID])]


# --------------------------------------------------------------------------- #
# anki_tags — generated Anki-safe hierarchical tags
# --------------------------------------------------------------------------- #
_SLUG_RE = re.compile(r"[^a-z0-9-]+")


def _slug(value: str) -> str:
    """Lowercase, '_'→'-', strip anything not [a-z0-9-]."""
    s = value.strip().lower().replace("_", "-")
    s = _SLUG_RE.sub("-", s)
    return s.strip("-")


def anki_tags_for(
    *, topic_primary: str, pos: str, risk_flags: str, bp_validity: str, tags: str
) -> str:
    """Build the space-joined Anki hierarchical-tag string for one sense.

      topic::<slug>     from the #topic-* tag
      pos::<pos>        from senses.pos
      bp::<slug>        from bp_validity (skipped when 'standard'/empty)
      risk::<slug>      one per risk_flags entry (skipped when 'none'/empty)
      meta::<slug>      every non-#topic- raw tag (#top2000, #single-word, …)

    All emitted tags match ^[a-z]+(::[a-z0-9-]+)+$.
    """
    out: list[str] = []

    if topic_primary.startswith("#topic-"):
        out.append(f"topic::{_slug(topic_primary[len('#topic-'):])}")
    elif topic_primary:
        out.append(f"topic::{_slug(topic_primary.lstrip('#'))}")

    if pos:
        out.append(f"pos::{_slug(pos)}")

    if bp_validity and bp_validity.lower() != "standard":
        out.append(f"bp::{_slug(bp_validity)}")

    for flag in (risk_flags or "").split("|"):
        flag = flag.strip()
        if flag and flag.lower() != "none":
            out.append(f"risk::{_slug(flag)}")

    for raw in (tags or "").split():
        raw = raw.strip()
        if not raw.startswith("#") or raw.startswith("#topic-"):
            continue
        slug = _slug(raw[1:])
        if slug:
            out.append(f"meta::{slug}")

    # De-dup preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return " ".join(uniq)
