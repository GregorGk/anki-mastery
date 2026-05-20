"""Stage 18 — sentence-first Anki card templates + shared CSS (v2).

Two card types, both built on the FULL example sentence:

  listening  hear the BP sentence → understand its meaning
  recall     hear the ENGLISH sentence → produce the whole BP sentence
             (shadowing / sentence production)

Listening front plays the Portuguese sentence (`audio_example`). Recall
front shows + plays the ENGLISH sentence (`example_en` + `audio_en_example`)
— no Portuguese, no IPA (would leak the answer). The shared back replays
the Portuguese sentence, reveals pt/en + sentence IPA, then the target
word + gloss.

Word audio is CLICK-ONLY: rendered as a native HTML5 `<audio controls
preload="none" src="{{audio_word_file}}">` (a bare basename), never a
`[sound:…]` tag — so it never autoplays. `audio_example` /
`audio_en_example` stay `[sound:…]` (autoplay where placed).

Field names match `NOTE_FIELDS` in `anki_models.py`. `{{#field}}…{{/field}}`
conditionals hide empty optionals. Every back carries the native-flag
footer. NO custom JS.

Imported by `build/lib/anki_models.py` and `tests/test_stage_18_apkg.py`.
"""
from __future__ import annotations

FOOTER = (
    '<div class="footer">\n'
    '  Issue? Flag this card red. Sense: {{sense_id}}\n'
    '</div>'
)

_OPTIONAL_BLOCKS = (
    '{{#usage_hint}}\n'
    '<div class="usage-hint">{{usage_hint}}</div>\n'
    '{{/usage_hint}}\n'
    '\n'
    '{{#risk_note}}\n'
    '<div class="risk-note">⚠ {{risk_note}}</div>\n'
    '{{/risk_note}}'
)

# Click-only word audio: native HTML5 element, bare-basename src, no autoplay.
_WORD_AUDIO = (
    '<div class="word-audio">\n'
    '  <div class="small">Word audio, click only</div>\n'
    '  <audio controls preload="none" src="{{audio_word_file}}"></audio>\n'
    '</div>'
)

# Identical answer side for both card types.
_SHARED_BACK = (
    '<div class="prompt small">Replay Portuguese sentence</div>\n'
    '<div class="audio-block">{{audio_example}}</div>\n'
    '\n'
    '<div class="sentence-pt">{{example_pt}}</div>\n'
    '<div class="sentence-en">{{example_en}}</div>\n'
    '\n'
    '<details class="ipa-details" open>\n'
    '  <summary>IPA sentence</summary>\n'
    '  <div class="ipa-example">{{ipa_example}}</div>\n'
    '</details>\n'
    '\n'
    '<hr>\n'
    '\n'
    '<div class="target">{{pt_display_safe}}</div>\n'
    '<div class="gloss">{{en_primary}}</div>\n'
    '\n'
    + _WORD_AUDIO + '\n'
    '\n'
    + _OPTIONAL_BLOCKS + '\n'
    '\n'
    + FOOTER
)

# card_type -> {"name", "qfmt", "afmt"}
TEMPLATES: dict[str, dict[str, str]] = {
    # 1. Sentence Listening — hear the PT sentence, understand the meaning.
    "listening": {
        "name": "Sentence Listening",
        "qfmt": (
            '<div class="prompt">Listen to the Portuguese sentence.</div>\n'
            '\n'
            '<div class="audio-block">{{audio_example}}</div>\n'
            '\n'
            '<details class="ipa-details">\n'
            '  <summary>IPA sentence</summary>\n'
            '  <div class="ipa-example">{{ipa_example}}</div>\n'
            '</details>'
        ),
        "afmt": _SHARED_BACK,
    },
    # 2. Sentence Recall — hear/read the ENGLISH sentence, produce the BP one.
    "recall": {
        "name": "Sentence Recall",
        "qfmt": (
            '<div class="recall-front">\n'
            '  <div class="prompt">Recall the whole Portuguese sentence.</div>\n'
            '\n'
            '  <div class="sentence-en-front">{{example_en}}</div>\n'
            '\n'
            '  <div class="audio-block">{{audio_en_example}}</div>\n'
            '\n'
            '  <div class="subprompt">Say it aloud, or write it before revealing.</div>\n'
            '</div>'
        ),
        "afmt": _SHARED_BACK,
    },
}


CARD_CSS = """
.card {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 22px;
  text-align: center;
  line-height: 1.35;
  color: #222;
  background: #fff;
}
.prompt {
  font-size: 15px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  opacity: 0.55;
  margin-bottom: 0.6rem;
}
.prompt.small, .small { font-size: 12px; margin-bottom: 0.3rem; }
.subprompt { font-size: 14px; opacity: 0.6; margin-top: 0.8rem; }
.audio-block { margin: 0.6rem 0; }
.recall-front { padding: 0.4rem 0; }
.sentence-en-front {
  font-size: 26px;
  font-weight: 600;
  margin: 0.8rem auto;
  max-width: 36ch;
}
.sentence-pt {
  font-size: 28px;
  font-weight: 600;
  margin: 0.8rem 0;
}
.sentence-en {
  font-size: 20px;
  opacity: 0.75;
}
details.ipa-details {
  margin: 0.7rem auto;
  max-width: 40ch;
  text-align: left;
}
details.ipa-details > summary {
  cursor: pointer;
  font-size: 13px;
  opacity: 0.6;
  text-align: center;
}
.ipa-example {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 18px;
  opacity: 0.85;
  margin-top: 0.4rem;
  text-align: center;
}
.target {
  font-size: 30px;
  font-weight: 700;
}
.gloss {
  font-size: 24px;
  margin-top: 0.4rem;
}
.word-audio { margin: 0.6rem 0; }
.word-audio audio { vertical-align: middle; }
.usage-hint {
  margin-top: 0.8rem;
  font-style: italic;
  opacity: 0.8;
}
.risk-note {
  margin-top: 0.8rem;
  font-weight: 700;
}
.footer {
  margin-top: 1.2rem;
  font-size: 13px;
  opacity: 0.45;
}
hr { border: none; border-top: 1px solid #ddd; margin: 0.9rem 0; }
.nightMode.card { color: #e6e9ef; background: #1b1b1b; }
.nightMode hr { border-top-color: #444; }
"""
