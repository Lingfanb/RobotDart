"""Decode BONES animator-style codes (e.g. 'jog ff stop 180') into natural
BABEL-style English (e.g. 'stop jogging backward').

Background:
    BONES `content_short_description_2` uses mocap industry shorthand:
        ff      = "forward-facing" starting orientation (drop in output)
        000-360 = direction of motion in degrees (forward / right / back / ...)
        start / loop / stop  = gait cycle phase
        opt N   = variant index (drop)
    Examples:
        'jog ff stop 180'        → 'stop jogging backward'
        'jump ff 360'            → 'jump in a full turn'
        'walk ff loop 090'       → 'walk right'
        'injured leg jog ff start 180'
                                 → 'start jogging backward with injured leg'

Coverage on BONES (142k clips):
    ~42% contain 3-digit direction code → benefit directly
    ~28% contain 'ff' → benefit directly
    ~58% are already plain English → pass through

For unrecognized strings (animator quirks), the decoder returns the input
unchanged. Optional LLM fallback can be plugged in later.
"""
from __future__ import annotations

import re
from typing import Optional


# ════════════════════════════════════════════════════════════════
# Vocabulary
# ════════════════════════════════════════════════════════════════

# Direction in degrees. 0/360 are both "no direction change" but 360 implies a full turn.
DIRECTION_MAP: dict[str, str] = {
    '000': 'forward',
    '045': 'forward right',
    '090': 'right',
    '135': 'back right',
    '180': 'backward',
    '225': 'back left',
    '270': 'left',
    '315': 'forward left',
    '360': 'in a full turn',   # special — full 360° rotation
}

PHASE_VERB: dict[str, str] = {
    'start': 'start',
    'stop':  'stop',
    'loop':  '',          # sustained motion — drop the phase word
}

# Action verbs we know how to gerund-ize. Maps base form → -ing form.
ACTION_GERUND: dict[str, str] = {
    'walk':   'walking',
    'jog':    'jogging',
    'run':    'running',
    'jump':   'jump',          # jump stays "jump" (verb noun)
    'crawl':  'crawling',
    'crouch': 'crouching',
    'climb':  'climbing',
    'turn':   'turning',
    'kneel':  'kneeling',
    'kick':   'kicking',
    'punch':  'punching',
    # 'reach' kept verbatim — BONES uses 'reach jump' as a clip-name noun
    'stretch': 'stretching',
    'lift':    'lifting',
    'dance':   'dancing',
    'sit':     'sitting',
    'stand':   'standing',
    'idle':    'idle',         # already noun-y
}

# Prefix qualifiers — recognized at the start of the string.
PREFIX_QUALIFIERS: list[tuple[str, str]] = [
    ('injured torso',     'with injured torso'),
    ('injured leg',       'with injured leg'),
    ('injured r leg',     'with injured right leg'),
    ('injured l leg',     'with injured left leg'),
    ('injured arm',       'with injured arm'),
    ('hurry',             'in a hurry'),
    ('old',               'with old gait'),
    ('lift crate',        'carrying a crate'),
    ('inj torso',         'with injured torso'),
    ('inj leg',           'with injured leg'),
    ('inj right leg',     'with injured right leg'),
    ('inj left leg',      'with injured left leg'),
]


# ════════════════════════════════════════════════════════════════
# Cleaning helpers
# ════════════════════════════════════════════════════════════════

# Drop these tokens unconditionally (anywhere in the string).
DROP_TOKENS = {
    'ff',          # forward-facing marker
    'r', 'l',      # leftover orientation suffix from filename
    'm',           # mirror suffix
}

# Patterns to strip via regex (run before tokenization).
# Caution: do NOT strip 3-digit numbers — those are direction codes (000-360).
_STRIP_PATTERNS = [
    re.compile(r'\bopt\s+\d+\b'),                    # 'opt 1', 'opt 2'
    re.compile(r'\bv\d+\b'),                          # 'v001'
    re.compile(r'(?<!\d)(\d{1,2})(?!\d)\s*$'),        # trailing 1-2 digit sequence num
    re.compile(r'(?<!\d)(\d{4,})(?!\d)'),             # 4+ digit numbers anywhere
]


def _normalize_whitespace(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()


def _strip_qualifier_prefix(s: str) -> tuple[str, Optional[str]]:
    """Detect and remove a prefix qualifier (injured, hurry, etc.).

    Returns (rest_of_string, qualifier_phrase_or_None).
    """
    for prefix_pat, qualifier in PREFIX_QUALIFIERS:
        if s.startswith(prefix_pat + ' ') or s == prefix_pat:
            return s[len(prefix_pat):].strip(), qualifier
    return s, None


def _extract_direction(tokens: list[str]) -> tuple[list[str], Optional[str]]:
    """Pull out a 3-digit direction token if present. Returns (tokens_left, direction_phrase)."""
    for i, t in enumerate(tokens):
        if t in DIRECTION_MAP:
            phrase = DIRECTION_MAP[t]
            return tokens[:i] + tokens[i + 1:], phrase
    return tokens, None


def _extract_phase(tokens: list[str]) -> tuple[list[str], Optional[str]]:
    """Pull out start/loop/stop. Returns (tokens_left, phase_verb)."""
    for i, t in enumerate(tokens):
        if t in PHASE_VERB:
            return tokens[:i] + tokens[i + 1:], PHASE_VERB[t]
    return tokens, None


def _drop_clutter(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in DROP_TOKENS and not t.isdigit()]


# ════════════════════════════════════════════════════════════════
# Main API
# ════════════════════════════════════════════════════════════════

def decode_animator_short(text: str) -> str:
    """Convert BONES animator-style short_2 to natural English.

    Strategy:
        1. Lowercase, normalize whitespace, strip variant tags ('opt 2', 'v001').
        2. Detect optional prefix qualifier ('injured leg', 'lift crate', ...).
        3. Pull out direction (000-360) and phase (start/loop/stop) tokens.
        4. The remaining tokens form the action; gerund-ize the first verb.
        5. Re-assemble: [phase] + [action gerund] + [direction] + [qualifier].

    Falls back to returning the cleaned-but-unparsed string if no animator
    structure is detected (so plain English already passes through).
    """
    if not isinstance(text, str) or not text.strip():
        return ''

    s = text.lower().strip()

    # Strip variant tags
    for pat in _STRIP_PATTERNS:
        s = pat.sub(' ', s)
    s = _normalize_whitespace(s)
    if not s:
        return ''

    # Detect / strip prefix qualifier
    s, qualifier = _strip_qualifier_prefix(s)

    tokens = s.split()
    if not tokens:
        # Whole string was a qualifier
        return qualifier or ''

    # Pull direction + phase out
    tokens, direction_phrase = _extract_direction(tokens)
    tokens, phase_verb = _extract_phase(tokens)
    tokens = _drop_clutter(tokens)

    if not tokens:
        # Edge case: only had ff + direction + phase
        action_phrase = 'move'
    else:
        # First token is typically the verb. Gerund-ize it if recognized.
        first = tokens[0]
        if first in ACTION_GERUND:
            tokens[0] = ACTION_GERUND[first]
        action_phrase = ' '.join(tokens)

    # Assemble
    parts: list[str] = []
    if phase_verb:
        parts.append(phase_verb)
    parts.append(action_phrase)
    if direction_phrase:
        parts.append(direction_phrase)
    if qualifier:
        parts.append(qualifier)

    out = _normalize_whitespace(' '.join(parts))
    return out


# ════════════════════════════════════════════════════════════════
# Self-test
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    test_cases = [
        # ── Direct animator codes (top frequency) ──
        ('jump ff 180',                      'jump backward'),
        ('jump ff 360',                      'jump in a full turn'),
        ('jump ff 000',                      'jump forward'),
        ('jog ff stop 180',                  'stop jogging backward'),
        ('jog ff start 180',                 'start jogging backward'),
        ('jog ff loop 180',                  'jogging backward'),
        ('walk ff loop 090',                 'walking right'),
        ('walk ff stop 270',                 'stop walking left'),

        # ── Compound prefixes ──
        ('injured leg jog ff loop 180',      'jogging backward with injured leg'),
        ('injured torso walk ff start 045',  'start walking forward right with injured torso'),
        ('inj right leg jog ff stop 180',    'stop jogging backward with injured right leg'),
        ('lift crate walk ff loop 180',      'walking backward carrying a crate'),
        ('hurry jog ff loop 000',            'jogging forward in a hurry'),

        # ── Variants/clutter ──
        ('dancing routine opt 2',            'dancing routine'),
        ('high jump opt 1',                  'high jump'),
        ('body stretch 4',                   'body stretch'),
        ('crouch ff loop 000',               'crouching forward'),

        # ── Already natural (should pass through) ──
        ('body check',                       'body check'),
        ('salute',                           'salute'),
        ('looking around horizontally',      'looking around horizontally'),
        ('idle loop',                        'idle'),
        ('reach jump',                       'reach jump'),
        ('crowd cheer',                      'crowd cheer'),

        # ── Edge cases ──
        ('',                                 ''),
        ('idle',                             'idle'),
    ]

    print(f'{"input":50s} → {"decoded":50s}  (expected)')
    print('─' * 130)
    pass_n = fail_n = 0
    for inp, expected in test_cases:
        out = decode_animator_short(inp)
        ok = out == expected
        if ok:
            pass_n += 1
            mark = '✓'
        else:
            fail_n += 1
            mark = '✗'
        print(f'  {mark} {inp!r:50s} → {out!r:50s}  ({expected!r})')
    print(f'\n{pass_n}/{pass_n + fail_n} pass')
