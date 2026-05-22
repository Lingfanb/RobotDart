"""Canonical action taxonomy for per-action VAD calibration.

BONES `content_type_of_movement` has 119 unique values, many of which are
composite labels like "walking, turning, climbing". For per-action μ/σ
calibration we collapse them into ~10 canonical groups whose members share
similar baseline motion vigor.

Rationale: VAD treats action category as a *given*; calibration normalizes
within-category variation. Two clips both labeled "walking" should differ in
VAD only if they actually express different affect — not because one is
"walking on a hill" and another is "walking, turning".

Order matters in `canonicalize()`: more specific patterns are checked first.
"""
from __future__ import annotations


# Canonical action classes ordered roughly by motion vigor.
# Each maps to a list of substring patterns that match raw BONES labels.
ACTION_CLASSES: list[str] = [
    'standing_idle',     # very low: ~0.001 mean_speed
    'sitting',
    'kneeling',          # on knees / on all fours
    'crawling',
    'walking',           # locomotion baseline
    'jogging',
    'jumping',           # includes flipping
    'climbing',
    'gesture',           # expressive arm-driven (no locomotion)
    'dancing',
    'action_dynamic',    # pulling/throwing/kicking — heavy whole-body
    'action_misc',       # everything else with 'action' label
    'transition',
    'other',             # fallback
]


# Substring matching priority (top-down — first match wins).
# Patterns are lowercased substring tests.
_PATTERNS: list[tuple[str, str]] = [
    ('jumping',     'jumping'),
    ('jumping',     'flipping'),
    ('jogging',     'jogging'),
    ('jogging',     'running'),
    ('climbing',    'climbing'),
    ('dancing',     'dancing'),
    ('crawling',    'crawling'),
    ('kneeling',    'on hands and knees'),
    ('kneeling',    'on all fours'),
    ('kneeling',    'kneeling'),
    ('action_dynamic', 'pulling'),
    ('action_dynamic', 'throwing'),
    ('action_dynamic', 'kicking'),
    ('walking',     'walking'),
    ('sitting',     'sitting'),
    ('standing_idle', 'standing idle'),
    ('standing_idle', 'crouching idle'),
    ('standing_idle', 'idle'),     # generic idle
    ('gesture',     'gesture'),
    ('gesture',     'pointing'),
    ('gesture',     'interacting'),
    ('gesture',     'playing instrument'),
    ('standing_idle', 'standing'),  # bare 'standing' = idle baseline
    ('action_misc', 'action'),
    ('transition',  'transition'),
]


def canonicalize(content_type_of_movement: str | None) -> str:
    """Map a raw `content_type_of_movement` string to a canonical class.

    Returns 'other' for unrecognized / empty / NaN labels.
    """
    if not content_type_of_movement or not isinstance(content_type_of_movement, str):
        return 'other'
    s = content_type_of_movement.lower().strip()
    for canonical, pattern in _PATTERNS:
        if pattern in s:
            return canonical
    return 'other'


def canonicalize_act_cats(act_cats: list[str] | None) -> str:
    """Pull `content_type_of_movement` from a primitive's act_cats and canonicalize.

    Two schemas exist:
      - Legacy bones_mp_data:  [category, content_type_of_movement, content_body_position]
        (length 3) — index 1 is content_type_of_movement.
      - New mp_data_g1_69_bones_clean: bones_npz segment_act_cat is a single
        string (length 1) — index 0 IS content_type_of_movement.

    Both index into content_type_of_movement at the right granularity for
    per-action VAD calibration. (Pre-2026-05-09 versions required len >= 2,
    which silently routed every new-schema primitive to 'other' — fixed now.)
    """
    if not act_cats:
        return 'other'
    if len(act_cats) >= 2:
        return canonicalize(act_cats[1])
    return canonicalize(act_cats[0])


# ════════════════════════════════════════════════════════════════
# 2-level hierarchy · Family → Leaf
# ════════════════════════════════════════════════════════════════
#
# For human-browseable directory layout AND fine-grained action
# disambiguation. The 14 canonical classes group into 7 families;
# expressive families (gesture, manipulation, dancing) further split
# into leaf actions via content_short_description keyword matching.

FAMILY_OF: dict[str, str] = {
    'walking':         'locomotion',
    'jogging':         'locomotion',
    'jumping':         'locomotion',
    'crawling':        'locomotion',
    'climbing':        'locomotion',
    'standing_idle':   'posture',
    'sitting':         'posture',
    'kneeling':        'posture',
    'gesture':         'gesture',
    'dancing':         'dancing',
    'action_dynamic':  'manipulation',
    'action_misc':     'manipulation',
    'transition':      'transition',
    'other':           'other',
}


import re

# Gesture leaf patterns — regex with word boundaries so 'bow' doesn't match
# 'elbow', 'pull' doesn't match 'pulling rope', etc. Order matters: more
# specific first. Patterns are matched against (short_description ' | ' natural_desc_1)
# so we catch labels like 'shrug' that only appear in long descriptions.
_GESTURE_LEAF_PATTERNS: list[tuple[str, list[str]]] = [
    # ── Social interaction (handover-relevant, ordered before generic gestures)
    ('hand_off',    [r'\bhand(s|ed|ing)?\s+(over|off)\b',
                     r'\bpass(es|ing|ed)?\s+(it|the|to)\b',
                     r'\bpass(es|ing|ed)?\s+\w+\s+to\b']),
    ('reach_out',   [r'reach(es|ing|ed)?\s+(out|toward|for)\b']),
    ('high_five',   [r'high[\s-]?five']),
    ('handshake',   [r'\bhandshake\b', r'shake[s]?\s+hand']),
    ('arms_crossed',[r'arms?\s+cross(ed)?\b', r'cross(es|ing)\s+arms?\b']),
    ('stop_gesture',[r'\bstop\s+gesture\b', r'\bhand[s]?\s+up\b\s*\(stop\)?']),
    ('shield',      [r'\bshield(s|ing|ed)?\b']),
    ('block',       [r'\bblock(s|ing|ed)?\b']),
    ('lean_against',[r'lean(s|ing|ed)?\s+against']),
    # ── Specific expressive gestures
    ('wave',        [r'\bwav(e|es|ing)\b']),
    ('clap',        [r'\bclap(s|ping)?\b', r'\bapplaus']),
    ('salute',      [r'\bsalut']),
    ('bow',         [r'\bbow(s|ing)?\b']),
    ('point',       [r'\bpoint(s|ing)?\b']),
    ('shrug',       [r'\bshrug(s|ging)?\b']),
    ('nod',         [r'\bnod(s|ding)?\b']),
    ('thumbs_up',   [r'thumbs[\s-]?up']),
    ('praying',     [r'\bpray(s|ing)?\b']),
    ('thinking',    [r'\bthink(s|ing)?\b']),
    ('confused',    [r'\bconfus']),
    ('triumphing',  [r'\btriumph']),
    ('welcoming',   [r'\bwelcom', r'\bgreet', r'\bhello\b']),
    ('face_palm',   [r'face[\s-]?palm', r'facing palm']),
    ('chef_kiss',   [r"chef'?s\s+kiss"]),
    ('yawn',        [r'\byawn']),
    ('lasso',       [r'\blasso']),
    ('check_time',  [r'check(ing)?\s+(time|watch)']),
    ('rub_eyes',    [r'rub(bing)?\s+eyes']),
    ('listening',   [r'\blistening\b']),
    ('not_seeing',  [r'not\s+seeing']),
    ('not_hearing', [r'not\s+hearing']),
    ('lamenting',   [r'\blamenting']),
    ('puking',      [r'\bpuk(e|ing)']),
    # ── Attention / orientation (looking)
    ('look_at',     [r'look(s|ing)?\s+at\s']),
    ('look_around', [r'look(s|ing)?\s+around']),
    ('head_turn',   [r'turn(s|ing)?\s+(their\s+)?head']),
]

_MANIPULATION_LEAF_PATTERNS: list[tuple[str, list[str]]] = [
    # Social/handover-adjacent manipulation (give/show/receive)
    ('give',        [r'\bgiv(es|ing|en)?\b', r'\bhand(s|ed|ing)?\s+to\s']),
    ('show',        [r'\bshow(s|ing|n)?\b']),
    ('offer',       [r'\boffer(s|ing|ed)?\b']),
    # Active manipulations
    ('kick',        [r'\bkick(s|ing|ed)?\b']),
    ('punch',       [r'\bpunch(es|ing|ed)?\b']),
    ('throw',       [r'\bthrow(s|ing|n)?\b', r'\btoss']),
    ('catch',       [r'\bcatch(es|ing)?\b']),
    ('pull',        [r'\bpull(s|ing|ed)?\b']),
    ('push',        [r'\bpush(es|ing|ed)?\b']),
    ('pick_up',     [r'pick(s|ing|ed)?\s+up', r'\bpickup\b']),
    ('put_down',    [r'put(s|ting)?\s+down', r'put(s|ting)?\s+away']),
    ('lift',        [r'\blift(s|ing|ed)?\b']),
    ('carry',       [r'\bcarry(ing)?\b', r'\bcarried\b']),
    ('grab',        [r'\bgrab(s|bing|bed)?\b']),
    ('drink',       [r'\bdrink(s|ing)?\b']),
    ('eat',         [r'\beat(s|ing)?\b']),
    ('itching',     [r'\bitch(ing)?\b', r'\bscratch']),
    ('grating',     [r'\bgrating\b']),
    ('opening',     [r'\bopen(s|ing)?\b']),
    ('closing',     [r'\bclos(es|ing|ed)\b']),
    ('lever',       [r'\blever']),
    ('knob',        [r'\bknob']),
    ('handle',      [r'\bhandle']),
]


# Pre-compile for speed (group all patterns per leaf into a single regex)
def _compile(patterns: list[tuple[str, list[str]]]) -> list[tuple[str, re.Pattern]]:
    return [(leaf, re.compile('|'.join(pats), re.IGNORECASE))
            for leaf, pats in patterns]


_GESTURE_RE = _compile(_GESTURE_LEAF_PATTERNS)
_MANIPULATION_RE = _compile(_MANIPULATION_LEAF_PATTERNS)


def get_family(canonical_class: str) -> str:
    """Map canonical class → top-level family (7 families)."""
    return FAMILY_OF.get(canonical_class, 'other')


def get_leaf_action(content_type_of_movement: str | None,
                    content_short_description: str | None,
                    content_natural_desc_1: str | None = None) -> tuple[str, str]:
    """Resolve (family, leaf_action) for a clip.

    Strategy:
      1. Search descriptions for gesture keywords → if matched, override family
         to 'gesture' regardless of content_type_of_movement.
      2. Likewise for manipulation keywords.
      3. Fall back to canonical class (= family + leaf).

    Why override family from keywords:
      - "Walks then waves" is tagged content_type_of_movement="walking, ...",
        but for browsing the wave aspect dominates → gesture/wave.
      - Calibration still happens at canonical level (see canonicalize()), so
        the (μ, σ) for these clips comes from their content_type_of_movement
        — only the directory layout uses the leaf.

    Args:
        content_type_of_movement: BONES metadata field
        content_short_description: BONES short description (may miss e.g. shrug)
        content_natural_desc_1: BONES long description (catches more keywords)

    Returns:
        (family, leaf_action) — both lowercase, dir-safe.
    """
    parts = []
    if isinstance(content_short_description, str):
        parts.append(content_short_description)
    if isinstance(content_natural_desc_1, str):
        parts.append(content_natural_desc_1)
    desc = ' | '.join(parts)

    if desc:
        for leaf, regex in _GESTURE_RE:
            if regex.search(desc):
                return 'gesture', leaf
        for leaf, regex in _MANIPULATION_RE:
            if regex.search(desc):
                return 'manipulation', leaf

    canonical = canonicalize(content_type_of_movement)
    family = get_family(canonical)

    # If content_type_of_movement says gesture but no keyword matched,
    # bucket as gesture_other (don't lose it to canonical leaf 'gesture').
    if family == 'gesture':
        return family, 'gesture_other'
    if family == 'manipulation':
        return family, 'manipulation_other'

    return family, canonical


# ════════════════════════════════════════════════════════════════
# v2 Taxonomy · 22 ACT_CLASSES + 4 families (locked 2026-04-27)
# ════════════════════════════════════════════════════════════════
#
# Spec: docs/knowledge/methods/primitive_schema_v2.md §4
# Source of truth: configs/act_classes.yaml — edit there to add/rename classes.
# Used by: cli.py NPZ output (segment_class_idx, primitive_class_idx)
#          DataLoader v2 (class_idx → embedding lookup)
#          Inference scripts (--act_class wave_one_arm → class_idx=11)
#
# v1 (ACTION_CLASSES above) stays as the calibration-side coarse-class
# taxonomy used by norm_params_by_action.yaml.

import re as _re_v2
import os as _os_v2

_V2_YAML_PATH = _os_v2.path.join(
    _os_v2.path.dirname(__file__), '..', '..', '..', '..',
    'configs', 'act_classes.yaml',
)

# Lazy-loaded module-level state (filled by `_load_v2()`).
ACT_CLASSES_V2: list[str] = []
FAMILY_OF_CLASS_IDX_V2: list[str] = []
CLASS_IDX_OF_NAME_V2: dict[str, int] = {}
NUM_ACT_CLASSES_V2: int = 0
NULL_ACT_CLASS_IDX_V2: int = 0
_V2_COMPILED: list[tuple[str, _re_v2.Pattern]] = []
_V2_LOADED: bool = False


def _load_v2() -> None:
    """Load classes + match rules from configs/act_classes.yaml.

    Populates module-level globals. Called once on first API use.
    """
    global ACT_CLASSES_V2, FAMILY_OF_CLASS_IDX_V2, CLASS_IDX_OF_NAME_V2
    global NUM_ACT_CLASSES_V2, NULL_ACT_CLASS_IDX_V2, _V2_COMPILED, _V2_LOADED

    if _V2_LOADED:
        return

    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML required to load configs/act_classes.yaml — install with "
            "`pip install pyyaml`") from e

    yaml_path = _os_v2.path.normpath(_V2_YAML_PATH)
    if not _os_v2.path.exists(yaml_path):
        raise FileNotFoundError(
            f'configs/act_classes.yaml not found at {yaml_path}. '
            f'This file is the source of truth for ACT_CLASSES_V2.')

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    # Classes (ordered list: idx = position)
    ACT_CLASSES_V2 = [c['name'] for c in cfg['classes']]
    FAMILY_OF_CLASS_IDX_V2 = [c['family'] for c in cfg['classes']]
    NUM_ACT_CLASSES_V2 = len(ACT_CLASSES_V2)
    NULL_ACT_CLASS_IDX_V2 = NUM_ACT_CLASSES_V2
    CLASS_IDX_OF_NAME_V2 = {n: i for i, n in enumerate(ACT_CLASSES_V2)}

    # Validate families against declared family vocab
    declared_families = set(cfg.get('families', []))
    if declared_families:
        for i, fam in enumerate(FAMILY_OF_CLASS_IDX_V2):
            assert fam in declared_families, (
                f'class[{i}]={ACT_CLASSES_V2[i]} has family={fam!r} not in '
                f'declared families {declared_families}')

    # Match rules — keep order; compile each rule's pattern list into one regex
    rules = cfg.get('match_rules', [])
    _V2_COMPILED = [
        (rule['target'],
         _re_v2.compile('|'.join(rule['patterns']), _re_v2.IGNORECASE))
        for rule in rules
    ]
    # Validate rule targets are real class names
    for target, _ in _V2_COMPILED:
        assert target in CLASS_IDX_OF_NAME_V2, (
            f'match_rule target {target!r} is not in ACT_CLASSES_V2')

    _V2_LOADED = True


def reload_v2_config() -> None:
    """Force-reload act_classes.yaml (e.g., after editing the file)."""
    global _V2_LOADED
    _V2_LOADED = False
    _load_v2()


def match_class_idx_v2(segment_label: str | None,
                       content_type_of_movement: str | None = None,
                       natural_desc_1: str | None = None) -> int:
    """Map a BONES/BABEL segment description to ACT_CLASSES_V2 index.

    Searches a combined corpus (segment_label + content_type + long desc) with
    keyword regex. First-match-wins (rules ordered specific → generic).

    Args:
        segment_label: BABEL raw_label / BONES short_description.
        content_type_of_movement: BONES `content_type_of_movement` field.
        natural_desc_1: optional BONES long description (catches e.g. shrug).

    Returns:
        int in [0, NUM_ACT_CLASSES_V2-1] if matched, else NULL_ACT_CLASS_IDX_V2.
    """
    if not _V2_LOADED:
        _load_v2()
    parts = []
    for s in (segment_label, content_type_of_movement, natural_desc_1):
        if isinstance(s, str) and s.strip():
            parts.append(s.lower())
    if not parts:
        return NULL_ACT_CLASS_IDX_V2
    corpus = " | ".join(parts)
    for name, regex in _V2_COMPILED:
        if regex.search(corpus):
            return CLASS_IDX_OF_NAME_V2[name]
    return NULL_ACT_CLASS_IDX_V2


# ════════════════════════════════════════════════════════════════
# Target-based classification (transition-aware)
# ════════════════════════════════════════════════════════════════
#
# Inference paradigm: user sequentially inputs target classes (e.g. run, stand)
# without a "transition" input. The model must learn transitions implicitly
# from (history, target_class) pairs. So at training time, transition primitives
# get labeled with their *target* state, not their current motion type.
#
# Rules:
#   1. BABEL `act_cat == "transition"` → look at neighbor segments:
#      - prefer next non-transition neighbor's class (target the model is moving INTO)
#      - fallback to previous non-transition neighbor (sequence-end transitions)
#      - fallback NULL
#   2. BONES text matches transition pattern → extract target from text:
#      - "stops X and Y" → Y
#      - "from Xing into Y" → Y
#      - "transitioning into Y" → Y
#      - "stops Xing" / "comes to a stop" alone → stand (implicit)
#   3. "starts to X" / "begins X" → keep in X (start phase belongs to target action)
#   4. Else → normal match_class_idx_v2

# Patterns that signal "this segment is a transition (NOT a steady action)".
# Conservative: ONLY decel/pure-transition phrases. Start phrases ("starts to X")
# are intentionally excluded — we keep those in their target X class.
_TRANSITION_PATTERNS_V2 = [
    r'\btransition(s|ing|ed)?\b',
    r'\bstops?\s+\w+ing\b',                                # "stops walking"
    r'\bcomes?\s+to\s+(a\s+|an\s+)?(stop|halt|standstill)\b',
    r'gradually\s+(slows?\s+down|stops?)\b',
    r'from\s+\w+ing\s+(and|to|into|comes?)\b',             # "from walking and"
]
_TRANSITION_RE_V2 = _re_v2.compile('|'.join(_TRANSITION_PATTERNS_V2), _re_v2.IGNORECASE)

# Patterns to extract the explicit *target* of a transition. Tried in order;
# the first match's captured group is what we re-classify against the 22-class
# regex (via match_class_idx_v2).
_TARGET_EXTRACT_PATTERNS_V2 = [
    r'(?:and\s+then|and|then)\s+([a-z]+(?:\s+[a-z]+){0,3})',          # "and stands [upright]"
    r'(?:into|to)\s+(?:an?\s+)?([a-z]+(?:\s+[a-z]+){0,3})',           # "into an idle stance"
    r'transition(?:s|ing|ed)?\s+(?:from\s+\w+\s+)?(?:in)?to\s+([a-z]+(?:\s+[a-z]+){0,3})',
]
_TARGET_EXTRACT_RE_V2 = [_re_v2.compile(p, _re_v2.IGNORECASE) for p in _TARGET_EXTRACT_PATTERNS_V2]


def is_transition_text_v2(text: str | None) -> bool:
    """True if text describes a transition (decel / pure transition)."""
    if not isinstance(text, str) or not text.strip():
        return False
    return bool(_TRANSITION_RE_V2.search(text))


def classify_segment_target_v2(
    segment_label: str | None,
    content_type_of_movement: str | None = None,
    neighbor_classes: tuple[int | None, int | None] = (None, None),
    natural_desc_1: str | None = None,
) -> int:
    """Target-based classification: transition segments get their target state's class.

    Args:
        segment_label: BABEL raw_label / BONES short_description.
        content_type_of_movement: BABEL act_cat / BONES content_type_of_movement.
        neighbor_classes: (prev_class_idx, next_class_idx) for BABEL-style lookup.
            None for either side if no qualified neighbor exists. Both are
            already-classified class indices of the neighboring NON-TRANSITION
            segments (so a "transition" right after another "transition" must
            scan past both).
        natural_desc_1: optional BONES long description.

    Returns:
        class_idx in [0, NUM_ACT_CLASSES_V2-1], or NULL_ACT_CLASS_IDX_V2.
    """
    if not _V2_LOADED:
        _load_v2()

    txt = str(segment_label) if segment_label else ''
    ac = str(content_type_of_movement) if content_type_of_movement else ''
    txt_l = txt.lower().strip()
    ac_l = ac.lower().strip()
    prev_cls, next_cls = neighbor_classes

    # 1. BABEL pure-transition act_cat → use neighbors
    is_babel_pure_trans = (ac_l == 'transition') or (txt_l == 'transition')
    if is_babel_pure_trans:
        if next_cls is not None and next_cls != NULL_ACT_CLASS_IDX_V2:
            return next_cls
        if prev_cls is not None and prev_cls != NULL_ACT_CLASS_IDX_V2:
            return prev_cls
        return NULL_ACT_CLASS_IDX_V2

    # 2. BONES transition text → extract target from text
    full_text = (txt_l + ' ' + ac_l).strip()
    if is_transition_text_v2(full_text):
        # Try to extract explicit target ("and Y", "into Y", "to Y")
        for regex in _TARGET_EXTRACT_RE_V2:
            m = regex.search(full_text)
            if m:
                target_phrase = m.group(1)
                target_idx = match_class_idx_v2(target_phrase, None, None)
                if target_idx != NULL_ACT_CLASS_IDX_V2:
                    return target_idx
        # Fallback: implicit target = stand (most decel/stops settle to stand)
        stand_idx = CLASS_IDX_OF_NAME_V2.get('stand', NULL_ACT_CLASS_IDX_V2)
        return stand_idx

    # 3. Not a transition — normal classification.
    # Match the legacy cli.py behavior: when segment_label has content, use ONLY
    # the label (act_cat is too broad — e.g. content_type='climbing' would label
    # any "stands on platform" text as climb). Fall back to act_cat only when
    # label is empty.
    if txt_l:
        return match_class_idx_v2(segment_label, None, natural_desc_1)
    return match_class_idx_v2(segment_label, content_type_of_movement, natural_desc_1)


def classify_segments_v2(
    segment_labels: list,
    segment_act_cats: list,
) -> list[int]:
    """Classify a sequence of segments with target-based logic + neighbor lookup.

    Two-pass:
      1. Tentative classify each segment ignoring transition-as-target rule
         (just match_class_idx_v2; pure-transition segs get their literal match,
         which is NULL).
      2. Re-classify each segment with neighbor context.

    For neighbor lookup we skip over consecutive transition segments to find
    the closest non-transition neighbor on each side.
    """
    if not _V2_LOADED:
        _load_v2()

    n = len(segment_labels)

    # Tentative classification (no transition handling) — used only to identify
    # which neighbors are NOT transitions for the lookup.
    tentative = []
    is_trans = []
    for i in range(n):
        txt = str(segment_labels[i]) if segment_labels[i] is not None else ''
        ac = str(segment_act_cats[i]) if segment_act_cats[i] is not None else ''
        ac_l = ac.lower().strip()
        txt_l = txt.lower().strip()
        babel_trans = (ac_l == 'transition') or (txt_l == 'transition')
        bones_trans = is_transition_text_v2(txt + ' ' + ac)
        is_trans.append(babel_trans or bones_trans)
        # Tentative: just normal regex; pure-transition segs become NULL here
        tentative.append(match_class_idx_v2(txt, ac if not txt_l else None, None))

    final = []
    for i in range(n):
        # Find prev non-transition neighbor (skip consecutive trans)
        prev_cls = None
        for j in range(i - 1, -1, -1):
            if not is_trans[j]:
                prev_cls = tentative[j]
                break
        # Find next non-transition neighbor
        next_cls = None
        for j in range(i + 1, n):
            if not is_trans[j]:
                next_cls = tentative[j]
                break

        cls = classify_segment_target_v2(
            segment_label=segment_labels[i],
            content_type_of_movement=segment_act_cats[i],
            neighbor_classes=(prev_cls, next_cls),
        )
        final.append(cls)

    return final


def family_of_class_idx_v2(class_idx: int) -> str:
    """Reverse-lookup family from class_idx. NULL → 'null'."""
    if not _V2_LOADED:
        _load_v2()
    if 0 <= class_idx < NUM_ACT_CLASSES_V2:
        return FAMILY_OF_CLASS_IDX_V2[class_idx]
    return "null"


def class_name_v2(class_idx: int) -> str:
    if not _V2_LOADED:
        _load_v2()
    if 0 <= class_idx < NUM_ACT_CLASSES_V2:
        return ACT_CLASSES_V2[class_idx]
    return "NULL"


# Eagerly load at import (so module-level constants like ACT_CLASSES_V2 are
# populated for callers that read them as plain attributes).
try:
    _load_v2()
except (FileNotFoundError, ImportError) as _e:
    # Don't crash module import — APIs will retry-load and report clearly.
    print(f'[action_taxonomy] warning: v2 config not loaded yet: {_e}')


if __name__ == '__main__':
    print('=== canonicalize() ===')
    samples = [
        'walking', 'walking, turning', 'walking, climbing',
        'jogging', 'jogging, turning',
        'jumping', 'jumping, dancing', 'flipping',
        'dancing', 'turning, dancing', 'gesture, dancing',
        'gesture', 'pointing', 'interacting',
        'sitting', 'kneeling', 'on hands and knees', 'crawling',
        'standing idle', 'standing', 'crouching idle',
        'action', 'pulling', 'throwing', 'kicking, walking',
        'transition', '', None, 'unknown_thing',
    ]
    for s in samples:
        print(f'  {str(s)!r:40} → {canonicalize(s)}')

    print('\n=== get_leaf_action(type_of_movement, short_description) ===')
    samples_2 = [
        ('gesture', 'wave two hands in front'),
        ('gesture', 'salute'),
        ('gesture', 'performing a handshake with a partner'),
        ('gesture', 'clapping'),
        ('gesture', 'bowing'),
        ('gesture', "chef's kiss"),
        ('gesture', 'pointing'),
        ('gesture', 'thinking'),
        ('action', 'kicks the ball'),
        ('action', 'pulling a rope'),
        ('action', 'picks up a small object from the ground'),
        ('action', 'drinking standing'),
        ('walking', 'walk forward'),
        ('jogging', 'jog forward and turn'),
        ('dancing', 'dance hip hop kriss cross'),
        ('standing idle', 'in a hurry position'),
    ]
    for ctom, desc in samples_2:
        fam, leaf = get_leaf_action(ctom, desc)
        print(f'  type={ctom!r:18}  desc={desc!r:50}  →  {fam}/{leaf}')

    print('\n=== v2 · match_class_idx_v2() — 22 类映射 ===')
    samples_3 = [
        # interaction (handover 核心)
        ('performing a handshake with a partner', None),
        ('hands the keys over to partner', None),
        ('picks up a small object from the ground', None),
        # gesture
        ('wave two hands in front', None),
        ('wave right hand', None),
        ('salute', None),
        ('bow', 'gesture'),
        ('clapping', None),
        ("expressing don't know without using hands", None),  # → shrug
        ('punching the bag', None),
        # expressive
        ('dance hip hop kriss cross', 'dancing'),
        # locomotion
        ('walks forward', None),
        ('jogs to the right', None),
        ('runs fast', None),
        ('jumps backward', None),
        ('turns 180 degrees', None),
        ('standing idle', 'standing idle'),
        ('crouching down', None),
        ('sits on chair', None),
        ('come up 50cm box', 'climbing box'),     # → climb
        ('on hands and knees forward', None),     # → crawl
        ('kicks the ball', None),
        # NULL
        ('itching head with right hand', None),
        ('triumphing two handed', None),
        ('thinking', None),
    ]
    for desc, ctom in samples_3:
        idx = match_class_idx_v2(desc, ctom)
        name = class_name_v2(idx)
        fam = family_of_class_idx_v2(idx)
        print(f'  {desc!r:55s} → {idx:>2d} {name:14s} ({fam})')
