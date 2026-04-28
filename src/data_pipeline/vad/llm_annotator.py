"""Annotate motion clips with VAD via LLM (Claude or GPT).

Reads BABEL text labels from mp_data_g1_69 pickle, sends them in batches to
Claude or GPT, parses JSON VAD scores, and writes `vad_labels_llm.json`.

Usage:
    # Dry run (no API, just prints prompts)
    python -m data_scripts.annotate_vad_llm --dry_run

    # Real run, requires ANTHROPIC_API_KEY
    ANTHROPIC_API_KEY=sk-ant-... python -m data_scripts.annotate_vad_llm \
        --provider claude \
        --split val \
        --out_path data/processed/vad_labels_llm_val.json

    # OpenAI fallback
    OPENAI_API_KEY=sk-... python -m data_scripts.annotate_vad_llm \
        --provider openai --model gpt-4o-mini

Schema of output:
    {
      "metadata": {...},
      "entries": {
        "<item_index>": {
          "text": "wave",
          "act_cats": ["wave", "gesture", "greet"],
          "V": 0.7, "A": 0.4, "D": 0.1,
          "reasoning": "waving is a warm greeting gesture..."
        },
        ...
      }
    }
"""
from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro


SYSTEM_PROMPT = """You are an expert in affective psychology and human motion analysis.
Your task is to annotate motion descriptions with VAD (Valence-Arousal-Dominance)
scores on Mehrabian's PAD dimensional model.

Each dimension is a continuous value in [-1, +1]:
- VALENCE (V): pleasantness, -1 = unpleasant/negative, +1 = pleasant/positive
- AROUSAL (A): activation, -1 = calm/drowsy, +1 = excited/alert
- DOMINANCE (D): control, -1 = submissive/yielding, +1 = dominant/in-control

Semantic anchors (for calibration):
- Joyful greeting:     V=+0.8, A=+0.6, D=+0.3
- Warm hospitality:    V=+0.7, A=+0.3, D=+0.1
- Calm confidence:     V=+0.3, A= 0.0, D=+0.5
- Polite neutral:      V=+0.2, A= 0.0, D=-0.1
- Hesitant offer:      V= 0.0, A=-0.2, D=-0.4
- Tired / low:         V=-0.3, A=-0.6, D=-0.3
- Sad withdraw:        V=-0.7, A=-0.4, D=-0.5
- Firm / assertive:    V=+0.1, A=+0.4, D=+0.7
- Urgent alarm:        V=-0.2, A=+0.9, D=+0.5
- Angry / aggressive:  V=-0.7, A=+0.8, D=+0.7

For each motion description, output a JSON object with fields:
    text, V, A, D, reasoning

For ambiguous or underspecified descriptions (e.g., "walk"), output a neutral
value near V=+0.1, A=+0.2, D=+0.1 (mildly positive, moderate arousal).

Respond with ONLY a valid JSON array, no markdown fences.
"""

USER_PROMPT_TEMPLATE = """Annotate the following {n} motion descriptions with VAD scores.

Each description has:
- "id": index
- "text": primary text label
- "act_cats": action categories

Input:
{items_json}

Output: a JSON array of {n} objects, each with {{id, text, V, A, D, reasoning}}.
"""


@dataclass
class Args:
    split: str = "val"
    """'train' or 'val'."""
    data_dir: str = "data/processed/mp_data_g1_69/Canonicalized_h2_f8_num1_fps30"
    out_path: str = ""
    """Output JSON path. Default: data/processed/vad_labels_llm_{split}.json"""
    provider: str = "claude"
    """'claude' or 'openai'."""
    model: str = ""
    """Model name. Default: claude-sonnet-4-5 for claude, gpt-4o-mini for openai."""
    batch_size: int = 25
    """Descriptions per API call."""
    limit: int = 0
    """Limit to first N items (0 = all)."""
    dedupe_by_text: bool = True
    """If True, dedupe by (text, act_cats tuple) to save API cost."""
    dry_run: bool = False
    """If True, print prompts but don't call API."""
    resume: bool = True
    """If True and out_path exists, skip already-annotated items."""


def build_items(data: list, args: Args) -> tuple[list[dict], dict]:
    """Flatten data items to annotation units, optionally dedupe.

    Returns:
        items: list of {id, text, act_cats}
        dedupe_map: if dedupe_by_text, maps index → canonical_id to fan-out later
    """
    canonical_items = []
    seen_keys = {}
    dedupe_map = {}  # original_idx → canonical_idx

    for i, item in enumerate(data):
        if args.limit > 0 and i >= args.limit:
            break
        primary_text = item["texts"][0] if item["texts"] else ""
        act_cats = tuple(sorted(item.get("act_cats", [])))
        key = (primary_text.lower().strip(), act_cats)
        if args.dedupe_by_text:
            if key not in seen_keys:
                cidx = len(canonical_items)
                canonical_items.append({
                    "id": cidx,
                    "text": primary_text,
                    "act_cats": list(act_cats),
                })
                seen_keys[key] = cidx
            dedupe_map[i] = seen_keys[key]
        else:
            canonical_items.append({
                "id": i,
                "text": primary_text,
                "act_cats": list(act_cats),
            })
            dedupe_map[i] = i
    return canonical_items, dedupe_map


def call_claude(messages, model="claude-sonnet-4-5") -> str:
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("pip install anthropic")
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return resp.content[0].text


def call_openai(messages, model="gpt-4o-mini") -> str:
    try:
        import openai
    except ImportError:
        raise RuntimeError("pip install openai")
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def annotate_batch(items: list[dict], args: Args) -> list[dict]:
    """Call LLM on one batch, return parsed annotations."""
    user_msg = USER_PROMPT_TEMPLATE.format(
        n=len(items),
        items_json=json.dumps(items, ensure_ascii=False, indent=2),
    )

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN — would send:")
        print(user_msg[:500] + ("..." if len(user_msg) > 500 else ""))
        return [
            {"id": it["id"], "text": it["text"],
             "V": 0.0, "A": 0.0, "D": 0.0,
             "reasoning": "[dry run]"}
            for it in items
        ]

    messages = [{"role": "user", "content": user_msg}]
    model = args.model or (
        "claude-sonnet-4-5" if args.provider == "claude" else "gpt-4o-mini")

    if args.provider == "claude":
        text = call_claude(messages, model=model)
    elif args.provider == "openai":
        text = call_openai(messages, model=model)
    else:
        raise ValueError(f"unknown provider: {args.provider}")

    # Extract JSON from the response
    text = text.strip()
    if text.startswith("```"):
        # strip markdown fences if present
        lines = text.split("\n")
        text = "\n".join(l for l in lines if not l.startswith("```"))

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse JSON from LLM: {text[:500]}") from e

    # openai returns a dict with key being the array, let's handle both
    if isinstance(parsed, dict):
        if "annotations" in parsed:
            parsed = parsed["annotations"]
        elif "results" in parsed:
            parsed = parsed["results"]
        else:
            # fallback: take the first list value
            for v in parsed.values():
                if isinstance(v, list):
                    parsed = v
                    break
    return parsed


def main():
    args = tyro.cli(Args)
    if not args.out_path:
        args.out_path = f"data/processed/vad_labels_llm_{args.split}.json"

    # Load motion data
    pkl_path = Path(args.data_dir) / f"{args.split}.pkl"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} items from {pkl_path}")

    # Prepare canonical items
    canonical_items, dedupe_map = build_items(data, args)
    print(f"After dedupe: {len(canonical_items)} unique descriptions "
          f"(from {len(dedupe_map)} clips)")

    # Resume from existing annotations
    out_path = Path(args.out_path)
    annotations = {}
    if args.resume and out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        annotations = existing.get("entries", {})
        print(f"Resumed: {len(annotations)} existing entries")

    todo = [it for it in canonical_items if str(it["id"]) not in annotations]
    print(f"To annotate: {len(todo)} items in batches of {args.batch_size}")

    # Batch-annotate
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i:i + args.batch_size]
        print(f"[{i}/{len(todo)}] Batch of {len(batch)}...", flush=True)
        try:
            annotated = annotate_batch(batch, args)
            for ann in annotated:
                aid = str(ann.get("id"))
                if aid in {str(b["id"]) for b in batch}:
                    annotations[aid] = ann
        except Exception as e:
            print(f"  ERROR: {e}")
            if args.dry_run:
                continue
            # Save partial progress
            with open(out_path, "w") as f:
                json.dump({"metadata": {"partial": True}, "entries": annotations}, f)
            raise
        if not args.dry_run:
            time.sleep(1.0)  # rate limit

    # Fan out to original indices using dedupe_map
    fan_out = {}
    for orig_idx, canonical_idx in dedupe_map.items():
        aid = str(canonical_idx)
        if aid in annotations:
            fan_out[str(orig_idx)] = annotations[aid]

    # Save
    output = {
        "metadata": {
            "split": args.split,
            "total_clips": len(data),
            "unique_descriptions": len(canonical_items),
            "annotated_descriptions": len(annotations),
            "provider": args.provider,
            "model": args.model or "default",
            "dedupe_by_text": args.dedupe_by_text,
        },
        "entries": fan_out,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_path} ({len(fan_out)} clip-level entries)")


if __name__ == "__main__":
    main()
