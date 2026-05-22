"""Backward-compat shim: data_scripts.annotate_vad_llm → data_pipeline.vad.llm_annotator.

Real code lives in data_pipeline/vad/llm_annotator.py. This wrapper preserves
the `python -m data_scripts.annotate_vad_llm ...` CLI entry point mentioned
in earlier logs.
"""
from MoGenAgent.data_pipeline.vad.llm_annotator import (
    Args,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    build_items,
    call_claude,
    call_openai,
    annotate_batch,
    main,
)

__all__ = [
    "Args", "SYSTEM_PROMPT", "USER_PROMPT_TEMPLATE",
    "build_items", "call_claude", "call_openai", "annotate_batch", "main",
]

if __name__ == "__main__":
    main()
