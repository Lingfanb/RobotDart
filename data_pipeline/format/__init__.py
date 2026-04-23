"""Dataset-specific parsers + canonical 69-dim feature computation.

Each parser produces a DatasetParser-compliant iterator of clips, which
downstream tools (segment, vad, retarget) consume uniformly.
"""
