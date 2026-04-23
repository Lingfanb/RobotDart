"""data_pipeline — unified motion data processing for VAD-conditioned G1.

See data_pipeline/README.md for architecture overview.

Public submodules:
    data_pipeline.vad      — VAD labeling + augmentation
    data_pipeline.retarget — skeleton retargeting
    data_pipeline.segment  — primitive slicing + action segmentation
    data_pipeline.format   — dataset parsers + 69-dim feature computation
"""

__version__ = "0.1.0"
