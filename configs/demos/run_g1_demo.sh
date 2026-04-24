#!/bin/bash
# Run G1 interactive demo — same experience as the original DART demo
# Type 'start' → type text prompts → robot moves in real-time

guidance=5
batch_size=1

model='./mld_denoiser/g1_mld_v2/checkpoint_300000.pt'

python -m mld.run_g1_demo \
    --denoiser_checkpoint "$model" \
    --batch_size $batch_size \
    --guidance_param $guidance
