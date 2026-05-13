#!/usr/bin/env bash
# Long video generation with EndlessWorld (3D-aware autoregressive rollout).
#
# Required:
#   --baseline_checkpoint_path : the Self-Forcing DMD warm-start checkpoint
#                                (text-only generator, used for the first chunk)
#   --checkpoint_path          : the EndlessWorld checkpoint (generator + 3D fusion)
#
# Both checkpoints are available at the project HuggingFace release page.

CUDA_VISIBLE_DEVICES=0 python inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --output_folder outputs/endlessworld/ \
    --baseline_checkpoint_path checkpoints/self_forcing_dmd.pt \
    --checkpoint_path checkpoints/endlessworld.pt \
    --data_path prompts/vidprom_filtered_extended.txt \
    --num_extension_steps 30 \
    --use_ema
