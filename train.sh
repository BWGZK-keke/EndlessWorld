#!/usr/bin/env bash
# Multi-GPU training for EndlessWorld 3D-aware fine-tuning.
#
# Before running, edit configs/self_forcing_dmd.yaml so that `generator_ckpt`
# points to the Self-Forcing DMD warm-start checkpoint, then run:

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --rdzv_id=endlessworld \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    train.py \
    --config_path configs/self_forcing_dmd.yaml \
    --logdir logs/endlessworld \
    --disable-wandb
