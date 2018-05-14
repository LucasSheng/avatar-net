#!/usr/bin/env bash

CUDA_ID=$1
# content image folders:
#   exemplar content images: ./data/contents/images/
#   exemplar content videos: ./data/contents/sequences/
CONTENT_DATASET_DIR=$2
# style image folders: ./data/styles/
STYLE_DATASET_DIR=$3
# output image folders: ./results/sequences/
EVAL_DATASET_DIR=$4

# network configuration
CONFIG_DIR=./configs/AvatarNet_config.yml

# the network path for the trained auto-encoding network (need to change accordingly)
MODEL_DIR=/DATA/AvatarNet

CUDA_VISIBLE_DEVICES=${CUDA_ID} \
    python evaluate_style_transfer.py \
        --checkpoint_dir=${MODEL_DIR} \
        --model_config_path=${CONFIG_DIR} \
        --content_dataset_dir=${CONTENT_DATASET_DIR} \
        --style_dataset_dir=${STYLE_DATASET_DIR} \
        --eval_dir=${EVAL_DATASET_DIR} \
        --inter_weight=0.8