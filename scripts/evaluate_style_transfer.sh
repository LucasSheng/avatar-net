#!/usr/bin/env bash

MODEL_TYPE=$1
CUDA_ID=$2

# blending weights between style and contents
WEIGHT=$3

# content image folders:
#   exemplar content images: ./data/contents/images/
#   exemplar content videos: ./data/contents/sequences/
CONTENT_DATASET_DIR=./data/contents/sequences/

# style image folders: ./data/styles/
STYLE_DATASET_DIR=./data/styles/

# output image folders: ./results/sequences/
EVAL_DIR=./results/sequences/

# network configuration
CONFIG_DIR=./configs

# the network path for the trained auto-encoding network
TRAIN_DIR=/DATA/lsheng/lsheng_model_checkpoints/style_transfer_models/${MODEL_TYPE}/train

CUDA_VISIBLE_DEVICES=${CUDA_ID} \
    python evaluate_style_transfer.py \
        --checkpoint_dir=${TRAIN_DIR} \
        --model_config_path=${CONFIG_DIR}/${MODEL_TYPE}_config.yml \
        --content_dataset_dir=${CONTENT_DATASET_DIR} \
        --style_dataset_dir=${STYLE_DATASET_DIR} \
        --eval_dir=${EVAL_DIR} \
        --inter_weight=${WEIGHT}