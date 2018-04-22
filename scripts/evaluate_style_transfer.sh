#!/usr/bin/env bash

MODEL_TYPE=$1
CUDA_ID=$2
WEIGHT=$3

CONTENT_DATASET_DIR=/DATA/lsheng/lsheng_data/content_examplar
STYLE_DATASET_DIR=/DATA/lsheng/lsheng_data/source_dataset/simple

CONFIG_DIR=/home/lsheng/lsheng_models/avatar-net/configs

TRAIN_DIR=/DATA/lsheng/lsheng_model_checkpoints/style_transfer_models/${MODEL_TYPE}/train
EVAL_DIR=/DATA/lsheng/lsheng_model_checkpoints/style_transfer_models/${MODEL_TYPE}/eval

CUDA_VISIBLE_DEVICES=${CUDA_ID} \
    python evaluate_style_transfer.py \
        --checkpoint_dir=${TRAIN_DIR} \
        --model_config_path=${CONFIG_DIR}/${MODEL_TYPE}_config.yml \
        --content_dataset_dir=${CONTENT_DATASET_DIR} \
        --style_dataset_dir=${STYLE_DATASET_DIR} \
        --eval_dir=${EVAL_DIR} \
        --inter_weight=${WEIGHT}