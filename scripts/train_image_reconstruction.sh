#!/usr/bin/env bash

CUDA_ID=$1
# MSCOCO tfexample dataset path
DATASET_DIR=$2
# model path
MODEL_DIR=$3

# network configuration
CONFIG_DIR=./configs/AvatarNet_config.yml

CUDA_VISIBLE_DEVICES=${CUDA_ID} \
    python train_image_reconstruction.py \
        --train_dir=${MODEL_DIR} \
        --model_config=${CONFIG_DIR} \
        --dataset_dir=${DATASET_DIR} \
        --dataset_name=MSCOCO \
        --dataset_split_name=train \
        --batch_size=8 \
        --max_number_of_step=120000 \
        --optimizer=adam \
        --learning_rate_decay_type=fixed \
        --learning_rate=0.0001