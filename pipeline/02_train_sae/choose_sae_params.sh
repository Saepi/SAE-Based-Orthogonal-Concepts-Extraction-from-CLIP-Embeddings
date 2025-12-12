#!/bin/bash

LAMBDA1S=(1e-8 1e-7 1e-6 1у-5 1у-4)
LATENTS=(1024 2048 4096 8182)

for L1 in "${LAMBDA1S[@]}"; do
    for DIM in "${LATENTS[@]}"; do

        MODEL_NAME="sae_l1${L1}_dim${DIM}"

        echo "Running $MODEL_NAME"

        python 02_train_sae/train_sae.py \
        --latent_dim $DIM \
        --lr 1e-4\
        --lambda1 $L1 \
        --max_epochs 100 \
        --model_name $MODEL_NAME \
        --model_dir ./model/sae \
        --logs_dir ./training_logs/sae \
        --device cuda

    done
done

