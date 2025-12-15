#!/bin/bash

L1_LAMBDAS=(1e-5)

EMB_DIRS=("noort" "frob" "ortsae" "srip")

BASE_EMB_PATH="data/sae_embeddings/cifar10/collapsed"

for L1 in "${L1_LAMBDAS[@]}"; do
    for EMB_TYPE in "${EMB_DIRS[@]}"; do
        MODEL_NAME="cbm_collapsed_${EMB_TYPE}_l1${L1}"

        echo "Running $MODEL_NAME"

        python pipeline/06_train_cbm/train_cbm.py \
        --train_emb "${BASE_EMB_PATH}/${EMB_TYPE}/train_latent.pt" \
        --val_emb   "${BASE_EMB_PATH}/${EMB_TYPE}/val_latent.pt" \
        --test_emb  "${BASE_EMB_PATH}/${EMB_TYPE}/test_latent.pt" \
        --l1_lambda $L1 \
        --epochs 250 \
        --model_dir "model/sae_collapsed" \
        --logs_dir "training_logs/cbm_" \
        --model_name "$MODEL_NAME" \
        --device cuda
    done
done
