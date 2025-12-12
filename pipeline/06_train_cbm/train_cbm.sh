#!/bin/bash

L1_LAMBDAS=(1e-6 1e-5 1e-4 1e-3)

EMB_DIRS=("noort" "frob" "ortsae" "srip")

BASE_EMB_PATH="data/sae_embeddings/cifar10"

for L1 in "${L1_LAMBDAS[@]}"; do
    for EMB_TYPE in "${EMB_DIRS[@]}"; do
        MODEL_NAME="cbm_${EMB_TYPE}_l1${L1}"

        echo "Running $MODEL_NAME"

        python step6_train_cbm/step6_train_cbm.py \
        --train_emb "${BASE_EMB_PATH}/${EMB_TYPE}/train_latent.pt" \
        --val_emb   "${BASE_EMB_PATH}/${EMB_TYPE}/val_latent.pt" \
        --test_emb  "${BASE_EMB_PATH}/${EMB_TYPE}/test_latent.pt" \
        --l1_lambda $L1 \
        --epochs 100 \
        --model_dir "model/cbm" \
        --logs_dir "training_logs/cbm" \
        --model_name "$MODEL_NAME" \
        --device cuda
    done
done
