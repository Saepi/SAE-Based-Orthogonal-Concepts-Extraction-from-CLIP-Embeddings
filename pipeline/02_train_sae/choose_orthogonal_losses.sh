#!/bin/bash

lambdas=(1e-8 1e-7 1e-6 1e-5 1e-4 1e-3)

declare -A regularizers
regularizers=( ["ortsae"]="--lambda_ortsae" ["frob"]="--lambda_frobenius" ["srip"]="--lambda_srip")

MAX_EPOCHS=100
DEVICE="cuda"

for lambda in "${lambdas[@]}"; do
        for reg in "${!regularizers[@]}"; do
        arg_name=${regularizers[$reg]}
        MODEL_NAME="sae_${reg}_l${lambda}_lr1e-4"

        echo "----------------------------------------"
        echo "Training $MODEL_NAME"

        python pipeline/02_train_sae/train_sae.py \
            --max_epochs $MAX_EPOCHS \
            --model_name $MODEL_NAME \
            $arg_name $lambda \
            --device $DEVICE \
            --model_dir ./model/sae_ort\
            --logs_dir ./training_logs/sae_ort \
            --lr 1e-4
    done
done
