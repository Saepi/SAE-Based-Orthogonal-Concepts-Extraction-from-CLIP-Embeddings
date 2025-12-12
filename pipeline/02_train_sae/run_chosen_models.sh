#!/bin/bash

MAX_EPOCHS=200
DEVICE="cuda"

echo "----------------------------------------"
echo "Training: ORTSAE (lambda = 1e-6)"
python 02_train_sae/train_sae.py \
    --max_epochs $MAX_EPOCHS \
    --model_name "sae_final_ortsae_1e-6" \
    --lambda_ortsae 1e-6 \
    --model_dir ./model/sae_final_models \
    --logs_dir ./training_logs/sae_final_models \
    --device $DEVICE


echo "----------------------------------------"
echo "Training: FROBENIUS (lambda = 5e-7)"
python 02_train_sae/train_sae.py \
    --max_epochs $MAX_EPOCHS \
    --model_name "sae_final_frobenius_5e-7" \
    --lambda_frobenius 5e-7 \
    --model_dir ./model/sae_final_models \
    --logs_dir ./training_logs/sae_final_models \
    --device $DEVICE

echo "----------------------------------------"
echo "Training: SRIP (lambda = 1e-7)"
python 02_train_sae/train_sae.py \
    --max_epochs $MAX_EPOCHS \
    --model_name "sae_final_srip_5e-7" \
    --lambda_frobenius 1e-7 \
    --model_dir ./model/sae_final_models \
    --logs_dir ./training_logs/sae_final_models \
    --device $DEVICE

echo "----------------------------------------"
echo "Training: NO REGULARIZERS"
python 02_train_sae/train_sae.py \
    --max_epochs $MAX_EPOCHS \
    --model_name "sae_final_noorth" \
    --device $DEVICE \
    --model_dir ./model/sae_final_models \
    --logs_dir ./training_logs/sae_final_models \

