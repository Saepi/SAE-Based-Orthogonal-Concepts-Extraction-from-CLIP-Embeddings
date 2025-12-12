python step5_extract_sae_cifar10_embeddings/step5_extract_sae_cifar10_embeddings.py \
    --output_dir data/sae_embeddings/cifar10/noort \
    --sae_checkpoint model/sae_final_models/sae_l11e-7_dim4096-epoch=199.ckpt

python step5_extract_sae_cifar10_embeddings/step5_extract_sae_cifar10_embeddings.py \
    --output_dir data/sae_embeddings/cifar10/frob \
    --sae_checkpoint model/sae_final_models/sae_final_frobenius_5e-7-epoch=175.ckpt

python step5_extract_sae_cifar10_embeddings/step5_extract_sae_cifar10_embeddings.py \
    --output_dir data/sae_embeddings/cifar10/ortsae \
    --sae_checkpoint model/sae_final_models/sae_ortsae_l1e-6-epoch=199.ckpt

python step5_extract_sae_cifar10_embeddings/step5_extract_sae_cifar10_embeddings.py \
    --output_dir data/sae_embeddings/cifar10/srip \
    --sae_checkpoint model/sae_final_models/sae_srip_l1e-7_lr1e-4-epoch=195.ckpt