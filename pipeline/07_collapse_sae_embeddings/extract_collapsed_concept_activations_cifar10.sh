python pipeline/05_cifar10_get_sae_embeddings/cifar10_get_sae_embeddings.py \
    --output_dir data/sae_embeddings/cifar10/collapsed/noort \
    --sae_checkpoint model/sae_collapsed/sae_collapsed_noort.ckpt

python pipeline/05_cifar10_get_sae_embeddings/cifar10_get_sae_embeddings.py \
    --output_dir data/sae_embeddings/cifar10/collapsed/frob \
    --sae_checkpoint model/sae_collapsed/sae_collapsed_frob.ckpt

python pipeline/05_cifar10_get_sae_embeddings/cifar10_get_sae_embeddings.py \
    --output_dir data/sae_embeddings/cifar10/collapsed/ortsae \
    --sae_checkpoint model/sae_collapsed/sae_collapsed_ortsae.ckpt

python pipeline/05_cifar10_get_sae_embeddings/cifar10_get_sae_embeddings.py \
    --output_dir data/sae_embeddings/cifar10/collapsed/srip \
    --sae_checkpoint model/sae_collapsed/sae_collapsed_srip.ckpt