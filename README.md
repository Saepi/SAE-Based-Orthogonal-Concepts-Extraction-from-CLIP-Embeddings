# SAE-Based Orthogonal Concepts Extraction from CLIP Embeddings

This is a repository of a project made for the course of Numerical Linear Algebra, Skoltech 2025 by the team of:
- Stepan Epifantsev, DS-1
- Valeria Yakupova, DS-1
- Andrej Mymrin, DS-1

<details> <summary><strong>Click to expand the full project tree</strong></summary>
.
├── pipeline/
│   ├── 01_cc3m_get_clip_embeddings/
│   │   └── extract_cc3m_embeddings.py
│   ├── 02_train_sae/
│   │   ├── choose_sae_params.sh
│   │   ├── choose_orthogonal_losses.sh
│   │   └── run_chosen_models.sh
│   ├── 03_name_concepts/
│   │   ├── generate_vocab_embeddings.py
│   │   └── name_concepts.py
│   ├── 04_cifar10_get_clip_embeddings/
│   │   └── cifar10_get_clip_embeddings.py
│   ├── 05_cifar10_get_sae_embeddings/
│   │   └── extract_concept_activations_cifar10.sh
│   └── 06_train_cbm/
│       └── train_cbm.sh
│
├── data/
│   ├── cifar-10-python.tar.gz
│   ├── concept_names/
│   ├── datasets/
│   │   └── cc3m/
│   ├── embeddings/
│   │   ├── cc3m/
│   │   └── cifar10/
│   ├── sae_embeddings/
│   │   ├── cc3m/
│   │   └── cifar10/
│   └── vocab/
│       └── 20k.txt
│
├── models/
│   ├── cbm/
│   ├── sae/
│   ├── sae_final_models/
│   └── sae_ort/
│
├── training_logs/
│   ├── sae/
│   ├── sae_ort/
│   └── cbm/
│
└── README.md

</details>

In “Discover-then-Name: Task-Agnostic Concept Bottlenecks via Automated Concept Discovery” (https://arxiv.org/abs/2407.14499), it was suggested to use a Sparse Autoencoder (SAE) on CLIP embeddings to extract neurons aligned with human-interpretable concepts for further implementation in Concept Bottleneck Models (CBMs). This approach eliminates the need for manually labeling a concept set in the dataset, significantly reducing the amount of time spent on data collection.

However, CBMs are known to be prone to polysemantic concepts, which reduces model interpretability and limits the ability to intervene effectively. We aim to explicitly enforce concept orthogonality during SAE training, with the goal of reducing information leakage in the concept layer and improving concept disentanglement.


## Step 0. Install requirements

```bash
python -m venv cbm_clip_sae
source cbm_clip_sae/bin/activate
pip install -r requirements
```


## Step 1. Get CLIP embeddings of CC3M dataset images

### Download data

Download the ‘Train_GCC-training.tsv’ and ‘Validation_GCC-1.1.0-Validation.tsv’ from https://ai.google.com/research/ConceptualCaptions/download. Change their names to cc3m_training.tsv and cc3m_test.tsv. Run the commands below to download images.

***Note:*** From further on we use only first 200,000 rows of the train file

```bash
sed -i '1s/^/caption\turl\n/' cc3m_training.tsv 

img2dataset --url_list cc3m_training.tsv --input_format "tsv" --url_col "url" --caption_col "caption" --output_format webdataset --output_folder training --processes_count 16 --thread_count 64 --image_size 256
```

```bash
sed -i '1s/^/caption\turl\n/' cc3m_test.tsv 

img2dataset --url_list cc3m_test.tsv --input_format "tsv" --url_col "url" --caption_col "caption" --output_format webdataset --output_folder test --processes_count 16 --thread_count 64 --image_size 256
```

The last two shards of training data are used as validation set.

### Extract CLIP embeddings

Run next command to obtain CLIP embeddings of the CC3M images.

```bash
python pipeline/01_cc3m_get_clip_embeddings.py/extract_cc3m_embeddings.py \
        --data_root data/datasets/cc3m \
        --train_dir training \
        --val_dir validation  \
        --test_dir test  \
        --train_range 00000..00047 \
        --val_range 00000..00001 \
        --test_range 00000..00001 \
        --output_dir data/embeddings/cc3m \
        --img_enc_name clip-RN50 \
        --batch_size 256 \
        --device cuda
```

## Step 2. Train SAE

Run the command to find the best dimensionality and sparse loss multiplier. We have chosen 4096 and 1e-7.

```bash
bash pipeline/02_train_sae/choose_sae_params.sh 
```

Run the command to find the best orthogonality loss multipliers. Check our results in the next script.

```bash
bash pipeline/02_train_sae/choose_orthogonal_losses.sh 
```

Run the command to find train the 4 SAE models (no orthogonalization, Frobenius norm approach, OrtSAE method and SRIP).

```bash
bash pipeline/02_train_sae/run_chosen_models.sh 
```

In order to monitor the loss and other metrics changes run the code (depending on the directory of interest):

```bash
tensorboard --logdir ./training_logs/sae
```

## Step 3. Name concepts

### Download vocabulary and generate text embeddings

Download the vocabulary of 20k words used by CLIP-Dissect (https://github.com/first20hours/google-10000-english/blob/master/20k.txt). 

Run the code to generate normalized CLIP embeddings

```bash
python pipeline/03_name_concepts/generate_vocab_embeddings.py
```

### Name SAE concepts

The following code runs the search for the closest text embeddings for each of the concept embeddings

```bash
bash pipeline/03_name_concepts/name_concepts.py
```

## Step 4. Get CLIP embeddings of CIFAR-10 dataset images

Launch the command below to download CIFAR-10 and extract CLIP embeddings of the images

```bash
python pipeline/04_cifar10_get_clip_embeddings/cifar10_get_clip_embeddings.py \
    --data_root ./data/datasets \
    --output_dir ./data/embeddings/cifar10/
```

## Step 5. Get SAE embeddings of CIFAR-10 dataset images

The following code gets the CIFAR-10 concept embeddings for each of the models

```bash
bash pipeline/05_cifar10_get_sae_embeddings/extract_concept_activations_cifar10.sh
```

## Step 6. Train CBM final layer

Run the command beow to train the linear layer of CBM with different levels of sparsity

```bash
bash pipeline/06_train_cbm/train_cbm.sh
```

Training process could be seen by the following commad

```bash
tensorboard --logdir ./training_logs/cbm
```

