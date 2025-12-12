import argparse

def get_args():
    parser = argparse.ArgumentParser("Sparse Autoencoder (SAE) Training")

    parser.add_argument("--train_path", type=str, default="data/embeddings/cc3m/train.pt")
    parser.add_argument("--val_path", type=str, default="data/embeddings/cc3m/validation.pt")
    parser.add_argument("--test_path", type=str, default="data/embeddings/cc3m/test.pt")

    parser.add_argument("--model_dir", type=str, default="./model/sae")
    parser.add_argument("--logs_dir", type=str, default="./training_logs/sae")
    parser.add_argument("--model_name", type=str, default="sae_model")

    parser.add_argument("--latent_dim", type=int, default=4096, help="Dimensionality of latent space")
    parser.add_argument("--lambda1", type=float, default=1e-7, help="L1 regularization on latent")

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--max_epochs", type=int, default=5)

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    parser.add_argument("--lambda_frobenius", type=float, default=0)
    parser.add_argument("--lambda_ortsae", type=float, default=0)
    parser.add_argument("--lambda_srip", type=float, default=0)

    return parser.parse_args()