import os
import sys
import torch
import pandas as pd
from collections import defaultdict
import argparse
import copy

sys.path.append(os.path.abspath("pipeline/02_train_sae"))
from train_sae import SAE

torch.serialization.add_safe_globals([argparse.Namespace])

models_folder = "model/sae_final_models/"
concept_names_folder = "data/concept_names/"
output_folder = "model/sae_collapsed/"
os.makedirs(output_folder, exist_ok=True)

model_files = [f for f in os.listdir(models_folder) if f.startswith("sae_final_") and f.endswith(".ckpt")]

for model_file in model_files:
    suffix = model_file[len("sae_final_"):-len(".ckpt")]

    print(f"Processing models with suffix: {suffix}")

    names_path = os.path.join(concept_names_folder, f"concept_names_{suffix}.csv")
    df = pd.read_csv(names_path)
    concept_names = df["concept_name"].tolist()

    groups = defaultdict(list)
    for i, name in enumerate(concept_names):
        groups[name].append(i)
    unique_names = list(groups.keys())
    new_latent_dim = len(unique_names)

    ckpt_path = os.path.join(models_folder, model_file)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    E = state_dict["encoder.0.weight"]
    b_e = state_dict["encoder.0.bias"]
    D = state_dict["decoder.weight"]
    b_d = state_dict["decoder.bias"]
    args_old = copy.deepcopy(ckpt["hyper_parameters"]["args"])

    input_dim = E.shape[1]

    E_new = torch.zeros(new_latent_dim, input_dim)
    b_e_new = torch.zeros(new_latent_dim)
    D_new = torch.zeros(input_dim, new_latent_dim)
    b_d_new = b_d.clone()

    for j, name in enumerate(unique_names):
        idxs = groups[name]

        E_new[j] = E[idxs].sum(dim=0)
        b_e_new[j] = b_e[idxs].sum()
        D_new[:, j] = D[:, idxs].sum(dim=1)

    args_new = copy.deepcopy(args_old)
    args_new.latent_dim = new_latent_dim
    sae_new = SAE(input_dim=input_dim, args=args_new)

    with torch.no_grad():
        sae_new.encoder[0].weight.copy_(E_new)
        sae_new.encoder[0].bias.copy_(b_e_new)
        sae_new.decoder.weight.copy_(D_new)
        sae_new.decoder.bias.copy_(b_d_new)

    output_path = os.path.join(output_folder, f"sae_collapsed_{suffix}.ckpt")
    torch.save({
        "state_dict": sae_new.state_dict(),
        "hyper_parameters": {"args": args_new},
    }, output_path)

    print(f"Collapsed SAE checkpoint saved at {output_path}")
