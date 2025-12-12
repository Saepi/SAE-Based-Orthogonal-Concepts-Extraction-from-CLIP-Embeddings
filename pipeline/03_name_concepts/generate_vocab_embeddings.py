import os
import sys
import torch
from tqdm import tqdm
from clip import clip

repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(repo_root) 

vocab_dir = "vocab"
vocab_file = os.path.join(vocab_dir, "clipdissect_20k.txt")
clip_model_name = "RN50"
batch_size = 256

with open(vocab_file, "r") as f:
    words = [w.strip() for w in f.readlines() if w.strip()]

print(f"Loaded {len(words)} words.")

device = "cuda"
model, preprocess = clip.load(clip_model_name, device=device)
model.eval()

all_features = []

for i in tqdm(range(0, len(words), batch_size)):
    batch_words = words[i:i+batch_size]
    tokens = clip.tokenize(batch_words).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_features.append(feats.cpu())

text_features = torch.cat(all_features, dim=0)

output_file = os.path.join(
    vocab_dir,
    f"embeddings_clip_{clip_model_name}_clipdissect_20k.pth"
)
torch.save(text_features, output_file)
print(f"Saved embeddings to {output_file}")