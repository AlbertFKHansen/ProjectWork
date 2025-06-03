from loadData import loadDataset
from geometry_metrics import compute_spectral_entropy, cosine_similarity_matrix

import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path
import ast
import torch
import clip
import torch.nn.functional as F
from tqdm import tqdm

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device)
base = Path("Data") / "Dataset"

object_to_GT = {
    "battery": "GP battery",
    "charger": "wall power adapter",
    "combination_lock": "combination lock",
    "controller": "playstation controller",
    "cup_no_handle": "stoneware tumbler",
    "cup_with_handle": "stoneware coffee mug",
    "duct_tape": "grey adhesive tape",
    "faxe_kondi": "Faxe Kondi",
    "lego_brick": "lego brick",
    "lego_man": "lego minifigure",
    "light_bulb": "electric bulb",
    "rubber_duck": "plastic duck",
    "rubiks_cube": "3x3 cube puzzle",
    "sun_glasses": "dark sunglasses",
    "thermo_cup": "metal coffee tumbler",
    "toy_car": "small grey toy car",
    "wrist_watch": "black wristwatch"
}

# Load embedding dictionary
with (base / "object_embeddings.json").open("r", encoding="utf-8") as f:
    embedding_dict = json.load(f)
embedding_dict = {k: np.array(v, dtype=np.float32) for k, v in embedding_dict.items()}

# Load label mapping
with (base / "labels.json").open("r", encoding="utf-8") as f:
    label_idx_dict = json.load(f)

# Encode labelset with CLIP:
def encode_labelset(merged_labels):
    labelset = torch.cat([clip.tokenize(f"a photo of a {c}") for c in merged_labels]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(labelset)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

def rotational_accuracy(image_features, text_features, idx):
    image_features = torch.tensor(image_features).to(device)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    logits = 100 * (image_features @ text_features.T)
    probs = F.softmax(logits, dim=1)
    GT_class_prob = probs[:, idx]
    return GT_class_prob.mean().item()

text_features = encode_labelset(label_idx_dict.keys())
correlation = {}
for obj_name, label in tqdm(object_to_GT.items(), desc="Evaluating"):
    idx = label_idx_dict[label]
    embeddings = embedding_dict[obj_name]
    acc = rotational_accuracy(embeddings, text_features, idx)
    SE = compute_spectral_entropy(cosine_similarity_matrix(embeddings))
    correlation[obj_name] = (acc,SE)
print(correlation)

objects = list(correlation.keys())
accuracies = [v[0] for v in correlation.values()]
entropies = [v[1] for v in correlation.values()]

plt.figure(figsize=(7, 5))
plt.scatter(entropies, accuracies, color="blue", alpha=0.7)

for obj, x, y in zip(objects, entropies, accuracies):
    plt.text(x + 0.003, y, obj, fontsize=8)

plt.xlabel("Spectral Entropy")
plt.ylabel("CLIP Accuracy")
plt.title("CLIP Accuracy vs. Spectral Entropy")
plt.grid(True)
plt.tight_layout()
plt.show()
