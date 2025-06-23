from loadData import loadDataset
from geometry_metrics import compute_spectral_entropy, cosine_similarity_matrix

import torch
import clip
import numpy as np
from tqdm import tqdm
from PIL import Image
import json

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device)
datasets = loadDataset("Data")
set_name = "Dataset" #or coil20
set_version_name = "rot" #or temp

# Takes PIL objects which are generated from loadData.py
def load_and_embed_images(PIL_objects):
    embeddings = []
    for obj in PIL_objects:
        image_input = preprocess(obj).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        # Convert to NumPy and enforce float32 for linalg + JSON
        np_embedding = image_features.cpu().numpy().astype(np.float32)
        embeddings.append(np_embedding)
    return np.vstack(embeddings)

SE_dict = {}
embedding_dict = {}

for name, objects in tqdm(datasets[set_name][set_version_name].items(), desc = "Generating SE and Embedding dictionaries:"):
    embeddings = load_and_embed_images(objects)
    SE = compute_spectral_entropy(cosine_similarity_matrix(embeddings))
    SE_dict[name] = SE
    embedding_dict[name] = embeddings.tolist()

# Save dictionaries to json
with open("object_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(embedding_dict, f, indent=2)

with open("spectral_entropy.json", "w", encoding="utf-8") as f:
    json.dump(SE_dict, f, indent=2)


