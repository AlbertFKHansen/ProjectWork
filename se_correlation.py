import torch
import torch.nn.functional as F
import clip
from tqdm import tqdm

import numpy as np
import json
import pandas as pd

import matplotlib.pyplot as plt
from scipy.stats import entropy as scipy_entropy
from geometry_metrics import compute_spectral_entropy, cosine_similarity_matrix
from labelfitting import get_image_embeddings

def encode_labelset(labels, model, device, batch_size=128):
    tokens = [clip.tokenize(f"a photo of a {label}.") for label in labels]
    tokens = torch.cat(tokens).to(device)

    features = []
    with torch.no_grad():
        for i in tqdm(range(0, len(tokens), batch_size), desc="Encoding Labelset"):
            batch = tokens[i:i + batch_size]
            feats = model.encode_text(batch)
            feats /= feats.norm(dim=-1, keepdim=True)
            features.append(feats)
    return torch.cat(features, dim=0)

def compute_entropies(image_features, text_features, idx, temperature=1):
    """
    Function to compute single or multuple embedding accuracy and semantic entropy.

    Args:
        image_features: n x m array for n image embeddings with dimension m 
        text_features: n x m array for n text embeddings with dimension m 
        idx: Ground truth index of the object you are evaluating
        temperature: Modifyable temperature, defaults to 100.
    Returns:
        accuracy: mean accuracy for multiple images/single accuracy for one
        density: mean of Semantic Density for multiple images/single density for one
    """
    image_features = torch.tensor(image_features).to(text_features.device)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    logits = (image_features @ text_features.T) * 100 / temperature
    probs = F.softmax(logits, dim=1)
    density_values = [float(scipy_entropy(p)) for p in probs.cpu().numpy()]
    density = float(np.mean(density_values))
    accuracy = probs[:, idx].mean().item()
    #top_preds = torch.argmax(probs, dim=1)
    return accuracy, density

def analyze_embeddings(embedding_path, label_path, GT_path, plot=False):
    """
    Loads embedding and labelset dictionaries to perform correlation analysis.
    Saves a CSV to use with R/Statistical analysis. Optional scatterplot.

    Args:
        embedding_path: String to path of object embeddings json
        label_path: String to the labelset json
        GT_path: String to json with object_name to ground truth, key: value pairs.
    Returns:
        Dictionary with accuracies, spectral and semantic density entropies. 

    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading CLIP ViT-L/14@336px...")
    #Force the model to use float32:
    model = clip.load("ViT-L/14@336px", jit=False, device=torch.device("cpu"))[0].to(device)

    embedding_dict = get_image_embeddings(Dataset)

    with open(label_path, "r", encoding="utf-8") as f:
        label_idx_dict = json.load(f)

    with open(GT_path, "r", encoding="utf-8") as f:
        Dataset_to_GT = json.load(f)
    
    label_names = list(label_idx_dict.keys())
    text_features = encode_labelset(label_names, model, device)

    for obj in embedding_dict:
        embedding_dict[obj] = np.array([embedding_dict[obj][key] for key in sorted(embedding_dict[obj].keys())], dtype=np.float32)

    results = {}
    for obj_name, label in Dataset_to_GT.items():
        idx = label_idx_dict[label]
        embeddings = embedding_dict[obj_name]
        acc, semantic_entropy = compute_entropies(embedding_dict[obj_name], text_features, idx)
        spectral_entropy = compute_spectral_entropy(cosine_similarity_matrix(embeddings))
        results[obj_name] = (acc, spectral_entropy, semantic_entropy)

    print("Analysis Complete")

    df = pd.DataFrame({
        "object": list(results.keys()),
        "accuracy": [v[0] for v in results.values()],
        "spectral_entropy": [v[1] for v in results.values()],
        "semantic_density": [v[2] for v in results.values()]
    })

    df.to_csv("entropy_analysis.csv", index=False)

    if plot:
        objs = list(results.keys())
        accs = [v[0] for v in results.values()]
        ents = [v[1] for v in results.values()]
        densities = [v[2] for v in results.values()]

        plt.figure(figsize=(8, 6))
        sc = plt.scatter(ents, accs, c=densities, cmap="viridis", alpha=0.8)
        plt.colorbar(sc, label="Semantic Density")
        for obj, x, y in zip(objs, ents, accs):
            plt.text(x + 0.002, y, obj, fontsize=8)
        plt.xlabel("Spectral Entropy")
        plt.ylabel("CLIP Accuracy")
        plt.title("Accuracy vs. Spectral Entropy")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return results

if __name__ == "__main__":
    #Usage example:
    Dataset = "combined" 
    embedding_path = f"Embeddings/{Dataset}.json"
    label_path = f"Data/{Dataset}/labels.json"
    GT_path = f"Data/{Dataset}/GT_labels.json"
    analyze_embeddings(embedding_path, label_path, GT_path, plot=True)
