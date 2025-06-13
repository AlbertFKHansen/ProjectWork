import torch
import torchvision
import clip
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import entropy as scipy_entropy, pearsonr
import matplotlib.pyplot as plt
from os.path import expanduser

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model and CIFAR-100 with CLIP preprocessing
model, preprocess = clip.load("ViT-L/14@336px", device=device)
data_root = expanduser("~/.torchvision_data")
testset = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=preprocess)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
class_names = testset.classes

# Encode label texts
with torch.no_grad():
    text_tokens = torch.cat([clip.tokenize(f"a photo of a {label}") for label in class_names]).to(device)
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Storage for class-wise stats
per_class_densities = [[] for _ in range(100)]       # holds softmax distributions
per_class_predictions = [[] for _ in range(100)]     # holds predicted labels

with torch.no_grad():
    for images, labels in tqdm(testloader, desc="Processing images"):
        images = images.to(device)
        labels = labels.to(device)

        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        logits = 100 * image_features @ text_features.T
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=1)

        for i in range(len(labels)):
            class_idx = labels[i].item()
            per_class_densities[class_idx].append(probs[i].cpu().numpy())
            per_class_predictions[class_idx].append(preds[i].item())


# Compute per class so the plot is readable
densities = []
accuracies = []

for c in range(100):
    class_probs = np.stack(per_class_densities[c])
    entropy_vals = [scipy_entropy(p) for p in class_probs]
    mean_entropy = np.mean(entropy_vals)
    accuracy = np.mean(np.array(per_class_predictions[c]) == c)

    densities.append(mean_entropy)
    accuracies.append(accuracy)

# Correlation
r, p = pearsonr(densities, accuracies)
print(f"Pearson correlation: r = {r:.3f}, p = {p:.3g}")

# Save or plot:
df = pd.DataFrame({
    "class": class_names,
    "semantic_density": densities,
    "accuracy": accuracies
})
df.to_csv("cifartest.csv", index=False)
print("Saved to cifartest.csv")

plot = True
if plot:
    plt.figure(figsize=(10, 6))
    plt.scatter(densities, accuracies, c='blue', alpha=0.7)
    for i, name in enumerate(class_names):
        plt.text(densities[i] + 0.005, accuracies[i], name, fontsize=6)
    plt.xlabel("Semantic Density (Mean Entropy)")
    plt.ylabel("Top-1 Accuracy")
    plt.title(f"CLIP ViT-L/14@336px on CIFAR-100\nPearson r = {r:.3f}, p = {p:.3g}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
