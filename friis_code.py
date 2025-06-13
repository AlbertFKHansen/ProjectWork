
# Imports
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
data_root = expanduser("~/.torchvision_data")

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model (ViT-L/14@336px) and its corresponding preprocess
model, preprocess = clip.load("ViT-L/14@336px", device=device)

# Load CIFAR-100 test set using CLIP's preprocess
testset = torchvision.datasets.CIFAR100(
    root=data_root,
    train=False,
    download=True,
    transform=preprocess
)

testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
class_names = testset.classes

# Encode label texts
with torch.no_grad():
    text_tokens = torch.cat([clip.tokenize(f"a photo of a {label}") for label in class_names]).to(device)
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Initialize storage
per_class_probs = [[] for _ in range(100)]
per_class_preds = [[] for _ in range(100)]

# Process each image
with torch.no_grad():
    for images, labels in tqdm(testloader, desc="Embedding images"):
        images = images.to(device)
        labels = labels.to(device)

        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        logits = 50 * image_features @ text_features.T
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=1)

        for i in range(len(labels)):
            per_class_probs[labels[i].item()].append(probs[i].cpu().numpy())
            per_class_preds[labels[i].item()].append(preds[i].item())

# Collect all image embeddings for saving
all_image_embeddings = []

with torch.no_grad():
    for images, labels in tqdm(testloader, desc="Collecting all image embeddings"):
        images = images.to(device)
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        all_image_embeddings.append(image_features.cpu().numpy())

# Stack and save to .npy
all_image_embeddings = np.vstack(all_image_embeddings)
np.save("clip_cifar100_image_features.npy", all_image_embeddings)
print(f"Saved {all_image_embeddings.shape} embeddings to clip_cifar100_image_features.npy")

# Compute per-class stats
semantic_densities = []
accuracies = []

for c in range(100):
    probs = np.stack(per_class_probs[c])
    entropy_vals = [scipy_entropy(p) for p in probs]
    avg_entropy = np.mean(entropy_vals)
    accuracy = np.mean(np.array(per_class_preds[c]) == c)

    semantic_densities.append(avg_entropy)
    accuracies.append(accuracy)

# Correlation
r, p = pearsonr(semantic_densities, accuracies)
print(f"Pearson correlation: r = {r:.3f}, p = {p:.3g}")

# Save results
df = pd.DataFrame({
    "class": class_names,
    "semantic_density": semantic_densities,
    "accuracy": accuracies
})
df.to_csv("clip_cifar100_entropy_accuracy.csv", index=False)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(semantic_densities, accuracies, c='blue', alpha=0.7)
for i, name in enumerate(class_names):
    plt.text(semantic_densities[i] + 0.005, accuracies[i], name, fontsize=6)
plt.xlabel("Semantic Density (Mean Entropy)")
plt.ylabel("Top-1 Accuracy")
plt.title(f"CLIP ViT-L/14@336px on CIFAR-100\nPearson r = {r:.3f}, p = {p:.3g}")
plt.grid(True)
plt.tight_layout()
plt.show()
