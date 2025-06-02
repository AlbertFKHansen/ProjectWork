import os
import ast
import torch
import clip
import numpy as np
from PIL import Image
from pathlib import Path
from collections import Counter

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device)

# Load ImageNet class names from txt
with open("Imagenet_classes.txt", "r") as f:
    class_dict = ast.literal_eval(f.read())
imagenet_labels = [v.split(',')[0].strip() for v in class_dict.values()]

# Tokenize labelset using clip.tokenize:
labelset = torch.cat([clip.tokenize(f"a photo of a {label}") for label in imagenet_labels]).to(device)

# Encode labelset embedding vectors
with torch.no_grad():
    text_features = model.encode_text(labelset)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Embed an image
def embed_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features

# Main loop
fitted_labels = {}

# Version that works for coil specifically ::: 
for obj_id in range(1, 21):  
    object_name = str(obj_id)
    image_folder = Path(f"../ProjectWork/Data/coil20/rot/{object_name}/").resolve()
    predictions = []

    for i in range(72):
        image_path = os.path.join(image_folder, f"obj{object_name}__{i}.png")
        image_features = embed_image(image_path)

        # Compute similarity (100X comes from CLIP's git)
        logits = 100 * image_features @ text_features.T
        probs = torch.softmax(logits, dim=1)
        top_idx = torch.argmax(probs, dim=1).item()
        predicted_label = imagenet_labels[top_idx]
        predictions.append(predicted_label)

    most_common_label = Counter(predictions).most_common(1)[0][0]
    fitted_labels[object_name] = most_common_label
    print(f"Object {object_name} -> {most_common_label}")

