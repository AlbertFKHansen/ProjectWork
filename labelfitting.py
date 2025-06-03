import os
import ast
import torch
import clip
import numpy as np
from PIL import Image
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import json
import glob

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Model is running on {device}', end='\n\n')
model, preprocess = clip.load("ViT-L/14@336px", device)

# Load ImageNet class names from txt
with open("Imagenet_classes.txt", "r") as f:
    class_dict = ast.literal_eval(f.read())
imagenet_labels = [v.split(',')[0].strip() for v in class_dict.values()]
print(f'All ImageNet labels are \n{imagenet_labels}', end='\n\n')

# Load our object class names from json
with open("object_labels.json", "r") as f:
    object_labels = json.load(f)
print(f'All object labels are: \n{object_labels}', end='\n\n')

# Merge labels
labels = imagenet_labels.copy()
for key, values in object_labels.items():
    for value in values:
        labels.append(value)

# Removing duplicates
length = len(labels)
labels = list(set(labels))
print(f'Removed {length - len(labels)} duplicate labels')
print(f'All labels are: \n{labels}', end='\n\n')

# Tokenize labelset using clip.tokenize:
labelset = torch.cat([clip.tokenize(f"a photo of a {label}") for label in labels]).to(device)

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

# Version that works for coil and our own data ::: Just choose which dataset you want to fit
coil = 'Data/coil20/rot'
dataset = 'Data/Dataset/rot'

obj_iterable = os.listdir(dataset) # <- Change the dataset here
for obj_name in tqdm(obj_iterable):
    image_folder = Path(f"{dataset}/{obj_name}/").resolve()
    images = os.listdir(image_folder)

    predictions = []
    for image in images:
        image_path = os.path.join(image_folder, image)
        image_features = embed_image(image_path)

        # Compute similarity (100X comes from CLIP's git)
        logits = 100 * image_features @ text_features.T
        probs = torch.softmax(logits, dim=1)
        top_idx = torch.argmax(probs, dim=1).item()
        predicted_label = labels[top_idx]
        predictions.append(predicted_label)

    most_common_label = Counter(predictions).most_common(1)[0][0]
    fitted_labels[obj_name] = most_common_label

print(f'{"Object:":<20} | Most common label:')
print('-' * (20*2 + 1))
for obj_name, most_common_label in fitted_labels.items():
    print(f'{obj_name:<20} | {most_common_label}')
print() # Empty space at end

# Saving the fitted labels
labels = list(fitted_labels.values())
imagenet_labels = [lbl for lbl in imagenet_labels if lbl not in labels]
merged_labels = imagenet_labels + labels
merged_labels = list(set(merged_labels))

label_to_index = {label: idx for idx, label in enumerate(merged_labels)}
print(f'Label dataset:\n{label_to_index}', end='\n\n',)

# Save the fitted labes
with open('Data/Dataset/labels.json', 'w') as f:
    f.write(json.dumps(label_to_index, indent=4))
print('Successively saved label dataset to "Data/Dataset/labels.json"!')
