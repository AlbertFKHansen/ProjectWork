import os
import ast
import torch
import clip
from PIL import Image
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import json

# Version that works for coil and our own data
datasets = {"coil20": {}, "Dataset": {}, "processed": {}, "coil100": {}}


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
datasets["Dataset"] = object_labels
datasets["processed"] = object_labels


# Load coil100 object class names from json
with open("COIL100_GPT_labels.json", "r") as f:
    object_labels_coil100 = json.load(f)
print(f'All coil100 object labels are: \n{object_labels_coil100}', end='\n\n')
datasets["coil100"] = object_labels_coil100


for dataset, label in datasets.items():
    print(f'Currently running on {dataset}')

    current_dataset = f'Data/{dataset}/rot'

    labels = imagenet_labels.copy()
    if len(label) > 0:
        for values in label.values():
            labels.extend(values)

        labels = list(set(labels))

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

    # Get prediction scores and the ground truth label
    obj_iterable = os.listdir(current_dataset)
    for obj_name in tqdm(obj_iterable):
        image_folder = Path(f"{current_dataset}/{obj_name}/").resolve()
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

    # Saving the fitted labels for each object
    with open(f'Data/{dataset}/GT_labels_{dataset}.json', 'w') as f:
        f.write(json.dumps(fitted_labels, indent=4))
    print(f'Successively saved label {dataset} to "Data/{dataset}/GT_labels_{dataset}.json"!')

    # Saving the fitted labels as a complete label dataset
    label_dataset = imagenet_labels.copy()
    for label in fitted_labels.values():
        label_dataset.append(label) if label not in label_dataset else None

    with open(f'Data/{dataset}/labels.json', 'w') as f:
        f.write(json.dumps(label_dataset, indent=4))
    print(f'Successively saved label dataset {dataset} to "Data/{dataset}/labels.json"!', end='\n\n')
