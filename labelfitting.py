from collections import Counter
import json
import torch
from tqdm import tqdm
import clip


def get_image_embeddings(dataset_name: str) -> dict[str, dict[str, list]]:
    """
    Args:
        dataset_name: Name of the dataset path. Could be "coil100"

    Returns:
        Dictionary of the "rot" json

    """
    with open(f'Embeddings/{dataset_name}.json', 'r', encoding='utf-8') as f:
        embeddings = json.load(f)
    return embeddings['rot']

def save_labels(dataset_name: str, fit_labels: dict, base_labels: list):
    # Saving the fitted labels for each object
    with open(f'Data/{dataset_name}/GT_labels.json', 'w') as f:
        f.write(json.dumps(fit_labels, indent=4))
    print(f'Successively saved label {dataset_name} to "Data/{dataset_name}/GT_labels.json"!')

    # Saving the fitted labels as a complete label dataset
    base_labels = base_labels.copy()
    for label in fitted_labels.values():
        base_labels.append(label) if label not in base_labels else None
    label_dict = {label: id for id, label in enumerate(base_labels)}

    with open(f'Data/{dataset_name}/labels.json', 'w') as f:
        f.write(json.dumps(label_dict, indent=4))
    print(f'Successively saved label dataset {dataset_name} to "Data/{dataset_name}/labels.json"!', end='\n\n')


if __name__ == '__main__':
    # Version that works for coil and our own data
    datasets = {"coil20": {}, "Dataset": {}, "processed": {}, "coil100": {}}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Model is running on {device}', end='\n\n')
    model, preprocess = clip.load("ViT-L/14@336px", device)

    # Load ImageNet class names from txt
    with open("Imagenet_classes.json", "r") as f:
        class_dict = json.load(f)
    imagenet_labels = []
    for label in class_dict.values():
        label_string: str = label[0]
        first_label = label_string.split(", ")[0].strip()

        imagenet_labels.append(first_label)
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

        labels = imagenet_labels.copy()  # Reset labels to be ImageNet labels
        if len(label) > 0:  # If the current dataset has any labels attached, then add them to the labels list
            for values in label.values():
                labels.extend(values)

            labels = list(set(labels))

        # Tokenize labelset using clip.tokenize:
        labelset = torch.cat([clip.tokenize(f"a photo of a {label}") for label in labels]).to(device)

        # Encode labelset embedding vectors
        with torch.no_grad():
            text_features = model.encode_text(labelset)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        # Main loop
        fitted_labels = {}

        # Get prediction scores and the ground truth label
        image_embeddings = get_image_embeddings(dataset)
        for obj_name, embeddings in tqdm(image_embeddings.items()):
            predictions = []
            for embed in embeddings.values():
                embed = torch.tensor(embed, device=device, dtype=text_features.dtype)

                # Compute similarity with a temperature of 0.01
                temp = 1/100
                logits = embed @ text_features.T / temp
                probs = torch.softmax(logits, dim=-1)
                top_idx = torch.argmax(probs, dim=-1).item()

                predicted_label = labels[top_idx]
                predictions.append(predicted_label)

            most_common_label = Counter(predictions).most_common(1)[0][0]
            fitted_labels[obj_name] = most_common_label

        print(f'{"Object:":<20} | Most common label:')
        print('-' * (20 * 2 + 1))
        for obj_name, most_common_label in fitted_labels.items():
            print(f'{obj_name:<20} | {most_common_label}')
        print()  # Empty space at end

        save_labels(dataset, fitted_labels, imagenet_labels)
