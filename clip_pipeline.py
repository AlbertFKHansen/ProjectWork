from PIL import Image
import numpy as np
from loadData import loadDataset
from rembg import remove
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import sys
from geometry_metrics import compute_spectral_entropy, cosine_similarity_matrix
import torch
import clip
import json


class CLIPModel():
    def __init__(self, name="ViT-L/14@336px", verbose=False):
        """
        Initializes the CLIP model with the specified name and device.
        Args:
            name (str): The name of the CLIP model to load. Default is "ViT-L/14@336px".
            verbose (bool): If True, prints the device being used and tqdm progress bars.
        """
        self.verbose = verbose
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if verbose:
            print(f"Using device: {torch.cuda.get_device_name(0)}") if self.device != "cpu" else print("Using CPU")
        self.model, self.preprocess = clip.load(name, self.device)

    def _embed_image(self, image):
        """
        processes a PIL image and returns its normalized CLIP embedding as a numpy f32 array.
        Args:
            image (PIL.Image): The image to embed.
        Returns:
            np.ndarray: A numpy array of shape (1, 512) containing the normalized CLIP embedding.
        """
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().astype(np.float32)
    
    def _embed_label(self, label):
        """
        processes "a photo of a {label}." and returns its normalized CLIP embedding as a numpy f32 array.
        Args:
            label (str): The label to embed, e.g., "cat", "dog", etc.
        Returns:
            np.ndarray: A numpy array of shape (1, 512) containing the normalized CLIP embedding.
        """
        text_input = clip.tokenize([f"a photo of a {label}."]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().astype(np.float32)
    
    def embed(self, item):
        """
        Embeds an item, which can be a PIL image or a label string.
        Returns the normalized CLIP embedding as a numpy f32 array.
        Args:
            item (PIL.Image or str): The item to embed. If a PIL Image, it will be processed as an image.
                                    If a string, it will be processed as a label.
        Returns:
            np.ndarray: A numpy array of shape (1, 512) containing the normalized CLIP embedding.
        """
        if isinstance(item, Image.Image):
            return self._embed_image(item)
        elif isinstance(item, str):
            return self._embed_label(item)
        else:
            raise ValueError("Item must be a PIL Image or a label string.")
        
    
    def embed_batch(self, items):
        """
        Embeds a batch of items, which can be a list of PIL images or label strings.
        Returns a numpy array of normalized CLIP embeddings.
        Args:
            items (list): A list of items to embed, where each item can be a PIL Image or a label string.
        Returns:
            np.ndarray: A 2D numpy array where each row is the normalized CLIP embedding of an item.
        """
        embeddings = []
        for item in tqdm(items, desc="Embedding items", disable=not self.verbose):
            embeddings.append(self.embed(item))
        return np.vstack(embeddings)





def __remove(img):
    img_no_bg = remove(img)
    np_img = np.array(img_no_bg)
    mask = np_img[:, :, 3] > 100  # Alpha channel
    return np_img, mask

def process_images(imgs, margin_ratio=0.1, output_size=None):
    """
    Processes a list of PIL images to crop all to a common bounding box (with margin),
    and return square images with a consistent background color.

    Args:
        imgs (list of PIL.Image): List of input images containing a single object each.
        margin_ratio (float): Fraction of the object size to use as margin (default is 10%).
        output_size (int): Optional; if provided, resizes the final images to this size.

    Returns:
        list of PIL.Image: List of processed, square, centered images.
    """

    masks = []
    np_imgs = []

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(__remove, imgs), total=len(imgs), desc="Removing background"))
    
    for np_img, mask in results:
        np_imgs.append(np_img)
        masks.append(mask)

    # Find combined bounding box over all masks
    all_coords = np.concatenate([np.argwhere(mask) for mask in masks], axis=0)
    y0, x0 = all_coords.min(axis=0)
    y1, x1 = all_coords.max(axis=0) + 1

    # Calculate margin in pixels (based on combined bbox)
    height = y1 - y0
    width = x1 - x0
    margin_y = int(margin_ratio * height)
    margin_x = int(margin_ratio * width)

    # Expand bounding box with margin
    y0 = y0 - margin_y
    y1 = y1 + margin_y
    x0 = x0 - margin_x
    x1 = x1 + margin_x

    # Make the crop square
    crop_height = y1 - y0
    crop_width = x1 - x0
    if crop_height > crop_width:
        diff = crop_height - crop_width
        x0 -= diff // 2
        x1 += diff - diff // 2
    elif crop_width > crop_height:
        diff = crop_width - crop_height
        y0 -= diff // 2
        y1 += diff - diff // 2

    # Calculate mean brightness for all visible pixels in all images
    all_visible_pixels = []
    for np_img, mask in zip(np_imgs, masks):
        rgb = np_img[:, :, :3]
        all_visible_pixels.append(rgb[mask])
    all_visible_pixels = np.concatenate(all_visible_pixels, axis=0)
    object_brightness = np.mean(all_visible_pixels)

    if object_brightness > 127:
        bg_color = (0, 0, 0, 255)
    else:
        bg_color = (255, 255, 255, 255)

    results = []
    for np_img, mask in tqdm(zip(np_imgs, masks), desc="Cropping and resizing images"):
        # Calculate padding if crop goes out of bounds
        y0_pad = max(0, -y0)
        x0_pad = max(0, -x0)
        y1_pad = max(0, y1 - np_img.shape[0])
        x1_pad = max(0, x1 - np_img.shape[1])

        # Crop (may be out of bounds)
        cropped_img = np_img[max(0, y0):min(np_img.shape[0], y1), max(0, x0):min(np_img.shape[1], x1)]
        cropped_mask = mask[max(0, y0):min(mask.shape[0], y1), max(0, x0):min(mask.shape[1], x1)]

        # Pad if needed
        if any([y0_pad, y1_pad, x0_pad, x1_pad]):
            cropped_img = np.pad(
                cropped_img,
                ((y0_pad, y1_pad), (x0_pad, x1_pad), (0, 0)),
                mode='constant',
                constant_values=0
            )
            cropped_mask = np.pad(
                cropped_mask,
                ((y0_pad, y1_pad), (x0_pad, x1_pad)),
                mode='constant',
                constant_values=0
            )

        # Set background color
        background = np.zeros_like(cropped_img, dtype=np.uint8)
        background[~cropped_mask] = bg_color
        output = np.where(cropped_mask[..., None], cropped_img, background)

        out_img = Image.fromarray(output)
        if output_size is not None:
            out_img = out_img.resize((output_size, output_size), Image.LANCZOS)
        results.append(out_img)

    return results


def dataset_to_embed_json(model, dataset, path):
    """
    Converts a dataset dictionary to a JSON file with embeddings and saves it.
    The key is defined as path_to_image from path/to/image.png, example:
    Dataset/rot/rubber_duck/35.png = {rot: {rubber_duck: {35: [embedding]}}}
    Args:
        data (dict): The dataset dictionary to save.
        path (str): The file path where the JSON will be saved.
    """

    embeddings_dict = {}
    for intervention in dataset.keys():
        embeddings_dict[intervention] = {}
        for obj in tqdm(dataset[intervention].keys(), desc=f"Processing {intervention} objects", disable=not model.verbose):
            embeddings_dict[intervention][obj] = {}
            embeddings = model.embed_batch(dataset[intervention][obj])
            for i, embedding in enumerate(embeddings):
                key = f"{i*5 if intervention == 'rot' else i*100 + 2000}"
                embeddings_dict[intervention][obj][key] = embedding.tolist()
            
            # Close images after processing to free memory
            for img in dataset[intervention][obj]:
                img.close()

    with open(path, "w") as f:
        json.dump(embeddings_dict, f, indent=4)



if __name__ == "__main__":

    data = loadDataset("Data/Dataset")

    def process_and_save(obj, imgs, path, name_fn):
        processed_imgs = process_images(imgs)
        for i, processed_img in enumerate(processed_imgs):
            save_path = os.path.join(path, obj, name_fn(i))
            processed_img.save(save_path)

        for img in imgs:
            img.close()
        for img in processed_imgs:
            img.close()

    def ensure_dir(obj, path):
        obj_path = os.path.join(path, obj)
        os.makedirs(obj_path, exist_ok=True)

    # Get object name from command line argument, or process all if not provided
    obj_args = sys.argv[1:] if len(sys.argv) > 1 else None

    rot = data["rot"]
    rot_path = "Data/processed/rot"
    objs = obj_args if obj_args else list(rot.keys())
    for obj in tqdm(objs, desc="Processing rotation images"):
        if obj not in rot:
            print(f"Object '{obj}' not found in rotation data.")
            continue
        ensure_dir(obj, rot_path)
        imgs = rot[obj]
        process_and_save(obj, imgs, rot_path, lambda x: f"{x*5}.png")

    temp = data["temp"]
    temp_path = "Data/processed/temp"
    objs = obj_args if obj_args else list(temp.keys())
    for obj in tqdm(objs, desc="Processing temperature images"):
        if obj not in temp:
            print(f"Object '{obj}' not found in temperature data.")
            continue
        ensure_dir(obj, temp_path)
        imgs = temp[obj]
        process_and_save(obj, imgs, temp_path, lambda x: f"{x*100+2000}.png")
        for img in imgs:
            img.close()


    model = CLIPModel(verbose=True)
    os.makedirs("Embeddings", exist_ok=True)

    sets = ["coil20", "Dataset", "processed"]
    for set in sets:
        dataset = loadDataset(f"Data/{set}")
        path = f"Embeddings/{set}.json"
        dataset_to_embed_json(model, dataset, path)
    

    """ simple usage example

    from clip_pipeline import CLIPModel, dataset_to_embed_json

    model = CLIPModel(verbose=True)
    dataset = loadDataset("Data/my_dataset")
    dataset_to_embed_json(model, dataset, "Embeddings/my_dataset.json")
    
    """





