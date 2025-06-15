import os
import re
from PIL import Image

def _is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

def _extract_image_index(filename):
    """
    Extract number before '.':

    in filenames like obj1__5.png, 10.jpg, temperature_4300.jpeg, etc.
    the number before the extension is considered the indexing value. (5,10,4300)
    This way we can sort images in a list automatically.
    """
    match = re.search(r'(\d+).', filename)
    return int(match.group(1)) if match else float('inf')

def _load_folder(path):
    """
    Recursively loads the directory into a nested dictionary.
    - Subdirs become dict keys
    - Image files are loaded as a list of PIL Image objects, sorted by index
    """
    entries = os.listdir(path)
    files = [f for f in entries if _is_image_file(f)]
    subdirs = [f for f in entries if os.path.isdir(os.path.join(path, f))]

    if files and not subdirs:
        # Folder only contains image files → return sorted list of images
        files.sort(key=_extract_image_index)
        return [Image.open(os.path.join(path, f)) for f in files]

    # Folder has subfolders (may or may not have images too)
    result = {}

    for entry in entries:
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            result[entry] = _load_folder(full_path)
        elif _is_image_file(entry):
            # If there are images *and* folders, put them under a special key to avoid confusion
            result.setdefault("_images", []).append((_extract_image_index(entry), Image.open(full_path)))

    # Sort images under "_images" key if present
    if "_images" in result:
        result["_images"].sort(key=lambda x: x[0])  # sort by index
        result["_images"] = [img for _, img in result["_images"]]  # strip index

    return result

def loadDataset(root_path:str):
    """
    Loads all top-level dataset dirs under root_path into a nested dictionary.
    Each dataset dir is expected to contain subdirs for objects, and each object dir may contain images.
    The images are loaded as PIL Image objects, and the subdirs are loaded recursively.
    The function returns a dictionary where keys are directory/dataset names and values are the loaded Images.
    If a dir contains both images and subdirs, the images are stored under the key "_images".

    The images are sorted by the number before the file extension in their names.
    This allows for automatic sorting of images in a list.


    Args:
        root_path (str): Path to the root dir containing dataset dirs.
    Returns:
        datasets (dict): A dictionary where keys are directory names and values are the loaded PIL.Image objects.
    
    Example:
    >>> root_path = 'Data'
    >>> datasets = loadDatasets(root_path)
    >>> # Result structure:
    >>> datasets = {
    ...     'dataset1': {
    ...         'object1': [...],
    ...         'object2': [...],
    ...         '_images': [...],
    ...     },
    ...     'dataset2': {
    ...         ...
    ...     }
    ... }
    """

    datasets = {}
    if not os.path.exists(root_path):
        raise FileNotFoundError(f"Path '{root_path}' does not exist.")
    if not os.path.isdir(root_path):
        raise NotADirectoryError(f"Path '{root_path}' is not a directory.")

    # Check if there are any directories in root_path
    dir_entries = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]
    if not dir_entries:
        datasets = _load_folder(root_path)
        return datasets
    else:
        for name in os.listdir(root_path):
            full_path = os.path.join(root_path, name)
            if os.path.isdir(full_path):
                datasets[name] = _load_folder(full_path)
        return datasets




if __name__ == '__main__':
    # Example usage
    root_path = 'Data'
    datasets = loadDataset(root_path)
    
    for dataset_name, dataset in datasets.items():
        print(f"Dataset: {dataset_name}")
        for key, value in dataset.items():
            if isinstance(value, list):
                print(f"  {key}: {len(value)} images")
            else:
                print(f"  {key}: {value}")