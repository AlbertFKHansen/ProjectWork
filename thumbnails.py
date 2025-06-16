from clip_pipeline import process_images
from loadData import loadDataset
import os
import sys
import argparse
from tqdm import tqdm


"""
This script is made to generate the thumbnails for a table in the report.
We are doing both thumbnails of the original images and the processed images.
for rotation and temperature datasets. but im just going to do it for a reduced granularity of them.
"""


# command line usage
parser = argparse.ArgumentParser(description="Generate thumbnails for dataset images.")
parser.add_argument("--dataset", type=str, default="Dataset",help="Path to the dataset directory")
parser.add_argument("--output_size", type=int, default=100, help="Thumbnail output size (pixels)")
args = parser.parse_args()



def generate_thumbnails(dataset_input, output_size):
    """
    Generate thumbnails for the images in the dataset.
    Args:
        dataset_input (str): Path to the dataset directory.
        output_size (int): Size of the thumbnails in pixels.
    """
    obj = dataset_input.split("/")[-1]  # get the last part of the path as the object name
    intervention = dataset_input.split("/")[1]  # get the second part of the path as the intervention name
    dataset_path = f"Data/{dataset_input}"
    output_dir = os.path.join("Thumbnails", dataset_input)
    os.makedirs(output_dir, exist_ok=True)

    data = loadDataset(dataset_path)

    # use all images to generate the correct bounding box
    # Process images (resize to output_size)
    processed = []
    for img in data:
        thumbnail = img.copy()
        # center crop
        width, height = thumbnail.size
        if width != height:
            if width > height:
                # Remove equally from left and right
                left = (width - height) // 2
                right = left + height
                top = 0
                bottom = height
            else:
                # Remove equally from top and bottom
                top = (height - width) // 2
                bottom = top + width
                left = 0
                right = width
            thumbnail = thumbnail.crop((left, top, right, bottom))
        thumbnail.thumbnail((output_size, output_size))
        processed.append(thumbnail)

    # save all processed images as thumbnails
    for idx, img in enumerate(processed):
        thumbnail_path = os.path.join(output_dir, f"{obj}_thumbnail_{idx*5 if intervention=='rot' else idx*100+2000}.png")
        img.save(thumbnail_path, "PNG")
        img.close()

    # close the original images to free up resources
    for img in data:
        img.close()


if __name__ == "__main__":
    # run the script with command line arguments
    args = parser.parse_args()
    print(f"Generating thumbnails for dataset:  {args.dataset}")
    print(f"Output size:                        {args.output_size} pixels")

    # Support multiple datasets from command line
    dataset_inputs = args.dataset.split()
    for dataset_input in tqdm(dataset_inputs, desc="Datasets"):
        for root, dirs, files in tqdm(os.walk(f"Data/{dataset_input}"), desc=f"Scanning {dataset_input}", leave=False):
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if image_files and not dirs:
                rel_path = os.path.relpath(root, "Data")
                print(f"Generating thumbnails for: {rel_path}")
                generate_thumbnails(rel_path, args.output_size)
