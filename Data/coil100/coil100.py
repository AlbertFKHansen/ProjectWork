"""
This script is made to restructure the coil100 dataset
so it has teh same structure as our other datasets.
"""

from pathlib import Path
from tqdm import tqdm

root = Path('./')
output_root = root / 'rot'

for file in tqdm(root.glob('*.png')):
    image_id = str(file).split('__')[0]
    target_dir = output_root / image_id

    target_dir.mkdir(parents=True, exist_ok=True)

    new_path = target_dir / file.name
    file.rename(new_path)