import os
from pathlib import Path
from PIL import Image
from rembg import remove

base_path = Path("Data/Dataset/rot")
output_dir = Path("Data/Dataset/Thumbnails")
output_dir.mkdir(exist_ok=True)
# Clear existing thumbnails
for file in output_dir.glob("*"):
    if file.is_file():
        file.unlink()  # delete the file

target_angle = 0
crop_width = 2500
crop_height = 1900

target_filename = f"{target_angle}.png"
for obj_folder in sorted(base_path.iterdir()):
    if obj_folder.is_dir():
        image_path = obj_folder / target_filename
        if image_path.exists():
            img = Image.open(image_path).convert("RGBA")
            width, height = img.size
            #print(width,height)
            #Remove background
            img = remove(img)
            white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(white_bg, img)

            #Rotate Image
            img = img.rotate(2, expand=False)
        
            #Crop Image
            offset = 0
            left = (width - crop_width) // 2
            top = ((height - crop_height) // 2) + offset
            right = left + crop_width
            bottom = top + crop_height - offset
            img = img.crop((left, top, right, bottom))

            #Save Image
            thumb_name = f"{obj_folder.name}_{target_angle}_thumbnail.png"
            output_path = output_dir / thumb_name
            img.save(output_path)

        else:
            print("Error: Missing filename for one or more images!")