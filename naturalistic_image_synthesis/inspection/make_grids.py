"""
Create 5×2 image grids for visual inspection of anchor images per object.

For each object in anchor_images/, arranges all images (sorted alphabetically)
in a grid of up to 5 columns × 2 rows at 256×256 px per cell, annotates each
cell with a truncated filename, and saves one JPEG grid per object into
anchor_grids_inspection/. Run after generation to quickly spot low-quality
images before running the similarity pipeline.
"""
import os
from PIL import Image, ImageDraw, ImageFont
import math
from naturalistic_image_synthesis.config import STIM_SET_DIR, ANCHOR_IMAGES_DIR  # generation_20240326_180652


def create_image_grid(source_folder, target_folder, required_string, image_size=(200, 200), font_size=20, single_strip=False, str_name=''):
    # Ensure target folder exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # List all image files in the source folder that contain the required string
    image_files = [f for f in os.listdir(source_folder) if f.endswith('.png')]
    image_files.sort()  # Sort the files for consistent ordering

    # Determine grid size based on the number of images
    num_images = len(image_files)
    if num_images ==10 and single_strip==False:
        grid_cols = 5
        grid_rows = 2
    elif single_strip:
        grid_cols = num_images
        grid_rows = 1
    else:
        grid_cols = int(math.ceil(math.sqrt(num_images)))
        grid_rows = int(math.ceil(num_images / grid_cols))

    # Create a new blank image for the grid
    grid_width = grid_cols * image_size[0]
    grid_height = grid_rows * image_size[1]
    grid_image = Image.new('RGB', (grid_width, grid_height))

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
        print("Custom font not found, using default.")

    # Paste each image into the grid and add the filename
    for index, image_file in enumerate(image_files):
        path = os.path.join(source_folder, image_file)
        try:
            with Image.open(path) as img:
                # Resize image to fit the grid cell
                img = img.resize(image_size)
                # Calculate position for pasting
                x_index = index % grid_cols
                y_index = index // grid_cols
                x = x_index * image_size[0]
                y = y_index * image_size[1]
                # Paste the image
                grid_image.paste(img, (x, y))
                # Draw the filename
                draw = ImageDraw.Draw(grid_image)
                # Truncate the filename if it's too long to fit in the image
                truncated_filename = (image_file[:25] + '..') if len(image_file) > 25 else image_file
                draw.text((x + 5, y + 5), truncated_filename, fill="white", font=font)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")


    # Save the grid image
    grid_image_path = os.path.join(target_folder, f'{required_string}_image_grid{str_name}.jpg')
    grid_image.save(grid_image_path)
    print(f"Grid image saved to {grid_image_path}")


# Usage example

source_folder = ANCHOR_IMAGES_DIR
target_folder = os.path.join(STIM_SET_DIR, 'anchor_grids_inspection')
os.makedirs(target_folder, exist_ok=True)
image_size = (256, 256)

for cat_dir_name in os.listdir(source_folder):
    for obj_dir_name in os.listdir(os.path.join(source_folder, cat_dir_name)):
        create_image_grid(os.path.join(source_folder, cat_dir_name, obj_dir_name), target_folder, required_string=f'{cat_dir_name}-{obj_dir_name}', image_size=image_size)
