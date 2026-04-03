"""
Create annotated image grids or strips from a flat stimulus directory.

Reads anchor images for a specified category/object, sorts them by a chosen
LPIPS score column (mean or median), and arranges them in a grid or horizontal
strip with filenames annotated in each cell. Primarily used to visually inspect
how anchor images rank by perceptual typicality before pair selection.
"""
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import math
from naturalistic_image_synthesis.config import STIM_SET_DIR, ANCHOR_IMAGES_DIR  # generation_20240326_180652


def create_image_grid(source_folder, target_folder, image_files, scores, required_string, image_size=(200, 200), font_size=20, single_strip=False, str_name=''):
    # Ensure target folder exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

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
            annotate_filename = image_file
            draw.text((x + 5, y + 5), annotate_filename, fill="white", font=font)



    # Save the grid image
    grid_image_path = os.path.join(target_folder, f'{required_string}_image_grid{str_name}.png')
    grid_image.save(grid_image_path)
    print(f"Grid image saved to {grid_image_path}")

# Load scores
scores_dir = os.path.join(STIM_SET_DIR, 'all_image_scores.csv')
scores = pd.read_csv(scores_dir)
category = 'animal'
object = 'horse'
subset_scores = scores[scores['object']==object]
source_folder = os.path.join(ANCHOR_IMAGES_DIR, category, object)

target_folder = os.path.join(STIM_SET_DIR, 'image_grids')
os.makedirs(target_folder, exist_ok=True)
image_size = (256, 256)
# Mean
subset_scores = subset_scores.sort_values(by='mean_score')
create_image_grid(source_folder=source_folder,
                  target_folder=target_folder,
                  image_files=subset_scores['image'],
                  scores=subset_scores['mean_score'],
                  required_string = category+'-'+object+'-mean',
                  image_size=image_size)
# Median
subset_scores = subset_scores.sort_values(by='median_score')
create_image_grid(source_folder=source_folder,
                  target_folder=target_folder,
                  image_files=subset_scores['image'],
                  scores=subset_scores['median_score'],
                  required_string = category+'-'+object+'-median',
                  image_size=image_size)