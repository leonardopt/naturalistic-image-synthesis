"""
Step 4b — Visual inspection: save side-by-side composites of selected pairs.

Reads the selected-pairs CSV and saves a side-by-side PNG for each object
showing the two endpoint images (median and mean selections). Run this to
visually verify pairs before deciding whether to swap any via pipe_04c.
Output goes into timestamped selected_pair_images_median_* and
selected_pair_images_mean_* directories.
"""
import os
from datetime import datetime

import pandas as pd
from PIL import Image
from naturalistic_image_synthesis.config import ANCHOR_IMAGES_DIR, SELECTED_PAIRS_PATH, STIM_SET_DIR

now = datetime.now()
date_time_str = now.strftime('%Y%m%d_%H%M%S')

# Define the path to the CSV file
csv_file_path = SELECTED_PAIRS_PATH
anchor_images_dir = ANCHOR_IMAGES_DIR

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

select_pair_dir = os.path.join(STIM_SET_DIR, f'selected_pair_images_median_{date_time_str}')
select_pair_dir_mean = os.path.join(STIM_SET_DIR, f'selected_pair_images_mean_{date_time_str}')

os.makedirs(select_pair_dir, exist_ok=True)
os.makedirs(select_pair_dir_mean, exist_ok=True)

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    # Get the paths to the images
    curr_category = row['category']
    curr_obj = row['object']
    img_path_0 = os.path.join(anchor_images_dir, curr_category, curr_obj, row['selected_pair_median_0'])
    img_path_1 = os.path.join(anchor_images_dir, curr_category, curr_obj, row['selected_pair_median_1'])
    img_path_0_mean = os.path.join(anchor_images_dir, curr_category, curr_obj, row['selected_pair_mean_0'])
    img_path_1_mean = os.path.join(anchor_images_dir, curr_category, curr_obj, row['selected_pair_mean_1'])

    if not (os.path.exists(img_path_0) and os.path.exists(img_path_1)):

        print(f"One or both images do not exist for pair {row['selected_pair_median_0'], row['selected_pair_median_1']}. Skipping...")
        print(f'Img 0 path: {img_path_0}')
        print(f'Img 1 path: {img_path_1}')

        continue

    # Load the images
    img_0 = Image.open(img_path_0)
    img_1 = Image.open(img_path_1)
    img_0_mean = Image.open(img_path_0_mean)
    img_1_mean = Image.open(img_path_1_mean)

    # Extract the first 8 characters of selected_pair_median_0 to create the image name
    image_name = os.path.basename(img_path_0)[:7] + ".png"
    image_name_mean = os.path.basename(img_path_0_mean)[:7] + ".png"

    # Create a new blank image with width double of the original images to accommodate both
    new_img = Image.new("RGB", (img_0.width * 2, img_0.height))
    new_img_mean = Image.new("RGB", (img_0_mean.width * 2, img_0_mean.height))

    # Paste images
    new_img.paste(img_0, (0, 0))
    new_img.paste(img_1, (img_0.width, 0))

    new_img_mean.paste(img_0_mean, (0, 0))
    new_img_mean.paste(img_1_mean, (img_0_mean.width, 0))

    # Save the combined image
    new_img.save(os.path.join(select_pair_dir, image_name))
    new_img_mean.save(os.path.join(select_pair_dir_mean, image_name_mean))

    print(f"Combined median image saved as {image_name}")
    print(f"Combined mean image saved as {image_name_mean}")

print('Done!')
