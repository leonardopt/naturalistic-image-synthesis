"""
Step 4d — Final inspection: save side-by-side composites of the definitive pairs.

Same as pipe_04b but reads the final-selection CSV (output of pipe_04c). Applies
use_backup flags to select the correct second endpoint for each object, then saves
composite images for a last visual check before running interpolations.
"""
import os
from datetime import datetime

import pandas as pd
from PIL import Image
from naturalistic_image_synthesis.config import ANCHOR_IMAGES_DIR, SELECTED_PAIRS_PATH, STIM_SET_DIR

# Setup initial variables and read the DataFrame
now = datetime.now()
date_time_str = now.strftime('%Y%m%d_%H%M%S')
base_folder = STIM_SET_DIR
csv_file_path = SELECTED_PAIRS_PATH
anchor_images_dir = ANCHOR_IMAGES_DIR
df = pd.read_csv(csv_file_path)

# Output directories
select_pair_dir = os.path.join(base_folder, f"selected_pair_images_median_{date_time_str}")
select_pair_dir_mean = os.path.join(base_folder, f"selected_pair_images_mean_{date_time_str}")

# Ensure output directories exist
os.makedirs(select_pair_dir, exist_ok=True)
os.makedirs(select_pair_dir_mean, exist_ok=True)


# Function to process and save images
def process_and_save_images(df_row, img_paths, save_dirs, image_name_median, image_name_mean):
    """Load two endpoint images and save them as a single side-by-side composite.

    Args:
        df_row: DataFrame row (unused beyond context; kept for call-site consistency).
        img_paths: Dict with keys 'path_0_median', 'path_1_median', 'path_0_mean',
            'path_1_mean' pointing to the four endpoint PNGs.
        save_dirs: Dict with keys 'dir_median' and 'dir_mean' for output directories.
        image_name_median: Output filename for the median-pair composite.
        image_name_mean: Output filename for the mean-pair composite.

    Returns:
        True on success, False if any image could not be loaded.
    """
    # Load the images
    try:
        img_0_median = Image.open(img_paths['path_0_median'])
        img_1_median = Image.open(img_paths['path_1_median'])
        img_0_mean = Image.open(img_paths['path_0_mean'])
        img_1_mean = Image.open(img_paths['path_1_mean'])
    except FileNotFoundError as e:
        print(f"Error loading images: {e}")
        return False

    # Create a new blank image for both median and mean images
    new_img_median = Image.new("RGB", (img_0_median.width * 2, img_0_median.height))
    new_img_mean = Image.new("RGB", (img_0_mean.width * 2, img_0_mean.height))

    # Paste images
    new_img_median.paste(img_0_median, (0, 0))
    new_img_median.paste(img_1_median, (img_0_median.width, 0))
    new_img_mean.paste(img_0_mean, (0, 0))
    new_img_mean.paste(img_1_mean, (img_0_mean.width, 0))

    # Save the combined image
    new_img_median.save(os.path.join(save_dirs['dir_median'], image_name_median))
    new_img_mean.save(os.path.join(save_dirs['dir_mean'], image_name_mean))

    print(f"Combined median image saved as {image_name_median}")
    print(f"Combined mean image saved as {image_name_mean}")
    return True


# Iterate over each row in the DataFrame to process images
for index, row in df.iterrows():
    use_backup = row.get('use_backup', 0)  # Assuming there's an 'use_backup' column
    print(f'Use backup {use_backup}')

    # Define paths for images
    img_paths = {
        'path_0_median': os.path.join(anchor_images_dir, row['category'], row['object'], row['selected_pair_median_0']),
        'path_1_median': os.path.join(anchor_images_dir, row['category'], row['object'],
                                      row[f'backup_image_median_{use_backup}'] if use_backup else row[
                                          'selected_pair_median_1']),
        'path_0_mean': os.path.join(anchor_images_dir, row['category'], row['object'], row['selected_pair_mean_0']),
        'path_1_mean': os.path.join(anchor_images_dir, row['category'], row['object'],
                                    row[f'backup_image_mean_{use_backup}'] if use_backup else row[
                                        'selected_pair_mean_1']),
    }

    save_dirs = {'dir_median': select_pair_dir, 'dir_mean': select_pair_dir_mean}
    image_name_median = os.path.basename(img_paths['path_0_median'])[:7] + ".png"
    image_name_mean = os.path.basename(img_paths['path_0_mean'])[:7] + ".png"

    # Process and save images, skip to next if any issue
    if not process_and_save_images(row, img_paths, save_dirs, image_name_median, image_name_mean):
        continue

print('Done! Processing complete and DataFrame updated.')
