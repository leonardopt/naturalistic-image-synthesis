"""
Step 4c — Apply manual pair substitutions and write the final-selection CSV.

For each row where use_backup > 0, replaces the guide image with the
corresponding backup_image_*_N. Saves side-by-side composites for changed
pairs into changed_pair_images_median_* and changed_pair_images_mean_*
directories, writes the updated CSV with final_select_* columns, and updates
SELECTED_PAIRS_FILE in .env to point to the final-selection CSV.
"""
import os
from datetime import datetime

import pandas as pd
from PIL import Image
from naturalistic_image_synthesis.config import ANCHOR_IMAGES_DIR, SELECTED_PAIRS_PATH, STIM_SET_DIR, update_env_value

now = datetime.now()
date_time_str = now.strftime('%Y%m%d_%H%M%S')

# Define the path to the CSV file
selected_pairs_file = os.path.basename(SELECTED_PAIRS_PATH)
base_folder = STIM_SET_DIR
csv_file_path = SELECTED_PAIRS_PATH
anchor_images_dir = ANCHOR_IMAGES_DIR

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

select_pair_dir = os.path.join(base_folder, f'changed_pair_images_median_{date_time_str}')
select_pair_dir_mean = os.path.join(base_folder, f'changed_pair_images_mean_{date_time_str}')

os.makedirs(select_pair_dir, exist_ok=True)
os.makedirs(select_pair_dir_mean, exist_ok=True)

# Initialize new columns in the DataFrame for final selections
df['final_select_mean_0'] = None
df['final_select_mean_1'] = None
df['final_select_mean_seed_0'] = None
df['final_select_mean_seed_1'] = None


# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    # Determine if this pair should use backup
    use_backup = row.get('use_backup', 0)  # Assuming there's an 'is_flagged' column
    print(f'Use backup {use_backup}')
    if use_backup:
        # Select either the primary or backup image based on flag
        img_path_1 = os.path.join(anchor_images_dir, row['category'], row['object'],
                                  row[f'backup_image_median_{use_backup}'] if use_backup else row[
                                      'selected_pair_median_1'])
        img_path_1_mean = os.path.join(anchor_images_dir, row['category'], row['object'],
                                       row[f'backup_image_mean_{use_backup}'] if use_backup else row[
                                           'selected_pair_mean_1'])

        # Paths to the always-selected '0' images remain unchanged
        img_path_0 = os.path.join(anchor_images_dir, row['category'], row['object'], row['selected_pair_median_0'])
        img_path_0_mean = os.path.join(anchor_images_dir, row['category'], row['object'], row['selected_pair_mean_0'])

        if not (os.path.exists(img_path_0) and os.path.exists(img_path_1)):
            print(
                f"One or both images do not exist for pair {row['selected_pair_median_0'], row['selected_pair_median_1']}. Skipping...")
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

        df.loc[index, 'final_select_mean_0'] = row['selected_pair_mean_0']
        df.loc[index, 'final_select_mean_1'] = row[f'backup_image_mean_{use_backup}']
        df.loc[index, 'final_select_mean_seed_0'] = row['seed_0_mean']
        df.loc[index, 'final_select_mean_seed_1'] = row[f'seed_mean_backup_{use_backup}']
    else:
        # No backup, use the original selections
        df.loc[index, 'final_select_mean_0'] = row['selected_pair_mean_0']
        df.loc[index, 'final_select_mean_1'] = row['selected_pair_mean_1']
        df.loc[index, 'final_select_mean_seed_0'] = row['seed_0_mean']
        df.loc[index, 'final_select_mean_seed_1'] = row['seed_1_mean']

final_selection_file = selected_pairs_file[:-4] + '_final_selection.csv'
df.to_csv(os.path.join(base_folder, final_selection_file), index=False)
update_env_value('SELECTED_PAIRS_FILE', final_selection_file)

print('Done!')
