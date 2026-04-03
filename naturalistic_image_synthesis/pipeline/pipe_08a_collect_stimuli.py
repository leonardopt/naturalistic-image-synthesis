"""
Step 8a — Assemble the final stimulus set.

Reads selected frames from selected_{INTERPOL_DIR_NAME}/, resizes each to
512×512 px (LANCZOS), renames to the canonical CAT-OBJ-NN.png format (NN =
00–09), and collects everything into a flat directory
stimulus_set_from_interpolations_{timestamp}/. Raises FileExistsError on any
filename collision. Saves stimulus_set_info.csv mapping old to new names.
"""
import os
from datetime import datetime

import pandas as pd
from PIL import Image
from naturalistic_image_synthesis.config import (
    ANCHOR_IMAGES_DIR,
    INTERPOL_DIR_NAME,
    SELECTED_INTERPOL_DIR,
    SIMILARITY_SCORES_INTERPOL_DIR,
    STIM_SET_FROM_INTERPOL_DIR,
)

from PIL import Image
import os

SELECTED_IMAGES_DIR = SELECTED_INTERPOL_DIR
SIMILARITY_SCORES_DIR = SIMILARITY_SCORES_INTERPOL_DIR


# Example usage
now = datetime.now()
date_time_str = now.strftime('%Y%m%d_%H%M%S')
interpols_id = INTERPOL_DIR_NAME.replace('interpolations_', '')
FINAL_STIM_SET = STIM_SET_FROM_INTERPOL_DIR
os.makedirs(FINAL_STIM_SET, exist_ok=False)
log = {'category':[], 'object':[], 'old_name':[], 'new_name':[]}
size = (512, 512)

for cat_name in os.listdir(SELECTED_IMAGES_DIR):
    for obj_name in os.listdir(os.path.join(SELECTED_IMAGES_DIR, cat_name)):
        object_dir_path = os.path.join(SELECTED_IMAGES_DIR, cat_name, obj_name)
        interpolated_images_names = sorted([file for file in os.listdir(object_dir_path) if file.endswith('.png')])

        for i, img_name in enumerate(interpolated_images_names):
            new_img_name = f'{img_name[:7]}-{i:02}.png'
            file_path = os.path.join(object_dir_path, img_name)
            resized_path = os.path.join(FINAL_STIM_SET, new_img_name)
            if os.path.exists(resized_path):
                raise FileExistsError(f'Output file already exists: {resized_path}')
            # Open the image
            with Image.open(file_path) as img:
                # Resize the image
                img_resized = img.resize(size, Image.Resampling.LANCZOS)
                # Save the resized image
                img_resized.save(resized_path)
            # shutil.copy2(os.path.join(object_dir_path, img_name), os.path.join(FINAL_STIM_SET, new_img_name))
                print(f'File {img_name} copied in {os.path.join(FINAL_STIM_SET, new_img_name)}')
            log['category'].append(cat_name)
            log['object'].append(obj_name)
            log['old_name'].append(img_name)
            log['new_name'].append(new_img_name)

log_df = pd.DataFrame(log)
log_df.to_csv(os.path.join(FINAL_STIM_SET, 'stimulus_set_info.csv'), index=False)
