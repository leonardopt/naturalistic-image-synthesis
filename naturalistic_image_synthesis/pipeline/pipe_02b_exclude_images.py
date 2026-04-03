"""Move user-flagged anchor images out of the active anchor set.

This script expects a user-provided CSV file at:
    STIM_SET_DIR/excluded_images.csv

The pipeline does not generate that CSV automatically. Create it manually if
you want to exclude specific anchor images after inspecting a run.
"""

import os
import pandas as pd
import shutil
from naturalistic_image_synthesis.config import ANCHOR_IMAGES_DIR, STIM_SET_DIR

stim_set_dir = STIM_SET_DIR
excluded_image_file = os.path.join(stim_set_dir, 'excluded_images.csv')
excluded_image_dir = os.path.join(stim_set_dir, 'excluded_images')
os.makedirs(excluded_image_dir, exist_ok=True)


df = pd.read_csv(excluded_image_file)
df_exclusion = df[df['exclude'] == 1]
for idx, row in df_exclusion.iterrows():
    img_dir = os.path.join(ANCHOR_IMAGES_DIR, row['category'], row['object'], row['ID'] + '.png')
    if os.path.isfile(img_dir):
        shutil.move(img_dir, os.path.join(excluded_image_dir, row['ID'] + '.png'))
    print(f'Moving {img_dir} to {os.path.join(excluded_image_dir,row["ID"] + ".png") }')
