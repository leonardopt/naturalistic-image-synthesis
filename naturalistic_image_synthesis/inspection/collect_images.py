"""
Copy all anchor images into a single flat directory for manual inspection.

Traverses the nested anchor_images/category/object/ structure and copies every
image into a flat anchor_images_inspection/ directory, prepending each filename
with its 3-letter category and object abbreviations so files remain identifiable
when browsed together.
"""
import os
import shutil
from naturalistic_image_synthesis.utils import get_unique_abbr
from naturalistic_image_synthesis.config import STIM_SET_DIR, ANCHOR_IMAGES_DIR  # generation_20240326_180652

stim_dir_path = ANCHOR_IMAGES_DIR
dst_dir = os.path.join(STIM_SET_DIR, 'anchor_images_inspection')
os.makedirs(dst_dir, exist_ok=False)

#
categ_short = get_unique_abbr(category_list=os.listdir(stim_dir_path), num_initials=3)

for cat_dir_name in os.listdir(stim_dir_path):
    for obj_dir_name in os.listdir(os.path.join(stim_dir_path, cat_dir_name)):
        obj_short = get_unique_abbr(category_list=os.listdir(os.path.join(stim_dir_path, cat_dir_name)), num_initials=3)

        object_dir_path = os.path.join(stim_dir_path, cat_dir_name, obj_dir_name)
        print(f'Processing directory: {object_dir_path}')
        for file in os.listdir(object_dir_path):
            if not obj_dir_name in file:
                new_file_id = '-'.join([categ_short[cat_dir_name], obj_short[obj_dir_name], str(file)]).upper()
            shutil.copy2(os.path.join(object_dir_path, file), os.path.join(dst_dir, new_file_id))
            print(f'File {new_file_id} copied in {os.path.join(dst_dir, new_file_id)}')
