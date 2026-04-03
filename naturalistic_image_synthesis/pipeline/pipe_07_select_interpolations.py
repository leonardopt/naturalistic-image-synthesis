"""
Step 7 — Select 10 representative frames per interpolation sequence.

For each object, reads LPIPS distances relative to the anchor frame (index 000)
and greedily picks 10 frames whose scores are as linearly spaced as possible
between the minimum and maximum. Supports manual index swaps via an optional
excluded_{INTERPOL_DIR_NAME}.csv file (columns: object, excluded, swap_with).
Selected frames are copied to selected_{INTERPOL_DIR_NAME}/category/object/
with per-object info.csv and interpolation_backups.csv metadata files.
"""
import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
from naturalistic_image_synthesis.config import (
    INTERPOL_DIR,
    INTERPOL_DIR_NAME,
    SELECTED_INTERPOL_DIR,
    SIMILARITY_SCORES_INTERPOL_DIR,
    STIM_SET_DIR,
)

def select_indices_linear_increase(similarity_scores, n_images, backup_range=0.01):
    """Greedily select n_images frame indices whose LPIPS scores are linearly spaced.

    Generates evenly-spaced target scores between the sequence min and max, then
    picks the closest unused index to each target. Index 0 (anchor, score=0) is
    always included first.

    Args:
        similarity_scores: DataFrame with an LPIPS score column at column index 1.
        n_images: Number of frames to select.
        backup_range: LPIPS tolerance for recording nearby frames as swap candidates.

    Returns:
        Tuple of (selected_indices, backups_dict) where selected_indices is a sorted
        list of integer frame indices and backups_dict maps each selected index to a
        list of nearby candidate indices within backup_range.
    """
    np_scores = similarity_scores.to_numpy()[:, 1]
    min_score = np_scores[1]
    max_score = np_scores[-1]
    target_scores = np.linspace(min_score, max_score, n_images-1, endpoint=True)
    target_scores = np.insert(target_scores, 0, 0)
    selected_indices = []
    backups_dict = {}

    for target in target_scores:
        # Calculate the absolute differences and get the sorted indices by their closeness
        distances = np.abs(np_scores - target)
        sorted_indices = np.argsort(distances)
        for idx in sorted_indices:
            if idx not in selected_indices:
                selected_indices.append(idx)
                backup_indices = np.where((distances <= backup_range) & (distances > 0))[0]
                backups_dict[idx] = backup_indices.tolist()
                break

    assert len(selected_indices) == n_images, 'Fewer indices than required selected.'
    selected_indices = sorted(selected_indices)
    return selected_indices, backups_dict

def find_similarity_scores_file(folder, category_name, object_name):
    """Return the filename of the LPIPS parquet for a given category-object pair.

    Args:
        folder: Directory containing LPIPS parquet files.
        category_name: Category string to match in the filename.
        object_name: Object string to match in the filename.

    Returns:
        Matching filename (str), or None if not found.
    """
    for file in os.listdir(folder):
        identifier = f'{category_name}-{object_name}.parquet'
        if identifier in file:
            return file

now = datetime.now()

interpolations_path = INTERPOL_DIR
similarity_scores_path = SIMILARITY_SCORES_INTERPOL_DIR

# check if there is a swap file 
swap_files = False
excluded_interpols_file = os.path.join(STIM_SET_DIR, f'excluded_{INTERPOL_DIR_NAME}.csv')
if os.path.exists(excluded_interpols_file):
    print(f'file with interpolation indices to swap found in {excluded_interpols_file}')
    swap_df = pd.read_csv(excluded_interpols_file)
    swap_files = True
    for index, row in swap_df.iterrows():
    # Attempt to convert or parse 'excluded' column
        try:
            # If 'excluded' is a single integer in string form, convert directly
            excluded = [int(x.strip()) for x in str(row['excluded']).split(',')] if pd.notna(row['excluded']) else ''
            swap_df.at[index, 'excluded'] = excluded
        except ValueError:
            # Handle the case where conversion is not possible (e.g., malformed data)
            print(f"Warning: Unable to parse 'excluded' for row {index}")
            swap_df.at[index, 'excluded'] = []
        
        # Repeat the process for 'swap_with' column
        try:
            swap_with = [int(x.strip()) for x in str(row['swap_with']).split(',')] if pd.notna(row['swap_with']) else []
            swap_df.at[index, 'swap_with'] = swap_with
        except ValueError:
            print(f"Warning: Unable to parse 'swap_with' for row {index}")
            swap_df.at[index, 'swap_with'] = []

net = 'squeeze'

for cat_name in os.listdir(interpolations_path):
    for obj_name in os.listdir(os.path.join(interpolations_path, cat_name)):
        object_dir_path = os.path.join(interpolations_path, cat_name, obj_name)
        dst_folder = os.path.join(SELECTED_INTERPOL_DIR, cat_name, obj_name)
        os.makedirs(dst_folder, exist_ok=True)
        scores_path = find_similarity_scores_file(similarity_scores_path, cat_name, obj_name)
        assert scores_path, f'scores path:{scores_path} for {obj_name}'
        # Load lpips scores
        all_scores = pd.read_parquet(os.path.join(similarity_scores_path, scores_path))
        # Subselect scores only for the first image:
        similarity_scores = all_scores.loc[:, all_scores.columns.str.endswith('000.png')]
        similarity_scores = similarity_scores.rename(columns={similarity_scores.columns[0]: "ref_img"})        
        similarity_scores = similarity_scores.reset_index(names='img_interpol_name') # reset index
        n_images = 10
        backup_range = 0.005
        selected_indices, backups_dict = select_indices_linear_increase(similarity_scores, n_images, backup_range)

        if swap_files:
            # Ensure we're working with the correct rows for the current object
            swap_rows = swap_df[swap_df['object'] == obj_name]
            for _, swap_row in swap_rows.iterrows():
                excluded_indices = swap_row['excluded']
                swap_with_indices = swap_row['swap_with']
                if excluded_indices and swap_with_indices:
                    for old_idx, new_idx in zip(excluded_indices, swap_with_indices):
                        if old_idx in selected_indices:
                            selected_indices[selected_indices.index(old_idx)] = new_idx
                            print(f'{cat_name}-{obj_name}: swapped idx {old_idx} with {new_idx}')
                    selected_indices.sort()
                    assert len(selected_indices) == 10, f'Selected indices are not 10, but {len(selected_indices)}'
            else:
                print('No indices to exclude found. Maintain original.')

        # Get images
        selected_images = similarity_scores['img_interpol_name'][selected_indices]
        assert len(selected_images) == 10, f'Selected images are not 10, but {len(selected_images)}'
        print(f'Selected indices: {selected_indices}')
        print(f'Selected images: {selected_images}')
        print(f'Selected LPIPS vals: {similarity_scores["ref_img"][selected_indices]}')



        for image_file in selected_images:

            try:
                img_src_path = os.path.join(object_dir_path, image_file)
                img_dst_path = os.path.join(dst_folder, image_file)
            except Exception as e:
                print(f'Error while loading the files: {e}\nFile path to load: {object_dir_path, image_file}')

            # Copy the file
            shutil.copy2(img_src_path, img_dst_path)
            print(f'Copied {image_file} to {dst_folder}')

        assert len(selected_images) == 10, f'only {len(selected_images)} images selected for {cat_name, obj_name}'

        # save information about the selected images
        info = {
            'selected_indices': selected_indices,
            'selected_images': selected_images,
            'similarity_scores': similarity_scores["ref_img"][selected_indices]
        }
        pd.DataFrame(info).to_csv(os.path.join(dst_folder, 'info.csv'), index=False)
        
        backup_info = []
        for main_idx, backup_idxs in backups_dict.items():
            main_img = similarity_scores.iloc[main_idx]['img_interpol_name']
            main_score = similarity_scores.iloc[main_idx]['ref_img']
            backups = similarity_scores.iloc[backup_idxs][['img_interpol_name', 'ref_img']]
            for _, backup_row in backups.iterrows():
                backup_info.append({
                    'Selected Interpolation ID': main_img,
                    'Selected Interpolation LPIPS score': main_score,
                    'Backup Image IDs': backup_row['img_interpol_name'],
                    f'Backup Image scores (\u00B1{backup_range})': backup_row['ref_img']
                })
        pd.DataFrame(backup_info).to_csv(os.path.join(dst_folder, 'interpolation_backups.csv'), index=False)        
computing_time = datetime.now() - now
print(f'Total computing time: {computing_time}')
