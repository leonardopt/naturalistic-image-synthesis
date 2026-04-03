"""
Step 4a — Select interpolation endpoint pairs (anchor + guide) for each object.

For each object, finds the most perceptually typical image (lowest mean LPIPS
across all pairwise comparisons), then restricts the search for the second
endpoint to the n_most_similar most typical images. The closest image within
that pool becomes the guide. Three backup alternatives are stored per pair.
Attaches prompts from the design parquet, saves selected_pairs_TIMESTAMP.csv,
and writes SELECTED_PAIRS_FILE to .env.
"""
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
from naturalistic_image_synthesis.config import STIM_SET_DIR, update_env_value


def read_lpips_mat_file(folder_path):
    """Return the first LPIPS parquet matrix found in folder_path, or None.

    Args:
        folder_path: Directory to search for a .parquet file whose name contains 'lpips'.

    Returns:
        DataFrame with the LPIPS distance matrix, or None if no file is found.
    """
    for file in os.listdir(folder_path):
        if file.endswith('.parquet') and 'lpips' in file:
            return pd.read_parquet(os.path.join(folder_path, file))
    return None

def fill_scores_dict(anchor_images_dir, similarity_scores_dir):
    """Build a nested {category: {object: lpips_df}} dict from parquet files.

    Matches each parquet filename against the category and object names found
    in anchor_images_dir. Each (category, object) pair is loaded at most once.

    Args:
        anchor_images_dir: Root directory with category/object subdirectories.
        similarity_scores_dir: Directory containing LPIPS parquet files.

    Returns:
        Nested dict mapping category → object → LPIPS DataFrame.
    """
    scores = {}
    processed = set()  # Set to keep track of processed (category, object) pairs

    for category in os.listdir(anchor_images_dir):
        category_path = os.path.join(anchor_images_dir, category)
        if os.path.isdir(category_path) and 'interpolations' not in category:
            scores[category] = {}
            for obj in os.listdir(category_path):
                for file in os.listdir(similarity_scores_dir):
                    if file.endswith('.parquet'):
                        # Normalize and split file name for exact matching
                        parts = file.replace('.parquet', '').lower().split('-')

                        # Prepare normalized category and obj for comparison
                        norm_category = category.lower()
                        norm_obj = obj.lower()

                        # Check for exact match in the parts of the file name
                        if norm_category in parts and norm_obj in parts:
                            # Create a unique identifier for the (category, object) pair
                            identifier = f"{norm_category}-{norm_obj}"

                            # Check if this pair has already been processed
                            if identifier not in processed:
                                # If not processed, read the file and update scores dictionary
                                lpips_df = pd.read_parquet(os.path.join(similarity_scores_dir, file))
                                scores[category][obj] = lpips_df

                                # Mark this pair as processed to avoid double dipping
                                processed.add(identifier)

    return scores


def find_most_and_least_typical_images(base_path):
    """Identify the most and least perceptually typical images for each object.

    Reads LPIPS matrices from similarity_scores_anchors/, computes per-image
    mean and median scores (diagonal excluded), and records the argmin (most
    typical) and argmax (least typical) for both metrics. Saves results to
    typical_and_atypical_images.csv in base_path.

    Args:
        base_path: Root stimulus-set directory (contains anchor_images/ and
            similarity_scores_anchors/).

    Returns:
        Path to the saved CSV file.
    """
    anchor_images_dir = os.path.join(base_path, 'anchor_images')
    similarity_scores_dir = os.path.join(base_path, 'similarity_scores_anchors')
    scores = fill_scores_dict(anchor_images_dir=anchor_images_dir,
                              similarity_scores_dir=similarity_scores_dir)

    image_info = []

    for cat, objs in scores.items():
        for obj, lpips_df in objs.items():
            # Calculate mean and median LPIPS scores, excluding self-comparison
            lpips_array = lpips_df.to_numpy()
            np.fill_diagonal(lpips_array, np.nan)
            mean_scores = np.nanmean(lpips_array, axis=1)
            median_scores = np.nanmedian(lpips_array, axis=1)

            # Identify the most and least typical images
            most_typical_index_mean = np.nanargmin(mean_scores)
            least_typical_index_mean = np.nanargmax(mean_scores)
            most_typical_index_median = np.nanargmin(median_scores)
            least_typical_index_median = np.nanargmax(median_scores)

            # Append the information to the list
            image_info.append({
                'category': cat,
                'object': obj,
                'most_typical_image_mean': lpips_df.index[most_typical_index_mean],
                'mean_lpips_score_most_typical': mean_scores[most_typical_index_mean],
                'most_typical_image_median': lpips_df.index[most_typical_index_median],
                'median_lpips_score_most_typical': median_scores[most_typical_index_median],
                'least_typical_image_mean': lpips_df.index[least_typical_index_mean],
                'mean_lpips_score_least_typical': mean_scores[least_typical_index_mean],
                'least_typical_image_median': lpips_df.index[least_typical_index_median],
                'median_lpips_score_least_typical': median_scores[least_typical_index_median],
            })

    # Convert to DataFrame
    image_info_df = pd.DataFrame(image_info)

    # Save to CSV
    csv_path = os.path.join(base_path, 'typical_and_atypical_images.csv')
    image_info_df.to_csv(csv_path, index=False)

    return csv_path


def select_images_for_interpolation(base_path):
    """Select interpolation endpoint pairs using the global-closest strategy.

    For each object: (1) find the most typical image (lowest mean LPIPS across
    all pairs), (2) find the image closest to it as the second endpoint.
    Seeds are extracted from filenames by regex.

    Args:
        base_path: Root stimulus-set directory (contains anchor_images/ and
            similarity_scores_anchors/).

    Returns:
        Tuple of (selected_pairs_df, all_scores_df).
    """
    anchor_images_dir = os.path.join(base_path, 'anchor_images')
    similarity_scores_dir = os.path.join(base_path, 'similarity_scores_anchors')
    scores = fill_scores_dict(anchor_images_dir=anchor_images_dir,
                              similarity_scores_dir=similarity_scores_dir)
    selected_pairs = []

    all_scores = []  # List to store all scores for all images
    
    for cat, objs in scores.items():
        for obj, lpips_df in objs.items():
            # Convert to numpy array and exclude self-comparisons
            lpips_array = lpips_df.to_numpy()
            np.fill_diagonal(lpips_array, np.nan)

            # Calculate mean and median LPIPS scores for each image
            mean_scores = np.nanmean(lpips_array, axis=1)
            median_scores = np.nanmedian(lpips_array, axis=1)

            # Store all scores
            for index, (mean_score, median_score) in enumerate(zip(mean_scores, median_scores)):
                all_scores.append({
                    'category': cat,
                    'object': obj,
                    'image': lpips_df.index[index],
                    'mean_score': mean_score,
                    'median_score': median_score
                })

            # Identify the most typical image based on the lowest mean score
            most_typical_index_mean = np.nanargmin(mean_scores)
            most_typical_image_mean = lpips_df.index[most_typical_index_mean]

            # Also identify the most typical image based on the lowest median score
            most_typical_index_median = np.nanargmin(median_scores)
            most_typical_image_median = lpips_df.index[most_typical_index_median]

            # Calculate LPIPS scores to the most typical images (mean and median)
            scores_to_most_typical_mean = lpips_array[most_typical_index_mean, :]
            scores_to_most_typical_median = lpips_array[most_typical_index_median, :]

            # Exclude the most typical images from consideration for the second images
            scores_to_most_typical_mean[most_typical_index_mean] = np.nan
            scores_to_most_typical_median[most_typical_index_median] = np.nan

            # Find the images closest to the most typical images (mean and median)
            second_image_index_mean = np.nanargmin(scores_to_most_typical_mean)
            second_image_mean = lpips_df.index[second_image_index_mean]

            second_image_index_median = np.nanargmin(scores_to_most_typical_median)
            second_image_median = lpips_df.index[second_image_index_median]

            # Concatenate file names for selected pairs
            selected_pair_mean_0 = f"{most_typical_image_mean}"
            selected_pair_mean_1 = f"{second_image_mean}"

            selected_pair_median_0 = f"{most_typical_image_median}"
            selected_pair_median_1 = f"{second_image_median}"

            # extract seeds from image name
            regex = r'\d+'
            match_0 = re.search(regex, selected_pair_median_0)
            seed_0_median = int(match_0.group())
            match_1 = re.search(regex, selected_pair_median_1)
            seed_1_median = int(match_1.group())

            # Store information about the selected pairs
            selected_pairs.append({
                'category': cat,
                'object': obj,
                'most_typical_image_mean': most_typical_image_mean,
                'closest_to_most_typical_image_mean': second_image_mean,
                'lpips_score_between_them_mean': scores_to_most_typical_mean[second_image_index_mean],
                'selected_pair_mean_0': selected_pair_mean_0,
                'selected_pair_mean_1': selected_pair_mean_1,
                'most_typical_image_median': most_typical_image_median,
                'closest_to_most_typical_image_median': second_image_median,
                'lpips_score_between_them_median': scores_to_most_typical_median[second_image_index_median],
                'selected_pair_median_0': selected_pair_median_0,
                'selected_pair_median_1': selected_pair_median_1,
                'seed_0_median': seed_0_median,
                'seed_1_median': seed_1_median,
            })

    # Convert results to DataFrames
    selected_pairs_df = pd.DataFrame(selected_pairs)
    all_scores_df = pd.DataFrame(all_scores).sort_values(by=['category', 'object', 'mean_score', 'median_score'])

    # Return DataFrames
    return selected_pairs_df, all_scores_df


def select_images_for_interpolation_from_most_similar(base_path, n_most_similar):
    """Select interpolation endpoint pairs restricting the search to the n most typical images.

    For each object: (1) find the most typical image (lowest mean/median LPIPS),
    (2) restrict the second-endpoint search to the n_most_similar most typical
    images by mean/median score, (3) pick the one closest to the anchor.
    Stores three backup candidates per pair. Seeds extracted from filenames by regex.

    Args:
        base_path: Root stimulus-set directory (contains anchor_images/ and
            similarity_scores_anchors/).
        n_most_similar: Pool size for second-endpoint candidate search.

    Returns:
        Tuple of (selected_pairs_df, all_scores_df).
    """
    anchor_images_dir = os.path.join(base_path, 'anchor_images')
    similarity_scores_dir = os.path.join(base_path, 'similarity_scores_anchors')
    scores = fill_scores_dict(anchor_images_dir=anchor_images_dir,
                              similarity_scores_dir=similarity_scores_dir)
    selected_pairs = []

    all_scores = []  # List to store all scores for all images

    for cat, objs in scores.items():
        for obj, lpips_df in objs.items():
            # Convert to numpy array and exclude self-comparisons
            lpips_array = lpips_df.to_numpy()
            np.fill_diagonal(lpips_array, np.nan)

            # Calculate mean and median LPIPS scores for each image
            mean_scores = np.nanmean(lpips_array, axis=1)
            median_scores = np.nanmedian(lpips_array, axis=1)

            # Mean Scores Processing
            # Identify the most typical image based on the lowest mean score
            most_typical_index_mean = np.nanargmin(mean_scores)
            most_typical_image_mean = lpips_df.index[most_typical_index_mean]

            # Calculate LPIPS scores to the most typical image (mean)
            scores_to_most_typical_mean = np.copy(lpips_array[most_typical_index_mean, :])
            scores_to_most_typical_mean[most_typical_index_mean] = np.nan

            # Identify the 10 most typical images based on their mean LPIPS scores
            sorted_indices_by_mean = np.argsort(mean_scores)
            top_n_typical_indices_mean = sorted_indices_by_mean[:n_most_similar]

            # Calculate the LPIPS scores between the most typical image and these top n images (mean)
            lpips_scores_to_first_typical_mean = lpips_array[most_typical_index_mean, top_n_typical_indices_mean]

            # Find the index of the image among the top n that has the highest similarity to the most typical image (mean)
            second_image_index_mean = np.nanargmin(lpips_scores_to_first_typical_mean)
            second_image_index_global_mean = top_n_typical_indices_mean[second_image_index_mean]
            second_image_mean = lpips_df.index[second_image_index_global_mean]

            # Median Scores Processing
            # Identify the most typical image based on the lowest median score
            most_typical_index_median = np.nanargmin(median_scores)
            most_typical_image_median = lpips_df.index[most_typical_index_median]

            # Calculate LPIPS scores to the most typical image (median)
            scores_to_most_typical_median = np.copy(lpips_array[most_typical_index_median, :])
            scores_to_most_typical_median[most_typical_index_median] = np.nan

            # Identify the n most typical images based on their median LPIPS scores
            sorted_indices_by_median = np.argsort(median_scores)
            top_n_typical_indices_median = sorted_indices_by_median[:n_most_similar]

            # Calculate the LPIPS scores between the most typical image and these top n images (median)
            lpips_scores_to_first_typical_median = lpips_array[most_typical_index_median, top_n_typical_indices_median]

            # Find the index of the image among the top n that has the highest similarity to the most typical image (median)
            second_image_index_median = np.nanargmin(lpips_scores_to_first_typical_median)
            second_image_index_global_median = top_n_typical_indices_median[second_image_index_median]
            second_image_median = lpips_df.index[second_image_index_global_median]

            # Concatenate file names for selected pairs
            selected_pair_mean_0 = f"{most_typical_image_mean}"
            selected_pair_mean_1 = f"{second_image_mean}"
            selected_pair_median_0 = f"{most_typical_image_median}"
            selected_pair_median_1 = f"{second_image_median}"

            # Extract seeds from image name
            regex = r'\d+'
            match_0 = re.search(regex, selected_pair_median_0)
            seed_0_median = int(match_0.group()) if match_0 else None
            match_1 = re.search(regex, selected_pair_median_1)
            seed_1_median = int(match_1.group()) if match_1 else None

            match_0 = re.search(regex, selected_pair_mean_0)
            match_1 = re.search(regex, selected_pair_mean_1)
            seed_0_mean = int(match_0.group()) if match_0 else None
            seed_1_mean = int(match_1.group()) if match_1 else None

            # For Mean Scores - Getting backups
            sorted_indices_mean = np.argsort(lpips_scores_to_first_typical_mean)[:4]  # Includes the primary and next 3
            backup_images_mean_indices = sorted_indices_mean[1:]  # Exclude the primary which is the first one
            backup_images_mean = lpips_df.index[top_n_typical_indices_mean[backup_images_mean_indices]].tolist()

            # For Median Scores - Getting backups
            sorted_indices_median = np.argsort(lpips_scores_to_first_typical_median)[
                                    :4]  # Includes the primary and next 3
            backup_images_median_indices = sorted_indices_median[1:]  # Exclude the primary which is the first one
            backup_images_median = lpips_df.index[top_n_typical_indices_median[backup_images_median_indices]].tolist()
            # Extracting backup images and seeds for mean scores
            backup_images_mean = lpips_df.index[top_n_typical_indices_mean[sorted_indices_mean[1:]]].tolist()
            seeds_mean = [int(re.search(r'\d+', img).group()) if re.search(r'\d+', img) else None for img in
                          backup_images_mean]

            # Extracting backup images and seeds for median scores
            backup_images_median = lpips_df.index[top_n_typical_indices_median[sorted_indices_median[1:]]].tolist()
            seeds_median = [int(re.search(r'\d+', img).group()) if re.search(r'\d+', img) else None for img in
                            backup_images_median]

            # Assuming cat, obj are defined earlier
            # Store information about the selected pairs
            selected_pairs.append({
                'category': cat,
                'object': obj,
                'use_backup': 0,
                'selected_pair_mean_0': selected_pair_mean_0,
                'selected_pair_mean_1': selected_pair_mean_1,
                'seed_0_mean': seed_0_mean,
                'seed_1_mean': seed_1_mean,
                'backup_image_mean_1': backup_images_mean[0] if len(backup_images_mean) > 0 else None,
                'backup_image_mean_2': backup_images_mean[1] if len(backup_images_mean) > 1 else None,
                'backup_image_mean_3': backup_images_mean[2] if len(backup_images_mean) > 2 else None,
                'seed_mean_backup_1': seeds_mean[0] if len(seeds_mean) > 0 else None,
                'seed_mean_backup_2': seeds_mean[1] if len(seeds_mean) > 1 else None,
                'seed_mean_backup_3': seeds_mean[2] if len(seeds_mean) > 2 else None,
                'selected_pair_median_0': selected_pair_median_0,
                'selected_pair_median_1': selected_pair_median_1,
                'seed_0_median': seed_0_median,
                'seed_1_median': seed_1_median,
                'backup_image_median_1': backup_images_median[0] if len(backup_images_median) > 0 else None,
                'backup_image_median_2': backup_images_median[1] if len(backup_images_median) > 1 else None,
                'backup_image_median_3': backup_images_median[2] if len(backup_images_median) > 2 else None,
                'seed_median_backup_1': seeds_median[0] if len(seeds_median) > 0 else None,
                'seed_median_backup_2': seeds_median[1] if len(seeds_median) > 1 else None,
                'seed_median_backup_3': seeds_median[2] if len(seeds_median) > 2 else None,
                'most_typical_image_mean': most_typical_image_mean,
                'closest_to_most_typical_image_mean': second_image_mean,
                'lpips_score_between_them_mean': scores_to_most_typical_mean[second_image_index_global_mean],
                'most_typical_image_median': most_typical_image_median,
                'closest_to_most_typical_image_median': second_image_median,
                'lpips_score_between_them_median': scores_to_most_typical_median[second_image_index_global_median],
            })

            # Store all scores
            for index, (mean_score, median_score, score_most_typical_mean, score_most_typical_median) in enumerate(zip(mean_scores, median_scores, scores_to_most_typical_mean, scores_to_most_typical_median)):
                all_scores.append({
                    'category': cat,
                    'object': obj,
                    'image': lpips_df.index[index],
                    'mean_score': mean_score,
                    'median_score': median_score,
                    'score_to_most_typical_mean': score_most_typical_mean,
                    'score_to_most_typical_median': score_most_typical_median,
                })

    # Convert results to DataFrames
    selected_pairs_df = pd.DataFrame(selected_pairs)
    all_scores_df = pd.DataFrame(all_scores).sort_values(by=['category', 'object', 'mean_score', 'score_to_most_typical_mean'])

    # Return DataFrames
    return selected_pairs_df, all_scores_df

# Directories
now = datetime.now()
date_time_str = now.strftime('%Y%m%d_%H%M%S')
base_path = STIM_SET_DIR

# select_unique_pairs(base_path=base_path, threshold=False, stat_threshold='median')
image_info_df = find_most_and_least_typical_images(base_path)
# Select images for interpolation
selected_pairs_df, all_scores_df = select_images_for_interpolation_from_most_similar(base_path, n_most_similar=5)
# Add information to the dataframe
stimulus_set_design = pd.read_parquet(os.path.join(base_path, 'stimulus_set_design.parquet'))

prompts = []
negative_prompts = []
for i, row in selected_pairs_df.iterrows():
    # Find matching prompts based on 'object' column
    curr_prompt = stimulus_set_design.loc[stimulus_set_design['object'] == row['object'], 'prompt'].unique()
    curr_negative_prompt = stimulus_set_design.loc[
        stimulus_set_design['object'] == row['object'], 'negative_prompt'].unique()

    # Assuming there is always one unique prompt and negative_prompt per object
    prompts.append(curr_prompt[0] if len(curr_prompt) > 0 else None)
    negative_prompts.append(curr_negative_prompt[0] if len(curr_negative_prompt) > 0 else None)

# Assign the collected prompts back to the original DataFrame
selected_pairs_df['prompt'] = prompts
selected_pairs_df['negative_prompt'] = negative_prompts

selected_inteprols_file_name = f'selected_pairs_for_interpolation_{date_time_str}.csv'
selected_pairs_df.to_csv(os.path.join(base_path,selected_inteprols_file_name), index=False)
update_env_value('SELECTED_PAIRS_FILE', selected_inteprols_file_name)
print(f'Saved in {os.path.join(base_path, selected_inteprols_file_name)}')
all_scores_df.to_csv(os.path.join(base_path, 'all_image_scores.csv'), index=False)
print(f'Saved in {os.path.join(base_path, "all_image_scores.csv")}')
