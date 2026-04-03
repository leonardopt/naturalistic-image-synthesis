"""
Shared utility functions used across the pipeline.

generate_ids(): adds an ID column to the design DataFrame by combining
  3-letter abbreviations of category, object, and seed into CAT-OBJ-SEED.
get_unique_abbr(): generates collision-free 3-letter abbreviations for a list
  of category or object names, extending into later characters when needed.
generate_exclusion_file(): creates a CSV template (from the design file) for
  manually marking anchor images to exclude before computing similarities.
"""
import os
import re
import pandas as pd

def generate_ids(df, num_initials=3, capitalize=True):
    """Add an 'ID' column to the design DataFrame.

    Each ID is formed as CAT-OBJ-SEED using collision-free abbreviations of
    the category and object names. Operates in-place and also returns df.

    Args:
        df: Design DataFrame with 'category', 'object', and 'seed' columns.
        num_initials: Number of characters to use for each abbreviation.
        capitalize: Whether to uppercase the final ID string.

    Returns:
        The input DataFrame with an 'ID' column added.
    """

    # Generate unique abbreviations for each category
    category_abbr = get_unique_abbr(df['category'].unique().tolist(), num_initials)
    object_abbr = get_unique_abbr(df['object'].unique().tolist(), num_initials)

    # Generate IDs
    def create_id(row):
        id_parts = [
            category_abbr[row['category']],
            object_abbr[row['object']],
            str(row['seed'])
        ]
        id_str = '-'.join(id_parts)
        return id_str.upper() if capitalize else id_str

    df['ID'] = df.apply(create_id, axis=1)

    return df



def get_unique_abbr(category_list, num_initials=3):
    """Build a collision-free abbreviation map for a list of names.

    Takes the first `num_initials` characters of each name. On collision,
    replaces the last character with successive characters from deeper in the
    name until a unique alphabetic abbreviation is found.

    Args:
        category_list: List of strings to abbreviate.
        num_initials: Target length for each abbreviation.

    Returns:
        Dict mapping each original name to its unique abbreviation.
    """
    abbr_dict = {}
    used_abbr = set()

    # print(category_list)
    # category_list = [''.join([char for char in name if char.isalpha()]) for name in category_list]
    # print(category_list)


    for category in category_list:
        abbr = category[:num_initials]

        # Check for duplicates and resolve them
        if abbr in used_abbr:
            for i in range(num_initials, len(category)):
                new_abbr = abbr[:-1] + category[i]

                if new_abbr not in used_abbr and new_abbr.isalpha():
                    abbr = new_abbr
                    break

        abbr_dict[category] = abbr
        used_abbr.add(abbr)

    return abbr_dict


def generate_exclusion_file(stim_set_design_file, target_dir):
    """Create a CSV template for manually flagging images to exclude.

    Strips generation-specific columns (seed, prompt, negative_prompt) from
    the design file and writes excluded_images.csv to target_dir. The 'exclude'
    column should be set to 1 for any image that should be moved out of the
    active set before computing similarities.

    Args:
        stim_set_design_file: Path to the stimulus design CSV.
        target_dir: Directory where excluded_images.csv will be written.
    """

    design = pd.read_csv(stim_set_design_file)
    design = design.drop(['seed', 'prompt', 'negative_prompt'], axis=1)
    design.to_csv(os.path.join(target_dir, 'excluded_images.csv'), index=False)