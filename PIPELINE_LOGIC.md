# Pipeline Logic

This document explains how the pipeline is meant to be run, which folders are
created automatically, which files are written at each stage, and how the
scripts pass information to one another through `.env`.

Important:

- at startup, only `BASE_DIR` needs to be set manually
- `DESIGN_FILE`, `STIM_SET_NAME`, `SELECTED_PAIRS_FILE`, and
  `INTERPOL_DIR_NAME` are expected to start blank and be populated by the
  pipeline as stages complete

## Overview

The pipeline has one user-chosen data root:

- `BASE_DIR`

Everything produced for a given run is written under:

- `BASE_DIR/generation_YYYYMMDD_HHMMSS`

The main exception is the design file from stage 01, which is stored in the
package folder:

- `naturalistic_image_synthesis/stimulus_set_designs`

The scripts communicate run-specific filenames through `.env`. In practice,
this means:

- stage 01 writes `DESIGN_FILE`
- stage 02a writes `STIM_SET_NAME`
- stage 04a writes `SELECTED_PAIRS_FILE`
- stage 04c updates `SELECTED_PAIRS_FILE` to the final-selection CSV
- stage 05 writes `INTERPOL_DIR_NAME`

Each later script reads those values back from `.env` when it starts.

## User-Managed Inputs

These are not created by the pipeline.

### `BASE_DIR`

You set this manually in `.env`.

Example:

```env
BASE_DIR=/path/to/your/output/directory
```

This is the root folder where run folders such as
`generation_20240326_180652` are created.

### LoRA Weights

The image-generation scripts expect the LoRA file to already exist in:

- `LoRAs/xl_more_art-full_v1.safetensors`

This dependency is user-managed. The pipeline does not download it.

### Optional `excluded_images.csv`

If you want to manually remove anchor images after inspecting them, you create:

- `BASE_DIR/generation_YYYYMMDD_HHMMSS/excluded_images.csv`

This file is used by `pipe_02b_exclude_images.py`.

Important:

- the pipeline does not generate `excluded_images.csv`
- it is an optional manual input

## Stage-by-Stage Logic

### Stage 01: Generate Design

File:

- `naturalistic_image_synthesis/pipeline/pipe_01_generate_design.py`

Reads:

- no input files

Creates if missing:

- `naturalistic_image_synthesis/stimulus_set_designs`

Writes:

- `naturalistic_image_synthesis/stimulus_set_designs/stim_design_TIMESTAMP.parquet`
- `naturalistic_image_synthesis/stimulus_set_designs/stim_design_TIMESTAMP.csv`

Updates `.env`:

- `DESIGN_FILE=stim_design_TIMESTAMP.parquet`

Motivation::

- stage 02a does not search for the newest design automatically
- it reads the exact filename stored in `DESIGN_FILE`

### Stage 02a: Generate Anchor Images

File:

- `naturalistic_image_synthesis/pipeline/pipe_02a_generate_images.py`

Reads:

- the design file referenced by `DESIGN_FILE`
- LoRA weights from `LoRAs/`

Creates:

- `BASE_DIR/generation_TIMESTAMP`
- `BASE_DIR/generation_TIMESTAMP/anchor_images`
- category folders inside `anchor_images`
- object folders inside each category

Writes:

- `BASE_DIR/generation_TIMESTAMP/stimulus_set_design.parquet`
- `BASE_DIR/generation_TIMESTAMP/stimulus_set_design.csv`
- `BASE_DIR/generation_TIMESTAMP/variables_log.txt`
- `BASE_DIR/generation_TIMESTAMP/anchor_images/{category}/{object}/{ID}.png`

Updates `.env`:

- `STIM_SET_NAME=generation_TIMESTAMP`

Motivation::

- this is the stage that creates the run folder
- after this point, the rest of the pipeline works inside that run folder

### Stage 02b: Exclude Images

File:

- `naturalistic_image_synthesis/pipeline/pipe_02b_exclude_images.py`

Reads:

- `BASE_DIR/generation_TIMESTAMP/excluded_images.csv`
- anchor images from `anchor_images/{category}/{object}/`

Creates:

- `BASE_DIR/generation_TIMESTAMP/excluded_images`

Moves files:

- from `anchor_images/{category}/{object}/{ID}.png`
- to `excluded_images/{ID}.png`

Notes:

- this is optional
- it depends on a user-provided `excluded_images.csv`

### Stage 03: Compute Similarities for Anchors

File:

- `naturalistic_image_synthesis/pipeline/pipe_03_compute_similarities.py`

Reads:

- all anchor images in `anchor_images/{category}/{object}/`

Creates:

- `BASE_DIR/generation_TIMESTAMP/similarity_scores_anchors`

Writes, for each object:

- `lpips-squeeze-distances-{category}-{object}.txt`
- `lpips-squeeze-mat-{category}-{object}.csv`
- `lpips-squeeze-mat-{category}-{object}.parquet`

### Stage 04a: Select Pairs

File:

- `naturalistic_image_synthesis/pipeline/pipe_04a_select_pairs.py`

Reads:

- `anchor_images`
- `similarity_scores_anchors`
- `stimulus_set_design.parquet`

Writes into the run folder:

- `typical_and_atypical_images.csv`
- `all_image_scores.csv`
- `selected_pairs_for_interpolation_TIMESTAMP.csv`

Updates `.env`:

- `SELECTED_PAIRS_FILE=selected_pairs_for_interpolation_TIMESTAMP.csv`

Motivation::

- later pair-inspection and interpolation stages read the exact CSV referenced
  by `SELECTED_PAIRS_FILE`

### Stage 04b: Collect Selected Pairs

File:

- `naturalistic_image_synthesis/pipeline/pipe_04b_collect_selected_pairs.py`

Reads:

- the selected-pairs CSV from `SELECTED_PAIRS_FILE`
- anchor images

Creates:

- `selected_pair_images_median_TIMESTAMP`
- `selected_pair_images_mean_TIMESTAMP`

Writes:

- preview PNGs showing selected endpoint pairs side by side

This is an inspection stage only.

### Stage 04c: Change Pair

File:

- `naturalistic_image_synthesis/pipeline/pipe_04c_change_pair.py`

Reads:

- the selected-pairs CSV from `SELECTED_PAIRS_FILE`
- anchor images

Creates:

- `changed_pair_images_median_TIMESTAMP`
- `changed_pair_images_mean_TIMESTAMP`

Writes:

- preview PNGs for changed pairs
- a final CSV:
  - `selected_pairs_for_interpolation_TIMESTAMP_final_selection.csv`

Updates `.env`:

- `SELECTED_PAIRS_FILE=selected_pairs_for_interpolation_TIMESTAMP_final_selection.csv`

Motivation::

- stage 05 uses this final-selection CSV

### Stage 04d: Collect Final Pairs

File:

- `naturalistic_image_synthesis/pipeline/pipe_04d_collect_final_pairs.py`

Reads:

- final selected-pairs CSV
- anchor images

Creates:

- `selected_pair_images_median_TIMESTAMP`
- `selected_pair_images_mean_TIMESTAMP`

Writes:

- preview PNGs for the final chosen pairs

This is another inspection stage only.

### Stage 05: Generate Interpolations

File:

- `naturalistic_image_synthesis/pipeline/pipe_05_generate_interpolations.py`

Reads:

- final selected-pairs CSV from `SELECTED_PAIRS_FILE`
- LoRA weights

Creates:

- `BASE_DIR/generation_TIMESTAMP/interpolations_TIMESTAMP`
- category/object folders inside it

Writes:

- interpolation PNGs:
  - `interpolations_TIMESTAMP/{category}/{object}/{ID}-interpol-XXX.png`

Updates `.env` after successful generation:

- `INTERPOL_DIR_NAME=interpolations_TIMESTAMP`

Motivation::

- stages 06, 07, and 08 all use `INTERPOL_DIR_NAME`

### Stage 06a: Compute Similarities for Interpolations

File:

- `naturalistic_image_synthesis/pipeline/pipe_06_compute_similarities_interpols.py`

Reads:

- `interpolations_TIMESTAMP/{category}/{object}/`

Creates:

- `BASE_DIR/generation_TIMESTAMP/similarity_scores_interpolations_TIMESTAMP`

Actual folder pattern in code:

- `similarity_scores_{INTERPOL_DIR_NAME}`

Writes, for each object:

- `lpips-squeeze-distances-interpols-{category}-{object}.txt`
- `lpips-squeeze-mat-interpols-{category}-{object}.csv`
- `lpips-squeeze-mat-interpols-{category}-{object}.parquet`

### Stage 06b: Compute Similarities for Interpolations Subset

File:


Folder logic:

- same as stage 06a

Difference:

- subset-oriented execution

### Stage 07: Select Interpolations

File:

- `naturalistic_image_synthesis/pipeline/pipe_07_select_interpolations.py`

Reads:

- interpolation images from `INTERPOL_DIR`
- interpolation LPIPS matrices from `similarity_scores_{INTERPOL_DIR_NAME}`
- optional manual swap file:
  - `BASE_DIR/generation_TIMESTAMP/excluded_{INTERPOL_DIR_NAME}.csv`

Creates:

- `BASE_DIR/generation_TIMESTAMP/selected_{INTERPOL_DIR_NAME}`
- category/object folders inside it

Writes, for each object:

- copied selected interpolation PNGs
- `info.csv`
- `interpolation_backups.csv`

Notes:

- this stage is now rerun-friendlier because existing destination folders are
  allowed

### Stage 08a: Collect Final Stimuli

File:

- `naturalistic_image_synthesis/pipeline/pipe_08a_collect_stimuli.py`

Reads:

- selected interpolation images from `selected_{INTERPOL_DIR_NAME}`

Creates:

- `BASE_DIR/generation_TIMESTAMP/stimulus_set_from_interpolations_{timestamp}`

The timestamp part comes from `INTERPOL_DIR_NAME`, not from a new clock time.

Writes:

- resized 512x512 final PNGs into that flat folder
- `stimulus_set_info.csv`

Naming rule:

- each output file is named with the first 7 characters of the original image
  name plus `-NN`

Safety behavior:

- the script now raises an error if a destination filename already exists,
  instead of silently overwriting it

### Stage 08b: Create Strips

File:

- `naturalistic_image_synthesis/pipeline/pipe_08b_create_strips.py`

Reads:

- final stimuli from `stimulus_set_from_interpolations_{timestamp}`

Creates:

- sibling folder:
  - `stimulus_set_from_interpolations_{timestamp}_strips`

Writes:

- one JPG strip per object ID

Behavior:

- it no longer assumes there must be exactly 108 IDs
- it only fails if no PNG files are found

## Folder Summary

### Outside `BASE_DIR`

- `naturalistic_image_synthesis/stimulus_set_designs/`

Contains:

- design parquet/csv files from stage 01

### Inside `BASE_DIR`

Each run gets:

- `BASE_DIR/generation_TIMESTAMP/`

Inside a run folder you can get:

- `anchor_images/`
- `excluded_images/`
- `similarity_scores_anchors/`
- `selected_pair_images_median_TIMESTAMP/`
- `selected_pair_images_mean_TIMESTAMP/`
- `changed_pair_images_median_TIMESTAMP/`
- `changed_pair_images_mean_TIMESTAMP/`
- `interpolations_TIMESTAMP/`
- `similarity_scores_{INTERPOL_DIR_NAME}/`
- `selected_{INTERPOL_DIR_NAME}/`
- `stimulus_set_from_interpolations_TIMESTAMP/`
- `stimulus_set_from_interpolations_TIMESTAMP_strips/`

And run-level CSV/parquet files such as:

- `stimulus_set_design.parquet`
- `stimulus_set_design.csv`
- `typical_and_atypical_images.csv`
- `all_image_scores.csv`
- `selected_pairs_for_interpolation_....csv`
- `stimulus_set_info.csv`

## Important Operational Notes

- `.env` values are used as handoff state between scripts
- the intended usage is one script per process, in order
- if you manually edit `.env`, later scripts will follow those values
- `excluded_images.csv` is manual input, not pipeline output
- `BASE_DIR` and the LoRA weights are the main required user-managed inputs

## Recommended Running Order

1. Set `BASE_DIR` in `.env`
2. Run `pipe_01_generate_design.py`
3. Run `pipe_02a_generate_images.py`
4. Optionally create `excluded_images.csv` and run `pipe_02b_exclude_images.py`
5. Run `pipe_03_compute_similarities.py`
6. Run `pipe_04a_select_pairs.py`
7. Optionally inspect with `pipe_04b_collect_selected_pairs.py`
8. Optionally adjust with `pipe_04c_change_pair.py`
9. Optionally inspect finals with `pipe_04d_collect_final_pairs.py`
10. Run `pipe_05_generate_interpolations.py`
11. Run `pipe_06_compute_similarities_interpols.py`
12. Run `pipe_07_select_interpolations.py`
13. Run `pipe_08a_collect_stimuli.py`
14. Run `pipe_08b_create_strips.py`
