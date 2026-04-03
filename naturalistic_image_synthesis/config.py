"""
Central path configuration for the pipeline.

Loads all run-specific identifiers from the .env file and derives every
filesystem path used across pipeline stages. Every other script that needs
filesystem paths should import from here rather than hardcoding paths directly.

Required .env keys:
    BASE_DIR            Root directory for all generated data.

Optional at startup, then written by the pipeline:
    STIM_SET_NAME       Timestamped subdirectory for one generation run,
                        written by pipe_02a after anchor generation succeeds.

Set after each stage produces its outputs:
    DESIGN_FILE         Filename of the design parquet produced by pipe_01.
    SELECTED_PAIRS_FILE Filename of the selected-pairs CSV produced by
                        pipe_04a and later updated by pipe_04c to point to the
                        final-selection CSV.
    INTERPOL_DIR_NAME   Full name of the interpolation directory produced by
                        pipe_05
                        (e.g. interpolations_YYYYMMDD_HHMMSS).
"""
import os
from dotenv import load_dotenv, find_dotenv, set_key

load_dotenv(find_dotenv())

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)
DESIGN_DIR = os.path.join(PACKAGE_DIR, "stimulus_set_designs")
LORA_DIR = os.path.join(PROJECT_ROOT, "LoRAs")

BASE_DIR = os.getenv("BASE_DIR")
STIM_SET_NAME = os.getenv("STIM_SET_NAME")
DESIGN_FILE = os.getenv("DESIGN_FILE")
SELECTED_PAIRS_FILE = os.getenv("SELECTED_PAIRS_FILE")
INTERPOL_DIR_NAME = os.getenv("INTERPOL_DIR_NAME")


def update_env_value(key, value):
    """Persist a pipeline handoff value into the project's .env file."""
    env_file = find_dotenv(usecwd=True)
    if not env_file:
        raise FileNotFoundError("Could not find a .env file to update.")
    set_key(env_file, key, value)
    os.environ[key] = value

# --- Paths available at all stages after STIM_SET_NAME is available ---
STIM_SET_DIR = os.path.join(BASE_DIR, STIM_SET_NAME) if STIM_SET_NAME else None
ANCHOR_IMAGES_DIR = os.path.join(STIM_SET_DIR, "anchor_images") if STIM_SET_DIR else None
SIMILARITY_SCORES_ANCHORS_DIR = (
    os.path.join(STIM_SET_DIR, "similarity_scores_anchors") if STIM_SET_DIR else None
)

# Design file — set DESIGN_FILE in .env after running pipe_01
DESIGN_FILE_PATH = os.path.join(DESIGN_DIR, DESIGN_FILE) if DESIGN_FILE else None

# Selected pairs CSV — set SELECTED_PAIRS_FILE in .env after running pipe_04c
SELECTED_PAIRS_PATH = os.path.join(STIM_SET_DIR, SELECTED_PAIRS_FILE) if SELECTED_PAIRS_FILE else None

# Interpolation directories — set INTERPOL_DIR_NAME in .env after running pipe_05
if INTERPOL_DIR_NAME and STIM_SET_DIR:
    INTERPOL_DIR = os.path.join(STIM_SET_DIR, INTERPOL_DIR_NAME)
    SELECTED_INTERPOL_DIR = os.path.join(STIM_SET_DIR, f"selected_{INTERPOL_DIR_NAME}")
    SIMILARITY_SCORES_INTERPOL_DIR = os.path.join(STIM_SET_DIR, f"similarity_scores_{INTERPOL_DIR_NAME}")
    _interpols_id = INTERPOL_DIR_NAME.replace("interpolations_", "")
    STIM_SET_FROM_INTERPOL_DIR = os.path.join(STIM_SET_DIR, f"stimulus_set_from_interpolations_{_interpols_id}")
else:
    INTERPOL_DIR = None
    SELECTED_INTERPOL_DIR = None
    SIMILARITY_SCORES_INTERPOL_DIR = None
    STIM_SET_FROM_INTERPOL_DIR = None
