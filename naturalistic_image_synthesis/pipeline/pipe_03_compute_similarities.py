"""
Step 3 — Compute pairwise LPIPS distances for all anchor image sets.

For each category/object subdirectory under anchor_images/, computes the full
symmetric LPIPS distance matrix and saves it as both CSV and Parquet. Results
go to similarity_scores_anchors/ and are read by pipe_04a to select the
interpolation endpoint pair. Parallelised across GPUs by category directory.
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
import lpips
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.multiprocessing as mp
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from tqdm import tqdm

from naturalistic_image_synthesis.config import ANCHOR_IMAGES_DIR, SIMILARITY_SCORES_ANCHORS_DIR


def compute_all_pairs(sourcedir, dstdir, net='squeeze', device="cuda", plot_results=False, recompute=False):
    """Compute the full pairwise LPIPS distance matrix for all images in sourcedir.

    Skips computation if output files already exist (unless recompute=True).
    Results are written as a .txt distances log, a .csv, and a .parquet matrix.
    Optionally plots a hierarchically clustered heatmap.

    Args:
        sourcedir: Directory containing the PNG/JPEG anchor images.
        dstdir: Directory where output files are saved.
        net: LPIPS backbone — 'squeeze', 'vgg', or 'alex'.
        device: CUDA device string or index.
        plot_results: If True, save a clustered LPIPS heatmap PNG alongside the images.
        recompute: If True, overwrite existing output files.

    Returns:
        Reordered distance DataFrame if plot_results=True, otherwise None.
    """
    # Initialize the LPIPS model
    loss_fn = lpips.LPIPS(net=net)
    loss_fn = loss_fn.cuda(device)

    # Extract source directory name for our file names
    directory_name = os.path.basename(sourcedir)
    sub_dirs = os.path.normpath(sourcedir).split(os.path.sep)[-2:]
    # Combine them with a hyphen
    directory_name = '-'.join(sub_dirs)

    # Define the output file name for distances
    distances_filename = os.path.join(dstdir, f"lpips-{net}-distances-{directory_name}.txt")
    csv_filename = os.path.join(dstdir, f"lpips-{net}-mat-{directory_name}.csv")
    parquet_filename = os.path.join(dstdir, f"lpips-{net}-mat-{directory_name}.parquet")


    # Check if file exists already
    if not recompute and os.path.exists(distances_filename) and os.path.exists(csv_filename) and os.path.exists(parquet_filename):
        print(f'Files {distances_filename, csv_filename, parquet_filename} exist. Skipping')
    else:
        # Filter out only .png and .jpg files from the source directory
        files = [file for file in os.listdir(sourcedir) if file.lower().endswith(('.png', '.jpg')) and 'lpips' not in file]

        # Store images in list
        images = []
        for i, file in enumerate(files):
            # load file using original file name
            img_path = os.path.join(sourcedir, file)
            if os.path.exists(img_path):
                img = lpips.im2tensor(lpips.load_image(img_path))  # RGB image from [-1,1]
                img = img.cuda()
                images.append(img)
            else:
                raise Exception(f'Could not load path: {img_path}')
        num_files = len(images)
        print(f'Number of files to analyse: {num_files}')
        if num_files == 0:
            raise Exception(f'No files found in the current directory {sourcedir}')
        dist_matrix = np.zeros((num_files, num_files))

        print(f'Starting computation for {directory_name}')
        # Write distances to the specified file in the output directory
        with open(distances_filename, 'w') as f:
            for i in tqdm(range(num_files)):
                for j in range(i + 1, num_files):  # Avoid redundant comparisons
                    dist = loss_fn.forward(images[i], images[j]).item()
                    dist_matrix[i, j] = dist_matrix[j, i] = dist  # Symmetric matrix
                    f.write('(%s, %s): %.6f\n' % (files[i], files[j], dist))
        # print('Success!')
        # After computing the distance matrix
        dist_matrix_df = pd.DataFrame(dist_matrix, index=files, columns=files)

        # Save the DataFrame to a CSV file
        dist_matrix_df.to_csv(csv_filename)
        dist_matrix_df.to_parquet(parquet_filename)
        print(f'Results written in:\n\t{csv_filename}\n\t{parquet_filename}')

        # Calculate average and standard error
        avg_dist = np.mean(dist_matrix[np.triu_indices(num_files, 1)])  # Exclude diagonal
        stderr_dist = np.std(dist_matrix[np.triu_indices(num_files, 1)]) / np.sqrt(num_files * (num_files - 1) / 2)

        print('Avg: %.5f +/- %.5f' % (avg_dist, stderr_dist))
        with open(distances_filename, 'a') as f:  # Append to the existing file
            f.write('Avg: %.6f +/- %.6f\n' % (avg_dist, stderr_dist))

        # Plot if selected
        if plot_results:
            # Convert the DataFrame to a condensed distance matrix
            condensed_dist_matrix = squareform(dist_matrix_df, checks=False)
            row_linkage = linkage(condensed_dist_matrix, method='average')
            col_linkage = linkage(condensed_dist_matrix, method='average')

            # Determine the order of rows and columns
            row_order = leaves_list(row_linkage)
            col_order = leaves_list(col_linkage)

            # Reorder the DataFrame according to the clustering
            sorted_matrix_df = dist_matrix_df.iloc[row_order, col_order]

            # Plot matrix
            fig = plt.figure(figsize=(17, 14))
            sns.heatmap(data=sorted_matrix_df,
                        cmap="magma",
                        annot=True,
                        )
            plot_name = os.path.basename(sourcedir).replace('/', '-')
            plt.title(f'LPIPS scores for object: {plot_name}')
            plt.tight_layout()
            plt.show()

            fig.savefig(os.path.join(sourcedir, f'lpips-mat-{plot_name}.png'))

            return sorted_matrix_df


def run_distributed(world_size, rootdir, scores_dstdir, net):
    """Split category directories across GPUs and spawn compute_similarities workers.

    Args:
        world_size: Number of GPUs to use.
        rootdir: Root anchor_images/ directory containing category subdirs.
        scores_dstdir: Output directory for LPIPS parquet/CSV files.
        net: LPIPS backbone name passed through to compute_all_pairs.
    """
    directories = [name for name in os.listdir(rootdir) if os.path.isdir(os.path.join(rootdir, name))]

    # Split DataFrame into chunks
    directories_splits = np.array_split(directories, world_size)

    # Spawn processes
    mp.spawn(compute_similarities, args=(directories_splits, world_size, rootdir, scores_dstdir, net), nprocs=world_size, join=True)


def compute_similarities(rank, directories_splits, world_size, rootdir, scores_dstdir, net):
    """Worker function: compute LPIPS matrices for the category subset assigned to this GPU.

    Args:
        rank: GPU index (set as the active CUDA device).
        directories_splits: List of per-GPU directory name arrays.
        world_size: Total number of GPU workers (unused directly, present for mp.spawn compat).
        rootdir: Root anchor_images/ directory.
        scores_dstdir: Output directory for results.
        net: LPIPS backbone name.
    """
    # Set the current device to the specific GPU
    torch.cuda.set_device(rank)

    # Get the chunk of DataFrame for this process
    dir_subset = directories_splits[rank]


    for cat_dir_name in dir_subset:
        for obj_dir_name in os.listdir(os.path.join(rootdir, cat_dir_name)):
            object_dir_path = os.path.join(rootdir, cat_dir_name, obj_dir_name)
            print(f'Processing directory: {object_dir_path}')
            compute_all_pairs(sourcedir=object_dir_path, dstdir=scores_dstdir, net=net, device=rank)


if __name__ == "__main__":

    try:
        world_size = torch.cuda.device_count()  # Number of available GPUs
        # Get Time
        now = datetime.now()
        date_time_str = now.strftime('%Y%m%d_%H%M%S')
        anchors_dir = ANCHOR_IMAGES_DIR
        scores_dstdir = SIMILARITY_SCORES_ANCHORS_DIR
        os.makedirs(scores_dstdir, exist_ok=True)
        print(f'Source folder: {anchors_dir}')
        print(f'Save scores in: {scores_dstdir}')
        # Run in parallel
        net = 'squeeze'

        run_distributed(world_size=world_size,
                        rootdir=anchors_dir,
                        scores_dstdir=scores_dstdir,
                        net=net)

        computing_time = datetime.now() - now
        print(f'Total computing time: {computing_time}')

    except Exception as error:
        print(error)

