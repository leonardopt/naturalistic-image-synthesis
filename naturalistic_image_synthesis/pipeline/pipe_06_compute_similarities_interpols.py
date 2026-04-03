"""
Step 6 — Compute pairwise LPIPS distances for all interpolated frames.

Same logic as pipe_03, but runs on interpolations_TIMESTAMP/ instead of
anchor_images/. Outputs go to similarity_scores_{INTERPOL_DIR_NAME}/, which
pipe_07 reads to select 10 representative frames per sequence.
Parallelised across GPUs by category. Must run after pipe_05.
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.multiprocessing as mp
import lpips
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from naturalistic_image_synthesis.config import INTERPOL_DIR, SIMILARITY_SCORES_INTERPOL_DIR


def compute_all_pairs(sourcedir, dstdir, net='squeeze', device="cuda", plot_results=False, recompute=False):
    """Compute all pairwise LPIPS distances for images in sourcedir.

    Skips computation if output files already exist, unless recompute=True.
    Writes a distance matrix as .txt, .csv, and .parquet. Optionally plots a
    hierarchically clustered heatmap.

    Args:
        sourcedir: Directory containing the interpolation frames to compare.
        dstdir: Directory where output files are written.
        net: LPIPS backbone network (default 'squeeze').
        device: CUDA device index (default 'cuda').
        plot_results: If True, render and save a clustered heatmap.
        recompute: If True, overwrite existing output files.

    Returns:
        Reordered distance matrix DataFrame if plot_results is True, else None.
    """
    loss_fn = lpips.LPIPS(net=net)
    loss_fn = loss_fn.cuda(device)

    sub_dirs = os.path.normpath(sourcedir).split(os.path.sep)[-2:]
    # Combine them with a hyphen
    directory_name = '-'.join(sub_dirs)

    # Define the output file name for distances
    distances_filename = os.path.join(dstdir, f"lpips-{net}-distances-interpols-{directory_name}.txt")
    csv_filename = os.path.join(dstdir, f"lpips-{net}-mat-interpols-{directory_name}.csv")
    parquet_filename = os.path.join(dstdir, f"lpips-{net}-mat-interpols-{directory_name}.parquet")

    # Check if file exists already
    if not recompute and os.path.exists(distances_filename) and os.path.exists(csv_filename) and os.path.exists(parquet_filename):
        print(f'Files {distances_filename, csv_filename, parquet_filename} exist. Skipping')
    else:
        # Filter out only .png and .jpg files from the source directory
        files = sorted([file for file in os.listdir(sourcedir) if file.lower().endswith(('.png', '.jpg')) and 'lpips' not in file])

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
    """Split category directories across GPUs and spawn one worker per GPU.

    Args:
        world_size: Number of available GPUs.
        rootdir: Root interpolations directory; subdirectories are categories.
        scores_dstdir: Directory where LPIPS output files are written.
        net: LPIPS backbone network identifier.
    """
    directories = [name for name in os.listdir(rootdir) if os.path.isdir(os.path.join(rootdir, name))]
    directories_splits = np.array_split(directories, world_size)
    mp.spawn(compute_similarities, args=(directories_splits, world_size, rootdir, scores_dstdir, net), nprocs=world_size, join=True)


def compute_similarities(rank, directories_splits, world_size, rootdir, scores_dstdir, net):
    """Worker function executed on a single GPU by mp.spawn.

    Args:
        rank: GPU index assigned by mp.spawn.
        directories_splits: List of category-name arrays, one slice per GPU.
        world_size: Total number of GPU workers (unused; kept for spawn symmetry).
        rootdir: Root interpolations directory.
        scores_dstdir: Directory where LPIPS output files are written.
        net: LPIPS backbone network identifier.
    """
    torch.cuda.set_device(rank)
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
        interpolations_dir_path = INTERPOL_DIR
        scores_dstdir = SIMILARITY_SCORES_INTERPOL_DIR
        os.makedirs(scores_dstdir, exist_ok=True)
        print(f'Source folder: {interpolations_dir_path}')
        print(f'Save scores in: {scores_dstdir}')
        # Run in parallel
        net = 'squeeze'
        run_distributed(world_size=world_size,
                        rootdir=interpolations_dir_path,
                        scores_dstdir=scores_dstdir,
                        net=net)

        computing_time = datetime.now() - now
        print(f'Total computing time: {computing_time}')

    except Exception as error:
        print(error)
