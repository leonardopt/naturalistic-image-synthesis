"""
Step 5 — Generate latent-space interpolation sequences for all objects.

For each endpoint pair in the final-selection CSV, generates a 200-step SLERP
interpolation between the two seed latents using SDXL base + refiner + LoRA.
The sequence is split into two batches of 100 (forward then reverse) to avoid
reloading the model. Outputs are saved as PNGs under
interpolations_TIMESTAMP/category/object/. Parallelised across GPUs. Writes
INTERPOL_DIR_NAME to .env after generation completes.

Reproducibility note: generators must be re-initialised on every call.
The base model does not need a generator (the seed is passed via init_latent),
but the refiner does. When interpolating, generator_1 is used for all frames
in both forward and reverse batches.
"""

import pandas as pd
import os
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch
import numpy as np
from diffusers.utils.torch_utils import randn_tensor
from pathlib import Path
from datetime import datetime
import torch.multiprocessing as mp
from naturalistic_image_synthesis.config import LORA_DIR, SELECTED_PAIRS_PATH, STIM_SET_DIR, update_env_value


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """Spherical linear interpolation between two latent vectors.

    Falls back to linear interpolation when the vectors are nearly parallel
    (dot product > DOT_THRESHOLD). Accepts both numpy arrays and torch tensors;
    always returns the same type as the input.

    Args:
        t: Interpolation factor in [0, 1].
        v0: Start vector (numpy array or torch.Tensor).
        v1: End vector (numpy array or torch.Tensor), same shape as v0.
        DOT_THRESHOLD: Cosine-similarity threshold above which linear
            interpolation is used instead of true SLERP.

    Returns:
        Interpolated vector with the same type and device as the inputs.

    Reference:
        https://github.com/nateraw/stable-diffusion-videos
    """

    inputs_are_torch = isinstance(v0, torch.Tensor)
    if inputs_are_torch:
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        if isinstance(v2, np.ndarray):
            v2 = torch.tensor(v2, device=input_device, dtype=torch.float16)
        else:
            # If v2 is unexpectedly a tensor, clone it to maintain the original tensor's properties without alteration
            v2 = v2.clone().detach().to(input_device).to(torch.float16)

    return v2


def generate_interpolation_batch(rank, start_seed, end_seed, num_interpolation_steps, batch_size, prompt,
                                 negative_prompt, lora_model_path, reverse_numbering, output_dir, obj_ID):
    """Generate one batch of SLERP-interpolated images and save them to output_dir.

    Loads SDXL base + refiner + LoRA fresh on each call (needed for multi-process
    safety). Samples start/end latents from start_seed/end_seed, builds
    num_interpolation_steps SLERP latents, generates the first batch_size of them.
    Frame filenames are zero-padded to three digits; reverse_numbering maps
    local index i to (num_interpolation_steps - 1 - i) so forward and reverse
    batches tile without overlap.

    Args:
        rank: GPU index.
        start_seed: Integer seed for the start latent.
        end_seed: Integer seed for the end latent.
        num_interpolation_steps: Total steps in the full SLERP arc (used for
            reverse-index computation and linspace target).
        batch_size: Number of frames this call actually generates.
        prompt: Positive text prompt.
        negative_prompt: Negative text prompt.
        lora_model_path: Path to the .safetensors LoRA weights.
        reverse_numbering: If True, number frames from the far end inward.
        output_dir: Directory where PNG files are saved.
        obj_ID: Filename prefix (CAT-OBJ).
    """
    # Load models
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to(rank)

    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to(rank)

    pipe.load_lora_weights(lora_model_path)
    # Optimizations
    pipe.enable_xformers_memory_efficient_attention()
    refiner.enable_xformers_memory_efficient_attention()

    # Define latents
    noise_shape = (1, 4, 1024 // pipe.vae_scale_factor, 1024 // pipe.vae_scale_factor)
    # Start latent
    generator_1 = torch.Generator(device=rank).manual_seed(start_seed)
    init_latent_1 = randn_tensor(noise_shape, generator=generator_1, device=torch.device(rank), dtype=torch.float16)
    # End latent
    generator_2 = torch.Generator(device=rank).manual_seed(end_seed)
    init_latent_2 = randn_tensor(noise_shape, generator=generator_2, device=torch.device(rank), dtype=torch.float16)

    interpolation_values = torch.linspace(0, 1, steps=num_interpolation_steps)
    interpolated_latents = [slerp(t, init_latent_1, init_latent_2) for t in interpolation_values][0:batch_size]

    # Generate image
    for i, curr_latent in enumerate(interpolated_latents):
        image = pipe(prompt=prompt, negative_prompt=negative_prompt, latents=curr_latent, num_inference_steps=50,
                     output_type="latent").images[0]
        refined = refiner(prompt=prompt, negative_prompt=negative_prompt, image=image[None, :], generator=generator_1).images[0]

        # Save image
        if reverse_numbering:
            interpol_number = num_interpolation_steps - 1 - i
        else:
            interpol_number = i

        # Save image
        refined.save(os.path.join(output_dir, f'{obj_ID}-interpol-{interpol_number:03}.png'))


def run_distributed(df, world_size, dstdir, metric, num_interpolation_steps):
    """Split the selected-pairs DataFrame and spawn one generate_interpolations worker per GPU.

    Args:
        df: Selected-pairs DataFrame (output of pipe_04a/c).
        world_size: Number of GPUs to use.
        dstdir: Root output directory for interpolation frames.
        metric: 'mean' or 'median' — selects which seed columns to use.
        num_interpolation_steps: Total SLERP steps per sequence (200 by default).
    """
    # Split DataFrame into chunks
    df_splits = np.array_split(df, world_size)

    # Spawn processes
    mp.spawn(generate_interpolations, args=(df_splits, world_size, dstdir, metric, num_interpolation_steps), nprocs=world_size, join=True)


def generate_interpolations(rank, df_splits, world_size, dstdir, metric, num_interpolation_steps):
    """Worker: generate all interpolation sequences assigned to this GPU.

    Calls generate_interpolation_batch twice per object — once forward
    (seed_0 → seed_1) and once in reverse (seed_1 → seed_0 with reverse
    numbering) — so the two batches together cover the full 200-step arc.

    Args:
        rank: GPU index.
        df_splits: List of per-GPU DataFrame chunks.
        world_size: Total number of GPU workers (unused directly; present for mp.spawn compat).
        dstdir: Root output directory.
        metric: 'mean' or 'median' — selects which seed columns to use.
        num_interpolation_steps: Total SLERP steps per sequence.
    """
    assert metric == 'mean' or metric == 'median'
    # Set the current device to the specific GPU
    torch.cuda.set_device(rank)

    # Get the chunk of DataFrame for this process
    df_subset = df_splits[rank]

    lora_model_id = 'xl_more_art-full_v1.safetensors'
    lora_model_path = os.path.join(LORA_DIR, lora_model_id)
    print(f'Load LoRA from: {lora_model_path}')

    # Generation parameters
    if num_interpolation_steps % 2 == 0:
        batch_size = num_interpolation_steps // 2
    else:
        batch_size = (num_interpolation_steps // 2) + 1


    for i, row in df_subset.iterrows():
        # Variables to generate current interpolations
        curr_cat = row['category']
        curr_obj = row['object']
        curr_prompt = row['prompt']
        curr_negative_prompt = row['negative_prompt']
        seed_0 = row[f'final_select_{metric}_seed_0']
        seed_1 = row[f'final_select_{metric}_seed_1']
        # Extract ID, removing seed and file extension
        curr_ID = row[f'final_select_{metric}_0']
        parts = curr_ID.split('-')
        id_part = '-'.join(parts[:2])

        # Create subfolder to store images
        obj_directory = os.path.join(dstdir, curr_cat, curr_obj)
        os.makedirs(obj_directory, exist_ok=True)

        print(f'{"-" * 10} Generate interpolations for {curr_cat}-{curr_obj} [BATCH-1, size={batch_size}] {"-" * 10}')
        print(f'seed_0: {seed_0}, seed_1: {seed_1}')
        generate_interpolation_batch(rank, seed_0, seed_1, num_interpolation_steps, batch_size, curr_prompt, curr_negative_prompt,
                                     lora_model_path, reverse_numbering=False, output_dir=obj_directory, obj_ID=id_part)
        print(f'{"-" * 10} Generate interpolations for {curr_cat}-{curr_obj} [BATCH-2, size={batch_size}] {"-" * 10}')
        generate_interpolation_batch(rank, seed_1, seed_0, num_interpolation_steps, batch_size, curr_prompt, curr_negative_prompt,
                                     lora_model_path, reverse_numbering=True, output_dir=obj_directory, obj_ID=id_part)




if __name__ == "__main__":

    try:
        world_size = torch.cuda.device_count()  # Number of available GPUs
        # Get Time
        now = datetime.now()
        date_time_str = now.strftime('%Y%m%d_%H%M%S')

        # Load csv file
        metric = 'mean'
        num_interpolation_steps = 200

        source_dir_path = STIM_SET_DIR

        # Make directory to store all generated interpolations
        interpol_dir_name = f'interpolations_{date_time_str}'
        dstdir = os.path.join(source_dir_path, interpol_dir_name)
        os.makedirs(dstdir, exist_ok=False)

        df = pd.read_csv(SELECTED_PAIRS_PATH)
        print('Design file loaded successfully')

        # Run in parallel
        run_distributed(df, world_size, dstdir, metric, num_interpolation_steps)
        update_env_value('INTERPOL_DIR_NAME', interpol_dir_name)

        computing_time = datetime.now() - now
        print(f'Total computing time: {computing_time}')

    except Exception as error:
        print(error)
