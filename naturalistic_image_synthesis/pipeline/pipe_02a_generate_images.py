"""
Step 2a — Generate anchor images with SDXL base + refiner.

Reads the design DataFrame from DESIGN_FILE_PATH, generates one image per row
(one object-scene per seed) using SDXL base (latent output) followed by the
SDXL refiner, with a LoRA loaded on top. The initial noise latent is sampled
from a fixed seed for reproducibility. Parallelised across GPUs. Creates the
generation run folder and writes STIM_SET_NAME to .env.
"""
import inspect
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.utils.torch_utils import randn_tensor
from naturalistic_image_synthesis.config import BASE_DIR, DESIGN_FILE_PATH, LORA_DIR, update_env_value

def save_variables_to_file(file_name):
    """
    Save all local variables and their values to a specified text file.

    Args:
    file_name (str): The name of the file where the variables will be saved.
    """
    # Retrieve local variables from the caller's frame
    variables = inspect.currentframe().f_back.f_locals

    with open(file_name, 'w') as file:
        for var_name, value in variables.items():
            # Write the variable name and its value to the file
            file.write(f"{var_name}: {repr(value)}\n")


def run_distributed(df, world_size, dstdir):
    """Split the design DataFrame and spawn one run_on_gpu worker per GPU.

    Args:
        df: Full design DataFrame (one row per image to generate).
        world_size: Number of GPUs to use.
        dstdir: Root output directory; category/object subdirs are created inside.
    """
    # Split DataFrame into chunks
    df_splits = np.array_split(df, world_size)

    # Spawn processes
    mp.spawn(run_on_gpu, args=(df_splits, world_size, dstdir), nprocs=world_size, join=True)


def run_on_gpu(rank, df_splits, world_size, dstdir):
    """Load SDXL base + refiner + LoRA on GPU rank and generate images for the assigned rows.

    Each row in df_splits[rank] produces one PNG saved to
    dstdir/category/object/ID.png. The initial noise latent is sampled from a
    fixed seed via randn_tensor for reproducibility; the same generator is
    passed to the refiner.

    Args:
        rank: GPU index (used as the CUDA device).
        df_splits: List of per-GPU DataFrame chunks.
        world_size: Total number of GPU workers (unused directly; present for mp.spawn compat).
        dstdir: Root output directory.
    """
    # Set the current device to the specific GPU
    torch.cuda.set_device(rank)

    # Get the chunk of DataFrame for this process
    df_subset = df_splits[rank]

    lora_model_id = 'xl_more_art-full_v1.safetensors'
    lora_model_path = os.path.join(LORA_DIR, lora_model_id)
    print(f'Load LoRA from: {lora_model_path}')

    print('Load model...')
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )

    pipe.to(rank)
    pipe.load_lora_weights(lora_model_path)
    refiner.to(rank)
    # Optimizations
    pipe.enable_xformers_memory_efficient_attention()
    # refiner.enable_xformers_memory_efficient_attention()

    num_inference_steps = 50

    for n, row in df_subset.iterrows():
        curr_seed = row['seed']
        curr_category = row['category']
        curr_object = row['object']
        curr_prompt = row['prompt']
        curr_negative_prompt = row['negative_prompt']
        curr_ID = row['ID']

        # Create subfolder for each category
        cat_dir = os.path.join(dstdir, curr_category)
        os.makedirs(cat_dir, exist_ok=True)
        obj_dir = os.path.join(cat_dir, curr_object)
        os.makedirs(obj_dir, exist_ok=True)

        print(f'{"-" * 10} generate prompt:\n"{curr_prompt}"\n{"-" * 10}')

        generator = torch.Generator(device=rank).manual_seed(curr_seed)
        noise_shape = (1,
                       4,
                       1024 // pipe.vae_scale_factor,
                       1024 // pipe.vae_scale_factor)
        init_latent = randn_tensor(noise_shape,
                                   generator=generator,
                                   device=torch.device(rank),
                                   dtype=torch.float16)

        image = pipe(prompt=curr_prompt,
                     negative_prompt=curr_negative_prompt,
                     # generator=generator,
                     latents=init_latent,
                     num_inference_steps=num_inference_steps,
                     output_type="latent",
                     ).images[0]

        refined = refiner(
            prompt=curr_prompt,
            negative_prompt=curr_negative_prompt,
            image=image[None, :],
            generator=generator,
        ).images[0]

        # Save refined image
        refined.save(os.path.join(obj_dir, f'{curr_ID}.png'))


if __name__ == "__main__":

    try:
        world_size = torch.cuda.device_count()  # Number of available GPUs
        # Get Time
        now = datetime.now()
        date_time_str = now.strftime('%Y%m%d_%H%M%S')
        # Set up dst directory
        stim_set_name = f'generation_{date_time_str}'
        stim_set_dir = os.path.join(BASE_DIR, stim_set_name)
        dstdir = os.path.join(stim_set_dir, 'anchor_images')
        os.makedirs(stim_set_dir, exist_ok=False)
        os.makedirs(dstdir, exist_ok=False)
        # Import DataFrame
        design_file_path = DESIGN_FILE_PATH
        print(f'Read design file from {design_file_path}')
        df = pd.read_parquet(design_file_path)
        df.to_parquet(os.path.join(stim_set_dir, 'stimulus_set_design.parquet'))
        df.to_csv(os.path.join(stim_set_dir, 'stimulus_set_design.csv'))
        print('Design file loaded successfully')

        # Run in parallel
        run_distributed(df, world_size, dstdir)

        # Save variables log
        save_variables_to_file(os.path.join(stim_set_dir, 'variables_log.txt'))
        update_env_value('STIM_SET_NAME', stim_set_name)

        computing_time = datetime.now() - now
        print(f'Total computing time: {computing_time}')

    except Exception as error:
        print(error)
