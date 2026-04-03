"""
Diagnostic: test interpolation generation for a single object.

Loads the SDXL base + refiner + LoRA stack and runs the full SLERP
interpolation routine for one object (giraffe by default) using a small number
of steps. Used to verify that the interpolation pipeline produces smooth,
reproducible sequences before committing to a full multi-GPU run.
"""
import os
from pathlib import Path

import pandas as pd
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch
from datetime import datetime
from diffusers.utils.torch_utils import randn_tensor
import numpy as np
from naturalistic_image_synthesis.config import STIM_SET_DIR, SELECTED_PAIRS_PATH
# generation_20240326_180652 / selected_pairs_for_interpolation_20240402_222029_final_selection.csv


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """modified version of function in https://github.com/nateraw/stable-diffusion-videos/blob/main/stable_diffusion_videos/utils.py"""

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
                                 negative_prompt, lora_model_path, reverse_numbering, output_dir):
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

        refined.save(os.path.join(output_dir, f'interpol-{interpol_number:03}.png'))


if __name__ == "__main__":
    # Configuration
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    lora_model_id = 'xl_more_art-full_v1.safetensors'
    lora_model_path = os.path.join(project_root, 'LoRAs', lora_model_id)
    print(f'Load LoRA from: {lora_model_path}')

    # ## Subselect dataframe if you want
    df = pd.read_csv(SELECTED_PAIRS_PATH)
    obj_name = 'jet_plane'

    df = df[df['object']== obj_name]
    prompt = str(df['prompt'])
    print(prompt)
    negative_prompt = str(df['negative_prompt'])

    now = datetime.now()
    date_time_str = now.strftime('%Y%m%d_%H%M%S')
    output_dir = f'test_images/interpolations_{date_time_str}'
    os.makedirs(output_dir, exist_ok=True)

    # Generate images with different seeds
    rank = "cuda:0"
    seed1 = int(df['seed_0_mean'].iloc[0])
    seed2 = int(df['seed_mean_backup_3'].iloc[0])
    print(seed1, seed2)
    num_interpolation_steps = 10
    if num_interpolation_steps % 2 == 0:
        batch_size = num_interpolation_steps // 2
    else:
        batch_size = (num_interpolation_steps // 2) + 1

    try:
        generate_interpolation_batch(rank, seed1, seed2, num_interpolation_steps, batch_size, prompt, negative_prompt,
                                         lora_model_path, reverse_numbering=False, output_dir=output_dir)
        generate_interpolation_batch(rank, seed2, seed1, num_interpolation_steps, batch_size, prompt, negative_prompt,
                                     lora_model_path, reverse_numbering=True, output_dir=output_dir)
    except Exception as e:
        print(e)