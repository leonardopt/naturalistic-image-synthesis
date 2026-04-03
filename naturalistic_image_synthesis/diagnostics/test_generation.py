"""
Diagnostic: test SDXL generation reproducibility.

Loads the full SDXL base + refiner + LoRA stack and generates two images from
the same seed and prompt. Both outputs should be pixel-identical; any difference
indicates a reproducibility issue in the pipeline setup (e.g. non-deterministic
ops, wrong generator initialisation).
"""
import os
from pathlib import Path
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch
from datetime import datetime
from diffusers.utils.torch_utils import randn_tensor

def setup_and_generate_image(seed, prompt, negative_prompt, lora_model_path, output_file):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()  # Attempt to reset CUDA state

    # Load models
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")

    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")

    pipe.load_lora_weights(lora_model_path)
    # Optimizations
    pipe.enable_xformers_memory_efficient_attention()
    refiner.enable_xformers_memory_efficient_attention()

    generator = torch.Generator(device="cuda:0").manual_seed(seed)
    noise_shape = (1, 4, 1024 // pipe.vae_scale_factor, 1024 // pipe.vae_scale_factor)
    init_latent = randn_tensor(noise_shape, generator=generator, device=torch.device("cuda:0"), dtype=torch.float16)

    # Generate image
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, latents=init_latent, num_inference_steps=50, output_type="latent").images[0]
    refined = refiner(prompt=prompt, negative_prompt=negative_prompt, image=image[None, :], generator=generator).images[0]

    # Save image
    refined.save(output_file)

if __name__ == "__main__":
    # Configuration
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    lora_model_id = 'xl_more_art-full_v1.safetensors'
    lora_model_path = os.path.join(project_root, 'LoRAs', lora_model_id)
    print(f'Load LoRA from: {lora_model_path}')

    prompt = "photo of a tractor parked in the middle of an agricultural field, hills in the background, bright light, high resolution photography, cinematic"
    negative_prompt = "people, person, human figures, human body parts, humans"
    seed = 363

    now = datetime.now()
    date_time_str = now.strftime('%Y%m%d_%H%M%S')
    output_file1 = f'test_images/test_object_{date_time_str}_1.png'
    output_file2 = f'test_images/test_object_{date_time_str}_2.png'

    # Generate images
    setup_and_generate_image(seed, prompt, negative_prompt, lora_model_path, output_file1)
    setup_and_generate_image(seed, prompt, negative_prompt, lora_model_path, output_file2)
