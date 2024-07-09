import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

# Dimensions of the Image
WIDTH = 512
HEIGHT = 512

# Dimensions of the Image in Latent Space
# of the Variational Autoencoder
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(
    prompt: str,
    uncond_prompt: str,
    input_image=None,
    strength: float = 0.8,
    do_cfg: bool = True,
    cfg_scale: float = 7.5,
    sampler_name: str = "ddpm",
    n_inference_steps: int = 50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
) -> None:
    """
    Generate Image from Text and/or Image

    prompt: text to generate image from
    uncond_prompt: unconditioned/negative prompt, concept from which generate
        image should move away
    input_image: t generate image from image
    strenght: how much input image will be considered for image-image generation,
        how much the output will resemble the input image.
    do_cfg: activite classifier-free guidance
    cfg_scale: how much pay attention to prompt instead of unconditioned prompt
    """

    with torch.inference_mode():
        if not (0 < strength <= 1):
            raise ValueError("strenght must be a float value between 0 and 1")

        if idle_device:
            to_idle: lambda x: x.to(idle_device)
        else:
            to_idle: lambda x: x.to(device)

        # Create random number generator
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        # Load CLIP model and send to device
        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            # Convert the prompt into tokens using the tokenizer
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_lenght", max_lenght=77
            ).input_ids
            # Convert input ids to tensor (batch_size, seq_len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # Forward pass prompt through CLIP model
            # (batch_size, seq_len) -> (batch_size, seq_len, dim)
            cond_context = clip(cond_tokens)

            # Convert the unconditioned prompt into tokens using the tokenizer
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_lenght", max_lenght=77
            ).input_ids
            # Convert input ids to tensor (batch_size, seq_len)
            uncond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # Forward pass unconditioned prompt through CLIP model
            # (batch_size, seq_len) -> (batch_size, seq_len, dim)
            uncond_context = clip(uncond_tokens)

            # Concatenate contexts. They will become input of the unit.
            # (batch_size, seq_len, dimension) = (2, 77, 768)
            context = torch.cat([cond_context, uncond_context])
        else:
            # Convert prompts into list of tokens
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_lenght", max_lenght=77
            ).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (1, 77, 768)
            context = clip(tokens)

        # Move CLIP model back to the CPU
        to_idle(clip)

        # Create sampler and set the number of inference steps
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            # The more denoisification steps, the better the quality
            # of the generated image. For inference 50 is ok,
            # even though for training it may need about 1000.
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler name {sampler_name}")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        # Consider input image for image-image generation
        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image = input_image.resize(HEIGHT, WIDTH)
            input_image = np.array(input_image)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            # Values for each pixel should be between -1 and 1 instead of between 0 and 255
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # Add batch_size dimension -> (batch_size, height, width, channels)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (batch_size, height, width, channels) -> (batch_size, channels, height, width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # Run the image through the encoder of the variational autoencoder
            latents = encoder(input_image_tensor)

            # Sample some noise for the encoder
            encoder_noise = torch.randn(
                latents_shape, generator=generator, device=device
            )

            # Add noise to the image and set time schedule for denoising
            sampler.set_strength(stregth=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # For text-to-image, start with random noise N(0, I)
            latents = torch.rand(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        # Predict amount of noise in the image using the UNET
        # The DDPM Sampler removes the noise from the image
        for i, timestep in tqdm(sampler.timesteps):
            # Convert timestep to tensor (1, 320)
            time_embedding = get_time_embeddings(timestep).to(device)

            # (batch_size, 4, height/8, width/8)
            model_input = latents

            if do_cfg:
                # Make two copies of the latents: one for using with
                # prompt and one without prompt
                # (batch_size, 4, height/8, width/8) -> (2*batch_size, 4, height/8, width/8)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model output is the predicted noise (in the image) by the UNET
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                # Combine conditioned and unconditioned prompt to generate output
                model_output = cfg_scale * output_cond + (1 - cfg_scale) * output_uncond

            # Remove noise predicted by the UNET
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        # Run denoised latents through the decoder to generate image
        decoder = models["decoder"]
        decoder.to(device)

        images = decoder(latents)
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]
    
def rescale(x, old_range, new_range, clamp=False):
    """
    Rescale values of pixels of image
    """
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    
    return x

def get_time_embeddings(timestep) -> torch.Tensor:
    """
    Convert timestep into a tensor of size (1, 320)

    Using sines and cosines formula from
    "All you need is attention"
    """
    # (160)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] ** freqs[None]
    # concatenate sine and cosine tensors -> (1, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1) 
