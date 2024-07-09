import torch
import numpy as np

class DDPMSampler:

    def __init__(self, generator: torch.Generator, num_training_steps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.0120) -> None:
        """
        Args:
        noise added to image varies according to beta variables, 1000 numbers
        from beta_start until beta_end 
        """
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32)**2
        self.alphas = 1.0 - self.betas
        # cumulative products
        # [alpha_0, alpha_0*alpha_1, alpha_0*alpha_1*alpha_2, ...]
        self.alpha_cumprod = torch.cumprod(self.alphas, 0)
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps
        # reverse timesteps from 1000 to 0 and convert to tensor
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps: int = 50) -> None:
        self.num_inference_steps = num_inference_steps
        # steps go from 1000 to 0 but in inference less steps
        # are done by skipping step_ratio steps after each step
        step_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        alpha_cumprod = self.alpha_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(device=original_samples.device)
        
        # Calculate mean
        sqrt_alpha_cumprod = alpha_cumprod[timesteps] ** 0.5
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.flatten()
        while len(sqrt_alpha_cumprod.shape) < len(original_samples.shape):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
        # standard deviation
        sqrt_one_minus_alpha_prod = (1 - sqrt_alpha_cumprod[timesteps]**0.5)
        while len(sqrt_one_minus_alpha_prod.shape) > len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        # sample noise and add it to the image
        # according to equation (4) of DDPM paper
        # X =  mean + std * Z
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = (sqrt_alpha_cumprod * original_samples) + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def get_previous_timestep(self, timestep: int) -> int:
        return timestep - (self.num_training_steps //self.num_inference_steps)
    
    def get_variance(self, timestep: int) -> torch.Tensor:
        prev_t = self.get_previous_timestep(timestep)

        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_prev_t = self.alpha_cumprod[timestep] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_prev_t

        # Compute variance using formula (7) of DDPM paper
        variance = (1 - alpha_prod_prev_t) / (1 - alpha_prod_t) * current_beta_t
        variance = torch.clamp(variance, min=1e-20)

        return variance
    
    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
        """
        Remove noise from latents

        Args:
        model_output: noise predicted by the UNET
        latents: compressed image in latent space
        """
        t = timestep
        prev_t = self.get_previous_timestep(t)

        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_prev_t = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_prev_t = 1 - alpha_prod_prev_t
        current_alpha_t = alpha_prod_t / alpha_prod_prev_t
        current_beta_t = 1 - current_alpha_t

        # Compute the predicted original sample using formula (15) of the DDPM paper
        pred_original_sample = (latents - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5

        # Compute the coefficients for pred_original_sample x_0 and current sample x_t
        pred_original_sample_coeff = (alpha_prod_prev_t**0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = (current_alpha_t**0.5 * beta_prod_prev_t) / beta_prod_t

        # Compute the mean of the predicted previous sample
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            variance = (self.get_variance(timestep)**0.5) * noise

        # N(0, 1) -> N(mu, sigma2)
        # X = mu + sigma Ü Z where Z ~ N(0, 1)
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def set_strength(self, strength: float = 1.0) -> None:
        """
        Skip some denoising steps so that the output image resembles less
        the original input image
        """
        self.start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[self.start_step:]
