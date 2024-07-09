# Coding stable diffusion from scratch using Pytorch

The model was coded following this [YouTube tutorial](https://www.youtube.com/watch?v=ZBKpAp_6TGI) by Umar Jamil.
The model is explained in the paper Denoising diffusion probabilistic models (Ho, J., Jain, A. and Abbeel P., 2020).

## Warning
The code doesn't work yet. The main purpose of this project is understanding the architecture of Stable Diffusion.

## Notes on Stable Diffusion

Stable diffusion is a text-to-image deep learning model, based on diffusion models.

Stable diffusion is a generative model. A generative model learns a joint probability distribution that depends on all variables/features of the data. This probability distribution can be sampled to create new instances of data.

Forward process (Noisification Markov Chain): Gaussian noise is added to an input image over T timesteps until the image becomes pure noise. The forward process admits sampling the image at a timestep t (between 0 and T) in closed form (not iteratively using markov chain).
Reverse process: Remove noise from pure noise using neural networks until you get the original image. The model learns parameters on how to denoise the data (mean and variance of the data).

We want to learn the parameters of a latent space (similar to a variational autoencoder). To do it, the likelihood of the the data given a set of parameters has to be maximized. As a result, the evidence lower bound (ELBO) is maximized.

During training, the model predicts/detects how much noise there is in noisified image at any timestep.

To generate new data, the model continuously remove noise from pure noise until a new image is generated. To control which image will be generated during denoisification, a conditional/context signal is used to influence how the model removes the noise, such that an image for a specific prompt is generated.

Classifier-free guidance: Train a model such that sometimes an image is generated using a prompt and sometimes without prompt. This way, an output can be conditioned to resemble a given prompt up to some configurable extent.

CLIP (Contrastive Language-Image Pre-training): Train model to match an image to some specific prompt.

Variational Autoencoder (VAE): used to compress image (high-dimensional data) step by step and learn latent variables of data

Text-To-Image: Text prompt is passed to CLIP Encoder. Sample pure noise, encode it with variational autoencoder and get latent representation of noise (compressed pure noise). U-Net detects how much noise is in the image. Conditional signal sent from the CLIP Encoder (passed through prompt embeddings) is sent to the U-Net. The U-Net calculates how much noise has to be removed for the image to look like an image that matches the prompt. The image is iteratively denoised over multiple timesteps until there is no more noise remaining in the image. The output of the U-Net is a latent variable that passes through the decoder of the variational autocoder to generate an image.

Image-To-Image: Instead of sampling pure noise at the beginning, an input image is given to the model. The model adds some noise to the image. The more noise is given, the more the model is able to modify the image.

## Notes

It uses a pretrained model.