import torch, os, time
from torch import optim, nn
from tqdm.auto import tqdm
import wandb
from .utils import config
from .data import train_loader
from .pixart_dit import PixartDit
from katara import count_params
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL


pixart_model = PixartDit()

params = count_params(pixart_model)

sd_vae: AutoencoderKL = AutoencoderKL.from_pretrained(config.sd_vae_id) 

def train_step(model: nn.Module = pixart_model, train_loader=train_loader):
    train_loss = 0.0
    
    for step, batch in tqdm(enumerate(train_loader), desc='training steps'):
        image, text_cond = batch
        latents = sd_vae.encode(image).latent_dist
        latents = latents.mode() * config.scale_factor

        batch_size = latents.shape[0]
        timesteps = torch.randint(0, config.sample_steps, (batch_size,), device=latents.device)
        