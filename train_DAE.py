import argparse

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import utils
import os
import random
from torch import optim
from torch.nn import functional as F

from IDEAS_models import Generator, StructureGenerator, DenoisingAutoencoder

if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--iter", type=int, default=100000)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.002)

    args = parser.parse_args()

    base_dir = f"experiments/{args.exp_name}"
    ideas_ckpt_dir = f"{base_dir}/checkpoints"

    # Load CheckPoints
    ckpt = torch.load(f"{ideas_ckpt_dir}/{args.ckpt}.pt",
                      map_location=lambda storage, loc: storage)
    ckpt_args = ckpt["args"]
    imgsize = ckpt_args.image_size

    dae_dir = f'experiments/{args.exp_name}/DAE'
    dae_ckpt_dir = f"{dae_dir}/checkpoints"
    dae_sample_dir = f"{dae_dir}/samples"

    # Make Dirs
    os.makedirs(f'{dae_ckpt_dir}', exist_ok=True)
    os.makedirs(f'{dae_sample_dir}', exist_ok=True)

    # Load Models
    generator = Generator(ckpt_args.channel).to(device)
    stru_generator = StructureGenerator(ckpt_args.channel, N=ckpt_args.N).to(device)
    dae = DenoisingAutoencoder(ckpt_args.channel).to(device)

    generator.load_state_dict(ckpt["generator_ema"])
    stru_generator.load_state_dict(ckpt["stru_generator_ema"])

    generator.eval()
    stru_generator.eval()
    dae.train()

    # Optimizer
    dae_optim = optim.Adam(
        dae.parameters(),
        lr=args.lr,
        betas=(0, 0.99)
    )

    batch = args.batch

    for i in range(1, args.iter + 1):
        # Generate Images
        tensor = torch.rand(size=(batch, ckpt_args.N, imgsize // 16, imgsize // 16)).cuda() * 2 - 1
        structure = stru_generator(tensor)
        texture = torch.rand(size=(batch, 2048)).cuda() * 2 - 1
        image = generator(structure, texture)

        # Add Gaussian Noising
        random_sigma = torch.rand(1)[0] * 0.2
        noised_image = image + torch.randn(size=image.shape).cuda() * (random_sigma)
        noised_image = noised_image.clamp(-1, 1)

        # Denoising
        denoised_image = dae(noised_image)

        # L1 Loss between Original_image and Denoised_image
        denoise_loss = F.l1_loss(denoised_image, image)
        # L1 Loss between Original_image and Noised_image
        noise_loss = F.l1_loss(noised_image, image)

        dae_optim.zero_grad()
        denoise_loss.backward()
        dae_optim.step()

        if i % 1000 == 0:
            print(f'[{i:06d}/{args.iter}] '
                  f'Denoised Loss:{torch.mean(torch.abs(image - denoised_image)) / 2 * 255}; '
                  f'Noised Loss: {torch.mean(torch.abs(image - noised_image)) / 2 * 255}')

        if i % 5000 == 0:
            utils.save_image([image[0], noised_image[0], denoised_image[0],
                              torch.zeros(size=image[0].shape).to(device),
                              torch.abs(image[0] - noised_image[0]) * 2 - 1,
                              torch.abs(image[0] - denoised_image[0]) * 2 - 1],
                             f'{dae_sample_dir}/{i:06d}.png',
                             nrow=3,
                             normalize=True,
                             range=(-1, 1))

        if i % 50000 == 0:
            torch.save(
                {
                    "dae": dae.state_dict(),
                    "dae_optim": dae_optim.state_dict(),
                    "args": args
                },
                f"{dae_ckpt_dir}/{i:06d}.pt"
            )
