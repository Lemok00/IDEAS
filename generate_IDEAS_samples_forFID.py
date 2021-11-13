import argparse

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import utils
import os
import warnings

warnings.simplefilter('ignore')

from IDEAS_models import Encoder, Generator, StructureGenerator
import random


def message_to_tensor(message, sigma, delta):
    secret_tensor = torch.zeros(size=(message.shape[0], message.shape[1] // sigma))
    step = 2 / 2 ** sigma
    random_interval_size = step * delta
    for i in range(secret_tensor.shape[0]):
        for j in range(secret_tensor.shape[1]):
            message_num = 0
            for idx in range(sigma):
                message_num += message[i][j * sigma + idx] * 2 ** (sigma - idx - 1)
            secret_tensor[i][j] = step * (message_num + 0.5) - 1
            secret_tensor[i][j] = secret_tensor[i][j] + (
                    torch.rand(1)[0] * random_interval_size * 2 - random_interval_size)
    return secret_tensor


def tensor_to_message(secret_tensor, sigma):
    message = torch.zeros(size=(secret_tensor.shape[0], secret_tensor.shape[1] * sigma))
    step = 2 / 2 ** sigma
    secret_tensor = torch.clamp(secret_tensor, min=-1, max=1) + 1
    for i in range(secret_tensor.shape[0]):
        for j in range(secret_tensor.shape[1]):
            message_num = int(secret_tensor[i][j] / step)
            for idx in range(sigma):
                if message_num >= 2 ** (sigma - idx - 1):
                    message[i][j * sigma + idx] = 1
                    message_num -= 2 ** (sigma - idx - 1)
    return message


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--ckpt", type=int, default=900000)
    parser.add_argument("--sigma", type=int, default=1)
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--num", type=int, default=10000)

    args = parser.parse_args()

    # Load CheckPoints
    ckpt = torch.load(f"experiments/{args.exp_name}/checkpoints/{args.ckpt}.pt",
                      map_location=lambda storage, loc: storage)
    ckpt_args = ckpt["args"]

    result_dir = f'results/generate_IDEAS_samples_forFID/{args.exp_name}/sigma={args.sigma}_delta={args.delta}'
    os.makedirs(result_dir, exist_ok=True)

    # Load Models
    encoder = Encoder(ckpt_args.channel).to(device)
    generator = Generator(ckpt_args.channel).to(device)
    stru_generator = StructureGenerator(ckpt_args.channel, N=ckpt_args.N).to(device)

    encoder.load_state_dict(ckpt["encoder_ema"])
    generator.load_state_dict(ckpt["generator_ema"])
    stru_generator.load_state_dict(ckpt["stru_generator_ema"])

    encoder.eval()
    generator.eval()
    stru_generator.eval()

    image_size = ckpt_args.image_size
    tensor_size = int(image_size / 16)

    messages = torch.randint(low=0, high=2, size=(args.num, ckpt_args.N * 16 * 16 * args.sigma))
    noises = message_to_tensor(messages, sigma=args.sigma, delta=args.delta)
    noises = noises.reshape(shape=(args.num, ckpt_args.N, tensor_size, tensor_size)).to(device)

    for i in range(1, args.num + 1):
        with torch.no_grad():
            message = messages[i - 1].unsqueeze(0)
            noise = noises[i - 1].unsqueeze(0)
            structure = stru_generator(noise)
            texture = torch.rand(size=(1, 2048)).to(device) * 2 - 1
            fake_image = generator(structure, texture)  # (-1,1)

            utils.save_image(fake_image,
                             f'{result_dir}/{i:06d}.png',
                             normalize=True,
                             range=(-1, 1))

        if i % 1000 == 0:
            print(f'Generating {args.exp_name} (Sigma={args.sigma}, Delta={args.delta}) Samples [{i}/{args.num}]!',
                  flush=True)

print(f'Generating {args.exp_name} (Sigma={args.sigma}, Delta={args.delta}) Samples Done!')
