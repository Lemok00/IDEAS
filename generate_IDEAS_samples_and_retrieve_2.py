# CUDA_VISIBLE_DEVICES=2 python generate_IDEAS_samples_and_retrieve.sh.py --exp_name IDEAS_Bedroom_N=1_lambdaREC=10 --dataset_type normal --dataset_path ../dataset/Bedroom/Samples_256/ --npz_path ../dataset/Statistics/dataset_pred_Bedroom.npy
# CUDA_VISIBLE_DEVICES=2 python generate_IDEAS_samples_and_retrieve.sh.py --exp_name IDEAS_FFHQ_N=1_lambdaREC=10 --dataset_type resized_lmdb --dataset_path ../dataset/FFHQ/prepared_train_256/ --npz_path ../dataset/Statistics/dataset_pred_FFHQ.npy
# CUDA_VISIBLE_DEVICES=2 python generate_IDEAS_samples_and_retrieve.sh.py --exp_name IDEAS_Church_N=1_lambdaREC=10 --dataset_type offical_lmdb --dataset_path ../dataset/Church/church_outdoor_train_lmdb/ --npz_path ../dataset/Statistics/dataset_pred_Church.npy

import argparse

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import utils, transforms
import os
import warnings
from tqdm import tqdm
from imutils.paths import list_files

from dataset import set_dataset

warnings.simplefilter('ignore')

from IDEAS_models import Generator, StructureGenerator, Encoder, Extractor
import random

from FID.inception import InceptionV3


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


IMG_EXTENSIONS = ['webp', '.png', '.jpg', '.jpeg', '.ppm', '.bmp', '.pgm', '.tif', '.tiff']

if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--ckpt", type=int, default=900000)
    parser.add_argument("--sigma", type=int, default=1)
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--num", type=int, default=100)

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--npz_path", type=str, required=True)
    parser.add_argument("--dataset_type", choices=['offical_lmdb', 'resized_lmdb', 'normal'])

    args = parser.parse_args()

    # Load CheckPoints
    ckpt = torch.load(f"experiments/{args.exp_name}/checkpoints/{args.ckpt}.pt",
                      map_location=lambda storage, loc: storage)
    ckpt_args = ckpt["args"]

    result_dir = f'results/generate_IDEAS_samples_and_retrieve_02/{args.exp_name}/sigma={args.sigma}_delta={args.delta}'
    os.makedirs(result_dir, exist_ok=True)

    # Load Models
    encoder = Encoder(ckpt_args.channel).to(device)
    generator = Generator(ckpt_args.channel).to(device)
    stru_generator = StructureGenerator(ckpt_args.channel, N=ckpt_args.N).to(device)
    extractor = Extractor(ckpt_args.channel, N=ckpt_args.N).to(device)

    # Find total parameters and trainable parameters
    encoder_total_params = sum(p.numel() for p in encoder.parameters())
    print(f'{encoder_total_params:,} encoder total parameters.')

    generator_total_params = sum(p.numel() for p in generator.parameters())
    print(f'{generator_total_params:,} generator total parameters.')

    stru_generator_total_params = sum(p.numel() for p in stru_generator.parameters())
    print(f'{stru_generator_total_params:,} stru_generator total parameters.')

    extractor_total_params = sum(p.numel() for p in extractor.parameters())
    print(f'{extractor_total_params:,} extractor total parameters.')

    print(
        f'{encoder_total_params + generator_total_params + stru_generator_total_params + extractor_total_params:,} total parameters.')

    encoder.load_state_dict(ckpt["encoder_ema"])
    generator.load_state_dict(ckpt["generator_ema"])
    stru_generator.load_state_dict(ckpt["stru_generator_ema"])
    extractor.load_state_dict(ckpt["extractor_ema"])

    encoder.eval()
    generator.eval()
    stru_generator.eval()
    extractor.eval()

    image_size = ckpt_args.image_size
    tensor_size = int(image_size / 16)

    messages = torch.randint(low=0, high=2, size=(args.num, ckpt_args.N * 16 * 16 * args.sigma))
    noises = message_to_tensor(messages, sigma=args.sigma, delta=args.delta)
    noises = noises.reshape(shape=(args.num, ckpt_args.N, tensor_size, tensor_size)).to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    ])

    dataset = set_dataset(args.dataset_type, args.dataset_path, transform, 256)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=2)

    training_structures = np.load(args.npz_path)
    # training_structures = training_structures[:10000]
    # print(training_structures)
    print(training_structures.shape)

    # training_structures = np.empty((len(dataset), 256))
    # start_idx = 0
    # for idx, batch in tqdm(enumerate(dataloader)):
    #     batch = batch.to(device)
    #
    #     with torch.no_grad():
    #         structure, _ = encoder(batch)
    #         structure = extractor(structure)
    #
    #     structure = structure.view(structure.shape[0], -1).cpu().numpy()
    #     training_structures[idx] = structure[0]
    #
    # np.save(f'../dataset/Statistics/noises_Church', training_structures)

    fake_images = []
    retrived_images = []
    for i in range(1, args.num + 1):
        with torch.no_grad():
            message = messages[i - 1].unsqueeze(0)
            noise = noises[i - 1].unsqueeze(0)
            structure = stru_generator(noise)
            # previous_structure =
            texture = torch.rand(size=(1, 2048)).to(device) * 2 - 1
            fake_image = generator(structure, texture)  # (-1,1)
            fake_images.append(fake_image)
            utils.save_image(fake_image,
                             f'{result_dir}/{i:06d}_synthesised.png',
                             normalize=True,
                             range=(-1, 1))

            # structure, _ = encoder(fake_image)

            structure = structure.view(structure.shape[0], -1).cpu().numpy()

            distance = np.linalg.norm(training_structures - noise.view(1, -1).cpu().numpy(), ord=2, axis=1)
            min_distance_index = np.argmin(distance)
            retrieved_image = dataset.__getitem__(min_distance_index).to(device).unsqueeze(0)
            retrived_images.append(retrieved_image)

            utils.save_image(retrieved_image,
                             f'{result_dir}/{i:06d}_retrieved.png',
                             normalize=True,
                             range=(-1, 1))

    fake_images = torch.concat(fake_images, dim=0)
    retrived_images = torch.concat(retrived_images, dim=0)

    utils.save_image(torch.cat([fake_images, retrived_images], dim=0),
                     f'{result_dir}/000000_total.png',
                     normalize=True,
                     nrow=args.num,
                     range=(-1, 1))
