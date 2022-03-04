#  CUDA_VISIBLE_DEVICES=2 python generate_dataset_npz.py --name Bedroom --dataset_type normal --dataset_path ../dataset/Bedroom/Samples_256/
# CUDA_VISIBLE_DEVICES=2 python generate_dataset_npz.py --name FFHQ --dataset_type resized_lmdb --dataset_path ../dataset/FFHQ/prepared_train_256/
# CUDA_VISIBLE_DEVICES=2 python generate_dataset_npz.py --name Church --dataset_type offical_lmdb --dataset_path ../dataset/Church/church_outdoor_train_lmdb/
import argparse

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import utils,transforms
import os
import warnings
from tqdm import tqdm

from dataset import set_dataset

warnings.simplefilter('ignore')

from IDEAS_models import Generator, StructureGenerator
import random

from IDEAS_models import Generator, StructureGenerator, Encoder, Extractor

if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--ckpt", type=int, default=900000)

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_type", choices=['offical_lmdb', 'resized_lmdb', 'normal'])
    parser.add_argument("--name", type=str,required=True)

    args = parser.parse_args()

    # Load CheckPoints
    ckpt = torch.load(f"experiments/{args.exp_name}/checkpoints/{args.ckpt}.pt",
                      map_location=lambda storage, loc: storage)
    ckpt_args = ckpt["args"]

    # Load Models
    encoder = Encoder(ckpt_args.channel).to(device)
    encoder.load_state_dict(ckpt["encoder_ema"])
    encoder.eval()

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

    training_structures = np.empty((len(dataset), 8 * 16 * 16))
    start_idx = 0
    for idx, batch in tqdm(enumerate(dataloader)):
        batch = batch.to(device)

        with torch.no_grad():
            structure, _ = encoder(batch)

        structure = structure.view(structure.shape[0], -1).cpu().numpy()
        training_structures[idx] = structure[0]

    np.save(f'../dataset/Statistics/structures_{args.name}',training_structures)
