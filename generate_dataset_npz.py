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

from FID.inception import InceptionV3



if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_type", choices=['offical_lmdb', 'resized_lmdb', 'normal'])
    parser.add_argument("--name", type=str,required=True)

    args = parser.parse_args()

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device)

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor()])

    dataset = set_dataset(args.dataset_type, args.dataset_path, transform, 256)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=50,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=2)

    pred_arr = np.empty((len(dataset), 2048))
    start_idx = 0
    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = inception(batch)[0]

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]

    np.save(f'../dataset/Statistics/dataset_pred_{args.name}',pred_arr)
