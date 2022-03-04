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

from IDEAS_models import Generator, StructureGenerator,Encoder,Extractor
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
    parser.add_argument("--num", type=int, default=1000)

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--npz_path", type=str, required=True)
    parser.add_argument("--dataset_type", choices=['offical_lmdb', 'resized_lmdb', 'normal'])

    args = parser.parse_args()

    # Load CheckPoints
    ckpt = torch.load(f"experiments/{args.exp_name}/checkpoints/{args.ckpt}.pt",
                      map_location=lambda storage, loc: storage)
    ckpt_args = ckpt["args"]

    result_dir = f'results/generate_IDEAS_samples_and_retrieve/{args.exp_name}/sigma={args.sigma}_delta={args.delta}'
    os.makedirs(result_dir, exist_ok=True)

    # Load Models
    encoder = Encoder(ckpt_args.channel).to(device)
    generator = Generator(ckpt_args.channel).to(device)
    stru_generator = StructureGenerator(ckpt_args.channel, N=ckpt_args.N).to(device)
    extractor = Extractor(ckpt_args.channel, N=ckpt_args.N).to(device)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device)

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
        # transforms.Resize((299, 299)),
        transforms.ToTensor()])

    dataset = set_dataset(args.dataset_type,args.dataset_path, transform,256)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=2)

    pred_arr = np.load(args.npz_path)
    print(pred_arr.shape)

    fake_images = []
    for i in range(1, args.num + 1):
        with torch.no_grad():
            message = messages[i - 1].unsqueeze(0)
            noise = noises[i - 1].unsqueeze(0)
            structure = stru_generator(noise)
            texture = torch.rand(size=(1, 2048)).to(device) * 2 - 1
            fake_image = generator(structure, texture)  # (-1,1)
            fake_images.append(fake_image)
            utils.save_image(fake_image,
                             f'{result_dir}/{i:06d}_generated.png',
                             normalize=True,
                             range=(-1, 1))
    fake_images = torch.concat(fake_images, dim=0)
    print(fake_images.shape)
    # fake_images = torch.nn.functional.interpolate(fake_images,size=(299,299),mode='bicubic')

    fake_preds = []
    for i in range(args.num // 50):
        fake_pred = inception(fake_images[i * 50:(i + 1) * 50])[0]
        fake_preds.append(fake_pred)
    fake_preds = torch.concat(fake_preds, dim=0)
    fake_preds = fake_preds.squeeze(3).squeeze(2).cpu().numpy()

    retrived_images = []
    retrived_preds = []
    BERs=[]
    for i in tqdm(range(1, fake_preds.shape[0] + 1)):
        distance = np.linalg.norm(pred_arr - fake_preds[i-1], ord=2, axis=1)
        min_distance_index = np.argmin(distance)
        retrieved_image = dataset.__getitem__(min_distance_index).to(device).unsqueeze(0)
        retrived_images.append(retrieved_image)
        retrived_preds.append(pred_arr[min_distance_index])

        retrieved_structure, _ = encoder(retrieved_image)
        retrieved_noise = extractor(retrieved_structure)
        retrieved_noise = retrieved_noise.reshape(shape=(1, ckpt_args.N * tensor_size * tensor_size))
        retrieved_message = tensor_to_message(retrieved_noise, sigma=args.sigma)

        BERs.append(torch.mean(torch.abs((messages[i-1].unsqueeze(0) - retrieved_message))).item())

        utils.save_image(dataset.__getitem__(min_distance_index),
                         f'{result_dir}/{i:06d}_training.png',
                         normalize=True,
                         range=(0, 1))

    # retrived_images = torch.cat(retrived_images,dim=0)
    # print(retrived_images.shape)
    # for i in range(args.num // 50):
    #     r_pred = inception(retrived_images[i * 50:(i + 1) * 50])[0]
    #     r_pred = r_pred.squeeze(3).squeeze(2).cpu().numpy()
    #     for j in range(r_pred.shape[0]):
    #         print((r_pred[j]-retrived_preds[i*50+j]))

    # utils.save_image(torch.cat([(fake_images+1)/2,retrived_images],dim=0),
    #                  f'{result_dir}/000000_total.png',
    #                  normalize=True,
    #                  nrow=args.num,
    #                  range=(0, 1))

        # print(fake_image_pred.shape)
        # fake_image_pred = np.repeat(fake_image_pred, pred_arr.shape[0], axis=0)
        # print(fake_image_pred.shape)

        # distance = pred_arr - fake_image_pred
        # distance = np.linalg.norm(pred_arr - fake_image_pred, ord=2, axis=1)
        #
        # min_distance_index = np.argmin(distance)
        #
        # print(distance[min_distance_index])
        # print(np.sort(distance))
        # print(pred_arr[min_distance_index])
        # print(fake_image_pred)
        # print(np.linalg.norm(pred_arr[min_distance_index] - fake_image_pred, ord=2, axis=1))
        #
        # retrieved_image = dataset.__getitem__(min_distance_index).to(device)
        # retrieved_image = retrieved_image.unsqueeze(0)
        #
        # retrieved_image_pred = inception(retrieved_image)[0]
        # retrieved_image_pred = retrieved_image_pred.squeeze(3).squeeze(2).cpu().numpy()
        #
        # print(retrieved_image_pred.shape)
        # print(fake_image_pred.shape)
        # retrived_distance = np.linalg.norm(retrieved_image_pred - fake_image_pred, ord=2, axis=1)
        # print(retrieved_image_pred)
        # print(retrived_distance)
        #
        # utils.save_image(dataset.__getitem__(min_distance_index),
        #                  f'{result_dir}/{i:06d}_training.png',
        #                  normalize=True,
        #                  range=(0, 1))
        # print(dataset.__getitem__(min_distance_index).shape)

        # print(distance.mean(axis=1).shape)
        # print(np.argmin(distance.mean(axis=1)))
    ACCs = 1 - np.array(BERs)
    ACC_avg = ACCs.mean()

    print(f"{args.exp_name} Sigma={args.sigma} Delta={args.delta}")
    print(f"ACC AVG: {ACC_avg:.6f}")
    print(f'Generating {args.exp_name} (Sigma={args.sigma}, Delta={args.delta}) Samples Done!')
