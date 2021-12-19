import argparse

import numpy as np
import torch
from torchvision import utils, transforms
import os
import warnings
from PIL import Image
import cv2 as cv
import kornia.filters as kornia
import pandas as pd

warnings.simplefilter('ignore')

from IDEAS_models import Encoder, Generator, StructureGenerator, Extractor
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
    parser.add_argument("--ckpt", type=int, default=100000)
    parser.add_argument("--sigma", type=int, default=1)
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--num", type=int, default=1000)
    parser.add_argument("--save_images", action='store_true')

    parser.add_argument('--GaussianNoise', action='store_true')
    parser.add_argument('--PepperNoise', action='store_true')
    parser.add_argument('--JPEG', action='store_true')
    parser.add_argument('--JPEG2000', action='store_true')
    parser.add_argument('--GaussianBlur', action='store_true')
    parser.add_argument('--MedianBlur', action='store_true')
    parser.add_argument('--all_attack', action='store_true')
    parser.add_argument('--show_tqdm', action='store_true')

    args = parser.parse_args()

    # Load CheckPoints
    ckpt = torch.load(f"experiments/{args.exp_name}/Robust_IDEAS/checkpoints/{args.ckpt}.pt",
                      map_location=lambda storage, loc: storage)
    ckpt_args = ckpt["args"]

    base_dir = f'results/test_Robust_IDEAS_accuracy_underattack/{args.exp_name}/sigma={args.sigma}_delta={args.delta}'

    # Make Dirs
    if args.save_images:
        os.makedirs(base_dir, exist_ok=True)

    if args.all_attack == True:
        args.GaussianNoise = True
        args.PepperNoise = True
        args.JPEG = True
        args.JPEG2000 = True
        args.GaussianBlur = True
        args.MedianBlur = True

    Normal_dir = f'{base_dir}/wo_Attack'
    GaussianNoise_dir = f'{base_dir}/GaussianNoise'
    PepperNoise_dir = f'{base_dir}/PepperNoise'
    JPEG_dir = f'{base_dir}/JPEG'
    JPEG2000_dir = f'{base_dir}/JPEG2000'
    GaussianBlur_dir = f'{base_dir}/GaussianBlur'
    MedianBlur_dir = f'{base_dir}/MedianBlur'

    Normal_BER = []
    GaussianNoise_BER = {}
    PepperNoise_BER = {}
    JPEG_BER = {}
    JPEG2000_BER = {}
    GaussianBlur_BER = {}
    MedianBlur_BER = {}

    GaussianNoise_Sigmas = [0.01, 0.05, 0.1]
    PepperNoise_Ps = [0.01, 0.05, 0.1]
    JPEG_Qualitys = [90, 70, 50]
    JPEG2000_Qualitys = [90, 70, 50]
    GaussianBlur_KSs = [3, 5, 7]
    MedianBlur_KSs = [3, 5, 7]

    # Load Models
    encoder = Encoder(ckpt_args.channel).to(device)
    generator = Generator(ckpt_args.channel).to(device)
    stru_generator = StructureGenerator(ckpt_args.channel, N=ckpt_args.N).to(device)
    extractor = Extractor(ckpt_args.channel, N=ckpt_args.N).to(device)

    encoder.load_state_dict(ckpt["encoder_ema"])
    generator.load_state_dict(ckpt["generator_ema"])
    stru_generator.load_state_dict(ckpt["stru_generator_ema"])
    extractor.load_state_dict(ckpt["extractor_ema"])

    encoder.eval()
    generator.eval()
    stru_generator.eval()
    extractor.eval()

    batch_size = 1
    image_size = ckpt_args.image_size
    tensor_size = int(image_size / 16)

    set_seed(1)

    messages = torch.randint(low=0, high=2, size=(args.num, ckpt_args.N * tensor_size * tensor_size * args.sigma))
    noises = message_to_tensor(messages, sigma=args.sigma, delta=args.delta)
    noises = noises.reshape(shape=(args.num, ckpt_args.N, tensor_size, tensor_size)).to(device)

    i = 1
    while i < args.num + 1:
        with torch.no_grad():
            # hiding
            message = messages[i - 1:i - 1 + batch_size]
            noise = noises[i - 1:i - 1 + batch_size]
            structure = stru_generator(noise)
            texture = torch.rand(size=(batch_size, 2048)).to(device) * 2 - 1
            fake_image = generator(structure, texture) / 2 + 0.5  # (0,1)

            if args.save_images:
                os.makedirs(Normal_dir, exist_ok=True)
                utils.save_image(
                    fake_image[0],
                    f'{Normal_dir}/{i:04d}.png',
                    normalize=True,
                    range=(0, 1)
                )

            if args.GaussianNoise == True:
                for sigma in GaussianNoise_Sigmas:
                    noised_image = fake_image + (torch.randn(size=fake_image.shape).to(device) * sigma)
                    noised_image = (noised_image.clamp(0, 1) * 255).int()
                    noised_image = (noised_image.float() / 255 * 2 - 1)  # -> (-1,1)

                    fake_structure, _ = encoder(noised_image)
                    fake_noise = extractor(fake_structure)
                    fake_noise = fake_noise.reshape(shape=(batch_size, ckpt_args.N * tensor_size * tensor_size))
                    fake_message = tensor_to_message(fake_noise, sigma=args.sigma)
                    BER = torch.mean(torch.abs(message - fake_message), dim=1).cpu().data.numpy()

                    if GaussianNoise_BER.get(f'sigma={sigma}') == None:
                        GaussianNoise_BER[f'sigma={sigma}'] = []
                    GaussianNoise_BER[f'sigma={sigma}'].append(BER)

                    if args.save_images:
                        result_dir = f'{GaussianNoise_dir}/sigma={sigma}'
                        os.makedirs(result_dir, exist_ok=True)
                        utils.save_image(noised_image[0],
                                         f'{result_dir}/{i:04d}.png',
                                         normalize=True,
                                         range=(-1, 1))

            if args.PepperNoise == True:
                for p in PepperNoise_Ps:
                    noise_mask = np.random.choice((0, 1, 2), size=(batch_size, 3, 256, 256), p=[(1 - p), p / 2, p / 2])
                    noised_image = fake_image.clone()
                    for j in range(batch_size):
                        for c in range(3):
                            for h in range(256):
                                for w in range(256):
                                    if noise_mask[j, c, h, w] == 1:
                                        noised_image[j, c, h, w] = 0
                                    elif noise_mask[j, c, h, w] == 2:
                                        noised_image[j, c, h, w] = 1
                    noised_image = (noised_image.clamp(0, 1) * 255).int()

                    noised_image = (noised_image.float() / 255 * 2 - 1)  # -> (-1,1)
                    fake_structure, _ = encoder(noised_image)
                    fake_noise = extractor(fake_structure)
                    fake_noise = fake_noise.reshape(shape=(batch_size, ckpt_args.N * tensor_size * tensor_size))
                    fake_message = tensor_to_message(fake_noise, sigma=args.sigma)
                    BER = torch.mean(torch.abs(message - fake_message), dim=1).cpu().data.numpy()

                    if PepperNoise_BER.get(f'p={p}') == None:
                        PepperNoise_BER[f'p={p}'] = []
                    PepperNoise_BER[f'p={p}'].append(BER)

                    if args.save_images:
                        result_dir = f'{PepperNoise_dir}/p={p}'
                        os.makedirs(result_dir, exist_ok=True)
                        utils.save_image(noised_image[0],
                                         f'{result_dir}/{i:04d}.png',
                                         normalize=True,
                                         range=(-1, 1))

            if args.JPEG == True:
                for quality in JPEG_Qualitys:
                    result_dir = f'{JPEG_dir}/quality={quality}'
                    os.makedirs(result_dir, exist_ok=True)

                    fake_jpeg_images = fake_image.clone()
                    for j in range(fake_image.shape[0]):
                        fake_jpeg_image = fake_image[j].cpu().clone()
                        fake_jpeg_image = transforms.ToPILImage()(fake_jpeg_image)
                        # Save as JPEG
                        fake_jpeg_image.save(f'{result_dir}/{i + j:04d}.jpg', quality=quality)
                        # PIL.Image -> Tensor
                        fake_jpeg_image = Image.open(f'{result_dir}/{i + j:04d}.jpg')
                        fake_jpeg_images[j] = transforms.ToTensor()(fake_jpeg_image).to(device)

                    fake_jpeg_images = fake_jpeg_images * 2 - 1  # (0,1) -> (-1,1)
                    fake_structure, _ = encoder(fake_jpeg_images)
                    fake_noise = extractor(fake_structure)
                    fake_noise = fake_noise.reshape(shape=(batch_size, ckpt_args.N * tensor_size * tensor_size))
                    fake_message = tensor_to_message(fake_noise, sigma=args.sigma)
                    BER = torch.mean(torch.abs(message - fake_message), dim=1).cpu().data.numpy()

                    if JPEG_BER.get(f'quality={quality}') == None:
                        JPEG_BER[f'quality={quality}'] = []
                    JPEG_BER[f'quality={quality}'].append(BER)

            if args.JPEG2000 == True:
                for quality in JPEG2000_Qualitys:
                    result_dir = f'{JPEG2000_dir}/quality={quality}'
                    os.makedirs(result_dir, exist_ok=True)

                    fake_jpeg_images = fake_image.clone() * 255  # ->(0,255)
                    for j in range(fake_jpeg_images.shape[0]):
                        fake_jpeg_image = fake_jpeg_images[j].cpu().clone()
                        fake_jpeg_image = np.asarray(fake_jpeg_image).transpose((1, 2, 0))
                        # np.array -> OpenCV Image
                        fake_jpeg_image = cv.cvtColor(fake_jpeg_image, cv.COLOR_RGB2BGR)
                        # Save as JPEG
                        cv.imwrite(f'{result_dir}/{i + j:04d}.jpg', fake_jpeg_image,
                                   [int(cv.IMWRITE_JPEG_QUALITY), quality])
                        # OpenCV.Image -> np.array
                        fake_jpeg_image = cv.imread(f'{result_dir}/{i + j:04d}.jpg')
                        fake_jpeg_image = cv.cvtColor(fake_jpeg_image, cv.COLOR_BGR2RGB).transpose((2, 0, 1))
                        # np.array -> Tensor
                        fake_jpeg_images[j] = torch.from_numpy(fake_jpeg_image).float().to(device)

                    fake_jpeg_images = fake_jpeg_images / 255 * 2 - 1  # (0,255) -> (-1,1)
                    fake_structure, _ = encoder(fake_jpeg_images)
                    fake_noise = extractor(fake_structure)
                    fake_noise = fake_noise.reshape(shape=(batch_size, ckpt_args.N * tensor_size * tensor_size))
                    fake_message = tensor_to_message(fake_noise, sigma=args.sigma)
                    BER = torch.mean(torch.abs(message - fake_message), dim=1).cpu().data.numpy()

                    if JPEG2000_BER.get(f'quality={quality}') == None:
                        JPEG2000_BER[f'quality={quality}'] = []
                    JPEG2000_BER[f'quality={quality}'].append(BER)

            if args.GaussianBlur == True:
                for ks in GaussianBlur_KSs:
                    # Gaussian Blur
                    blurred_image = kornia.gaussian_blur2d(fake_image, kernel_size=(ks, ks), sigma=(1, 1))
                    blurred_image = (blurred_image.clamp(0, 1) * 255).int()

                    blurred_image = (blurred_image.float() / 255 * 2 - 1)  # -> (-1,1)
                    fake_structure, _ = encoder(blurred_image)
                    fake_noise = extractor(fake_structure)
                    fake_noise = fake_noise.reshape(shape=(batch_size, ckpt_args.N * tensor_size * tensor_size))
                    fake_message = tensor_to_message(fake_noise, sigma=args.sigma)
                    BER = torch.mean(torch.abs(message - fake_message), dim=1).cpu().data.numpy()

                    if GaussianBlur_BER.get(f'ks={ks}') == None:
                        GaussianBlur_BER[f'ks={ks}'] = []
                    GaussianBlur_BER[f'ks={ks}'].append(BER)

                    if args.save_images:
                        result_dir = f'{GaussianBlur_dir}/KernelSize_{ks}'
                        os.makedirs(result_dir, exist_ok=True)
                        utils.save_image(blurred_image[0],
                                         f'{result_dir}/{i:04d}.png',
                                         normalize=True,
                                         range=(-1, 1))

            if args.MedianBlur == True:
                for ks in MedianBlur_KSs:

                    # Median Blur
                    blurred_image = kornia.median_blur(fake_image, kernel_size=(ks, ks))
                    blurred_image = (blurred_image.clamp(0, 1) * 255).int()

                    blurred_image = (blurred_image.float() / 255 * 2 - 1)  # -> (-1,1)
                    fake_structure, _ = encoder(blurred_image)
                    fake_noise = extractor(fake_structure)
                    fake_noise = fake_noise.reshape(shape=(batch_size, ckpt_args.N * tensor_size * tensor_size))
                    fake_message = tensor_to_message(fake_noise, sigma=args.sigma)
                    BER = torch.mean(torch.abs(message - fake_message), dim=1).cpu().data.numpy()

                    if MedianBlur_BER.get(f'ks={ks}') == None:
                        MedianBlur_BER[f'ks={ks}'] = []
                    MedianBlur_BER[f'ks={ks}'].append(BER)

                    if args.save_images:
                        result_dir = f'{MedianBlur_dir}/KernelSize={ks}'
                        os.makedirs(result_dir, exist_ok=True)
                        utils.save_image(blurred_image[0],
                                         f'{result_dir}/{i:04d}.png',
                                         normalize=True,
                                         range=(-1, 1))

            fake_structure, _ = encoder(fake_image * 2 - 1)
            fake_noise = extractor(fake_structure)
            fake_noise = fake_noise.reshape(shape=(batch_size, ckpt_args.N * tensor_size * tensor_size))
            fake_message = tensor_to_message(fake_noise, sigma=args.sigma)

            Normal_BER.append(torch.mean(torch.abs((message - fake_message))).item())

            i += 1

    dataframe = pd.DataFrame()
    AttName_list = []
    Acc_list = []

    Acc = 1 - np.mean(np.hstack(Normal_BER))
    print(f'\n Robust {args.exp_name}')
    print(f'Acc w/o attack: {Acc}')
    AttName_list.append('w/o Att.')
    Acc_list.append(Acc)

    if args.GaussianNoise:
        for sigma in GaussianNoise_Sigmas:
            print(f'Gaussian Noise Sigma={args.sigma} Sigma={sigma}')
            Acc = 1 - np.mean(np.hstack(GaussianNoise_BER[f'sigma={sigma}']))
            print(f"ACC: {Acc}")
            AttName_list.append(f'GN sig.={sigma}')
            Acc_list.append(Acc)

    if args.PepperNoise:
        for p in PepperNoise_Ps:
            print(f'Pepper Noise Sigma={args.sigma} P={p}')
            Acc = 1 - np.mean(np.hstack(PepperNoise_BER[f'p={p}']))
            print(f"ACC: {Acc}")
            AttName_list.append(f'PN p={p}')
            Acc_list.append(Acc)

    if args.JPEG:
        for quality in JPEG_Qualitys:
            print(f'JPEG Compression Sigma={args.sigma} Quality={quality}')
            Acc = 1 - np.mean(np.hstack(JPEG_BER[f'quality={quality}']))
            print(f"ACC: {Acc}")
            AttName_list.append(f'JC qua.={quality}')
            Acc_list.append(Acc)

    if args.JPEG2000:
        for quality in JPEG2000_Qualitys:
            print(f'JPEG2000 Compression Sigma={args.sigma} Quality={quality}')
            Acc = 1 - np.mean(np.hstack(JPEG2000_BER[f'quality={quality}']))
            print(f"ACC: {Acc}")
            AttName_list.append(f'J2C qua.={quality}')
            Acc_list.append(Acc)

    if args.GaussianBlur:
        for ks in GaussianBlur_KSs:
            print(f'Gaussian Blur Sigma={args.sigma} KS={ks}')
            Acc = 1 - np.mean(np.hstack(GaussianBlur_BER[f'ks={ks}']))
            print(f"ACC: {Acc}")
            AttName_list.append(f'GB K.S.={ks}')
            Acc_list.append(Acc)

    if args.MedianBlur:
        for ks in MedianBlur_KSs:
            print(f'Median Blur Sigma={args.sigma} KS={ks}')
            Acc = 1 - np.mean(np.hstack(MedianBlur_BER[f'ks={ks}']))
            print(f"ACC: {Acc}")
            AttName_list.append(f'MB K.S.={ks}')
            Acc_list.append(Acc)

    dataframe['Att. Name'] = AttName_list
    dataframe['Acc'] = Acc_list
    dataframe.to_csv(f'{base_dir}/acc_result.csv')
