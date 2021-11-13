import argparse
import random
import os

import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils

import time
from mytime import time_change

from IDEAS_models import Encoder, Generator, StructureGenerator, Discriminator, CooccurDiscriminator, \
    DistributionDiscriminator, Extractor
from dataset import set_dataset


def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


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


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    (grad_real,) = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_recovery_loss(real_code, fake_code):
    loss = F.l1_loss(fake_code, real_code)

    return loss.mean()


def patchify_image(img, n_crop, min_size=1 / 8, max_size=1 / 4):
    crop_size = torch.rand(n_crop) * (max_size - min_size) + min_size
    batch, channel, height, width = img.shape
    target_h = int(height * max_size)
    target_w = int(width * max_size)
    crop_h = (crop_size * height).type(torch.int64).tolist()
    crop_w = (crop_size * width).type(torch.int64).tolist()

    patches = []
    for c_h, c_w in zip(crop_h, crop_w):
        c_y = random.randrange(0, height - c_h)
        c_x = random.randrange(0, width - c_w)

        cropped = img[:, :, c_y: c_y + c_h, c_x: c_x + c_w]
        cropped = F.interpolate(
            cropped, size=(target_h, target_w), mode="bilinear", align_corners=False
        )

        patches.append(cropped)

    patches = torch.stack(patches, 1).view(-1, channel, target_h, target_w)

    return patches


def train(
        args,
        loader,
        encoder,
        generator,
        stru_generator,
        extractor,
        discriminator,
        cooccur_discriminator,
        distribution_discriminator,
        g_optim,
        ex_optim,
        d_optim,
        encoder_ema,
        generator_ema,
        stru_generator_ema,
        extractor_ema,
        device,
        exp_name
):
    loader = sample_data(loader)

    loss_dict = {}

    accum = 0.5 ** (32 / (10 * 1000))

    start_time = time.time()

    for idx in range(1, args.num_iters + 1):
        iter_idx = idx + args.start_iter

        if iter_idx > args.num_iters:
            print("Done!")
            break

        real_img = next(loader)
        real_img = real_img.to(device)

        requires_grad(encoder, False)
        requires_grad(generator, False)
        requires_grad(stru_generator, False)
        requires_grad(extractor, False)
        requires_grad(discriminator, True)
        requires_grad(cooccur_discriminator, True)
        requires_grad(distribution_discriminator, True)

        '''
        Training Discriminators
        '''

        # Feature Encoding & Generation
        structure1, texture1 = encoder(real_img)
        secret_tensor = torch.rand(size=(structure1.shape[0], args.N, structure1.shape[2], structure1.shape[3]),
                           dtype=torch.float).cuda() * 2 - 1
        structure2 = stru_generator(secret_tensor)
        texture2 = torch.rand_like(texture1) * 2 - 1

        # Image Synthesis
        fake_img1 = generator(structure1, texture1)
        fake_img2 = generator(structure2, texture1)
        fake_img3 = generator(structure2, texture2)

        # L_{D,real}
        fake_pred = discriminator(torch.cat((fake_img1, fake_img2, fake_img3), 0))
        real_pred = discriminator(real_img)

        D_real_loss = d_logistic_loss(real_pred, fake_pred)

        # L_{D,texture}
        fake_patch = patchify_image(fake_img2, args.n_crop)
        real_patch = patchify_image(real_img, args.n_crop)
        ref_patch = patchify_image(real_img, args.ref_crop * args.n_crop)

        fake_patch_pred, ref_input = cooccur_discriminator(fake_patch, ref_patch, ref_batch=args.ref_crop)
        real_patch_pred, _ = cooccur_discriminator(real_patch, ref_input=ref_input)

        D_cooccur_loss = d_logistic_loss(real_patch_pred, fake_patch_pred)

        # L_{D,distribution}
        fake_uniform_pred = distribution_discriminator(texture1)
        real_uniform_pred = distribution_discriminator(texture2)

        D_dist_loss = d_logistic_loss(real_uniform_pred, fake_uniform_pred)

        # Record D Losses and optimize
        loss_dict["D_real_loss"] = D_real_loss
        loss_dict["D_cooccur_loss"] = D_cooccur_loss
        loss_dict["D_dist_loss"] = D_dist_loss

        d_optim.zero_grad()
        (D_real_loss + D_cooccur_loss + D_dist_loss).backward()
        d_optim.step()

        # Regularization

        if iter_idx % args.d_reg_every == 0:
            real_img.requires_grad = True
            real_pred = discriminator(real_img)
            D_real_r1_loss = d_r1_loss(real_pred, real_img)

            real_patch.requires_grad = True
            real_patch_pred, _ = cooccur_discriminator(real_patch, ref_patch, ref_batch=args.ref_crop)
            D_cooccur_r1_loss = d_r1_loss(real_patch_pred, real_patch)

            texture2.requires_grad = True
            real_uniform_pred = distribution_discriminator(texture2)
            D_dist_r1_loss = d_r1_loss(real_uniform_pred, texture2)

            d_optim.zero_grad()

            r1_loss_sum = args.r1 / 3 * D_real_r1_loss * args.d_reg_every
            r1_loss_sum += args.cooccur_r1 / 3 * D_cooccur_r1_loss * args.d_reg_every
            r1_loss_sum += args.dist_r1 / 3 * D_dist_r1_loss * args.d_reg_every
            r1_loss_sum.backward()

            d_optim.step()

            loss_dict["D_real_r1_loss"] = D_real_r1_loss
            loss_dict["D_cooccur_r1_loss"] = D_cooccur_r1_loss
            loss_dict["D_dist_r1_loss"] = D_dist_r1_loss

        requires_grad(encoder, True)
        requires_grad(generator, True)
        requires_grad(stru_generator, True)
        requires_grad(extractor, True)
        requires_grad(discriminator, False)
        requires_grad(cooccur_discriminator, False)
        requires_grad(distribution_discriminator, False)

        '''
        Training main components.
        '''

        # Feature Encoding & Generation
        structure1, texture1 = encoder(real_img)
        secret_tensor = torch.rand(
            size=(structure1.shape[0], args.N, structure1.shape[2], structure1.shape[3]),
            dtype=torch.float).cuda() * 2 - 1
        structure2 = stru_generator(secret_tensor)
        texture2 = torch.rand_like(texture1) * 2 - 1

        # Image Synthesis
        fake_img1 = generator(structure1, texture1)
        fake_img2 = generator(structure2, texture1)
        fake_img3 = generator(structure2, texture2)

        # L_{G,rec}
        G_rec_loss = F.l1_loss(fake_img1, real_img)

        # L_{G,real}
        fake_pred = discriminator(torch.cat((fake_img1, fake_img2, fake_img3), 0))
        G_real_loss = g_nonsaturating_loss(fake_pred)

        # L_{E,dist}
        fake_uniform_pred = distribution_discriminator(texture1)
        E_dist_loss = g_nonsaturating_loss(fake_uniform_pred)

        # L_{G,texture}
        fake_patch = patchify_image(fake_img2, args.n_crop)
        ref_patch = patchify_image(real_img, args.ref_crop * args.n_crop)
        fake_patch_pred, _ = cooccur_discriminator(fake_patch, ref_patch, ref_batch=args.ref_crop)
        G_texture_loss = g_nonsaturating_loss(fake_patch_pred)

        # L_{E,stru}
        if iter_idx > args.num_iters * 0.8:
            container_image = fake_img3
        else:
            container_image = fake_img2
        fake_structure2, _ = encoder(container_image)
        E_stru_loss = F.l1_loss(fake_structure2, structure2)

        # L_{REC}
        fake_tensor = extractor(fake_structure2)
        Ex_rec_loss = F.l1_loss(fake_tensor, secret_tensor)

        # Record Loss
        loss_dict["G_rec_loss"] = G_rec_loss
        loss_dict["G_real_loss"] = G_real_loss
        loss_dict["G_texture_loss"] = G_texture_loss
        loss_dict["E_dist_loss"] = E_dist_loss
        loss_dict["E_stru_loss"] = E_stru_loss
        loss_dict["Ex_rec_loss"] = Ex_rec_loss

        # L_G
        Loss_G = G_rec_loss + G_texture_loss + 2 * G_real_loss

        # L_E
        Loss_E = E_dist_loss + E_stru_loss

        # L_REC
        Loss_REC = Ex_rec_loss

        # L_{total}
        Loss_total = args.lambda_G * Loss_G + args.lambda_E * Loss_E + args.lambda_REC * Loss_REC

        # optimize Encoder and Generator
        g_optim.zero_grad()
        Loss_total.backward(retain_graph=True)
        g_optim.step()

        # optimize Extractor
        ex_optim.zero_grad()
        Loss_REC.backward()
        ex_optim.step()

        accumulate(encoder_ema, encoder, accum)
        accumulate(generator_ema, generator, accum)
        accumulate(stru_generator_ema, stru_generator, accum)
        accumulate(extractor_ema, extractor, accum)

        # Log
        if iter_idx % args.log_every == 0:
            G_rec_val = loss_dict["G_rec_loss"].mean().item()
            G_texture_val = loss_dict["G_texture_loss"].mean().item()
            G_real_val = loss_dict["G_real_loss"].mean().item()

            E_dist_val = loss_dict["E_dist_loss"].mean().item()
            E_stru_val = loss_dict["E_stru_loss"].mean().item()

            Ex_rec_val = loss_dict["Ex_rec_loss"].mean().item()

            now_time = time.time()
            used_time = now_time - start_time
            rest_time = (now_time - start_time) / idx * (args.num_iters - iter_idx)

            log_output = f"[{iter_idx:07d}/{args.num_iters:07}] Total_loss: {Loss_total.item():.4f}; " \
                         f"G_rec: {G_rec_val:.4f}; G_texture: {G_texture_val:.4f}; G_real: {G_real_val:.4f}; " \
                         f"E_dist: {E_dist_val:.4f}; E_stru: {E_stru_val:.4f}; Ex_rec: {Ex_rec_val:.4f} " \
                         f"used time: {time_change(used_time)};" \
                         f"rest time: {time_change(rest_time)}"

            print(log_output, flush=True)
            with open(f'{base_dir}/training_logs.txt', 'a') as fp:
                fp.write(f'{log_output}\n')


        # Output Samples
        if iter_idx % args.show_every == 0:
            with torch.no_grad():
                encoder_ema.eval()
                generator_ema.eval()
                stru_generator_ema.eval()
                extractor_ema.eval()

                # Sample a secret message and map it to secret tensor
                structure1, texture1 = encoder_ema(real_img)
                message = torch.randint(low=0, high=2, size=(
                    structure1.shape[0], args.N * structure1.shape[2] * structure1.shape[3]),
                                        dtype=torch.float).cuda()
                secret_tensor = message_to_tensor(message, sigma=1, delta=0.5).cuda()
                secret_tensor = secret_tensor.reshape(
                    shape=(structure1.shape[0], args.N, structure1.shape[2], structure1.shape[3]))

                # Translate the secret tensor to structure code
                structure2 = stru_generator_ema(secret_tensor)

                # Sample a texture
                texture2 = torch.rand_like(texture1) * 2 - 1

                # Image Synthesis
                fake_img1 = generator_ema(structure1, texture1)
                fake_img2 = generator_ema(structure2, texture1)
                fake_img3 = generator_ema(structure2, texture2)

                # Secret Tensor Extracting
                if iter_idx > args.num_iters * 0.8:
                    container_image = fake_img3
                    fake_img_used_as_container = 3
                else:
                    container_image = fake_img2
                    fake_img_used_as_container = 2
                recovered_structure2, _ = encoder_ema(container_image)
                recovered_secret_tensor = extractor_ema(recovered_structure2)

                tensor_recovering_loss = torch.mean(torch.abs(secret_tensor - recovered_secret_tensor))

                recovered_secret_tensor = recovered_secret_tensor.reshape(
                    shape=(structure1.shape[0], args.N * structure1.shape[2] * structure1.shape[3]))
                recovered_message = tensor_to_message(recovered_secret_tensor, sigma=1).cuda()

                BER = torch.mean(torch.abs(message - recovered_message))
                ACC = 1 - BER

                print(f'[Testing {iter_idx:07d}/{args.num_iters:07d}] sigma=1 delta=50% '
                      f'using synthesised image X_{fake_img_used_as_container} '
                      f'ACC of Msg: {ACC:.4f}; L1 loss of tensor: {tensor_recovering_loss:.4f}')

                sample = torch.cat((real_img, fake_img1, fake_img2, fake_img3), 0)

                utils.save_image(
                    sample,
                    f"{sample_dir}/{iter_idx:07d}.png",
                    nrow=int(args.batch_size),
                    normalize=True,
                    range=(-1, 1),
                )

                print(f'Sample images are saved in experiments/{exp_name}/samples')

        # Save models
        if iter_idx % args.save_every == 0:
            torch.save(
                {
                    'iter_idx': iter_idx,
                    'N': args.N,
                    "encoder": encoder.state_dict(),
                    "generator": generator.state_dict(),
                    "stru_generator": stru_generator.state_dict(),
                    "extractor": extractor.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "cooccur_discriminator": cooccur_discriminator.state_dict(),
                    "distribution_discriminator": distribution_discriminator.state_dict(),
                    "encoder_ema": encoder_ema.state_dict(),
                    "generator_ema": generator_ema.state_dict(),
                    "stru_generator_ema": stru_generator_ema.state_dict(),
                    "extractor_ema": extractor_ema.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "ex_optim": ex_optim.state_dict(),
                    "d_optim": d_optim.state_dict(),
                    "args": args,
                },
                f"{ckpt_dir}/{iter_idx}.pt",
            )

            print(f'Checkpoint is saved in experiments/{exp_name}/checkpoints')


if __name__ == "__main__":
    device = "cuda"

    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_type", choices=['offical_lmdb', 'resized_lmdb', 'normal'])
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--num_iters", type=int, default=1000000)
    parser.add_argument("--N", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.002)

    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=256)

    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--cooccur_r1", type=float, default=1)
    parser.add_argument("--dist_r1", type=float, default=1)

    parser.add_argument("--lambda_G", type=float, default=1)
    parser.add_argument("--lambda_E", type=float, default=1)
    parser.add_argument("--lambda_REC", type=float, default=5)

    parser.add_argument("--ref_crop", type=int, default=4)
    parser.add_argument("--n_crop", type=int, default=8)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--channel", type=int, default=32)
    parser.add_argument("--channel_multiplier", type=int, default=1)

    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--show_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=100000)

    args = parser.parse_args()

    base_dir = f"experiments/{args.exp_name}"
    ckpt_dir = f"{base_dir}/checkpoints"
    sample_dir = f"{base_dir}/samples"

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    with open(f"{base_dir}/training_config.txt", "wt") as fp:
        for k, v in vars(args).items():
            fp.write(f'{k}: {v}\n')
        fp.close()

    args.start_iter = 0

    encoder = Encoder(args.channel).to(device)
    generator = Generator(args.channel).to(device)
    stru_generator = StructureGenerator(args.channel, N=args.N).to(device)
    extractor = Extractor(args.channel, N=args.N).to(device)

    discriminator = Discriminator(args.image_size, channel_multiplier=args.channel_multiplier).to(device)
    cooccur_discriminator = CooccurDiscriminator(args.channel).to(device)
    distribution_discriminator = DistributionDiscriminator().to(device)

    encoder_ema = Encoder(args.channel).to(device)
    generator_ema = Generator(args.channel).to(device)
    stru_generator_ema = StructureGenerator(args.channel, N=args.N).to(device)
    extractor_ema = Extractor(args.channel, N=args.N).to(device)

    encoder_ema.eval()
    generator_ema.eval()
    stru_generator_ema.eval()
    extractor_ema.eval()

    accumulate(encoder_ema, encoder, 0)
    accumulate(generator_ema, generator, 0)
    accumulate(stru_generator_ema, stru_generator, 0)
    accumulate(extractor_ema, extractor, 0)

    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        list(generator.parameters())
        + list(encoder.parameters())
        + list(stru_generator.parameters()),
        lr=args.lr,
        betas=(0, 0.99),
    )
    ex_optim = optim.Adam(
        extractor.parameters(),
        lr=args.lr,
        betas=(0, 0.99),
    )
    d_optim = optim.Adam(
        list(discriminator.parameters())
        + list(cooccur_discriminator.parameters())
        + list(distribution_discriminator.parameters()),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(f"{ckpt_dir}/{args.ckpt}.pt",
                          map_location=lambda storage, loc: storage)

        args.start_iter = ckpt['iter_idx']

        encoder.load_state_dict(ckpt["encoder"])
        generator.load_state_dict(ckpt["generator"])
        stru_generator.load_state_dict(ckpt["stru_generator"])
        extractor.load_state_dict(ckpt["extractor"])

        discriminator.load_state_dict(ckpt["discriminator"])
        cooccur_discriminator.load_state_dict(ckpt["cooccur_discriminator"])
        distribution_discriminator.load_state_dict(ckpt["distribution_discriminator"])

        encoder_ema.load_state_dict(ckpt["encoder_ema"])
        generator_ema.load_state_dict(ckpt["generator_ema"])
        stru_generator_ema.load_state_dict(ckpt["stru_generator_ema"])
        extractor_ema.load_state_dict(ckpt["extractor_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        ex_optim.load_state_dict(ckpt["ex_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = set_dataset(args.dataset_type, args.dataset_path, transform, args.image_size)

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=data_sampler(dataset, shuffle=True)
    )

    print('Data Loaded')

    exp_name = args.exp_name

    train(
        args,
        loader,
        encoder,
        generator,
        stru_generator,
        extractor,
        discriminator,
        cooccur_discriminator,
        distribution_discriminator,
        g_optim,
        ex_optim,
        d_optim,
        encoder_ema,
        generator_ema,
        stru_generator_ema,
        extractor_ema,
        device,
        exp_name
    )
