import random
import os
import argparse
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils

import time
from mytime import time_change

from IDEAS_models import Encoder, Generator, StructureGenerator, Discriminator, CooccurDiscriminator, \
    DistributionDiscriminator, Extractor, DenoisingAutoencoder
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
        dae,
        device
):
    loader = sample_data(loader)

    loss_dict = {}

    accum = 0.5 ** (32 / (10 * 1000))

    start_time = time.time()

    for idx in range(1, args.num_iters+1):
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
        requires_grad(dae, False)
        requires_grad(discriminator, True)
        requires_grad(cooccur_discriminator, True)
        requires_grad(distribution_discriminator, True)

        '''
        Training Discriminators
        '''

        # Feature Encoding & Generation
        structure1, texture1 = encoder(real_img)
        secret_tensor = torch.rand(size=(structure1.shape[0], args.N, structure1.shape[2], structure1.shape[3]),
                                   dtype=torch.float).cuda() * 2 - 1  # batch*in_channel*8*8
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

        # Add Gaussian Noise
        random_sigma = torch.rand(1)[0] * 0.2
        noised_fake_img2 = fake_img2 + torch.randn(size=fake_img2.shape).cuda() * (random_sigma)
        noised_fake_img3 = fake_img3 + torch.randn(size=fake_img3.shape).cuda() * (random_sigma)
        noised_fake_img2 = noised_fake_img2.clamp(-1, 1)
        noised_fake_img3 = noised_fake_img3.clamp(-1, 1)

        with torch.no_grad():
            denoised_fake_img2 = dae(noised_fake_img2)
            denoised_fake_img3 = dae(noised_fake_img3)

        # Re-Extract
        recovered_structure_2, _ = encoder(fake_img2)
        recovered_structure_3, _ = encoder(fake_img3)
        noised_recovered_structure_2, _ = encoder(noised_fake_img2)
        noised_recovered_structure_3, _ = encoder(noised_fake_img3)
        denoised_recovered_structure_2, _ = encoder(denoised_fake_img2)
        denoised_recovered_structure_3, _ = encoder(denoised_fake_img3)

        original_stru_extract_loss = F.l1_loss(recovered_structure_2, structure2) + \
                                     F.l1_loss(recovered_structure_3, structure2)
        noised_stru_extract_loss = F.l1_loss(noised_recovered_structure_2, structure2) + \
                                   F.l1_loss(noised_recovered_structure_3, structure2)
        denoised_stru_extract_loss = F.l1_loss(denoised_recovered_structure_2, structure2) + \
                                     F.l1_loss(denoised_recovered_structure_3, structure2)

        E_stru_loss = (original_stru_extract_loss + noised_stru_extract_loss + denoised_stru_extract_loss) / 6

        recovered_tensor_2 = extractor(recovered_structure_2)
        recovered_tensor_3 = extractor(recovered_structure_3)
        noised_recovered_tensor_2 = extractor(noised_recovered_structure_2)
        noised_recovered_tensor_3 = extractor(noised_recovered_structure_3)
        denoised_recovered_tensor_2 = extractor(denoised_recovered_structure_2)
        denoised_recovered_tensor_3 = extractor(denoised_recovered_structure_3)

        original_mess_extract_loss = F.l1_loss(recovered_tensor_2, secret_tensor) + \
                                     F.l1_loss(recovered_tensor_3, secret_tensor)
        noised_mess_extract_loss = F.l1_loss(noised_recovered_tensor_2, secret_tensor) + \
                                   F.l1_loss(noised_recovered_tensor_3, secret_tensor)
        denoised_mess_extract_loss = F.l1_loss(denoised_recovered_tensor_2, secret_tensor) + \
                                     F.l1_loss(denoised_recovered_tensor_3, secret_tensor)

        Ex_rec_loss = (original_mess_extract_loss + noised_mess_extract_loss + denoised_mess_extract_loss)

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
                fake_img3 = generator_ema(structure2, texture2)

                with torch.no_grad():
                    # Add Gaussian Noise
                    random_sigma = torch.rand(1)[0] * 0.2
                    noised_fake_img3 = fake_img3 + torch.randn(size=fake_img3.shape).cuda() * (random_sigma)
                    noised_fake_img3 = noised_fake_img3.clamp(-1, 1)
                    denoised_fake_img3 = dae(noised_fake_img3)

                # Re-Extract
                original_recovered_structure, _ = encoder_ema(fake_img3)
                noised_recovered_structure, _ = encoder_ema(noised_fake_img3)
                denoised_recovered_structure, _ = encoder_ema(denoised_fake_img3)

                original_recovered_tensor = extractor_ema(original_recovered_structure)
                noised_recovered_tensor = extractor_ema(noised_recovered_structure)
                denoised_recovered_tensor = extractor_ema(denoised_recovered_structure)

                original_extract_loss = F.l1_loss(original_recovered_tensor, secret_tensor)
                noised_extract_loss = torch.mean(torch.abs(noised_recovered_tensor - secret_tensor))
                denoised_extract_loss = torch.mean(torch.abs(denoised_recovered_tensor - secret_tensor))

                original_recovered_tensor = original_recovered_tensor.reshape(
                    shape=(structure1.shape[0], args.N * structure1.shape[2] * structure1.shape[3]))
                original_recovered_message = tensor_to_message(original_recovered_tensor, sigma=1).cuda()

                noised_recovered_tensor = noised_recovered_tensor.reshape(
                    shape=(structure1.shape[0], args.N * structure1.shape[2] * structure1.shape[3]))
                noised_recovered_message = tensor_to_message(noised_recovered_tensor, sigma=1).cuda()

                denoised_recovered_tensor = denoised_recovered_tensor.reshape(
                    shape=(structure1.shape[0], args.N * structure1.shape[2] * structure1.shape[3]))
                denoised_recovered_message = tensor_to_message(denoised_recovered_tensor, sigma=1).cuda()

                original_BER = torch.mean(torch.abs(message - original_recovered_message))
                noised_BER = torch.mean(torch.abs(message - noised_recovered_message))
                denoised_BER = torch.mean(torch.abs(message - denoised_recovered_message))

                original_ACC = 1 - original_BER
                noised_ACC = 1 - noised_BER
                denoised_ACC = 1 - denoised_BER

                print(f'[Testing {iter_idx:06d}/{args.num_iters:06d}] sigma=1 delta=50% '
                      f'ACC of Original Image: {original_ACC:.4f}; L1 loss of tensor: {original_extract_loss:.4f}'
                      f'ACC of   Noised Image: {noised_ACC:.4f}; L1 loss of tensor: {noised_extract_loss:.4f}'
                      f'ACC of Denoised Image: {denoised_ACC:.4f}; L1 loss of tensor: {denoised_extract_loss:.4f}')

                sample = torch.cat((fake_img3, noised_fake_img3, denoised_fake_img3), dim=3)

                utils.save_image(
                    sample,
                    f"{sample_dir}/{iter_idx:06d}.png",
                    nrow=int(args.batch_size),
                    normalize=True,
                    range=(-1, 1),
                )

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
                    "dae": dae.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "ex_optim": ex_optim.state_dict(),
                    "d_optim": d_optim.state_dict(),
                    "args": args,
                },
                f"{ckpt_dir}/{idx}.pt",
            )


if __name__ == "__main__":
    device = "cuda"

    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    # Arg parameters

    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--ideas_ckpt", type=str, default=800000)
    parser.add_argument("--dae_ckpt", type=str, default=100000)

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_type", choices=['offical_lmdb', 'resized_lmdb', 'normal'])
    parser.add_argument("--num_iters", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--show_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=100000)

    args = parser.parse_args()

    # Make Dirs

    base_dir = f"experiments/{args.exp_name}/Robust_IDEAS"
    ckpt_dir = f"{base_dir}/checkpoints"
    sample_dir = f"{base_dir}/samples"

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # Load ckpt

    print("load model:", args.ideas_ckpt)

    ideas_ckpt = torch.load(f"experiments/{args.exp_name}/checkpoints/{args.ideas_ckpt}.pt",
                            map_location=lambda storage, loc: storage)

    dae_ckpt = torch.load(f"experiments/{args.exp_name}/DAE/checkpoints/{args.dae_ckpt}.pt",
                          map_location=lambda storage, loc: storage)

    # Load arg parameters in trained IDEAS
    ideas_args = ideas_ckpt['args']

    args.start_iter = 0

    args.N = ideas_args.N
    args.lr = ideas_args.lr

    args.image_size = ideas_args.image_size

    args.r1 = ideas_args.r1
    args.cooccur_r1 = ideas_args.cooccur_r1
    args.dist_r1 = ideas_args.dist_r1

    args.lambda_G = ideas_args.lambda_G
    args.lambda_E = ideas_args.lambda_E
    args.lambda_REC = ideas_args.lambda_REC

    args.ref_crop = ideas_args.ref_crop
    args.n_crop = ideas_args.n_crop
    args.d_reg_every = ideas_args.d_reg_every
    args.channel = ideas_args.channel
    args.channel_multiplier = ideas_args.channel_multiplier

    # Initialize Networks

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

    dae = DenoisingAutoencoder(args.channel).to(device)
    dae.load_state_dict(dae_ckpt["dae"])
    dae.eval()

    encoder.load_state_dict(ideas_ckpt["encoder"])
    generator.load_state_dict(ideas_ckpt["generator"])
    stru_generator.load_state_dict(ideas_ckpt["stru_generator"])
    extractor.load_state_dict(ideas_ckpt["extractor"])

    discriminator.load_state_dict(ideas_ckpt["discriminator"])
    cooccur_discriminator.load_state_dict(ideas_ckpt["cooccur_discriminator"])
    distribution_discriminator.load_state_dict(ideas_ckpt["distribution_discriminator"])

    encoder_ema.load_state_dict(ideas_ckpt["encoder_ema"])
    generator_ema.load_state_dict(ideas_ckpt["generator_ema"])
    stru_generator_ema.load_state_dict(ideas_ckpt["stru_generator_ema"])
    extractor_ema.load_state_dict(ideas_ckpt["extractor_ema"])

    encoder_ema.eval()
    generator_ema.eval()
    stru_generator_ema.eval()
    extractor_ema.eval()

    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        list(generator.parameters()) + list(encoder.parameters()) +
        list(stru_generator.parameters()),
        lr=args.lr,
        betas=(0, 0.99),
    )
    ex_optim = optim.Adam(
        extractor.parameters(),
        lr=args.lr,
        betas=(0, 0.99),
    )
    d_optim = optim.Adam(
        list(discriminator.parameters()) + list(cooccur_discriminator.parameters()) + list(
            distribution_discriminator.parameters()),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    g_optim.load_state_dict(ideas_ckpt["g_optim"])
    ex_optim.load_state_dict(ideas_ckpt["ex_optim"])
    d_optim.load_state_dict(ideas_ckpt["d_optim"])

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
        dae,
        device
    )
