import argparse
import os
import numpy as np

import torch
from torch import autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils

import time
from utils import time_change
from utils import data_sampler, requires_grad, accumulate, sample_data
from utils import tensor_to_message, message_to_tensor
from utils import d_logistic_loss, d_r1_loss, g_nonsaturating_loss, patchify_image

from models import init_model
from dataset import set_dataset


def train(
        exp_name,
        args,
        loader,
        trainer,
        device
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

        # The reference image X
        X = next(loader)
        X = X.to(device)

        '''
        Training Discriminators
        '''

        requires_grad(trainer['E'], False)
        requires_grad(trainer['G'], False)
        requires_grad(trainer['Gstru'], False)
        requires_grad(trainer['Ex'], False)
        requires_grad(trainer['Dreal'], True)
        requires_grad(trainer['Dco'], True)
        requires_grad(trainer['Ddist'], True)

        # Feature Encoding & Generation
        # Structure S1 and Texture T1
        S1, T1 = trainer['E'](X)
        # Secret tensor Z
        Z = torch.rand(size=(S1.shape[0], args.N, S1.shape[2], S1.shape[3]),
                       dtype=torch.float).cuda() * 2 - 1
        # Structure S2 and Texture T2
        S2 = trainer['Gstru'](Z)
        T2 = torch.rand_like(T1) * 2 - 1

        # Image Synthesis
        # Reconstructed image \hat{X}_1
        hat_X1 = trainer['G'](S1, T1)
        # Synthesised image \hat{X}_2 and \hat{X}_3
        hat_X2 = trainer['G'](S2, T1)
        hat_X3 = trainer['G'](S2, T2)

        # L_{D,real}
        fake_pred = trainer['Dreal'](torch.cat((hat_X1, hat_X2, hat_X3), 0))
        real_pred = trainer['Dreal'](X)

        D_real_loss = d_logistic_loss(real_pred, fake_pred)

        # L_{D,texture}
        fake_patch = patchify_image(hat_X2, args.n_crop)
        real_patch = patchify_image(X, args.n_crop)
        ref_patch = patchify_image(X, args.ref_crop * args.n_crop)

        fake_texture_pred, ref_input = trainer['Dco'](fake_patch, ref_patch, ref_batch=args.ref_crop)
        real_texture_pred, _ = trainer['Dco'](real_patch, ref_input=ref_input)

        D_texture_loss = d_logistic_loss(real_texture_pred, fake_texture_pred)

        # L_{D,distribution}
        fake_dist_pred = trainer['Ddist'](T1)
        real_dist_pred = trainer['Ddist'](T2)

        D_dist_loss = d_logistic_loss(real_dist_pred, fake_dist_pred)

        # Record D Losses and optimise
        loss_dict["D_real_loss"] = D_real_loss
        loss_dict["D_texture_loss"] = D_texture_loss
        loss_dict["D_dist_loss"] = D_dist_loss

        trainer['d_optim'].zero_grad()
        (D_real_loss + D_texture_loss + D_dist_loss).backward()
        trainer['d_optim'].step()

        # Regularization
        if iter_idx % args.d_reg_every == 0:
            X.requires_grad = True
            real_pred = trainer['Dreal'](X)
            D_real_r1_loss = d_r1_loss(real_pred, X)

            real_patch.requires_grad = True
            real_patch_pred, _ = trainer['Dco'](real_patch, ref_patch, ref_batch=args.ref_crop)
            D_texture_r1_loss = d_r1_loss(real_patch_pred, real_patch)

            T2.requires_grad = True
            real_uniform_pred = trainer['Ddist'](T2)
            D_dist_r1_loss = d_r1_loss(real_uniform_pred, T2)

            trainer['d_optim'].zero_grad()

            r1_loss_sum = args.real_r1 / 3 * D_real_r1_loss * args.d_reg_every
            r1_loss_sum += args.texture_r1 / 3 * D_texture_r1_loss * args.d_reg_every
            r1_loss_sum += args.dist_r1 / 3 * D_dist_r1_loss * args.d_reg_every
            r1_loss_sum.backward()

            trainer['d_optim'].step()

            loss_dict["D_real_r1_loss"] = D_real_r1_loss
            loss_dict["D_texture_r1_loss"] = D_texture_r1_loss
            loss_dict["D_dist_r1_loss"] = D_dist_r1_loss

        '''
        Training main components.
        '''

        requires_grad(trainer['E'], True)
        requires_grad(trainer['G'], True)
        requires_grad(trainer['Gstru'], True)
        requires_grad(trainer['Ex'], True)
        requires_grad(trainer['Dreal'], False)
        requires_grad(trainer['Dco'], False)
        requires_grad(trainer['Ddist'], False)

        # Feature Encoding & Generation
        # Structure S1 and Texture T1
        S1, T1 = trainer['E'](X)
        # Secret tensor Z
        Z = torch.rand(size=(S1.shape[0], args.N, S1.shape[2], S1.shape[3]),
                       dtype=torch.float).cuda() * 2 - 1
        # Structure S2 and Texture T2
        S2 = trainer['Gstru'](Z)
        T2 = torch.rand_like(T1) * 2 - 1

        # Image Synthesis
        # Reconstructed image \hat{X}_1
        hat_X1 = trainer['G'](S1, T1)
        # Synthesised image \hat{X}_2 and \hat{X}_3
        hat_X2 = trainer['G'](S2, T1)
        hat_X3 = trainer['G'](S2, T2)

        # L_{G,rec}
        G_rec_loss = F.l1_loss(hat_X1, X)

        # L_{G,real}
        fake_pred = trainer['Dreal'](torch.cat((hat_X1, hat_X2, hat_X3), 0))
        G_real_loss = g_nonsaturating_loss(fake_pred)

        # L_{E,dist}
        fake_dist_pred = trainer['Ddist'](T1)
        E_dist_loss = g_nonsaturating_loss(fake_dist_pred)

        # L_{G,texture}
        fake_patch = patchify_image(hat_X2, args.n_crop)
        ref_patch = patchify_image(X, args.ref_crop * args.n_crop)
        fake_patch_pred, _ = trainer['Dco'](fake_patch, ref_patch, ref_batch=args.ref_crop)
        G_texture_loss = g_nonsaturating_loss(fake_patch_pred)

        # L_{E,stru}
        if iter_idx > args.num_iters * 0.8:
            container_image = hat_X3
        else:
            container_image = hat_X2
        # The recovered structure \hat{S}_2
        hat_S2, _ = trainer['E'](container_image)
        E_stru_loss = F.l1_loss(hat_S2, S2)

        # L_{REC}
        # The extracted secret tensor \hat{Z}
        hat_Z = trainer['Ex'](hat_S2)
        Ex_loss = F.l1_loss(hat_Z, Z)

        # Record Loss
        loss_dict["G_rec_loss"] = G_rec_loss
        loss_dict["G_real_loss"] = G_real_loss
        loss_dict["G_texture_loss"] = G_texture_loss
        loss_dict["E_dist_loss"] = E_dist_loss
        loss_dict["E_stru_loss"] = E_stru_loss
        loss_dict["Ex_loss"] = Ex_loss

        # L_G
        Loss_G = G_rec_loss + G_texture_loss + 2 * G_real_loss
        # L_E
        Loss_E = E_dist_loss + E_stru_loss
        # L_Ex
        Loss_Ex = Ex_loss
        # L_{total}
        Loss_total = Loss_G + Loss_E + args.lambda_Ex * Loss_Ex

        # optimize Encoder and Generator
        trainer['g_optim'].zero_grad()
        Loss_total.backward(retain_graph=True)
        trainer['g_optim'].step()

        # optimize Extractor
        trainer['ex_optim'].zero_grad()
        Loss_Ex.backward()
        trainer['ex_optim'].step()

        accumulate(trainer['E_ema'], trainer['E'], accum)
        accumulate(trainer['G_ema'], trainer['G'], accum)
        accumulate(trainer['Gstru_ema'], trainer['Gstru'], accum)
        accumulate(trainer['Ex_ema'], trainer['Ex'], accum)

        # Log
        if iter_idx % args.log_every == 0:
            G_rec_val = loss_dict["G_rec_loss"].mean().item()
            G_texture_val = loss_dict["G_texture_loss"].mean().item()
            G_real_val = loss_dict["G_real_loss"].mean().item()

            E_dist_val = loss_dict["E_dist_loss"].mean().item()
            E_stru_val = loss_dict["E_stru_loss"].mean().item()

            Ex_val = loss_dict["Ex_loss"].mean().item()

            now_time = time.time()
            used_time = now_time - start_time
            rest_time = (now_time - start_time) / idx * (args.num_iters - iter_idx)

            log_output = f"[{iter_idx:07d}/{args.num_iters:07}] Total: {Loss_total.item():.4f}; " \
                         f"G,rec: {G_rec_val:.4f}; G,texture: {G_texture_val:.4f}; G,real: {G_real_val:.4f}; " \
                         f"E,dist: {E_dist_val:.4f}; E,stru: {E_stru_val:.4f}; Ex: {Ex_val:.4f} " \
                         f"used time: {time_change(used_time)};" \
                         f"rest time: {time_change(rest_time)}"

            print(log_output, flush=True)
            with open(f'{base_dir}/training_logs.txt', 'a') as fp:
                fp.write(f'{log_output}\n')

        # Output Samples
        if iter_idx % args.show_every == 0:
            with torch.no_grad():
                # Sample a secret message and map it to secret tensor
                S1, T1 = trainer['E_ema'](X)
                # The secret message M
                M = torch.randint(low=0, high=2, dtype=torch.float,
                                  size=(S1.shape[0], args.N * S1.shape[2] * S1.shape[3]))
                Z = message_to_tensor(M, sigma=1, delta=0.5).to(device)
                Z = Z.reshape(shape=(S1.shape[0], args.N, S1.shape[2], S1.shape[3]))

                # Generate structure S2 from the secret tensor
                S2 = trainer['Gstru_ema'](Z)

                # Sample a texture T2
                T2 = torch.rand_like(T1) * 2 - 1

                # Image Synthesis
                hat_X1 = trainer['G_ema'](S1, T1)
                hat_X2 = trainer['G_ema'](S2, T1)
                hat_X3 = trainer['G_ema'](S2, T2)

                # Secret Tensor Extracting
                if iter_idx > args.num_iters * 0.8:
                    container_image = hat_X3
                    fake_img_used_as_container = 3
                else:
                    container_image = hat_X2
                    fake_img_used_as_container = 2
                hat_S2, _ = trainer['E_ema'](container_image)
                hat_Z = trainer['Ex_ema'](hat_S2)

                tensor_recovering_loss = torch.mean(torch.abs(hat_Z - Z))
                hat_Z = hat_Z.reshape(shape=(Z.shape[0], -1))
                # The extracted secret message \hat_{M}
                hat_M = tensor_to_message(hat_Z, sigma=1)

                BER = torch.mean(torch.abs(M - hat_M))
                ACC = 1 - BER

                log_output = f'[Testing {iter_idx:07d}/{args.num_iters:07d}] sigma=1 delta=50% ' \
                             f'using synthesised image \hatX_{fake_img_used_as_container} ' \
                             f'ACC of Msg: {ACC:.4f}; L1 loss of tensor: {tensor_recovering_loss:.4f}'
                print(log_output, flush=True)
                with open(f'{base_dir}/training_logs.txt', 'a') as fp:
                    fp.write(f'{log_output}\n')

                sample = torch.cat((X, hat_X1, hat_X2, hat_X3), 0)

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
            trainer_ckpt = {}
            for key in trainer.keys():
                trainer_ckpt[key] = trainer[key].state_dict()
            torch.save(
                {
                    'iter_idx': iter_idx,
                    'N': args.N,
                    "trainer": trainer_ckpt,
                    "args": args,
                },
                f"{ckpt_dir}/{iter_idx}.pt",
            )

            print(f'Checkpoint is saved in experiments/{exp_name}/checkpoints')


if __name__ == "__main__":
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()

    # Working directory: experiments/exp_name
    parser.add_argument("--exp_name", type=str, required=True)
    # Training dataset
    parser.add_argument("--dataset_path", type=str, required=True)
    # Select 'lmdb' for the lmdb files, like LSUN (https://github.com/fyu/lsun)
    # Select 'normal' for the dataset storing files (e.g., in PNG format) in a folder, like FFHQ (https://github.com/NVlabs/ffhq-dataset)
    parser.add_argument("--dataset_type", choices=['lmdb', 'normal'], required=True)

    # We recommend training at least 80k iterations
    parser.add_argument("--num_iters", type=int, required=True)
    # Hyper-parameters
    parser.add_argument("--N", type=int, default=1)
    parser.add_argument("--lambda_Ex", type=float, default=10)
    # Resume training
    parser.add_argument("--ckpt", type=str, default=None)

    # Trainig parameters
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--real_r1", type=float, default=10)
    parser.add_argument("--texture_r1", type=float, default=1)
    parser.add_argument("--dist_r1", type=float, default=1)
    parser.add_argument("--ref_crop", type=int, default=4)
    parser.add_argument("--n_crop", type=int, default=8)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--channel", type=int, default=32)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--structure_channel", type=int, default=8)
    parser.add_argument("--texture_channel", type=int, default=2048)

    # Output logs every 'log_every' iterations
    parser.add_argument("--log_every", type=int, default=200)
    # Save example images every 'show_every' iterations
    parser.add_argument("--show_every", type=int, default=1000)
    # Save models every 'save_every' iterations
    parser.add_argument("--save_every", type=int, default=200000)

    args = parser.parse_args()
    args.start_iter = 0
    args.blur_kernel = (1, 3, 3, 1)

    # Create folders for working directory
    base_dir = f"experiments/{args.exp_name}"
    ckpt_dir = f"{base_dir}/checkpoints"
    sample_dir = f"{base_dir}/samples"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # Log training configurations
    with open(f"{base_dir}/training_config.txt", "wt") as fp:
        for k, v in vars(args).items():
            fp.write(f'{k}: {v}\n')
        fp.close()

    # Clear training logs
    with open(f"{base_dir}/training_logs.txt", "wt") as fp:
        fp.close()

    # Init models
    trainer = {
        'E': init_model('DisentanglementEncoder', args).to(device),
        'G': init_model('Generator', args).to(device),
        'Gstru': init_model('StructureGenerator', args).to(device),
        'Ex': init_model('TensorExtractor', args).to(device),

        'Dreal': init_model('ImageLevelDiscriminator', args).to(device),
        'Dco': init_model('CooccurenceDiscriminator', args).to(device),
        'Ddist': init_model('DistributionDiscriminator', args).to(device),

        'E_ema': init_model('DisentanglementEncoder', args).to(device),
        'G_ema': init_model('Generator', args).to(device),
        'Gstru_ema': init_model('StructureGenerator', args).to(device),
        'Ex_ema': init_model('TensorExtractor', args).to(device),
    }

    trainer['E_ema'].eval()
    trainer['G_ema'].eval()
    trainer['Gstru_ema'].eval()
    trainer['Ex_ema'].eval()

    accumulate(trainer['E_ema'], trainer['E'], 0)
    accumulate(trainer['G_ema'], trainer['G'], 0)
    accumulate(trainer['Gstru_ema'], trainer['Gstru'], 0)
    accumulate(trainer['Ex_ema'], trainer['Ex'], 0)

    # Init optimizers
    trainer['g_optim'] = optim.Adam(
        list(trainer['E'].parameters()) + list(trainer['G'].parameters()) + list(trainer['Gstru'].parameters()),
        lr=args.lr,
        betas=(0, 0.99),
    )
    trainer['ex_optim'] = optim.Adam(
        trainer['Ex'].parameters(),
        lr=args.lr,
        betas=(0, 0.99),
    )
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    trainer['d_optim'] = optim.Adam(
        list(trainer['Dreal'].parameters()) + list(trainer['Dco'].parameters()) + list(trainer['Ddist'].parameters()),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    # Resume training from the 'experiments/exp_name/checkpoints/{ckpt}.pt' file
    if args.ckpt is not None:
        print("load model:", args.ckpt)
        ckpt = torch.load(f"{ckpt_dir}/{args.ckpt}.pt", map_location=lambda storage, loc: storage)
        args.start_iter = ckpt['iter_idx']
        for key in trainer.keys():
            trainer[key].load_state_dict(ckpt['trainer'][key])
    else:
        args.start_iter = 0

    # Init transforms
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    # Init dataset and dataloader
    dataset = set_dataset(
        type=args.dataset_type,
        path=args.dataset_path,
        transform=transform,
        resolution=args.image_size
    )
    loader = data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        sampler=data_sampler(dataset=dataset, shuffle=True)
    )

    print('Data Loaded')

    exp_name = args.exp_name

    train(
        exp_name=exp_name,
        args=args,
        loader=loader,
        trainer=trainer,
        device=device
    )
