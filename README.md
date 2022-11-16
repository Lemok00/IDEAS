# IDEAS - Official PyTorch implementation

![result 1](imgs/result_1.png "The synthesised images of IDEAS.")

**Image Disentanglement Autoencoder for Steganography without Embedding**

[Xiyao Liu](https://faculty.csu.edu.cn/liuxiyao/en/index.htm), [Ziping Ma](https://lemok00.github.io), Junxing Ma, Jian Zhang, [Gerald Schaefer](https://www.lboro.ac.uk/departments/compsci/staff/gerald-schaefer/), [Hui Fang](https://www.lboro.ac.uk/departments/compsci/staff/hui-fang/)

This repo is the official implementation of "[Image Disentanglement Autoencoder for Steganography without Embedding](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Image_Disentanglement_Autoencoder_for_Steganography_Without_Embedding_CVPR_2022_paper.html)"

## Abstract
> Conventional steganography approaches embed a secret
message into a carrier for concealed communication but
are prone to attack by recent advanced steganalysis tools.
In this paper, we propose Image DisEntanglement Autoencoder
for Steganography (IDEAS) as a novel steganography
without embedding (SWE) technique. Instead of directly
embedding the secret message into a carrier image, our approach
hides it by transforming it into a synthesised image,
and is thus fundamentally immune to any steganalysis attack.
By disentangling an image into two representations
for structure and texture, we exploit the stability of structure
representation to improve secret message extraction while
increasing synthesis diversity via randomising texture representations
to enhance steganography security. In addition,
we design an adaptive mapping mechanism to further
enhance the diversity of synthesised images when ensuring
different required extraction levels. Experimental results
convincingly demonstrate IDEAS to achieve superior
performance in terms of enhanced security, reliable secret
message extraction and flexible adaptation for different extraction
levels, compared to state-of-the-art SWE methods.

## Main Results of IDEAS

IDEAS N = 1, σ = 1

|         |  LSUN Bedroom  |  LSUN Church   |      FFHQ      |
|---------|:--------------:|:--------------:|:--------------:|
| Δ = 0%  |  100% / 16.88  |  100% / 15.90  |  100% / 32.88  |
| Δ = 25% |  100% / 15.56  |  100% / 15.50  |  100% / 31.10  |
| Δ = 50% | 99.54% / 13.39 | 99.55% / 14.48 | 99.49% / 29.31 |

IDEAS N = 2, σ = 1

|         |  LSUN Bedroom  |  LSUN Church   |      FFHQ      |
|---------|:--------------:|:--------------:|:--------------:|
| Δ = 0%  |  100% / 14.17  |  100% / 17.15  |  100% / 29.76  |
| Δ = 25% |  100% / 14.01  |  100% / 16.32  |  100% / 29.02  |
| Δ = 50% | 99.32% / 13.51 | 99.29% / 16.34 | 99.42% / 28.45 |

Main results of IDEAS in terms of extraction accuracy (values on the left) and FID scores (value on the right).

## Requirements
* **Only Linux is supported.** 
* Ninja >= 1.10.2, GCC/G++ >= 9.4.0.
* One high-end NVIDIA GPU with at least 11GB of memory. We have done all development and testing using a NVIDIA RTX 2080Ti.
* Python >= 3.7 and PyTorch >= 1.8.2. See [https://pytorch.org/](https://pytorch.org/) for PyTorch install instructions.
* CUDA toolkit 11.2 or later.
* Python libraries: `pip install lmdb imutils opencv-python pandas tqdm`. We use the Anaconda3 2020.11 distribution which installs most of these by default.

## Training
Train a model using the dataset with path of `PATH` and type of `TYPE`.
```shell
python train.py --exp_name NAME --dataset_type TYPE --dataset_path PATH --num_iters ITERS
```    
The training configuration can be customized with command line option:

| args             | Description                                                                                          |
|:-----------------|:-----------------------------------------------------------------------------------------------------|
| `exp_name`       | The working directory `./experiments/{exp_name}`.                                                    |
| `dataset_type`   | The type of dataset. Select `lmdb` for LMDB files, or `normal` for the folder storing files.         |
| `dataset_path`   | The path of dataset.                                                                                 |
| `num_iters`      | Num of training iterations.                                                                          |
| `N`, `lambda_Ex` | The hyper-parameters of IDEAS.                                                                       |
| `ckpt`           | Train from scratch if ignored, else resume training from `./experiments/NAME/checkpoints/{ckpt}.pt`. |
| `log_every`      | Output logs every `log_every` iterations.                                                            |
| `show_every`     | Save example images every `show_every` iterations under `./experiments/NAME/samples/`.               |
| `save_every`     | Save models every `save_every` iterations under `./experiments/NAME/checkpoints/`.                   |


## Citation
```
@inproceedings{liu2022image,
  title={Image Disentanglement Autoencoder for Steganography Without Embedding},
  author={Liu, Xiyao and Ma, Ziping and Ma, Junxing and Zhang, Jian and Schaefer, Gerald and Fang, Hui},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2303--2312},
  year={2022}
}
```