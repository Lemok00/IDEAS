# IDEAS - Official PyTorch implementation

![result 1](imgs/result_1.png "The synthesised images of IDEAS.")

**Image Disentanglement Autoencoder for Steganography without Embedding**

Xiyao Liu, Ziping Ma, Junxing Ma, Jian Zhang, Gerald Schaefer, Hui Fang

This repo is the official implementation of "Image Disentanglement Autoencoder for Steganography without Embedding"

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

|         | LSUN Bedroom   | LSUN Church    | FFHQ           |
|---------|----------------|----------------|----------------|
| Δ = 0%  | 100% / 16.88   | 100% / 15.90   | 100% / 32.88   |
| Δ = 25% | 100% / 15.56   | 100% / 15.50   | 100% / 31.10   |
| Δ = 50% | 99.54% / 13.39 | 99.55% / 14.48 | 99.49% / 29.31 |

IDEAS N = 2, σ = 1

|         | LSUN Bedroom   | LSUN Church    | FFHQ           |
|---------|----------------|----------------|----------------|
| Δ = 0%  | 100% / 14.17   | 100% / 17.15   | 100% / 29.76   |
| Δ = 25% | 100% / 14.01   | 100% / 16.32   | 100% / 29.02   |
| Δ = 50% | 99.32% / 13.51 | 99.29% / 16.34 | 99.42% / 28.45 |

Main results of IDEAS in terms of extraction accuracy (values on the left) and FID scores (value on the right).

## Requirements