# IDEAS - Official PyTorch implementation

![result 1](imgs/result_1.png "The synthesised images of IDEAS.")

**Image Disentanglement Autoencoder for Steganography without Embedding**

Xiyao Liu, Ziping Ma, Junxing Ma, Jian Zhang, Gerald Schaefer, Hui Fang

This repo is the official implementation of "Image Disentanglement Autoencoder for Steganography without Embedding"

#Abstract
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

#Main Results of IDEAS

<img src="http://latex.codecogs.com/svg.latex?N=1,&space;\sigma&space;=&space;1" title="http://latex.codecogs.com/svg.latex?N=1, \sigma = 1" />

$$\sigma = 1$$

|                | LSUN Bedroom | LSUN Church | FFHQ  |
|----------------| ---- | ---- |-------|
|  | 94.01%/283.32 |

