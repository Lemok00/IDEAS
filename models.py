import math

import torch
from torch import nn
from torch.nn import functional as F

from stylegan2.model import StyledConv_without_noise as StyledConv, Blur, EqualLinear, EqualConv2d, ScaledLeakyReLU
from stylegan2.op import FusedLeakyReLU


class EqualConvTranspose2d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv_transpose2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[0]}, {self.weight.shape[1]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class ConvLayer(nn.Sequential):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            upsample=False,
            downsample=False,
            blur_kernel=(1, 3, 3, 1),
            bias=True,
            activate=True,
            padding="zero",
            tanh=False
    ):
        layers = []

        self.padding = 0
        stride = 1

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2

        if upsample:
            layers.append(
                EqualConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=0,
                    stride=2,
                    bias=bias and not activate,
                )
            )

            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

        else:
            if not downsample:
                if padding == "zero":
                    self.padding = (kernel_size - 1) // 2

                elif padding == "reflect":
                    padding = (kernel_size - 1) // 2

                    if padding > 0:
                        layers.append(nn.ReflectionPad2d(padding))

                    self.padding = 0

                elif padding != "valid":
                    raise ValueError('Padding should be "zero", "reflect", or "valid"')

            layers.append(
                EqualConv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    bias=bias and not activate,
                )
            )

        if activate:
            if tanh:
                layers.append(nn.Tanh())
            else:
                if bias:
                    layers.append(FusedLeakyReLU(out_channel))

                else:
                    layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class StyledResBlock(nn.Module):
    def __init__(
            self, in_channel, out_channel, style_dim, upsample, blur_kernel=(1, 3, 3, 1)
    ):
        super().__init__()

        self.conv1 = StyledConv(
            in_channel,
            out_channel,
            3,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
        )

        self.conv2 = StyledConv(out_channel, out_channel, 3, style_dim)

        if upsample or in_channel != out_channel:
            self.skip = ConvLayer(
                in_channel,
                out_channel,
                1,
                upsample=upsample,
                blur_kernel=blur_kernel,
                bias=False,
                activate=False,
            )

        else:
            self.skip = None

    def forward(self, input, style, noise=None):
        out = self.conv1(input, style, noise)
        out = self.conv2(out, style, noise)

        if self.skip is not None:
            skip = self.skip(input)

        else:
            skip = input

        return (out + skip) / math.sqrt(2)


class ResBlock(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            downsample,
            padding="zero",
            blur_kernel=(1, 3, 3, 1),
    ):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, out_channel, 3, padding=padding)

        self.conv2 = ConvLayer(
            out_channel,
            out_channel,
            3,
            downsample=downsample,
            padding=padding,
            blur_kernel=blur_kernel,
        )

        if downsample or in_channel != out_channel:
            self.skip = ConvLayer(
                in_channel,
                out_channel,
                1,
                downsample=downsample,
                blur_kernel=blur_kernel,
                bias=False,
                activate=False,
            )

        else:
            self.skip = None

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        if self.skip is not None:
            skip = self.skip(input)

        else:
            skip = input

        return (out + skip) / math.sqrt(2)


class DisentanglementEncoder(nn.Module):
    def __init__(
            self,
            channel,
            structure_channel=8,
            texture_channel=2048,
            blur_kernel=(1, 3, 3, 1),
    ):
        super().__init__()

        stem = [ConvLayer(3, channel, 1)]

        in_channel = channel
        for i in range(1, 5):
            ch = channel * (2 ** i)
            stem.append(ResBlock(in_channel, ch, downsample=True, padding="reflect", blur_kernel=blur_kernel))
            in_channel = ch

        self.stem = nn.Sequential(*stem)

        self.structure = nn.Sequential(
            ConvLayer(in_channel, in_channel, 1, blur_kernel=blur_kernel),
            ConvLayer(in_channel, structure_channel, 1, blur_kernel=blur_kernel)
        )

        self.texture = nn.Sequential(
            ConvLayer(in_channel, in_channel * 2, 3, downsample=True, padding="valid", blur_kernel=blur_kernel),
            ConvLayer(in_channel * 2, in_channel * 4, 3, downsample=True, padding="valid", blur_kernel=blur_kernel),
            nn.AdaptiveAvgPool2d(1),
            ConvLayer(in_channel * 4, texture_channel, 1, tanh=True, blur_kernel=blur_kernel),
        )

    def forward(self, input):
        out = self.stem(input)

        structure = self.structure(out)
        texture = torch.flatten(self.texture(out), 1)

        return structure, texture


class Generator(nn.Module):
    def __init__(
            self,
            channel,
            structure_channel=8,
            texture_channel=2048,
            blur_kernel=(1, 3, 3, 1),
    ):
        super().__init__()

        ch_multiplier = (4, 8, 12, 16, 16, 16, 8, 4)
        upsample = (False, False, False, False, True, True, True, True)

        self.layers = nn.ModuleList()
        in_ch = structure_channel
        for ch_mul, up in zip(ch_multiplier, upsample):
            self.layers.append(
                StyledResBlock(
                    in_ch, channel * ch_mul, texture_channel, up, blur_kernel
                )
            )
            in_ch = channel * ch_mul

        self.to_rgb = ConvLayer(in_ch, 3, 1, activate=False)

    def forward(self, structure, texture, noises=None):
        if noises is None:
            noises = [None] * len(self.layers)

        out = structure
        for layer, noise in zip(self.layers, noises):
            out = layer(out, texture, noise)

        out = self.to_rgb(out)

        return out


class StructureGenerator(nn.Module):
    def __init__(
            self,
            channel,
            N=1,
            structure_channel=8,
            blur_kernel=(1, 3, 3, 1),
    ):
        super().__init__()

        stem = [ConvLayer(N, channel, 1, blur_kernel=blur_kernel)]
        stem.append(ResBlock(channel, channel * 2, downsample=False, padding="reflect", blur_kernel=blur_kernel))
        stem.append(ResBlock(channel * 2, channel * 4, downsample=False, padding="reflect", blur_kernel=blur_kernel))
        stem.append(ResBlock(channel * 4, channel * 2, downsample=False, padding="reflect", blur_kernel=blur_kernel))
        stem.append(ConvLayer(channel * 2, structure_channel, 1, blur_kernel=blur_kernel))

        self.structure = nn.Sequential(*stem)

    def forward(self, noise):
        out = self.structure(noise)
        return out


class ImageLevelDiscriminator(nn.Module):
    def __init__(self, size, channel_multiplier=1, blur_kernel=(1, 3, 3, 1)):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1, blur_kernel=blur_kernel)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, downsample=True, blur_kernel=blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.final_conv = ConvLayer(in_channel, channels[4], 3, blur_kernel=blur_kernel)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)
        out = self.final_conv(out)

        out = out.view(out.shape[0], -1)
        out = self.final_linear(out)

        return out


class CooccurenceDiscriminator(nn.Module):
    def __init__(self, channel, size=256):
        super().__init__()

        encoder = [ConvLayer(3, channel, 1)]

        ch_multiplier = (2, 4, 8, 12, 12, 24)
        downsample = (True, True, True, True, True, False)
        in_ch = channel
        for ch_mul, down in zip(ch_multiplier, downsample):
            encoder.append(ResBlock(in_ch, channel * ch_mul, down))
            in_ch = channel * ch_mul

        if size > 511:
            k_size = 3
            feat_size = 2 * 2

        else:
            k_size = 2
            feat_size = 1 * 1

        encoder.append(ConvLayer(in_ch, channel * 12, k_size, padding="valid"))

        self.encoder = nn.Sequential(*encoder)

        self.linear = nn.Sequential(
            EqualLinear(
                channel * 12 * 2 * feat_size, channel * 32, activation="fused_lrelu"
            ),
            EqualLinear(channel * 32, channel * 32, activation="fused_lrelu"),
            EqualLinear(channel * 32, channel * 16, activation="fused_lrelu"),
            EqualLinear(channel * 16, 1),
        )

    def forward(self, input, reference=None, ref_batch=None, ref_input=None):
        out_input = self.encoder(input)

        if ref_input is None:
            ref_input = self.encoder(reference)
            _, channel, height, width = ref_input.shape
            ref_input = ref_input.view(-1, ref_batch, channel, height, width)
            ref_input = ref_input.mean(1)

        out = torch.cat((out_input, ref_input), 1)
        out = torch.flatten(out, 1)
        out = self.linear(out)

        return out, ref_input


class DistributionDiscriminator(nn.Module):
    def __init__(self, texture_channel=2048):
        super().__init__()
        self.model = nn.Sequential(
            EqualLinear(texture_channel, texture_channel // 4, activation="fused_lrelu"),
            EqualLinear(texture_channel // 4, texture_channel // 16, activation="fused_lrelu"),
            EqualLinear(texture_channel // 16, texture_channel // 64, activation="fused_lrelu"),
            EqualLinear(texture_channel // 64, 1, activation="fused_lrelu")
        )

    def forward(self, input):
        out = self.model(input)
        return out


class TensorExtractor(nn.Module):
    def __init__(
            self,
            channel,
            N=1,
            structure_channel=8,
            blur_kernel=(1, 3, 3, 1),
    ):
        super().__init__()

        stem = [ConvLayer(structure_channel, channel * 2, 1, blur_kernel=blur_kernel)]
        stem.append(ResBlock(channel * 2, channel * 4, downsample=False, padding="reflect", blur_kernel=blur_kernel))
        stem.append(ResBlock(channel * 4, channel * 2, downsample=False, padding="reflect", blur_kernel=blur_kernel))
        stem.append(ResBlock(channel * 2, channel, downsample=False, padding="reflect", blur_kernel=blur_kernel))
        stem.append(ConvLayer(channel, N, 1, blur_kernel=blur_kernel))

        self.extract = nn.Sequential(*stem)

    def forward(self, input):
        out = self.extract(input)

        return out


def init_model(model, args):
    if model == 'DisentanglementEncoder':
        return DisentanglementEncoder(
            channel=args.channel,
            structure_channel=args.structure_channel,
            texture_channel=args.texture_channel,
            blur_kernel=args.blur_kernel
        )
    elif model == 'Generator':
        return Generator(
            channel=args.channel,
            structure_channel=args.structure_channel,
            texture_channel=args.texture_channel,
            blur_kernel=args.blur_kernel
        )
    elif model == 'StructureGenerator':
        return StructureGenerator(
            channel=args.channel,
            N=args.N,
            structure_channel=args.structure_channel,
            blur_kernel=args.blur_kernel
        )
    elif model == 'ImageLevelDiscriminator':
        return ImageLevelDiscriminator(
            size=args.image_size,
            channel_multiplier=args.channel_multiplier,
            blur_kernel=args.blur_kernel
        )
    elif model == 'CooccurenceDiscriminator':
        return CooccurenceDiscriminator(
            channel=args.channel,
            size=args.image_size
        )
    elif model == 'DistributionDiscriminator':
        return DistributionDiscriminator(
            texture_channel=args.texture_channel
        )
    elif model == 'TensorExtractor':
        return TensorExtractor(
            channel=args.channel,
            N=args.N,
            structure_channel=args.structure_channel,
            blur_kernel=args.blur_kernel
        )
    else:
        raise NotImplementedError
