import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

def pad_edges(x, pads):
    (pad_u, pad_d), (pad_l, pad_r) = pads
    x = torch.cat([x[:, :, :pad_u, :], x], dim=2)
    x = torch.cat([x, x[:, :, -pad_d:, :]], dim=2)
    x = torch.cat([x[:, :, :, :pad_l], x], dim=3)
    x = torch.cat([x, x[:, :, :, -pad_r:]], dim=3)
    return x

def pad_min(x, pads):
    min_value = torch.min(x)
    paddings = [(0, 0), *pads, (0, 0)]
    x = F.pad(x, paddings, value=min_value)
    return x

class MaxProjection(nn.Module):
    def forward(self, inputs):
        return torch.max(inputs, dim=3)[0]

class AvgProjection(nn.Module):
    def forward(self, inputs):
        return torch.mean(inputs, dim=3)

class Simplex(nn.Module):
    def forward(self, x):
        x = torch.clamp(x, 0, float('inf'))
        x = x / torch.sum(x, dim=3, keepdim=True)
        return x

class ClipValuesBetween(nn.Module):
    def __init__(self, min_value=0, max_value=1):
        super(ClipValuesBetween, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        return torch.clamp(x, self.min_value, self.max_value)

class IdLifting(nn.Module):
    def __init__(self, n_scales):
        super(IdLifting, self).__init__()
        self.n_scales = n_scales

    def forward(self, x):
        x = x.unsqueeze(3)
        return x.repeat(1, 1, 1, self.n_scales, 1)

class ScaleGaussian(nn.Module):
    def __init__(self, zero_scale=1.0, max_width=5, strides=(1, 1), base=2, start_at_one=False, trainable_params=False):
        super(ScaleGaussian, self).__init__()
        self.zero_scale = zero_scale
        self.max_width = max_width
        self.strides = strides
        self.base = base
        self.start_at_one = start_at_one
        self.trainable_params = trainable_params

    def build(self, input_shape):
        self.n_scales = input_shape[-2]
        if self.start_at_one:
            self.scales = [float((self.base ** i) - 1) * self.zero_scale * 2 for i in range(self.n_scales)]
        else:
            self.scales = [float((self.base ** i) - 1) * self.zero_scale * 2 for i in range(1, self.n_scales + 1)]

        vmin = ((2 * 1.645 * self.zero_scale) ** 2) / self.max_width ** 2
        vmax = ((2 * 1.645 * self.zero_scale) ** 2)

        self.coef = nn.Parameter(torch.ones(1, input_shape[-1]), requires_grad=self.trainable_params)
        self.coef = ClipValuesBetween(min_value=vmin, max_value=vmax)(self.coef)

        self.widths = [int(self.max_width * scale) for scale in self.scales]
        filters = [torch.linspace(-w, w, 2 * w + 1).float() for w in self.widths]
        self.filters = [-(k[:, None] ** 2) / (scale ** 2) for k, scale in zip(filters, self.scales)]

    def forward(self, x):
        out = []
        for i in range(self.n_scales):
            if i == 0 and self.start_at_one:
                out.append(x[..., 0, :])
                continue
            k = torch.exp(self.filters[i] * self.coef)
            k = k / torch.sum(k, dim=0, keepdim=True)
            kx = k[:, None, :, None]
            ky = k[None, :, :, None]
            x0 = x[..., i, :]
            s = self.widths[i]
            pad = [s, s]
            pads = [pad, pad]
            x0 = pad_edges(x0, pads)

            x0 = F.conv2d(x0, kx, stride=(1, self.strides[0]), padding=0)
            x0 = F.conv2d(x0, ky, stride=(self.strides[1], 1), padding=0)
            out.append(x0)

        return torch.stack(out, dim=-2)

if __name__ == '__main__':
    n_scales = 5

    im = imread('poivrons.png')
    im = im.astype(np.float32) / 255
    im = im[np.newaxis, ...]

    inputs = torch.tensor(im)
    x = IdLifting(n_scales)(inputs)
    x = ScaleGaussian(0.5)(x)
    model1 = nn.Sequential(IdLifting(n_scales), ScaleGaussian(0.5))

    inputs = torch.tensor(im)
    x = IdLifting(n_scales)(inputs)
    x = ScaleQuadraticDilation(0.5)(x)
    model2 = nn.Sequential(IdLifting(n_scales), ScaleQuadraticDilation(0.5))

    im_g = model1(im)[0, ...]
    im_d = model2(im)[0, ...]

    fig = plt.figure()
    for i in range(n_scales):
        plt.subplot(2, n_scales, i + 1)
        plt.imshow(im_g[:, :, i, :])
        plt.title(f'gaussian {i}')
    for i in range(n_scales):
        plt.subplot(2, n_scales, n_scales + i + 1)
        plt.imshow(im_d[:, :, i, :])
        plt.title(f'dilation {i}')
    plt.show()
