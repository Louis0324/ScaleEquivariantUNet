import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from scale_cross_corr_pytorch import *
from scale_space_layers import *

class MaxProjection(nn.Module):
    def forward(self, inputs):
        return torch.max(inputs, dim=3)[0]

class UpsampleEquivariant(nn.Module):
    def __init__(self, size=(2, 2)):
        super(UpsampleEquivariant, self).__init__()
        self.size = size

    def forward(self, inputs):
        shape = inputs.shape
        out = inputs.view(shape[0], shape[1], shape[2], shape[3] * shape[4])
        h = self.size[0] * shape[1]
        w = self.size[1] * shape[2]
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        return out.view(shape[0], h, w, shape[3], shape[4])

class ScaleConvBlock(nn.Module):
    def __init__(self, filters, n_scales, scale_dim, name, batchnorm_momentum):
        super(ScaleConvBlock, self).__init__()
        self.sconv1 = ScaleConv(filters, (3, 3, 1), n_scales)
        self.bn1 = nn.BatchNorm3d(filters, momentum=batchnorm_momentum)
        self.relu1 = nn.ReLU()
        self.sconv2 = ScaleConv(filters, (3, 3, scale_dim), n_scales)
        self.bn2 = nn.BatchNorm3d(filters, momentum=batchnorm_momentum)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.sconv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.sconv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, res, filters, n_scales, scale_dim, name, batchnorm_momentum):
        super(UpsampleBlock, self).__init__()
        self.sconv1 = ScaleConv(filters, (3, 3, 1), n_scales)
        self.bn1 = nn.BatchNorm3d(filters, momentum=batchnorm_momentum)
        self.relu1 = nn.ReLU()
        self.sconv2 = ScaleConv(filters, (3, 3, scale_dim), n_scales)
        self.bn2 = nn.BatchNorm3d(filters, momentum=batchnorm_momentum)
        self.relu2 = nn.ReLU()
        self.upsample = UpsampleEquivariant(size=(2, 2))
        self.concat = nn.Concatenate(dim=-1)
        self.conv3d = nn.Conv3d(filters, filters, kernel_size=1)
        self.bn3 = nn.BatchNorm3d(filters, momentum=batchnorm_momentum)
        self.relu3 = nn.ReLU()

    def forward(self, x, res):
        x = self.sconv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.sconv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.upsample(x)
        x = self.concat((x, res))
        x = self.conv3d(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

class SEUNet(nn.Module):
    def __init__(self, img_size, lifting, n_scales, scale_dim, n_filters, n_classes, depth=4, dropout=0,
                 activation=None, pooling_method='Quadratic', batchnorm_momentum=0.98):
        super(SEUNet, self).__init__()
        self.id_lifting = IdLifting(n_scales)
        self.lifting = lifting

        self.block1_sconv = ScaleConv(n_filters, (3, 3, 1), n_scales)
        self.block1_bn = nn.BatchNorm3d(n_filters, momentum=batchnorm_momentum)
        self.block1_relu = nn.ReLU()

        self.res = []

        for i in range(depth):
            filters = n_filters * (2 ** i)
            block = ScaleConvBlock(filters, n_scales, scale_dim, f'down{i+1}', batchnorm_momentum)
            self.add_module(f'down{i+1}', block)
            self.res.append(block)

        for i in range(depth, 0, -1):
            filters = n_filters * (2 ** i)
            residual = self.res.pop()
            block = UpsampleBlock(residual, filters, n_scales, scale_dim, f'up{i}', batchnorm_momentum)
            self.add_module(f'up{i}', block)

        self.block_out = ScaleConv(n_filters, (3, 3, 1), n_scales)
        self.block_out_bn = nn.BatchNorm3d(n_filters, momentum=batchnorm_momentum)
        self.block_out_relu = nn.ReLU()

        self.dropout = nn.Dropout3d(p=dropout)
        self.proj = MaxProjection()
        self.classifier = nn.Conv2d(n_filters, n_classes, kernel_size=1, padding=0, stride=1)

    def forward(self, inputs):
        x = self.id_lifting(inputs)
        x = self.lifting(x)

        x = self.block1_sconv(x)
        x = self.block1_bn(x)
        x = self.block1_relu(x)

        for i in range(1, self.depth + 1):
            x = self.downsample(x, self.pooling_method, f'down{i}_pool')

        for i in range(self.depth, 0, -1):
            residual = self.res.pop()
            x = self.upsample_block(x, residual, f'up{i}')

        x = self.block_out(x)
        x = self.block_out_bn(x)
        x = self.block_out_relu(x)

        if self.dropout > 0:
            x = self.dropout(x)
        x = self.proj(x)
        x = self.classifier(x)

        return x
