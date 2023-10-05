import torch
import torch.nn as nn

class ScaleDropout(nn.Module):
    def __init__(self, rate):
        super(ScaleDropout, self).__init__()
        self.rate = rate

    def forward(self, x):
        if self.training:
            return (1 - self.rate) * torch.nn.functional.dropout(x, p=self.rate)
        return x

class ScaleConv(nn.Module):
    def __init__(self, n_out, kernel_size, n_scales_out, use_bias=True, base=2, strides=(1, 1),
                 kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(ScaleConv, self).__init__()

        self.n_out = n_out
        self.kernel_size = kernel_size
        self.strides = strides
        self.n_scales_out = n_scales_out
        self.base = base
        self.dilations = [base ** i for i in range(n_scales_out)]
        self.use_bias = use_bias

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        k = torch.arange(self.kernel_size[2])
        k = 2 ** (3 * k)
        k = k.view(1, 1, 1, -1, 1)
        self.k = 1 / k.type(torch.float32)
        
        # Define kernel as a learnable parameter
        self.kernel = nn.Parameter(torch.empty(self.kernel_size[0], self.kernel_size[1],
                                               self.kernel_size[2] * input_shape[-1], self.n_out))

        # Initialize the kernel using the specified initializer
        if self.kernel_initializer == 'glorot_uniform':
            nn.init.xavier_uniform_(self.kernel)
        else:
            raise NotImplementedError("Custom initialization not implemented. You can add your initializer.")
    def forward(self, x):
        shape = x.size()
        out = []

        for i in range(self.n_scales_out):
            d = self.dilations[i]
            smax = min(self.n_scales_in, i + self.kernel_size[2])
            x0 = x[:, :, :, i:smax, :]

            if smax - i < self.kernel_size[2]:
                x0 = torch.cat([x0] + [x0[:, :, :, -1:, :]] * (self.kernel_size[2] + i - smax), dim=-2)

            x0 = (x0 * self.k).view(shape[0], shape[1], shape[2], -1)
            sr = d * (self.kernel_size[0] // 2)
            sc = d * (self.kernel_size[1] // 2)
            pad_r = [sr, sr]
            pad_c = [sc, sc]
            paddings = (0, 0), pad_r, pad_c, (0, 0)
            x0 = nn.functional.pad(x0, paddings)

            x0 = nn.functional.conv2d(x0, self.kernel, stride=self.strides, dilation=d)
            out.append(x0)
        
        out = torch.stack(out, dim=3)

        if self.use_bias:
            out = out + self.bias.view(1, 1, 1, -1)

        return out
