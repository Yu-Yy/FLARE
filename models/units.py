# -*- encoding: utf-8 -*-
'''
@File    :   units.py
@Time    :   2026/02/28 18:31:49
@Author  :   panzhiyu 
@Version :   1.0
@Contact :   pzy20@mails.tsinghua.edu.cn
@License :   Copyright (c) 2026, Zhiyu Pan, Tsinghua University. All rights reserved
'''
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageGradient(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
        kernel_y = [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        grad_x = F.conv2d(x, self.weight_x, padding=1)
        grad_y = F.conv2d(x, self.weight_y, padding=1)
        return grad_x, grad_y

class ImageGaussian(nn.Module):
    def __init__(self, win_size, std):
        super().__init__()
        self.win_size = max(3, win_size // 2 * 2 + 1)

        n = np.arange(0, win_size) - (win_size - 1) / 2.0
        gkern1d = np.exp(-(n ** 2) / (2 * std ** 2)) / (np.sqrt(2 * np.pi) * std)
        gkern2d = np.outer(gkern1d, gkern1d)
        gkern2d = torch.FloatTensor(gkern2d).unsqueeze(0).unsqueeze(0)
        self.gkern2d = nn.Parameter(data=gkern2d, requires_grad=False)

    def forward(self, x):
        x_gaussian = F.conv2d(x, self.gkern2d, padding=self.win_size // 2)
        return x_gaussian

class DecoderSkip2(nn.Module):
    def __init__(self, in_channel, num_layers, expansion=1, do_bn=True) -> None:
        super().__init__()
        self.n_layer = len(num_layers)

        for ii, cur_channel in enumerate(num_layers):
            setattr(self, f"upsample{ii}", nn.ConvTranspose2d(in_channel * expansion, cur_channel * expansion, 2, 2))
            setattr(self, f"layer{ii}", DoubleConv((cur_channel + cur_channel) * expansion, cur_channel * expansion, do_bn))
            in_channel = cur_channel

    def forward(self, inputs):
        y = inputs[0]
        for ii in range(self.n_layer):
            y = getattr(self, f"upsample{ii}")(y)
            y = getattr(self, f"layer{ii}")(torch.cat((inputs[ii + 1], y), dim=1))

        return y

class FastCartoonTexture(nn.Module):
    def __init__(self, sigma=2.5, eps=1e-6) -> None:
        super().__init__()
        self.sigma = sigma
        self.eps = eps
        self.cmin = 0.3
        self.cmax = 0.7
        self.lim = 20

        self.img_grad = ImageGradient()

    def lowpass_filtering(self, img, L):
        img_fft = torch.fft.fftshift(torch.fft.fft2(img), dim=(-2, -1)) * L

        img_rec = torch.fft.ifft2(torch.fft.fftshift(img_fft, dim=(-2, -1)))
        img_rec = torch.real(img_rec)

        return img_rec

    def gradient_norm(self, img):
        Gx, Gy = self.img_grad(img)
        return torch.sqrt(Gx ** 2 + Gy ** 2) + self.eps

    def forward(self, input):
        H, W = input.size(-2), input.size(-1)
        grid_y, grid_x = torch.meshgrid(torch.linspace(-0.5, 0.5, H), torch.linspace(-0.5, 0.5, W), indexing="ij")
        grid_radius = torch.sqrt(grid_x ** 2 + grid_y ** 2) + self.eps

        L = (1.0 / (1 + (2 * np.pi * grid_radius * self.sigma) ** 4)).type_as(input)[None, None]

        grad_img1 = self.gradient_norm(input)
        grad_img1 = self.lowpass_filtering(grad_img1, L)

        img_low = self.lowpass_filtering(input, L)
        grad_img2 = self.gradient_norm(img_low)
        grad_img2 = self.lowpass_filtering(grad_img2, L)

        diff = grad_img1 - grad_img2
        flag = torch.abs(grad_img1)
        diff = torch.where(flag > 1, diff / flag.clamp_min(self.eps), torch.zeros_like(diff))

        weight = (diff - self.cmin) / (self.cmax - self.cmin)
        weight = torch.clamp(weight, 0, 1)

        cartoon = weight * img_low + (1 - weight) * input
        texture = (input - cartoon + self.lim) * 255 / (2 * self.lim)
        texture = torch.clamp(texture, 0, 255)
        return texture


class NormalizeModule(nn.Module):
    def __init__(self, m0=0.0, var0=1.0, eps=1e-6):
        super(NormalizeModule, self).__init__()
        self.m0 = m0
        self.var0 = var0
        self.eps = eps

    def forward(self, x):
        x_m = x.mean(dim=(1, 2, 3), keepdim=True)
        x_var = x.var(dim=(1, 2, 3), keepdim=True)
        y = (self.var0 * (x - x_m) ** 2 / x_var.clamp_min(self.eps)).sqrt()
        y = torch.where(x > x_m, self.m0 + y, self.m0 - y)
        return y

class ConvBnPRelu(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_chn, out_chn, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_chn, eps=0.001, momentum=0.99)
        self.relu = nn.PReLU(out_chn, init=0)

    def forward(self, input):
        y = self.conv(input)
        y = self.bn(y)
        y = self.relu(y)
        return y

class FingerprintCompose(nn.Module):
    def __init__(self, win_size=8, do_norm=False, m0=0, var0=1.0, eps=1e-6):
        super().__init__()
        self.win_size = max(3, win_size // 2 * 2 + 1)

        self.norm = NormalizeModule(m0=m0, var0=var0, eps=eps) if do_norm else nn.Identity()
        self.conv_grad = ImageGradient()
        self.conv_gaussian = ImageGaussian(self.win_size, self.win_size / 3.0)
        mean_kernel = torch.ones([self.win_size, self.win_size], dtype=torch.float32)[None, None] / self.win_size ** 2
        self.weight_avg = nn.Parameter(data=mean_kernel, requires_grad=False)

    def forward(self, x):
        assert x.size(1) == 1

        Gx, Gy = self.conv_grad(x)
        Gxx = self.conv_gaussian(Gx ** 2)
        Gyy = self.conv_gaussian(Gy ** 2)
        Gxy = self.conv_gaussian(-Gx * Gy)
        sin2 = F.conv2d(2 * Gxy, self.weight_avg, padding=self.win_size // 2)
        cos2 = F.conv2d(Gxx - Gyy, self.weight_avg, padding=self.win_size // 2)

        x = torch.cat((x, sin2, cos2), dim=1)

        x = self.norm(x)

        return x

class ChannelPad(nn.Module):
    def __init__(self, after_C, before_C=0, value=0) -> None:
        super().__init__()
        self.before_C = before_C
        self.after_C = after_C
        self.value = value

    def forward(self, x):
        prev_0 = [0] * (x.ndim - 2) * 2
        out = F.pad(x, (*prev_0, self.before_C, self.after_C), value=self.value)
        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )  # verify bias false
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BasicDeConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, outpadding=0):
        super(BasicDeConv2d, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=outpadding,
            bias=False,
        )  # verify bias false
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_chn, out_chn, do_bn=True, do_res=False):
        super().__init__()
        self.conv = (
            nn.Sequential(
                nn.Conv2d(in_chn, out_chn, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_chn),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_chn, out_chn, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_chn),
                nn.LeakyReLU(inplace=True),
            )
            if do_bn
            else nn.Sequential(
                nn.Conv2d(in_chn, out_chn, 3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_chn, out_chn, 3, padding=1),
                nn.LeakyReLU(inplace=True),
            )
        )

        self.do_res = do_res
        if self.do_res:
            if out_chn < in_chn:
                self.original = nn.Conv2d(in_chn, out_chn, 1, padding=0)
            elif out_chn == in_chn:
                self.original = nn.Identity()
            else:
                self.original = ChannelPad(out_chn - in_chn)

    def forward(self, x):
        out = self.conv(x)
        if self.do_res:
            res = self.original(x)
            out = out + res
        return out
    
class PositionEncoding2D(nn.Module):
    def __init__(self, in_size, ndim): 
        super().__init__()
        n_encode = ndim // 2
        self.in_size = in_size
        coordinate = torch.meshgrid(torch.arange(in_size[0]), torch.arange(in_size[1]), indexing="ij")
        div_term = torch.exp(torch.arange(0, n_encode, 2).float() * (-math.log(10000.0) / n_encode)).view(-1, 1, 1) 
        pe = torch.cat(
            (
                torch.sin(coordinate[0].unsqueeze(0) * div_term),
                torch.cos(coordinate[0].unsqueeze(0) * div_term),
                torch.sin(coordinate[1].unsqueeze(0) * div_term),
                torch.cos(coordinate[1].unsqueeze(0) * div_term),
            ),
            dim=0,
        )
        self.div_term = div_term # B, d_model, 1, 1
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe