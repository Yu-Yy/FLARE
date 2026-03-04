'''
Description: 
Author: Xiongjun Guan
Date: 2024-06-24 10:40:07
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-08-23 14:27:13

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''

import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups: int = 1,
                 BN: bool = True,
                 Act: nn.Module = nn.ReLU(inplace=True)):
        super().__init__()
        self.bias = False if BN else True
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=self.bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = Act

    def forward(self, x):
        x = self.conv(x)
        if not self.bias:
            x = self.bn(x)
        return self.relu(x)


class ResNextBlock(nn.Module):

    def __init__(self, in_channels, out_channels, groups, stride):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels=in_channels,
                      out_channels=out_channels // 2,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            ConvBlock(in_channels=out_channels // 2,
                      out_channels=out_channels // 2,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      groups=groups),
            ConvBlock(in_channels=out_channels // 2,
                      out_channels=out_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      Act=nn.Identity()),
        )
        self.identity = ConvBlock(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=1,
                                  stride=stride,
                                  padding=0,
                                  Act=nn.Identity())
        if stride == 1 and in_channels == out_channels:
            self.identity = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.identity(x) + self.conv(x))


class ResNeXt50_32x4d(nn.Module):

    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.conv1 = ConvBlock(in_channels=in_channels,
                               out_channels=64,
                               kernel_size=7,
                               stride=2,
                               padding=3)
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layers(in_channels=64,
                              out_channels=256,
                              groups=32,
                              stride=1,
                              num_layers=3))
        self.conv3 = self._make_layers(in_channels=256,
                                       out_channels=512,
                                       groups=32,
                                       stride=2,
                                       num_layers=4)
        self.conv4 = self._make_layers(in_channels=512,
                                       out_channels=1024,
                                       groups=32,
                                       stride=2,
                                       num_layers=6)
        self.conv5 = self._make_layers(in_channels=1024,
                                       out_channels=2048,
                                       groups=32,
                                       stride=2,
                                       num_layers=3)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim=1),
            nn.Linear(in_features=2048, out_features=n_classes, bias=True))

    def _make_layers(self, in_channels, out_channels, groups, stride,
                     num_layers):
        layers = []
        layers.append(
            ResNextBlock(in_channels=in_channels,
                         out_channels=out_channels,
                         groups=groups,
                         stride=stride))
        for _ in range(num_layers - 1):
            layers.append(
                ResNextBlock(in_channels=out_channels,
                             out_channels=out_channels,
                             groups=groups,
                             stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        print('# Conv1 output shape:', x.shape)
        x = self.conv2(x)
        print('# Conv2 output shape:', x.shape)
        x = self.conv3(x)
        print('# Conv3 output shape:', x.shape)
        x = self.conv4(x)
        print('# Conv4 output shape:', x.shape)
        x = self.conv5(x)
        print('# Conv5 output shape:', x.shape)
        x = self.classifier(x)
        print('# Classifier output shape:', x.shape)
        return x


if __name__ == "__main__":
    inputs = torch.randn(4, 3, 224, 224)
    cnn = ResNeXt50_32x4d(in_channels=3, n_classes=1000)
    outputs = cnn(inputs)
