import argparse
from typing import Type

import torch
from torch import Tensor, nn


def add_subparser(subparsers):
    parser = subparsers.add_parser("residualconv", help="Residual convolution model (not pre-trained)")
    parser.add_argument("-hs", "--hidden_size", default=4, choices=[2, 4, 8], type=int)
    parser.add_argument("-mp", "--max_pool", action="store_true", help="use max pool instead of average pool")
    # parser.add_argument("-hl", "--hidden_layers", default=18, choices=[18, 50], type=int)
    # parser.add_argument("-ac", "--activation", default="torch.nn.ReLU", help="arbitrary activation functions, default is torch.nn.ReLU", type=str)


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None,
        activation_module: nn.Module = torch.nn.ReLU,
    ) -> None:
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = activation_module(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Model(nn.Module):
    def __init__(
        self,
        hparams,
        n_channels: int = 1,
    ) -> None:
        hs = hparams["model_hparams"].get("hidden_size", 8)
        activation_module = torch.nn.ReLU

        block = BasicBlock
        img_channels = n_channels
        num_classes = 1

        super(Model, self).__init__()

        layers = [2, 2, 2, 2]
        self.expansion = 1

        self.in_channels = 64
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = activation_module(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, hs, activation_module = activation_module)
        # self.layer1 = self._make_layer(block, 64, layers[0], activation_module = activation_module)
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2, activation_module = activation_module)
        self.layer2 = self._make_layer(block, 128, hs, activation_module = activation_module, expansion=True)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2, activation_module = activation_module)
        self.layer3 = self._make_layer(block, 256, hs, stride=2, activation_module = activation_module, expansion=True)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2, activation_module = activation_module)
        if hparams["model_hparams"]["max_pool"]:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * self.expansion, num_classes)
        # self.fc = nn.Linear(128 * self.expansion, num_classes)
        # self.fc = nn.Linear(64 * self.expansion, num_classes)
        # self.fc = nn.Linear(512 * self.expansion, num_classes)
        self.cam_target_layers = self.layer3[-1]

    def _make_layer(self, block: Type[BasicBlock], out_channels: int, blocks: int, stride: int = 1, activation_module = torch.nn.ReLU, expansion = False) -> nn.Sequential:
        downsample = None
        if stride != 1 or expansion:
            """
            This should pass from `layer2` to `layer4` or
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, self.expansion, downsample, activation_module=activation_module))
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, expansion=self.expansion, activation_module=activation_module))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        # The spatial dimension of the final layer's feature
        # map should be (7, 7) for all ResNets.
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Model = torchvision.models.resnet18
