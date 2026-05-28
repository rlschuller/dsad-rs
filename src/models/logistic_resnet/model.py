import argparse
from typing import Type, Dict
import copy
import json

import torch
from torch import Tensor, nn

from src.models.resnet.model import Model as ResNet

def add_subparser(subparsers):
    parser = subparsers.add_parser("logistic_resnet", help="multiple resnet models with outputs joined by a linear layer, similar to logistic regression")
    parser.add_argument("-hl", "--hidden_layers", default=18, choices=[18, 50], type=int)
    parser.add_argument("-ac", "--activation", default="torch.nn.ReLU", help="arbitrary activation functions, default is torch.nn.ReLU", type=str)


class Model(nn.Module):
    def __init__(
        self,
        hparams,
        n_channels: int = 1,
        n_channels_time_series: int = 1,
        length_time_series: int = 1,
    ) -> None:


        super(Model, self).__init__()

        self.input_2_resnet = nn.ModuleDict()

        for k in n_channels:
            self.input_2_resnet[k] = ResNet(hparams, n_channels[k])

        self.fc = nn.Linear(len(n_channels), 1)


    def forward(self, X):
        out = []
        for k in X:
            out.append(self.input_2_resnet[k](X[k]))

        out = torch.cat(out, dim=-1)
        return self.fc(out)

