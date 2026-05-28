import argparse
from typing import Type, Dict
import copy
import json

import torch
from torch import Tensor, nn

from src.models.resnet.model import Model as ResNet
from src.models.lstm.model import Model as LSTM

def add_subparser(subparsers):
    parser = subparsers.add_parser("multi_resnet", help="resnet models with embeddings joined at the end w/ a fully connected linear layer")
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

            # output to embedding instead of 1-dim
            self.input_2_resnet[k].fc = nn.Identity() # nn.Linear(512, hparams["model_hparams"]["hidden_dim_lstm"])

        self.fc = nn.Linear(512*len(n_channels), 1)


    def forward(self, X):
        out = []
        for k in X:
            out.append(self.input_2_resnet[k](X[k]))

        out = torch.cat(out, dim=-1)
        return self.fc(out)

