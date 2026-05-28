import argparse
from typing import Type, Dict
import copy
import json

import torch
from torch import Tensor, nn

from src.models.resnet.model import Model as ResNet
from src.models.lstm.model import Model as LSTM

def add_subparser(subparsers):
    parser = subparsers.add_parser("multi_resnet-lstm", help="resnet models and an lstm model with embeddings joined at the end w/ a fully connected linear layer. Note that the embeddings of the resnets were downscaled to match the lstm one.")
    parser.add_argument("-hlr", "--hidden_layers_resnet", default=18, choices=[18, 50], type=int)
    parser.add_argument("-hll", "--hidden_layers_lstm", default=2, type=int)
    parser.add_argument("-hdl", "--hidden_dim_lstm", default=20, type=int)
    parser.add_argument(
        "-dl",
        "--deterministic_lstm",
        action="store_true",
        help="Make forward initialization deterministic by using torch.zero instead of randn. Note: it might decrease accuracy."
    )
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

        print(json.dumps(hparams,indent=2))
        resnet_hparams = copy.deepcopy(hparams)
        resnet_hparams["model_hparams"]["hidden_layers"] = resnet_hparams["model_hparams"]["hidden_layers_resnet"]

        for k in n_channels:
            self.input_2_resnet[k] = ResNet(resnet_hparams, n_channels[k])

            # output to embedding instead of 1-dim
            self.input_2_resnet[k].fc = nn.Linear(512, hparams["model_hparams"]["hidden_dim_lstm"])

        lstm_hparams = copy.deepcopy(hparams)
        lstm_hparams["model_hparams"]["hidden_layers"] = resnet_hparams["model_hparams"]["hidden_layers_lstm"]
        lstm_hparams["model_hparams"]["hidden_dim"] = resnet_hparams["model_hparams"]["hidden_dim_lstm"]
        lstm_hparams["model_hparams"]["deterministic"] = resnet_hparams["model_hparams"]["deterministic_lstm"]

        print(f"{n_channels_time_series=}")
        self.lstm = LSTM(
            lstm_hparams,
            n_channels=None,
            n_channels_time_series = n_channels_time_series,
            length_time_series=length_time_series
        )

        # select embedding corresponding do latest time
        print(self.lstm.linear)
        self.lstm.linear = nn.Identity()

        self.fc = nn.Linear(hparams["model_hparams"]["hidden_dim_lstm"]*(len(n_channels)+1), 1)


    #def to(self, device):
    #    for k in self.input_2_resnet:
    #        self.input_2_resnet[k] = self.input_2_resnet[k].to(device)

    #    self.lstm = self.lstm.to(device)
    #    self.fc = self.fc.to(device)

    def forward(self, X):
        out = []
        for k in X[1]:
            out.append(self.input_2_resnet[k](X[1][k]))

        out.append(self.lstm(X))
        out = torch.cat(out, dim=-1)
        return self.fc(out)

