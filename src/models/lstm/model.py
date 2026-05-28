import argparse
from typing import Type

import torch
from torch import Tensor, nn


def add_subparser(subparsers):
    parser = subparsers.add_parser("lstm", help="lstm classifier")
    parser.add_argument("-hl", "--hidden_layers", default=2, type=int)
    parser.add_argument("-hd", "--hidden_dim", default=20, type=int)
    parser.add_argument(
        "-d",
        "--deterministic",
        action="store_true",
        help="Make forward initialization deterministic by using torch.zero instead of randn. Note: it might decrease accuracy."
    )



class Model(nn.Module):
    def __init__(
        self,
        hparams,
        n_channels: int = 1,
        n_channels_time_series: int = 1,
        length_time_series: int = 1,
    ) -> None:
        super(Model, self).__init__()

        self.lstm_layers = hparams["model_hparams"]["hidden_layers"]
        self.hidden_dim = hparams["model_hparams"]["hidden_dim"]
        self.rnn = nn.LSTM(n_channels_time_series,
                           self.hidden_dim,
                           self.lstm_layers,
                           batch_first=True)

        self.linear = nn.Linear(self.hidden_dim, 1)


        self.bn = nn.BatchNorm1d(n_channels_time_series, affine=False)
        self.deterministic = hparams["model_hparams"]["deterministic"]

        self.cam_target_layers = self.rnn

    def forward(self, x: tuple[Tensor, Tensor]) -> Tensor:
        time_series = self.bn(x[0])

        if self.deterministic:
            # init hidden
            h0 = torch.zeros(
                self.lstm_layers,
                time_series.shape[0],
                self.hidden_dim,
                #generator=gen
            ).to(time_series.device)

            c0 = torch.zeros(
                self.lstm_layers,
                time_series.shape[0],
                self.hidden_dim,
                #generator=gen
            ).to(time_series.device)

        else:
            #gen = torch.Generator()

            # init hidden
            h0 = torch.randn(
                self.lstm_layers,
                time_series.shape[0],
                self.hidden_dim,
                #generator=gen
            ).to(time_series.device)

            c0 = torch.zeros(
                self.lstm_layers,
                time_series.shape[0],
                self.hidden_dim,
                #generator=gen
            ).to(time_series.device)


        time_series = time_series.permute(0, 2, 1)
        output, (hn, cn) = self.rnn(time_series, (h0, c0))


        return self.linear(output[:, -1, :])

