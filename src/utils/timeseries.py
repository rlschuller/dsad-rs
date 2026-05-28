import torch
from torch import nn


class LinearTimeSeries(nn.Module):
    def __init__(self,
                 ts_size: int) -> None:
        super(LinearTimeSeries, self).__init__()

        # create a few liner layer
        self._ts_model = nn.Sequential(
            # skip bias since we are going to use a batchnorm layer
            nn.Linear(in_features=ts_size, out_features=4096, bias=False),
            nn.BatchNorm1d(num_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU()
        )

    def output_size(self):
        return torch.Size((4096,))

    def forward(self, time_series):
        # reshape time_series into a vector
        time_series = time_series.flatten(-2, -1)
        return self._ts_model(time_series)


class ConvTimeSeries(nn.Module):
    def __init__(self,
                 ts_channels: int,
                 num_intermediate_channels=4,
                 dropout=0.5,
                 num_linear_layers=1,
                 kernel_size=3,
                 ) -> None:
        super(ConvTimeSeries, self).__init__()

        layers = list()
        num_channels = ts_channels
        # input shape: (N, 11, 141)
        for i in range(3):
            layers += [
                nn.Conv1d(
                    in_channels=num_channels,
                    out_channels=num_intermediate_channels,
                    kernel_size=kernel_size,
                ),
                nn.BatchNorm1d(num_features=num_intermediate_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ]
            num_channels = num_intermediate_channels
        # output shape: (N, 16, 15)

        layers += [
            nn.Conv1d(
                in_channels=num_intermediate_channels,
                out_channels=num_intermediate_channels,
                kernel_size=kernel_size,
            ),
            nn.BatchNorm1d(num_features=num_intermediate_channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=8)
        ]
        # output shape: (N, 4, 8)

        layers += [
            nn.Flatten(),
            nn.Dropout(p=dropout),
        ]
        
        in_f = out_f = num_intermediate_channels*8
        for i in range(num_linear_layers):
            if i == num_linear_layers - 1:
                out_f = 8

            layers += [
                nn.Linear(in_features=in_f, out_features=out_f, bias=False),
                nn.BatchNorm1d(num_features=out_f),
                nn.ReLU()
            ]

        self._ts_model = nn.Sequential(
            *layers
        )

    def output_size(self):
        return torch.Size((8,))

    def forward(self, time_series):
        # reshape time_series into a vector
        return self._ts_model(time_series)

def test_size():
    import torchinfo

    print("aqui")
    batch_size = 2
    num_channels = 11
    length = 128

    convts = ConvTimeSeries(ts_channels=11, kernel_size=5)
    torchinfo.summary(convts, torch.Size((batch_size, num_channels, length)))
    

if __name__ == "__main__":
    test_size()
