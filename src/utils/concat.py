import torch
from torch import nn

class ConcatLinear(nn.Module):
    def __init__(self,
                model1_shape: torch.Size,
                model2_shape: torch.Size,
                hidden_layer_size = 4096,
                num_classes: int = 1) -> None:
        super(ConcatLinear, self).__init__()

        infeatures = (torch.prod(torch.tensor(model1_shape), dtype=torch.int) + 
                        torch.prod(torch.tensor(model2_shape), dtype=torch.int))

        self._head = nn.Sequential(
            nn.Linear(in_features=infeatures, out_features=hidden_layer_size),
            # nn.ReLU(),
            # nn.Linear(in_features=hidden_layer_size, out_features=hidden_layer_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layer_size, out_features=num_classes),
        )

    def forward(self, X: tuple[torch.Tensor, torch.Tensor]):
        model1_y = X[0]#.flatten(-2, -1)
        model2_y = X[1]#.flatten(-2, -1)

        z = torch.cat((model1_y, model2_y), dim=1)

        return self._head(z)
