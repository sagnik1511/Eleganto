import torch
import torch.nn as nn
from typing import Tuple
import src.config.model as m
from torchsummary import summary
from src.layers import base_classes
from src.utils.model_utils import network_shape_update_conv, network_shape_update_pool


class ElegantoNN(nn.Module):

    def __init__(self, in_channels: int, config: dict, base_filter_dim: int = 16,
                 num_classes: int = 100, input_shape: Tuple[int, int] = (224, 224)):
        super(ElegantoNN, self).__init__()
        self.nC = num_classes
        self.in_channels = in_channels
        self.conf = config
        self.filter_size = base_filter_dim
        self.inp_shape = input_shape
        self.model = self._build_model()

    def _build_model(self) -> nn.Sequential:
        layers = []
        grid_shape = self.inp_shape
        inC, outC = self.in_channels, self.filter_size
        for conf_layer in self.conf:
            ltype = conf_layer[0]
            desc = conf_layer[1]
            if ltype == "C":
                layer = base_classes.Conv(inC, outC, desc[0], desc[1], desc[2])
                inC = outC
                outC *= 2
                grid_shape = network_shape_update_conv(grid_shape, desc[0], desc[1], desc[2])
            elif ltype == "AP":
                layer = nn.AvgPool2d(desc)
                grid_shape = network_shape_update_pool(grid_shape, desc)
            elif ltype == "MP":
                layer = nn.MaxPool2d(desc)
                grid_shape = network_shape_update_pool(grid_shape, desc)
            elif ltype == "L":
                if isinstance(desc, int):
                    layer = nn.Linear(inC, desc)
                    inC = desc
                else:
                    layer = nn.Sequential(
                        nn.Linear(inC, desc[0]),
                        nn.Sigmoid() if desc[1] == "S" else nn.Tanh())
            elif ltype == "F":
                layer = nn.Flatten()
                inC = inC * grid_shape[0] * grid_shape[1]
            elif ltype == "T":
                layer = nn.Linear(inC, self.nC)
            else:
                raise NotImplementedError

            layers.append(layer)

        model = nn.Sequential(*layers)
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def test():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev)
    B, C, BFD = 2, 3, 16
    inp_shape = (C, 224, 224)
    config = m.ELEGANTONN_A
    model = ElegantoNN(C, config, BFD).to(device)
    summary(model, inp_shape, B, dev)


if __name__ == '__main__':
    test()
