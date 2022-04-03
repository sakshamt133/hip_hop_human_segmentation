import torch.nn as nn
from doub_conv import DoubleConv


class DownSample(nn.Module):
    def __init__(self, in_channels, features):
        super(DownSample, self).__init__()
        self.layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        for feat in features:
            self.layers.append(DoubleConv(in_channels, feat))
            in_channels = feat

    def forward(self, x):
        out = []
        for layer in self.layers:
            x = layer(x)
            out.append(x)
            x = self.pool(x)
        return out, x
