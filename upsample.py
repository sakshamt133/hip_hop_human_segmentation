import torch.nn as nn


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2)):
        super(UpSample, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.up(x)
