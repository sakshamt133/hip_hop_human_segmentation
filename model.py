import torch.nn as nn
from DownSample import DownSample
import torch
from upsample import UpSample
from doub_conv import DoubleConv


class U_Net(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(U_Net, self).__init__()
        self.down = DownSample(in_channels, features)
        self.bottle = DoubleConv(features[-1], 1024, (1, 1), padding=(0, 0), pool=False)
        self.ups = nn.ModuleList()
        for feat in features[::-1]:
            self.ups.append(UpSample(feat * 2, feat))
            self.ups.append(DoubleConv(feat * 2, feat))
        self.final_layer = nn.Conv2d(features[0], out_channels, kernel_size=(1, 1))

    def forward(self, x):
        out, x = self.down(x)
        x = self.bottle(x)
        out = out[::-1]
        for i in range(0, len(self.ups), 2):
            up = self.ups[i](x)
            skip_add = torch.cat((up, out[i//2]), dim=1)
            x = self.ups[i+1](skip_add)

        return self.final_layer(x)
