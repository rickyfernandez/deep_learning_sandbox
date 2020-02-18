from torch import nn

def conv2d_layer(in_channels, out_channels, kernel_size=3, stride=2, **kwargs):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                kernel_size, padding=kernel_size//2, stride=stride),
            nn.ReLU(**kwargs))

