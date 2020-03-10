from torch import nn

#class ConvLayer(nn.Sequential):
#    def __init__(self, in_channels, out_channels, kernel_size=3, padding=None, stride=1, zero=True, **kwargs):
#        if padding is None: padding = (kernel_size-1)//2
#        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)
#        batch_norm = nn.BatchNorm2d(out_channels)
#        if batch_norm.affine:
#            batch_norm.bias.data.fill_(1e-3)
#            batch_norm.weight.data.fill_(0. if zero else 1.)
#        layers = [conv]
#        super().__init__(*layers)

def noop(x): return x

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

def conv(in_channels, out_channels, kernel_size=3, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=ks//2, bias=bias)


def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m (nn.Conv2d, nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)

def conv_layer(in_channels, out_channels, kernel_size=3, stride=1, zero_bn=False, act=True):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
    layers = [conv(in_channels, out_channels, kernel_size, stride=stride), bn]
    if act: layers.append(act_fn)
    return nn.Sequential(*layers)

class ResBlock(nn.Module):
    def __init__(self, expansion, in_channels, hid_channels, stride=1):
        super().__init__()
        out_channels, in_channels = hid_channels*expansion, in_channels*expansion
        layers = [conv_layer(in_channels,  hid_channels, 3, stride=stride),
                  conv_layer(hid_channels, out_channels, 3, zero_bn=True, act=False)
        ] if expansion == 1 else [
                conv_layer(in_channels,  hid_channels, 1),
                conv_layer(hid_channels, hid_channels, 3, stride=stride),
                conv_layer(hid_channels, out_channels, 1, zero_bn=True, act=False)
        ]
        self.convs = nn.Sequential(*layers)
        self.idconv = noop if in_channels==out_channels else conv-layer(in_channels, out_channels, 1, act=False)
        self.pool = noop if stride==1 else nn.AvgPool2d(2, ciel_mode=True)

    def forward(self, x): return act_fn(self.convs(x) + self.idconv(self.pool(x)))


class XResNet(nn.Sequential):
    def create(cls, expansion, layers, in_channels=3, out_channels=1000):
        # create stem of resnet
        num_filters = [in_channels, (in_channels+1)*8, 64, 64]
        stem = [conv_layer(num_filters[i], num_filters[i+1], stride=2 if i==0 else 1)
                for i in range(3)]

        num_filters = [64//expansion, 64, 128, 256, 512]
        res_layers = [cls._make_layer(epxansion, num_filters[i], num_filters[i+1],
            n_blocks=l, stride=1 if i==0 else 2)
            for i,l in enumerate(layers)]

        res = cls(
                *stem,
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                *res_layers,
                nn.AdaptiveAvgPool2d(1), Flatten(),
                nn.Linear(num_filters[-1]*expansion, out_channels),
                )
        init_cnn(res)
        return res

    @staticmethod
    def _make_layer(expansion, in_channels, out_channels, n_blocks, stride):
        return nn.Sequential(
                *[ResBlock(expansion, in_channels if i==0 else out_channels, stride if i==0 else 1)
                    for i in range(n_blocks)])

def xresnet18 (**kwargs(): return XResNet.create(1, [2, 2,  2, 2], **kwargs)
def xresnet34 (**kwargs(): return XResNet.create(1, [3, 4,  6, 3], **kwargs)
def xresnet50 (**kwargs(): return XResNet.create(4, [3, 4,  6, 3], **kwargs)
def xresnet101(**kwargs(): return XResNet.create(4, [3, 4, 23, 3], **kwargs)
def xresnet152(**kwargs(): return XResNet.create(4, [3, 8, 36, 3], **kwargs)
