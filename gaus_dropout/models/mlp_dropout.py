import torch.nn as nn

from continuous_dropouts import *

class MLP_dropout(nn.Module):
    def __init__(self, p, input_shape, output_shape, layers_num=0, layers_shape=0, bottleneck=False):
        super(MLP_dropout, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        modules = []
        if layers_num < 2:
            raise ValueError("For dropout MLP you need at least 2 hidden layers")
        else:
            if layers_shape==0:
                raise ValueError("If there are hidden layers, please specify their shape with layers_shape")
            modules.append(nn.Linear(input_shape, layers_shape))
            modules.append(nn.ReLU(True))
            for i in range(layers_num-2):
                modules.append(nn.Linear(layers_shape, layers_shape))
                modules.append(nn.ReLU(True))
            modules.append(dropout(p, 'gaussian'))
            modules.append(nn.Linear(layers_shape, layers_shape))
            modules.append(nn.ReLU(True))

        self.net = nn.Sequential(*modules)
        self.bottleneck = None
        if bottleneck:
            self.bottleneck = nn.Sequential(nn.Linear(layers_shape, 2), nn.ReLU(True))
            self.classifier = nn.Linear(2, output_shape)
        else:
            self.classifier = nn.Linear(layers_shape, output_shape)

    def forward(self, x, return_repr=False):
        x_flat = x.view(x.size(0), -1)
        out = self.net(x_flat)
        if self.bottleneck:
            out = self.bottleneck(out)
        if return_repr:
            return self.classifier(out), out
        return self.classifier(out)

    def representation(self, x):
        x_flat = x.view(x.size(0), -1)
        out = self.net(x_flat)
        if self.bottleneck:
            out = self.bottleneck(out)
        return out

    def predictor(self, repr):
        return self.classifier(repr)


