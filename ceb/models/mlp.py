import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_shape, output_shape, layers_num=0, layers_shape=0):
        super(MLP, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        modules = []
        if layers_num == 0:
            modules.append(nn.Linear(input_shape, output_shape))
        else:
            if layers_shape==0:
                raise ValueError("If there are hidden layers, please specify their shape with layers_shape")
            modules.append(nn.Linear(input_shape, layers_shape))
            modules.append(nn.ReLU(True))
            for i in range(layers_num-1):
                modules.append(nn.Linear(layers_shape, layers_shape))
                modules.append(nn.ReLU(True))
            modules.append(nn.Linear(layers_shape, output_shape))

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        return self.net(x_flat)
