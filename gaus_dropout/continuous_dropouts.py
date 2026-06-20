import torch
import torch.nn as nn

class GaussianDropout(nn.Module):
    def __init__(self, p=0.5):
        super(GaussianDropout, self).__init__()
        if p < 0 or p >= 1:
            raise Exception("p value should accomplish 0 < p < 1")
        self.p = p
    
    def forward(self, x):
        if self.training and self.p != 0:
            # in order to have correct normalization, like in standard binary dropout
            stddev = (self.p / (1.0 - self.p)) ** 0.5
            # multiplicative Gaussian noise with mean 1
            epsilon = torch.randn_like(x) * stddev + 1
            return x * epsilon
        else:
            return x

def dropout(p, method='standard', layer=None, prior=None):
    if method == 'standard':
        return nn.Dropout(p)
    elif method == 'gaussian':
        return GaussianDropout(p)
    else:
        raise Exception("gaussian or standard methods are possible")
