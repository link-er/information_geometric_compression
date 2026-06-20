import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

def reparametrize(data):
    size = int(data.size(-1) // 2)
    mu = data[:, :size]
    std = F.softplus(data[:, size:], beta=1) + 1e-5
    dist = Normal(mu, std)
    return dist

class CEB(nn.Module):
    def __init__(self, beta, encoder_model, decoder_model, latent_size, backencoder_model, num_classes, sample_size=8):
        super(CEB, self).__init__()
        self.beta = beta
        self.num_classes = num_classes
        self.sample_size = sample_size
        self.latent_size = latent_size

        self.encoder = encoder_model
        self.decoder = decoder_model
        self.backencoder = backencoder_model

        # logging
        self.info_dict = {}

    def rsample_z(self, x):
        """For each x, sample self.sample_size z"""
        mean_logit = self.encoder(x)
        dist = reparametrize(mean_logit)
        z = dist.rsample((self.sample_size,))
        return z, dist

    def get_loss(self, x, y):
        # reparametrize and sample from the encoder output distribution
        z, dist = self.rsample_z(x)

        # Get crossentropy loss
        y_pred = self.decoder(z.flatten(0, 1))
        y_expand = y.repeat(self.sample_size, 1).flatten(0, 1)
        class_loss = nn.CrossEntropyLoss()(y_pred, y_expand)

        self.info_dict["acc"] = (y_expand == y_pred.max(1)[1]).float().mean().cpu().item()
        self.info_dict["class_loss"] = class_loss.cpu().item()
        self.info_dict["I(Z;Y)_bound"] = np.log(self.num_classes) - self.info_dict["class_loss"]

        # Get I(X;Z):
        z_loglikl = dist.log_prob(z).sum(-1)
        enc_loss = z_loglikl.mean()

        self.info_dict["enc_loss"] = -enc_loss.cpu().item()

        # Backward encoder:
        y_onehot = F.one_hot(y, self.num_classes).float()
        dist_backenc = reparametrize(self.backencoder(y_onehot))
        backenc_loglikl = dist_backenc.log_prob(z).sum(-1)
        backenc_loss = -backenc_loglikl.mean()

        self.info_dict["backenc_loss"] = backenc_loss.cpu().item()
        self.info_dict["I(X;Z|Y)"] = (enc_loss + backenc_loss).cpu().item()

        # beta-1 makes it equivalent to VIB
        loss = enc_loss + backenc_loss + class_loss * (self.beta - 1)
        return loss

    def representation(self, x):
        mean_logit = self.encoder(x)
        dist = reparametrize(mean_logit)
        z = dist.rsample()
        return z

    def mean_repr(self, x):
        mean_logit = self.encoder(x)
        dist = reparametrize(mean_logit)
        return dist.mean

    def backward_representation(self, y):
        y_onehot = F.one_hot(y, self.num_classes).float()
        dist_backenc = reparametrize(self.backencoder(y_onehot))
        return dist_backenc.loc, dist_backenc.variance