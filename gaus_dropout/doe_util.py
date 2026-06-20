import torch
import math

# parametric mutual-information estimator built from a marginal qY and conditional qY_X
class DoE(torch.nn.Module):
    def __init__(self, dimX, dimY, hidden, layers, pdf):
        super(DoE, self).__init__()
        self.qY = PDF(dimY, pdf)
        self.qY_X = ConditionalPDF(dimX, dimY, hidden, layers, pdf)

    def forward(self, X, Y):
        # computes the (mean) negative log-likelihood of Y under the marginal qY
        hY = self.qY(Y)
        # computes the (mean) negative log-likelihood of Y under the conditional distribution predicted from X
        hY_X = self.qY_X(Y, X)

        loss = hY + hY_X
        mi_loss = hY_X - hY
        # Numeric value returned equals mi_loss
        # Gradient flow: gradients flow only through loss (because the (mi_loss - loss).detach() piece is detached)
        # Practically: the backward pass sees gradient loss, not mi_loss
        # Why do this? Often mi_loss is high-variance/unreliable; using loss for gradients stabilizes optimization
        # while reporting the mi_loss value
        return (mi_loss - loss).detach() + loss

class ConditionalPDF(torch.nn.Module):
    def __init__(self, dimX, dimY, hidden, layers, pdf):
        super(ConditionalPDF, self).__init__()
        assert pdf in {'gauss', 'logistic'}
        self.dimX = dimX
        self.dimY = dimY
        self.pdf = pdf
        # feedforward module; it outputs 2 * dimY values per input sample
        # these are split into mu and ln_var
        self.X2Y = ConvX2Y(dimY) # for CIFAR10, CIFAR100, SVHN
        #self.X2Y = TransformerX2Y(dimX, dimY) # for Bert
        #self.X2Y = FF(dimX, hidden, 2 * dimY, layers)

    def forward(self, Y, X):
        mu, ln_var = torch.split(self.X2Y(X), self.dimY, dim=1)
        cross_entropy = compute_negative_ln_prob(Y, mu, ln_var, self.pdf)
        return cross_entropy

class PDF(torch.nn.Module):
    def __init__(self, dimY, pdf):
        super(PDF, self).__init__()
        assert pdf in {'gauss', 'logistic'}
        self.dimY = dimY
        self.pdf = pdf
        # compact way to create (1, dimY) learnable parameters
        # the marginal qY is a diagonal distribution with learnable mu and ln_var shared across all samples
        # equivalent to nn.Parameter(torch.zeros(1, dimY)) for each
        self.mu = torch.nn.Embedding(1, self.dimY)
        self.ln_var = torch.nn.Embedding(1, self.dimY)  # ln(s) in logistic

    def forward(self, Y):
        cross_entropy = compute_negative_ln_prob(Y, self.mu.weight, self.ln_var.weight, self.pdf)
        return cross_entropy

class FF(torch.nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, num_layers,
                 activation='relu', dropout_rate=0.1, layer_norm=True,
                 residual_connection=False):
        super(FF, self).__init__()
        assert (not residual_connection) or (dim_hidden == dim_input)
        self.residual_connection = residual_connection

        self.stack = torch.nn.ModuleList()
        for l in range(num_layers):
            layer = []
            if layer_norm:
                layer.append(torch.nn.LayerNorm(dim_input if l == 0 else dim_hidden))
            layer.append(torch.nn.Linear(dim_input if l == 0 else dim_hidden, dim_hidden))
            layer.append({'tanh': torch.nn.Tanh(), 'relu': torch.nn.ReLU()}[activation])
            layer.append(torch.nn.Dropout(dropout_rate))
            self.stack.append(torch.nn.Sequential(*layer))
        self.out = torch.nn.Linear(dim_input if num_layers < 1 else dim_hidden, dim_output)

    def forward(self, x):
        for layer in self.stack:
            x = x + layer(x) if self.residual_connection else layer(x)
        return self.out(x)

# for 32*32*3 X
class ConvX2Y(torch.nn.Module):
    def __init__(self, dimY, hidden=512):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, stride=2, padding=1),  # 16×16
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1), # 8×8
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, 3, stride=2, padding=1), # 4×4
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(256*4*4, hidden),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Linear(hidden, 2*dimY)

    def forward(self, x):
        h = self.encoder(x)
        return self.out(h)

class TransformerX2Y(torch.nn.Module):
    def __init__(self, dimX, dimY, hidden=256, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        # Optional projection if BERT dim is different
        self.input_proj = torch.nn.Linear(dimX, hidden)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pool sequence → single representation (CLS-style)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)  # mean-pool across tokens

        # Output distribution parameters
        self.out = torch.nn.Linear(hidden, 2 * dimY)

    def forward(self, x, attention_mask=None):
        h = self.input_proj(x)  # (B, seq_len, dim_hidden)
        h = self.encoder(h, src_key_padding_mask=(~attention_mask.bool()) if attention_mask is not None else None)
        h = h.mean(dim=1)  # (B, dim_hidden)
        return self.out(h)

def compute_negative_ln_prob(Y, mu, ln_var, pdf):
    eps = 1e-8
    # Clamp ln_var for stability
    ln_var = torch.clamp(ln_var, min=-20.0, max=20.0)
    var = ln_var.exp() + eps

    if pdf == 'gauss':
        negative_ln_prob = 0.5 * ((Y - mu) ** 2 / var).sum(1).mean() + \
                           0.5 * Y.size(1) * math.log(2 * math.pi) + \
                           0.5 * ln_var.sum(1).mean()

    elif pdf == 'logistic':
        whitened = (Y - mu) / var
        adjust = torch.nn.functional.softplus(-whitened)
        negative_ln_prob = whitened.sum(1).mean() + \
                           2 * adjust.sum(1).mean() + \
                           ln_var.sum(1).mean()

    else:
        raise ValueError('Unknown PDF: %s' % (pdf))

    return negative_ln_prob