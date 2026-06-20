import torch.optim as optim
from torch.optim import lr_scheduler
import os

from local_datasets.datasets import get_dataloaders
from models import *

dset_dir = "local_datasets/AGNews"
num_classes = 4

batch_size = int(os.environ.get('BATCH_SIZE', 256))
lr = float(os.environ.get('LEARN_RATE', 1e-5))
p = float(os.environ.get('DROP_P', 0.6))
epochs = 20
beta = 5
inspect_step = 2

try:
    val_cut = args.val
except NameError:
    val_cut = 0
data_loaders = get_dataloaders(dset_dir, batch_size, val=val_cut)

model = MiniBERTWithBottleneck(num_classes=num_classes, dropout_p=p).cuda()

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
