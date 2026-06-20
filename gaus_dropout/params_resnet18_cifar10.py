import torch.optim as optim
from torch.optim import lr_scheduler
import os

from local_datasets.datasets import get_dataloaders
from models import *

dset_dir = "local_datasets/CIFAR10"
num_classes = 10

batch_size = int(os.environ.get('BATCH_SIZE', 128))
lr = float(os.environ.get('LEARN_RATE', 0.1))
p = float(os.environ.get('DROP_P', 0.4))
epochs = 100
beta = 5
inspect_step = 20

try:
    val_cut = args.val
except NameError:
    val_cut = 0
data_loaders = get_dataloaders(dset_dir, batch_size, val=val_cut)

model = ResNet18_cifar10_model(p).cuda()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)