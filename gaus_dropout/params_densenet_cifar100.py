import torch.optim as optim
from torch.optim import lr_scheduler
import os

from local_datasets.datasets import get_dataloaders
from models import *

dset_dir = "local_datasets/CIFAR100"
num_classes = 100

batch_size = int(os.environ.get('BATCH_SIZE', 256))
lr = float(os.environ.get('LEARN_RATE', 0.1))
p = float(os.environ.get('DROP_P', 0.3))
epochs = 200
beta = 5
inspect_step = 20

try:
    val_cut = args.val
except NameError:
    val_cut = 0
data_loaders = get_dataloaders(dset_dir, batch_size, val=val_cut)

# Densenet 121
model = DenseNet(32, (6, 12, 24, 16), 64, 4, p, num_classes, False).cuda()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)