from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST, MNIST, CIFAR10, CIFAR100

def get_dataloaders(dset_dir, batch_size):
    if '/MNIST' in dset_dir:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])
        train_data = MNIST(root=dset_dir, train=True, transform=transform, download=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
        test_data = MNIST(root=dset_dir, train=False, transform=transform, download=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)
    elif '/FashionMNIST' in dset_dir:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])
        train_data = FashionMNIST(root=dset_dir, train=True, transform=transform, download=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
        test_data = FashionMNIST(root=dset_dir, train=False, transform=transform, download=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)
    elif '/CIFAR10' in dset_dir and not '/CIFAR100' in dset_dir:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
        train_data = CIFAR10(root=dset_dir, train=True, transform=train_transform, download=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=drop_last)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_data = CIFAR10(root=dset_dir, train=False, transform=test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=drop_last)
    elif '/CIFAR100' in dset_dir:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), ])
        train_data = CIFAR100(root=dset_dir, train=True, transform=train_transform, download=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        test_data = CIFAR100(root=dset_dir, train=False, transform=test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)

    data_loaders = dict()
    data_loaders['train'] = train_loader
    data_loaders['test'] = test_loader

    return data_loaders
