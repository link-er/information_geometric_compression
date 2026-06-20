from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST, CIFAR10, CIFAR100, SVHN
from transformers import AutoTokenizer
from datasets import load_from_disk
from pathlib import Path

def get_dataloaders(dset_dir, batch_size, val=0, drop_last = False):
    if '/FashionMNIST' in dset_dir:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_data = FashionMNIST(root=dset_dir, train=True, transform=transform, download=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=drop_last)
        test_data = FashionMNIST(root=dset_dir, train=False, transform=transform, download=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=drop_last)
    elif '/CIFAR10' in dset_dir and not '/CIFAR100' in dset_dir:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        train_data = CIFAR10(root=dset_dir, train=True, transform=train_transform, download=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=drop_last)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_data = CIFAR10(root=dset_dir, train=False, transform=test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=drop_last)
    elif '/CIFAR100' in dset_dir:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])
        train_data = CIFAR100(root=dset_dir, train=True, transform=train_transform, download=False)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=drop_last)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        test_data = CIFAR100(root=dset_dir, train=False, transform=test_transform, download=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=drop_last)
    elif '/SVHN' in dset_dir:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),])
        train_data = SVHN(root=dset_dir, split='train', transform=train_transform, download=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=drop_last)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        test_data = SVHN(root=dset_dir, split='test', transform=test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=drop_last)
    elif "/AGNews" in dset_dir:
        tokenized_path = Path(dset_dir / "ag_news_tokenized")
        if tokenized_path.exists():
            tokenized_datasets = load_from_disk(tokenized_path)
        else:
            dataset = load_from_disk(dset_dir)
            tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-mini")
            def tokenize_function(batch):
                return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
            tokenized_datasets = dataset.map(tokenize_function, batched=True)
            tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
            tokenized_path.mkdir(parents=False, exist_ok=True)
            tokenized_datasets.save_to_disk(tokenized_path)
        train_data = tokenized_datasets["train"]
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(tokenized_datasets["test"], batch_size=batch_size)

    data_loaders = dict()
    if val > 0:
        val_size = int(val * len(train_data))
        train_size = len(train_data) - val_size
        trainval_data, val_data = random_split(train_data, [train_size, val_size])
        train_loader = DataLoader(trainval_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=drop_last)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=drop_last)
        data_loaders['val'] = val_loader
    data_loaders['train'] = train_loader
    data_loaders['test'] = test_loader

    return data_loaders
