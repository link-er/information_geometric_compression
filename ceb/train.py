import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import pickle
import argparse

from datasets.datasets import get_dataloaders
from models import *

def validation(model, test_loader):
    model.eval()
    loss_record = 0.0
    # Taking the average of all metrics:
    for j, (X_batch, y_batch) in enumerate(test_loader):
        loss_ele = model.get_loss(X_batch.cuda(), y_batch.cuda()).cpu().item()
        loss_record = loss_record + loss_ele
        if j==0:
            all_info_dict = {key: 0 for key in model.info_dict.keys()}
        for key in model.info_dict:
            all_info_dict[key] += model.info_dict[key]
    j += 1
    for key in all_info_dict:
        all_info_dict[key] /= j
    loss_record /= j
    return loss_record, all_info_dict

def print_dict(info_dict):
    for k in info_dict:
        print(k, ":", round(info_dict[k], 4), end="  ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', dest='directory', type=str, help='directory')
    parser.add_argument('--beta', dest='beta', type=float, help='beta')
    parser.add_argument('--seed', dest='seed', type=int, help='random seed')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dir = args.directory
    EXP_PATH = 'checkpoints/densenet121_cifar100/' + dir + "/"

    # not very clean way for sharing and saving setup of the experiment
    exec(open(EXP_PATH + "params.py").read())

    epochs = 150

    # Inspection
    inspect_step = 1

    # Optimizer
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    # Get original loss:
    print("-------Initialization----------")
    loss_original, info_dict = validation(model, data_loaders['test'])
    print("Loss", round(loss_original, 4))
    print_dict(info_dict)

    train_history = {}
    test_history = {}

    # Training:
    for i in range(epochs):
        model.train()

        loss_record = 0.0
        for k, (X_batch, y_batch) in enumerate(data_loaders['train']):
            optimizer.zero_grad()
            loss = model.get_loss(X_batch.cuda(), y_batch.cuda())
            loss.backward()
            optimizer.step()
            loss_record = loss_record + loss.cpu().item()
            if k == 0:
                all_info_dict = {key: 0 for key in model.info_dict.keys()}
            for key in model.info_dict:
                all_info_dict[key] += model.info_dict[key]

        scheduler.step()

        k += 1
        for key in all_info_dict:
            all_info_dict[key] /= k
        loss_record /= k
        print("\nEpoch", i+1)
        print("Training loss", round(loss_record, 4))
        print_dict(all_info_dict)
        if i == 0:
            train_history = {key: [all_info_dict[key]] for key in all_info_dict.keys()}
            train_history["epoch"] = [1]
        else:
            for key in all_info_dict.keys():
                train_history[key].append(all_info_dict[key])
            train_history["epoch"].append(i+1)

        if (i+1)%10 == 0:
            torch.save(model.state_dict(), EXP_PATH+"chkp_"+str(i+1))

        if i % inspect_step == 0:
            val_loss, info_dict = validation(model, data_loaders['test'])
            print("\nTest loss", round(val_loss, 4))
            print_dict(info_dict)
        if i == 0:
            test_history = {key: [info_dict[key]] for key in info_dict.keys()}
            test_history["epoch"] = [1]
            test_history['total_loss'] = [val_loss]
        else:
            for key in info_dict.keys():
                test_history[key].append(info_dict[key])
            test_history["epoch"].append(i+1)
            test_history['total_loss'].append(val_loss)

    pickle.dump(test_history, open(EXP_PATH + "test_history.pkl", "wb"))
    pickle.dump(train_history, open(EXP_PATH + "train_history.pkl", "wb"))
