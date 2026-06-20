import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import pickle
from pathlib import Path
from matplotlib import pyplot as plt
import torch.optim as optim
from torch.utils.data import Subset
import importlib.util
import argparse

from local_datasets.datasets import get_dataloaders
from continuous_dropouts import *
from doe_util import DoE
from NC_regularizer import compute_cluster

# EXP_PATH = Path('checkpoints/densenet_cifar100/optuna_lambda_lmbd0.001')
DATA_DIM = 3*32*32 # experiment specific! 3*32*32 for CIFAR and SVHN, 28*28 for FMNIST, 256 for mini-Bert embeddings
REPR_DIM = 128 # experiment specific!

#BINS = 5
critic_params = {
    'layers': 2,
    'hidden': 128 # align with REPR_DIM
}
SAMPLES = 4
LR = 0.0001
GRAD_CLIP = 1
BS = 256
EPOCHS = 3

def extract_representations(model, dataloader, device='cuda', llm_mode=False):
    reps = []
    labels = []
    with torch.no_grad():
        for B in dataloader:
            if llm_mode:
                input_ids = B["input_ids"].to(device)
                attention_mask = B["attention_mask"].to(device)
                y = B["label"].to(device)
                r = model.representation(input_ids=input_ids, attention_mask=attention_mask)
            else:
                X, y = B
                X, y = X.to(device), y.to(device)
                r = model.representation(X)        # (B, REPR_DIM), on device
            reps.append(r)
            labels.append(y)
    reps = torch.cat(reps, dim=0)   # (N, D)
    labels = torch.cat(labels, dim=0)
    return reps, labels

def compute_cdnv_from_reprs(reprs: torch.Tensor, labels: torch.Tensor, device=None):
    cdnv, _, _ = compute_cluster(reprs, labels)
    return cdnv

# single batch training for DoE with SAMPLES stochastic reps
def train_doe(doe, optimizer, model, loader, device, llm_mode=False):
    doe.train()
    mi_loss_log = []
    for step, B in enumerate(loader):
        if llm_mode:
            input_ids = B["input_ids"].to(device)
            attention_mask = B["attention_mask"].to(device)
            curX = model.bert.embeddings(input_ids)
        else:
            batch_X, _ = B
            batch_X = batch_X.to(device)  # (B, C, H, W)
            # flatten samples once on GPU
            curX = batch_X #.flatten(start_dim=1)  # (B, DATA_DIM)
        # collect SAMPLES representations on GPU
        losses = []
        # we have to avoid passing the same example with multiple noisy representations,
        # because then DoE will match negative pairs wrongly (same X with another representation is not negative)
        for _ in range(SAMPLES):
            # enable train mode for model if you want stochasticity (dropout)
            if llm_mode:
                r = model.representation(input_ids=input_ids, attention_mask=attention_mask)
            else:
                r = model.representation(batch_X)    # (B, REPR_DIM) on device
            loss = doe(curX, r)
            losses.append(loss)
        # Average the losses over SAMPLES
        total_loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(doe.parameters(), GRAD_CLIP)
        optimizer.step()

        mi_loss_log.append(total_loss.item())

        if (step+1) % 10 == 0:
            print('step {:4d} | '.format(step+1), end='')
            print('doe_l: {:6.2f} | '.format(-loss.item()))
    return loss, mi_loss_log

def plot_mi_loss_log(loss_log, plot_name, smoothing=50):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_log, alpha=0.7, label="DoE loss (raw)")

    if len(loss_log) > smoothing:
        # simple moving average for smoothing
        smoothed = [
            sum(loss_log[i - smoothing:i]) / smoothing
            for i in range(smoothing, len(loss_log))
        ]
        plt.plot(range(smoothing, len(loss_log)), smoothed, c="red", label=f"smoothed ({smoothing})")

    plt.xlabel("Training step")
    plt.ylabel("DoE loss")
    plt.title("DoE critic training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(EXP_PATH / (plot_name+".jpg"))
    plt.close()

def enable_dropout(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, GaussianDropout)):
            m.train()

def report_MI(doe_loss, eps=1e-6):
    """
    Convert DoE loss to MI for reporting.
    - If loss is slightly positive (numerical noise), report 0.
    - If loss is significantly positive, report NaN (numerical instability).
    - Otherwise, return -loss.
    """
    value = doe_loss.cpu().item()  # scalar
    if value > eps:
        return float('nan')
    elif value > 0:  # tiny positive values
        return 0.0
    else:
        return -value

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', dest='exp_dir', type=str, help='experiment directory')
    parser.add_argument('--llm_mode', dest='llm_mode', type=str, help='is it for a text task transformer?', default="False")
    args = parser.parse_args()
    EXP_PATH = Path(args.exp_dir)
    args.llm_mode = args.llm_mode.lower() == "true"

    spec = importlib.util.spec_from_file_location("param_setup", (EXP_PATH / "params.py"))
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)

    cur_lmbd = float(str(args.exp_dir).split("/")[-1].split('_')[0][4:])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_loaders = get_dataloaders(params.dset_dir, BS, drop_last=False)
    train_data = data_loaders['train'].dataset
    test_data = data_loaders['test'].dataset

    # plot and save learning history
    train_hist = pickle.load(open((EXP_PATH/"train_history.pkl"), "rb"))
    test_hist = pickle.load(open((EXP_PATH/"test_history.pkl"), "rb"))
    plt.plot(test_hist['epoch'], test_hist['loss'], c='orange', label='classification loss')
    plt.plot(train_hist['epoch'], train_hist['loss'], c='orange', linestyle='--', alpha=0.6, label='train classification loss')
    plt.plot(test_hist['epoch'], test_hist['acc'], c='blue', label='test accuracy')
    plt.plot(train_hist['epoch'], train_hist['acc'], c='blue', linestyle='--', alpha=0.6, label='train accuracy')
    plt.legend()
    #plt.show()
    plt.savefig(EXP_PATH/"training_hist.jpg")
    plt.close()

    cdnv_hist = pickle.load(open((EXP_PATH/"cdnv_history.pkl"), "rb"))
    fig, ax1 = plt.subplots(figsize=(8, 5))
    # --- first y-axis: loss + accuracy ---
    ax1.plot(test_hist['epoch'], test_hist['loss'], c='orange', label='test loss')
    ax1.plot(train_hist['epoch'], train_hist['loss'], c='orange', linestyle='--', alpha=0.6, label='train loss')
    ax1.plot(test_hist['epoch'], test_hist['acc'], c='blue', label='test accuracy')
    ax1.plot(train_hist['epoch'], train_hist['acc'], c='blue', linestyle='--', alpha=0.6, label='train accuracy')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss / Accuracy")
    ax1.tick_params(axis='y')
    ax1.legend(loc="upper left")
    # --- second y-axis: CDNV ---
    ax2 = ax1.twinx()
    ax2.plot(cdnv_hist['epoch'], cdnv_hist['cdnv'], c='green', label='train CDNV')
    ax2.set_ylabel("CDNV")
    ax2.tick_params(axis='y', labelcolor='green')
    # combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2)
    fig.tight_layout()
    plt.savefig(EXP_PATH / "training_hist_with_cdnv.jpg")
    plt.close()

    last_epoch = train_hist['epoch'][-1]
    params.model.load_state_dict(torch.load(EXP_PATH/("chkp_"+str(last_epoch)), weights_only=True))
    params.model.to(device)
    enable_dropout(params.model)

    print("MI for train data")
    doe_l = DoE(DATA_DIM, REPR_DIM, critic_params['hidden'], critic_params['layers'], 'logistic').to(device)
    optimizer = optim.AdamW(doe_l.parameters(), lr=LR)
    mi_critic_logs = []
    for ep in range(EPOCHS):
        L_doe_l, logs = train_doe(doe_l, optimizer, params.model, data_loaders['train'], device, args.llm_mode)
        mi_critic_logs += logs
    train_mi_xz = report_MI(L_doe_l)
    print("Full MI train ", train_mi_xz)
    plot_mi_loss_log(mi_critic_logs, "DoE_train_data")

    print("conditional on label MI for train data")
    # compute also I(X;Z∣Y) as average among all the per label MIs
    train_mi_xz_byy = []
    for targ in range(params.num_classes):
        print("class ", targ, end=":")

        target_indices = np.where(np.array(train_data["label"]) == targ)[0] # train_data.targets or train_data.labels or train_data["label"]
        print(len(target_indices), end=".")
        target_subset = Subset(train_data, target_indices)
        target_subset_loader = torch.utils.data.DataLoader(target_subset, batch_size=BS, shuffle=True, num_workers=1, pin_memory=True, drop_last=False)

        doe_l = DoE(DATA_DIM, REPR_DIM, critic_params['hidden'], critic_params['layers'], 'logistic').to(device)
        optimizer = optim.AdamW(doe_l.parameters(), lr=LR)
        mi_critic_logs = []
        # since per class amounts are extremely small, one epoch is better not to bias
        for ep in range(1):
            L_doe_l, logs = train_doe(doe_l, optimizer, params.model, target_subset_loader, device, args.llm_mode)
            mi_critic_logs += logs
        train_mi_xz_byy.append(report_MI(L_doe_l))
    print(train_mi_xz_byy)
    train_mi_xz_byy = np.mean(train_mi_xz_byy)

    print("MI for test data")
    doe_l_test = DoE(DATA_DIM, REPR_DIM, critic_params['hidden'], critic_params['layers'], 'logistic').to(device)
    optim_test = optim.AdamW(doe_l_test.parameters(), lr=LR)
    mi_critic_logs = []
    # for test set it might be also better not to bias
    for ep in range(1):
        L_doe_l_test, logs = train_doe(doe_l_test, optim_test, params.model, data_loaders['test'], device, args.llm_mode)
        mi_critic_logs += logs
    test_mi_xz = report_MI(L_doe_l_test)
    print("Full MI test ", test_mi_xz)
    plot_mi_loss_log(mi_critic_logs, "DoE_test_data")

    print("conditional on label MI for test data")
    test_mi_xz_byy = []
    for targ in range(params.num_classes):
        print("class ", targ, end=":")

        target_indices = np.where(np.array(test_data["label"]) == targ)[0] # test_data.targets or test_data.labels or test_data["label"]
        print(len(target_indices), end=".")
        target_subset = Subset(test_data, target_indices)
        target_subset_loader = torch.utils.data.DataLoader(target_subset, batch_size=BS, shuffle=True, num_workers=1, pin_memory=True, drop_last=False)

        doe_l_test = DoE(DATA_DIM, REPR_DIM, critic_params['hidden'], critic_params['layers'], 'logistic').to('cuda')
        optim_test = optim.AdamW(doe_l_test.parameters(), lr=LR)
        mi_critic_logs = []
        # since per class amounts are extremely small, one epoch is better not to bias
        for ep in range(1):
            L_doe_l_test, logs = train_doe(doe_l_test, optim_test, params.model, target_subset_loader, device, args.llm_mode)
            mi_critic_logs += logs
        test_mi_xz_byy.append(report_MI(L_doe_l_test))
    print(test_mi_xz_byy)
    test_mi_xz_byy = np.mean(test_mi_xz_byy)

    # (g1) Compute neural collapse geometric characteristic
    # cdnv(Q1, Q2) = (Var(Q1) + Var(Q2))/(2|mu(Q1) - mu(Q2)|^2)
    # for final sets we take the penultimate representation f, for each class find mean and E[|f - mu|^2] (var)
    # tends to 0 when collapse is happening
    # "ON THE ROLE OF NEURAL COLLAPSE IN TRANSFER LEARNING" Galanti
    # representations are sampled also with dropout, so they correspond to the distributions in which we measure MI
    train_reps, train_labels = extract_representations(params.model, data_loaders['train'], device=device, llm_mode=args.llm_mode)
    test_reps, test_labels = extract_representations(params.model, data_loaders['test'], device=device, llm_mode=args.llm_mode)
    train_cdnv = compute_cdnv_from_reprs(train_reps, train_labels, device=device)
    test_cdnv = compute_cdnv_from_reprs(test_reps, test_labels, device=device)
    print("Avg train CDNV", train_cdnv.item())
    print("Avg test CDNV", test_cdnv.item())

    pickle.dump({
        'train_acc': train_hist['acc'][-1],
        'test_acc': test_hist['acc'][-1],
        'train_loss': train_hist['loss'][-1],
        'test_loss': test_hist['loss'][-1],
        'train_IXZ_givenY': train_mi_xz_byy,
        'test_IXZ_givenY': test_mi_xz_byy,
        'train_IXZ': train_mi_xz,
        'test_IXZ': test_mi_xz,
        # geometric characteristics
        'train_NC_g1': train_cdnv.item(),
        'test_NC_g1': test_cdnv.item()
    }, open(EXP_PATH / "characteristics.pkl", "wb"))
