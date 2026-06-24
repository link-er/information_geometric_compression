import numpy as np
import torch
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import pickle
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

from datasets.datasets import get_dataloaders
from models import *

if __name__ == "__main__":
    EXP_PATH_parent = Path('checkpoints/wrn28-4_cifar10')

    exp_data = []
    for exp in EXP_PATH_parent.iterdir():
        EXP_PATH = exp

        # not very clean way for sharing and saving setup of the experiment
        exec(open(EXP_PATH / "params.py").read())

        test_history = pickle.load(open(EXP_PATH/"test_history.pkl", "rb"))
        train_history = pickle.load(open(EXP_PATH/"train_history.pkl", "rb"))

        cmap = matplotlib.colormaps['viridis']
        colors = cmap(np.linspace(0.1, 0.9, 4))

        plt.plot(test_history['epoch'], test_history['I(X;Z|Y)'], c=colors[0], label='test I(X;Z|Y)')
        plt.plot(train_history['epoch'], train_history['I(X;Z|Y)'], c=colors[0], linestyle='--', alpha=0.6, label='train I(X;Z|Y)')
        plt.legend()
        #plt.show()
        plt.savefig(EXP_PATH/"mi_xz.jpg")
        plt.close()

        plt.plot(test_history['epoch'], test_history['class_loss'], c=colors[1], label='classification loss')
        plt.plot(train_history['epoch'], train_history['class_loss'], c=colors[1], linestyle='--', alpha=0.6, label='train classification loss')
        plt.plot(test_history['epoch'], test_history['total_loss'], c=colors[2], label='total loss')
        plt.plot(test_history['epoch'], test_history['acc'], c=colors[3], label='test accuracy')
        plt.plot(train_history['epoch'], train_history['acc'], c=colors[3], linestyle='--', alpha=0.6, label='train accuracy')
        plt.legend()
        #plt.show()
        plt.savefig(EXP_PATH/"training_hist.jpg")
        plt.close()

        # Plot information plane
        COLORBAR_MAX_EPOCHS = train_history['epoch'][-1]
        sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=COLORBAR_MAX_EPOCHS))

        for ind, epoch in enumerate(train_history['epoch']):
            c = sm.to_rgba(epoch)
            xmvals = train_history['I(X;Z|Y)'][ind]
            ymvals = train_history['I(Z;Y)_bound'][ind]
            plt.scatter(xmvals, ymvals, s=20, facecolors=c, zorder=2)
        # plt.ylim([1, 3.5])
        # plt.xlim([4, 14])
        # to avoid ugly numbers on the axis
        ax = plt.gca()
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        # ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        plt.xlabel('I(X;Z|Y)')
        plt.ylabel('I(Y;Z)')
        plt.colorbar(sm, ax = ax, label='Epoch')
        plt.tight_layout()
        #plt.show()
        plt.savefig(EXP_PATH/"inf_plain.jpg")
        plt.close()

        # get representations based on the last model
        model.load_state_dict(torch.load(EXP_PATH/("chkp_"+str(train_history['epoch'][-1]))))
        # collect indices of different class samples for geometric compression measures
        train_class_inds, test_class_inds = {}, {}
        train_labels, test_labels = [], []
        for c in range(num_classes):
            train_class_inds[c], test_class_inds[c] = [], []
        train_reprs = []
        for ind, d in enumerate(data_loaders['train']):
            train_reprs += model.representation(d[0].cuda()).data.cpu().numpy().tolist()
            train_labels += d[1].cpu().data.numpy().tolist()
            for c in range(num_classes):
                train_class_inds[c] += ((d[1]==c).nonzero().flatten()+ind*batch_size).tolist()
        train_reprs = np.array(train_reprs)
        test_reprs = []
        for ind, d in enumerate(data_loaders['test']):
            test_reprs += model.representation(d[0].cuda()).data.cpu().numpy().tolist()
            test_labels += d[1].cpu().data.numpy().tolist()
            for c in range(num_classes):
                test_class_inds[c] += ((d[1]==c).nonzero().flatten()+ind*batch_size).tolist()
        test_reprs = np.array(test_reprs)

        # Compute neural collapse geometric characteristic
        # cdnv(Q1, Q2) = (Var(Q1) + Var(Q2))/(2|mu(Q1) - mu(Q2)|^2)
        # for final sets we take the penultimate representation f, for each class find mean and E[|f - mu|^2] (var)
        # tends to 0 when collapse is happening
        # "ON THE ROLE OF NEURAL COLLAPSE IN TRANSFER LEARNING" Galanti
        train_cdnvs = []
        for c1 in range(num_classes):
            ind_c1 = train_class_inds[c1]
            for c2 in range(num_classes):
                if c2 <= c1:
                    continue
                ind_c2 = train_class_inds[c2]
                mu1 = np.mean(train_reprs[ind_c1], axis=0)
                var1 = np.mean([np.linalg.norm(f - mu1)**2 for f in train_reprs[ind_c1]])
                mu2 = np.mean(train_reprs[ind_c2], axis=0)
                var2 = np.mean([np.linalg.norm(f - mu2)**2 for f in train_reprs[ind_c2]])
                train_cdnvs.append((var1 + var2)/(2*np.linalg.norm(mu1 - mu2)**2))
        print("Avg train CDNV", np.mean(train_cdnvs))
        test_cdnvs = []
        for c1 in range(num_classes):
            ind_c1 = test_class_inds[c1]
            for c2 in range(num_classes):
                if c2 <= c1:
                    continue
                ind_c2 = test_class_inds[c2]
                mu1 = np.mean(test_reprs[ind_c1], axis=0)
                var1 = np.mean([np.linalg.norm(f - mu1)**2 for f in test_reprs[ind_c1]])
                mu2 = np.mean(test_reprs[ind_c2], axis=0)
                var2 = np.mean([np.linalg.norm(f - mu2)**2 for f in test_reprs[ind_c2]])
                test_cdnvs.append((var1 + var2)/(2*np.linalg.norm(mu1 - mu2)**2))
        print("Avg test CDNV", np.mean(test_cdnvs))

        #the neural collapse for backwards encoder; we simply need one hot encoded version from each class and send it through the weights to obtain mean and variance
        backw_means, backw_vars = [], []
        for c in range(num_classes):
            m, v = model.backward_representation(torch.tensor([c]).to('cuda'))
            backw_means.append(m.data.cpu().numpy())
            backw_vars.append(v.data.cpu().numpy())
        backw_cdnvs = []
        for c1 in range(num_classes):
            for c2 in range(num_classes):
                if c2 <= c1:
                    continue
                backw_cdnvs.append((backw_vars[c1] + backw_vars[c2])/(2*np.linalg.norm(backw_means[c1] - backw_means[c2])**2))
        print("Avg backward encoder CDNV", np.mean(backw_cdnvs))

        # Save the values for the final model: train\test avg accuracy, train\test class loss, train\test I_XZ, train\test I_YZ, g1, g2
        pickle.dump({
            'train_avg_acc': train_history['acc'][-1],
            'test_avg_acc': test_history['acc'][-1],
            'train_class_loss': train_history['class_loss'][-1],
            'test_class_loss': test_history['class_loss'][-1],
            # conditional mutual information, conditioned on Y
            'train_IXZ': train_history['I(X;Z|Y)'][-1],
            'test_IXZ': test_history['I(X;Z|Y)'][-1],
            'train_IYZ': train_history['I(Z;Y)_bound'][-1],
            'test_IYZ': test_history['I(Z;Y)_bound'][-1],
            # geometric characteristics
            'train_NC_g1': np.mean(train_cdnvs),
            'test_NC_g1': np.mean(test_cdnvs),
            'backward_NC': np.mean(backw_cdnvs)
        }, open(EXP_PATH/"characteristics.pkl", "wb"))
